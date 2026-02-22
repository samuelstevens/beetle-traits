# inference.py
"""Run a trained model over labeled datasets and save per-sample results to Parquet.

Produces predictions, errors, heatmap diagnostics, and CLS embeddings for every sample. The output Parquet drives the active learning analysis notebook.
"""

import dataclasses
import logging
import pathlib
import typing as tp

import beartype
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import tyro
from jaxtyping import Array, Float, jaxtyped

import btx.configs
import btx.data
import btx.data.transforms
import btx.helpers
import btx.metrics
import btx.modeling
import btx.objectives.heatmap

SCHEMA = pl.Schema({
    # Metadata
    "beetle_id": pl.String,
    "scientific_name": pl.String,
    "group_img_basename": pl.String,
    "img_fpath": pl.String,
    "dataset": pl.String,
    # Scalar metrics (NaN when ground truth is missing/masked)
    "sample_loss": pl.Float32,
    "width_line_err_cm": pl.Float32,
    "length_line_err_cm": pl.Float32,
    # Heatmap diagnostics
    "mean_entropy": pl.Float32,
    # Coordinates: flat [w0_x, w0_y, w1_x, w1_y, l0_x, l0_y, l1_x, l1_y]
    "pred_coords_px": pl.Array(pl.Float32, 8),
    "gt_coords_px": pl.Array(pl.Float32, 8),
    # CLS embedding from frozen ViT backbone
    "cls_embedding": pl.List(pl.Float32),
})


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    ckpt_fpath: pathlib.Path = pathlib.Path("model.eqx")
    """Path to saved model checkpoint (.eqx file). Contains both model config and weights."""
    hawaii: btx.data.HawaiiConfig = btx.data.HawaiiConfig()
    """Hawaii dataset config. Set --hawaii.go=False to skip."""
    beetlepalooza: btx.data.BeetlePaloozaConfig = btx.data.BeetlePaloozaConfig()
    """BeetlePalooza dataset config. Set --beetlepalooza.go=False to skip."""
    biorepo: btx.data.BioRepoConfig = btx.data.BioRepoConfig()
    """BioRepo dataset config. Set --biorepo.go=False to skip."""
    batch_size: int = 256
    """Inference batch size."""
    n_workers: int = 4
    """Number of dataloader workers."""
    out_fpath: pathlib.Path = pathlib.Path("inference_results.parquet")
    """Output Parquet file path."""
    slurm_acct: str = ""
    """Slurm account. Empty means run locally."""
    slurm_partition: str = ""
    """Slurm partition."""
    n_hours: float = 1.0
    """Slurm job length in hours."""
    log_to: pathlib.Path = pathlib.Path("./logs")
    """Where to write Slurm logs."""


log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("inference")


@jaxtyped(typechecker=beartype.beartype)
class InferAux(eqx.Module):
    """Per-sample outputs from one inference batch."""

    pred_coords_px: Float[Array, "batch 2 2 2"]
    """Predicted endpoints in original-image coordinates."""
    cls_embedding: Float[Array, "batch embed_dim"]
    """CLS token from frozen ViT backbone."""
    mean_entropy: Float[Array, " batch"]
    """Mean heatmap entropy across 4 channels."""
    # Error metrics (NaN when ground truth is masked/missing).
    sample_loss: Float[Array, " batch"]
    width_line_err_cm: Float[Array, " batch"]
    length_line_err_cm: Float[Array, " batch"]


@eqx.filter_jit()
@jaxtyped(typechecker=beartype.beartype)
def forward_batch(
    model: eqx.Module,
    batch: dict[str, Array],
    *,
    heatmap_cfg: btx.objectives.heatmap.Config,
) -> InferAux:
    """Single-pass inference: forward, decode, diagnostics, and optional error metrics.

    Hardcoded for the heatmap objective. If we add new objectives, this function needs a new branch or a parallel implementation. The caller (infer()) asserts the objective type at runtime.

    All outputs are in original-image coordinates. Error metrics are NaN when ground truth is masked or scalebar is invalid.
    """
    # 1) Forward pass: logits + CLS embeddings.
    extract = tp.cast(tp.Callable, model.extract_features)
    logits_bchw, cls_bd = jax.vmap(extract)(batch["img"])

    # 2) Decode heatmap logits to image-space coordinates.
    preds_aug = jax.vmap(
        lambda logits_chw: btx.objectives.heatmap.heatmaps_to_coords(
            logits_chw, cfg=heatmap_cfg
        )
    )(logits_bchw)
    preds_orig = btx.metrics.apply_affine(batch["t_orig_from_aug"], preds_aug)

    # 3) Heatmap diagnostics: mean entropy across 4 channels.
    _, entropy_bc, _ = btx.objectives.heatmap.get_diagnostics(
        logits_bchw, cfg=heatmap_cfg
    )
    mean_entropy = jnp.mean(entropy_bc, axis=1)

    # 4) Error metrics (only meaningful when ground truth + scalebar are valid).
    mask_line = batch["loss_mask"]
    tgts_orig = btx.metrics.choose_endpoint_matching(preds_orig, batch["points_px"])

    scalebar_valid, px_per_cm = btx.metrics.get_scalebar_mask(
        batch["scalebar_px"], batch["scalebar_valid"]
    )
    metric_mask_line = mask_line * scalebar_valid[:, None]

    # Line-length errors in cm.
    pred_line_len = jnp.linalg.norm(preds_orig[:, :, 0] - preds_orig[:, :, 1], axis=-1)
    tgt_line_len = jnp.linalg.norm(tgts_orig[:, :, 0] - tgts_orig[:, :, 1], axis=-1)
    line_err_cm = jnp.abs(pred_line_len - tgt_line_len) / px_per_cm[:, None]
    line_err_cm = jnp.where(metric_mask_line > 0, line_err_cm, jnp.nan)

    # Per-sample heatmap CE loss (reuse objective code for consistency with training).
    heatmap_tgt = jax.vmap(
        lambda tgt: btx.objectives.heatmap.make_targets(tgt, cfg=heatmap_cfg)
    )(batch["tgt"])
    sample_loss_raw = jax.vmap(
        lambda pred, tgt, mask: btx.objectives.heatmap.heatmap_ce_loss(
            pred, tgt, mask, cfg=heatmap_cfg
        )
    )(logits_bchw, heatmap_tgt, mask_line)
    active = jnp.sum(mask_line, axis=1)
    sample_loss = jnp.where(active > 0, sample_loss_raw, jnp.nan)

    return InferAux(
        pred_coords_px=preds_orig,
        cls_embedding=cls_bd,
        mean_entropy=mean_entropy,
        sample_loss=sample_loss,
        width_line_err_cm=line_err_cm[:, 0],
        length_line_err_cm=line_err_cm[:, 1],
    )


@beartype.beartype
def batch_to_rows(aux: InferAux, metadata: dict, dataset_key: str) -> list[dict]:
    """Convert one batch of InferAux + metadata into a list of row dicts for the output DataFrame."""
    n = aux.pred_coords_px.shape[0]
    rows = []
    for i in range(n):
        rows.append({
            "beetle_id": metadata["beetle_id"][i],
            "scientific_name": metadata["scientific_name"][i],
            "group_img_basename": metadata["group_img_basename"][i],
            "img_fpath": metadata["img_fpath"][i],
            "dataset": dataset_key,
            "sample_loss": float(aux.sample_loss[i]),
            "width_line_err_cm": float(aux.width_line_err_cm[i]),
            "length_line_err_cm": float(aux.length_line_err_cm[i]),
            "mean_entropy": float(aux.mean_entropy[i]),
            "pred_coords_px": np.asarray(aux.pred_coords_px[i]).reshape(8).tolist(),
            "gt_coords_px": metadata["points_px"][i].reshape(8).tolist()
            if "points_px" in metadata
            else [float("nan")] * 8,
            "cls_embedding": np.asarray(aux.cls_embedding[i]).tolist(),
        })
    return rows


@beartype.beartype
def infer(cfg: Config):
    """Run inference on a single config. Called directly or via submitit."""
    model, _model_cfg, objective_cfg = btx.modeling.load_ckpt(cfg.ckpt_fpath)
    msg = f"Inference currently only supports heatmap objectives, got {type(objective_cfg)}"
    assert isinstance(objective_cfg, btx.objectives.heatmap.Config), msg
    heatmap_cfg = objective_cfg

    ds_cfgs: list[btx.data.Config] = [cfg.hawaii, cfg.beetlepalooza, cfg.biorepo]
    eval_aug = btx.data.transforms.AugmentConfig()
    rows: list[dict] = []
    for ds_cfg in ds_cfgs:
        if not ds_cfg.go:
            continue
        ds = ds_cfg.dataset(ds_cfg)
        dl = btx.data.make_dataloader(
            [ds],
            [eval_aug],
            seed=0,
            batch_size=cfg.batch_size,
            n_workers=cfg.n_workers,
            shuffle=False,
            finite=True,
            is_train=False,
        )
        logger.info("Running inference on %s (%d samples).", ds_cfg.key, len(ds))
        for batch in dl:
            batch, metadata = btx.helpers.to_device(batch)
            aux = forward_batch(model, batch, heatmap_cfg=heatmap_cfg)
            rows.extend(batch_to_rows(aux, metadata, ds_cfg.key))

    df = pl.DataFrame(rows, schema=SCHEMA)
    assert len(df.columns) == len(SCHEMA), (
        f"Column count mismatch: {len(df.columns)} != {len(SCHEMA)}"
    )
    cfg.out_fpath.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cfg.out_fpath)
    logger.info("Wrote %d rows to '%s'.", len(df), cfg.out_fpath)


@beartype.beartype
def main(
    cfg: tp.Annotated[Config, tyro.conf.arg(name="")],
    sweep: pathlib.Path | None = None,
):
    if sweep is None:
        cfgs = [cfg]
    else:
        sweep_dcts = btx.configs.load_sweep(sweep)
        if not sweep_dcts:
            logger.error("No valid sweeps found in '%s'.", sweep)
            return

        cfgs, errs = btx.configs.load_cfgs(cfg, default=Config(), sweep_dcts=sweep_dcts)
        if errs:
            for err in errs:
                logger.warning("Error in config: %s", err)
            return

    base = cfgs[0]
    for c in cfgs[1:]:
        msg = "Sweep configs must share slurm_acct, slurm_partition, n_hours, n_workers, and log_to."
        assert c.slurm_acct == base.slurm_acct, msg
        assert c.slurm_partition == base.slurm_partition, msg
        assert c.n_hours == base.n_hours, msg
        assert c.n_workers == base.n_workers, msg
        assert c.log_to == base.log_to, msg

    if base.slurm_acct:
        import submitit

        executor = submitit.SlurmExecutor(folder=base.log_to)
        executor.update_parameters(
            job_name="beetle-infer",
            time=int(base.n_hours * 60),
            partition=base.slurm_partition,
            gpus_per_node=1,
            ntasks_per_node=1,
            cpus_per_task=base.n_workers + 4,
            stderr_to_stdout=True,
            account=base.slurm_acct,
        )
    else:
        import submitit

        executor = submitit.DebugExecutor(folder=base.log_to)

    with executor.batch():
        jobs = [executor.submit(infer, c) for c in cfgs]

    for job in jobs:
        logger.info("Running job %s.", job.job_id)

    for job in jobs:
        job.result()


if __name__ == "__main__":
    tyro.cli(main)
