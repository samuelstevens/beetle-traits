# train.py
import dataclasses
import heapq
import logging
import pathlib
import typing as tp
from collections.abc import Iterable

import beartype
import einops
import equinox as eqx
import grain
import jax
import jax.numpy as jnp
import numpy as np
import optax
import polars as pl
import wandb
from jaxtyping import Array, Float, Int, PyTree, jaxtyped
from PIL import Image, ImageDraw

import btx.data
import btx.metrics
import btx.modeling
import btx.objectives

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)

logger = logging.getLogger("train.py")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class ValConfig:
    every: int = 1_000
    """How often to run the validation loop."""
    n_fixed: int = 5
    """Number of fixed validation images to track across training."""
    n_worst: int = 1
    """Number of worst predictions (highest error) to log."""
    n_random: int = 1
    """Number of randomly selected validation images per validation step."""


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class ValRunSpec:
    """Static inputs for one validation run.

    One instance corresponds to one `validate()` call and one logging namespace (for example, `val/hawaii`). It binds the underlying dataset source, the dataloader built from that source, and per-dataset indexing/context used for metrics and image logging.
    """

    ds: btx.data.Dataset
    """Underlying validation dataset source; used to derive `key` and `n_samples`."""
    dl: Iterable[dict[str, object]]
    """Finite validation dataloader for this dataset."""
    fixed_indices: Int[np.ndarray, " n_fixed"]
    """Stable sample indices to visualize at every validation step."""
    seen_species: frozenset[str] = frozenset()
    """Training species used for seen/unseen metrics. Empty means split metrics are disabled."""

    @property
    def key(self) -> str:
        return self.ds.cfg.key

    @property
    def n_samples(self) -> int:
        return len(self.ds)

    @property
    def prefix(self) -> str:
        return f"val/{self.key}"

    def __post_init__(self):
        msg = f"Expected positive n_samples, got {self.n_samples}"
        assert self.n_samples > 0, msg
        if self.fixed_indices.size == 0:
            return
        lo = int(np.min(self.fixed_indices))
        hi = int(np.max(self.fixed_indices))
        msg = f"Expected fixed_indices in [0, {self.n_samples}), got [{lo}, {hi}]"
        assert lo >= 0 and hi < self.n_samples, msg


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    seed: int = 17
    """Random seed."""
    model: btx.modeling.Config = btx.modeling.heatmap.Heatmap()
    """Neural network config."""
    hawaii: btx.data.HawaiiConfig = btx.data.HawaiiConfig()
    """Hawaii data config."""
    beetlepalooza: btx.data.BeetlePaloozaConfig = btx.data.BeetlePaloozaConfig()
    """BeetlePalooza data config."""
    biorepo: btx.data.BioRepoConfig = btx.data.BioRepoConfig()
    """BioRepo data config."""
    aug_hawaii: btx.data.AugmentConfig = btx.data.AugmentConfig(
        crop_scale_min=0.9,
        crop_scale_max=1.0,
        crop_ratio_min=0.95,
        crop_ratio_max=1.067,
    )
    """Augmentation config for Hawaii data."""
    aug_beetlepalooza: btx.data.AugmentConfig = btx.data.AugmentConfig(
        crop_scale_min=0.9,
        crop_scale_max=1.0,
        crop_ratio_min=0.95,
        crop_ratio_max=1.067,
    )
    """Augmentation config for Beetlepalooza data."""
    aug_biorepo: btx.data.AugmentConfig = btx.data.AugmentConfig()
    """Augmentation config for BioRepo data."""
    objective: btx.objectives.Config = btx.objectives.Heatmap()
    """Training objective configuration."""
    batch_size: int = 256
    """Batch size."""
    n_workers: int = 4
    """Number of dataloader workers."""
    tags: list[str] = dataclasses.field(default_factory=list)
    """List of wandb tags to include."""
    save_every: int = 200
    """How often to save predictions."""
    log_every: int = 200
    """How often to log to stderr."""
    val: ValConfig = ValConfig()
    """Validation schedule and sampling settings."""
    n_steps: int = 100_000
    """Total number of training steps."""
    log_to: pathlib.Path = pathlib.Path("./logs")
    learning_rate: float = 3e-4

    wandb_project: str = "beetle-traits"
    slurm_acct: str = ""
    slurm_partition: str = ""
    n_hours: float = 2.0


@beartype.beartype
def get_aug_for_dataset(cfg: Config, ds_cfg: btx.data.Config) -> btx.data.AugmentConfig:
    aug_cfg_by_key = {
        "hawaii": cfg.aug_hawaii,
        "beetlepalooza": cfg.aug_beetlepalooza,
        "biorepo": cfg.aug_biorepo,
    }
    msg = f"No augment config for dataset key '{ds_cfg.key}'"
    assert ds_cfg.key in aug_cfg_by_key, msg
    return aug_cfg_by_key[ds_cfg.key]


@beartype.beartype
def make_dataloader(
    cfg: Config,
    dss: list[btx.data.Dataset],
    *,
    shuffle: bool,
    finite: bool,
    is_train: bool,
):
    """Build a mixed Grain dataloader from one or more dataset sources.

    Args:
        cfg: Global train config, used to look up per-dataset augmentation configs.
        dss: Dataset sources to include in this loader (plural of ds).
        shuffle: Whether to shuffle each source dataset before transforms.
        finite: Whether the iterator should stop after one epoch (`True`) or repeat forever (`False`).
        is_train: Whether to build the train transform pipeline (`True`) or eval pipeline (`False`).

    Returns:
        Grain iterable dataset yielding transformed, batched samples.
    """

    datasets = []
    weights = []

    for ds in dss:
        aug_cfg = get_aug_for_dataset(cfg, ds.cfg)
        source = tp.cast(tp.Sequence[object], ds)
        mapped_ds = grain.MapDataset.source(source).seed(cfg.seed)
        if shuffle:
            mapped_ds = mapped_ds.shuffle()

        for i, tfm in enumerate(
            btx.data.transforms.make_transforms(aug_cfg, is_train=is_train)
        ):
            if isinstance(tfm, grain.transforms.RandomMap):
                mapped_ds = mapped_ds.random_map(tfm, seed=cfg.seed + i)
            else:
                mapped_ds = mapped_ds.map(tfm)

        datasets.append(mapped_ds)
        weights.append(len(ds))

    assert datasets, "No datasets provided."

    if len(datasets) == 1:
        mixed = datasets[0]
    else:
        total = sum(weights)
        assert total > 0, "All datasets are empty."
        mix_weights = [w / total for w in weights]
        mixed = grain.MapDataset.mix(datasets, weights=mix_weights)

    mixed = mixed.repeat(num_epochs=None if not finite else 1)
    mixed = mixed.batch(batch_size=cfg.batch_size, drop_remainder=False)

    iter_ds = mixed.to_iter_dataset(
        read_options=grain.ReadOptions(num_threads=2, prefetch_buffer_size=8)
    )

    if cfg.n_workers > 0:
        iter_ds = iter_ds.mp_prefetch(
            grain.multiprocessing.MultiprocessingOptions(
                num_workers=cfg.n_workers, per_worker_buffer_size=2
            )
        )

    return iter_ds


@jaxtyped(typechecker=beartype.beartype)
class TrainAux(eqx.Module):
    """Training/validation outputs that combine objective and geometry metrics.

    Groups:
    1) Core objective outputs (`preds`, `loss`, `sample_loss`) in augmented-image space.
    2) `sample_loss` is per-sample masked objective loss in augmented-image space.
    3) Error tensors split by geometry: `point_*` are endpoint distances, `line_*` are absolute line-length errors (both in centimeters).
    4) Optional per-trait summaries (`width_*`, `length_*`) and batch-level data-quality metric (`oob_points_frac`).
    5) `obj_metrics` stores objective-specific per-sample diagnostics keyed by metric name.
    """

    # Core optimization/prediction outputs.
    loss: Float[Array, ""]
    """Scalar training loss in augmented-image space."""
    preds: Float[Array, "batch 2 2 2"]
    """Predicted endpoints in augmented-image coordinates, shape [batch, lines, points, xy]."""

    # Per-sample augmented-space loss proxy and cm geometric errors.
    sample_loss: Float[Array, " batch"]
    """Per-sample masked objective loss in augmented-image space, shape [batch]."""
    point_err_cm: Float[Array, " batch points"]
    """Per-point Euclidean error in centimeters in original-image space, flattened per sample."""
    line_err_cm: Float[Array, " batch lines"]
    """Absolute line-length error in centimeters in original-image space, shape [batch, lines]."""

    # Width/length summaries from point errors in centimeters.
    width_point_err_cm: Float[Array, " batch"]
    """Mean width-point error in centimeters, shape [batch]."""
    length_point_err_cm: Float[Array, " batch"]
    """Mean length-point error in centimeters, shape [batch]."""

    # Width/length summaries from line-length errors in centimeters.
    width_line_err_cm: Float[Array, " batch"]
    """Width line-length absolute error in centimeters, shape [batch]."""
    length_line_err_cm: Float[Array, " batch"]
    """Length line-length absolute error in centimeters, shape [batch]."""

    # Batch-level data quality.
    oob_points_frac: Float[Array, ""]
    """Batch-mean fraction of out-of-bounds target points after augmentation."""
    obj_metrics: dict[str, Float[Array, " batch"]]
    """Objective-specific per-sample metric arrays keyed by metric name."""

    def metrics(self) -> dict:
        metrics = {
            "loss": self.loss,
            "point_err_cm": self.point_err_cm,
            "line_err_cm": self.line_err_cm,
            "width_point_err_cm": self.width_point_err_cm,
            "length_point_err_cm": self.length_point_err_cm,
            "width_line_err_cm": self.width_line_err_cm,
            "length_line_err_cm": self.length_line_err_cm,
            "oob_points_frac": self.oob_points_frac,
        }
        metrics |= self.obj_metrics
        return metrics


@eqx.filter_jit()
@jaxtyped(typechecker=beartype.beartype)
def loss_and_aux(
    diff_model: eqx.Module,
    static_model: eqx.Module,
    batch: dict[str, Array],
    *,
    obj: btx.objectives.Obj,
) -> tuple[Float[Array, ""], TrainAux]:
    required_keys = [
        "img",
        "tgt",
        "loss_mask",
        "points_px",
        "scalebar_px",
        "scalebar_valid",
        "t_orig_from_aug",
        "oob_points_frac",
    ]
    missing = tuple(key for key in required_keys if key not in batch)
    msg = f"Missing required batch keys for loss: {missing}"
    assert not missing, msg

    # 1) Forward pass in augmented-image space.
    model = eqx.combine(diff_model, static_model)
    forward = tp.cast(tp.Callable[[Array], Array], model)
    preds_raw = jax.vmap(forward)(batch["img"])

    mask_line = batch["loss_mask"]
    msg = f"Expected loss_mask shape [batch, 2], got {mask_line.shape}"
    assert mask_line.ndim == 2 and mask_line.shape[1] == 2, msg
    mask_point = mask_line[:, :, None]
    msg = f"Expected tgt shape [batch, 2, 2, 2], got {batch['tgt'].shape}"
    assert batch["tgt"].shape[1:] == (2, 2, 2), msg

    # 2) Objective-specific loss path via polymorphic objective implementation.
    obj_aux = obj.get_loss_aux(preds_raw=preds_raw, batch=batch, mask_line=mask_line)
    obj_metrics = {f"{obj.key}/{k}": v for k, v in obj_aux.metrics.items()}

    # 3) Move predictions back to original-image coordinates and resolve endpoint ordering.
    preds_orig = btx.metrics.apply_affine(batch["t_orig_from_aug"], obj_aux.preds)
    tgts_orig = btx.metrics.choose_endpoint_matching(preds_orig, batch["points_px"])

    # 4) Compute physical errors in centimeters.
    point_err_px_line = jnp.linalg.norm(preds_orig - tgts_orig, axis=-1)
    scalebar_valid, px_per_cm = btx.metrics.get_scalebar_mask(
        batch["scalebar_px"], batch["scalebar_valid"]
    )
    point_err_cm_line = point_err_px_line / px_per_cm[:, None, None]
    tgts_start_orig, tgts_end_orig = jnp.unstack(tgts_orig, axis=2)
    tgts_line_orig = jnp.linalg.norm(tgts_start_orig - tgts_end_orig, axis=-1)
    preds_start_orig, preds_end_orig = jnp.unstack(preds_orig, axis=2)
    preds_line_orig = jnp.linalg.norm(preds_start_orig - preds_end_orig, axis=-1)
    line_err_cm = jnp.abs(preds_line_orig - tgts_line_orig) / px_per_cm[:, None]

    # 5) Apply both supervision and unit-conversion validity masks, then flatten point errors.
    metric_mask_line = mask_line * scalebar_valid[:, None]
    metric_mask_point = mask_point * scalebar_valid[:, None, None]
    point_err_cm_line = jnp.where(metric_mask_point > 0, point_err_cm_line, jnp.nan)
    line_err_cm = jnp.where(metric_mask_line > 0, line_err_cm, jnp.nan)
    point_err_cm = einops.rearrange(
        point_err_cm_line, "batch lines points -> batch (lines points)"
    )

    # 6) Build per-trait summaries and batch-level data quality.
    width_point_err_cm = jnp.nanmean(point_err_cm_line[:, 0], axis=1)
    length_point_err_cm = jnp.nanmean(point_err_cm_line[:, 1], axis=1)

    width_line_err_cm = line_err_cm[:, 0]
    length_line_err_cm = line_err_cm[:, 1]
    oob_points_frac = jnp.nanmean(batch["oob_points_frac"])

    # 7) Package all logging/analysis outputs.
    return obj_aux.loss, TrainAux(
        obj_aux.loss,
        obj_aux.preds,
        obj_aux.sample_loss,
        point_err_cm,
        line_err_cm,
        width_point_err_cm,
        length_point_err_cm,
        width_line_err_cm,
        length_line_err_cm,
        oob_points_frac,
        obj_metrics,
    )


@eqx.filter_jit()
@jaxtyped(typechecker=beartype.beartype)
def step_model(
    model: eqx.Module,
    optim: optax.GradientTransformation,
    state: tp.Any,
    batch: dict[str, Array],
    filter_spec: PyTree[bool],
    *,
    obj: btx.objectives.Obj,
) -> tuple[eqx.Module, tp.Any, TrainAux]:
    diff_model, static_model = eqx.partition(model, filter_spec)

    loss_fn = eqx.filter_value_and_grad(loss_and_aux, has_aux=True)
    (loss, aux), grads = loss_fn(diff_model, static_model, batch, obj=obj)

    updates, new_state = optim.update(grads, state, diff_model)
    diff_model = eqx.apply_updates(diff_model, updates)
    model = eqx.combine(diff_model, static_model)

    return model, new_state, aux


@beartype.beartype
def get_trainable_filter_spec(model: eqx.Module) -> PyTree[bool]:
    """Build a trainable-parameter mask that freezes ViT and trains non-ViT arrays.

    Args:
        model: Full model module used for training.

    Returns:
        Pytree of booleans matching `model` leaves. `True` marks differentiable
        parameters to optimize, `False` marks frozen/static leaves.
    """
    filter_spec = jax.tree_util.tree_map(eqx.is_array, model)
    if not hasattr(model, "vit"):
        return filter_spec

    vit = tp.cast(eqx.Module, getattr(model, "vit"))
    frozen_vit = jax.tree_util.tree_map(lambda _: False, vit)
    return eqx.tree_at(lambda tree: tree.vit, filter_spec, frozen_vit)


@jaxtyped(typechecker=beartype.beartype)
def plot_preds(
    batch: dict[str, Array],
    metadata: dict[str, object],
    preds: Float[Array, "batch 2 2 2"],
    sample_idx: int,
) -> tuple[str, Image.Image]:
    img_fpaths = tp.cast(list[str], metadata["img_fpath"])
    img_fpath = img_fpaths[sample_idx]
    img = Image.open(img_fpath)

    # Get ground truth points in original image coordinates.
    gt_width_px, gt_length_px = np.asarray(batch["points_px"][sample_idx])

    # Map predictions from augmented image space back to original image space.
    i = sample_idx
    pred_orig = btx.metrics.apply_affine(
        batch["t_orig_from_aug"][i : i + 1], preds[i : i + 1]
    )[0]
    pred_width_px, pred_length_px = np.asarray(pred_orig)

    # Draw on image
    draw = ImageDraw.Draw(img)

    # Draw ground truth in green
    draw.line(
        [tuple(gt_length_px[0]), tuple(gt_length_px[1])], fill=(0, 255, 0), width=3
    )
    draw.line([tuple(gt_width_px[0]), tuple(gt_width_px[1])], fill=(0, 255, 0), width=3)

    # Draw predictions in red
    draw.line(
        [tuple(pred_length_px[0]), tuple(pred_length_px[1])], fill=(255, 0, 0), width=3
    )
    draw.line(
        [tuple(pred_width_px[0]), tuple(pred_width_px[1])], fill=(255, 0, 0), width=3
    )

    # Draw points as circles for clarity
    radius = 4
    for pt in gt_length_px:
        x, y = pt
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=(0, 255, 0))
    for pt in gt_width_px:
        x, y = pt
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=(0, 255, 0))
    for pt in pred_length_px:
        x, y = pt
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=(255, 0, 0))
    for pt in pred_width_px:
        x, y = pt
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=(255, 0, 0))

    # Extract individual ID from filepath
    beetle_ids = tp.cast(list[str], metadata["beetle_id"])
    beetle_id = beetle_ids[sample_idx]

    return beetle_id, img


@jaxtyped(typechecker=beartype.beartype)
def validate(
    cfg: Config,
    model: eqx.Module,
    filter_spec: PyTree[bool],
    spec: ValRunSpec,
    *,
    obj: btx.objectives.Obj,
) -> dict[str, float | wandb.Image]:
    """Run validation on one dataset loader and return scalar metrics and image artifacts.

    Args:
        cfg: Global training config with logging/selection settings.
        model: Current model parameters.
        filter_spec: Equinox filter spec used to partition trainable/static leaves.
        spec: Per-dataset validation inputs (dataset source, dataloader, fixed indices, and seen-species split context).

    Returns:
        Flat dict containing aggregate validation metrics and optional `wandb.Image` entries.
    """

    metrics = []

    # Fixed indices - same samples tracked across training
    fixed_batch_idxs = spec.fixed_indices // cfg.batch_size
    fixed_sample_batch_idxs = spec.fixed_indices % cfg.batch_size

    # Random indices - different samples each validation step
    random_indices = np.random.default_rng().choice(
        spec.n_samples, size=min(cfg.val.n_random, spec.n_samples), replace=False
    )
    random_batch_idxs = random_indices // cfg.batch_size
    random_sample_batch_idxs = random_indices % cfg.batch_size

    images = {}
    worst_candidates = []
    seen_metrics = []
    unseen_metrics = []

    diff_model, static_model = eqx.partition(model, filter_spec)
    for i, batch in enumerate(spec.dl):
        batch, metadata = to_device(batch)
        loss, aux = loss_and_aux(
            diff_model,
            static_model,
            batch,
            obj=obj,
        )
        metrics.append(aux.metrics())

        # Track seen vs unseen species metrics
        if spec.seen_species:
            is_seen = jnp.array([
                name in spec.seen_species for name in metadata["scientific_name"]
            ])

            seen_metrics.append({
                "line_err_cm": jnp.where(is_seen[:, None], aux.line_err_cm, jnp.nan),
                "point_err_cm": jnp.where(is_seen[:, None], aux.point_err_cm, jnp.nan),
                "loss": jnp.where(is_seen, aux.sample_loss, jnp.nan),
            })
            unseen_metrics.append({
                "line_err_cm": jnp.where(~is_seen[:, None], aux.line_err_cm, jnp.nan),
                "point_err_cm": jnp.where(~is_seen[:, None], aux.point_err_cm, jnp.nan),
                "loss": jnp.where(~is_seen, aux.sample_loss, jnp.nan),
            })

        # check to print any of the fixed images
        actual_batch_size = aux.preds.shape[0]
        for batch_idx, sample_idx in zip(fixed_batch_idxs, fixed_sample_batch_idxs):
            if i == batch_idx and sample_idx < actual_batch_size:
                beetle_id, img = plot_preds(batch, metadata, aux.preds, int(sample_idx))
                images[f"images/fixed/beetle{beetle_id}"] = wandb.Image(img)

        # check to print any of the random images
        for batch_idx, sample_idx in zip(random_batch_idxs, random_sample_batch_idxs):
            if i == batch_idx and sample_idx < actual_batch_size:
                beetle_id, img = plot_preds(batch, metadata, aux.preds, int(sample_idx))
                images[f"images/random/beetle{beetle_id}"] = wandb.Image(img)

        # track worst predictions by line_err_cm using a min-heap for efficiency
        # Extract only the data needed for plotting to avoid keeping entire batch arrays in memory
        sample_errors = jnp.nanmean(aux.line_err_cm, axis=1)  # take mean of the lines
        for j, err in enumerate(sample_errors):
            if jnp.isnan(err):
                continue
            err_val = float(err)
            # Extract single-sample batch/metadata/preds for plot_preds compatibility
            sample_batch = {
                "points_px": jnp.asarray(batch["points_px"][j])[jnp.newaxis],
                "t_orig_from_aug": jnp.asarray(batch["t_orig_from_aug"][j])[
                    jnp.newaxis
                ],
            }
            sample_metadata = {
                "img_fpath": [metadata["img_fpath"][j]],
                "beetle_id": [metadata["beetle_id"][j]],
            }
            sample_preds = jnp.asarray(aux.preds[j])[jnp.newaxis]
            candidate = (err_val, sample_batch, sample_metadata, sample_preds)
            if len(worst_candidates) < cfg.val.n_worst:
                heapq.heappush(worst_candidates, candidate)
            elif err_val > worst_candidates[0][0]:
                heapq.heapreplace(worst_candidates, candidate)

    # plot worst predictions
    for err, sample_batch, sample_metadata, sample_preds in worst_candidates:
        beetle_id, img = plot_preds(sample_batch, sample_metadata, sample_preds, 0)
        images[f"images/worst/beetle{beetle_id}"] = wandb.Image(img)

    metrics = {
        k: jnp.concatenate([dct[k].reshape(-1) for dct in metrics]) for k in metrics[0]
    }

    means = {k: jnp.nanmean(v).item() for k, v in metrics.items()}
    maxes = {f"max_{k}": jnp.nanmax(v).item() for k, v in metrics.items()}
    medians = {f"median_{k}": jnp.nanmedian(v).item() for k, v in metrics.items()}

    # Compute seen vs unseen species metrics
    seen_unseen_metrics = {}
    if spec.seen_species and seen_metrics:
        all_seen = {
            k: jnp.concatenate([m[k].reshape(-1) for m in seen_metrics], axis=0)
            for k in seen_metrics[0]
        }
        all_unseen = {
            k: jnp.concatenate([m[k].reshape(-1) for m in unseen_metrics], axis=0)
            for k in unseen_metrics[0]
        }

        for key in ["line_err_cm", "point_err_cm", "loss"]:
            seen_unseen_metrics[f"seen_{key}"] = jnp.nanmean(all_seen[key]).item()
            seen_unseen_metrics[f"unseen_{key}"] = jnp.nanmean(all_unseen[key]).item()
            seen_unseen_metrics[f"seen_max_{key}"] = jnp.nanmax(all_seen[key]).item()
            seen_unseen_metrics[f"unseen_max_{key}"] = jnp.nanmax(
                all_unseen[key]
            ).item()

    return {
        **means,
        **maxes,
        **medians,
        **seen_unseen_metrics,
        **images,
    }


@beartype.beartype
def get_training_species(cfg: "Config") -> frozenset[str]:
    """Collect scientific names from training datasets (Hawaii train + BeetlePalooza).

    Important: Only includes species from the TRAIN split, not validation.
    """
    species: set[str] = set()

    # Hawaii training species - must filter by train split
    if cfg.hawaii.go:
        hawaii_train_cfg = dataclasses.replace(cfg.hawaii, split="train")
        hawaii_source = hawaii_train_cfg.dataset(hawaii_train_cfg)
        hawaii_species = (
            hawaii_source.df.get_column("scientific_name").unique().to_list()
        )
        species.update(hawaii_species)
        logger.info("Hawaii training species: %d unique", len(hawaii_species))

    # BeetlePalooza species (no splits, all samples are training)
    if cfg.beetlepalooza.go:
        palooza_df = pl.read_json(cfg.beetlepalooza.annotations)
        palooza_species = palooza_df.get_column("scientific_name").unique().to_list()
        species.update(palooza_species)
        logger.info("BeetlePalooza training species: %d unique", len(palooza_species))

    logger.info("Total unique training species: %d", len(species))
    return frozenset(species)


@beartype.beartype
def is_device_array(x: object) -> bool:
    if isinstance(x, (jax.Array, np.ndarray)):
        dt = getattr(x, "dtype", None)
        if dt is None:
            return False
        return (
            np.issubdtype(dt, np.bool_)
            or np.issubdtype(dt, np.integer)
            or np.issubdtype(dt, np.unsignedinteger)
            or np.issubdtype(dt, np.floating)
            or np.issubdtype(dt, np.complexfloating)
        )
    return False


@beartype.beartype
def to_device(batch: dict[str, object], device=None) -> tuple:
    numeric = {k: v for k, v in batch.items() if is_device_array(v)}
    aux = {k: v for k, v in batch.items() if not is_device_array(v)}
    # device_put works on pytrees; leaves become jax.Arrays on the target device
    numeric = jax.device_put(numeric, device)
    return numeric, aux


@beartype.beartype
def train(cfg: Config):
    key = jax.random.key(seed=cfg.seed)
    biorepo_val_cfg = dataclasses.replace(cfg.biorepo, split="val")
    msg = "BioRepo val must be included: set cfg.biorepo.go=True."
    assert biorepo_val_cfg.go, msg

    val_cfgs = [dataclasses.replace(cfg.hawaii, split="val"), biorepo_val_cfg]
    val_cfgs = [dataset_cfg for dataset_cfg in val_cfgs if dataset_cfg.go]
    training_species = get_training_species(cfg)

    # Create separate validation dataloaders for each dataset
    val_run_specs = []
    for val_cfg in val_cfgs:
        ds = val_cfg.dataset(val_cfg)
        msg = (
            f"{val_cfg.key} val has {len(ds)} samples, expected at least "
            f"cfg.val.n_fixed={cfg.val.n_fixed}."
        )
        assert len(ds) >= cfg.val.n_fixed, msg
        dl = make_dataloader(cfg, [ds], shuffle=False, finite=True, is_train=False)

        rng = np.random.default_rng(seed=cfg.seed)
        fixed_indices = rng.choice(len(ds), size=cfg.val.n_fixed, replace=False)
        seen_species = training_species if val_cfg.key == "biorepo" else frozenset()

        val_run_specs.append(ValRunSpec(ds, dl, fixed_indices, seen_species))
        logger.info("%s validation: fixed indices: %s", val_cfg.key, fixed_indices)

    train_cfgs = [dataclasses.replace(cfg.hawaii, split="train"), cfg.beetlepalooza]
    train_dss = [ds_cfg.dataset(ds_cfg) for ds_cfg in train_cfgs if ds_cfg.go]
    train_dl = make_dataloader(
        cfg, train_dss, shuffle=True, finite=False, is_train=True
    )

    model = btx.modeling.make(cfg.model, key)

    # Freeze ViT arrays, optimize non-ViT arrays.
    filter_spec = get_trainable_filter_spec(model)

    # Optimize only the differentiable part of the model.
    diff_model, _ = eqx.partition(model, filter_spec)
    optim = optax.adamw(learning_rate=cfg.learning_rate)
    state = optim.init(diff_model)
    obj_cfg = cfg.objective
    obj = obj_cfg.get_obj()

    run = wandb.init(
        project=cfg.wandb_project,
        config=dataclasses.asdict(cfg),
        tags=cfg.tags,
        # Hidden wandb folder
        dir=".wandb",
    )

    # Training
    for step, batch in enumerate(train_dl):
        batch, metadata = to_device(batch)
        model, state, aux = step_model(
            model,
            optim,
            state,
            batch,
            filter_spec,
            obj=obj,
        )

        if step % cfg.save_every == 0:
            # since batch is shuffled, choose elem 0
            beetle_id, img = plot_preds(batch, metadata, aux.preds, 0)
            run.log(
                {"step": step, f"images/train/beetle{beetle_id}": wandb.Image(img)},
                step=step,
            )

        if step % cfg.log_every == 0:
            metrics = {
                f"train/{key}": jnp.nanmean(value).item()
                for key, value in aux.metrics().items()
            }
            metrics["step"] = step
            run.log(metrics, step=step)
            logger.info(
                "Step: %d %s",
                step,
                {"step": step, "train/loss": metrics["train/loss"]},
            )

        if step % cfg.val.every == 0:
            val_metrics = {}
            for run_spec in val_run_specs:
                run_metrics = validate(cfg, model, filter_spec, run_spec, obj=obj)
                run_metrics = {
                    f"{run_spec.prefix}/{key}": value
                    for key, value in run_metrics.items()
                }
                val_metrics.update(run_metrics)

            run.log(val_metrics, step=step)
            logger.info("Validation: %d %s", step, val_metrics)

        if step >= cfg.n_steps:
            break
