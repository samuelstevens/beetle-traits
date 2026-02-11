# train.py
import dataclasses
import heapq
import logging
import pathlib
import typing as tp

import beartype
import einops
import equinox as eqx
import grain
import jax
import jax.numpy as jnp
import numpy as np
import optax
import polars as pl
from jaxtyping import Array, Float, PyTree, jaxtyped
from PIL import Image, ImageDraw

import btx.data
import btx.metrics
import btx.modeling
import wandb

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("train.py")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    seed: int = 17
    """Random seed."""
    model: btx.modeling.Config = btx.modeling.frozen.Frozen()
    """Neural network config."""
    hawaii: btx.data.HawaiiConfig = btx.data.HawaiiConfig()
    """Hawaii data config."""
    beetlepalooza: btx.data.BeetlePaloozaConfig = btx.data.BeetlePaloozaConfig()
    """BeetlePalooza data config."""
    biorepo: btx.data.BioRepoConfig = btx.data.BioRepoConfig()
    """BioRepo data config."""
    augment: btx.data.AugmentConfig = btx.data.AugmentConfig()
    """Default train/eval augmentation + normalization config."""
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
    val_every: int = 1_000
    """How often to run the validation loop."""
    n_val_fixed: int = 5
    """Number of fixed validation images to track across training."""
    n_val_worst: int = 1
    """Number of worst predictions (highest error) to log."""
    n_val_random: int = 1
    """Number of randomly selected validation images per validation step."""
    n_steps: int = 100_000
    """Total number of training steps."""
    log_to: pathlib.Path = pathlib.Path("./logs")
    learning_rate: float = 3e-4

    wandb_project: str = "beetle-traits"
    slurm_acct: str = ""
    slurm_partition: str = ""
    n_hours: float = 2.0


@beartype.beartype
def get_transforms(
    cfg: btx.data.AugmentConfig, *, train_split: bool
) -> list[grain.transforms.Map | grain.transforms.RandomMap]:
    tfms: list[grain.transforms.Map | grain.transforms.RandomMap] = [
        btx.data.utils.DecodeRGB(),
        btx.data.augment.InitAugState(size=cfg.size, min_px_per_cm=cfg.min_px_per_cm),
    ]
    if not train_split or not cfg.go:
        tfms.append(btx.data.augment.Resize(size=cfg.size))
        tfms.append(btx.data.augment.FinalizeTargets(cfg=cfg))
        if cfg.normalize:
            tfms.append(btx.data.utils.Normalize())
        return tfms

    tfms.extend([
        btx.data.augment.RandomResizedCrop(cfg=cfg),
        btx.data.augment.RandomFlip(cfg=cfg),
        btx.data.augment.RandomRotation90(cfg=cfg),
        btx.data.augment.ColorJitter(cfg=cfg),
        btx.data.augment.FinalizeTargets(cfg=cfg),
    ])
    if cfg.normalize:
        tfms.append(btx.data.utils.Normalize())
    return tfms


@beartype.beartype
def get_aug_for_dataset(
    train_cfg: Config, dataset_cfg: btx.data.Config
) -> btx.data.AugmentConfig:
    if isinstance(dataset_cfg, btx.data.HawaiiConfig):
        return train_cfg.aug_hawaii
    if isinstance(dataset_cfg, btx.data.BeetlePaloozaConfig):
        return train_cfg.aug_beetlepalooza
    if isinstance(dataset_cfg, btx.data.BioRepoConfig):
        return train_cfg.aug_biorepo
    return train_cfg.augment


@beartype.beartype
def make_dataset(
    cfgs: list[btx.data.Config],
    *,
    seed: int,
    batch_size: int,
    n_workers: int,
    shuffle: bool,
    finite: bool,
    train_cfg: Config,
    train: bool,
):
    datasets = []
    weights = []

    for dataset_cfg in cfgs:
        source = dataset_cfg.dataset(dataset_cfg)
        assert isinstance(source, grain.sources.RandomAccessDataSource)

        if len(source) == 0:
            continue

        ds = grain.MapDataset.source(source).seed(seed)
        if shuffle:
            ds = ds.shuffle()

        aug_cfg = get_aug_for_dataset(train_cfg, dataset_cfg)
        for i, tfm in enumerate(get_transforms(aug_cfg, train_split=train)):
            if isinstance(tfm, grain.transforms.RandomMap):
                ds = ds.random_map(tfm, seed=seed + i)
            else:
                ds = ds.map(tfm)

        datasets.append(ds)
        weights.append(len(source))

    assert datasets, "No datasets provided."

    if len(datasets) == 1:
        mixed = datasets[0]
    else:
        total = sum(weights)
        assert total > 0, "All datasets are empty."
        mix_weights = [w / total for w in weights]
        mixed = grain.MapDataset.mix(datasets, weights=mix_weights)

    epochs = None if not finite else 1
    mixed = mixed.repeat(num_epochs=epochs)

    mixed = mixed.batch(batch_size=batch_size, drop_remainder=False)

    iter_ds = mixed.to_iter_dataset(
        read_options=grain.ReadOptions(num_threads=2, prefetch_buffer_size=8)
    )

    if n_workers > 0:
        iter_ds = iter_ds.mp_prefetch(
            grain.multiprocessing.MultiprocessingOptions(
                num_workers=n_workers, per_worker_buffer_size=2
            )
        )

    return iter_ds


@jaxtyped(typechecker=beartype.beartype)
class Aux(eqx.Module):
    loss: Float[Array, ""]
    preds: Float[Array, "batch 2 2 2"]
    point_err_raw: Float[Array, " batch points"]
    point_err_cm: Float[Array, " batch points"]
    line_err_raw: Float[Array, " batch lines"]
    line_err_cm: Float[Array, " batch lines"]
    width_point_err_raw: Float[Array, " batch"]
    length_point_err_raw: Float[Array, " batch"]
    width_point_err_cm: Float[Array, " batch"]
    length_point_err_cm: Float[Array, " batch"]
    width_line_err_raw: Float[Array, " batch"]
    length_line_err_raw: Float[Array, " batch"]
    width_line_err_cm: Float[Array, " batch"]
    length_line_err_cm: Float[Array, " batch"]
    oob_points_frac: Float[Array, ""]

    def metrics(self):
        return {
            "loss": self.loss,
            "point_err_cm": self.point_err_cm,
            "line_err_cm": self.line_err_cm,
            "width_point_err_cm": self.width_point_err_cm,
            "length_point_err_cm": self.length_point_err_cm,
            "width_line_err_cm": self.width_line_err_cm,
            "length_line_err_cm": self.length_line_err_cm,
            "oob_points_frac": self.oob_points_frac,
        }


@eqx.filter_jit()
@jaxtyped(typechecker=beartype.beartype)
def loss_and_aux(
    diff_model: eqx.Module,
    static_model: eqx.Module,
    batch: dict[str, Array],
    *,
    min_px_per_cm: float,
) -> tuple[Float[Array, ""], Aux]:
    model = eqx.combine(diff_model, static_model)
    preds = jax.vmap(model)(batch["img"])

    # Apply loss mask to exclude certain measurements (e.g., BeetlePalooza width)
    squared_error = (preds - batch["tgt"]) ** 2
    mask = einops.rearrange(batch["loss_mask"], "b l -> b l () ()")
    masked_error = squared_error * mask
    # Each mask element covers 2 points x 2 coords = 4 values, so active_lines counts unmasked coordinate values.
    active_lines = jnp.sum(mask) * 4
    mse = jnp.where(active_lines > 0, jnp.sum(masked_error) / active_lines, 0.0)

    # Metrics
    mask_line = batch["loss_mask"]
    mask_point = mask_line[:, :, None]

    # Raw space (augmented image space, same space as loss)
    point_err_raw_line = jnp.linalg.norm(preds - batch["tgt"], axis=-1)
    tgts_start_raw, tgts_end_raw = jnp.unstack(batch["tgt"], axis=2)
    tgts_line_raw = jnp.linalg.norm(tgts_start_raw - tgts_end_raw, axis=-1)
    preds_start_raw, preds_end_raw = jnp.unstack(preds, axis=2)
    preds_line_raw = jnp.linalg.norm(preds_start_raw - preds_end_raw, axis=-1)
    line_err_raw = jnp.abs(preds_line_raw - tgts_line_raw)

    # Physical metrics in original-image space.
    preds_orig = btx.metrics.apply_affine_jax(batch["t_orig_from_aug"], preds)
    tgts_orig = btx.metrics.choose_endpoint_matching(preds_orig, batch["points_px"])
    point_err_px_line = jnp.linalg.norm(preds_orig - tgts_orig, axis=-1)
    metric_mask_cm, px_per_cm = btx.metrics.get_metric_mask_cm(
        batch["scalebar_px"], batch["metric_mask_cm"], min_px_per_cm=min_px_per_cm
    )
    point_err_cm_line = point_err_px_line / px_per_cm[:, None, None]
    tgts_start_orig, tgts_end_orig = jnp.unstack(tgts_orig, axis=2)
    tgts_line_orig = jnp.linalg.norm(tgts_start_orig - tgts_end_orig, axis=-1)
    preds_start_orig, preds_end_orig = jnp.unstack(preds_orig, axis=2)
    preds_line_orig = jnp.linalg.norm(preds_start_orig - preds_end_orig, axis=-1)
    line_err_cm = jnp.abs(preds_line_orig - tgts_line_orig) / px_per_cm[:, None]

    metric_mask_line = mask_line * metric_mask_cm[:, None]
    metric_mask_point = mask_point * metric_mask_cm[:, None, None]
    point_err_raw_line = jnp.where(mask_point > 0, point_err_raw_line, jnp.nan)
    point_err_cm_line = jnp.where(metric_mask_point > 0, point_err_cm_line, jnp.nan)
    line_err_raw = jnp.where(mask_line > 0, line_err_raw, jnp.nan)
    line_err_cm = jnp.where(metric_mask_line > 0, line_err_cm, jnp.nan)

    point_err_raw = einops.rearrange(
        point_err_raw_line, "batch lines points -> batch (lines points)"
    )
    point_err_cm = einops.rearrange(
        point_err_cm_line, "batch lines points -> batch (lines points)"
    )

    width_point_err_raw = jnp.nanmean(point_err_raw_line[:, 0], axis=1)
    length_point_err_raw = jnp.nanmean(point_err_raw_line[:, 1], axis=1)
    width_point_err_cm = jnp.nanmean(point_err_cm_line[:, 0], axis=1)
    length_point_err_cm = jnp.nanmean(point_err_cm_line[:, 1], axis=1)

    width_line_err_raw = line_err_raw[:, 0]
    length_line_err_raw = line_err_raw[:, 1]
    width_line_err_cm = line_err_cm[:, 0]
    length_line_err_cm = line_err_cm[:, 1]
    oob_points_frac = jnp.nanmean(batch["oob_points_frac"])

    return mse, Aux(
        mse,
        preds,
        point_err_raw,
        point_err_cm,
        line_err_raw,
        line_err_cm,
        width_point_err_raw,
        length_point_err_raw,
        width_point_err_cm,
        length_point_err_cm,
        width_line_err_raw,
        length_line_err_raw,
        width_line_err_cm,
        length_line_err_cm,
        oob_points_frac,
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
    min_px_per_cm: float,
) -> tuple[eqx.Module, tp.Any, Aux]:
    diff_model, static_model = eqx.partition(model, filter_spec)

    loss_fn = eqx.filter_value_and_grad(loss_and_aux, has_aux=True)
    (loss, aux), grads = loss_fn(
        diff_model, static_model, batch, min_px_per_cm=min_px_per_cm
    )

    updates, new_state = optim.update(grads, state, diff_model)
    diff_model = eqx.apply_updates(diff_model, updates)
    model = eqx.combine(diff_model, static_model)

    return model, new_state, aux


@jaxtyped(typechecker=beartype.beartype)
def plot_preds(
    batch: dict[str, Array],
    metadata: dict[str, object],
    preds: Float[Array, "batch 2 2 2"],
    sample_idx: int,
) -> tuple[str, Image.Image]:
    img_fpath = metadata["img_fpath"][sample_idx]
    img = Image.open(img_fpath)

    # Get ground truth points in original image coordinates.
    gt_width_px, gt_length_px = np.asarray(batch["points_px"][sample_idx])

    # Map predictions from augmented image space back to original image space.
    i = sample_idx
    pred_orig = btx.metrics.apply_affine_jax(
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
    beetle_id = metadata["beetle_id"][sample_idx]

    return beetle_id, img


@beartype.beartype
def validate(
    cfg: Config,
    model: eqx.Module,
    val_dl,
    filter_spec: PyTree[bool],
    *,
    min_px_per_cm: float,
    fixed_indices: np.ndarray,
    val_total_samples: int,
    prefix: str = "val",
    training_species: set[str] | None = None,
):
    metrics = []

    # Fixed indices - same samples tracked across training
    fixed_batch_idxs = fixed_indices // cfg.batch_size
    fixed_sample_batch_idxs = fixed_indices % cfg.batch_size

    # Random indices - different samples each validation step
    random_rng = np.random.default_rng()
    random_indices = random_rng.choice(
        val_total_samples, size=min(cfg.n_val_random, val_total_samples), replace=False
    )
    random_batch_idxs = random_indices // cfg.batch_size
    random_sample_batch_idxs = random_indices % cfg.batch_size

    images = {}
    worst_candidates = []
    seen_metrics = []
    unseen_metrics = []

    for i, batch in enumerate(val_dl):
        batch, metadata = to_device(batch)
        diff_model, static_model = eqx.partition(model, filter_spec)
        loss, aux = loss_and_aux(
            diff_model,
            static_model,
            batch,
            min_px_per_cm=min_px_per_cm,
        )
        metrics.append(aux.metrics())

        # Track seen vs unseen species metrics
        if training_species is not None:
            is_seen = jnp.array([
                name in training_species for name in metadata["scientific_name"]
            ])
            sample_loss = jnp.nanmean(aux.point_err_raw**2, axis=1)

            seen_metrics.append({
                "line_err_cm": jnp.where(is_seen[:, None], aux.line_err_cm, jnp.nan),
                "point_err_cm": jnp.where(is_seen[:, None], aux.point_err_cm, jnp.nan),
                "loss": jnp.where(is_seen, sample_loss, jnp.nan),
            })
            unseen_metrics.append({
                "line_err_cm": jnp.where(~is_seen[:, None], aux.line_err_cm, jnp.nan),
                "point_err_cm": jnp.where(~is_seen[:, None], aux.point_err_cm, jnp.nan),
                "loss": jnp.where(~is_seen, sample_loss, jnp.nan),
            })

        # check to print any of the fixed images
        actual_batch_size = aux.preds.shape[0]
        for batch_idx, sample_idx in zip(fixed_batch_idxs, fixed_sample_batch_idxs):
            if i == batch_idx and sample_idx < actual_batch_size:
                beetle_id, img = plot_preds(batch, metadata, aux.preds, int(sample_idx))
                images[f"images/{prefix}/fixed/beetle{beetle_id}"] = wandb.Image(img)

        # check to print any of the random images
        for batch_idx, sample_idx in zip(random_batch_idxs, random_sample_batch_idxs):
            if i == batch_idx and sample_idx < actual_batch_size:
                beetle_id, img = plot_preds(batch, metadata, aux.preds, int(sample_idx))
                images[f"images/{prefix}/random/beetle{beetle_id}"] = wandb.Image(img)

        # track worst predictions by line_err_raw using a min-heap for efficiency
        # Extract only the data needed for plotting to avoid keeping entire batch arrays in memory
        sample_errors = jnp.nanmean(aux.line_err_raw, axis=1)  # take mean of the lines
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
            if len(worst_candidates) < cfg.n_val_worst:
                heapq.heappush(worst_candidates, candidate)
            elif err_val > worst_candidates[0][0]:
                heapq.heapreplace(worst_candidates, candidate)

    # plot worst predictions
    for err, sample_batch, sample_metadata, sample_preds in worst_candidates:
        beetle_id, img = plot_preds(sample_batch, sample_metadata, sample_preds, 0)
        images[f"images/{prefix}/worst/beetle{beetle_id}"] = wandb.Image(img)

    metrics = {
        k: jnp.concatenate([dct[k].reshape(-1) for dct in metrics]) for k in metrics[0]
    }

    means = {f"{prefix}/{k}": jnp.nanmean(v).item() for k, v in metrics.items()}
    maxes = {f"{prefix}/max_{k}": jnp.nanmax(v).item() for k, v in metrics.items()}
    medians = {
        f"{prefix}/median_{k}": jnp.nanmedian(v).item() for k, v in metrics.items()
    }

    # Compute seen vs unseen species metrics
    seen_unseen_metrics = {}
    if training_species is not None and seen_metrics:
        all_seen = {
            k: jnp.concatenate([m[k].reshape(-1) for m in seen_metrics], axis=0)
            for k in seen_metrics[0]
        }
        all_unseen = {
            k: jnp.concatenate([m[k].reshape(-1) for m in unseen_metrics], axis=0)
            for k in unseen_metrics[0]
        }

        for key in ["line_err_cm", "point_err_cm", "loss"]:
            seen_unseen_metrics[f"{prefix}/seen_{key}"] = jnp.nanmean(
                all_seen[key]
            ).item()
            seen_unseen_metrics[f"{prefix}/unseen_{key}"] = jnp.nanmean(
                all_unseen[key]
            ).item()
            seen_unseen_metrics[f"{prefix}/seen_max_{key}"] = jnp.nanmax(
                all_seen[key]
            ).item()
            seen_unseen_metrics[f"{prefix}/unseen_max_{key}"] = jnp.nanmax(
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
def get_training_species(cfg: "Config") -> set[str]:
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
    return species


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


def to_device(batch: dict[str, object], device=None) -> tuple:
    numeric = {k: v for k, v in batch.items() if is_device_array(v)}
    aux = {k: v for k, v in batch.items() if not is_device_array(v)}
    # device_put works on pytrees; leaves become jax.Arrays on the target device
    numeric = jax.device_put(numeric, device)
    return numeric, aux


@beartype.beartype
def train(cfg: Config):
    key = jax.random.key(seed=cfg.seed)
    val_cfgs = [
        dataclasses.replace(cfg.hawaii, split="val"),
        dataclasses.replace(cfg.biorepo, split="val"),
    ]
    val_cfgs = [c for c in val_cfgs if c.go]
    val_min_px_per_cm = min(
        get_aug_for_dataset(cfg, dataset_cfg).min_px_per_cm for dataset_cfg in val_cfgs
    )

    # Get training species for seen/unseen validation metrics
    training_species = get_training_species(cfg)

    # Create separate validation dataloaders for each dataset
    # Each entry: (dataloader, fixed_indices, total_samples, use_species_split)
    val_datasets = {}

    hawaii_val_cfg = dataclasses.replace(cfg.hawaii, split="val")
    if hawaii_val_cfg.go:
        hawaii_source = hawaii_val_cfg.dataset(hawaii_val_cfg)
        if len(hawaii_source) > 0:
            hawaii_total = len(hawaii_source)
            fixed_rng = np.random.default_rng(seed=cfg.seed)
            hawaii_fixed_indices = fixed_rng.choice(
                hawaii_total, size=min(cfg.n_val_fixed, hawaii_total), replace=False
            )
            hawaii_val_dl = make_dataset(
                [hawaii_val_cfg],
                seed=cfg.seed,
                batch_size=cfg.batch_size,
                n_workers=cfg.n_workers,
                shuffle=False,
                finite=True,
                train_cfg=cfg,
                train=False,
            )
            val_datasets["val/hawaii"] = (
                hawaii_val_dl,
                hawaii_fixed_indices,
                hawaii_total,
                False,
            )
            logger.info(
                "Hawaii validation: %d samples, fixed indices: %s",
                hawaii_total,
                hawaii_fixed_indices,
            )

    biorepo_val_cfg = dataclasses.replace(cfg.biorepo, split="val")
    if biorepo_val_cfg.go:
        biorepo_source = biorepo_val_cfg.dataset(biorepo_val_cfg)
        if len(biorepo_source) > 0:
            biorepo_total = len(biorepo_source)
            fixed_rng = np.random.default_rng(seed=cfg.seed)
            biorepo_fixed_indices = fixed_rng.choice(
                biorepo_total, size=min(cfg.n_val_fixed, biorepo_total), replace=False
            )
            biorepo_val_dl = make_dataset(
                [biorepo_val_cfg],
                seed=cfg.seed,
                batch_size=cfg.batch_size,
                n_workers=cfg.n_workers,
                shuffle=False,
                finite=True,
                train_cfg=cfg,
                train=False,
            )
            val_datasets["val/biorepo"] = (
                biorepo_val_dl,
                biorepo_fixed_indices,
                biorepo_total,
                True,
            )
            logger.info(
                "BioRepo validation: %d samples, fixed indices: %s",
                biorepo_total,
                biorepo_fixed_indices,
            )

    train_cfgs = [
        dataclasses.replace(cfg.hawaii, split="train"),
        cfg.beetlepalooza,
    ]
    train_cfgs = [c for c in train_cfgs if c.go]
    train_min_px_per_cm = min(
        get_aug_for_dataset(cfg, dataset_cfg).min_px_per_cm
        for dataset_cfg in train_cfgs
    )
    train_dl = make_dataset(
        train_cfgs,
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        n_workers=cfg.n_workers,
        shuffle=True,
        finite=False,
        train_cfg=cfg,
        train=True,
    )

    model = btx.modeling.make(cfg.model, key)

    # freeze the Vit

    # Step 1 create the filter_specification
    # Create a pytree with the same shape as model setting each leaf to false
    filter_spec = jax.tree_util.tree_map(lambda _: False, model)
    # If a leaf is part of the head, make it trainable
    filter_spec = eqx.tree_at(
        lambda tree: tree.head,
        filter_spec,
        jax.tree_util.tree_map(eqx.is_array, model.head),
    )

    # Optimize only the differentiable part of the model.
    diff_model, _ = eqx.partition(model, filter_spec)
    optim = optax.adamw(learning_rate=cfg.learning_rate)
    state = optim.init(diff_model)

    run = wandb.init(
        project=cfg.wandb_project, config=dataclasses.asdict(cfg), tags=cfg.tags
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
            min_px_per_cm=train_min_px_per_cm,
        )

        if step % cfg.save_every == 0:
            beetle_id, img = plot_preds(
                batch, metadata, aux.preds, 0
            )  # since batch is shuffled, choose elem 0
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

        if step % cfg.val_every == 0:
            all_val_metrics = {}
            for prefix, (
                val_dl,
                fixed_indices,
                val_total_samples,
                use_species_split,
            ) in val_datasets.items():
                species_set = training_species if use_species_split else None
                metrics = validate(
                    cfg,
                    model,
                    val_dl,
                    filter_spec,
                    min_px_per_cm=val_min_px_per_cm,
                    fixed_indices=fixed_indices,
                    val_total_samples=val_total_samples,
                    prefix=prefix,
                    training_species=species_set,
                )
                all_val_metrics.update(metrics)
            run.log(all_val_metrics, step=step)
            logger.info("Validation: %d %s", step, all_val_metrics)

        if step >= cfg.n_steps:
            break
