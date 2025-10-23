# train.py
import dataclasses
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
from jaxtyping import Array, Float, jaxtyped
from PIL import Image, ImageDraw
import os
import btx.data
import btx.modeling
import wandb

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("train.py")

_NUMERIC_KINDS = set("biufc")  # bool,int,uint,float,complex


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    seed: int = 17
    """Random seed."""
    model: btx.modeling.Config = btx.modeling.frozen.Frozen()
    """Neural network config."""
    data: btx.data.HawaiiConfig = btx.data.HawaiiConfig()
    """Data config."""
    tags: list[str] = dataclasses.field(default_factory=list)
    """List of wandb tags to include."""
    save_every: int = 200
    """How often to save predictions."""
    log_every: int = 200
    """How often to log to stderr."""
    val_every: int = 1_000
    """How often to run the validation loop."""
    n_steps: int = 100_000
    """Total number of training steps."""
    log_to: pathlib.Path = pathlib.Path("./logs")
    learning_rate: float = 3e-4
    # flexible wandb logging
    wandb_project: str = "beetle-traits"
    wandb_entity: str = os.environ.get("WANDB_ENTITY")
    slurm_acct: str = ""
    slurm_partition: str = ""
    n_hours: float = 2.0


# @beartype.beartype
# def make_dataloader(cfg: btx.data.HawaiiConfig):
#     source = btx.data.HawaiiDataset(cfg)
#     shuffle = cfg.split == "train"
#     n_epochs = None if cfg.split == "train" else 1
#     sampler = grain.samplers.IndexSampler(
#         num_records=len(source), shuffle=shuffle, num_epochs=n_epochs, seed=cfg.seed
#     )
#     ops = [
#         btx.data.utils.DecodeRGB(),
#         btx.data.utils.Resize(size=cfg.size),
#         btx.data.utils.GaussianHeatmap(
#             size=cfg.size,
#             sigma=getattr(cfg, "sigma", 3.0),  # fallback if not present on data cfg
#         ),
#         grain.transforms.Batch(batch_size=cfg.batch_size, drop_remainder=False),
#     ]
#     loader = grain.DataLoader(
#         data_source=source,
#         sampler=sampler,
#         operations=ops,
#         worker_count=cfg.n_workers,
#         worker_buffer_size=2,  # per-worker prefetch of output batches
#         read_options=grain.ReadOptions(num_threads=2, prefetch_buffer_size=64),
#     )

#     return loader

@beartype.beartype
def make_dataloader(cfg: btx.data.HawaiiConfig | object):
    """
    accepts either:
      - data config (btx.data.HawaiiConfig)
      - or top-level train.Config (with a .data field)

    normalize to `data_cfg` and read everything from there.
    """
    data_cfg = cfg.data if hasattr(cfg, "data") else cfg
    split       = getattr(data_cfg, "split", "train")
    seed        = getattr(data_cfg, "seed", 17)
    batch_size  = getattr(data_cfg, "batch_size", 8)
    n_workers   = getattr(data_cfg, "n_workers", 4)
    img_size    = getattr(data_cfg, "size", 256)
    sigma       = getattr(data_cfg, "sigma", 3.0)

    source = btx.data.HawaiiDataset(data_cfg)
    shuffle = split == "train"
    n_epochs = None if split == "train" else 1

    sampler = grain.samplers.IndexSampler(
        num_records=len(source), shuffle=shuffle, num_epochs=n_epochs, seed=seed
    )

    ops = [
        btx.data.utils.DecodeRGB(),
        btx.data.utils.Resize(size=img_size),
        btx.data.utils.GaussianHeatmap(size=img_size, sigma=sigma),
        grain.transforms.Batch(batch_size=batch_size, drop_remainder=False),
    ]

    loader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=ops,
        worker_count=n_workers,
        worker_buffer_size=2, 
        read_options=grain.ReadOptions(num_threads=2, prefetch_buffer_size=64),
    )

    return loader


@jaxtyped(typechecker=beartype.beartype)
class Aux(eqx.Module):
    loss: Float[Array, ""]
    preds: Float[Array, "batch 2 2 2"]
    point_err_px: Float[Array, " batch points"]
    point_err_cm: Float[Array, " batch points"]
    line_err_px: Float[Array, " batch lines"]
    line_err_cm: Float[Array, " batch lines"]

    def metrics(self):
        return {
            "loss": self.loss,
            "point_err_px": self.point_err_px,
            "point_err_cm": self.point_err_cm,
            "line_err_px": self.line_err_px,
            "line_err_cm": self.line_err_cm,
        }


@eqx.filter_jit()
@jaxtyped(typechecker=beartype.beartype)
def loss_and_aux(
    model: eqx.Module, batch: dict[str, Array]
) -> tuple[Float[Array, ""], Aux]:
    """
    handles both coordinate and heatmap output modes.
    """
    preds = jax.vmap(model)(batch["img"])

    # Coordinate mode
    if preds.ndim == 4 and preds.shape[1:] == (2, 2, 2):
        # Coordinate loss
        mse = jnp.mean((preds - batch["tgt"]) ** 2)
        preds_coords = preds  # (B, 2, 2, 2)

    # Heatmap mode
    elif preds.ndim == 4 and preds.shape[2] == batch["tgt_pixel_probs"].shape[2]:
        assert preds.shape == batch["tgt_pixel_probs"].shape, (
            f"Pred heatmaps {preds.shape} != tgt heatmaps {batch['tgt_pixel_probs'].shape}"
        )
        mse = jnp.mean((preds - batch["tgt_pixel_probs"]) ** 2)
        B, K, H, W = preds.shape
        idx_flat = jnp.argmax(preds.reshape(B, K, H * W), axis=-1)  
        ys = (idx_flat // W).astype(jnp.float32)
        xs = (idx_flat %  W).astype(jnp.float32)
        preds_k2 = jnp.stack([xs, ys], axis=-1)  
        preds_coords = preds_k2.reshape(B, 2, 2, 2)  
    else:
        raise RuntimeError(
            f"Unrecognized model output shape {preds.shape}. "
            "Expected (B, 2, 2, 2) for coordinates or (B, K, H, W) for heatmaps."
        )

    scale_x, scale_y = jnp.unstack(batch["scale"], axis=-1)  # (B,)
    scale_x = jnp.expand_dims(scale_x, (1, 2))               # (B,1,1)
    scale_y = jnp.expand_dims(scale_y, (1, 2))               # (B,1,1)

    # Pred & tgt in original px
    preds_px = preds_coords.at[:, :, :, 0].divide(scale_x).at[:, :, :, 1].divide(scale_y)
    tgts_px  = batch["tgt"].at[:, :, :, 0].divide(scale_x).at[:, :, :, 1].divide(scale_y)

    # Centimeters
    scalebar_start, scalebar_end = jnp.unstack(batch["scalebar_px"], axis=1)
    px_per_cm = jnp.linalg.norm(scalebar_start - scalebar_end, axis=1)  # (B,)

    # Point MAE
    point_err_px = einops.rearrange(
        jnp.linalg.norm(preds_px - tgts_px, axis=-1),  # (B, 2, 2)
        "batch lines points -> batch (lines points)",  # (B, 4)
    )
    point_err_cm = point_err_px / px_per_cm[:, None]

    # Line length MAE (per line)
    tgts_start_px, tgts_end_px = jnp.unstack(tgts_px, axis=2)   # (B,2,2) each
    tgts_line_px = jnp.linalg.norm(tgts_start_px - tgts_end_px, axis=-1)  # (B,2)

    preds_start_px, preds_end_px = jnp.unstack(preds_px, axis=2)
    preds_line_px = jnp.linalg.norm(preds_start_px - preds_end_px, axis=-1)  # (B,2)

    line_err_px = jnp.abs(preds_line_px - tgts_line_px)          # (B,2)
    line_err_cm = line_err_px / px_per_cm[:, None]

    return mse, Aux(mse, preds_coords, point_err_px, point_err_cm, line_err_px, line_err_cm)


@eqx.filter_jit()
@jaxtyped(typechecker=beartype.beartype)
def step_model(
    model: eqx.Module,
    optim: optax.GradientTransformation,
    state: tp.Any,
    batch: dict[str, Array],
) -> tuple[eqx.Module, tp.Any, Aux]:
    loss_fn = eqx.filter_value_and_grad(loss_and_aux, has_aux=True)
    (loss, aux), grads = loss_fn(model, batch)

    updates, new_state = optim.update(grads, state, model)
    model = eqx.apply_updates(model, updates)

    return model, new_state, aux


@jaxtyped(typechecker=beartype.beartype)
def plot_preds(
    batch: dict[str, Array],
    metadata: dict[str, object],
    preds: Float[Array, "batch 2 2 2"],
) -> tuple[str, Image.Image]:
    # Process first image in batch
    img_fpath = metadata["img_fpath"][0]
    img = Image.open(img_fpath)

    # Get ground truth points
    gt_length_px, gt_width_px = batch["points_px"][0]

    scale_x, scale_y = batch["scale"][0]

    # Get predicted points
    pred_px = preds[0].at[:, :, 0].divide(scale_x).at[:, :, 1].divide(scale_y)
    pred_length_px, pred_width_px = pred_px

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
    beetle_id = metadata["beetle_id"][0]

    return beetle_id, img


@beartype.beartype
def validate(cfg: Config, model: eqx.Module, val_dl):
    metrics = []
    for batch in val_dl:
        batch, metadata = to_device(batch)
        loss, aux = loss_and_aux(model, batch)
        metrics.append(aux.metrics())

    metrics = {
        k: jnp.concatenate([dct[k].reshape(-1) for dct in metrics]) for k in metrics[0]
    }

    means = {f"val/{k}": v.mean().item() for k, v in metrics.items()}
    maxes = {f"val/max_{k}": v.max().item() for k, v in metrics.items()}
    medians = {f"val/median_{k}": jnp.median(v).item() for k, v in metrics.items()}

    beetle_id, img = plot_preds(batch, metadata, aux.preds)
    return {
        **means,
        **maxes,
        **medians,
        f"images/val/beetle{beetle_id}": wandb.Image(img),
    }


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
    val_cfg = dataclasses.replace(cfg.data, split="val", seed=cfg.seed)
    train_cfg = dataclasses.replace(cfg.data, split="train", seed=cfg.seed)
    val_dl = make_dataloader(val_cfg)
    train_dl = make_dataloader(train_cfg)

    model = btx.modeling.make(cfg.model, key)

    optim = optax.adamw(learning_rate=cfg.learning_rate)
    state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    run = wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        config=dataclasses.asdict(cfg),
        tags=cfg.tags,
    )

    # Training
    for step, batch in enumerate(train_dl):
        batch, metadata = to_device(batch)
        model, state, aux = step_model(model, optim, state, batch)

        if step % cfg.save_every == 0:
            beetle_id, img = plot_preds(batch, metadata, aux.preds)
            metrics = {
                f"train/{key}": value.mean().item()
                for key, value in aux.metrics().items()
            }
            metrics["step"] = step
            metrics[f"images/train/beetle{beetle_id}"] = wandb.Image(img)
            run.log(metrics, step=step)

        if step % cfg.log_every == 0:
            metrics = {"step": step, "train/loss": aux.loss.item()}
            run.log(metrics, step=step)
            logger.info("Step: %d %s", step, metrics)

        if step % cfg.val_every == 0:
            metrics = validate(cfg, model, val_dl)
            run.log(metrics, step=step)
            logger.info("Validation: %d %s", step, metrics)

        if step >= cfg.n_steps:
            break