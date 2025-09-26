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

    wandb_project: str = "beetle-traits"
    wandb_entity: str = "samuelstevens"
    slurm_acct: str = ""
    slurm_partition: str = ""
    n_hours: float = 2.0


@beartype.beartype
def make_dataloader(cfg: btx.data.HawaiiConfig):
    source = btx.data.HawaiiDataset(cfg)
    shuffle = cfg.split == "train"
    n_epochs = None if cfg.split == "train" else 1
    sampler = grain.samplers.IndexSampler(
        num_records=len(source), shuffle=shuffle, num_epochs=n_epochs, seed=cfg.seed
    )
    ops = [
        btx.data.utils.DecodeRGB(),
        btx.data.utils.Resize(),
        grain.transforms.Batch(batch_size=cfg.batch_size, drop_remainder=False),
    ]
    loader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=ops,
        worker_count=cfg.n_workers,
        worker_buffer_size=2,  # per-worker prefetch of output batches
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


@jaxtyped(typechecker=beartype.beartype)
@eqx.filter_jit()
def loss_and_aux(
    model: eqx.Module, batch: dict[str, Array]
) -> tuple[Float[Array, ""], Aux]:
    preds = jax.vmap(model)(batch["img"])
    mse = jnp.mean((preds - batch["tgt"]) ** 2)

    # Metrics
    # Pixels
    scale_x, scale_y = jnp.unstack(batch["scale"], axis=-1)
    scale_x = jnp.expand_dims(scale_x, (1, 2))
    scale_y = jnp.expand_dims(scale_y, (1, 2))
    preds_px = preds.at[:, :, :, 0].divide(scale_x).at[:, :, :, 1].divide(scale_y)
    tgts_px = batch["tgt"].at[:, :, :, 0].divide(scale_x).at[:, :, :, 1].divide(scale_y)

    # Centimeters
    scalebar_start, scalebar_end = jnp.unstack(batch["scalebar_px"], axis=1)
    px_per_cm = jnp.linalg.norm(scalebar_start - scalebar_end, axis=1)

    # Point MAE
    point_err_px = einops.rearrange(
        jnp.linalg.norm(preds_px - tgts_px, axis=-1),
        "batch lines points -> batch (lines points)",
    )
    point_err_cm = point_err_px / px_per_cm[:, None]

    # Line length MAE
    tgts_start_px, tgts_end_px = jnp.unstack(tgts_px, axis=2)
    tgts_line_px = jnp.linalg.norm(tgts_start_px - tgts_end_px, axis=-1)

    preds_start_px, preds_end_px = jnp.unstack(preds_px, axis=2)
    preds_line_px = jnp.linalg.norm(preds_start_px - preds_end_px, axis=-1)
    line_err_px = jnp.abs(preds_line_px - tgts_line_px)
    line_err_cm = line_err_px / px_per_cm[:, None]

    return mse, Aux(mse, preds, point_err_px, point_err_cm, line_err_px, line_err_cm)


@jaxtyped(typechecker=beartype.beartype)
@eqx.filter_jit()
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
