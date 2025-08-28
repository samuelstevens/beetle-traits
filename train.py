# train.py
import dataclasses
import logging
import pathlib
import typing as tp

import beartype
import equinox as eqx
import grain
import jax
import jax.numpy as jnp
import optax
import tyro
from jaxtyping import Array, Float, jaxtyped
from PIL import Image, ImageDraw

import btx.data
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
    learning_rate: float = 1e-4
    wandb_project: str = "beetle-traits"
    wandb_entity: str = "samuelstevens"
    pck_rs: list[int] = dataclasses.field(default_factory=lambda: [1, 2, 4, 8])
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
        grain.transforms.Batch(batch_size=2, drop_remainder=True),
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
def compute_loss(
    model: eqx.Module,
    imgs: Float[Array, "batch width height channels"],
    tgts: Float[Array, "batch 2 2 2"],
) -> Float[Array, ""]:
    preds = jax.vmap(model)(imgs)
    loss = jnp.mean((preds - tgts) ** 2)
    return loss


@jaxtyped(typechecker=beartype.beartype)
def compute_metrics(
    cfg: Config, preds: Float[Array, "batch 2 2 2"], batch: dict[str, object]
) -> dict[str, Float[Array, ""]]:
    breakpoint()
    mse = jnp.mean((preds - tgts) ** 2)
    diffs = jnp.abs(preds - tgts)
    # TODO: add a metric that tracks the difference in predicted line difference.
    mae = jnp.mean(diffs)
    pck = {f"pck@{r}": jnp.sum(diffs < r) for r in cfg.pck_rs}

    return {"mse": mse, "mae": mae, **pck}


@jaxtyped(typechecker=beartype.beartype)
@eqx.filter_jit(donate="all")
def step_model(
    model: eqx.Module,
    optim: optax.GradientTransformation,
    state: tp.Any,
    imgs: Float[Array, "batch width height channels"],
    tgts: Float[Array, "batch 2 2 2"],
) -> tuple[eqx.Module, tp.Any, Float[Array, ""]]:
    loss, grads = eqx.filter_value_and_grad(compute_loss)(model, imgs, tgts)
    (updates,), new_state = optim.update([grads], state, [model])

    model = eqx.apply_updates(model, updates)

    return model, new_state, loss


@jaxtyped(typechecker=beartype.beartype)
def plot_preds(
    batch: dict[str, object], preds: Float[Array, "batch 2 2 2"]
) -> tuple[str, Image.Image]:
    # Process first image in batch
    img_fpath = batch["img_fpath"][0]
    img = Image.open(img_fpath)

    # Get ground truth points
    gt_length_px, gt_width_px = batch["points_px"][0]

    scale_x, scale_y = batch["scale"][0]
    tgts_px = batch["tgt"][0].copy()
    tgts_px[:, :, 0] /= scale_x
    tgts_px[:, :, 1] /= scale_y

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
    beetle_id = batch["beetle_id"][0]

    return beetle_id, img


@beartype.beartype
def train(cfg: Config):
    key = jax.random.key(seed=cfg.seed)
    val_cfg = dataclasses.replace(cfg.data, split="val", seed=cfg.seed)
    train_cfg = dataclasses.replace(cfg.data, split="train", seed=cfg.seed)
    val_dl = make_dataloader(val_cfg)
    train_dl = make_dataloader(train_cfg)

    model = btx.modeling.make(cfg.model, key)

    optim = optax.sgd(learning_rate=cfg.learning_rate)
    state = optim.init(eqx.filter([model], eqx.is_inexact_array))

    run = wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        config=dataclasses.asdict(cfg),
        tags=cfg.tags,
    )

    # Training
    batch = next(iter(train_dl))
    for step, _ in enumerate(train_dl):
        imgs = jnp.array(batch["img"])
        tgts = jnp.array(batch["tgt"])

        model, new_state, loss = step_model(model, optim, state, imgs, tgts)

        if step % cfg.save_every == 0:
            preds = jax.vmap(model)(imgs)
            beetle_id, img = plot_preds(batch, preds)
            metrics = {"step": step, "train/loss": loss.item()}
            run.log({**metrics, f"images/train/beetle{beetle_id}": wandb.Image(img)})

        if step % cfg.log_every == 0:
            metrics = {"step": step, "train/loss": loss.item()}
            run.log(metrics)
            logger.info("Step: %d %s", step, metrics)

        # if step % cfg.val_every == 0:
        #     metrics = []
        #     for batch in val_dl:
        #         imgs = jnp.array(batch["img"])
        #         tgts = jnp.array(batch["tgt"])

        #         preds = jax.vmap(model)(imgs)
        #         metrics.append(compute_metrics(cfg, preds, tgts))

        #     metrics = {
        #         f"val/{k}": jnp.array([dct[k] for dct in metrics]).mean().item()
        #         for k in metrics[0]
        #     }
        #     beetle_id, img = save_preds(cfg, step, batch, preds)
        #     run.log({**metrics, f"images/val/beetle{beetle_id}": wandb.Image(img)})

        if step >= cfg.n_steps:
            break


@beartype.beartype
def main(cfg: Config):
    import submitit

    if cfg.slurm_acct:
        executor = submitit.SlurmExecutor(folder=cfg.log_to)
        executor.update_parameters(
            time=int(cfg.n_hours * 60),
            partition=cfg.slurm_partition,
            gpus_per_node=1,
            ntasks_per_node=1,
            cpus_per_task=cfg.data.n_workers + 4,
            stderr_to_stdout=True,
            account=cfg.slurm_acct,
        )

    else:
        executor = submitit.DebugExecutor(folder=cfg.log_to)

    job = executor.submit(train, cfg)
    logger.info("Running job %s", job.job_id)
    job.result()


if __name__ == "__main__":
    main(tyro.cli(Config))
