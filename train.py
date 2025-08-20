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
import numpy as np
import optax
import tyro
from jaxtyping import Array, Float, jaxtyped
from PIL import Image, ImageDraw

import btx.data
import btx.modeling

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("train.py")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    seed: int = 17
    """Random seed."""
    model: btx.modeling.Config = btx.modeling.Config()
    """Neural network config."""
    data: btx.data.HawaiiConfig = btx.data.HawaiiConfig()
    """Data config."""
    tags: list[str] = dataclasses.field(default_factory=list)
    """List of wandb tags to include."""
    save_every: int = 1000
    """How often to save predictions."""
    log_every: int = 1000
    log_to: pathlib.Path = pathlib.Path("./logs")
    learning_rate: float = 1e-4


@dataclasses.dataclass(frozen=True)
class DecodeRGB(grain.transforms.Map):
    def map(self, sample: btx.data.hawaii.Sample):
        # Heavy I/O lives in a transform so workers can parallelize it
        with Image.open(sample["img_fpath"]) as im:
            sample["img"] = np.array(im.convert("RGB"))
        return sample


@dataclasses.dataclass(frozen=True)
class Resize(grain.transforms.Map):
    size: int = 256

    def map(self, sample: dict[str, object]) -> dict[str, object]:
        img = sample["img"]
        orig_h, orig_w = img.shape[:2]

        # simple resize with PIL for brevity
        img = np.array(Image.fromarray(img).resize((self.size, self.size)))
        sample["img"] = img.astype(np.float32) / 255.0

        # Rescale the measurements according to the new size
        # TODO: is it better for data in grain transforms to be numpy or jax arrays?
        # TODO: if we update the fields, then the save_preds method is messed up.
        scale_x = self.size / orig_w
        scale_y = self.size / orig_h

        # elytra_length_px and elytra_width_px are 2x2 arrays: [[x1, y1], [x2, y2]]
        elytra_length = np.array(sample["elytra_length_px"])
        elytra_length[:, 0] *= scale_x  # scale x coordinates
        elytra_length[:, 1] *= scale_y  # scale y coordinates
        sample["elytra_length_px"] = elytra_length

        elytra_width = np.array(sample["elytra_width_px"])
        elytra_width[:, 0] *= scale_x  # scale x coordinates
        elytra_width[:, 1] *= scale_y  # scale y coordinates
        sample["elytra_width_px"] = elytra_width

        return sample


@beartype.beartype
def make_dataloader(cfg: btx.data.HawaiiConfig):
    source = btx.data.HawaiiDataset(cfg)
    shuffle = cfg.split == "train"
    sampler = grain.samplers.IndexSampler(
        num_records=len(source), shuffle=shuffle, num_epochs=None, seed=cfg.seed
    )
    ops = [
        DecodeRGB(),
        Resize(size=256),
        grain.transforms.Batch(batch_size=32, drop_remainder=True),
    ]
    loader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=ops,
        worker_count=0,  # >0 -> do work in child processes
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
    # jax.debug.print("preds={preds}, tgts={tgts}", preds=preds[0], tgts=tgts[0])
    loss = jnp.mean((preds - tgts) ** 2)

    return loss


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
def save_preds(
    cfg: Config, step: int, batch: dict[str, object], preds: Float[Array, "batch 2 2 2"]
):
    # Create output directory if it doesn't exist
    output_dir = cfg.log_to / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process first image in batch
    img_fpath = batch["img_fpath"][0]
    img = Image.open(img_fpath)

    # Get ground truth points
    gt_length_px = batch["elytra_length_px"][0]
    gt_width_px = batch["elytra_width_px"][0]

    # Get predicted points
    pred_length_px = preds[0, 0]
    pred_width_px = preds[0, 1]

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

    # Save image
    output_path = output_dir / f"step{step}_{beetle_id}.png"
    img.save(output_path)
    logger.info(f"Saved predictions to {output_path}")


@beartype.beartype
def main(cfg: Config):
    key = jax.random.key(seed=cfg.seed)
    train_cfg = dataclasses.replace(cfg.data, split="train", seed=cfg.seed)
    train_dl = make_dataloader(train_cfg)
    model = btx.modeling.make(cfg.model, key)

    optim = optax.sgd(learning_rate=cfg.learning_rate)
    state = optim.init(eqx.filter([model], eqx.is_inexact_array))

    # Training
    batch = next(iter(train_dl))
    for b in range(100_000):
        imgs = jnp.array(batch["img"])
        tgts = jnp.stack([batch["elytra_length_px"], batch["elytra_width_px"]], axis=1)

        model, new_state, loss = step_model(model, optim, state, imgs, tgts)

        if b % cfg.save_every == 0:
            preds = jax.vmap(model)(imgs)
            save_preds(cfg, b, batch, preds)

        if b % cfg.log_every == 0:
            print(b, loss.item())


if __name__ == "__main__":
    main(tyro.cli(Config))
