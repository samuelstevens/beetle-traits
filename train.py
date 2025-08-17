# train.py
import dataclasses
import logging
import pathlib

import beartype
import grain
import jax
import numpy as np
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
    save_every: int = 100
    """How often to save predictions."""
    log_to: pathlib.Path = pathlib.Path("./logs")


@dataclasses.dataclass(frozen=True)
class DecodeRGB(grain.transforms.Map):
    def map(self, sample: btx.data.hawaii.Sample):
        # Heavy I/O lives in a transform so workers can parallelize it
        with Image.open(sample["img_filepath"]) as im:
            sample["img"] = np.array(im.convert("RGB"))
        return sample


@dataclasses.dataclass(frozen=True)
class Resize(grain.transforms.Map):
    size: int = 256

    def map(self, sample: dict[str, object]) -> dict[str, object]:
        img = sample["img"]
        # simple resize with PIL for brevity
        img = np.array(Image.fromarray(img).resize((self.size, self.size)))
        sample["img"] = img.astype(np.float32) / 255.0
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
        grain.transforms.Batch(batch_size=2, drop_remainder=True),
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
def save_preds(
    cfg: Config, step: int, batch: dict[str, object], preds: Float[Array, "batch 2 2 2"]
):
    # Create output directory if it doesn't exist
    output_dir = cfg.log_to / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process first image in batch
    pil_img = Image.open(batch["img_filepath"][0])

    # Get ground truth points
    gt_length_px = batch["elytra_length_points"][0]
    gt_width_px = batch["elytra_width_points"][0]

    # Get predicted points
    pred_length_px = preds[0, 0]
    pred_width_px = preds[0, 1]

    # Draw on image
    draw = ImageDraw.Draw(pil_img)

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
    filepath = str(batch["img_filepath"][0])
    filename = pathlib.Path(filepath).name
    # Extract ID from filename like "IMG_0551_specimen_2_TREOBT_NEON.BET.D20.001690.png"
    individual_id = filename.split("_")[-1].split(".")[0]  # Gets "001690"

    # Save image
    output_path = output_dir / f"step{step}_id{individual_id}.png"
    pil_img.save(output_path)
    logger.info(f"Saved predictions to {output_path}")


@beartype.beartype
def main(cfg: Config):
    key = jax.random.key(seed=cfg.seed)
    train_cfg = dataclasses.replace(cfg.data, split="train", seed=cfg.seed)
    train_dl = make_dataloader(train_cfg)
    model = btx.modeling.make(cfg.model, key)

    # Training
    for b, batch in enumerate(train_dl):
        preds = jax.vmap(model)(batch["img"])

        if b % cfg.save_every == 0:
            save_preds(cfg, b, batch, preds)
            break


if __name__ == "__main__":
    main(tyro.cli(Config))
