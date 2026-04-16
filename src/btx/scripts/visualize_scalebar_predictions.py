"""Visualize scalebar predictions from px_per_cm.json on group images.

For each sampled image, crops to the predicted bbox (with padding), draws the
predicted scalebar line and bbox rectangle, and saves as a JPEG.

USAGE:
------
  uv run python -m btx.scripts.visualize_scalebar_predictions \
      --results-fpath docs/experiments/013-scalebar-training/results/fdpj3qnm_px_per_cm.json \
      --n 20
"""

import dataclasses
import json
import logging
import pathlib
import random

import beartype
import tyro
from PIL import Image, ImageDraw, ImageFont

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logger = logging.getLogger("visualize-scalebar-predictions")

_DEFAULT_RESULTS = pathlib.Path(
    "docs/experiments/013-scalebar-training/results/fdpj3qnm_px_per_cm.json"
)
_DEFAULT_OUT = pathlib.Path(
    "docs/experiments/013-scalebar-training/results/fdpj3qnm_predictions_viz"
)

# Padding around the bbox crop, in pixels (on original image scale).
_PADDING = 100


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    results_fpath: pathlib.Path = _DEFAULT_RESULTS
    """px_per_cm.json produced by scalebar_infer.py."""
    out_dpath: pathlib.Path = _DEFAULT_OUT
    """Directory to write JPEG visualizations."""
    n: int = 20
    """Number of samples to visualize. Ignored if all=True."""
    seed: int = 0
    """Random seed for sampling."""
    all: bool = False
    """Visualize all predictions instead of sampling n."""


@beartype.beartype
def visualize_one(img_fpath: str, record: dict, out_dpath: pathlib.Path) -> None:
    orig_path = pathlib.Path(img_fpath)
    if not orig_path.exists():
        logger.warning("Image not found: '%s'", img_fpath)
        return

    bbox = record["bbox"]
    bx, by, bw, bh = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    pts = record["scalebar_pts"]  # [[x0,y0],[x1,y1]] in original image coords
    px_per_cm = record["px_per_cm"]

    with Image.open(orig_path) as im:
        im = im.convert("RGB")
        iw, ih = im.size

        # Padded crop bounds (clamped to image).
        cx0 = max(0, bx - _PADDING)
        cy0 = max(0, by - _PADDING)
        cx1 = min(iw, bx + bw + _PADDING)
        cy1 = min(ih, by + bh + _PADDING)
        crop = im.crop((cx0, cy0, cx1, cy1))

    draw = ImageDraw.Draw(crop)

    # Shift coordinates into crop space.
    def to_crop(x: float, y: float) -> tuple[float, float]:
        return x - cx0, y - cy0

    # Draw bbox rectangle (yellow).
    rx0, ry0 = to_crop(bx, by)
    draw.rectangle([rx0, ry0, rx0 + bw, ry0 + bh], outline=(255, 220, 0), width=4)

    # Draw predicted scalebar line (cyan).
    p0 = to_crop(pts[0][0], pts[0][1])
    p1 = to_crop(pts[1][0], pts[1][1])
    draw.line([p0, p1], fill=(0, 220, 255), width=6)
    r = 8
    for px, py in [p0, p1]:
        draw.ellipse([px - r, py - r, px + r, py + r], fill=(0, 220, 255))

    # Draw label.
    label = f"{px_per_cm:.1f} px/cm"
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 36)
    except OSError:
        font = ImageFont.load_default()
    lx, ly = rx0, max(0, ry0 - 44)
    draw.rectangle([lx - 2, ly - 2, lx + 300, ly + 44], fill=(0, 0, 0))
    draw.text((lx + 4, ly + 4), label, fill=(255, 255, 255), font=font)

    out_fname = orig_path.stem + "_pred.jpg"
    crop.save(out_dpath / out_fname, quality=90)
    logger.info("Saved '%s'", out_dpath / out_fname)


@beartype.beartype
def main(cfg: Config) -> None:
    logging.basicConfig(level=logging.INFO, format=log_format)

    assert cfg.results_fpath.exists(), f"Results not found: '{cfg.results_fpath}'"
    with cfg.results_fpath.open() as fd:
        results: dict[str, dict] = json.load(fd)
    logger.info("Loaded %d results from '%s'", len(results), cfg.results_fpath)

    keys = sorted(results.keys())
    if cfg.all:
        sample_keys = keys
    else:
        rng = random.Random(cfg.seed)
        sample_keys = rng.sample(keys, min(cfg.n, len(keys)))
    logger.info("Visualizing %d samples", len(sample_keys))

    cfg.out_dpath.mkdir(parents=True, exist_ok=True)
    for key in sample_keys:
        visualize_one(key, results[key], cfg.out_dpath)

    logger.info("Done. Output in '%s'", cfg.out_dpath)


if __name__ == "__main__":
    tyro.cli(main)
