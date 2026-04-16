"""Core inference logic for scalebar localization.

Separated from the launcher because submitit and jaxtyping cannot coexist
in the same module.
"""

import dataclasses
import json
import logging
import pathlib
import typing as tp

import beartype
import equinox as eqx
import jax
import numpy as np
from jaxtyping import Array, Float, jaxtyped
from PIL import Image

import btx.helpers
import btx.modeling
import btx.objectives.heatmap

logger = logging.getLogger("scalebar-infer")

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    ckpt_fpath: pathlib.Path
    """Checkpoint produced by train_scalebar.py."""
    bboxes_fpath: pathlib.Path = pathlib.Path(
        "/fs/scratch/PAS2136/cain429/scalebar_bboxes.json"
    )
    """Output of detect_scalebar_bboxes.py."""
    out_fpath: pathlib.Path = pathlib.Path(
        "docs/experiments/013-scalebar-training/results/px_per_cm.json"
    )
    """Output JSON: group image path -> {px_per_cm, scalebar_pts, bbox}."""
    batch_size: int = 64
    log_to: pathlib.Path = pathlib.Path("./logs")
    slurm_acct: str = ""
    slurm_partition: str = ""
    n_hours: float = 1.0


@eqx.filter_jit()
@jaxtyped(typechecker=beartype.beartype)
def _predict(
    model: eqx.Module,
    imgs_bhwc: Float[Array, "batch height width 3"],
    *,
    obj_cfg: btx.objectives.heatmap.Config,
) -> Float[Array, "batch 2 2 2"]:
    """Batched forward pass + heatmap decode to image-space coordinates."""
    forward = tp.cast(tp.Callable[[Array], Array], model)
    preds_raw = jax.vmap(forward)(imgs_bhwc)
    decode = lambda chw: btx.objectives.heatmap.heatmaps_to_coords(chw, cfg=obj_cfg)
    return jax.vmap(decode)(preds_raw)


@beartype.beartype
def _load_clip(
    img_fpath: str, bbox: dict, *, size: int
) -> tuple[np.ndarray, int, int]:
    """Crop to bbox, resize to size x size, normalize. Returns (arr_hwc, clip_w, clip_h)."""
    with Image.open(img_fpath) as im:
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        clip = im.crop((x, y, x + w, y + h)).convert("RGB")
    clip_w, clip_h = clip.size
    resized = np.asarray(
        clip.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32
    )
    arr = (resized / 255.0 - _MEAN) / _STD
    return arr, clip_w, clip_h


@beartype.beartype
def run(cfg: Config) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    )
    assert cfg.ckpt_fpath.exists(), f"Checkpoint not found: '{cfg.ckpt_fpath}'"
    assert cfg.bboxes_fpath.exists(), (
        f"Bboxes not found: '{cfg.bboxes_fpath}'. Run detect_scalebar_bboxes.py first."
    )

    model, _model_cfg, obj_cfg = btx.modeling.load_ckpt(cfg.ckpt_fpath)
    assert isinstance(obj_cfg, btx.objectives.heatmap.Config), (
        f"Expected heatmap objective, got {type(obj_cfg)}"
    )
    size = obj_cfg.image_size
    logger.info("Loaded checkpoint '%s', image_size=%d", cfg.ckpt_fpath, size)

    with cfg.bboxes_fpath.open() as fd:
        bboxes: dict[str, dict] = json.load(fd)
    img_fpaths = sorted(bboxes.keys())
    logger.info("Running inference on %d images", len(img_fpaths))

    results: dict[str, dict] = {}
    batches = btx.helpers.batched_idx(len(img_fpaths), cfg.batch_size)
    for start, end in btx.helpers.progress(batches, every=10, desc="scalebar-infer"):
        batch_arrs: list[np.ndarray] = []
        batch_meta: list[tuple[str, dict, int, int]] = []

        for fpath in img_fpaths[start:end]:
            try:
                arr, clip_w, clip_h = _load_clip(fpath, bboxes[fpath], size=size)
            except Exception as e:
                logger.warning("Failed to load '%s': %s", fpath, e)
                continue
            batch_arrs.append(arr)
            batch_meta.append((fpath, bboxes[fpath], clip_w, clip_h))

        if not batch_arrs:
            continue

        imgs_bhwc = jax.device_put(np.stack(batch_arrs))
        coords_b222 = np.asarray(_predict(model, imgs_bhwc, obj_cfg=obj_cfg))

        for i, (fpath, bbox, clip_w, clip_h) in enumerate(batch_meta):
            # coords_b222[i, 0] = scalebar line (channel 0 = scalebar endpoints)
            # in aug space (size x size); undo resize, then add bbox offset.
            pts_aug = coords_b222[i, 0]  # (2, 2): [[x0,y0],[x1,y1]]
            pts_clip = pts_aug * np.array([clip_w / size, clip_h / size])
            pts_orig = pts_clip + np.array([bbox["x"], bbox["y"]])
            px_per_cm = float(np.linalg.norm(pts_orig[1] - pts_orig[0]))
            results[fpath] = {
                "px_per_cm": px_per_cm,
                "scalebar_pts": pts_orig.tolist(),
                "bbox": bbox,
            }

    cfg.out_fpath.parent.mkdir(parents=True, exist_ok=True)
    with cfg.out_fpath.open("w") as fd:
        json.dump(results, fd, indent=2)
    logger.info("Wrote %d results to '%s'", len(results), cfg.out_fpath)
