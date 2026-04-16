"""Dataset for scalebar localization on full group images.

Each sample is one group image; the target is the scalebar line ([[x0,y0],[x1,y1]])
in the original group image pixel coordinates.

Annotations are drawn from two sources:
  - biorepo-formatted/annotations.json
  - active_learning_round2/round2/annotations.json

The scalebar endpoints are returned in points_px[0] (the "width" line slot), with
loss_mask=[1.0, 0.0] so the model trains only on the first pair of heatmap channels
and ignores the second pair. scalebar_px is zeroed out and marked invalid since the
scalebar IS what we're detecting.
"""

import dataclasses
import json
import logging
import pathlib
import typing as tp

import beartype
import numpy as np
from PIL import Image

from . import utils

logger = logging.getLogger("scalebar-group")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config(utils.Config):
    go: bool = True
    biorepo_annotations: pathlib.Path = pathlib.Path(
        "/fs/scratch/PAS2136/cain429/biorepo-formatted/annotations.json"
    )
    round2_annotations: pathlib.Path = pathlib.Path(
        "/fs/scratch/PAS2136/cain429/active_learning_round2/round2/annotations.json"
    )
    split: tp.Literal["train", "val", "all"] = "train"
    val_fraction: float = 0.2
    """Fraction of group images held out for validation (deterministic)."""
    cache: bool = True
    """Pre-load all images into RAM at dataset init to eliminate per-step disk I/O."""
    cache_size: int | None = None
    """If set, images are pre-resized to this square size during caching (~25x memory saving vs full-res). Must match aug.size. Set automatically by the trainer from cfg.aug.size."""

    @property
    def key(self) -> str:
        return "scalebar_group"

    @property
    def dataset(self) -> type["Dataset"]:
        return Dataset


@beartype.beartype
def _load_group_scalebars(anns_fpath: pathlib.Path) -> list[dict]:
    """Return one dict per unique group image that has a scalebar annotation."""
    anns = json.loads(anns_fpath.read_text())
    seen: set[str] = set()
    rows = []
    for ann in anns:
        fname = ann.get("group_img", "")
        if not fname or fname in seen:
            continue
        scalebar = next(
            (
                m["polyline"]
                for m in ann.get("measurements", [])
                if m.get("measurement_type") == "scalebar"
            ),
            None,
        )
        if scalebar is None:
            continue
        seen.add(fname)
        rows.append({
            "group_img": fname,
            "abs_group_img_path": ann["abs_group_img_path"],
            "scalebar": scalebar,
        })
    return rows


@beartype.beartype
class Dataset(utils.Dataset):
    _cfg: Config
    _rows: list[dict]
    _img_cache: dict[str, np.ndarray]

    def __init__(self, cfg: Config):
        self._cfg = cfg

        # Load from both sources, dedup by abs path (round2 wins on overlap).
        rows_by_path: dict[str, dict] = {}
        for anns_fpath in [cfg.biorepo_annotations, cfg.round2_annotations]:
            assert anns_fpath.exists(), f"Annotations not found: '{anns_fpath}'"
            for row in _load_group_scalebars(anns_fpath):
                rows_by_path[row["abs_group_img_path"]] = row

        all_keys = sorted(rows_by_path.keys())
        n_val = max(1, round(len(all_keys) * cfg.val_fraction))
        val_keys = set(all_keys[-n_val:])

        if cfg.split == "train":
            keys = [k for k in all_keys if k not in val_keys]
        elif cfg.split == "val":
            keys = [k for k in all_keys if k in val_keys]
        else:
            keys = all_keys

        self._rows = [rows_by_path[k] for k in keys]
        assert self._rows, f"ScalebarGroup dataset is empty for split='{cfg.split}'."
        logger.info("ScalebarGroup %s: %d group images", cfg.split, len(self._rows))

        if cfg.cache:
            logger.info("Pre-loading %d images into RAM...", len(self._rows))
            self._img_cache = {}
            for row in self._rows:
                fpath = row["abs_group_img_path"]
                with Image.open(fpath) as im:
                    row["orig_size"] = im.size  # (w, h) in original pixels
                    if cfg.cache_size is not None:
                        im = im.resize(
                            (cfg.cache_size, cfg.cache_size),
                            Image.Resampling.BILINEAR,
                        )
                    self._img_cache[fpath] = np.asarray(
                        im.convert("RGB"), dtype=np.uint8
                    )
            logger.info(
                "Image cache ready (~%.1f GB).",
                sum(a.nbytes for a in self._img_cache.values()) / 1e9,
            )
        else:
            self._img_cache = {}

    @property
    def cfg(self) -> Config:
        return self._cfg

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: tp.SupportsIndex) -> utils.Sample:
        row = self._rows[int(idx)]
        img_fpath = pathlib.Path(row["abs_group_img_path"])
        assert img_fpath.is_file(), f"Group image not found: '{img_fpath}'"

        pts = np.array(row["scalebar"], dtype=np.float64)
        if self._cfg.cache_size is not None:
            orig_w, orig_h = row["orig_size"]
            pts = pts * np.array([
                self._cfg.cache_size / orig_w,
                self._cfg.cache_size / orig_h,
            ])
        scalebar_arr = np.stack([
            pts[0],
            pts[-1],
        ])  # endpoints only; middle points are annotation artifacts
        dummy = np.zeros((2, 2), dtype=np.float64)
        points_px = np.stack([scalebar_arr, dummy])  # (2, 2, 2)

        return utils.Sample(
            img_fpath=str(img_fpath),
            points_px=points_px,
            scalebar_px=np.zeros((2, 2), dtype=np.float64),
            scalebar_valid=np.bool_(False),
            loss_mask=np.array([1.0, 0.0]),
            beetle_id="",
            beetle_position=0,
            group_img_basename=pathlib.Path(row["group_img"]).stem,
            scientific_name="",
            split=self._cfg.split,
        )
