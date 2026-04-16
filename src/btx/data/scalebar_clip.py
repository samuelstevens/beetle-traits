"""Dataset for scalebar localization on bbox-clipped group images.

Each sample is a crop of a group image tightly around the scalebar region,
detected by detect_scalebar_bboxes.py and saved by make_scalebar_clips.py.
The scalebar line annotation is stored pre-transformed into clip-space coordinates.

Compared to ScalebarGroupDataset (full group images), this dataset gives the
model a much easier task: the scalebar fills most of the frame.
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

logger = logging.getLogger("scalebar-clip")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config(utils.Config):
    go: bool = True
    clips_json: pathlib.Path = pathlib.Path(
        "/fs/scratch/PAS2136/cain429/beetle-traits/scalebar_clips/clips.json"
    )
    """Output of make_scalebar_clips.py."""
    split: tp.Literal["train", "val", "all"] = "train"
    val_fraction: float = 0.2
    cache: bool = True
    """Pre-load all clip images into RAM at dataset init."""
    cache_size: int | None = None
    """If set, clips are pre-resized to this square during caching. Set automatically by the trainer from cfg.aug.size."""

    @property
    def key(self) -> str:
        return "scalebar_clip"

    @property
    def dataset(self) -> type["Dataset"]:
        return Dataset


@beartype.beartype
class Dataset(utils.Dataset):
    _cfg: Config
    _rows: list[dict]
    _img_cache: dict[str, np.ndarray]

    def __init__(self, cfg: Config):
        self._cfg = cfg

        assert cfg.clips_json.exists(), (
            f"clips.json not found: '{cfg.clips_json}'. Run make_scalebar_clips.py first."
        )
        with cfg.clips_json.open() as fd:
            all_rows: list[dict] = json.load(fd)
        assert all_rows, "clips.json is empty."

        all_keys = sorted(r["clip_fpath"] for r in all_rows)
        rows_by_key = {r["clip_fpath"]: r for r in all_rows}
        n_val = max(1, round(len(all_keys) * cfg.val_fraction))
        val_keys = set(all_keys[-n_val:])

        if cfg.split == "train":
            keys = [k for k in all_keys if k not in val_keys]
        elif cfg.split == "val":
            keys = [k for k in all_keys if k in val_keys]
        else:
            keys = all_keys

        self._rows = [rows_by_key[k] for k in keys]
        assert self._rows, f"ScalebarClip dataset is empty for split='{cfg.split}'."
        logger.info("ScalebarClip %s: %d clips", cfg.split, len(self._rows))

        if cfg.cache:
            logger.info("Pre-loading %d clip images into RAM...", len(self._rows))
            self._img_cache = {}
            for row in self._rows:
                fpath = row["clip_fpath"]
                with Image.open(fpath) as im:
                    row["clip_size"] = im.size  # (w, h)
                    if cfg.cache_size is not None:
                        im = im.resize(
                            (cfg.cache_size, cfg.cache_size),
                            Image.Resampling.BILINEAR,
                        )
                    self._img_cache[fpath] = np.asarray(
                        im.convert("RGB"), dtype=np.uint8
                    )
            logger.info(
                "Clip cache ready (~%.1f GB).",
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
        clip_fpath = pathlib.Path(row["clip_fpath"])
        assert clip_fpath.is_file(), f"Clip image not found: '{clip_fpath}'"

        pts = np.array(row["scalebar"], dtype=np.float64)  # [[x0,y0],[x1,y1]]
        if self._cfg.cache_size is not None:
            clip_w, clip_h = row["clip_size"]
            pts = pts * np.array([
                self._cfg.cache_size / clip_w,
                self._cfg.cache_size / clip_h,
            ])

        scalebar_arr = pts  # already endpoints only
        dummy = np.zeros((2, 2), dtype=np.float64)
        points_px = np.stack([scalebar_arr, dummy])  # (2, 2, 2)

        orig_fpath = pathlib.Path(row["orig_fpath"])
        return utils.Sample(
            img_fpath=str(clip_fpath),
            points_px=points_px,
            scalebar_px=np.zeros((2, 2), dtype=np.float64),
            scalebar_valid=np.bool_(False),
            loss_mask=np.array([1.0, 0.0]),
            beetle_id="",
            beetle_position=0,
            group_img_basename=orig_fpath.stem,
            scientific_name="",
            split=self._cfg.split,
        )
