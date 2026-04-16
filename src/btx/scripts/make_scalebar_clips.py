"""Clip scalebar bounding boxes from group images and write training data.

For each annotated group image that has a detected bbox (from detect_scalebar_bboxes.py),
this script:
  1. Crops the original image to the bbox.
  2. Transforms the scalebar line annotation into clip-space coordinates.
  3. Saves the crop as a PNG.
  4. Writes a JSON index usable by ScalebarClipDataset.

INPUTS:
- scalebar_bboxes.json (from detect_scalebar_bboxes.py)
- biorepo-formatted/annotations.json
- active_learning_round2/round2/annotations.json

OUTPUTS:
- clips_dpath/: PNG crops, one per annotated image with a bbox.
- clips_dpath/clips.json: [{clip_fpath, scalebar, orig_fpath, bbox}, ...]
  where scalebar = [[x0,y0],[x1,y1]] in clip-space pixel coordinates.

USAGE:
------
  uv run python -m btx.scripts.make_scalebar_clips
"""

import dataclasses
import json
import logging
import pathlib

import beartype
import numpy as np
import tyro
from PIL import Image

from btx.data.scalebar import _load_group_scalebars

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logger = logging.getLogger("make-scalebar-clips")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    bboxes_fpath: pathlib.Path = pathlib.Path(
        "/fs/scratch/PAS2136/cain429/scalebar_bboxes.json"
    )
    """Output of detect_scalebar_bboxes.py."""
    biorepo_annotations: pathlib.Path = pathlib.Path(
        "/fs/scratch/PAS2136/cain429/biorepo-formatted/annotations.json"
    )
    round2_annotations: pathlib.Path = pathlib.Path(
        "/fs/scratch/PAS2136/cain429/active_learning_round2/round2/annotations.json"
    )
    clips_dpath: pathlib.Path = pathlib.Path(
        "/fs/scratch/PAS2136/cain429/beetle-traits/scalebar_clips"
    )
    """Directory to write clip PNGs and clips.json."""


@beartype.beartype
def main(cfg: Config) -> None:
    logging.basicConfig(level=logging.INFO, format=log_format)

    assert cfg.bboxes_fpath.exists(), (
        f"Bboxes not found: '{cfg.bboxes_fpath}'. Run detect_scalebar_bboxes.py first."
    )
    with cfg.bboxes_fpath.open() as fd:
        bboxes: dict[str, dict] = json.load(fd)
    logger.info("Loaded %d bboxes from '%s'", len(bboxes), cfg.bboxes_fpath)

    # Load annotations, dedup by path (round2 wins on overlap).
    rows_by_path: dict[str, dict] = {}
    for anns_fpath in [cfg.biorepo_annotations, cfg.round2_annotations]:
        assert anns_fpath.exists(), f"Annotations not found: '{anns_fpath}'"
        for row in _load_group_scalebars(anns_fpath):
            rows_by_path[row["abs_group_img_path"]] = row
    logger.info("Loaded annotations for %d group images", len(rows_by_path))

    overlap = set(rows_by_path) & set(bboxes)
    logger.info(
        "%d images have both annotation and bbox (%d annotation-only, %d bbox-only)",
        len(overlap),
        len(rows_by_path) - len(overlap),
        len(bboxes) - len(overlap),
    )

    cfg.clips_dpath.mkdir(parents=True, exist_ok=True)

    clips = []
    n_oob = 0

    for orig_fpath_str in sorted(overlap):
        row = rows_by_path[orig_fpath_str]
        bbox = bboxes[orig_fpath_str]
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]

        orig_fpath = pathlib.Path(orig_fpath_str)
        assert orig_fpath.exists(), f"Original image missing: '{orig_fpath}'"

        with Image.open(orig_fpath) as im:
            orig_w, orig_h = im.size
            clip = im.crop((x, y, x + w, y + h))

        # Transform annotation endpoints from original to clip coordinates.
        pts = np.array(row["scalebar"], dtype=np.float64)
        scalebar_orig = np.stack([pts[0], pts[-1]])  # [[x0,y0],[x1,y1]]
        scalebar_clip = scalebar_orig - np.array([x, y], dtype=np.float64)

        # Warn if either endpoint is outside the clip.
        if (
            np.any(scalebar_clip < 0)
            or np.any(scalebar_clip[:, 0] > w)
            or np.any(scalebar_clip[:, 1] > h)
        ):
            logger.warning(
                "Scalebar endpoints outside bbox for '%s': pts=%s bbox=(%d,%d,%d,%d)",
                orig_fpath.name,
                scalebar_clip.tolist(),
                x,
                y,
                w,
                h,
            )
            n_oob += 1

        clip_fname = orig_fpath.stem + "_clip.png"
        clip_fpath = cfg.clips_dpath / clip_fname
        clip.save(clip_fpath)

        clips.append({
            "clip_fpath": str(clip_fpath),
            "scalebar": scalebar_clip.tolist(),
            "orig_fpath": orig_fpath_str,
            "orig_size": [orig_w, orig_h],
            "bbox": {"x": x, "y": y, "w": w, "h": h},
        })

    clips_json_fpath = cfg.clips_dpath / "clips.json"
    with clips_json_fpath.open("w") as fd:
        json.dump(clips, fd, indent=2)

    logger.info("=" * 60)
    logger.info("DONE")
    logger.info("  Clips saved:                %d", len(clips))
    logger.info("  Endpoint outside bbox:      %d", n_oob)
    logger.info("  Clips dir: '%s'", cfg.clips_dpath)
    logger.info("  Clips JSON: '%s'", clips_json_fpath)


if __name__ == "__main__":
    tyro.cli(main)
