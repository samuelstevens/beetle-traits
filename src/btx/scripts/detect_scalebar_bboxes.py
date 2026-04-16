"""Detect red bounding boxes around scalebars in Plotted group images.

The Plotted folder contains group images with red rectangles drawn around the
scalebar. This script finds those red bounding boxes using connected-component
analysis on a red-pixel mask, then records the bounding box alongside the path
to the original (un-annotated) image in plotted_trays/.

INPUTS:
- /fs/ess/PAS2136/CarabidImaging/Output/Scalebar/Plotted/: Group images with
  red bounding boxes drawn around scalebars.
- /fs/ess/PAS2136/CarabidImaging/Output/plotted_trays/: Original group images
  (same filenames, no annotations drawn).

OUTPUTS:
- JSON file keyed by abs original image path, each value is:
    {x, y, w, h, plotted_fpath}
  where (x, y, w, h) is the detected red bbox in original image coordinates.

USAGE:
------
  uv run python -m btx.scripts.detect_scalebar_bboxes

  # With Slurm:
  uv run python -m btx.scripts.detect_scalebar_bboxes --slurm-acct PAS2136 --slurm-partition standard
"""

import dataclasses
import json
import logging
import pathlib

import beartype
import cv2
import numpy as np
import tyro

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logger = logging.getLogger("detect-scalebar-bboxes")

_CARB_DIR = pathlib.Path("/fs/ess/PAS2136/CarabidImaging")
_PLOTTED_DPATH = _CARB_DIR / "Output" / "Scalebar" / "Plotted"
_ORIGINALS_DPATH = _CARB_DIR / "Output" / "plotted_trays"

# Red-pixel threshold: R > _RED_MIN, G < _RED_MAX_GB, B < _RED_MAX_GB
_RED_MIN = 150
_RED_MAX_GB = 80

# Minimum area (pixels) to be considered a real bbox vs. noise.
_MIN_AREA = 200


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    plotted_dpath: pathlib.Path = _PLOTTED_DPATH
    """Folder containing Plotted images with red bounding boxes."""
    originals_dpath: pathlib.Path = _ORIGINALS_DPATH
    """Folder containing the original group images (same filenames, no annotations)."""
    out: pathlib.Path = pathlib.Path("/fs/scratch/PAS2136/cain429/scalebar_bboxes.json")
    """Output JSON path."""
    log_to: pathlib.Path = pathlib.Path("./logs")
    """Directory for submitit logs."""
    slurm_acct: str = ""
    """Slurm account. If set, submits to Slurm; otherwise runs locally."""
    slurm_partition: str = ""
    """Slurm partition."""
    n_hours: float = 4.0
    """Wall-clock time limit in hours."""


@beartype.beartype
def detect_red_bbox(img_arr: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return (x, y, w, h) of the largest red connected component, or None."""
    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    red_mask = ((r > _RED_MIN) & (g < _RED_MAX_GB) & (b < _RED_MAX_GB)).astype(
        np.uint8
    ) * 255

    n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        red_mask, connectivity=8
    )
    # stats[0] is the background component; skip it.
    if n <= 1:
        return None

    best_i, best_area = -1, 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            best_i = i

    if best_area < _MIN_AREA:
        return None

    x = int(stats[best_i, cv2.CC_STAT_LEFT])
    y = int(stats[best_i, cv2.CC_STAT_TOP])
    w = int(stats[best_i, cv2.CC_STAT_WIDTH])
    h = int(stats[best_i, cv2.CC_STAT_HEIGHT])
    return x, y, w, h


@beartype.beartype
def run(cfg: Config) -> None:
    logging.basicConfig(level=logging.INFO, format=log_format)

    plotted_fpaths = sorted(cfg.plotted_dpath.glob("*.png"))
    logger.info(
        "Found %d Plotted images in '%s'", len(plotted_fpaths), cfg.plotted_dpath
    )

    results: dict[str, dict] = {}
    n_missing = 0
    n_no_bbox = 0

    for i, plotted_fpath in enumerate(plotted_fpaths):
        if i % 100 == 0:
            logger.info("Progress: %d / %d", i, len(plotted_fpaths))

        orig_fpath = cfg.originals_dpath / plotted_fpath.name
        if not orig_fpath.exists():
            logger.warning("Original image not found: '%s'", plotted_fpath.name)
            n_missing += 1
            continue

        img_arr = cv2.imread(str(plotted_fpath))
        assert img_arr is not None, f"Failed to read: '{plotted_fpath}'"
        # cv2 loads BGR; convert to RGB for the red mask.
        img_arr = img_arr[:, :, ::-1]

        bbox = detect_red_bbox(img_arr)
        if bbox is None:
            logger.warning("No red bbox found in '%s'", plotted_fpath.name)
            n_no_bbox += 1
            continue

        x, y, w, h = bbox
        results[str(orig_fpath)] = {
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "plotted_fpath": str(plotted_fpath),
        }

    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    with cfg.out.open("w") as fd:
        json.dump(results, fd, indent=2)

    logger.info("=" * 60)
    logger.info("DONE")
    logger.info("  Images processed:  %d", len(plotted_fpaths))
    logger.info("  Bboxes detected:   %d", len(results))
    logger.info("  Missing originals: %d", n_missing)
    logger.info("  No bbox detected:  %d", n_no_bbox)
    logger.info("  Output: '%s'", cfg.out)


@beartype.beartype
def main(cfg: Config) -> None:
    logging.basicConfig(level=logging.INFO, format=log_format)

    import submitit

    if cfg.slurm_acct:
        executor = submitit.SlurmExecutor(folder=cfg.log_to)
        executor.update_parameters(
            job_name="detect-scalebar-bboxes",
            time=int(cfg.n_hours * 60),
            partition=cfg.slurm_partition,
            ntasks_per_node=1,
            cpus_per_task=4,
            account=cfg.slurm_acct,
            mem="16GB",
            stderr_to_stdout=True,
            setup=["unset SLURM_CPUS_PER_TASK"],
        )
    else:
        executor = submitit.DebugExecutor(folder=cfg.log_to)

    job = executor.submit(run, cfg)
    logger.info("Submitted job %s.", job.job_id)
    job.result()


if __name__ == "__main__":
    tyro.cli(main)
