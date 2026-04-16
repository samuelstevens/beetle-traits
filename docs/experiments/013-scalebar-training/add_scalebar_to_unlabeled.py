"""Join scalebar predictions into the unlabeled biorepo CSV.

Reads fdpj3qnm_px_per_cm.json (keyed by abs_group_img_path) and
all_unlabeled_biorepo_annotations.csv, joins on abs_group_img_path, and writes
a new CSV with four extra columns added per row:

  px_per_cm       - pixels per centimetre for that group image
  scalebar_x0/y0  - first scalebar endpoint in group-image pixel coordinates
  scalebar_x1/y1  - second scalebar endpoint in group-image pixel coordinates

Rows whose group image is not in the JSON are kept but the four columns are
null (scalebar_valid will remain False in biorepo.py for those rows).

USAGE:
------
  uv run python docs/experiments/013-scalebar-training/add_scalebar_to_unlabeled.py
"""

import json
import logging
import pathlib

import polars as pl

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logger = logging.getLogger("add-scalebar-to-unlabeled")

_ROOT = pathlib.Path("docs/experiments/013-scalebar-training")
_JSON_FPATH = _ROOT / "results" / "fdpj3qnm_px_per_cm.json"
_CSV_FPATH = pathlib.Path(
    "/fs/scratch/PAS2136/cain429/all_unlabeled_biorepo_annotations.csv"
)
_OUT_FPATH = pathlib.Path("/fs/scratch/PAS2136/cain429/unlabeled_with_scalebar.csv")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format=log_format)

    assert _JSON_FPATH.exists(), f"JSON not found: '{_JSON_FPATH}'"
    assert _CSV_FPATH.exists(), f"CSV not found: '{_CSV_FPATH}'"

    with _JSON_FPATH.open() as fd:
        predictions: dict[str, dict] = json.load(fd)
    logger.info(
        "Loaded %d scalebar predictions from '%s'", len(predictions), _JSON_FPATH
    )

    # Build a lookup frame: abs_group_img_path -> four scalebar columns.
    rows = []
    for img_fpath, rec in predictions.items():
        pts = rec["scalebar_pts"]  # [[x0,y0],[x1,y1]]
        rows.append({
            "abs_group_img_path": img_fpath,
            "px_per_cm": rec["px_per_cm"],
            "scalebar_x0": pts[0][0],
            "scalebar_y0": pts[0][1],
            "scalebar_x1": pts[1][0],
            "scalebar_y1": pts[1][1],
        })
    scalebar_df = pl.DataFrame(rows)

    unlabeled_df = pl.read_csv(_CSV_FPATH)
    logger.info("Loaded %d unlabeled rows from '%s'", unlabeled_df.height, _CSV_FPATH)

    out_df = unlabeled_df.join(scalebar_df, on="abs_group_img_path", how="left")

    n_matched = out_df.filter(pl.col("px_per_cm").is_not_null()).height
    n_missing = out_df.height - n_matched
    logger.info(
        "%d / %d rows matched a scalebar prediction (%d missing)",
        n_matched,
        out_df.height,
        n_missing,
    )

    _OUT_FPATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_csv(_OUT_FPATH)
    logger.info("Wrote '%s'", _OUT_FPATH)


if __name__ == "__main__":
    main()
