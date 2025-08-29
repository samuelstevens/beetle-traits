"""Hawaii beetle dataset loader for trait prediction model training.

Goal:
Load individual beetle images with their trait annotations (elytra max length,
basal pronotum width, elytra max width) for training keypoint detection models.
Each sample should contain an individual beetle image cropped from the group
photo and the corresponding trait polylines in pixel coordinates.

Dataset Structure:
- annotations.json: Contains all beetle annotations with:
  - Individual image paths (relative to HuggingFace dataset root)
  - Origin coordinates (x, y) in the group image
  - Trait measurements as polylines in individual image coordinates
  - NCC score from template matching (confidence metric)
- Individual images: Located at HF_ROOT/individual_specimens/
- Group images: Located at HF_ROOT/group_images/ (not needed for training)

Implementation Challenges:

1. Path resolution:
   - Annotations contain absolute paths that need conversion to relative
   - Must handle both local and scratch filesystem paths
   Solution: Extract relative path components from indiv_img_rel_path field

2. Coordinate systems:
   - Polylines are already in individual image pixel coordinates
   - No transformation needed, just validation
   Solution: Directly use polyline_px coordinates from annotations

3. Data filtering:
   - Some beetles may have missing or incomplete annotations
   - NCC scores indicate template matching confidence
   Solution: Filter by NCC threshold (e.g., > 0.8) and validate all traits present

4. Memory efficiency:
   - Dataset has ~1600 beetles, loading all at once may be excessive
   - Images vary in size (typically 400-1000px per dimension)
   Solution: Use grain's lazy loading, load images on-demand in __getitem__

Unresolved Challenges:
- Handling variable image sizes for batch training (requires padding/resizing)
- Dealing with outlier beetle sizes (some are 1500+ pixels)
- Normalizing trait measurements across different beetle scales
- Variable keypoint counts: elytra_max_width has 2 points (73%) or 4 points (27%)
  when wings are spread. With 4 points, only outer segments matter (middle segment
  crosses the gap between wings). Models need fixed keypoint counts, so options:
  a) Always predict 4 points, ignore middle segment when present
  b) Only predict 2 endpoints, but this overestimates width for spread wings
  c) Predict wing spread as separate classification task, then variable points

Testing Strategy:
1. Load a few samples and visualize with trait polylines overlaid
2. Verify polyline coordinates fall within image bounds
3. Check distribution of NCC scores and image dimensions
4. Test with grain's DataLoader for batching compatibility
5. Validate against saved example images in random-examples/
"""

import dataclasses
import logging
import pathlib
import typing as tp

import beartype
import grain
import numpy as np
import polars as pl

from . import utils

logger = logging.getLogger("hawaii")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    hf_root: pathlib.Path = pathlib.Path("data/hawaii")
    """Path to the dataset root downloaded from HuggingFace."""
    annotations: pathlib.Path = pathlib.Path("data/hawaii-formatted/annotations.json")
    """Path to the annotations.json file made by running format_hawaii.py."""
    include_polylines: bool = True
    """Whether to include polylines (lines with more than 2 points)."""
    split: tp.Literal["train", "val"] = "train"
    """Which split."""
    # Split-related configuration.
    seed: int = 0
    """Random seed for split."""
    min_val_groups: int = 2
    """Minimum group images per species in validation."""
    min_val_beetles: int = 20
    """Minimum beetles per species in validation."""
    n_workers: int = 4

    batch_size: int = 16

    def __post_init__(self):
        # TODO: Check that hf_root exists and is a directory
        # TODO: Check that annotations.json exists and is a file.
        pass


@beartype.beartype
def _grouped_split(cfg: Config) -> pl.DataFrame:
    """
    Group-aware train/val split.

    For each species, try to grab at least two group images and 10 samples per species. All of the individuals in a single group image are either in train OR test.
    """
    df = pl.read_json(cfg.annotations)

    # Create group-level statistics dataframe
    group_stats = df.group_by(["group_img_basename", "taxon_id"]).agg(
        pl.len().alias("n_beetles")
    )

    # Add taxon-level group counts
    taxon_group_counts = group_stats.group_by("taxon_id").agg(
        pl.col("group_img_basename").n_unique().alias("n_groups")
    )

    # Join to get n_groups for each row
    group_stats = group_stats.join(taxon_group_counts, on="taxon_id")

    # Species with limited group images go ENTIRELY to validation
    val_group_imgs = set()
    val_group_imgs.update(
        group_stats.filter(pl.col("n_groups") <= cfg.min_val_groups)
        .get_column("group_img_basename")
        .to_list()
    )

    # For species with more groups, select up to cfg.min_val_groups groups and 10 samples
    for (taxon_id,) in (
        taxon_group_counts.filter(pl.col("n_groups") > cfg.min_val_groups)
        .select("taxon_id")
        .iter_rows()
    ):
        taxon_groups = group_stats.filter(pl.col("taxon_id") == taxon_id).filter(
            ~pl.col("group_img_basename").is_in(val_group_imgs)
        )

        # Take groups until we have 2 groups or 10 samples
        total_beetles = 0
        total_groups = 0
        for group_img, n_beetles in (
            taxon_groups.select("group_img_basename", "n_beetles")
            .sample(fraction=1.0, with_replacement=False, shuffle=True, seed=cfg.seed)
            .iter_rows()
        ):
            if (
                total_groups >= cfg.min_val_groups
                and total_beetles >= cfg.min_val_beetles
            ):
                break

            val_group_imgs.add(group_img)
            total_beetles += n_beetles
            total_groups += 1

    # Log the split statistics
    total = len(df)
    val_df = df.filter(pl.col("group_img_basename").is_in(val_group_imgs))
    val_total = len(val_df)

    # Get per-taxon statistics
    val_stats = val_df.group_by("taxon_id").agg([
        pl.len().alias("val_samples"),
        pl.col("group_img_basename").n_unique().alias("val_groups"),
    ])

    # Join with totals
    taxon_totals = df.group_by("taxon_id").agg([
        pl.len().alias("total_samples"),
        pl.col("group_img_basename").n_unique().alias("total_groups"),
    ])

    split_stats = taxon_totals.join(val_stats, on="taxon_id", how="left").fill_null(0)

    logger.info(
        f"Total samples: {total}, Val: {val_total} ({100 * val_total / total:.1f}%)"
    )

    for row in split_stats.sort("taxon_id").iter_rows(named=True):
        taxon_id = row["taxon_id"]
        total_count = row["total_samples"]
        val_count = row["val_samples"]
        total_groups = row["total_groups"]
        val_groups_count = row["val_groups"]

        split_type = "ALL VAL" if total_groups <= 2 else "MIXED"
        pct = 100 * val_count / total_count if total_count > 0 else 0
        logger.info(
            f"  {taxon_id} [{split_type}]: {val_count}/{total_count} samples ({pct:.1f}%), "
            f"{val_groups_count}/{total_groups} groups"
        )

    # Build split table (each beetle inherits its group's split)
    return df.with_columns(
        pl.when(pl.col("group_img_basename").is_in(val_group_imgs))
        .then(pl.lit("val"))
        .otherwise(pl.lit("train"))
        .alias("split")
    )


@beartype.beartype
class Dataset(grain.sources.RandomAccessDataSource):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.df = _grouped_split(cfg).filter(pl.col("split") == cfg.split)

        self.logger = logging.getLogger("hawaii-ds")

        if self.cfg.include_polylines:
            raise NotImplementedError()
        else:
            # measurements: list[struct[2]]
            # Example:
            # [{'measurement_type': 'elytra_max_length', 'polyline_px': [231.3699999999999, 410.8299999999999, 190.32999999999993, 239.17999999999984]}, {'measurement_type': 'basal_pronotum_width', 'polyline_px': [187.57999999999993, 237.21000000000004, 250.0300000000002, 216.0]}, {'measurement_type': 'elytra_max_width', 'polyline_px': [178.92999999999984, 341.8899999999999, 293.03999999999996, 316.3600000000001]}]
            #
            # Only pick examples where all measurements only have four floats (two points)
            self.df = self.df.filter(
                pl.col("measurements").map_elements(
                    lambda measurements: all(
                        len(m["polyline_px"]) == 4 for m in measurements
                    ),
                    return_dtype=pl.Boolean,
                )
            )

    def __len__(self) -> int:
        return self.df.height

    def __getitem__(self, idx: int) -> utils.Sample:
        """Load image and annotations for given index."""
        row = self.df.row(index=idx, named=True)
        fpath = self.cfg.hf_root / "individual_specimens" / row["indiv_img_rel_path"]
        # TODO: include error message.
        assert fpath.is_file()

        elytra_width_px = None
        elytra_length_px = None
        scalebar_px = None
        for measurement in row["measurements"]:
            if measurement["measurement_type"] == "elytra_max_length":
                elytra_length_px = measurement["polyline_px"]
            if measurement["measurement_type"] == "elytra_max_width":
                elytra_width_px = measurement["polyline_px"]
            if measurement["measurement_type"] == "scalebar":
                scalebar_px = measurement["polyline_px"]

        if elytra_width_px is None:
            self.logger.error(
                "Image %s beetle %d has no elytra width.",
                row["group_img_rel_path"],
                row["beetle_position"],
            )
            elytra_width_px = [0.0, 0.0, 0.0, 0.0]
        if elytra_length_px is None:
            self.logger.error(
                "Image %s beetle %d has no elytra length.",
                row["group_img_rel_path"],
                row["beetle_position"],
            )
            elytra_length_px = [0.0, 0.0, 0.0, 0.0]
        if scalebar_px is None:
            self.logger.error(
                "Image %s beetle %d has no scalebar.",
                row["group_img_rel_path"],
                row["beetle_position"],
            )
            scalebar_px = [0.0, 0.0, 1.0, 1.0]

        if self.cfg.include_polylines:
            raise NotImplementedError()

        return utils.Sample(
            img_fpath=str(fpath),
            points_px=np.array(elytra_width_px + elytra_length_px).reshape(2, 2, 2),
            scalebar_px=np.array(scalebar_px).reshape(2, 2),
            beetle_id=row["individual_id"],
            beetle_position=row["beetle_position"],
            group_img_basename=row["group_img_basename"],
        )
