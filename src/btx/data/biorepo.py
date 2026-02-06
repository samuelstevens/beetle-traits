import dataclasses

import logging
import numpy as np
import pathlib
import polars as pl


import beartype
import grain

import typing as tp
from . import utils

logger = logging.getLogger("biorepo")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config(utils.Config):
    go: bool = True
    """Whether to include this dataset in training."""
    hf_root: pathlib.Path = pathlib.Path("data/biorepo")
    """Path to the dataset root"""
    annotations: pathlib.Path = pathlib.Path("data/biorepo-formatted/annotations.json")
    """Path to the annotations.json file made by running format_biorepo.py."""
    split: tp.Literal["train", "val", "test"] = "val"
    """Which split."""

    @property
    def dataset(self):
        return Dataset
    
@beartype.beartype
def _grouped_split(cfg: Config) -> pl.DataFrame:
    """
    Group-aware train/val split.

    For each species, try to grab at least two group images and 10 samples per species. All of the individuals in a single group image are either in train OR test.
    """
    df = pl.read_json(cfg.annotations)

    # Create group-level statistics dataframe
    group_stats = df.group_by(["rel_group_img_path", "taxon_id"]).agg(
        pl.len().alias("n_beetles")
    )

    # Add taxon-level group counts
    taxon_group_counts = group_stats.group_by("taxon_id").agg(
        pl.col("rel_group_img_path").n_unique().alias("n_groups")
    )

    # Join to get n_groups for each row
    group_stats = group_stats.join(taxon_group_counts, on="taxon_id")

    # Species with limited group images go ENTIRELY to validation
    val_group_imgs = set()
    val_group_imgs.update(
        group_stats
        .filter(pl.col("n_groups") <= 2)
        .get_column("rel_group_img_path")
        .to_list()
    )

    # For species with more groups, select up to 2 groups and 10 samples
    for (taxon_id,) in (
        taxon_group_counts
        .filter(pl.col("n_groups") > 2)
        .select("taxon_id")
        .iter_rows()
    ):
        taxon_groups = group_stats.filter(pl.col("taxon_id") == taxon_id).filter(
            ~pl.col("rel_group_img_path").is_in(val_group_imgs)
        )

        # Take groups until we have 2 groups or 10 samples
        total_beetles = 0
        total_groups = 0
        for group_img, n_beetles in (
            taxon_groups
            .select("rel_group_img_path", "n_beetles")
            .sample(fraction=1.0, with_replacement=False, shuffle=True, seed=0)
            .iter_rows()
        ):
            if (
                total_groups >= 2
                and total_beetles >= 20
            ):
                break

            val_group_imgs.add(group_img)
            total_beetles += n_beetles
            total_groups += 1

    # Log the split statistics
    total = len(df)
    val_df = df.filter(pl.col("rel_group_img_path").is_in(val_group_imgs))
    val_total = len(val_df)

    # Get per-taxon statistics
    val_stats = val_df.group_by("taxon_id").agg([
        pl.len().alias("val_samples"),
        pl.col("rel_group_img_path").n_unique().alias("val_groups"),
    ])

    # Join with totals
    taxon_totals = df.group_by("taxon_id").agg([
        pl.len().alias("total_samples"),
        pl.col("rel_group_img_path").n_unique().alias("total_groups"),
    ])

    split_stats = taxon_totals.join(val_stats, on="taxon_id", how="left").fill_null(0)

    # Thresholds for per-species reporting (from issue #8)
    min_groups = 2
    min_beetles = 20

    logger.info(
        f"Total samples: {total}, Val: {val_total} ({100 * val_total / total:.1f}%)"
    )

    below_threshold_species = []
    for row in split_stats.sort("taxon_id").iter_rows(named=True):
        taxon_id = row["taxon_id"]
        total_count = row["total_samples"]
        val_count = row["val_samples"]
        total_groups = row["total_groups"]
        val_groups_count = row["val_groups"]

        split_type = "ALL VAL" if val_groups_count == total_groups else "MIXED"
        pct = 100 * val_count / total_count if total_count > 0 else 0

        # Check if species is below threshold for reliable per-species reporting
        below_threshold = val_groups_count < min_groups or val_count < min_beetles
        flag = " [BELOW THRESHOLD]" if below_threshold else ""
        if below_threshold:
            below_threshold_species.append(taxon_id)

        logger.info(
            f"  {taxon_id} [{split_type}]: {val_count}/{total_count} samples ({pct:.1f}%), "
            f"{val_groups_count}/{total_groups} groups{flag}"
        )

    if below_threshold_species:
        logger.warning(
            f"{len(below_threshold_species)} species below threshold for per-species reporting "
            f"(min {min_groups} groups, {min_beetles} beetles): {below_threshold_species}. "
            "These species are included in aggregate metrics but may have unreliable per-species results."
        )

    # Build split table (each beetle inherits its group's split)
    return df.with_columns(
        pl
        .when(pl.col("rel_group_img_path").is_in(val_group_imgs))
        .then(pl.lit("val"))
        .otherwise(pl.lit("train"))
        .alias("split"),
        pl.col("rel_group_img_path")
            .str.strip_prefix("Images/")
            .str.strip_suffix(".png")
            .alias("group_img_basename")
    )


@beartype.beartype
class Dataset(grain.sources.RandomAccessDataSource):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.df = _grouped_split(cfg).filter(pl.col("split") == cfg.split)

        self.logger = logging.getLogger("biorepo-ds")

        # do not include annotations with more than 2 points
        self.df = self.df.filter(
                pl.col("measurements").map_elements(
                    lambda measurements: all(
                        len(m["polyline"]) == 2 for m in measurements
                    ),
                    return_dtype=pl.Boolean,
                )
            )
        
    def __len__(self) -> int:
        return self.df.height
    
    def __getitem__(self, idx: int) -> utils.Sample:
        """Load image and annotations for given index."""
        row = self.df.row(index=idx, named=True)
        fpath = self.cfg.hf_root / row["rel_individual_img_path"]
        assert fpath.is_file(), f"Image not found: {fpath}"

        elytra_width = None
        elytra_length = None
        scalebar = None
        for measurement in row["measurements"]:
            if measurement["measurement_type"] == "elytra_length":
                elytra_length = measurement["polyline"]
            if measurement["measurement_type"] == "elytra_width":
                elytra_width = measurement["polyline"]
            if measurement["measurement_type"] == "scalebar":
                scalebar = measurement["polyline"]

        if elytra_width is None:
            msg = f"Image {row['rel_group_img_path']} beetle {row['beetle_position']} has no elytra width."
            self.logger.error(msg)
            
        if elytra_length is None:
            msg = f"Image {row['rel_group_img_path']} beetle {row['beetle_position']} has no elytra length."
            self.logger.error(msg)
            
        if scalebar is None:
            msg = f"Image {row['rel_group_img_path']} beetle {row['beetle_position']} has no scalebar."
            self.logger.error(msg)
            

        loss_mask = np.array([1.0, 1.0])  # Train on both width and length
        msg = f"Expected loss_mask shape (2,), got {loss_mask.shape}"
        assert loss_mask.shape == (2,), msg
        return utils.Sample(
            img_fpath=str(fpath),
            points_px=np.array(elytra_width + elytra_length).reshape(2, 2, 2),
            scalebar_px=np.array(scalebar),
            loss_mask=loss_mask,
            beetle_id=row["individual_id"],
            beetle_position=row["beetle_position"],
            group_img_basename=row["group_img_basename"],
            scientific_name=row["scientific_name"],
        )
