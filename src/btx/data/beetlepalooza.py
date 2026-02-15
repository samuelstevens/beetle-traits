"""BeetlePalooza dataset

Context: This dataset will be mixed with Hawaii and BioRepo via grain.MapDataset, so we need consistent per-beetle samples matching utils.Sample.
Problem - annotator variation: Labels come from multiple annotators, so targets may be inconsistent across copies.
Why fix it: Mixed-quality labels would leak noise into the shared training pipeline and destabilize convergence.
Proposed solution: Filter rows to a trusted annotator list (currently ['isa fluck']) before producing samples.

Problem - elytra width quality: Elytra width measurements are known junk.
Why fix it: Downstream models still expect a fixed target shape, so dropping them would break shapes while keeping them may mislead metrics.
Proposed solution: Keep widths in the sample for shape compatibility but note they are low-trust until a better handling strategy exists.

Problem - annotation count drift: Some images have 2-6 annotations instead of exactly 2.
Why fix it: Variable annotation counts would break batching and target shapes.
Proposed solution: Filtering by annotator may collapse this to 2; otherwise pick a deterministic rule (e.g., first two or majority vote) to enforce exactly two annotations before batching.
"""

import dataclasses
import logging
import pathlib
import typing as tp

import beartype
import numpy as np
import polars as pl

from . import utils

logger = logging.getLogger("beetlepalooza")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config(utils.Config):
    go: bool = True
    """Whether to include this dataset in training."""
    hf_root: pathlib.Path = pathlib.Path("data/beetlepalooza")
    """Path to the dataset root downloaded from HuggingFace."""
    annotations: pathlib.Path = pathlib.Path(
        "data/beetlepalooza-formatted/annotations.json"
    )
    """Path to the annotations.json file."""

    include_polylines: bool = False
    """Whether to include polylines (lines with more than 2 points)."""
    annotators: list[str] = dataclasses.field(default_factory=lambda: ["IsaFluck"])
    """According to Aly, we need to filter by `annotator="IsaFluck"`. See https://hdrimageomics.slack.com/archives/C08T6MCFME1/p1763993618130059?thread_ts=1763935266.449869&cid=C08T6MCFME1 for more context."""

    @property
    def key(self) -> str:
        return "beetlepalooza"

    @property
    def dataset(self):
        return Dataset

    def __post_init__(self):
        # TODO: Check that hf_root exists and is a directory
        # TODO: Check that annotations.json exists and is a file.
        pass


@beartype.beartype
def _trusted_data(cfg: Config) -> pl.DataFrame:
    """
    Filter the annotations to only include trusted annotators
    """
    annotations_df = pl.read_json(cfg.annotations)
    annotations_df_ex = annotations_df.explode("measurements").with_columns(
        pl.col("measurements").struct.unnest()
    )

    measurement_count_before = annotations_df_ex.height
    logger.info(
        f"Number of annotation measurements before filtering: {measurement_count_before}\nNumber of annotations: {annotations_df.height}"
    )

    # remove annotations with no measurements
    annotations_df_ex = annotations_df_ex.filter(~pl.col("measurements").is_null())

    # check if each group of individual_id has 1 elytra_width and 1 elytra_length measurements from trusted sources
    annotations_trusted = (
        annotations_df_ex
        .with_columns([pl.col("annotator").is_in(cfg.annotators).alias("is_trusted")])
        # compute per-group whether there's any trusted annotator
        .with_columns([
            # True if any trusted annotator produced an 'elytra_width' measurement for this individual
            ((pl.col("is_trusted")) & (pl.col("measurement_type") == "elytra_width"))
            .any()
            .over("individual_id")
            .alias("group_has_trusted_width"),
            # True if any trusted annotator produced an 'elytra_length' measurement for this individual
            ((pl.col("is_trusted")) & (pl.col("measurement_type") == "elytra_length"))
            .any()
            .over("individual_id")
            .alias("group_has_trusted_length"),
        ])
        # keep rows where the group doesn't have a trusted width or length, is trusted,
        # isn't trusted but is a length measurement
        # and group doesn't have a trusted length measurement, or isn't trusted but is width measurement and group doesn't have a trusted length measurement
        .filter(
            (
                (
                    ~pl.col("group_has_trusted_length")
                    & ~pl.col("group_has_trusted_width")
                )
                | (pl.col("is_trusted"))
                | (
                    ~pl.col("group_has_trusted_length")
                    & (pl.col("measurement_type") == "elytra_length")
                )
                | (
                    ~pl.col("group_has_trusted_width")
                    & (pl.col("measurement_type") == "elytra_width")
                )
            )
        )
        .drop([
            "is_trusted",
            "group_has_trusted_width",
            "group_has_trusted_length",
        ])  # optional cleanup
    )

    logger.info(
        f"{measurement_count_before - annotations_trusted.height} measurements were filtered out when finding high trust measurements"
    )

    # select the first elytra width and the first elytra length measurement for each individual_id
    choose_first_per_individual_id = annotations_trusted.group_by(
        ["individual_id", "measurement_type"], maintain_order=True
    ).agg(pl.all().first())

    # now implode the df to get measurements with the length and width
    annotations_filtered = choose_first_per_individual_id.group_by([
        "individual_id",
        "group_img_basename",
        "group_img_rel_path",
        "beetle_position",
        "indiv_img_rel_path",
        "indiv_img_abs_path",
        "scalebar_px",
        "scientific_name",
    ]).agg(pl.col("measurements"))
    # Keep dataset indexing deterministic across runs.
    annotations_filtered = annotations_filtered.sort([
        "individual_id",
        "beetle_position",
        "group_img_basename",
        "indiv_img_rel_path",
    ])
    logger.info(
        f"{annotations_filtered.height} annotations remain after enforcing 2 measurements per annotation."
    )

    return annotations_filtered


@beartype.beartype
class Dataset(utils.Dataset):
    _cfg: Config

    def __init__(self, cfg: Config):
        self._cfg = cfg
        self.df = _trusted_data(cfg)
        self.logger = logging.getLogger("beetlepalooza-ds")
        self.logger.warning("Warning: elytra_width measurements are inaccurate")
        msg = "BeetlePalooza dataset is empty after annotator/measurement filtering."
        assert self.df.height > 0, msg

    @property
    def cfg(self) -> Config:
        return self._cfg

    def __len__(self) -> int:
        return self.df.height

    def __getitem__(self, idx: tp.SupportsIndex) -> utils.Sample:
        """Load image and annotations for given index."""
        row = self.df.row(index=int(idx), named=True)

        if "part_000" in row["indiv_img_abs_path"]:
            row["indiv_img_rel_path"] = "part_000/" + row["indiv_img_rel_path"]
        else:
            row["indiv_img_rel_path"] = "part_001/" + row["indiv_img_rel_path"]

        fpath = self.cfg.hf_root / "individual_specimens" / row["indiv_img_rel_path"]
        assert fpath.is_file(), f"Image not found: {fpath}"

        elytra_width_px = None
        elytra_length_px = None

        # Convert scalebar struct to list
        scalebar_struct = row.get("scalebar_px", None)
        if scalebar_struct is None:
            self.logger.error(
                "Image %s beetle %d has no scalebar struct in row; using default.",
                row.get("group_img_rel_path", "<unknown>"),
                row.get("beetle_position", -1),
            )
            scalebar_px = [0.0, 0.0, 1.0, 1.0]
        else:
            # Defensive extraction in case the struct shape differs; ensure floats
            scalebar_px = [
                float(scalebar_struct["x1"]),
                float(scalebar_struct["y1"]),
                float(scalebar_struct["x2"]),
                float(scalebar_struct["y2"]),
            ]

        for measurement in row["measurements"]:
            coords = measurement["coords_px"]
            if measurement["measurement_type"] == "elytra_length":
                elytra_length_px = [
                    float(coords["x1"]),
                    float(coords["y1"]),
                    float(coords["x2"]),
                    float(coords["y2"]),
                ]
            else:
                elytra_width_px = [
                    float(coords["x1"]),
                    float(coords["y1"]),
                    float(coords["x2"]),
                    float(coords["y2"]),
                ]

        if elytra_width_px is None:
            msg = f"Image {row['group_img_rel_path']} beetle {row['beetle_position']} has no elytra width."
            self.logger.error(msg)
            elytra_width_px = [0.0, 0.0, 0.0, 0.0]
        if elytra_length_px is None:
            msg = f"Image {row['group_img_rel_path']} beetle {row['beetle_position']} has no elytra length."
            self.logger.error(msg)
            elytra_length_px = [0.0, 0.0, 0.0, 0.0]

        if self.cfg.include_polylines:
            raise NotImplementedError()

        # Skip width (inaccurate), train on length only
        loss_mask = np.array([0.0, 1.0])
        msg = f"Expected loss_mask shape (2,), got {loss_mask.shape}"
        assert loss_mask.shape == (2,), msg
        return utils.Sample(
            img_fpath=str(fpath),
            points_px=np.array(elytra_width_px + elytra_length_px).reshape(2, 2, 2),
            scalebar_px=np.array(scalebar_px).reshape(2, 2),
            loss_mask=loss_mask,
            beetle_id=row["individual_id"],
            beetle_position=row["beetle_position"],
            group_img_basename=row["group_img_basename"],
            scientific_name=row["scientific_name"],
        )
