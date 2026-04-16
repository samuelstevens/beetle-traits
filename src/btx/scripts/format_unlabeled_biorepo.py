"""
Format unlabeled CarabidImaging beetle images without template matching.

Enumerates individual beetle images in Output/cropped_images/ as-is, using the
position already encoded in each filename. No template matching, no renaming.
Writes an annotations CSV with paths and taxon metadata.

INPUTS:
- /fs/ess/PAS2136/CarabidImaging/Output/cropped_images/: Individual beetle images
- /fs/ess/PAS2136/CarabidImaging/Images/FinalImages/: Group images (for paths only)
- /fs/ess/PAS2136/CarabidImaging/allIndividuals.csv: Beetle metadata

OUTPUTS:
- annotations.csv: One row per beetle with paths and taxon metadata

USAGE:
------
  uv run python -m btx.scripts.format_unlabeled_biorepo_v2

Exclude already-labeled beetles:
  uv run python -m btx.scripts.format_unlabeled_biorepo_v2 \
      --labeled-annotations data/biorepo-formatted/annotations.json
"""

import csv
import dataclasses
import json
import logging
import pathlib

import beartype
import tyro

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logger = logging.getLogger("format-unlabeled-biorepo-v2")

_ANN_COLS = [
    "beetle_position",
    "group_img",
    "rel_group_img_path",
    "abs_group_img_path",
    "rel_individual_img_path",
    "abs_individual_img_path",
    "individual_id",
    "taxon_id",
    "scientific_name",
]


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    carb_dir: pathlib.Path = pathlib.Path("/fs/ess/PAS2136/CarabidImaging")
    """Root of the CarabidImaging dataset."""

    labeled_annotations: pathlib.Path | None = None
    #pathlib.Path(
    #    "/fs/scratch/PAS2136/cain429/biorepo-formatted/annotations.json"
    #)
    """Path to labeled annotations JSON. If provided, beetles whose individual_id
    appears in the labeled data are excluded from the output CSV."""

    @property
    def images_dir(self) -> pathlib.Path:
        return self.carb_dir / "Output" / "plotted_trays"

    @property
    def cropped_dir(self) -> pathlib.Path:
        return self.carb_dir / "Output" / "cropped_images"

    @property
    def metadata_csv(self) -> pathlib.Path:
        return self.carb_dir / "allIndividuals.csv"

    
    annotations_fpath: pathlib.Path = pathlib.Path(
        "/fs/scratch/PAS2136/cain429/unlabeled_biorepo_annotations.csv")


@beartype.beartype
def build_image_index(cfg: Config) -> dict[str, pathlib.Path]:
    """Return a mapping of group image stem -> absolute path by scanning images_dir."""
    index: dict[str, pathlib.Path] = {}
    for f in cfg.images_dir.glob("*.png"):
        index[f.stem] = f
    logger.info("Indexed %d group images under %s", len(index), cfg.images_dir)
    return index


@beartype.beartype
def load_metadata(cfg: Config) -> dict[tuple[str, int], dict]:
    """Return {(group_stem, order): {individual_id, taxon_id, scientific_name}}."""
    meta: dict[tuple[str, int], dict] = {}
    with cfg.metadata_csv.open(encoding="utf-8") as fd:
        for row in csv.DictReader(fd):
            image_id = row["imageID"]
            if not image_id or image_id == "NA":
                continue
            stem = image_id.removesuffix(".png")
            order_str = row.get("Order", "")
            if not order_str or order_str == "NA":
                continue
            try:
                order = int(order_str)
            except ValueError:
                continue
            meta[(stem, order)] = {
                "individual_id": row.get("individualID") or None,
                "taxon_id": row.get("taxonID") or None,
                "scientific_name": row.get("scientificName") or None,
            }
    logger.info("Loaded metadata for %d (group, position) pairs", len(meta))
    return meta


@beartype.beartype
def load_labeled_exclusions(labeled_fpath: pathlib.Path) -> set[str]:
    """Return set of individual_ids present in the labeled annotations JSON."""
    with labeled_fpath.open(encoding="utf-8") as fd:
        labeled = json.load(fd)
    excluded = {ann["individual_id"] for ann in labeled if ann.get("individual_id")}
    logger.info(
        "Loaded %d individual_ids to exclude from labeled annotations at %s",
        len(excluded),
        labeled_fpath,
    )
    return excluded


@beartype.beartype
def main(cfg: Config) -> None:
    logging.basicConfig(level=logging.INFO, format=log_format)

    logger.info("=" * 80)
    logger.info("FORMAT UNLABELED BIOREPO V2 (no template matching)")
    logger.info("=" * 80)

    image_index = build_image_index(cfg)
    metadata = load_metadata(cfg)
    excluded_ids = (
        load_labeled_exclusions(cfg.labeled_annotations)
        if cfg.labeled_annotations
        else set()
    )

    group_dirs = sorted(d for d in cfg.cropped_dir.iterdir() if d.is_dir())
    logger.info("Found %d group directories", len(group_dirs))

    total_written = 0
    total_excluded = 0

    with cfg.annotations_fpath.open("w", newline="", encoding="utf-8") as fd:
        writer = csv.DictWriter(fd, fieldnames=_ANN_COLS)
        writer.writeheader()

        for group_dir in group_dirs:
            group_stem = group_dir.name
            individual_paths = sorted(group_dir.glob(f"{group_stem}_*.png"))
            if not individual_paths:
                logger.warning("No individual images found in %s", group_dir)
                continue

            group_image_path = image_index.get(group_stem)
            rel_group = (
                str(group_image_path.relative_to(cfg.carb_dir))
                if group_image_path
                else None
            )
            abs_group = str(group_image_path.resolve()) if group_image_path else None
            group_img_fname = group_image_path.name if group_image_path else None

            for indiv_path in individual_paths:
                try:
                    pos = int(indiv_path.stem.split("_")[-1])
                except (ValueError, IndexError):
                    logger.warning("Could not parse position from %s", indiv_path.name)
                    continue

                meta = metadata.get((group_stem, pos), {})
                individual_id = meta.get("individual_id")
                taxon_id = meta.get("taxon_id")
                scientific_name = meta.get("scientific_name")

                if individual_id and individual_id in excluded_ids:
                    total_excluded += 1
                    continue
                missing_required = any(
                    v is None
                    for v in (
                        group_img_fname,
                        rel_group,
                        abs_group,
                        individual_id,
                        taxon_id,
                        scientific_name,
                    )
                )
                if missing_required:
                    total_excluded += 1
                    continue

                writer.writerow({
                    "beetle_position": pos,
                    "group_img": group_img_fname,
                    "rel_group_img_path": rel_group,
                    "abs_group_img_path": abs_group,
                    "rel_individual_img_path": str(
                        indiv_path.relative_to(cfg.carb_dir)
                    ),
                    "abs_individual_img_path": str(indiv_path.resolve()),
                    "individual_id": individual_id,
                    "taxon_id": taxon_id,
                    "scientific_name": scientific_name,
                })
                total_written += 1

    logger.info("=" * 80)
    logger.info("DONE")
    logger.info("  Annotation rows written:          %d", total_written)
    logger.info("  Annotation rows excluded (labeled): %d", total_excluded)
    logger.info("  Annotations CSV: %s", cfg.annotations_fpath)


if __name__ == "__main__":
    tyro.cli(main)
