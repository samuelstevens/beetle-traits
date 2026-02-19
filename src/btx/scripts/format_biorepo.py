"""
Format and validate beetle position annotations from biorepo group images.

This script processes beetle collections by:
1. Template matching individual beetles on group images to find their spatial locations
2. Grouping beetles into rows and inferring correct positions (top-to-bottom, left-to-right)
3. Renaming individual beetle files to match inferred positions
4. Creating measurement annotations with taxon info and measurement coordinates
5. Validating all annotations ensure coordinates stay within image bounds
6. Generating validation statistics and outputting validated annotations

INPUTS:
- Group images (PNG): Original images with multiple beetles
- Individual beetles (PNG): Pre-extracted individual beetle images
- Beetle metadata (CSV): Scientific names and taxon IDs
- TORAS annotations (JSON): Manual measurements with coordinates

OUTPUTS:
- annotations.json: Validated beetle annotations with measurements and metadata
- validation_stats.json: Statistics on validation results and data quality

USAGE:
------
Local execution:
  python -m btx.scripts.format_biorepo

With Slurm (parallel processing):
  python -m btx.scripts.format_biorepo --slurm-acct=YOUR_ACCOUNT --slurm-partition=parallel

Custom paths:
  python -m btx.scripts.format_biorepo --biorepo-dir=./data/biorepo --dump-to=./data/biorepo-formatted

All parameters are configurable via command line (see Config class for available options).
"""

import dataclasses
import gc
import json
import logging
import pathlib
import time

import beartype
import numpy as np
import polars as pl
import skimage.feature
import submitit
import tyro
from PIL import Image, ImageDraw, ImageFont

import btx.helpers

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logger = logging.getLogger("format-biorepo")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    biorepo_dir: pathlib.Path = pathlib.Path("./data/biorepo")
    """Where biorepo data is stored."""

    dump_to: pathlib.Path = pathlib.Path("./data/biorepo-formatted")
    """Where to save formatted data."""

    toras_anns: pathlib.Path = pathlib.Path("./data/biorepo/completed_annotations")
    """Where the toras annotations are stored"""

    @property
    def images_dir(self) -> pathlib.Path:
        """Where group images are stored."""
        return self.biorepo_dir / "Images"

    @property
    def output_dir(self) -> pathlib.Path:
        """Where individual beetle images are stored."""
        return self.biorepo_dir / "Output"

    @property
    def viz_output_dir(self) -> pathlib.Path:
        """Where to save visualizations."""
        return self.biorepo_dir / "template_match_output"

    @property
    def metadata_csv(self) -> pathlib.Path:
        """Metadata CSV file."""
        return self.biorepo_dir / "allIndividuals_subset.csv"

    row_center_tolerance_ratio: float = 50 / 221.4
    """Ratio of tolerance to average beetle height for row grouping."""

    create_template_match_images: bool = True
    """Whether to create template matching visualization images."""

    # Slurm configuration
    slurm_acct: str = ""
    """Slurm account to use. If empty, uses DebugExecutor (local execution)."""

    slurm_partition: str = "parallel"
    """Slurm partition to use."""

    log_to: pathlib.Path = pathlib.Path("./logs")
    """Where to save submitit/slurm logs."""

    n_hours: float = 2.0
    """Number of hours to request for each job."""

    groups_per_job: int = 4
    """Number of group images to process per job."""

    # Visualization settings
    bbox_color: tuple[int, int, int] = (255, 0, 0)
    """Red for bounding boxes."""

    bbox_width: int = 4
    """Width of bounding box lines."""

    text_color: tuple[int, int, int] = (255, 255, 255)
    """White for text labels."""


@beartype.beartype
@dataclasses.dataclass
class BeetleMatch:
    """Result of template matching for one beetle."""

    x: float  # Top-left x coordinate
    y: float  # Top-left y coordinate
    width: int  # Width of individual image
    height: int  # Height of individual image
    ncc: float  # Normalized cross-correlation score
    individual_path: pathlib.Path  # Full path to individual image
    filename_position: int  # Position claimed in filename


@beartype.beartype
@dataclasses.dataclass
class RenameOperation:
    """Information about a file rename operation."""

    old_path: pathlib.Path
    new_path: pathlib.Path
    group_image: str
    old_position: int
    new_position: int


@beartype.beartype
def template_match_beetles(
    group_image_path: pathlib.Path,
    individual_paths: list[pathlib.Path],
) -> list[BeetleMatch]:
    """
    Template match all individual beetles on the group image.

    Args:
        group_image_path: Path to group image
        individual_paths: List of individual beetle image paths

    Returns:
        List of BeetleMatch objects with coordinates and scores
    """
    logger.info("Loading group image: %s", group_image_path)
    group_img = np.asarray(Image.open(group_image_path).convert("L"))
    logger.info("Group image size: %s", group_img.shape)

    matches = []

    for indiv_path in individual_paths:
        try:
            filename_pos = int(indiv_path.stem.split("_")[-1])
        except (ValueError, IndexError):
            logger.warning("Could not parse position from %s", indiv_path.name)
            continue

        logger.info(
            "Template matching: %s (position %d)", indiv_path.name, filename_pos
        )

        try:
            template = np.asarray(Image.open(indiv_path).convert("L"))
            logger.info("Template size: %s", template.shape)

            corr = skimage.feature.match_template(group_img, template, pad_input=False)

            if corr.size == 0.0:
                logger.warning("Empty correlation map for %s", indiv_path.name)
                continue

            max_idx = np.argmax(corr)
            y, x = np.unravel_index(max_idx, corr.shape)
            ncc_score = float(corr.flat[max_idx])

            match = BeetleMatch(
                x=float(x),
                y=float(y),
                width=template.shape[1],
                height=template.shape[0],
                ncc=ncc_score,
                individual_path=indiv_path,
                filename_position=filename_pos,
            )

            logger.info(
                "Found at: (%.1f, %.1f), NCC: %.3f", match.x, match.y, match.ncc
            )
            matches.append(match)

        except Exception as e:
            logger.exception(
                "Template match failed for %s", indiv_path.name, exc_info=e
            )

    return matches


@beartype.beartype
def beetles_on_same_row(
    beetle1: BeetleMatch, beetle2: BeetleMatch, tolerance: float = 30
) -> bool:
    """
    Check if two beetles are on the same row based on their vertical alignment.

    Checks if centers, tops, OR bottoms are within tolerance. This handles cases where
    beetles are at different heights but clearly on the same row.

    Args:
        beetle1: First beetle
        beetle2: Second beetle
        tolerance: Vertical tolerance in pixels

    Returns:
        True if any of (centers, tops, or bottoms) are within tolerance
    """
    # Calculate vertical positions
    top1, top2 = beetle1.y, beetle2.y
    bottom1 = beetle1.y + beetle1.height
    bottom2 = beetle2.y + beetle2.height
    center1_y = beetle1.y + beetle1.height / 2
    center2_y = beetle2.y + beetle2.height / 2

    # Check if any alignment metric is within tolerance
    centers_aligned = abs(center1_y - center2_y) <= tolerance
    tops_aligned = abs(top1 - top2) <= tolerance
    bottoms_aligned = abs(bottom1 - bottom2) <= tolerance

    return centers_aligned or tops_aligned or bottoms_aligned


@beartype.beartype
def group_beetles_into_rows(
    matches: list[BeetleMatch], center_tolerance: float = 30
) -> list[list[BeetleMatch]]:
    """
    Group beetles into rows based on their vertical centers.

    Args:
        matches: List of BeetleMatch objects
        center_tolerance: Max distance between vertical centers for beetles to be in same row

    Returns:
        List of rows, where each row is a list of BeetleMatch objects sorted left-to-right
    """
    if not matches:
        return []

    # Start with each beetle in its own row
    rows = [[match] for match in matches]

    # Iteratively merge rows that have beetles with similar vertical centers
    merged = True
    while merged:
        merged = False
        new_rows = []
        used = set()

        for i, row1 in enumerate(rows):
            if i in used:
                continue

            # Try to find another row that should be merged with this one
            for j, row2 in enumerate(rows[i + 1 :], start=i + 1):
                if j in used:
                    continue

                # Check if any beetle in row1 is on same row as any beetle in row2
                same_row_found = False
                for beetle1 in row1:
                    for beetle2 in row2:
                        if beetles_on_same_row(beetle1, beetle2, center_tolerance):
                            same_row_found = True
                            break
                    if same_row_found:
                        break

                if same_row_found:
                    # Merge the rows
                    new_rows.append(row1 + row2)
                    used.add(i)
                    used.add(j)
                    merged = True
                    break

            if i not in used:
                new_rows.append(row1)
                used.add(i)

        rows = new_rows

    # Sort each row left-to-right by x coordinate
    for row in rows:
        row.sort(key=lambda m: m.x)

    # Sort rows top-to-bottom by average center y coordinate
    rows.sort(key=lambda row: sum(m.y + m.height / 2 for m in row) / len(row))

    return rows


@beartype.beartype
def infer_positions(
    matches: list[BeetleMatch], row_tolerance: float
) -> dict[int, BeetleMatch]:
    """
    Infer correct beetle positions based on row grouping and spatial layout.

    Args:
        matches: List of BeetleMatch objects
        row_tolerance: Tolerance for grouping beetles into same row (pixels)

    Returns:
        Dictionary mapping inferred position (1, 2, 3...) to BeetleMatch
    """
    logger.info("INFERRING BEETLE POSITIONS (tolerance=%.1fpx)", row_tolerance)

    rows = group_beetles_into_rows(matches, center_tolerance=row_tolerance)
    logger.info("Grouped into %d rows", len(rows))

    for row_idx, row in enumerate(rows):
        avg_center_y = sum(m.y + m.height / 2 for m in row) / len(row)
        avg_top_y = sum(m.y for m in row) / len(row)
        avg_bottom_y = sum(m.y + m.height for m in row) / len(row)
        logger.info(
            "Row %d: %d beetles | avg_top=%.1f avg_center=%.1f avg_bottom=%.1f",
            row_idx + 1,
            len(row),
            avg_top_y,
            avg_center_y,
            avg_bottom_y,
        )
        for beetle in row:
            center_y = beetle.y + beetle.height / 2
            bottom_y = beetle.y + beetle.height
            logger.info(
                "  x=%.1f top=%.1f center=%.1f bottom=%.1f size=(%dx%d)",
                beetle.x,
                beetle.y,
                center_y,
                bottom_y,
                beetle.width,
                beetle.height,
            )

    position_map = {}
    current_pos = 1

    logger.info(
        "Assigning positions: rows top-to-bottom, beetles left-to-right within each row"
    )
    for idx, row in enumerate(rows):
        logger.info("Assigning positions for row %d", idx + 1)
        for beetle in row:
            position_map[current_pos] = beetle
            logger.info(
                "  Position %d -> (%.1f, %.1f)", current_pos, beetle.x, beetle.y
            )
            current_pos += 1

    return position_map


@beartype.beartype
def visualize_results(
    group_image_path: pathlib.Path,
    position_map: dict[int, BeetleMatch],
    output_path: pathlib.Path,
    cfg: Config,
):
    """Draw bounding boxes on the group image."""
    logger.info("Creating visualization for %s", group_image_path)

    img = Image.open(group_image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except OSError:
        font = ImageFont.load_default()

    for inferred_pos, match in position_map.items():
        logger.info(
            "Drawing beetle #%d at (%.1f, %.1f) labeled %d",
            inferred_pos,
            match.x,
            match.y,
            match.filename_position,
        )

        bbox = [match.x, match.y, match.x + match.width, match.y + match.height]
        draw.rectangle(bbox, outline=cfg.bbox_color, width=cfg.bbox_width)

        label = f"#{inferred_pos}"
        text_pos = (match.x + 10, match.y + 10)

        bbox_text = draw.textbbox(text_pos, label, font=font)
        draw.rectangle(bbox_text, fill=(0, 0, 0))
        draw.text(text_pos, label, fill=cfg.text_color, font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    logger.info("Saved visualization to: %s", output_path)


@beartype.beartype
def execute_renames(
    rename_ops: list[RenameOperation],
) -> tuple[int, int, list[RenameOperation]]:
    """
    Execute file rename operations using a two-phase approach to avoid conflicts.

    Args:
        rename_ops: List of rename operations to perform

    Returns:
        Tuple of (successful_renames, failed_renames, completed_operations)
    """
    successful = 0
    failed = 0
    completed_ops = []

    # Two-phase rename to avoid conflicts
    temp_mappings = []

    logger.info("PHASE 1: Renaming to temporary names")

    for idx, op in enumerate(rename_ops, 1):
        temp_path = op.old_path.with_suffix(op.old_path.suffix + f".tmp_rename_{idx}")

        logger.info(
            "[%d/%d] %s: Temp rename phase", idx, len(rename_ops), op.group_image
        )

        try:
            if not op.old_path.exists():
                logger.error("Source file not found: %s", op.old_path)
                failed += 1
                continue

            op.old_path.rename(temp_path)
            temp_mappings.append((temp_path, op.new_path, op))
            logger.info("Phase 1 OK: %s -> %s", op.old_path.name, temp_path.name)

        except Exception as e:
            logger.exception("Phase 1 rename failed for %s", op.old_path, exc_info=e)
            failed += 1

    logger.info("PHASE 2: Renaming to final names")

    for idx, (temp_path, new_path, op) in enumerate(temp_mappings, 1):
        logger.info(
            "[%d/%d] %s: position %d -> %d",
            idx,
            len(temp_mappings),
            op.group_image,
            op.old_position,
            op.new_position,
        )

        try:
            new_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path.rename(new_path)
            logger.info("Renamed successfully: %s -> %s", op.old_path, op.new_path)
            successful += 1
            completed_ops.append(op)

        except Exception as e:
            logger.exception("Final rename failed for %s", temp_path, exc_info=e)
            failed += 1

            try:
                temp_path.rename(op.old_path)
                logger.warning("Restored to original: %s", op.old_path)
            except Exception as restore_error:
                logger.error(
                    "Could not restore temp file %s: %s", temp_path, restore_error
                )

    return successful, failed, completed_ops


@beartype.beartype
def process_single_image(
    group_image_name: str,
    cfg: Config,
) -> tuple[list[RenameOperation], pl.DataFrame]:
    """
    Process a single group image.

    Returns:
        Tuple of (list of RenameOperation objects, DataFrame with beetle match info)
    """
    logger.info("Processing group image: %s", group_image_name)

    rename_operations = []

    group_image_path = cfg.images_dir / group_image_name
    if not group_image_path.exists():
        logger.error("Group image not found at %s", group_image_path)
        return rename_operations, pl.DataFrame()

    group_stem = group_image_path.stem
    individual_dir = cfg.output_dir / group_stem

    if not individual_dir.exists():
        logger.warning("No individual directory found at %s", individual_dir)
        return rename_operations, pl.DataFrame()

    individual_paths = sorted(individual_dir.glob(f"{group_stem}_*.png"))
    logger.info("Found %d individual beetles", len(individual_paths))

    if len(individual_paths) == 0:
        logger.warning("No individual beetles found, skipping %s", group_image_name)
        return rename_operations, pl.DataFrame()

    # Template match all beetles
    matches = template_match_beetles(group_image_path, individual_paths)

    if not matches:
        logger.error("No successful template matches for %s", group_image_name)
        return rename_operations, pl.DataFrame()

    logger.info("Successfully matched %d beetles", len(matches))

    avg_height = sum(m.height for m in matches) / len(matches)
    avg_width = sum(m.width for m in matches) / len(matches)
    min_height = min(m.height for m in matches)
    max_height = max(m.height for m in matches)

    logger.info(
        "Beetle size stats | avg_height=%.1f avg_width=%.1f height_range=%d-%d",
        avg_height,
        avg_width,
        min_height,
        max_height,
    )

    row_center_tolerance = avg_height * cfg.row_center_tolerance_ratio
    logger.info("Row center tolerance: %.1f pixels", row_center_tolerance)

    position_map = infer_positions(matches, row_center_tolerance)

    output_path = None
    if cfg.create_template_match_images:
        output_path = cfg.viz_output_dir / f"{group_stem}_annotated.png"
        visualize_results(group_image_path, position_map, output_path, cfg)

    logger.info("FILE RENAMING ANALYSIS")

    for inferred_pos, match in sorted(position_map.items()):
        if inferred_pos != match.filename_position:
            old_path = match.individual_path
            new_filename = f"{group_stem}_{inferred_pos}.png"
            new_path = individual_dir / new_filename

            rename_op = RenameOperation(
                old_path=old_path,
                new_path=new_path,
                group_image=group_image_name,
                old_position=match.filename_position,
                new_position=inferred_pos,
            )
            rename_operations.append(rename_op)
            logger.info(
                "Position %d needs rename (currently %d)",
                inferred_pos,
                match.filename_position,
            )
        else:
            logger.info("Position %d correct (no rename)", inferred_pos)

    logger.info("Creating match info dataframe")

    match_records = []
    for inferred_pos, match in sorted(position_map.items()):
        match_records.append({
            "group_image": group_image_name,
            "individual_filename": match.individual_path.name,
            "inferred_position": inferred_pos,
            "filename_position": match.filename_position,
            "offset_x": match.x,
            "offset_y": match.y,
            "width": match.width,
            "height": match.height,
            "center_x": match.x + match.width / 2,
            "center_y": match.y + match.height / 2,
            "ncc_score": match.ncc,
        })

    match_df = pl.DataFrame(match_records)
    logger.info("Created dataframe with %d beetle matches", len(match_df))

    logger.info(
        "Summary | image=%s beetles=%d rename_ops=%d viz=%s",
        group_image_name,
        len(matches),
        len(rename_operations),
        output_path if output_path else "(skipped)",
    )

    return rename_operations, match_df


@beartype.beartype
def worker_fn(
    cfg: Config,
    group_image_names: list[str],
) -> tuple[list[RenameOperation], list[pl.DataFrame]]:
    """
    Worker function for parallel processing of group images.

    Processes a batch of group images and returns rename operations and match dataframes.
    This function is designed to be run in parallel via Slurm.

    Args:
        cfg: Configuration object
        group_image_names: List of group image filenames to process

    Returns:
        Tuple of (list of RenameOperation, list of match DataFrames)
    """
    logging.basicConfig(level=logging.INFO, format=log_format)
    worker_logger = logging.getLogger("format-biorepo.worker")

    worker_logger.info("Starting worker with %d group images", len(group_image_names))

    all_rename_ops = []
    all_match_dfs = []

    for idx, group_image_name in enumerate(group_image_names):
        worker_logger.info(
            "Processing group %d/%d: %s",
            idx + 1,
            len(group_image_names),
            group_image_name,
        )

        try:
            rename_ops, match_df = process_single_image(group_image_name, cfg)
            all_rename_ops.extend(rename_ops)
            if not match_df.is_empty():
                all_match_dfs.append(match_df)
            worker_logger.info("Successfully processed %s", group_image_name)
        except Exception as e:
            worker_logger.error("Error processing %s: %s", group_image_name, e)
            import traceback

            traceback.print_exc()
            continue

        gc.collect()

    worker_logger.info(
        "Worker completed. Processed %d groups, %d rename ops, %d match dfs",
        len(group_image_names),
        len(all_rename_ops),
        len(all_match_dfs),
    )

    return all_rename_ops, all_match_dfs


@beartype.beartype
def offset_polyline(polyline: list, offset_x: float, offset_y: float) -> list:
    """Offset all points in a polyline by the given x and y offsets.

    Input polyline from polars .to_list() is: [[[x1, y1, x2, y2, ...]]]
    Output is in paired format: [[x1+ox, y1+oy], [x2+ox, y2+oy], ...]
    """
    if not polyline or not polyline[0] or not polyline[0][0]:
        return []
    points = polyline[0][0]  # Unwrap two levels: [[[...]]] -> [...]
    # Points are in flat format: [x1, y1, x2, y2, ...]
    result = []
    for i in range(0, len(points), 2):
        result.append([points[i] - offset_x, points[i + 1] - offset_y])
    return result


@beartype.beartype
def create_measurements_annotations(
    cfg: Config, template_match_df: pl.DataFrame, beetle_metadata_df: pl.DataFrame
) -> list[dict]:
    annotations_path = cfg.toras_anns
    with annotations_path.open("r", encoding="utf-8") as f:
        toras_data = json.load(f)

    measurements_per_beetle = []  # build json measurements for each beetle

    image_count = len(toras_data["images"])

    toras_measurements_df = pl.DataFrame(toras_data["annotations"], strict=False)
    toras_measurements_df = toras_measurements_df.with_columns(
        pl.col("name").str.replace("Entity ", "").cast(pl.Int64).alias("Entity")
    )
    for i in range(image_count):
        image_df = toras_measurements_df.filter(pl.col("image_id") == i)
        group_img_name = (
            toras_data["images"][i]["toras_path"].lstrip("/").replace(".jpg", ".png")
        )

        scalebar_row = image_df.filter(pl.col("Entity") == pl.col("Entity").max())[0]

        for j in range(
            int(scalebar_row["Entity"].item() / 3)
        ):  # loops through each beetle since 3 measurements per beetle
            entity_num = j * 3 + 1
            beetle_num = j + 1

            # Look up offsets from template_match_df
            match_row = template_match_df.filter(
                (pl.col("group_image") == group_img_name)
                & (pl.col("inferred_position") == beetle_num)
            )

            if match_row.is_empty():
                logger.warning(
                    "No match found for %s beetle %d", group_img_name, beetle_num
                )
                continue

            offset_x = match_row["offset_x"].item()
            offset_y = match_row["offset_y"].item()

            metadata_row = beetle_metadata_df.filter(
                (pl.col("imageID") == group_img_name) & (pl.col("Order") == beetle_num)
            )

            if metadata_row.is_empty():
                logger.warning(
                    "No metadata found for %s beetle %d", group_img_name, beetle_num
                )
                taxon_id = None
                scientific_name = None
                individual_id = None
            else:
                taxon_id = metadata_row["taxonID"][0]
                scientific_name = metadata_row["scientificName"][0]
                individual_id = metadata_row["individualID"][0]

            length_row = image_df.filter(pl.col("Entity") == entity_num)[0]
            width_row = image_df.filter(pl.col("Entity") == entity_num + 1)[0]
            pronotum_row = image_df.filter(pl.col("Entity") == entity_num + 2)[0]

            # Convert polylines from group image coordinates to individual beetle image coordinates
            # by subtracting the offset (original coords are in group space, we want individual space)
            length_json = {
                "measurement_type": "elytra_length",
                "polyline": offset_polyline(
                    length_row["polyline"].to_list(), offset_x, offset_y
                ),
            }
            width_json = {
                "measurement_type": "elytra_width",
                "polyline": offset_polyline(
                    width_row["polyline"].to_list(), offset_x, offset_y
                ),
            }
            pronotum_json = {
                "measurement_type": "pronotum_width",
                "polyline": offset_polyline(
                    pronotum_row["polyline"].to_list(), offset_x, offset_y
                ),
            }
            # Scalebar stays in group image coordinates but convert to paired format [[x1,y1], [x2,y2], ...]
            scalebar_json = {
                "measurement_type": "scalebar",
                "polyline": offset_polyline(
                    scalebar_row["polyline"].to_list(), 0.0, 0.0
                ),
            }

            # Build image paths
            group_stem = group_img_name.replace(".png", "")
            individual_filename = f"{group_stem}_{beetle_num}.png"

            rel_group_img_path = f"Images/{group_img_name}"
            abs_group_img_path = str((cfg.images_dir / group_img_name).resolve())
            rel_individual_img_path = f"Output/{group_stem}/{individual_filename}"
            abs_individual_img_path = str(
                (cfg.output_dir / group_stem / individual_filename).resolve()
            )

            beetle_annotation_json = {
                "beetle_position": beetle_num,
                "group_img": group_img_name,
                "rel_group_img_path": rel_group_img_path,
                "abs_group_img_path": abs_group_img_path,
                "rel_individual_img_path": rel_individual_img_path,
                "abs_individual_img_path": abs_individual_img_path,
                "offset_x": offset_x,
                "offset_y": offset_y,
                "taxon_id": taxon_id,
                "individual_id": individual_id,
                "scientific_name": scientific_name,
                "measurements": [length_json, width_json, pronotum_json, scalebar_json],
            }
            measurements_per_beetle.append(beetle_annotation_json)

    logger.info("Created %d beetle annotations", len(measurements_per_beetle))

    return measurements_per_beetle


@beartype.beartype
def validate_annotations(cfg: Config, annotations: list[dict]) -> list[dict]:
    """
    Validate annotations ensuring coordinates are within individual image bounds.

    Args:
        cfg: Configuration object
        annotations: List of beetle annotation dictionaries

    Returns:
        List of valid annotations (invalid ones are removed)
    """
    from PIL import Image

    valid_annotations = []
    invalid_annotations = []
    stats = {
        "total_annotations": len(annotations),
        "valid_annotations": 0,
        "invalid_annotations": 0,
        "invalid_beetles": [],
        "polylines_with_more_than_2_points": [],
        "empty_polylines": [],
        "measurement_stats": {
            "elytra_length": {"total": 0, "valid": 0, "out_of_bounds": 0, "empty": 0},
            "elytra_width": {"total": 0, "valid": 0, "out_of_bounds": 0, "empty": 0},
            "pronotum_width": {"total": 0, "valid": 0, "out_of_bounds": 0, "empty": 0},
            "scalebar": {"total": 0, "valid": 0, "empty": 0},
        },
        "images_processed": set(),
        "unique_species": set(),
    }

    for ann in annotations:
        group_img = ann["group_img"]
        beetle_pos = ann["beetle_position"]
        individual_path = cfg.biorepo_dir / ann["rel_individual_img_path"]

        stats["images_processed"].add(group_img)
        if ann.get("scientific_name"):
            stats["unique_species"].add(ann["scientific_name"])

        # Check for missing taxon information
        if ann.get("taxon_id") is None or ann.get("scientific_name") is None:
            invalid_annotations.append({
                "group_img": group_img,
                "beetle_position": beetle_pos,
                "reason": "missing taxon_id or scientific_name",
            })
            stats["invalid_annotations"] += 1
            stats["invalid_beetles"].append(
                f"{group_img} beetle {beetle_pos}: missing taxon info"
            )
            continue

        # Check if individual image exists
        if not individual_path.exists():
            invalid_annotations.append({
                "group_img": group_img,
                "beetle_position": beetle_pos,
                "reason": "individual image not found",
                "path": str(individual_path),
            })
            stats["invalid_annotations"] += 1
            stats["invalid_beetles"].append(
                f"{group_img} beetle {beetle_pos}: image not found"
            )
            continue

        # Get individual image dimensions
        try:
            with Image.open(individual_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            invalid_annotations.append({
                "group_img": group_img,
                "beetle_position": beetle_pos,
                "reason": f"failed to open image: {e}",
            })
            stats["invalid_annotations"] += 1
            stats["invalid_beetles"].append(
                f"{group_img} beetle {beetle_pos}: failed to open image"
            )
            continue

        # Validate each measurement (except scalebar which is in group image coords)
        is_valid = True
        for measurement in ann["measurements"]:
            mtype = measurement["measurement_type"]
            polyline = measurement["polyline"]

            if mtype in stats["measurement_stats"]:
                stats["measurement_stats"][mtype]["total"] += 1

            # Check for empty polyline
            if not polyline:
                if mtype in stats["measurement_stats"]:
                    stats["measurement_stats"][mtype]["empty"] += 1
                stats["empty_polylines"].append(
                    f"{group_img} beetle {beetle_pos}: {mtype}"
                )
                continue

            # Check for polylines with more than 2 points
            if len(polyline) > 2:
                stats["polylines_with_more_than_2_points"].append({
                    "group_img": group_img,
                    "beetle_position": beetle_pos,
                    "measurement_type": mtype,
                    "num_points": len(polyline),
                })

            # Skip bounds checking for scalebar (it's in group image coordinates)
            if mtype == "scalebar":
                stats["measurement_stats"][mtype]["valid"] += 1
                continue

            # Check if all points are within image bounds (with 5px tolerance)
            tolerance = 5
            for pt in polyline:
                x, y = pt[0], pt[1]
                if (
                    x < -tolerance
                    or x > img_width + tolerance
                    or y < -tolerance
                    or y > img_height + tolerance
                ):
                    is_valid = False
                    if mtype in stats["measurement_stats"]:
                        stats["measurement_stats"][mtype]["out_of_bounds"] += 1
                    stats["invalid_beetles"].append(
                        f"{group_img} beetle {beetle_pos}: {mtype} out of bounds "
                        f"(point [{x:.1f}, {y:.1f}] outside {img_width}x{img_height} + {tolerance}px tolerance)"
                    )
                    break
            else:
                # All points valid
                if mtype in stats["measurement_stats"]:
                    stats["measurement_stats"][mtype]["valid"] += 1

        if is_valid:
            valid_annotations.append(ann)
            stats["valid_annotations"] += 1
        else:
            invalid_annotations.append({
                "group_img": group_img,
                "beetle_position": beetle_pos,
                "reason": "coordinates out of bounds",
            })
            stats["invalid_annotations"] += 1

    # Convert sets to lists for JSON serialization
    stats["images_processed"] = len(stats["images_processed"])
    stats["unique_species"] = list(stats["unique_species"])
    stats["num_unique_species"] = len(stats["unique_species"])
    stats["num_polylines_with_more_than_2_points"] = len(
        stats["polylines_with_more_than_2_points"]
    )
    stats["num_empty_polylines"] = len(stats["empty_polylines"])

    # Create output directory if it doesn't exist
    cfg.dump_to.mkdir(parents=True, exist_ok=True)

    # Save statistics
    stats_path = cfg.dump_to / "validation_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info("Saved validation statistics to %s", stats_path)

    valid_annotations_path = cfg.dump_to / "annotations.json"
    with valid_annotations_path.open("w", encoding="utf-8") as f:
        json.dump(valid_annotations, f, indent=2)
    logger.info(
        "Saved %d valid annotations to %s",
        len(valid_annotations),
        valid_annotations_path,
    )

    logger.info("VALIDATION SUMMARY")
    logger.info("Total annotations: %d", stats["total_annotations"])
    logger.info("Valid annotations: %d", stats["valid_annotations"])
    logger.info("Invalid annotations: %d", stats["invalid_annotations"])
    logger.info("Images processed: %d", stats["images_processed"])
    logger.info("Unique species: %d", stats["num_unique_species"])
    logger.info(
        "Polylines with >2 points: %d",
        stats["num_polylines_with_more_than_2_points"],
    )
    logger.info("Empty polylines: %d", stats["num_empty_polylines"])
    for mtype, mstats in stats["measurement_stats"].items():
        logger.info("Measurement %s: %s", mtype, mstats)

    if stats["invalid_beetles"]:
        logger.info("First 10 invalid beetles:")
        for beetle in stats["invalid_beetles"][:10]:
            logger.info("  - %s", beetle)

    return valid_annotations


@beartype.beartype
def main(cfg: Config) -> tuple[int, pl.DataFrame]:
    logging.basicConfig(level=logging.INFO, format=log_format)

    logger.info("=" * 80)
    logger.info("BIOREPO BEETLE POSITION INFERENCE AND TEMPLATE MATCHING")
    logger.info("=" * 80)

    # Get all group images
    group_images = sorted(cfg.images_dir.glob("*.png"))
    all_group_names = [img.name for img in group_images]
    logger.info("Found %d group images in %s", len(group_images), cfg.images_dir)

    if not all_group_names:
        logger.error("No group images found!")
        return 1, pl.DataFrame()

    # Batch group images into chunks for each job
    group_batches = list(
        btx.helpers.batched_idx(len(all_group_names), cfg.groups_per_job)
    )
    logger.info(
        "Will process %d group images in %d jobs (%d groups per job)",
        len(all_group_names),
        len(group_batches),
        cfg.groups_per_job,
    )

    # Set up executor based on whether we're using Slurm
    if cfg.slurm_acct:
        # Calculate safe limits for Slurm
        max_array_size = btx.helpers.get_slurm_max_array_size()
        max_submit_jobs = btx.helpers.get_slurm_max_submit_jobs()

        safe_array_size = min(int(max_array_size * 0.95), max_array_size - 2)
        safe_array_size = max(1, safe_array_size)

        safe_submit_jobs = min(int(max_submit_jobs * 0.95), max_submit_jobs - 2)
        safe_submit_jobs = max(1, safe_submit_jobs)

        logger.info(
            "Using Slurm with safe limits - Array size: %d (max: %d), Submit jobs: %d (max: %d)",
            safe_array_size,
            max_array_size,
            safe_submit_jobs,
            max_submit_jobs,
        )

        executor = submitit.SlurmExecutor(folder=cfg.log_to)
        executor.update_parameters(
            time=int(cfg.n_hours * 60),  # Convert hours to minutes
            partition=cfg.slurm_partition,
            gpus_per_node=0,
            ntasks_per_node=1,
            cpus_per_task=8,
            mem_per_cpu="10gb",
            stderr_to_stdout=True,
            account=cfg.slurm_acct,
            array_parallelism=safe_array_size,
        )
    else:
        logger.info("Using DebugExecutor for local execution")
        executor = submitit.DebugExecutor(folder=cfg.log_to)
        safe_array_size = len(group_batches)
        safe_submit_jobs = len(group_batches)

    # Submit jobs in batches to respect Slurm limits
    all_jobs = []
    job_batches = list(btx.helpers.batched_idx(len(group_batches), safe_array_size))

    for batch_idx, (start, end) in enumerate(job_batches):
        current_batches = group_batches[start:end]

        # Check current job count and wait if needed (only for Slurm)
        if cfg.slurm_acct:
            current_jobs = btx.helpers.get_slurm_job_count()
            jobs_available = max(0, safe_submit_jobs - current_jobs)

            while jobs_available < len(current_batches):
                logger.info(
                    "Can only submit %d jobs but need %d. Waiting for jobs to complete...",
                    jobs_available,
                    len(current_batches),
                )
                time.sleep(60)  # Wait 1 minute
                current_jobs = btx.helpers.get_slurm_job_count()
                jobs_available = max(0, safe_submit_jobs - current_jobs)

        logger.info(
            "Submitting job batch %d/%d: jobs %d-%d",
            batch_idx + 1,
            len(job_batches),
            start,
            end - 1,
        )

        # Submit jobs for this batch
        with executor.batch():
            for group_start, group_end in current_batches:
                group_batch = all_group_names[group_start:group_end]
                job = executor.submit(worker_fn, cfg, group_batch)
                all_jobs.append(job)

        logger.info("Submitted job batch %d/%d", batch_idx + 1, len(job_batches))

    logger.info("Submitted %d total jobs. Waiting for results...", len(all_jobs))

    # Collect results from all jobs
    all_rename_operations = []
    all_match_dfs = []
    successful_jobs = 0
    failed_jobs = 0

    for job_idx, job in enumerate(all_jobs):
        try:
            rename_ops, match_dfs = job.result()
            all_rename_operations.extend(rename_ops)
            all_match_dfs.extend(match_dfs)
            successful_jobs += 1
            logger.info("Job %d/%d completed", job_idx + 1, len(all_jobs))
        except Exception as e:
            logger.error("Job %d/%d failed: %s", job_idx + 1, len(all_jobs), e)
            failed_jobs += 1

    # Summary of parallel processing
    logger.info("=" * 80)
    logger.info("PARALLEL PROCESSING COMPLETE")
    logger.info("=" * 80)
    logger.info("Jobs succeeded: %d", successful_jobs)
    logger.info("Jobs failed: %d", failed_jobs)
    logger.info("Total rename operations: %d", len(all_rename_operations))
    logger.info("Total match dataframes: %d", len(all_match_dfs))

    # Combine all match dataframes
    combined_df = pl.concat(all_match_dfs) if all_match_dfs else pl.DataFrame()

    # Execute renames sequentially (must happen after all parallel jobs complete)
    logger.info("=" * 80)
    logger.info("EXECUTING FILE RENAMES")
    logger.info("=" * 80)

    successful, failed, completed_ops = execute_renames(all_rename_operations)

    logger.info("Rename results:")
    logger.info("  Successful: %d", successful)
    logger.info("  Failed: %d", failed)

    # Update dataframe with new filenames after successful renames
    if completed_ops and not combined_df.is_empty():
        logger.info("Updating match dataframe with new filenames...")

        # Create mapping of old filename to new filename
        filename_map = {op.old_path.name: op.new_path.name for op in completed_ops}

        # Update the individual_filename column
        combined_df = combined_df.with_columns(
            pl.col("individual_filename").replace(
                filename_map, default=pl.col("individual_filename")
            )
        )
        logger.info("Updated %d filenames in dataframe", len(completed_ops))

    logger.info("Files have been renamed!")

    # Open beetle metadata csv (contains scientific name)
    beetle_metadata_df = pl.read_csv(cfg.metadata_csv)
    annotations = create_measurements_annotations(cfg, combined_df, beetle_metadata_df)

    # Validate annotations and save statistics
    valid_annotations = validate_annotations(cfg, annotations)

    logger.info("=" * 80)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 80)
    logger.info("Beetles with valid annotations: %d", len(valid_annotations))

    return 0, combined_df


if __name__ == "__main__":
    try:
        exit_code, match_df = main(tyro.cli(Config))
        raise SystemExit(exit_code)
    except KeyboardInterrupt:
        logger.info("Interrupted.")
        raise SystemExit(130)
