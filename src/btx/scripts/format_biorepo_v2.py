"""Append new TORAS annotations to an existing annotations file.

This script processes new beetle group images (not yet annotated) by:
1. Loading existing annotations from a previous run (optional)
2. Determining which group images have TORAS annotations not yet in the formatted data
3. Template matching and renaming individual beetles for the new group images only
4. Creating annotations for the new group images from TORAS data
5. Validating and appending new annotations to the existing file

INPUTS:
- New group image names (optional, via --new-group-images; auto-detected from TORAS if omitted)
- Existing annotations.json (optional, from a previous run)
- TORAS annotations (JSON)
- Beetle metadata (CSV)
- Individual beetle images in per-group subdirectories under --individuals-dpath

OUTPUTS:
- annotations.json: Combined existing + new valid annotations
- validation_stats.json: Statistics for the newly added annotations

USAGE:
------
Auto-detect new images from TORAS (compares against existing annotations):
  python -m btx.scripts.format_biorepo_v2 \\
    --existing-annotations-fpath ./data/biorepo-formatted/annotations.json

Explicit image list (first run, no existing annotations):
  python -m btx.scripts.format_biorepo_v2 --new-group-images img1.png img2.png

Append new images to existing annotations:
  python -m btx.scripts.format_biorepo_v2 \\
    --new-group-images img3.png img4.png \\
    --existing-annotations-fpath ./data/biorepo-formatted/annotations.json

With Slurm:
  python -m btx.scripts.format_biorepo_v2 \\
    --slurm-acct=YOUR_ACCOUNT --slurm-partition=parallel
"""

import dataclasses
import gc
import json
import logging
import pathlib
import re
import time
import traceback

import beartype
import polars as pl
import submitit
import tyro
from PIL import Image, ImageDraw, ImageFont

import btx.helpers
from btx.scripts.format_biorepo import (
    BeetleMatch,
    RenameOperation,
    execute_renames,
    infer_positions,
    log_format,
    offset_polyline,
    template_match_beetles,
)

logger = logging.getLogger("format-biorepo-v2")


@beartype.beartype
def clean_toras_path(toras_path: str) -> str:
    """Convert a TORAS image path to a clean group image filename.

    Strips leading slash, removes Windows duplicate-file suffixes like ' (1)',
    and converts .jpg to .png.
    """
    # Take only the filename, discarding any leading directory components (e.g. "round1_groups/").
    path = pathlib.Path(toras_path).name
    # Remove " (N)" before the extension, e.g. "img (1).jpg" -> "img.jpg"
    path = re.sub(r" \(\d+\)(?=\.)", "", path)
    return path.replace(".jpg", ".png")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    group_images_dpath: pathlib.Path = pathlib.Path(
        "/fs/ess/PAS2136/CarabidImaging/Output/plotted_trays"
    )
    """Directory containing group images (PNG)."""

    individuals_dpath: pathlib.Path = pathlib.Path(
        "/fs/ess/PAS2136/CarabidImaging/Output/cropped_images"
    )
    """Directory containing per-group subdirectories of individual beetle images."""

    dump_to: pathlib.Path = pathlib.Path("./data/biorepo-formatted")
    """Where to save formatted data."""

    toras_anns: pathlib.Path = pathlib.Path(
        "/fs/scratch/PAS2136/cain429/coco_anns.json"
    )
    """Path to TORAS annotations JSON file."""

    metadata_csv: pathlib.Path = pathlib.Path(
        "/fs/ess/PAS2136/CarabidImaging/allIndividuals.csv"
    )
    """Beetle metadata CSV file."""

    new_group_images: list[str] = dataclasses.field(default_factory=list)
    """List of new group image filenames to process (e.g. img1.png img2.png)."""

    existing_annotations_fpath: pathlib.Path | None = None
    """Path to existing annotations.json to append to. If None, starts fresh."""

    viz_output_dpath: pathlib.Path = pathlib.Path(
        "/fs/scratch/PAS2136/cain429/biorepo-formatted/template_match"
    )
    """Where to save template matching visualizations."""

    row_center_tolerance_ratio: float = 50 / 221.4
    """Ratio of tolerance to average beetle height for row grouping."""

    create_template_match_images: bool = True
    """Whether to create template matching visualization images."""

    # Slurm configuration
    slurm_acct: str = ""
    """Slurm account to use. If empty, uses DebugExecutor (local execution)."""

    slurm_partition: str = "nextgen"
    """Slurm partition to use."""

    log_to: pathlib.Path = pathlib.Path("./logs")
    """Where to save submitit/slurm logs."""

    n_hours: float = 1.0
    """Number of hours to request for each job."""

    groups_per_job: int = 8
    """Number of group images to process per job."""

    # Visualization settings
    bbox_color: tuple[int, int, int] = (255, 0, 0)
    bbox_width: int = 4
    text_color: tuple[int, int, int] = (255, 255, 255)


@beartype.beartype
def visualize_results(
    group_image_path: pathlib.Path,
    position_map: dict[int, BeetleMatch],
    output_path: pathlib.Path,
    cfg: Config,
):
    """Draw bounding boxes and position labels on the group image."""
    img = Image.open(group_image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except OSError:
        font = ImageFont.load_default()

    for inferred_pos, match in position_map.items():
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
def process_single_image(
    group_image_name: str,
    cfg: Config,
) -> tuple[list[RenameOperation], pl.DataFrame]:
    """Process a single group image: template match, infer positions, prepare renames."""
    logger.info("Processing group image: %s", group_image_name)

    group_image_path = cfg.group_images_dpath / group_image_name
    if not group_image_path.exists():
        logger.error("Group image not found at %s", group_image_path)
        return [], pl.DataFrame()

    group_stem = group_image_path.stem
    individual_dir = cfg.individuals_dpath / group_stem

    if not individual_dir.exists():
        logger.warning("No individual directory found at %s", individual_dir)
        return [], pl.DataFrame()

    individual_paths = sorted(individual_dir.glob(f"{group_stem}_*.png"))
    logger.info("Found %d individual beetles", len(individual_paths))

    if not individual_paths:
        logger.warning("No individual beetles found, skipping %s", group_image_name)
        return [], pl.DataFrame()

    matches = template_match_beetles(group_image_path, individual_paths)

    if not matches:
        logger.error("No successful template matches for %s", group_image_name)
        return [], pl.DataFrame()

    avg_height = sum(m.height for m in matches) / len(matches)
    tolerance_ratio = (
        1.5 if "ctray" in group_image_name.lower() else cfg.row_center_tolerance_ratio
    )
    row_center_tolerance = avg_height * tolerance_ratio
    position_map = infer_positions(matches, row_center_tolerance)

    if cfg.create_template_match_images:
        output_path = cfg.viz_output_dpath / f"{group_stem}_annotated.png"
        visualize_results(group_image_path, position_map, output_path, cfg)

    rename_operations = []
    for inferred_pos, match in sorted(position_map.items()):
        if inferred_pos != match.filename_position:
            rename_operations.append(
                RenameOperation(
                    old_path=match.individual_path,
                    new_path=individual_dir / f"{group_stem}_{inferred_pos}.png",
                    group_image=group_image_name,
                    old_position=match.filename_position,
                    new_position=inferred_pos,
                )
            )

    match_records = [
        {
            "group_image": group_image_name,
            "inferred_position": inferred_pos,
            "offset_x": match.x,
            "offset_y": match.y,
        }
        for inferred_pos, match in sorted(position_map.items())
    ]

    logger.info(
        "Summary | image=%s beetles=%d rename_ops=%d",
        group_image_name,
        len(matches),
        len(rename_operations),
    )
    return rename_operations, pl.DataFrame(match_records)


@beartype.beartype
def worker_fn(
    cfg: Config,
    group_image_names: list[str],
) -> tuple[list[RenameOperation], list[pl.DataFrame]]:
    """Worker function for parallel processing of group images."""
    logging.basicConfig(level=logging.INFO, format=log_format)
    worker_logger = logging.getLogger("format-biorepo-v2.worker")

    all_rename_ops: list[RenameOperation] = []
    all_match_dfs: list[pl.DataFrame] = []

    for idx, name in enumerate(group_image_names):
        worker_logger.info(
            "Processing %d/%d: %s", idx + 1, len(group_image_names), name
        )
        try:
            rename_ops, match_df = process_single_image(name, cfg)
            all_rename_ops.extend(rename_ops)
            if not match_df.is_empty():
                all_match_dfs.append(match_df)
        except Exception as e:
            worker_logger.error("Error processing %s: %s", name, e)
            traceback.print_exc()
        gc.collect()

    return all_rename_ops, all_match_dfs


@beartype.beartype
def create_new_measurements_annotations(
    cfg: Config,
    template_match_df: pl.DataFrame,
    beetle_metadata_df: pl.DataFrame,
    new_group_images: set[str],
) -> list[dict]:
    """Create measurement annotations for the new group images only."""
    with cfg.toras_anns.open("r", encoding="utf-8") as fd:
        toras_data = json.load(fd)

    toras_measurements_df = pl.DataFrame(toras_data["annotations"], strict=False)
    toras_measurements_df = toras_measurements_df.with_columns(
        pl.col("name").str.replace("Entity ", "").cast(pl.Int64).alias("Entity")
    )

    measurements_per_beetle = []

    for i, img_meta in enumerate(toras_data["images"]):
        group_img_name = clean_toras_path(img_meta["toras_path"])
        if group_img_name not in new_group_images:
            continue

        image_df = toras_measurements_df.filter(pl.col("image_id") == i)
        if image_df.is_empty():
            logger.warning(
                "No TORAS annotations for %s (image_id=%d)", group_img_name, i
            )
            continue

        scalebar_row = image_df.filter(pl.col("Entity") == pl.col("Entity").max())[0]

        for j in range(int(scalebar_row["Entity"].item() / 3)):
            entity_num = j * 3 + 1
            beetle_num = j + 1

            match_row = template_match_df.filter(
                (pl.col("group_image") == group_img_name)
                & (pl.col("inferred_position") == beetle_num)
            )
            if match_row.is_empty():
                logger.warning("No match for %s beetle %d", group_img_name, beetle_num)
                continue

            offset_x = match_row["offset_x"].item()
            offset_y = match_row["offset_y"].item()

            metadata_row = beetle_metadata_df.filter(
                (pl.col("imageID") == group_img_name) & (pl.col("Order") == beetle_num)
            )
            if metadata_row.is_empty():
                logger.warning(
                    "No metadata for %s beetle %d", group_img_name, beetle_num
                )
                taxon_id = None
                scientific_name = None
                individual_id = None
            else:
                taxon_id = metadata_row["taxonID"][0]
                scientific_name = metadata_row["scientificName"][0]
                individual_id = metadata_row["individualID"][0]

            length_df = image_df.filter(pl.col("Entity") == entity_num)
            width_df = image_df.filter(pl.col("Entity") == entity_num + 1)
            pronotum_df = image_df.filter(pl.col("Entity") == entity_num + 2)
            if length_df.is_empty() or width_df.is_empty() or pronotum_df.is_empty():
                logger.warning(
                    "Missing measurements for %s beetle %d, skipping",
                    group_img_name,
                    beetle_num,
                )
                continue
            length_row = length_df[0]
            width_row = width_df[0]
            pronotum_row = pronotum_df[0]

            group_stem = group_img_name.replace(".png", "")
            individual_filename = f"{group_stem}_{beetle_num}.png"
            abs_individual_img_path = str(
                (cfg.individuals_dpath / group_stem / individual_filename).resolve()
            )

            measurements_per_beetle.append({
                "beetle_position": beetle_num,
                "group_img": group_img_name,
                "rel_group_img_path": str(cfg.group_images_dpath / group_img_name),
                "abs_group_img_path": str(
                    (cfg.group_images_dpath / group_img_name).resolve()
                ),
                "rel_individual_img_path": str(
                    cfg.individuals_dpath / group_stem / individual_filename
                ),
                "abs_individual_img_path": abs_individual_img_path,
                "offset_x": offset_x,
                "offset_y": offset_y,
                "taxon_id": taxon_id,
                "individual_id": individual_id,
                "scientific_name": scientific_name,
                "measurements": [
                    {
                        "measurement_type": "elytra_length",
                        "polyline": offset_polyline(
                            length_row["polyline"].to_list(), offset_x, offset_y
                        ),
                    },
                    {
                        "measurement_type": "elytra_width",
                        "polyline": offset_polyline(
                            width_row["polyline"].to_list(), offset_x, offset_y
                        ),
                    },
                    {
                        "measurement_type": "pronotum_width",
                        "polyline": offset_polyline(
                            pronotum_row["polyline"].to_list(), offset_x, offset_y
                        ),
                    },
                    {
                        "measurement_type": "scalebar",
                        "polyline": offset_polyline(
                            scalebar_row["polyline"].to_list(), 0.0, 0.0
                        ),
                    },
                ],
            })

    logger.info("Created %d new beetle annotations", len(measurements_per_beetle))
    return measurements_per_beetle


@beartype.beartype
def validate_and_append_annotations(
    cfg: Config,
    new_annotations: list[dict],
    existing_annotations: list[dict],
) -> list[dict]:
    """Validate new annotations and append valid ones to existing, then save."""
    valid_new = []
    stats: dict = {
        "total_new": len(new_annotations),
        "valid_new": 0,
        "invalid_new": 0,
        "invalid_beetles": [],
    }

    for ann in new_annotations:
        group_img = ann["group_img"]
        beetle_pos = ann["beetle_position"]
        individual_path = pathlib.Path(ann["abs_individual_img_path"])

        if ann.get("taxon_id") is None or ann.get("scientific_name") is None:
            stats["invalid_new"] += 1
            stats["invalid_beetles"].append(
                f"{group_img} beetle {beetle_pos}: missing taxon info"
            )
            continue

        if not individual_path.exists():
            stats["invalid_new"] += 1
            stats["invalid_beetles"].append(
                f"{group_img} beetle {beetle_pos}: image not found at {individual_path}"
            )
            continue

        try:
            with Image.open(individual_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            stats["invalid_new"] += 1
            stats["invalid_beetles"].append(
                f"{group_img} beetle {beetle_pos}: failed to open image: {e}"
            )
            continue

        is_valid = True
        tolerance = 5
        for measurement in ann["measurements"]:
            mtype = measurement["measurement_type"]
            polyline = measurement["polyline"]
            if not polyline or mtype == "scalebar":
                continue
            for pt in polyline:
                x, y = pt[0], pt[1]
                if (
                    x < -tolerance
                    or x > img_width + tolerance
                    or y < -tolerance
                    or y > img_height + tolerance
                ):
                    is_valid = False
                    stats["invalid_beetles"].append(
                        f"{group_img} beetle {beetle_pos}: {mtype} out of bounds "
                        f"(point [{x:.1f}, {y:.1f}] outside {img_width}x{img_height} + {tolerance}px tolerance)"
                    )
                    break
            if not is_valid:
                break

        if is_valid:
            valid_new.append(ann)
            stats["valid_new"] += 1
        else:
            stats["invalid_new"] += 1

    combined = existing_annotations + valid_new

    cfg.dump_to.mkdir(parents=True, exist_ok=True)

    stats_fpath = cfg.dump_to / "validation_stats.json"
    with stats_fpath.open("w", encoding="utf-8") as fd:
        json.dump(stats, fd, indent=2)
    logger.info("Saved validation stats to %s", stats_fpath)

    annotations_fpath = cfg.dump_to / "annotations.json"
    with annotations_fpath.open("w", encoding="utf-8") as fd:
        json.dump(combined, fd, indent=2)
    logger.info(
        "Saved %d total annotations (%d existing + %d new) to %s",
        len(combined),
        len(existing_annotations),
        len(valid_new),
        annotations_fpath,
    )

    logger.info(
        "VALIDATION SUMMARY | total_new=%d valid=%d invalid=%d",
        stats["total_new"],
        stats["valid_new"],
        stats["invalid_new"],
    )
    if stats["invalid_beetles"]:
        logger.info("First 10 invalid:")
        for b in stats["invalid_beetles"][:10]:
            logger.info("  - %s", b)

    return combined


@beartype.beartype
def main(cfg: Config) -> int:
    logging.basicConfig(level=logging.INFO, format=log_format)

    logger.info("=" * 80)
    logger.info("BIOREPO FORMAT V2: APPENDING NEW ANNOTATIONS")
    logger.info("=" * 80)

    # Load existing annotations
    existing_annotations: list[dict] = []
    if cfg.existing_annotations_fpath is not None:
        assert cfg.existing_annotations_fpath.exists(), (
            f"Existing annotations not found: {cfg.existing_annotations_fpath}"
        )
        with cfg.existing_annotations_fpath.open(encoding="utf-8") as fd:
            existing_annotations = json.load(fd)
        logger.info("Loaded %d existing annotations", len(existing_annotations))

    existing_group_images = {ann["group_img"] for ann in existing_annotations}

    # Determine which images are truly new
    if cfg.new_group_images:
        already_done = [
            img for img in cfg.new_group_images if img in existing_group_images
        ]
        new_group_images = [
            img for img in cfg.new_group_images if img not in existing_group_images
        ]
        if already_done:
            logger.warning(
                "Skipping %d images already in existing annotations: %s",
                len(already_done),
                already_done,
            )
    else:
        # Auto-detect: find all group images in TORAS not yet in existing annotations
        assert cfg.toras_anns.exists(), f"TORAS annotations not found: {cfg.toras_anns}"
        with cfg.toras_anns.open("r", encoding="utf-8") as fd:
            toras_data = json.load(fd)
        all_toras_images = [
            clean_toras_path(img["toras_path"]) for img in toras_data["images"]
        ]
        new_group_images = [
            img for img in all_toras_images if img not in existing_group_images
        ]
        logger.info(
            "Auto-detected %d new group images from TORAS (out of %d total)",
            len(new_group_images),
            len(all_toras_images),
        )

    logger.info("Processing %d new group images", len(new_group_images))

    if not new_group_images:
        logger.info("No new group images to process. Exiting.")
        return 0

    # Batch and submit jobs
    group_batches = list(
        btx.helpers.batched_idx(len(new_group_images), cfg.groups_per_job)
    )
    logger.info(
        "Processing %d images in %d jobs", len(new_group_images), len(group_batches)
    )

    if cfg.slurm_acct:
        max_array_size = btx.helpers.get_slurm_max_array_size()
        max_submit_jobs = btx.helpers.get_slurm_max_submit_jobs()
        safe_array_size = max(1, min(int(max_array_size * 0.95), max_array_size - 2))
        safe_submit_jobs = max(1, min(int(max_submit_jobs * 0.95), max_submit_jobs - 2))
        executor = submitit.SlurmExecutor(folder=cfg.log_to)
        executor.update_parameters(
            time=int(cfg.n_hours * 60),
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

    all_jobs = []
    job_batches = list(btx.helpers.batched_idx(len(group_batches), safe_array_size))

    for batch_idx, (start, end) in enumerate(job_batches):
        current_batches = group_batches[start:end]

        if cfg.slurm_acct:
            current_jobs = btx.helpers.get_slurm_job_count()
            jobs_available = max(0, safe_submit_jobs - current_jobs)
            while jobs_available < len(current_batches):
                logger.info(
                    "Waiting for Slurm jobs to free up (%d/%d available)...",
                    jobs_available,
                    len(current_batches),
                )
                time.sleep(60)
                current_jobs = btx.helpers.get_slurm_job_count()
                jobs_available = max(0, safe_submit_jobs - current_jobs)

        logger.info("Submitting job batch %d/%d", batch_idx + 1, len(job_batches))
        with executor.batch():
            for group_start, group_end in current_batches:
                group_batch = new_group_images[group_start:group_end]
                all_jobs.append(executor.submit(worker_fn, cfg, group_batch))

    logger.info("Submitted %d jobs. Waiting for results...", len(all_jobs))

    all_rename_ops: list[RenameOperation] = []
    all_match_dfs: list[pl.DataFrame] = []
    for job_idx, job in enumerate(all_jobs):
        try:
            rename_ops, match_dfs = job.result()
            all_rename_ops.extend(rename_ops)
            all_match_dfs.extend(match_dfs)
            logger.info("Job %d/%d completed", job_idx + 1, len(all_jobs))
        except Exception as e:
            logger.error("Job %d/%d failed: %s", job_idx + 1, len(all_jobs), e)

    assert all_match_dfs, (
        f"No template match results collected for {len(new_group_images)} new images. Check job logs for errors (missing group images, individual directories, or failed template matching)."
    )
    combined_df = pl.concat(all_match_dfs)

    successful, failed, _ = execute_renames(all_rename_ops)
    logger.info("Renames: %d successful, %d failed", successful, failed)

    beetle_metadata_df = pl.read_csv(
        cfg.metadata_csv, null_values=["NA", "N/A", "", "-Inf"]
    )
    new_annotations = create_new_measurements_annotations(
        cfg, combined_df, beetle_metadata_df, set(new_group_images)
    )

    validate_and_append_annotations(cfg, new_annotations, existing_annotations)

    logger.info("=" * 80)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(tyro.cli(Config)))
    except KeyboardInterrupt:
        logger.info("Interrupted.")
        raise SystemExit(130)
