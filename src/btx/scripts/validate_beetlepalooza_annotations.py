"""
Validate that elytra measurements are within the bounding box of individual beetle images.

This script reads annotations.json and checks if the measurement coordinates
(elytra_length, elytra_width) fall within the bounds of the individual image.
"""

import json
import logging
import pathlib
import dataclasses
import typing as tp

import beartype
import numpy as np
import tyro
from PIL import Image, ImageDraw

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    annotations_file: pathlib.Path = pathlib.Path(
        "./data/beetlepalooza-formatted/annotations.json"
    )
    """Path to the annotations.json file."""

    hf_root: pathlib.Path = pathlib.Path("./data/beetlepalooza/individual_specimens")
    """Path to individual specimens directory (to get image dimensions)."""

    resized_root: pathlib.Path = pathlib.Path("./data/beetlepalooza/group_images_resized")
    """Path to group images."""

    output_dir: pathlib.Path = pathlib.Path("./data/beetlepalooza-formatted/validation-examples")
    """Where to save visualization images."""

    visualize_sample_rate: float = 0.1
    """Fraction of invalid measurements to visualize (0.1 = 10%)."""

    seed: int = 42
    """Random seed for sampling which invalid measurements to visualize."""

    delete_invalid: bool = True
    """If True, remove invalid annotations from annotations.json."""

    save_invalid_separately: bool = True
    """If True, save invalid annotations to a separate file."""


@beartype.beartype
@dataclasses.dataclass
class ValidationResult:
    """Results from validating one annotation."""

    individual_id: str
    group_img_basename: str
    beetle_position: int
    image_width: int
    image_height: int
    measurements_checked: int
    measurements_valid: int
    measurements_invalid: int
    invalid_details: list[str] = dataclasses.field(default_factory=list)
    # Store the full annotation for visualization
    annotation: dict = dataclasses.field(default_factory=dict)


@beartype.beartype
def validate_measurement_coords(
    coords: dict[str, float],
    img_width: int,
    img_height: int,
    measurement_type: str,
) -> tuple[bool, str]:
    """
    Check if measurement coordinates are within image bounds.

    Returns (is_valid, error_message)
    """
    x1, y1 = coords["x1"], coords["y1"]
    x2, y2 = coords["x2"], coords["y2"]

    errors = []

    if x1 < 0 or x1 > img_width:
        errors.append(f"x1={x1:.1f} out of bounds [0, {img_width}]")
    if x2 < 0 or x2 > img_width:
        errors.append(f"x2={x2:.1f} out of bounds [0, {img_width}]")
    if y1 < 0 or y1 > img_height:
        errors.append(f"y1={y1:.1f} out of bounds [0, {img_height}]")
    if y2 < 0 or y2 > img_height:
        errors.append(f"y2={y2:.1f} out of bounds [0, {img_height}]")

    if errors:
        return False, f"{measurement_type}: " + ", ".join(errors)
    return True, ""


@beartype.beartype
def validate_annotation(
    annotation: dict, hf_root: pathlib.Path, logger: logging.Logger
) -> ValidationResult | None:
    """Validate measurements for a single annotation."""
    individual_id = annotation.get("individual_id", "unknown")
    group_img_basename = annotation.get("group_img_basename", "unknown")
    beetle_position = annotation.get("beetle_position", -1)

    # Get the individual image path (use absolute path which includes part_000/part_001 etc.)
    indiv_img_abs_path = annotation.get("indiv_img_abs_path", "")
    if not indiv_img_abs_path:
        logger.warning("No indiv_img_abs_path for %s", individual_id)
        return None

    indiv_img_path = pathlib.Path(indiv_img_abs_path)

    # Get image dimensions
    if not indiv_img_path.exists():
        logger.warning("Image not found: %s", indiv_img_path)
        return None

    try:
        with Image.open(indiv_img_path) as img:
            img_width, img_height = img.size
    except Exception as e:
        logger.warning("Failed to open image %s: %s", indiv_img_path, e)
        return None

    # Check measurements
    measurements = annotation.get("measurements", [])
    measurements_checked = 0
    measurements_valid = 0
    measurements_invalid = 0
    invalid_details = []

    for measurement in measurements:
        measurement_type = measurement.get("measurement_type", "")

        # Only check elytra measurements
        if not measurement_type.startswith("elytra_"):
            continue

        coords = measurement.get("coords_px", {})
        if not coords or "x1" not in coords:
            continue

        measurements_checked += 1

        is_valid, error_msg = validate_measurement_coords(
            coords, img_width, img_height, measurement_type
        )

        if is_valid:
            measurements_valid += 1
        else:
            measurements_invalid += 1
            invalid_details.append(error_msg)

    return ValidationResult(
        individual_id=individual_id,
        group_img_basename=group_img_basename,
        beetle_position=beetle_position,
        image_width=img_width,
        image_height=img_height,
        measurements_checked=measurements_checked,
        measurements_valid=measurements_valid,
        measurements_invalid=measurements_invalid,
        invalid_details=invalid_details,
        annotation=annotation,
    )


@beartype.beartype
def visualize_invalid_measurement(
    result: ValidationResult,
    cfg: Config,
    logger: logging.Logger,
) -> None:
    """Draw the group image with bounding box and invalid measurements."""
    annotation = result.annotation

    # Load group image
    group_img_basename = result.group_img_basename
    group_img_path = cfg.resized_root / group_img_basename

    if not group_img_path.exists():
        logger.warning("Group image not found: %s", group_img_path)
        return

    try:
        group_img = Image.open(group_img_path).convert("RGB")
    except Exception as e:
        logger.warning("Failed to open group image %s: %s", group_img_path, e)
        return

    draw = ImageDraw.Draw(group_img)

    # Get bounding box coordinates
    origin_x = annotation.get("origin_x", 0)
    origin_y = annotation.get("origin_y", 0)
    bbox_width = result.image_width
    bbox_height = result.image_height

    # Draw bounding box in red
    draw.rectangle(
        [origin_x, origin_y, origin_x + bbox_width, origin_y + bbox_height],
        outline=(255, 0, 0),
        width=8,
    )

    # Draw measurements
    measurements = annotation.get("measurements", [])
    for measurement in measurements:
        measurement_type = measurement.get("measurement_type", "")

        # Only draw elytra measurements
        if not measurement_type.startswith("elytra_"):
            continue

        coords = measurement.get("coords_px", {})
        if not coords or "x1" not in coords:
            continue

        # Convert from individual image coordinates to group image coordinates
        x1 = coords["x1"] + origin_x
        y1 = coords["y1"] + origin_y
        x2 = coords["x2"] + origin_x
        y2 = coords["y2"] + origin_y

        # Check if this measurement is valid or invalid
        is_valid, _ = validate_measurement_coords(
            coords, bbox_width, bbox_height, measurement_type
        )

        # Use green for valid, yellow for invalid
        color = (0, 255, 0) if is_valid else (255, 255, 0)

        # Draw the measurement line
        draw.line([(x1, y1), (x2, y2)], fill=color, width=6)

        # Draw circles at endpoints for invalid measurements
        if not is_valid:
            radius = 10
            draw.ellipse([x1-radius, y1-radius, x1+radius, y1+radius], fill=(255, 255, 0))
            draw.ellipse([x2-radius, y2-radius, x2+radius, y2+radius], fill=(255, 255, 0))

    # Resize for easier viewing
    group_w, group_h = group_img.size
    resized_group = group_img.resize((group_w // 10, group_h // 10))

    # Save visualization
    output_path = cfg.output_dir / f"{result.individual_id}_validation.png"
    resized_group.save(output_path)
    logger.info("Saved visualization to %s", output_path)


@beartype.beartype
def main(cfg: Config) -> int:
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger("validate-bp")

    # Load annotations
    logger.info("Loading annotations from %s", cfg.annotations_file)
    if not cfg.annotations_file.exists():
        logger.error("Annotations file not found: %s", cfg.annotations_file)
        return 1

    with open(cfg.annotations_file, "r") as f:
        annotations = json.load(f)

    logger.info("Loaded %d annotations", len(annotations))

    # Validate each annotation
    results = []
    for idx, annotation in enumerate(annotations):
        if (idx + 1) % 1000 == 0:
            logger.info("Validated %d/%d annotations...", idx + 1, len(annotations))

        result = validate_annotation(annotation, cfg.hf_root, logger)
        if result:
            results.append(result)

    # Calculate totals
    total_measurements = sum(r.measurements_checked for r in results)
    total_valid = sum(r.measurements_valid for r in results)
    total_invalid = sum(r.measurements_invalid for r in results)

    # Report results
    logger.info("=" * 60)
    logger.info("VALIDATION RESULTS")
    logger.info("=" * 60)
    logger.info("Annotations validated: %d", len(results))
    logger.info("Total elytra measurements checked: %d", total_measurements)
    logger.info("Measurements within bounds: %d", total_valid)
    logger.info("Measurements out of bounds: %d", total_invalid)

    if total_measurements > 0:
        percent_valid = (total_valid / total_measurements) * 100
        logger.info("Accuracy: %.2f%%", percent_valid)

    # Show examples of invalid measurements
    if total_invalid > 0:
        logger.info("\nExamples of out-of-bounds measurements:")
        count = 0
        for result in results:
            if result.measurements_invalid > 0:
                logger.info(
                    "  %s (beetle %d, image size: %dx%d):",
                    result.individual_id,
                    result.beetle_position,
                    result.image_width,
                    result.image_height,
                )
                for detail in result.invalid_details:
                    logger.info("    - %s", detail)
                count += 1
                if count >= 10:
                    remaining = sum(
                        1 for r in results if r.measurements_invalid > 0
                    ) - count
                    if remaining > 0:
                        logger.info("  ... and %d more annotations with errors", remaining)
                    break

    # Visualize a sample of invalid measurements
    if total_invalid > 0:
        # Create output directory
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        # Get all results with invalid measurements
        invalid_results = [r for r in results if r.measurements_invalid > 0]

        # Sample 10% (or at least 1 if there are any invalid)
        num_to_visualize = max(1, int(len(invalid_results) * cfg.visualize_sample_rate))

        logger.info("\nVisualizing %d/%d annotations with invalid measurements...",
                   num_to_visualize, len(invalid_results))

        # Use numpy for reproducible random sampling
        rng = np.random.default_rng(seed=cfg.seed)
        indices = rng.choice(len(invalid_results), size=num_to_visualize, replace=False)

        for idx in indices:
            result = invalid_results[idx]
            visualize_invalid_measurement(result, cfg, logger)

        logger.info("Saved visualizations to %s", cfg.output_dir)

    # Save detailed results to file
    output_file = cfg.annotations_file.parent / "validation_results.json"
    detailed_results = [
        {
            "individual_id": r.individual_id,
            "group_img_basename": r.group_img_basename,
            "beetle_position": r.beetle_position,
            "image_dimensions": [r.image_width, r.image_height],
            "measurements_checked": r.measurements_checked,
            "measurements_valid": r.measurements_valid,
            "measurements_invalid": r.measurements_invalid,
            "errors": r.invalid_details,
        }
        for r in results
        if r.measurements_invalid > 0  # Only save entries with errors
    ]

    with open(output_file, "w") as f:
        json.dump(detailed_results, f, indent=2)

    logger.info("\nDetailed results saved to %s", output_file)

    # Remove invalid measurements from annotations if requested
    if cfg.delete_invalid and total_invalid > 0:
        logger.info("\n" + "=" * 60)
        logger.info("REMOVING INVALID MEASUREMENTS")
        logger.info("=" * 60)

        # Create a set of individual_ids with invalid measurements for quick lookup
        invalid_ids = {r.individual_id for r in results if r.measurements_invalid > 0}

        # Track statistics
        total_measurements_removed = 0
        annotations_modified = 0
        invalid_annotations_saved = []

        # Process each annotation
        cleaned_annotations = []
        for annotation in annotations:
            individual_id = annotation.get("individual_id", "")

            if individual_id not in invalid_ids:
                # No invalid measurements, keep as-is
                cleaned_annotations.append(annotation)
                continue

            # This annotation has invalid measurements - filter them out
            original_measurements = annotation.get("measurements", [])
            valid_measurements = []

            # Find the corresponding result to get image dimensions
            result = next((r for r in results if r.individual_id == individual_id), None)
            if not result:
                # Couldn't validate, keep as-is
                cleaned_annotations.append(annotation)
                continue

            img_width = result.image_width
            img_height = result.image_height

            # Check each measurement
            for measurement in original_measurements:
                measurement_type = measurement.get("measurement_type", "")

                # Only validate elytra measurements
                if not measurement_type.startswith("elytra_"):
                    valid_measurements.append(measurement)
                    continue

                coords = measurement.get("coords_px", {})
                if not coords or "x1" not in coords:
                    valid_measurements.append(measurement)
                    continue

                # Check if measurement is valid
                is_valid, _ = validate_measurement_coords(
                    coords, img_width, img_height, measurement_type
                )

                if is_valid:
                    valid_measurements.append(measurement)
                else:
                    total_measurements_removed += 1

            # Update annotation with filtered measurements
            cleaned_annotation = annotation.copy()
            cleaned_annotation["measurements"] = valid_measurements

            if len(valid_measurements) < len(original_measurements):
                annotations_modified += 1
                # Save this annotation to invalid list
                if cfg.save_invalid_separately:
                    invalid_annotation_info = {
                        "annotation": annotation,
                        "removed_measurements": len(original_measurements) - len(valid_measurements),
                    }
                    invalid_annotations_saved.append(invalid_annotation_info)

            cleaned_annotations.append(cleaned_annotation)

        logger.info("Removed %d invalid measurements from %d annotations",
                   total_measurements_removed, annotations_modified)

        # Save invalid annotations separately if requested
        if cfg.save_invalid_separately and invalid_annotations_saved:
            invalid_file = cfg.annotations_file.parent / "invalid_annotations.json"
            invalid_data = [item["annotation"] for item in invalid_annotations_saved]
            with open(invalid_file, "w") as f:
                json.dump(invalid_data, f, indent=2)
            logger.info("Saved %d annotations with invalid measurements to %s",
                       len(invalid_annotations_saved), invalid_file)

        # Save cleaned annotations
        logger.info("\nSaving cleaned annotations to %s", cfg.annotations_file)
        with open(cfg.annotations_file, "w") as f:
            json.dump(cleaned_annotations, f, indent=2)

        logger.info("✓ Annotations file updated - %d measurements removed",
                   total_measurements_removed)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(tyro.cli(Config)))
    except KeyboardInterrupt:
        print("Interrupted.")
        raise SystemExit(130)
