"""
Visualize measurements on a specific individual beetle image.

This script loads annotations.json, finds a specific beetle by individual_id,
and draws all measurements on the individual beetle image.
"""

import dataclasses
import json
import logging
import pathlib

import beartype
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

    output_dir: pathlib.Path = pathlib.Path(
        "./data/beetlepalooza-formatted/individual-visualization"
    )
    """Where to save visualization images."""

    individual_id: str = "A00000046137_9"
    """The individual_id to visualize."""


@beartype.beartype
def visualize_individual(
    annotation: dict,
    cfg: Config,
    logger: logging.Logger,
) -> None:
    """Draw measurements on the individual beetle image."""
    individual_id = annotation.get("individual_id", "unknown")

    # Get the individual image path
    indiv_img_abs_path = annotation.get("indiv_img_abs_path", "")
    if not indiv_img_abs_path:
        logger.error("No indiv_img_abs_path for %s", individual_id)
        return

    indiv_img_path = pathlib.Path(indiv_img_abs_path)

    if not indiv_img_path.exists():
        logger.error("Individual image not found: %s", indiv_img_path)
        return

    try:
        indiv_img = Image.open(indiv_img_path).convert("RGB")
    except Exception as e:
        logger.error("Failed to open individual image %s: %s", indiv_img_path, e)
        return

    draw = ImageDraw.Draw(indiv_img)
    img_width, img_height = indiv_img.size

    logger.info("Image dimensions: %dx%d", img_width, img_height)

    # Define colors for different measurement types
    measurement_colors = {
        "elytra_length": (0, 255, 0),  # Green
        "elytra_width": (255, 255, 0),  # Yellow
        "pronotum_width": (0, 0, 255),  # Blue
    }

    # Draw measurements
    measurements = annotation.get("measurements", [])
    logger.info("Found %d measurements", len(measurements))

    for idx, measurement in enumerate(measurements):
        measurement_type = measurement.get("measurement_type", "")
        coords = measurement.get("coords_px", {})
        dist_cm = measurement.get("dist_cm", None)

        if not coords or "x1" not in coords:
            logger.warning("Measurement %d has no valid coords", idx)
            continue

        x1, y1 = coords["x1"], coords["y1"]
        x2, y2 = coords["x2"], coords["y2"]

        # Get color for this measurement type
        color = measurement_colors.get(
            measurement_type, (255, 0, 255)
        )  # Magenta default

        # Check if coordinates are within bounds
        in_bounds = (
            0 <= x1 <= img_width
            and 0 <= x2 <= img_width
            and 0 <= y1 <= img_height
            and 0 <= y2 <= img_height
        )

        if not in_bounds:
            logger.warning(
                "Measurement %s has out-of-bounds coords: (%d,%d)-(%d,%d) for image %dx%d",
                measurement_type,
                int(x1),
                int(y1),
                int(x2),
                int(y2),
                img_width,
                img_height,
            )
            # Use red for out-of-bounds
            color = (255, 0, 0)

        # Draw the measurement line
        draw.line([(x1, y1), (x2, y2)], fill=color, width=3)

        # Draw circles at endpoints
        radius = 5
        draw.ellipse([x1 - radius, y1 - radius, x1 + radius, y1 + radius], fill=color)
        draw.ellipse([x2 - radius, y2 - radius, x2 + radius, y2 + radius], fill=color)

        # Try to add text label (may fail if no font available)
        try:
            label = f"{measurement_type}"
            if dist_cm:
                label += f" ({dist_cm:.3f} cm)"

            # Put label near the midpoint of the line
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2

            # Offset label slightly so it doesn't overlap the line
            text_pos = (int(mid_x) + 10, int(mid_y) + 10)
            draw.text(text_pos, label, fill=color)
        except Exception:
            # If font rendering fails, just skip the label
            pass

        logger.info(
            "  %s: (%d,%d)->(%d,%d) [%s] %s",
            measurement_type,
            int(x1),
            int(y1),
            int(x2),
            int(y2),
            "in bounds" if in_bounds else "OUT OF BOUNDS",
            f"{dist_cm:.3f} cm" if dist_cm else "",
        )

    # Save visualization
    output_path = cfg.output_dir / f"{individual_id}_measurements.png"
    indiv_img.save(output_path)
    logger.info("Saved visualization to %s", output_path)


@beartype.beartype
def main(cfg: Config) -> int:
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger("visualize-individual")

    # Load annotations
    logger.info("Loading annotations from %s", cfg.annotations_file)
    if not cfg.annotations_file.exists():
        logger.error("Annotations file not found: %s", cfg.annotations_file)
        return 1

    with open(cfg.annotations_file, "r") as f:
        annotations = json.load(f)

    logger.info("Loaded %d annotations", len(annotations))

    # Find the specific annotation
    logger.info("Searching for individual_id: %s", cfg.individual_id)
    target_annotation = None
    for annotation in annotations:
        if annotation.get("individual_id") == cfg.individual_id:
            target_annotation = annotation
            break

    if not target_annotation:
        logger.error("Individual ID '%s' not found in annotations", cfg.individual_id)
        logger.info("Available individual IDs (first 10):")
        for i, ann in enumerate(annotations[:10]):
            logger.info("  - %s", ann.get("individual_id", "unknown"))
        return 1

    logger.info("Found annotation for %s", cfg.individual_id)
    logger.info("  Group image: %s", target_annotation.get("group_img_basename"))
    logger.info("  Beetle position: %s", target_annotation.get("beetle_position"))
    logger.info(
        "  Origin: (%s, %s)",
        target_annotation.get("origin_x"),
        target_annotation.get("origin_y"),
    )
    logger.info("  NCC score: %.4f", target_annotation.get("ncc", 0.0))

    # Create output directory
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Visualize the annotation
    visualize_individual(target_annotation, cfg, logger)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(tyro.cli(Config)))
    except KeyboardInterrupt:
        print("Interrupted.")
        raise SystemExit(130)
