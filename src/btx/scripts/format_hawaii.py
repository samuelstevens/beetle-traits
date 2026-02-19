# src/btx/scripts/format_hawaii.py
"""
Some context:

[I] samuelstevens@ascend-login01 /f/s/P/s/d/hawaii-beetles [1]> pwd
/fs/scratch/PAS2136/samuelstevens/datasets/hawaii-beetles
[I] samuelstevens@ascend-login01 /f/s/P/s/d/hawaii-beetles> ls
group_images  images_metadata.csv  individual_specimens  README.md  trait_annotations.csv
[I] samuelstevens@ascend-login01 /f/s/P/s/d/hawaii-beetles> ls group_images/ | head
IMG_0093.png
IMG_0095.png
IMG_0109.png
IMG_0110.png
IMG_0111.png
IMG_0112.png
IMG_0113.png
IMG_0114.png
IMG_0115.png
IMG_0116.png
[I] samuelstevens@ascend-login01 /f/s/P/s/d/hawaii-beetles> ls individual_specimens/ | head
IMG_0093_specimen_1_MECKON_NEON.BET.D20.000001.png
IMG_0093_specimen_2_MECKON_NEON.BET.D20.000003.png
IMG_0093_specimen_3_MECKON_NEON.BET.D20.000004.png
IMG_0095_specimen_1_MECKON_NEON.BET.D20.000005.png
IMG_0095_specimen_2_MECKON_NEON.BET.D20.000007.png
IMG_0095_specimen_3_MECKON_NEON.BET.D20.000010.png
IMG_0109_specimen_1_MECKON_NEON.BET.D20.000011.png
IMG_0109_specimen_2_MECKON_NEON.BET.D20.000017.png
IMG_0109_specimen_3_MECKON_NEON.BET.D20.000026.png
IMG_0110_specimen_1_MECKON_NEON.BET.D20.000035.png
[I] samuelstevens@ascend-login01 /f/s/P/s/d/hawaii-beetles> head trait_annotations.csv
groupImageFilePath,BeetlePosition,individualID,coords_scalebar,coords_elytra_max_length,coords_basal_pronotum_width,coords_elytra_max_width,px_scalebar,px_elytra_max_length,px_basal_pronotum_width,px_elytra_max_width,cm_scalebar,cm_elytra_max_length,cm_basal_pronotum_width,cm_elytra_max_width
group_images/IMG_0093.png,1,NEON.BET.D20.000001,"[[5713.91, 3045.68, 5701.21, 2265.92]]","[[3865.5, 1245.87, 3881.25, 1045.81]]","[[3922.92, 1046.2, 3872.53, 1035.06]]","[[3960.08, 1145.79, 3814.38, 1123.85]]",779.8634159902615,200.67901260470657,51.606702084128464,147.34264012837542,1.0,0.257,0.066,0.189
group_images/IMG_0093.png,2,NEON.BET.D20.000003,"[[5713.91, 3045.68, 5701.21, 2265.92]]","[[3899.77, 2528.67, 3939.62, 2338.98]]","[[3961.39, 2339.49, 3912.19, 2329.79]]","[[3974.62, 2440.72, 3895.98, 2421]]",779.8634159902615,193.8306441200669,50.14708366395775,81.07482963287664,1.0,0.249,0.064,0.104
group_images/IMG_0093.png,3,NEON.BET.D20.000004,"[[5713.91, 3045.68, 5701.21, 2265.92]]","[[3998.51, 3859.69, 4002.25, 3686.53]]","[[4047.58, 3684.09, 4013.58, 3681.39]]","[[4080.85, 3774.4, 3959.48, 3763.86]]",779.8634159902615,173.2003845261319,34.1070373969948,121.82679713429216,1.0,0.222,0.044,0.156
group_images/IMG_0095.png,1,NEON.BET.D20.000005,"[[4361.9, 3001.51, 4356.57, 2218.33]]","[[2727.56, 991.13, 2777.05, 746.39]]","[[2771.23, 737.52, 2719.33, 714.82]]","[[2819.12, 840.3, 2651.68, 832.24]]",783.1981366806234,249.69366772106983,56.64715350306674,167.63387843750445,1.0,0.319,0.072,0.214
group_images/IMG_0095.png,2,NEON.BET.D20.000007,"[[4361.9, 3001.51, 4356.57, 2218.33]]","[[2637.35, 2011.99, 2701.51, 1851.59]]","[[2703.39, 1846.64, 2660.39, 1834.34]]","[[2721.27, 1936.92, 2601.72, 1899.58]]",783.1981366806234,172.7560870128751,44.72460173103842,125.24567098307251,1.0,0.221,0.057,0.16
group_images/IMG_0095.png,3,NEON.BET.D20.000010,"[[4361.9, 3001.51, 4356.57, 2218.33]]","[[2665.12, 2947.47, 2726.98, 2725.72]]","[[2725.91, 2711.99, 2675.41, 2705.29]]","[[2755.34, 2836.81, 2597.89, 2812.56]]",783.1981366806234,230.2166851034043,50.94251662413232,159.30651273566968,1.0,0.294,0.065,0.203
group_images/IMG_0109.png,1,NEON.BET.D20.000011,"[[5963.41, 2293.36, 5928.69, 1510.68]]","[[4369.81, 1151.98, 4434.72, 964.11]]","[[4435.89, 958.48, 4390.69, 941.68]]","[[4453.82, 1063.75, 4319.9, 1024.04]]",783.4497181057634,198.767313711284,48.22115718229985,139.68339378752228,1.0,0.254,0.062,0.178
group_images/IMG_0109.png,2,NEON.BET.D20.000017,"[[5963.41, 2293.36, 5928.69, 1510.68]]","[[4332.92, 1979.73, 4335.18, 1781.63]]","[[4395.78, 1775.08, 4340.7, 1755.66]]","[[4428.95, 1868.03, 4287.33, 1862.43]]",783.4497181057634,198.1128910495225,58.40327730530185,141.73067557871858,1.0,0.253,0.075,0.181
group_images/IMG_0109.png,3,NEON.BET.D20.000026,"[[5963.41, 2293.36, 5928.69, 1510.68]]","[[4264.26, 2738.34, 4313.08, 2537.98]]","[[4362.27, 2538.03, 4321.57, 2526.78]]","[[4386.88, 2641.59, 4329.2, 2638.75, 4324.27, 2637.61, 4245.53, 2614.46]]",783.4497181057634,206.2220211325649,42.2262063178787,139.82246488940592,1.0,0.263,0.054,0.178
[I] samuelstevens@ascend-login01 /f/s/P/s/d/hawaii-beetles> head images_metadata.csv
individualImageFilePath,groupImageFilePath,individualID,taxonID,scientificName,plotID,trapID,plotTrapID,collectDate,ownerInstitutionCode,catalogNumber
individual_specimens/IMG_0093_specimen_1_MECKON_NEON.BET.D20.000001.png,group_images/IMG_0093.png,NEON.BET.D20.000001,MECKON,Mecyclothorax konanus,6,W,006W,20190424,NEON,DP1.10022.001
individual_specimens/IMG_0093_specimen_2_MECKON_NEON.BET.D20.000003.png,group_images/IMG_0093.png,NEON.BET.D20.000003,MECKON,Mecyclothorax konanus,16,W,016W,20190424,NEON,DP1.10022.001
individual_specimens/IMG_0093_specimen_3_MECKON_NEON.BET.D20.000004.png,group_images/IMG_0093.png,NEON.BET.D20.000004,MECKON,Mecyclothorax konanus,6,E,006E,20190424,NEON,DP1.10022.001
individual_specimens/IMG_0095_specimen_1_MECKON_NEON.BET.D20.000005.png,group_images/IMG_0095.png,NEON.BET.D20.000005,MECKON,Mecyclothorax konanus,14,S,014S,20190424,NEON,DP1.10022.001
individual_specimens/IMG_0095_specimen_2_MECKON_NEON.BET.D20.000007.png,group_images/IMG_0095.png,NEON.BET.D20.000007,MECKON,Mecyclothorax konanus,6,E,006E,20190424,NEON,DP1.10022.001
individual_specimens/IMG_0095_specimen_3_MECKON_NEON.BET.D20.000010.png,group_images/IMG_0095.png,NEON.BET.D20.000010,MECKON,Mecyclothorax konanus,14,W,014W,20190424,NEON,DP1.10022.001
individual_specimens/IMG_0109_specimen_1_MECKON_NEON.BET.D20.000011.png,group_images/IMG_0109.png,NEON.BET.D20.000011,MECKON,Mecyclothorax konanus,14,E,014E,20190424,NEON,DP1.10022.001
individual_specimens/IMG_0109_specimen_2_MECKON_NEON.BET.D20.000017.png,group_images/IMG_0109.png,NEON.BET.D20.000017,MECKON,Mecyclothorax konanus,14,S,014S,20190424,NEON,DP1.10022.001
individual_specimens/IMG_0109_specimen_3_MECKON_NEON.BET.D20.000026.png,group_images/IMG_0109.png,NEON.BET.D20.000026,MECKON,Mecyclothorax konanus,14,W,014W,20190424,NEON,DP1.10022.001
"""

import dataclasses
import gc
import itertools
import json
import logging
import pathlib
import resource
import time
import typing as tp

import beartype
import numpy as np
import polars as pl
import skimage.feature
import submitit
import tyro
from jaxtyping import Float, UInt
from PIL import Image, ImageDraw

import btx.helpers

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    hf_root: pathlib.Path = pathlib.Path("./data/hawaii")
    """Where you dumped data when using download_hawaii.py."""

    log_to: pathlib.Path = pathlib.Path("./logs")
    """Where to save submitit/slurm logs."""

    dump_to: pathlib.Path = pathlib.Path("./data/hawaii-formatted")
    """Where to save formatted data."""

    ignore_errors: bool = False
    """Skip the user error check and always proceed (equivalent to answering 'yes')."""

    seed: int = 42
    """Random seed for sampling which annotations to save as examples."""

    sample_rate: int = 20
    """Save 1 in sample_rate annotations as example images (default: 1 in 20)."""

    # Slurm configuration
    slurm_acct: str = ""
    """Slurm account to use. If empty, uses DebugExecutor."""

    slurm_partition: str = "parallel"
    """Slurm partition to use."""

    n_hours: float = 2.0
    """Number of hours to request for each job."""

    groups_per_job: int = 4
    """Number of group images to process per job."""


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class ValidationError:
    """Data validation error with type and count."""

    error_type: tp.Literal[
        "trait_duplicates",
        "image_duplicates",
        "images_without_traits",
        "traits_without_images",
        "missing_files",
        "corrupted_files",
        "dimension_errors",
    ]
    count: int
    details: list[str] = dataclasses.field(default_factory=list)
    """Optional list of example error details for logging."""

    def log_summary(self, logger: logging.Logger) -> None:
        """Log a summary of this error."""
        logger.error("Found %d %s", self.count, self.error_type.replace("_", " "))
        for detail in self.details[:10]:  # Show first 10 examples
            logger.error("  %s", detail)
        if len(self.details) > 10:
            logger.error("  ... and %d more", len(self.details) - 10)

    @property
    def display_name(self) -> str:
        """Human-readable error type name."""
        return self.error_type.replace("_", " ").title()


@beartype.beartype
def img_as_arr(
    img: Image.Image | pathlib.Path,
) -> Float[np.ndarray, "height width channels"]:
    img = img if isinstance(img, Image.Image) else Image.open(img)
    return np.asarray(img, dtype=np.float32)


@beartype.beartype
def img_as_grayscale(
    img: Image.Image | pathlib.Path,
) -> UInt[np.ndarray, "height width"]:
    img = img if isinstance(img, Image.Image) else Image.open(img)
    # Convert to grayscale using PIL (more efficient than loading RGB then converting)
    return np.asarray(img.convert("L"))


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class WorkerError(Exception):
    """Base class for worker errors with context."""

    group_img_basename: str
    message: str

    def __str__(self):
        return f"{self.group_img_basename}: {self.message}"


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class TemplateMatchingError(WorkerError):
    """Error during template matching."""

    beetle_position: int
    indiv_img_path: str

    def __str__(self):
        return f"{self.group_img_basename} (beetle {self.beetle_position}): Template matching failed for {self.indiv_img_path} - {self.message}"


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class ImageLoadError(WorkerError):
    """Error loading an image file."""

    img_path: str

    def __str__(self):
        return f"{self.group_img_basename}: Failed to load {self.img_path} - {self.message}"


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Annotation:
    group_img_basename: str
    beetle_position: int
    group_img_abs_path: pathlib.Path
    indiv_img_abs_path: pathlib.Path
    indiv_offset_px: tuple[float, float]
    individual_id: str
    ncc: float
    """Normalized cross-correlation score from template matching."""
    taxon_id: str
    """Six letter taxon ID code."""
    scientific_name: str
    """Scientific name (genus species)."""

    def to_dict(self) -> dict:
        """Convert annotation to dictionary for JSON serialization."""
        return {
            "group_img_basename": self.group_img_basename,
            "beetle_position": self.beetle_position,
            "group_img_rel_path": f"group_images/{self.group_img_basename.upper()}.png",
            "indiv_img_rel_path": str(self.indiv_img_abs_path).split("/hawaii/")[-1]
            if "/hawaii/" in str(self.indiv_img_abs_path)
            else str(self.indiv_img_abs_path.name),
            "indiv_img_abs_path": str(self.indiv_img_abs_path),
            "individual_id": self.individual_id,
            "origin_x": int(self.indiv_offset_px[0]),
            "origin_y": int(self.indiv_offset_px[1]),
            "ncc": self.ncc,
            "taxon_id": self.taxon_id,
            "scientific_name": self.scientific_name,
        }


@beartype.beartype
def save_example_images(
    dump_to: pathlib.Path, annotation: Annotation, trait_data: dict[str, object]
) -> None:
    """Save example images with annotations drawn on them."""
    # Define colors for different trait types (RGB)
    trait_colors = {
        "coords_elytra_max_length": (0, 255, 0),  # Green
        "coords_basal_pronotum_width": (0, 0, 255),  # Blue
        "coords_elytra_max_width": (255, 255, 0),  # Yellow
    }
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger("save-imgs")

    # Load images for drawing
    try:
        group_img_pil = Image.open(annotation.group_img_abs_path).convert("RGB")
        indiv_img_pil = Image.open(annotation.indiv_img_abs_path).convert("RGB")
    except Exception as e:
        logger.warning(
            "Failed to load images for example: %s beetle %d - %s",
            annotation.group_img_basename,
            annotation.beetle_position,
            e,
        )
        return

    # Get individual image dimensions
    indiv_w, indiv_h = indiv_img_pil.size
    x, y = annotation.indiv_offset_px

    # Draw on group image
    group_draw = ImageDraw.Draw(group_img_pil)

    # Draw bounding box around individual beetle
    group_draw.rectangle(
        (x, y, x + indiv_w, y + indiv_h),
        outline=(255, 0, 0),
        width=12,
    )

    # Draw polylines for each trait type on group image
    for trait_name, color in trait_colors.items():
        if trait_name not in trait_data:
            continue

        coords = trait_data[trait_name]
        if not coords:
            continue

        for polyline in coords:
            if len(polyline) < 2:
                continue

            # Check that polyline has even number of coordinates (x,y pairs)
            if len(polyline) % 2 != 0:
                logger.warning(
                    "Polyline for %s has odd number of coordinates (%d) for %s beetle %d. Skipping.",
                    trait_name,
                    len(polyline),
                    annotation.group_img_basename,
                    annotation.beetle_position,
                )
                continue

            # Convert to list of tuples for PIL
            points = list(itertools.batched(polyline, 2))
            if len(points) < 2:
                continue

            group_draw.line(points, fill=color, width=8)

    # Draw on individual image with adjusted coordinates
    indiv_draw = ImageDraw.Draw(indiv_img_pil)

    for trait_name, color in trait_colors.items():
        if trait_name not in trait_data:
            continue

        coords = trait_data[trait_name]
        if not coords:
            continue

        for polyline in coords:
            if len(polyline) < 2:
                continue

            # Check that polyline has even number of coordinates (x,y pairs)
            if len(polyline) % 2 != 0:
                logger.warning(
                    "Polyline for %s has odd number of coordinates (%d) for %s beetle %d. Skipping.",
                    trait_name,
                    len(polyline),
                    annotation.group_img_basename,
                    annotation.beetle_position,
                )
                continue

            # Adjust coordinates relative to individual image
            adjusted_points = [
                (pt[0] - x, pt[1] - y) for pt in itertools.batched(polyline, 2)
            ]

            # Filter and warn about out-of-bounds points
            valid_points = []
            invalid_count = 0
            for px, py in adjusted_points:
                if 0 <= px < indiv_w and 0 <= py < indiv_h:
                    valid_points.append((px, py))
                else:
                    invalid_count += 1

            if invalid_count > 0:
                logger.warning(
                    "Found %d out-of-bounds points for %s on %s beetle %d (discarded from drawing)",
                    invalid_count,
                    trait_name,
                    annotation.group_img_basename,
                    annotation.beetle_position,
                )

            if len(valid_points) < 2:
                continue

            indiv_draw.line(valid_points, fill=color, width=3)

    # Resize group image for viewing
    group_w, group_h = group_img_pil.size
    resized_group = group_img_pil.resize((group_w // 10, group_h // 10))

    # Save images
    examples_dir = dump_to / "random-examples"
    group_path = (
        examples_dir
        / f"{annotation.group_img_basename}_beetle{annotation.beetle_position}_group.png"
    )
    indiv_path = (
        examples_dir
        / f"{annotation.group_img_basename}_beetle{annotation.beetle_position}_individual.png"
    )

    resized_group.save(group_path)
    indiv_img_pil.save(indiv_path)

    logger.info(
        "Saved example images for %s beetle %d",
        annotation.group_img_basename,
        annotation.beetle_position,
    )


@beartype.beartype
def get_memory_info() -> dict[str, float]:
    """Get current memory usage information."""
    try:
        with open("/proc/meminfo", "r") as f:
            lines = f.readlines()
        meminfo = {}
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0].rstrip(":")
                value = int(parts[1])
                meminfo[key] = value

        mem_total_gb = meminfo.get("MemTotal", 0) / (1024 * 1024)
        mem_available_gb = meminfo.get("MemAvailable", 0) / (1024 * 1024)
        mem_free_gb = meminfo.get("MemFree", 0) / (1024 * 1024)

        # Also get process-specific memory
        usage = resource.getrusage(resource.RUSAGE_SELF)
        process_mem_gb = usage.ru_maxrss / (1024 * 1024)  # Linux reports in KB

        return {
            "total_gb": round(mem_total_gb, 2),
            "available_gb": round(mem_available_gb, 2),
            "free_gb": round(mem_free_gb, 2),
            "used_gb": round(mem_total_gb - mem_available_gb, 2),
            "process_gb": round(process_mem_gb, 2),
            "percent_used": round(
                (mem_total_gb - mem_available_gb) / mem_total_gb * 100, 1
            ),
        }
    except Exception as e:
        return {"error": str(e)}


@beartype.beartype
def worker_fn(
    cfg: Config, group_img_basenames: list[str]
) -> list[Annotation | WorkerError]:
    """Worker. Processing group_img_basenames and returns a list of annotations or errors."""
    logging.basicConfig(level=logging.DEBUG, format=log_format)
    logger = logging.getLogger("worker")

    # Log initial memory state
    mem_info = get_memory_info()
    logger.info(
        "Starting worker with %d group images. Memory: %s",
        len(group_img_basenames),
        mem_info,
    )

    # Load dataframes
    img_df = load_img_df(cfg)
    logger.info(
        "Loaded img_df with %d rows. Memory: %s", len(img_df), get_memory_info()
    )

    trait_df = load_trait_df(cfg)
    logger.info(
        "Loaded trait_df with %d rows. Memory: %s", len(trait_df), get_memory_info()
    )

    # Filter to only relevant data to save memory
    logger.info("Filtering dataframes to relevant groups...")
    img_df = img_df.filter(pl.col("GroupImgBasename").is_in(group_img_basenames))
    trait_df = trait_df.filter(pl.col("GroupImgBasename").is_in(group_img_basenames))
    logger.info(
        "After filtering - img_df: %d rows, trait_df: %d rows. Memory: %s",
        len(img_df),
        len(trait_df),
        get_memory_info(),
    )

    results = []

    # Initialize random number generator for sampling
    rng = np.random.default_rng(seed=cfg.seed)

    for idx, group_img_basename in enumerate(group_img_basenames):
        logger.info(
            "Processing group %d/%d: %s. Memory before: %s",
            idx + 1,
            len(group_img_basenames),
            group_img_basename,
            get_memory_info(),
        )

        # Construct the group image path (need to uppercase the basename for filesystem)
        group_img_abs_path = (
            cfg.hf_root / "group_images" / f"{group_img_basename.upper()}.png"
        )

        # Load the group image in grayscale for matching.
        try:
            group_img_gray = img_as_grayscale(group_img_abs_path)
            logger.info(
                "Loaded group image %s, gray shape: %s. Memory: %s",
                group_img_basename,
                group_img_gray.shape,
                get_memory_info(),
            )
        except Exception as e:
            logger.error("Failed to load group image %s: %s", group_img_basename, e)
            results.append(
                ImageLoadError(
                    group_img_basename=group_img_basename,
                    message=str(e),
                    img_path=str(group_img_abs_path),
                )
            )
            continue

        # Find all individual images for this group image
        group_rows = img_df.filter(pl.col("GroupImgBasename") == group_img_basename)
        logger.info(
            "Found %d individual images for group %s",
            len(group_rows),
            group_img_basename,
        )

        for row in group_rows.iter_rows(named=True):
            beetle_position = row["BeetlePosition"]
            indiv_img_rel_path = row["individualImageFilePath"]
            indiv_img_abs_path = cfg.hf_root / indiv_img_rel_path

            # Load the individual image (grayscale for matching)
            try:
                template_gray = img_as_grayscale(indiv_img_abs_path)
                logger.debug(
                    "Loaded individual image for beetle %d, gray shape: %s. Memory: %s",
                    beetle_position,
                    template_gray.shape,
                    get_memory_info(),
                )
            except Exception as e:
                logger.error(
                    "Failed to load individual image for beetle %d: %s",
                    beetle_position,
                    e,
                )
                results.append(
                    ImageLoadError(
                        group_img_basename=group_img_basename,
                        message=str(e),
                        img_path=str(indiv_img_abs_path),
                    )
                )
                continue

            # Perform template matching (using grayscale images)
            try:
                logger.debug(
                    "Starting template matching for beetle %d...", beetle_position
                )
                corr = skimage.feature.match_template(
                    group_img_gray, template_gray, pad_input=False
                )
                logger.debug(
                    "Template matching complete, corr shape: %s. Memory: %s",
                    corr.shape,
                    get_memory_info(),
                )

                if corr.size == 0:
                    raise ValueError(
                        "Empty correlation map - template may be larger than image"
                    )

                max_corr_idx = np.argmax(corr)
                iy, ix = np.unravel_index(max_corr_idx, corr.shape)
                offset_px = float(ix), float(iy)

                # Get the normalized cross-correlation score at the best match
                ncc_score = float(corr.flat[max_corr_idx])

                # Clean up correlation matrix to free memory
                del corr
                del template_gray
                gc.collect()

                # Get individual ID from the row data
                individual_id = row.get("individualID", "")
                taxon_id = row.get("taxonID", "")
                scientific_name = row.get("scientificName", "")

                annotation = Annotation(
                    group_img_basename=group_img_basename,
                    beetle_position=beetle_position,
                    group_img_abs_path=group_img_abs_path,
                    indiv_img_abs_path=indiv_img_abs_path,
                    indiv_offset_px=offset_px,
                    individual_id=individual_id,
                    ncc=ncc_score,
                    taxon_id=taxon_id,
                    scientific_name=scientific_name,
                )

                # Save a random subset of annotations as example images
                if rng.integers(0, cfg.sample_rate) == 0:
                    trait_row = trait_df.filter(
                        (pl.col("GroupImgBasename") == annotation.group_img_basename)
                        & (pl.col("BeetlePosition") == annotation.beetle_position)
                    )

                    if not trait_row.is_empty():
                        trait_data = trait_row.to_dicts()[0]
                        save_example_images(cfg.dump_to, annotation, trait_data)
                        logger.debug(
                            "Saved example image. Memory: %s", get_memory_info()
                        )
                    else:
                        # TODO: log that there are no traits for this group image and beetle position.
                        pass

                results.append(annotation)

            except Exception as e:
                results.append(
                    TemplateMatchingError(
                        group_img_basename=group_img_basename,
                        message=str(e),
                        beetle_position=beetle_position,
                        indiv_img_path=str(indiv_img_abs_path),
                    )
                )

        # Clean up group images after processing all individuals
        del group_img_gray
        gc.collect()
        logger.info(
            "Finished processing group %s. Memory after cleanup: %s",
            group_img_basename,
            get_memory_info(),
        )

    logger.info(
        "Worker completed. Processed %d groups, %d results. Final memory: %s",
        len(group_img_basenames),
        len(results),
        get_memory_info(),
    )
    return results


@beartype.beartype
def load_trait_df(cfg: Config) -> pl.DataFrame:
    polyline_dtype = pl.List(pl.List(pl.Float64))
    return pl.read_csv(cfg.hf_root / "trait_annotations.csv").with_columns(
        pl
        .col("groupImageFilePath")
        .str.to_lowercase()
        .str.strip_prefix("group_images/")
        .str.strip_suffix(".png")
        .alias("GroupImgBasename"),
        pl.col("coords_scalebar").str.json_decode(polyline_dtype),
        pl.col("coords_elytra_max_length").str.json_decode(polyline_dtype),
        pl.col("coords_basal_pronotum_width").str.json_decode(polyline_dtype),
        pl.col("coords_elytra_max_width").str.json_decode(polyline_dtype),
    )


@beartype.beartype
def load_img_df(cfg: Config) -> pl.DataFrame:
    return pl.read_csv(cfg.hf_root / "images_metadata.csv").with_columns(
        pl
        .col("groupImageFilePath")
        .str.to_lowercase()
        .str.strip_prefix("group_images/")
        .str.strip_suffix(".png")
        .alias("GroupImgBasename"),
        pl
        .col("individualImageFilePath")
        .str.to_lowercase()
        .str.extract(r"specimen_(\d+)", 1)
        .cast(pl.Int64)
        .alias("BeetlePosition"),
    )


@beartype.beartype
def main(cfg: Config) -> int:
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger("format-hawaii")
    errors = []

    trait_df = load_trait_df(cfg)
    # Check that there are no duplicate unique keys
    trait_dups = (
        trait_df
        .group_by("GroupImgBasename", "BeetlePosition")
        .len()
        .filter(pl.col("len") > 1)
    )
    if not trait_dups.is_empty():
        error_msg = f"Found {len(trait_dups)} duplicate GroupImgBasename/BeetlePosition combinations in trait annotations"
        logger.error(error_msg)
        errors.append(("trait_duplicates", len(trait_dups)))
        for row in trait_dups.iter_rows(named=True):
            logger.error(
                "  GroupImgBasename: %s, BeetlePosition: %s, count: %d",
                row["GroupImgBasename"],
                row["BeetlePosition"],
                row["len"],
            )

    img_df = load_img_df(cfg)
    img_dups = (
        img_df
        .group_by("GroupImgBasename", "BeetlePosition")
        .len()
        .filter(pl.col("len") > 1)
    )
    if not img_dups.is_empty():
        error_msg = f"Found {len(img_dups)} duplicate GroupImgBasename/BeetlePosition combinations in images metadata"
        logger.error(error_msg)
        errors.append(("image_duplicates", len(img_dups)))
        for row in img_dups.iter_rows(named=True):
            logger.error(
                "  GroupImgBasename: %s, BeetlePosition: %s, count: %d",
                row["GroupImgBasename"],
                row["BeetlePosition"],
                row["len"],
            )

    # Check for rows in img_df that are not in trait_df
    img_keys = img_df.select("GroupImgBasename", "BeetlePosition").unique()
    trait_keys = trait_df.select("GroupImgBasename", "BeetlePosition").unique()

    img_not_in_trait = img_keys.join(
        trait_keys, on=["GroupImgBasename", "BeetlePosition"], how="anti"
    )
    if not img_not_in_trait.is_empty():
        error_msg = f"Found {len(img_not_in_trait)} image entries without corresponding trait annotations"
        logger.error(error_msg)
        errors.append(("images_without_traits", len(img_not_in_trait)))
        for row in img_not_in_trait.head(10).iter_rows(named=True):
            logger.error(
                "  GroupImgBasename: %s, BeetlePosition: %s",
                row["GroupImgBasename"],
                row["BeetlePosition"],
            )
        if len(img_not_in_trait) > 10:
            logger.error("  ... and %d more", len(img_not_in_trait) - 10)

    # Check for rows in trait_df that are not in img_df
    trait_not_in_img = trait_keys.join(
        img_keys, on=["GroupImgBasename", "BeetlePosition"], how="anti"
    )
    if not trait_not_in_img.is_empty():
        error_msg = f"Found {len(trait_not_in_img)} trait annotations without corresponding image entries"
        logger.error(error_msg)
        errors.append(("traits_without_images", len(trait_not_in_img)))
        for row in trait_not_in_img.head(10).iter_rows(named=True):
            logger.error(
                "  GroupImgBasename: %s, BeetlePosition: %s",
                row["GroupImgBasename"],
                row["BeetlePosition"],
            )
        if len(trait_not_in_img) > 10:
            logger.error("  ... and %d more", len(trait_not_in_img) - 10)

    # Check that image files exist
    logger.info("Checking that image files exist.")

    # Get all unique file paths
    all_group_paths = set(img_df["groupImageFilePath"].unique().to_list())
    all_individual_paths = set(img_df["individualImageFilePath"].unique().to_list())
    all_paths = all_group_paths | all_individual_paths

    missing_files = []
    corrupted_files = []

    for i, rel_path in enumerate(
        btx.helpers.progress(
            sorted(all_paths), every=len(all_paths) // 10, desc="file existence"
        )
    ):
        full_path = cfg.hf_root / rel_path
        if not full_path.exists():
            missing_files.append(rel_path)
        else:
            # Try to open the file to check if it's corrupted
            try:
                with Image.open(full_path) as img:
                    _ = img.size  # Force loading to check corruption
            except Exception as e:
                corrupted_files.append((rel_path, str(e)))

    if missing_files:
        error_msg = f"Found {len(missing_files)} missing image files"
        logger.error(error_msg)
        errors.append(("missing_files", len(missing_files)))
        for path in missing_files[:10]:
            logger.error("  Missing: %s", path)
        if len(missing_files) > 10:
            logger.error("  ... and %d more", len(missing_files) - 10)

    if corrupted_files:
        error_msg = f"Found {len(corrupted_files)} corrupted image files"
        logger.error(error_msg)
        errors.append(("corrupted_files", len(corrupted_files)))
        for path, error in corrupted_files[:10]:
            logger.error("  Corrupted: %s (%s)", path, error)
        if len(corrupted_files) > 10:
            logger.error("  ... and %d more", len(corrupted_files) - 10)

    logger.info("Checking image dimensions.")

    # Get unique group images with their corresponding individual images
    group_to_individuals = (
        img_df
        .group_by("groupImageFilePath")
        .agg(pl.col("individualImageFilePath").unique())
        .to_dicts()
    )

    dimension_errors = []
    for group_data in btx.helpers.progress(
        group_to_individuals,
        every=len(group_to_individuals) // 10,
        desc="dimension check",
    ):
        group_path = cfg.hf_root / group_data["groupImageFilePath"]

        # Skip if file doesn't exist (already reported)
        if not group_path.exists():
            continue

        try:
            with Image.open(group_path) as group_img:
                group_width, group_height = group_img.size

                for individual_rel_path in group_data["individualImageFilePath"]:
                    individual_path = cfg.hf_root / individual_rel_path

                    # Skip if file doesn't exist (already reported)
                    if not individual_path.exists():
                        continue

                    try:
                        with Image.open(individual_path) as ind_img:
                            ind_width, ind_height = ind_img.size

                            if ind_width >= group_width or ind_height >= group_height:
                                dimension_errors.append({
                                    "group": group_data["groupImageFilePath"],
                                    "individual": individual_rel_path,
                                    "group_size": (group_width, group_height),
                                    "individual_size": (ind_width, ind_height),
                                })
                    except Exception:
                        # Skip corrupted files (already reported)
                        pass

        except Exception:
            # Skip corrupted files (already reported)
            pass

    if dimension_errors:
        error_msg = (
            f"Found {len(dimension_errors)} dimension validation errors in sample"
        )
        logger.error(error_msg)
        errors.append(("dimension_errors", len(dimension_errors)))
        for err in dimension_errors[:5]:
            logger.error(
                "  Individual %s (%dx%d) >= Group %s (%dx%d)",
                err["individual"],
                err["individual_size"][0],
                err["individual_size"][1],
                err["group"],
                err["group_size"][0],
                err["group_size"][1],
            )
        if len(dimension_errors) > 5:
            logger.error("  ... and %d more", len(dimension_errors) - 5)

    # If there are errors, summarize and ask user if they want to proceed
    if errors:
        print("\n" + "=" * 60)
        print("DATA VALIDATION SUMMARY")
        print("=" * 60)
        print("\nThe following issues were found:")
        for error_type, count in errors:
            print(f"  - {error_type.replace('_', ' ').title()}: {count}")

        print("\n" + "=" * 60)

        if cfg.ignore_errors:
            logger.warning("Ignoring errors due to --ignore-errors flag. Continuing.")
        else:
            response = input(
                "\nDo you want to continue with template matching despite these errors? (yes/no): "
            )
            if response.lower() not in ["yes", "y"]:
                logger.info("Exiting.")
                return 1
            logger.info("Continuing.")
    else:
        logger.info("No data validation errors found.")

    logger.info("Ready for parallel processing implementation.")

    # Create output directory for example images
    examples_dir = cfg.dump_to / "random-examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Example images will be saved to %s", examples_dir)

    # Get all unique group images to process
    all_group_basenames = trait_df.get_column("GroupImgBasename").unique().to_list()
    logger.info("Found %d unique group images to process", len(all_group_basenames))

    # Batch group images into chunks for each job
    group_batches = list(
        btx.helpers.batched_idx(len(all_group_basenames), cfg.groups_per_job)
    )
    logger.info(
        "Will process %d group images in %d jobs (%d groups per job)",
        len(all_group_basenames),
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
                group_batch = all_group_basenames[group_start:group_end]
                job = executor.submit(worker_fn, cfg, group_batch)
                all_jobs.append(job)

        logger.info("Submitted job batch %d/%d", batch_idx + 1, len(job_batches))

    logger.info("Submitted %d total jobs. Waiting for results...", len(all_jobs))

    # Collect results and count annotations/errors
    all_annotations = []
    all_errors = []

    for job_idx, job in enumerate(all_jobs):
        try:
            results = job.result()
            for result in results:
                if isinstance(result, WorkerError):
                    all_errors.append(result)
                else:
                    all_annotations.append(result)
            logger.info("Job %d/%d completed", job_idx + 1, len(all_jobs))
        except Exception as e:
            logger.error("Job %d/%d failed: %s", job_idx + 1, len(all_jobs), e)

    # Report final statistics
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info("Total annotations: %d", len(all_annotations))
    logger.info("Total errors: %d", len(all_errors))

    if all_errors:
        logger.info("\nError summary:")
        error_types = {}
        for error in all_errors:
            error_type = type(error).__name__
            error_types[error_type] = error_types.get(error_type, 0) + 1
        for error_type, count in error_types.items():
            logger.info("  %s: %d", error_type, count)

    # Check expected vs actual annotation count
    expected_count = sum(
        len(img_df.filter(pl.col("GroupImgBasename") == gb))
        for gb in all_group_basenames
    )
    logger.info("\nExpected annotations: %d", expected_count)
    logger.info("Actual annotations: %d", len(all_annotations))
    if expected_count != len(all_annotations) + len(all_errors):
        logger.warning(
            "Mismatch in annotation count! Missing: %d",
            expected_count - len(all_annotations) - len(all_errors),
        )

    if not all_annotations:
        return 1

    # Save annotations to disk
    output_file = cfg.dump_to / "annotations.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving annotations to %s", output_file)

    # Convert annotations to JSON format with trait polylines
    json_data = []

    for annotation in all_annotations:
        # Get base annotation data
        ann_dict = annotation.to_dict()

        # Get trait annotations for this beetle
        trait_row = trait_df.filter(
            (pl.col("GroupImgBasename") == annotation.group_img_basename)
            & (pl.col("BeetlePosition") == annotation.beetle_position)
        )

        # Add trait polylines if available
        if trait_row.is_empty():
            continue

        measurements = []
        trait_data = trait_row.to_dicts()[0]

        # Process each trait type
        trait_types = {
            "coords_scalebar": "scalebar",
            "coords_elytra_max_length": "elytra_max_length",
            "coords_basal_pronotum_width": "basal_pronotum_width",
            "coords_elytra_max_width": "elytra_max_width",
        }

        for coords_key, measurement_type in trait_types.items():
            if coords_key not in trait_data:
                continue

            coords = trait_data[coords_key]
            if not coords:
                continue

            for polyline in coords:
                if len(polyline) < 2:
                    continue

                if len(polyline) % 2 != 0:
                    logger.warning(
                        "Skipping polyline with odd length %d for %s in %s beetle %d",
                        len(polyline),
                        measurement_type,
                        annotation.group_img_basename,
                        annotation.beetle_position,
                    )
                    continue

                # Convert to individual image coordinates
                origin_x, origin_y = annotation.indiv_offset_px
                adjusted_polyline = []
                for i in range(0, len(polyline), 2):
                    adjusted_x = polyline[i] - origin_x
                    adjusted_y = polyline[i + 1] - origin_y
                    adjusted_polyline.extend([adjusted_x, adjusted_y])

                measurements.append({
                    "measurement_type": measurement_type,
                    "polyline_px": adjusted_polyline,
                })

        ann_dict["measurements"] = measurements
        json_data.append(ann_dict)

    # Save to JSON file
    with open(output_file, "w") as f:
        json.dump(json_data, f, indent=2)

    logger.info("Saved %d annotations to %s", len(json_data), output_file)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(tyro.cli(Config)))
    except KeyboardInterrupt:
        print("Interrupted.")
        raise SystemExit(130)
