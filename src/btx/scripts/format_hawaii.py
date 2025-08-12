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
import logging
import pathlib

import beartype
import numpy as np
import polars as pl
import skimage.feature
import tyro
from jaxtyping import Float
from PIL import Image

import btx.helpers

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("format_hawaii.py")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    hf_root: pathlib.Path = pathlib.Path(
        "/fs/scratch/PAS2136/samuelstevens/datasets/hawaii-beetles"
    )
    """Where you dumped data when using download_hawaii.py."""


@beartype.beartype
def img_as_arr(
    img: Image.Image | pathlib.Path,
) -> Float[np.ndarray, "width height channels"]:
    img = img if isinstance(img, Image.Image) else Image.open(img)
    return np.asarray(img, dtype=np.float32)


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
    group_img_abs_path: str
    indiv_img_abs_path: str
    indiv_offset_px: tuple[float, float]

    @property
    def indiv_bbox_px(self) -> tuple[float, float, float, float]:
        # Use indiv_img.size to figure out the lower right corner.
        raise NotImplementedError()


@beartype.beartype
def worker_fn(
    cfg: Config, group_img_basenames: list[str]
) -> list[Annotation | WorkerError]:
    """Worker. Processing group_img_basenames and returns a list of annotations or errors."""
    img_df = load_img_df(cfg)
    results = []

    for group_img_basename in group_img_basenames:
        # Construct the group image path (need to uppercase the basename for filesystem)
        group_img_abs_path = (
            cfg.hf_root / "group_images" / f"{group_img_basename.upper()}.png"
        )

        # Load the group image
        try:
            group_img = img_as_arr(group_img_abs_path)
        except Exception as e:
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

        for row in group_rows.iter_rows(named=True):
            beetle_position = row["BeetlePosition"]
            indiv_img_rel_path = row["individualImageFilePath"]
            indiv_img_abs_path = cfg.hf_root / indiv_img_rel_path

            # Load the individual image
            try:
                template = img_as_arr(indiv_img_abs_path)
            except Exception as e:
                results.append(
                    ImageLoadError(
                        group_img_basename=group_img_basename,
                        message=str(e),
                        img_path=str(indiv_img_abs_path),
                    )
                )
                continue

            # Perform template matching
            try:
                corr = skimage.feature.match_template(
                    group_img, template, pad_input=False
                )
                if corr.size == 0:
                    raise ValueError(
                        "Empty correlation map - template may be larger than image"
                    )

                iy, ix = np.unravel_index(np.argmax(corr), corr.shape)
                offset_px = (int(ix), int(iy))

                annotation = Annotation(
                    group_img_basename=group_img_basename,
                    beetle_position=beetle_position,
                    group_img_abs_path=str(group_img_abs_path),
                    indiv_img_abs_path=str(indiv_img_abs_path),
                    indiv_offset_px=offset_px,
                )
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

    return results


@beartype.beartype
def load_trait_df(cfg: Config) -> pl.DataFrame:
    return pl.read_csv(cfg.hf_root / "trait_annotations.csv").with_columns(
        pl.col("groupImageFilePath")
        .str.to_lowercase()
        .str.strip_prefix("group_images/")
        .str.strip_suffix(".png")
        .alias("GroupImgBasename"),
        pl.col("coords_scalebar").str.json_decode(),
        pl.col("coords_elytra_max_length").str.json_decode(),
        pl.col("coords_basal_pronotum_width").str.json_decode(),
        pl.col("coords_elytra_max_width").str.json_decode(),
    )


@beartype.beartype
def load_img_df(cfg: Config) -> pl.DataFrame:
    return pl.read_csv(cfg.hf_root / "images_metadata.csv").with_columns(
        pl.col("groupImageFilePath")
        .str.to_lowercase()
        .str.strip_prefix("group_images/")
        .str.strip_suffix(".png")
        .alias("GroupImgBasename"),
        pl.col("individualImageFilePath")
        .str.to_lowercase()
        .str.extract(r"specimen_(\d+)", 1)
        .cast(pl.Int64)
        .alias("BeetlePosition"),
    )


@beartype.beartype
def main(cfg: Config) -> int:
    errors = []

    trait_df = load_trait_df(cfg)
    # Check that there are no duplicate unique keys
    trait_dups = (
        trait_df.group_by("GroupImgBasename", "BeetlePosition")
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
        img_df.group_by("GroupImgBasename", "BeetlePosition")
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

    # Check that all image files exist
    logger.info("Checking that all image files exist.")

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

    # Check dimensions for a sample of 100 group images
    logger.info("Checking image dimensions for a sample of group images...")

    # Get unique group images with their corresponding individual images
    group_to_individuals = (
        img_df.group_by("groupImageFilePath")
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

    # TODO: (future) run parallel jobs for template matching.
    #
    # TODO: check that len(annotations) = sum(individuals per group image x n group images)
    # Don't error out. Just report an error.
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(tyro.cli(Config)))
    except KeyboardInterrupt:
        print("Interrupted.")
        raise SystemExit(130)
