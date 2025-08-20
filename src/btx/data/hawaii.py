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
import jax.numpy as jnp
import polars as pl
from jaxtyping import Array, Float, jaxtyped


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
    seed: int = 0

    def __post_init__(self):
        # TODO: Check that hf_root exists and is a directory
        # TODO: Check that annotations.json exists and is a file.
        pass


@jaxtyped(typechecker=beartype.beartype)
class Sample(tp.TypedDict):
    img_fpath: str
    elytra_width_px: Float[Array, "2 2"]
    elytra_length_px: Float[Array, "2 2"]
    beetle_id: str
    beetle_position: int
    group_img_basename: str


@beartype.beartype
class Dataset(grain.sources.RandomAccessDataSource):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.df = pl.read_json(cfg.annotations)

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

    def __getitem__(self, idx: int) -> Sample:
        """Load image and annotations for given index."""
        row = self.df.row(index=idx, named=True)
        fpath = self.cfg.hf_root / "individual_specimens" / row["indiv_img_rel_path"]
        # TODO: include error message.
        assert fpath.is_file()

        elytra_width_px = None
        elytra_length_px = None
        for measurement in row["measurements"]:
            if measurement["measurement_type"] == "elytra_max_length":
                elytra_length_px = measurement["polyline_px"]
            if measurement["measurement_type"] == "elytra_max_width":
                elytra_width_px = measurement["polyline_px"]

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

        if self.cfg.include_polylines:
            raise NotImplementedError()

        return Sample(
            img_fpath=str(fpath),
            elytra_width_px=jnp.array(elytra_width_px).reshape(2, 2),
            elytra_length_px=jnp.array(elytra_length_px).reshape(2, 2),
            beetle_id=row["individual_id"],
            beetle_position=row["beetle_position"],
            group_img_basename=row["group_img_basename"],
        )
