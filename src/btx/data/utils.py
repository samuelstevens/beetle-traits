import dataclasses
import typing as tp

import beartype
import grain
import numpy as np
from jaxtyping import Float, jaxtyped
from PIL import Image


@jaxtyped(typechecker=beartype.beartype)
class Sample(tp.TypedDict):
    img_fpath: str

    points_px: Float[np.ndarray, "lines 2 2"]
    """{width, length} x two points x {x, y}."""
    scalebar_px: Float[np.ndarray, "2 2"]
    """two points x {x, y}."""

    # Metadata
    beetle_id: str
    beetle_position: int
    group_img_basename: str


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class DecodeRGB(grain.transforms.Map):
    def map(self, sample: Sample) -> Sample:
        # Heavy I/O lives in a transform so workers can parallelize it
        with Image.open(sample["img_fpath"]) as im:
            sample["img"] = im.convert("RGB")
        return sample


@dataclasses.dataclass(frozen=True)
class Resize(grain.transforms.Map):
    size: int = 256

    def map(self, sample: dict[str, object]) -> dict[str, object]:
        img = sample["img"]
        orig_w, orig_h = img.size

        img = np.array(img.resize((self.size, self.size)))
        sample["img"] = img.astype(np.float32) / 255.0

        # Rescale the measurements according to the new size
        scale_x = self.size / orig_w
        scale_y = self.size / orig_h

        points = sample["points_px"].copy()
        points[:, :, 0] *= scale_x
        points[:, :, 1] *= scale_y
        sample["tgt"] = points

        sample["scale"] = np.array([scale_x, scale_y])

        return sample

@beartype.beartype
@dataclasses.dataclass(frozen=True)
class GaussianHeatmap(grain.transforms.Map):
    size: int = 256
    """Image size in pixels (used if we can't infer from sample['img'])."""
    sigma: float = 3.0
    """Standard deviation of the Gaussian in pixels."""

    def map(self, sample: dict[str, object]) -> dict[str, object]:
        """
        Reads 'tgt' (keypoint coordinates) and adds:
          sample['tgt_pixel_probs']: Float[np.ndarray, "points height width"]

        Each channel is a 2D Gaussian centered at the corresponding keypoint,
        with peak value 1.0 and std-dev = self.sigma (in pixels).
        """
        # --- 1) Get image size (H, W) robustly --------------------------------
        if "img" in sample and isinstance(sample["img"], np.ndarray):
            H, W = sample["img"].shape[:2]
        else:
            H = W = int(self.size)

        # --- 2) Read and normalize target keypoints ----------------------------
        # Expected 'tgt' shape from your pipeline: (lines, 2, 2)  -> {width,length} x two points x {x,y}
        # We treat each endpoint as its own keypoint: reshape to (num_points, 2) with columns (x, y).
        tgt = np.asarray(sample["tgt"], dtype=np.float32)  # shape: (lines, 2, 2)
        pts = tgt.reshape(-1, 2)                           # shape: (num_points, 2)

        # --- 3) Build pixel grid for Gaussian computation ----------------------
        # yy[i, j] = i (row / y), xx[i, j] = j (col / x)
        yy, xx = np.meshgrid(
            np.arange(H, dtype=np.float32),
            np.arange(W, dtype=np.float32),
            indexing="ij",
        )  # shapes: (H, W)

        # Prepare for broadcasting: add a leading 'points' axis to the grid
        xx = xx[None, ...]  # (1, H, W)
        yy = yy[None, ...]  # (1, H, W)

        # --- 4) Compute Gaussian heatmaps per point ----------------------------
        # x0, y0: (num_points, 1, 1) so they broadcast over (H, W)
        x0 = pts[:, 0][:, None, None]
        y0 = pts[:, 1][:, None, None]

        sigma2 = float(self.sigma) ** 2
        # Gaussian: exp(-((x-x0)^2 + (y-y0)^2) / (2*sigma^2))
        probs = np.exp(-(((xx - x0) ** 2 + (yy - y0) ** 2) / (2.0 * sigma2)))  # (points, H, W)

        # --- 5) Handle invalid coordinates (NaN/Inf) by zeroing their channel --
        invalid = ~np.isfinite(pts).all(axis=1)
        if invalid.any():
            probs[invalid] = 0.0

        # --- 6) Store result (float32, points x H x W) -------------------------
        sample["tgt_pixel_probs"] = probs.astype(np.float32)
        # Optional: keep sigma for reference/debugging
        sample["tgt_pixel_sigma"] = np.array(self.sigma, dtype=np.float32)

        return sample
