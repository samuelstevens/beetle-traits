import abc
import dataclasses
import typing as tp

import beartype
import grain
import numpy as np
from jaxtyping import Float, jaxtyped
from PIL import Image


@beartype.beartype
class Config(abc.ABC):
    @property
    @abc.abstractmethod
    def dataset(self): ...


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


@beartype.beartype
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
    """Image size in pixels."""
    sigma: float = 3.0
    """Standard deviation in pixels."""

    def map(self, sample: dict[str, object]) -> dict[str, object]:
        """Reads the 'tgt' key and adds a 'heatmap': Float[np.ndarray, "height width"] key-value pair to the sample dict.

        The heatmap should be a Gaussian/normal distribution centered on the tgt keypoint, with a peak of 1.0 and a std dev of self.sigma.
        """
