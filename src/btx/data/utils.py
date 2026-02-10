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
    loss_mask: Float[np.ndarray, "2"]
    """Mask for {width, length} indicating which measurements to train on. 1.0 = train, 0.0 = skip."""

    # Metadata
    beetle_id: str
    beetle_position: int
    group_img_basename: str
    scientific_name: str


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
    imagenet_normalize: bool = False
    """Apply ImageNet mean/std normalization (for pretrained DINOv2/v3 ViTs)."""

    def map(self, sample: dict[str, object]) -> dict[str, object]:
        img = sample["img"]
        orig_w, orig_h = img.size

        img = np.array(img.resize((self.size, self.size)))
        img = img.astype(np.float32) / 255.0
        if self.imagenet_normalize:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std
        sample["img"] = img

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


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Normalize(grain.transforms.Map):
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def map(self, sample: dict[str, object]) -> dict[str, object]:
        img = sample["img"]
        msg = f"Expected ndarray image, got {type(img)}"
        assert isinstance(img, np.ndarray), msg
        assert img.ndim == 3 and img.shape[-1] == 3, (
            f"Expected HWC image with 3 channels, got {img.shape}"
        )

        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        assert mean.shape == (3,), f"Expected mean shape (3,), got {mean.shape}"
        assert std.shape == (3,), f"Expected std shape (3,), got {std.shape}"
        assert np.all(np.isfinite(mean)), "mean must be finite"
        assert np.all(np.isfinite(std)), "std must be finite"
        assert np.all(std > 0.0), "std must be positive"

        img_f32 = img.astype(np.float32, copy=False)
        sample["img"] = (img_f32 - mean[None, None, :]) / std[None, None, :]
        return sample
