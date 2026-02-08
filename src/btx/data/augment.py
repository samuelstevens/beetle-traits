import dataclasses
import typing as tp

import beartype
import grain
import numpy as np
from jaxtyping import Bool, Float, jaxtyped

OobPolicy = tp.Literal["mask_any_oob", "mask_all_oob", "supervise_oob"]


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class AugmentConfig:
    """Configuration for train-time augmentation and cm-metric masking behavior."""

    go: bool = True
    """Whether to enable the augmentation pipeline."""
    size: int = 256
    """Output image side length in pixels. Fixed to 256 for this experiment."""

    crop_scale_min: float = 0.5
    """Minimum random crop area scale for RandomResizedCrop."""
    crop_scale_max: float = 1.0
    """Maximum random crop area scale for RandomResizedCrop."""
    crop_ratio_min: float = 0.75
    """Minimum random crop aspect ratio for RandomResizedCrop."""
    crop_ratio_max: float = 1.333
    """Maximum random crop aspect ratio for RandomResizedCrop."""

    hflip_prob: float = 0.5
    """Probability of applying horizontal flip."""
    vflip_prob: float = 0.5
    """Probability of applying vertical flip."""
    rotation_prob: float = 0.75
    """Probability of applying a non-identity k*90-degree rotation."""

    brightness: float = 0.2
    """Color jitter brightness strength."""
    contrast: float = 0.2
    """Color jitter contrast strength."""
    saturation: float = 0.2
    """Color jitter saturation strength."""
    hue: float = 0.1
    """Color jitter hue strength."""

    oob_policy: OobPolicy = "supervise_oob"
    """Out-of-bounds supervision policy."""
    min_px_per_cm: float = 1e-6
    """Minimum valid scalebar length in pixels for cm metrics."""

    def __post_init__(self):
        msg = f"Expected fixed size 256 for experiment, got {self.size}"
        assert self.size == 256, msg


@jaxtyped(typechecker=beartype.beartype)
def get_identity_affine() -> Float[np.ndarray, "3 3"]:
    raise NotImplementedError()


@jaxtyped(typechecker=beartype.beartype)
def get_crop_resize_affine(
    x0: float,
    y0: float,
    crop_w: float,
    crop_h: float,
    *,
    size: int,
) -> Float[np.ndarray, "3 3"]:
    raise NotImplementedError()


@jaxtyped(typechecker=beartype.beartype)
def get_hflip_affine(*, size: int) -> Float[np.ndarray, "3 3"]:
    raise NotImplementedError()


@jaxtyped(typechecker=beartype.beartype)
def get_vflip_affine(*, size: int) -> Float[np.ndarray, "3 3"]:
    raise NotImplementedError()


@jaxtyped(typechecker=beartype.beartype)
def get_rot90_affine(k: int, *, size: int) -> Float[np.ndarray, "3 3"]:
    raise NotImplementedError()


@jaxtyped(typechecker=beartype.beartype)
def apply_affine_to_points(
    affine_33: Float[np.ndarray, "3 3"],
    points_l22: Float[np.ndarray, "lines 2 2"],
) -> Float[np.ndarray, "lines 2 2"]:
    raise NotImplementedError()


@jaxtyped(typechecker=beartype.beartype)
def is_in_bounds(
    points_l22: Float[np.ndarray, "lines 2 2"],
    *,
    size: int,
) -> Bool[np.ndarray, "lines 2"]:
    raise NotImplementedError()


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class InitAugState(grain.transforms.Map):
    size: int = 256
    min_px_per_cm: float = 1e-6

    def map(self, element: object) -> object:
        raise NotImplementedError()


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class RandomResizedCrop(grain.transforms.RandomMap):
    cfg: AugmentConfig

    def random_map(
        self,
        element: object,
        rng: np.random.Generator,
    ) -> object:
        raise NotImplementedError()


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class RandomFlip(grain.transforms.RandomMap):
    cfg: AugmentConfig

    def random_map(
        self,
        element: object,
        rng: np.random.Generator,
    ) -> object:
        raise NotImplementedError()


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class RandomRotation90(grain.transforms.RandomMap):
    cfg: AugmentConfig

    def random_map(
        self,
        element: object,
        rng: np.random.Generator,
    ) -> object:
        raise NotImplementedError()


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class ColorJitter(grain.transforms.RandomMap):
    cfg: AugmentConfig

    def random_map(
        self,
        element: object,
        rng: np.random.Generator,
    ) -> object:
        raise NotImplementedError()


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class FinalizeTargets(grain.transforms.Map):
    cfg: AugmentConfig

    def map(self, element: object) -> object:
        raise NotImplementedError()
