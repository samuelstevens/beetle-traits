import dataclasses
import typing as tp

import beartype
import grain
import numpy as np
from jaxtyping import Bool, Float, jaxtyped
from PIL import Image, ImageEnhance

OobPolicy = tp.Literal["mask_any_oob", "mask_all_oob", "supervise_oob"]


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class AugmentConfig:
    """Configuration for train-time augmentation and cm-metric masking behavior."""

    go: bool = True
    """Whether to enable the augmentation pipeline."""
    size: int = 256
    """Output image side length in pixels. Fixed to 256 for this experiment."""
    crop: bool = True
    """Whether to use RandomResizedCrop (True) or plain Resize (False) during training. Set to False for tightly-cropped datasets where random cropping would lose too much content."""

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
    """Probability of applying a random rotation (uniform 0-360 degrees)."""

    brightness: float = 0.2
    """Color jitter brightness strength."""
    contrast: float = 0.2
    """Color jitter contrast strength."""
    saturation: float = 0.2
    """Color jitter saturation strength."""
    hue: float = 0.1
    """Color jitter hue strength."""
    color_jitter_prob: float = 1.0
    """Probability of applying color jitter when any jitter strength is nonzero."""
    normalize: bool = True
    """Whether to apply ImageNet normalization at the end of the transform pipeline."""

    oob_policy: OobPolicy = "supervise_oob"
    """Out-of-bounds supervision policy."""
    min_px_per_cm: float = 1e-6
    """Minimum valid scalebar length in pixels for cm metrics."""

    def __post_init__(self):
        msg = f"Expected fixed size 256 for experiment, got {self.size}"
        assert self.size == 256, msg
        msg = f"Expected color_jitter_prob in [0, 1], got {self.color_jitter_prob}"
        assert 0.0 <= self.color_jitter_prob <= 1.0, msg


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class HeatmapTargetConfig:
    """Configuration for optional Gaussian endpoint heatmap targets."""

    go: bool = False
    """Whether to generate endpoint Gaussian heatmap targets."""
    heatmap_size: int = 64
    """Square target heatmap size in pixels."""
    sigma: float = 2.0
    """Gaussian standard deviation in heatmap pixels."""
    in_key: str = "tgt"
    """Input key containing endpoint coordinates with shape `(2, 2, 2)`."""
    out_key: str = "heatmap_tgt"
    """Output key where generated endpoint heatmaps are stored."""

    def __post_init__(self):
        msg = f"Expected positive heatmap_size, got {self.heatmap_size}"
        assert self.heatmap_size > 0, msg
        msg = f"Expected positive sigma, got {self.sigma}"
        assert self.sigma > 0.0, msg


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class DecodeRGB(grain.transforms.Map):
    def map(self, element: object) -> object:
        sample = _sample_dct(element)
        img_fpath = sample["img_fpath"]
        msg = f"Expected string image path, got {type(img_fpath)}"
        assert isinstance(img_fpath, str), msg
        # Heavy I/O lives in a transform so workers can parallelize it.
        with Image.open(img_fpath) as im:
            sample["img"] = im.convert("RGB")
        return sample


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class GaussianHeatmap(grain.transforms.Map):
    image_size: int = 256
    """Square input image size in pixels."""
    heatmap_size: int = 64
    """Square target heatmap size in pixels."""
    sigma: float = 2.0
    """Gaussian standard deviation in heatmap pixels."""
    in_key: str = "tgt"
    """Sample key containing endpoint coordinates with shape `(2, 2, 2)`."""
    out_key: str = "heatmap_tgt"
    """Sample key where generated heatmaps with shape `(4, H, W)` are stored."""
    downsample: float = dataclasses.field(init=False, repr=False)
    """Image-to-heatmap stride."""
    x_w: np.ndarray = dataclasses.field(init=False, repr=False)
    """Heatmap x-axis coordinates."""
    y_h: np.ndarray = dataclasses.field(init=False, repr=False)
    """Heatmap y-axis coordinates."""
    gauss_denom: float = dataclasses.field(init=False, repr=False)
    """Gaussian denominator `2 * sigma^2` reused in map."""

    def __post_init__(self):
        msg = (
            "Expected integer downsample ratio between image and heatmap sizes, got "
            f"image_size={self.image_size}, heatmap_size={self.heatmap_size}"
        )
        assert self.image_size % self.heatmap_size == 0, msg
        msg = f"Expected positive sigma, got {self.sigma}"
        assert self.sigma > 0.0, msg
        downsample = float(self.image_size // self.heatmap_size)
        x_w = np.arange(self.heatmap_size, dtype=np.float32)
        y_h = np.arange(self.heatmap_size, dtype=np.float32)
        gauss_denom = float(2.0 * self.sigma**2)
        object.__setattr__(
            self,
            "downsample",
            downsample,
        )
        object.__setattr__(self, "x_w", x_w)
        object.__setattr__(self, "y_h", y_h)
        object.__setattr__(self, "gauss_denom", gauss_denom)

    def map(self, element: object) -> object:
        """Attach endpoint heatmap targets to a sample.

        Args:
            element: Dataset sample dictionary containing transformed endpoint coordinates.

        Returns:
            The same sample dictionary with an additional `out_key` entry containing endpoint heatmaps.

        Notes:
            Coordinates are interpreted as image-space points in the configured `image_size` and are converted to Gaussian heatmaps with UDP alignment.
        """
        sample = _sample_dct(element)
        msg = f"Missing required coordinate key '{self.in_key}'"
        assert self.in_key in sample, msg
        points_l22 = np.asarray(sample[self.in_key], dtype=np.float32)
        msg = f"Expected {self.in_key} shape (2, 2, 2), got {points_l22.shape}"
        assert points_l22.shape == (2, 2, 2), msg
        msg = f"Expected finite {self.in_key}, got {points_l22}"
        assert np.all(np.isfinite(points_l22)), msg

        points_n2 = points_l22.reshape(4, 2)
        hx_n = (points_n2[:, 0] + 0.5) / self.downsample - 0.5
        hy_n = (points_n2[:, 1] + 0.5) / self.downsample - 0.5

        dx_nw = self.x_w[None, :] - hx_n[:, None]
        dy_nh = self.y_h[None, :] - hy_n[:, None]
        dist2_nhw = dy_nh[:, :, None] ** 2 + dx_nw[:, None, :] ** 2
        heatmap_tgt = np.exp(-dist2_nhw / self.gauss_denom).astype(np.float32)
        msg = f"Expected heatmap_tgt shape (4, H, W), got {heatmap_tgt.shape}"
        assert heatmap_tgt.shape == (4, self.heatmap_size, self.heatmap_size), msg
        sample[self.out_key] = heatmap_tgt
        return sample


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Normalize(grain.transforms.Map):
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def map(self, element: object) -> object:
        sample = _sample_dct(element)
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

        img_f32 = np.asarray(img, dtype=np.float32)
        sample["img"] = (img_f32 - mean[None, None, :]) / std[None, None, :]
        return sample


@jaxtyped(typechecker=beartype.beartype)
def get_identity_affine() -> Float[np.ndarray, "3 3"]:
    return np.eye(3, dtype=np.float32)


@jaxtyped(typechecker=beartype.beartype)
def get_crop_resize_affine(
    x0: float, y0: float, crop_w: float, crop_h: float, *, size: int
) -> Float[np.ndarray, "3 3"]:
    msg = f"Expected positive crop size, got crop_w={crop_w}, crop_h={crop_h}"
    assert crop_w > 0.0 and crop_h > 0.0, msg
    sx = size / crop_w
    sy = size / crop_h
    return np.array(
        [
            [sx, 0.0, -sx * x0],
            [0.0, sy, -sy * y0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


@jaxtyped(typechecker=beartype.beartype)
def get_hflip_affine(*, size: int) -> Float[np.ndarray, "3 3"]:
    return np.array(
        [
            [-1.0, 0.0, size - 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


@jaxtyped(typechecker=beartype.beartype)
def get_vflip_affine(*, size: int) -> Float[np.ndarray, "3 3"]:
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, size - 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


@jaxtyped(typechecker=beartype.beartype)
def apply_affine_to_points(
    affine_33: Float[np.ndarray, "3 3"], points_l22: Float[np.ndarray, "lines 2 2"]
) -> Float[np.ndarray, "lines 2 2"]:
    msg = f"Expected affine shape (3, 3), got {affine_33.shape}"
    assert affine_33.shape == (3, 3), msg
    msg = f"Expected points last shape (2, 2), got {points_l22.shape}"
    assert points_l22.shape[1:] == (2, 2), msg

    dt = np.result_type(affine_33.dtype, points_l22.dtype)
    pts = points_l22.reshape(-1, 2).astype(dt, copy=False)
    ones = np.ones((pts.shape[0], 1), dtype=dt)
    hom = np.concatenate([pts, ones], axis=1).T
    out = (affine_33.astype(dt, copy=False) @ hom).T[:, :2]
    return out.reshape(points_l22.shape)


@jaxtyped(typechecker=beartype.beartype)
def is_in_bounds(
    points_l22: Float[np.ndarray, "lines 2 2"], *, size: int
) -> Bool[np.ndarray, "lines 2"]:
    x = points_l22[:, :, 0]
    y = points_l22[:, :, 1]
    return (x >= 0.0) & (x < size) & (y >= 0.0) & (y < size)


@beartype.beartype
def _sample_dct(element: object) -> dict[str, object]:
    msg = f"Expected sample dict, got {type(element)}"
    assert isinstance(element, dict), msg
    return tp.cast(dict[str, object], element)


@beartype.beartype
def _as_img_f32(img: object) -> Float[np.ndarray, "h w c"]:
    if isinstance(img, Image.Image):
        return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0

    msg = f"Expected ndarray or PIL image, got {type(img)}"
    assert isinstance(img, np.ndarray), msg
    assert img.ndim == 3 and img.shape[2] == 3, f"Expected HWC image, got {img.shape}"
    if np.issubdtype(img.dtype, np.integer):
        return np.asarray(img, dtype=np.float32) / 255.0
    return np.asarray(img, dtype=np.float32)


@beartype.beartype
def _resize_img(
    img_hwc: Float[np.ndarray, "h w c"], *, size: int
) -> Float[np.ndarray, "size size c"]:
    arr_u8 = np.clip(np.rint(img_hwc * 255.0), 0.0, 255.0).astype(np.uint8)
    pil = Image.fromarray(arr_u8)
    out = pil.resize((size, size), resample=Image.Resampling.BILINEAR)
    return np.asarray(out, dtype=np.float32) / 255.0


@beartype.beartype
def _compose_affine(
    sample: dict[str, object],
    next_from_prev_33: Float[np.ndarray, "3 3"],
) -> None:
    msg = "Missing running affine key 't_aug_from_orig'"
    assert "t_aug_from_orig" in sample, msg
    t_prev = np.asarray(sample["t_aug_from_orig"], dtype=np.float64)
    assert t_prev.shape == (3, 3), f"Expected affine shape (3, 3), got {t_prev.shape}"
    sample["t_aug_from_orig"] = np.asarray(next_from_prev_33, dtype=np.float64) @ t_prev


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class InitAugState(grain.transforms.Map):
    size: int = 256
    min_px_per_cm: float = 1e-6

    def map(self, element: object) -> object:
        sample = _sample_dct(element)
        sample["img"] = _as_img_f32(sample["img"])

        points_px = np.asarray(sample["points_px"], dtype=np.float32)
        scalebar_px = np.asarray(sample["scalebar_px"], dtype=np.float32)
        loss_mask = np.asarray(sample["loss_mask"], dtype=np.float32)
        assert points_px.shape == (2, 2, 2), (
            f"Expected points_px shape (2, 2, 2), got {points_px.shape}"
        )
        assert scalebar_px.shape == (2, 2), (
            f"Expected scalebar_px shape (2, 2), got {scalebar_px.shape}"
        )
        assert loss_mask.shape == (2,), (
            f"Expected loss_mask shape (2,), got {loss_mask.shape}"
        )
        assert np.all(np.isfinite(points_px)), "points_px must be finite"
        assert np.all(np.isfinite(scalebar_px)), "scalebar_px must be finite"

        sample["points_px"] = points_px
        sample["scalebar_px"] = scalebar_px
        sample["loss_mask"] = loss_mask

        px_per_cm = np.linalg.norm(scalebar_px[1] - scalebar_px[0])
        valid = np.isfinite(px_per_cm) and (px_per_cm > self.min_px_per_cm)
        existing = sample.get("scalebar_valid", True)
        sample["scalebar_valid"] = np.bool_(existing and valid)

        ident = np.eye(3, dtype=np.float64)
        sample["t_aug_from_orig"] = ident
        sample["t_orig_from_aug"] = ident.copy()
        sample["oob_points_frac"] = np.float32(0.0)
        return sample


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Resize(grain.transforms.Map):
    size: int = 256

    def map(self, element: object) -> object:
        sample = _sample_dct(element)
        img = _as_img_f32(sample["img"])
        h, w, _ = img.shape
        sample["img"] = _resize_img(img, size=self.size)
        next_from_prev = get_crop_resize_affine(
            x0=0.0,
            y0=0.0,
            crop_w=float(w),
            crop_h=float(h),
            size=self.size,
        )
        _compose_affine(sample, next_from_prev)
        return sample


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class RandomResizedCrop(grain.transforms.RandomMap):
    cfg: AugmentConfig

    _max_attempts: int = 10

    def random_map(self, element: object, rng: np.random.Generator) -> object:
        sample = _sample_dct(element)
        img = _as_img_f32(sample["img"])
        h, w, _ = img.shape
        area = float(h * w)

        # Resample until the crop fits within the image (following PyTorch convention).
        crop_w = crop_h = 0
        for _ in range(self._max_attempts):
            scale = float(rng.uniform(self.cfg.crop_scale_min, self.cfg.crop_scale_max))
            ratio = float(rng.uniform(self.cfg.crop_ratio_min, self.cfg.crop_ratio_max))
            cw = int(round(np.sqrt(area * scale * ratio)))
            ch = int(round(np.sqrt(area * scale / ratio)))
            if 0 < cw <= w and 0 < ch <= h:
                crop_w, crop_h = cw, ch
                break

        if crop_w == 0:
            # Fallback: largest center crop at geometric mean of ratio range.
            target_ratio = np.sqrt(self.cfg.crop_ratio_min * self.cfg.crop_ratio_max)
            if w / h < target_ratio:
                crop_w, crop_h = w, max(1, int(round(w / target_ratio)))
            else:
                crop_w, crop_h = max(1, int(round(h * target_ratio))), h

        assert 0 < crop_w <= w and 0 < crop_h <= h

        x0 = 0 if crop_w == w else int(rng.integers(0, w - crop_w + 1))
        y0 = 0 if crop_h == h else int(rng.integers(0, h - crop_h + 1))
        crop = img[y0 : y0 + crop_h, x0 : x0 + crop_w]

        sample["img"] = _resize_img(crop, size=self.cfg.size)
        next_from_prev = get_crop_resize_affine(
            x0=float(x0),
            y0=float(y0),
            crop_w=float(crop_w),
            crop_h=float(crop_h),
            size=self.cfg.size,
        )
        _compose_affine(sample, next_from_prev)
        return sample


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class RandomFlip(grain.transforms.RandomMap):
    cfg: AugmentConfig

    def random_map(self, element: object, rng: np.random.Generator) -> object:
        sample = _sample_dct(element)
        img = _as_img_f32(sample["img"])

        if rng.random() < self.cfg.hflip_prob:
            img = np.flip(img, axis=1).copy()
            _compose_affine(sample, get_hflip_affine(size=self.cfg.size))
        if rng.random() < self.cfg.vflip_prob:
            img = np.flip(img, axis=0).copy()
            _compose_affine(sample, get_vflip_affine(size=self.cfg.size))

        sample["img"] = img
        return sample


@jaxtyped(typechecker=beartype.beartype)
def get_rotation_affine(angle_deg: float, *, size: int) -> Float[np.ndarray, "3 3"]:
    """Affine for counterclockwise rotation by angle_deg around the image center."""
    rad = np.deg2rad(angle_deg)
    c, s = float(np.cos(rad)), float(np.sin(rad))
    cx = cy = (size - 1) / 2.0
    return np.array(
        [
            [c, s, cx * (1 - c) - cy * s],
            [-s, c, cx * s + cy * (1 - c)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class RandomRotation(grain.transforms.RandomMap):
    """Uniform random rotation from 0-360 degrees."""

    cfg: AugmentConfig

    def random_map(self, element: object, rng: np.random.Generator) -> object:
        sample = _sample_dct(element)
        img = _as_img_f32(sample["img"])
        if rng.random() < self.cfg.rotation_prob:
            angle = float(rng.uniform(0.0, 360.0))
        else:
            angle = 0.0

        if angle != 0.0:
            arr_u8 = np.clip(np.rint(img * 255.0), 0.0, 255.0).astype(np.uint8)
            pil = Image.fromarray(arr_u8)
            pil = pil.rotate(
                angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0)
            )
            img = np.asarray(pil, dtype=np.float32) / 255.0

        sample["img"] = img
        _compose_affine(sample, get_rotation_affine(angle, size=self.cfg.size))
        return sample


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class ColorJitter(grain.transforms.RandomMap):
    cfg: AugmentConfig

    def random_map(self, element: object, rng: np.random.Generator) -> object:
        sample = _sample_dct(element)
        img = _as_img_f32(sample["img"])
        if self.cfg.color_jitter_prob <= 0.0:
            sample["img"] = img
            return sample
        if (
            self.cfg.brightness == 0.0
            and self.cfg.contrast == 0.0
            and self.cfg.saturation == 0.0
            and self.cfg.hue == 0.0
        ):
            sample["img"] = img
            return sample
        if rng.random() >= self.cfg.color_jitter_prob:
            sample["img"] = img
            return sample

        arr_u8 = np.clip(np.rint(img * 255.0), 0.0, 255.0).astype(np.uint8)
        pil = Image.fromarray(arr_u8)

        if self.cfg.brightness > 0.0:
            low = max(0.0, 1.0 - self.cfg.brightness)
            high = 1.0 + self.cfg.brightness
            pil = ImageEnhance.Brightness(pil).enhance(float(rng.uniform(low, high)))
        if self.cfg.contrast > 0.0:
            low = max(0.0, 1.0 - self.cfg.contrast)
            high = 1.0 + self.cfg.contrast
            pil = ImageEnhance.Contrast(pil).enhance(float(rng.uniform(low, high)))
        if self.cfg.saturation > 0.0:
            low = max(0.0, 1.0 - self.cfg.saturation)
            high = 1.0 + self.cfg.saturation
            pil = ImageEnhance.Color(pil).enhance(float(rng.uniform(low, high)))
        if self.cfg.hue > 0.0:
            delta = float(rng.uniform(-self.cfg.hue, self.cfg.hue))
            hsv = np.asarray(pil.convert("HSV"), dtype=np.uint8).copy()
            hue = hsv[:, :, 0].astype(np.int16)
            hue = (hue + int(round(delta * 255.0))) % 256
            hsv[:, :, 0] = hue.astype(np.uint8)
            h, w, _ = hsv.shape
            hsv_img = Image.frombuffer("HSV", (w, h), hsv.tobytes(), "raw", "HSV", 0, 1)
            pil = hsv_img.convert("RGB")

        sample["img"] = np.asarray(pil, dtype=np.float32) / 255.0
        return sample


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class FinalizeTargets(grain.transforms.Map):
    cfg: AugmentConfig

    def map(self, element: object) -> object:
        sample = _sample_dct(element)
        points_px = np.asarray(sample["points_px"], dtype=np.float32)
        t_aug_from_orig = np.asarray(sample["t_aug_from_orig"], dtype=np.float64)
        assert points_px.shape == (2, 2, 2), (
            f"Expected points_px shape (2, 2, 2), got {points_px.shape}"
        )
        assert t_aug_from_orig.shape == (3, 3), (
            f"Expected affine shape (3, 3), got {t_aug_from_orig.shape}"
        )
        assert np.all(np.isfinite(t_aug_from_orig)), (
            "Non-finite values in t_aug_from_orig"
        )

        t_orig_from_aug = np.linalg.inv(t_aug_from_orig)
        assert np.all(np.isfinite(t_orig_from_aug)), (
            "Non-finite values in t_orig_from_aug"
        )
        ident = t_orig_from_aug @ t_aug_from_orig
        assert np.allclose(ident, np.eye(3, dtype=np.float64), atol=1e-9, rtol=0.0), (
            f"Affine inverse invariant failed: {ident}"
        )

        tgt = apply_affine_to_points(t_aug_from_orig, points_px).astype(
            np.float32, copy=False
        )

        in_bounds_l2 = is_in_bounds(tgt, size=self.cfg.size)
        if self.cfg.oob_policy == "mask_any_oob":
            keep_l = np.all(in_bounds_l2, axis=1)
        elif self.cfg.oob_policy == "mask_all_oob":
            keep_l = np.any(in_bounds_l2, axis=1)
        else:
            keep_l = np.ones(2, dtype=bool)

        loss_mask = np.asarray(sample["loss_mask"], dtype=np.float32)
        assert loss_mask.shape == (2,), (
            f"Expected loss_mask shape (2,), got {loss_mask.shape}"
        )

        sample["loss_mask"] = loss_mask * keep_l.astype(np.float32)
        sample["tgt"] = tgt
        sample["t_aug_from_orig"] = t_aug_from_orig.astype(np.float32, copy=False)
        sample["t_orig_from_aug"] = t_orig_from_aug.astype(np.float32, copy=False)
        sample["oob_points_frac"] = np.float32(1.0 - np.mean(in_bounds_l2))
        return sample


@beartype.beartype
def make_transforms(
    cfg: AugmentConfig,
    *,
    is_train: bool,
    heatmap_tgt_cfg: HeatmapTargetConfig | None = None,
) -> list[grain.transforms.Map | grain.transforms.RandomMap]:
    """Build the Grain transform list for train or eval.

    Args:
        cfg: Augmentation settings, target sizing, and optional normalization controls.
        is_train: Whether to build the train pipeline. If false, build the eval pipeline.
        heatmap_tgt_cfg: Optional Gaussian heatmap target settings. If enabled, appends `GaussianHeatmap` after `FinalizeTargets`.

    Returns:
        Ordered `grain.transforms.Map`/`RandomMap` transforms to apply to each sample.

    The pipeline always starts with `DecodeRGB` and `InitAugState` and always applies `FinalizeTargets` before optional normalization. If heatmap targets are enabled, `GaussianHeatmap` is applied after `FinalizeTargets`. If `is_train` and `cfg.go` are both true, the pipeline includes stochastic spatial/color augmentation (`RandomResizedCrop`, `RandomFlip`, `RandomRotation`, `ColorJitter`). Otherwise it uses deterministic `Resize`.
    """
    if heatmap_tgt_cfg is None:
        heatmap_tgt_cfg = HeatmapTargetConfig()

    tfms: list[grain.transforms.Map | grain.transforms.RandomMap] = [
        DecodeRGB(),
        InitAugState(size=cfg.size, min_px_per_cm=cfg.min_px_per_cm),
    ]
    if not is_train or not cfg.go:
        tfms.append(Resize(size=cfg.size))
        tfms.append(FinalizeTargets(cfg=cfg))
        if heatmap_tgt_cfg.go:
            tfms.append(
                GaussianHeatmap(
                    image_size=cfg.size,
                    heatmap_size=heatmap_tgt_cfg.heatmap_size,
                    sigma=heatmap_tgt_cfg.sigma,
                    in_key=heatmap_tgt_cfg.in_key,
                    out_key=heatmap_tgt_cfg.out_key,
                )
            )
        if cfg.normalize:
            tfms.append(Normalize())
        return tfms

    if cfg.crop:
        tfms.append(RandomResizedCrop(cfg=cfg))
    else:
        tfms.append(Resize(size=cfg.size))
    tfms.extend([
        RandomFlip(cfg=cfg),
        RandomRotation(cfg=cfg),
        ColorJitter(cfg=cfg),
        FinalizeTargets(cfg=cfg),
    ])
    if heatmap_tgt_cfg.go:
        tfms.append(
            GaussianHeatmap(
                image_size=cfg.size,
                heatmap_size=heatmap_tgt_cfg.heatmap_size,
                sigma=heatmap_tgt_cfg.sigma,
                in_key=heatmap_tgt_cfg.in_key,
                out_key=heatmap_tgt_cfg.out_key,
            )
        )
    if cfg.normalize:
        tfms.append(Normalize())
    return tfms
