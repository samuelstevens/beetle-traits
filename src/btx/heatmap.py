"""Heatmap target/loss/decoding utilities for the current line-endpoint task.

Channel contract is intentionally fixed to four endpoint channels:
`[width_p0, width_p1, length_p0, length_p1]`.
If we later switch to a different annotation shape (for example, polylines with
more than two endpoints), update `CHANNEL_NAMES` and the reshape assumptions in
`make_targets`/`heatmaps_to_coords`, plus their tests.
"""

import dataclasses

import beartype
import jax.nn as jnn
import jax.numpy as jnp
from jaxtyping import Array, Float, jaxtyped

CHANNEL_NAMES = ("width_p0", "width_p1", "length_p0", "length_p1")
NEAR_UNIFORM_ENTROPY_FRAC = 0.98


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    """Configuration for heatmap target generation and coordinate decoding."""

    image_size: int = 256
    """Square image size in pixels used by the model input pipeline."""
    heatmap_size: int = 64
    """Square heatmap size in pixels used for supervision and decoding."""
    sigma: float = 2.0
    """Gaussian standard deviation in heatmap pixels."""
    eps: float = 1e-8
    """Small positive constant used for safe normalizations."""

    def __post_init__(self):
        """Validate static heatmap geometry assumptions.

        Args:
            None. Validation uses dataclass fields only.
        """
        msg = f"Expected positive image_size, got {self.image_size}"
        assert self.image_size > 0, msg
        msg = f"Expected positive heatmap_size, got {self.heatmap_size}"
        assert self.heatmap_size > 0, msg
        msg = f"Expected positive sigma, got {self.sigma}"
        assert self.sigma > 0.0, msg
        msg = f"Expected positive eps, got {self.eps}"
        assert self.eps > 0.0, msg
        msg = (
            "Expected integer downsample ratio between image and heatmap sizes, got "
            f"image_size={self.image_size}, heatmap_size={self.heatmap_size}"
        )
        assert self.image_size % self.heatmap_size == 0, msg

    @property
    def downsample(self) -> int:
        """Integer image-to-heatmap stride (`image_size // heatmap_size`)."""
        return self.image_size // self.heatmap_size


@jaxtyped(typechecker=beartype.beartype)
def image_to_heatmap_udp(
    points_b2: Float[Array, "*batch 2"], *, cfg: Config
) -> Float[Array, "*batch 2"]:
    """Map image-space coordinates to heatmap-space coordinates using UDP.

    Args:
        points_b2: Coordinates in image space with trailing `[..., 2]` order `(x, y)`.
        cfg: Heatmap geometry and numerical configuration.

    Returns:
        Coordinates in heatmap space with trailing `[..., 2]` order `(hx, hy)`.
    """
    s = float(cfg.downsample)
    return (points_b2 + 0.5) / s - 0.5


@jaxtyped(typechecker=beartype.beartype)
def heatmap_to_image_udp(
    points_b2: Float[Array, "*batch 2"], *, cfg: Config
) -> Float[Array, "*batch 2"]:
    """Map heatmap-space coordinates back to image-space coordinates using UDP.

    Args:
        points_b2: Coordinates in heatmap space with trailing `[..., 2]` order `(hx, hy)`.
        cfg: Heatmap geometry and numerical configuration.

    Returns:
        Coordinates in image space with trailing `[..., 2]` order `(x, y)`.
    """
    s = float(cfg.downsample)
    return (points_b2 + 0.5) * s - 0.5


@jaxtyped(typechecker=beartype.beartype)
def _get_softargmax_axis(*, cfg: Config) -> Float[Array, " heatmap"]:
    """Build the 1D coordinate axis used for spatial soft-argmax.

    Args:
        cfg: Heatmap geometry and numerical configuration.

    Returns:
        Monotonic coordinate axis with `cfg.heatmap_size` entries spanning
        `[-0.5, heatmap_size - 0.5]`.
    """
    return jnp.linspace(
        -0.5,
        cfg.heatmap_size - 0.5,
        cfg.heatmap_size,
        dtype=jnp.float32,
    )


@jaxtyped(typechecker=beartype.beartype)
def _softargmax_points(
    logits_nhw: Float[Array, "n height width"], *, cfg: Config
) -> Float[Array, "n 2"]:
    """Decode heatmap logits into `(x, y)` points with UDP-compatible soft-argmax.

    Args:
        logits_nhw: Per-keypoint logits with shape `[n_keypoints, H, W]`.
        cfg: Heatmap geometry and numerical configuration.

    Returns:
        Decoded heatmap-space coordinates with shape `[n_keypoints, 2]`.

    """
    msg = (
        "Expected square logits matching heatmap config. "
        f"Got logits {logits_nhw.shape} and heatmap_size {cfg.heatmap_size}."
    )
    assert logits_nhw.ndim == 3, msg
    assert logits_nhw.shape[1:] == (cfg.heatmap_size, cfg.heatmap_size), msg
    n, h, w = logits_nhw.shape
    axis_h = _get_softargmax_axis(cfg=cfg)
    logits_centered_nhw = logits_nhw - jnp.max(logits_nhw, axis=(1, 2), keepdims=True)
    probs_nhw = jnn.softmax(jnp.reshape(logits_centered_nhw, (n, h * w)), axis=1)
    probs_nhw = jnp.reshape(probs_nhw, (n, h, w))
    x_n = jnp.sum(probs_nhw * axis_h[None, None, :], axis=(1, 2))
    y_n = jnp.sum(probs_nhw * axis_h[None, :, None], axis=(1, 2))
    return jnp.stack([x_n, y_n], axis=1)


@jaxtyped(typechecker=beartype.beartype)
def _gaussian_targets(
    points_n2: Float[Array, "n 2"], *, cfg: Config
) -> Float[Array, "n height width"]:
    """Generate unnormalized Gaussian targets centered at UDP heatmap coordinates.

    Args:
        points_n2: Heatmap-space keypoint centers with shape `[n_keypoints, 2]`.
        cfg: Heatmap geometry and numerical configuration.

    Returns:
        Per-keypoint target heatmaps with shape `[n_keypoints, H, W]`.
    """
    msg = f"Expected keypoint centers with shape [n, 2], got {points_n2.shape}."
    assert points_n2.ndim == 2 and points_n2.shape[1] == 2, msg
    h = cfg.heatmap_size
    w = cfg.heatmap_size
    x = jnp.arange(w, dtype=points_n2.dtype)[None, None, :]
    y = jnp.arange(h, dtype=points_n2.dtype)[None, :, None]
    hx = points_n2[:, 0][:, None, None]
    hy = points_n2[:, 1][:, None, None]
    dist2_nhw = (x - hx) ** 2 + (y - hy) ** 2
    denom = 2.0 * (cfg.sigma**2)
    return jnp.exp(-dist2_nhw / denom)


@jaxtyped(typechecker=beartype.beartype)
def _line_loss_permutation_invariant(
    pred_chw: Float[Array, "channels height width"],
    tgt_chw: Float[Array, "channels height width"],
    loss_mask_l: Float[Array, "2"],
    *,
    cfg: Config,
) -> Float[Array, ""]:
    """Compute permutation-invariant masked MSE over width and length endpoint pairs.

    Args:
        pred_chw: Predicted heatmap logits with channel order
            `[width_p0, width_p1, length_p0, length_p1]`.
        tgt_chw: Target Gaussian heatmaps with the same channel order as `pred_chw`.
        loss_mask_l: Per-line supervision mask for `[width, length]`.
        cfg: Heatmap geometry and numerical configuration.

    Returns:
        Scalar masked MSE value.
    """
    n_channels = len(CHANNEL_NAMES)
    msg = f"Expected pred shape ({n_channels}, H, W), got {pred_chw.shape}"
    assert pred_chw.ndim == 3 and pred_chw.shape[0] == n_channels, msg
    msg = f"Expected tgt shape matching pred, got {tgt_chw.shape} vs {pred_chw.shape}"
    assert tgt_chw.shape == pred_chw.shape, msg
    msg = f"Expected loss_mask shape (2,), got {loss_mask_l.shape}"
    assert loss_mask_l.shape == (2,), msg
    sqerr_chw = (pred_chw - tgt_chw) ** 2

    direct_width = jnp.sum(sqerr_chw[0]) + jnp.sum(sqerr_chw[1])
    direct_length = jnp.sum(sqerr_chw[2]) + jnp.sum(sqerr_chw[3])

    swapped_width = jnp.sum((pred_chw[0] - tgt_chw[1]) ** 2) + jnp.sum(
        (pred_chw[1] - tgt_chw[0]) ** 2
    )
    swapped_length = jnp.sum((pred_chw[2] - tgt_chw[3]) ** 2) + jnp.sum(
        (pred_chw[3] - tgt_chw[2]) ** 2
    )

    line_loss_l = jnp.array(
        [
            jnp.minimum(direct_width, swapped_width),
            jnp.minimum(direct_length, swapped_length),
        ],
        dtype=pred_chw.dtype,
    )

    masked_sum = jnp.sum(line_loss_l * loss_mask_l)
    _, h, w = pred_chw.shape
    active = jnp.sum(loss_mask_l) * (2.0 * h * w)
    denom = jnp.maximum(active, cfg.eps)
    return masked_sum / denom


@jaxtyped(typechecker=beartype.beartype)
def _channel_ce_loss(
    pred_hw: Float[Array, "height width"],
    tgt_hw: Float[Array, "height width"],
    *,
    cfg: Config,
) -> Float[Array, ""]:
    """Cross-entropy from target heatmap values to predicted heatmap logits."""
    msg = (
        f"Expected pred and tgt to match shape, got {pred_hw.shape} and {tgt_hw.shape}"
    )
    assert pred_hw.shape == tgt_hw.shape, msg
    tgt_sum = jnp.sum(tgt_hw)
    tgt_prob_hw = tgt_hw / jnp.maximum(tgt_sum, cfg.eps)
    pred_flat = jnp.reshape(pred_hw, (-1,))
    tgt_prob_flat = jnp.reshape(tgt_prob_hw, (-1,))
    return -jnp.sum(tgt_prob_flat * jnn.log_softmax(pred_flat, axis=0))


@jaxtyped(typechecker=beartype.beartype)
def _line_ce_loss_permutation_invariant(
    pred_chw: Float[Array, "channels height width"],
    tgt_chw: Float[Array, "channels height width"],
    loss_mask_l: Float[Array, "2"],
    *,
    cfg: Config,
) -> Float[Array, ""]:
    """Compute permutation-invariant masked CE over width and length endpoint pairs."""
    n_channels = len(CHANNEL_NAMES)
    msg = f"Expected pred shape ({n_channels}, H, W), got {pred_chw.shape}"
    assert pred_chw.ndim == 3 and pred_chw.shape[0] == n_channels, msg
    msg = f"Expected tgt shape matching pred, got {tgt_chw.shape} vs {pred_chw.shape}"
    assert tgt_chw.shape == pred_chw.shape, msg
    msg = f"Expected loss_mask shape (2,), got {loss_mask_l.shape}"
    assert loss_mask_l.shape == (2,), msg

    direct_width = _channel_ce_loss(
        pred_chw[0], tgt_chw[0], cfg=cfg
    ) + _channel_ce_loss(pred_chw[1], tgt_chw[1], cfg=cfg)
    direct_length = _channel_ce_loss(
        pred_chw[2], tgt_chw[2], cfg=cfg
    ) + _channel_ce_loss(pred_chw[3], tgt_chw[3], cfg=cfg)

    swapped_width = _channel_ce_loss(
        pred_chw[0], tgt_chw[1], cfg=cfg
    ) + _channel_ce_loss(pred_chw[1], tgt_chw[0], cfg=cfg)
    swapped_length = _channel_ce_loss(
        pred_chw[2], tgt_chw[3], cfg=cfg
    ) + _channel_ce_loss(pred_chw[3], tgt_chw[2], cfg=cfg)

    line_loss_l = jnp.array(
        [
            jnp.minimum(direct_width, swapped_width),
            jnp.minimum(direct_length, swapped_length),
        ],
        dtype=pred_chw.dtype,
    )
    masked_sum = jnp.sum(line_loss_l * loss_mask_l)
    active = jnp.sum(loss_mask_l)
    denom = jnp.maximum(active, cfg.eps)
    return masked_sum / denom


@jaxtyped(typechecker=beartype.beartype)
def make_targets(
    points_l22: Float[Array, "2 2 2"], *, cfg: Config
) -> Float[Array, "4 height width"]:
    """Build endpoint heatmap targets from line-endpoint coordinates.

    Args:
        points_l22: Line-endpoint coordinates in image space with shape
            `[line, endpoint, (x, y)]`.
        cfg: Heatmap geometry and numerical configuration.

    Returns:
        Endpoint heatmaps with channel order
        `[width_p0, width_p1, length_p0, length_p1]`.
    """
    msg = f"Expected points shape (2, 2, 2), got {points_l22.shape}"
    assert points_l22.shape == (2, 2, 2), msg
    points_n2 = jnp.reshape(points_l22, (4, 2))
    points_hm_n2 = image_to_heatmap_udp(points_n2, cfg=cfg)
    return _gaussian_targets(points_hm_n2, cfg=cfg)


@jaxtyped(typechecker=beartype.beartype)
def heatmap_loss(
    pred_chw: Float[Array, "channels height width"],
    tgt_chw: Float[Array, "channels height width"],
    loss_mask_l: Float[Array, "2"],
    *,
    cfg: Config,
) -> Float[Array, ""]:
    """Compute permutation-invariant masked MSE for endpoint heatmaps.

    Args:
        pred_chw: Predicted heatmap logits with channel order
            `[width_p0, width_p1, length_p0, length_p1]`.
        tgt_chw: Target endpoint heatmaps with the same channel order as `pred_chw`.
        loss_mask_l: Per-line supervision mask for `[width, length]`.
        cfg: Heatmap geometry and numerical configuration.

    Returns:
        Scalar masked MSE value.
    """
    n_channels = len(CHANNEL_NAMES)
    msg = f"Expected pred shape ({n_channels}, H, W), got {pred_chw.shape}"
    assert pred_chw.ndim == 3 and pred_chw.shape[0] == n_channels, msg
    msg = f"Expected tgt shape matching pred, got {tgt_chw.shape} vs {pred_chw.shape}"
    assert tgt_chw.shape == pred_chw.shape, msg
    msg = f"Expected loss_mask shape (2,), got {loss_mask_l.shape}"
    assert loss_mask_l.shape == (2,), msg
    return _line_loss_permutation_invariant(
        pred_chw,
        tgt_chw,
        loss_mask_l,
        cfg=cfg,
    )


@jaxtyped(typechecker=beartype.beartype)
def heatmap_ce_loss(
    pred_chw: Float[Array, "channels height width"],
    tgt_chw: Float[Array, "channels height width"],
    loss_mask_l: Float[Array, "2"],
    *,
    cfg: Config,
) -> Float[Array, ""]:
    """Compute permutation-invariant masked CE for endpoint heatmaps."""
    n_channels = len(CHANNEL_NAMES)
    msg = f"Expected pred shape ({n_channels}, H, W), got {pred_chw.shape}"
    assert pred_chw.ndim == 3 and pred_chw.shape[0] == n_channels, msg
    msg = f"Expected tgt shape matching pred, got {tgt_chw.shape} vs {pred_chw.shape}"
    assert tgt_chw.shape == pred_chw.shape, msg
    msg = f"Expected loss_mask shape (2,), got {loss_mask_l.shape}"
    assert loss_mask_l.shape == (2,), msg
    return _line_ce_loss_permutation_invariant(
        pred_chw,
        tgt_chw,
        loss_mask_l,
        cfg=cfg,
    )


@jaxtyped(typechecker=beartype.beartype)
def heatmaps_to_coords(
    logits_chw: Float[Array, "channels height width"], *, cfg: Config
) -> Float[Array, "2 2 2"]:
    """Decode four endpoint heatmaps into line-endpoint coordinates.

    Args:
        logits_chw: Predicted endpoint logits with channel order
            `[width_p0, width_p1, length_p0, length_p1]`.
        cfg: Heatmap geometry and numerical configuration.

    Returns:
        Coordinates in image space with shape `[2, 2, 2]` and order
        `[line, endpoint, (x, y)]`.
    """
    n_channels = len(CHANNEL_NAMES)
    msg = f"Expected logits shape ({n_channels}, H, W), got {logits_chw.shape}"
    assert logits_chw.ndim == 3 and logits_chw.shape[0] == n_channels, msg
    msg = (
        "Expected logits to match heatmap config size, got "
        f"{logits_chw.shape[1:]} and {cfg.heatmap_size}."
    )
    assert logits_chw.shape[1:] == (cfg.heatmap_size, cfg.heatmap_size), msg
    points_hm_n2 = _softargmax_points(logits_chw, cfg=cfg)
    points_img_n2 = heatmap_to_image_udp(points_hm_n2, cfg=cfg)
    return jnp.reshape(points_img_n2, (2, 2, 2))


@jaxtyped(typechecker=beartype.beartype)
def get_diagnostics(
    logits_bchw: Float[Array, "batch channels height width"], *, cfg: Config
) -> tuple[
    Float[Array, "batch channels"],
    Float[Array, "batch channels"],
    Float[Array, "batch channels"],
]:
    """Compute collapse diagnostics from predicted heatmap logits.

    Args:
        logits_bchw: Predicted heatmap logits with shape `[batch, channels, H, W]`.
        cfg: Heatmap geometry and numerical configuration.

    Returns:
        Tuple `(max_logit_bc, entropy_bc, near_uniform_bc)` where:
        - `max_logit_bc` is max logit per sample/channel.
        - `entropy_bc` is spatial softmax entropy per sample/channel.
        - `near_uniform_bc` is 1.0 when normalized entropy exceeds threshold, else 0.0.
    """
    n_channels = len(CHANNEL_NAMES)
    msg = f"Expected logits shape [batch, {n_channels}, H, W], got {logits_bchw.shape}"
    assert logits_bchw.ndim == 4 and logits_bchw.shape[1] == n_channels, msg
    _, _, h_hm, w_hm = logits_bchw.shape
    msg = f"Expected square heatmaps, got {logits_bchw.shape}"
    assert h_hm == w_hm, msg
    msg = (
        "Expected logits spatial size to match cfg.heatmap_size, got "
        f"{(h_hm, w_hm)} and cfg.heatmap_size={cfg.heatmap_size}"
    )
    assert (h_hm, w_hm) == (cfg.heatmap_size, cfg.heatmap_size), msg

    max_logit_bc = jnp.max(logits_bchw, axis=(2, 3))
    logits_centered = logits_bchw - jnp.max(logits_bchw, axis=(2, 3), keepdims=True)
    probs_bc_hw = jnn.softmax(
        jnp.reshape(logits_centered, (-1, n_channels, h_hm * w_hm)), axis=2
    )
    entropy_bc = -jnp.sum(
        probs_bc_hw * jnp.log(jnp.maximum(probs_bc_hw, cfg.eps)), axis=-1
    )
    max_entropy = jnp.log(jnp.array(h_hm * w_hm, dtype=entropy_bc.dtype))
    entropy_frac_bc = entropy_bc / jnp.maximum(max_entropy, cfg.eps)
    near_uniform_bc = (entropy_frac_bc >= NEAR_UNIFORM_ENTROPY_FRAC).astype(
        entropy_bc.dtype
    )
    return max_logit_bc, entropy_bc, near_uniform_bc
