import dataclasses
import pathlib

import beartype
import chex
import equinox as eqx
import jax
from jaxtyping import Array, Float, jaxtyped

from . import dinov3


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Heatmap:
    """Configuration for the frozen-DINOv3 + deconvolution heatmap model."""

    dinov3_ckpt: pathlib.Path = pathlib.Path("models/dinov3_vitb16.eqx")
    """Path to serialized DINOv3 Equinox checkpoint."""
    heatmap_size: int = 64
    """Output heatmap side length in pixels."""
    out_channels: int = 4
    """Number of endpoint heatmaps `[width_p0, width_p1, length_p0, length_p1]`."""
    deconv1_channels: int = 256
    """Output channels for the first transpose-convolution block."""
    deconv2_channels: int = 128
    """Output channels for the second transpose-convolution block."""
    groupnorm_groups: int = 32
    """Number of groups for GroupNorm in each decoder block."""

    def __post_init__(self):
        """Validate static decoder configuration.

        Args:
            None. Validation uses dataclass fields only.
        """
        msg = f"Expected positive heatmap_size, got {self.heatmap_size}"
        assert self.heatmap_size > 0, msg
        msg = f"Expected out_channels=4, got {self.out_channels}"
        assert self.out_channels == 4, msg
        msg = (
            "Expected GroupNorm groups to divide decoder channel counts, got "
            f"groups={self.groupnorm_groups}, deconv1={self.deconv1_channels}, "
            f"deconv2={self.deconv2_channels}"
        )
        assert self.deconv1_channels % self.groupnorm_groups == 0, msg
        assert self.deconv2_channels % self.groupnorm_groups == 0, msg


@jaxtyped(typechecker=beartype.beartype)
class Model(eqx.Module):
    """Heatmap decoder head on top of a frozen DINOv3 ViT backbone."""

    vit: dinov3.VisionTransformer
    """Frozen DINOv3 vision transformer."""
    deconv1: eqx.nn.ConvTranspose2d
    """First upsampling block (`embed_dim -> deconv1_channels`, stride 2)."""
    gn1: eqx.nn.GroupNorm
    """GroupNorm after first upsampling block."""
    deconv2: eqx.nn.ConvTranspose2d
    """Second upsampling block (`deconv1_channels -> deconv2_channels`, stride 2)."""
    gn2: eqx.nn.GroupNorm
    """GroupNorm after second upsampling block."""
    out_conv: eqx.nn.Conv2d
    """Final `1x1` projection from decoder channels to endpoint heatmaps."""

    def __init__(self, cfg: Heatmap, *, key: chex.PRNGKey):
        """Initialize the heatmap decoder and load frozen DINOv3 weights.

        Args:
            cfg: Heatmap model configuration.
            key: JAX PRNG key used to initialize decoder layers.
        """
        k1, k2, k3 = jax.random.split(key, 3)
        self.vit = dinov3.load(cfg.dinov3_ckpt)
        d_model = self.vit.cfg.embed_dim

        self.deconv1 = eqx.nn.ConvTranspose2d(
            in_channels=d_model,
            out_channels=cfg.deconv1_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            key=k1,
        )
        self.gn1 = eqx.nn.GroupNorm(
            groups=cfg.groupnorm_groups, channels=cfg.deconv1_channels
        )
        self.deconv2 = eqx.nn.ConvTranspose2d(
            in_channels=cfg.deconv1_channels,
            out_channels=cfg.deconv2_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            key=k2,
        )
        self.gn2 = eqx.nn.GroupNorm(
            groups=cfg.groupnorm_groups, channels=cfg.deconv2_channels
        )
        self.out_conv = eqx.nn.Conv2d(
            in_channels=cfg.deconv2_channels,
            out_channels=cfg.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            key=k3,
        )

    def __call__(
        self, x_hwc: Float[Array, "h w c"], *, key: chex.PRNGKey | None = None
    ) -> Float[Array, "channels height width"]:
        """Run a forward pass and produce endpoint heatmap logits.

        Args:
            x_hwc: Input image tensor in `HWC` format.
            key: Optional PRNG key for Equinox module compatibility.

        Returns:
            Heatmap logits with channel-first shape `[4, 64, 64]`.

        Notes:
            This is intentionally a skeleton implementation and will be completed
            during the TDD phase.
        """
        _ = x_hwc
        _ = key
        raise NotImplementedError("Heatmap model forward pass is not implemented yet.")
