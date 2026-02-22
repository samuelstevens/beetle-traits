import dataclasses
import pathlib

import beartype
import chex
import einops
import equinox as eqx
import jax
import jax.nn as jnn
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
    heatmap_size: int
    """Expected side length for decoder output heatmaps."""

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
        self.heatmap_size = cfg.heatmap_size

    def __call__(
        self, x_hwc: Float[Array, "h w c"], *, key: chex.PRNGKey | None = None
    ) -> Float[Array, "channels height width"]:
        """Run a forward pass and produce endpoint heatmap logits.

        Args:
            x_hwc: Input image tensor in `HWC` format.
            key: Optional PRNG key for Equinox module compatibility.

        Returns:
            Heatmap logits with channel-first shape `[4, 64, 64]`.
        """
        msg = f"Expected image shape (H, W, 3), got {x_hwc.shape}"
        assert x_hwc.ndim == 3 and x_hwc.shape[2] == 3, msg
        h_img, w_img, _ = x_hwc.shape
        patch_size = self.vit.cfg.patch_size
        msg = (
            "Expected image spatial size to be divisible by ViT patch_size, got "
            f"({h_img}, {w_img}) and patch_size={patch_size}"
        )
        assert h_img % patch_size == 0 and w_img % patch_size == 0, msg
        _ = key

        x_chw = einops.rearrange(x_hwc, "h w c -> c h w")
        vit_out = self.vit(x_chw)
        patch_nd = vit_out["patches"]
        msg = f"Expected patch tokens shape (N, D), got {patch_nd.shape}"
        assert patch_nd.ndim == 2, msg

        grid_h = h_img // patch_size
        grid_w = w_img // patch_size
        expected_n_patches = grid_h * grid_w
        n_patches = patch_nd.shape[0]
        msg = (
            "Expected patch count implied by ViT config, got "
            f"n_patches={n_patches} and expected_n_patches={expected_n_patches}"
        )
        assert n_patches == expected_n_patches, msg
        msg = f"Expected square patch grid, got ({grid_h}, {grid_w})"
        assert grid_h == grid_w, msg

        feat_dhw = einops.rearrange(patch_nd, "(h w) d -> d h w", h=grid_h, w=grid_w)
        x = self.deconv1(feat_dhw)
        x = jnn.relu(self.gn1(x))
        x = self.deconv2(x)
        x = jnn.relu(self.gn2(x))
        logits_chw = self.out_conv(x)

        msg = (
            "Expected logits shape "
            f"({self.out_conv.out_channels}, {self.heatmap_size}, {self.heatmap_size}), "
            f"got {logits_chw.shape}"
        )
        assert logits_chw.shape == (
            self.out_conv.out_channels,
            self.heatmap_size,
            self.heatmap_size,
        ), msg
        return logits_chw

    def extract_features(
        self, x_hwc: Float[Array, "h w c"]
    ) -> tuple[Float[Array, "channels height width"], Float[Array, " embed_dim"]]:
        """Forward pass returning both heatmap logits and the CLS embedding.

        Args:
            x_hwc: Input image tensor in `HWC` format.

        Returns:
            Tuple of (logits_chw, cls_d) where cls_d is the ViT CLS token.
        """
        msg = f"Expected image shape (H, W, 3), got {x_hwc.shape}"
        assert x_hwc.ndim == 3 and x_hwc.shape[2] == 3, msg
        h_img, w_img, _ = x_hwc.shape
        patch_size = self.vit.cfg.patch_size
        msg = f"Expected image spatial size divisible by patch_size={patch_size}, got ({h_img}, {w_img})"
        assert h_img % patch_size == 0 and w_img % patch_size == 0, msg

        x_chw = einops.rearrange(x_hwc, "h w c -> c h w")
        vit_out = self.vit(x_chw)
        cls_d = vit_out["cls"]
        patch_nd = vit_out["patches"]

        grid_h = h_img // patch_size
        grid_w = w_img // patch_size
        feat_dhw = einops.rearrange(patch_nd, "(h w) d -> d h w", h=grid_h, w=grid_w)
        x = self.deconv1(feat_dhw)
        x = jnn.relu(self.gn1(x))
        x = self.deconv2(x)
        x = jnn.relu(self.gn2(x))
        logits_chw = self.out_conv(x)

        return logits_chw, cls_d
