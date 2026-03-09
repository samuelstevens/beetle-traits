Module btx.modeling.heatmap
===========================

Classes
-------

`Heatmap(dinov3_ckpt: pathlib.Path = PosixPath('models/dinov3_vitb16.eqx'), heatmap_size: int = 64, out_channels: int = 4, deconv1_channels: int = 256, deconv2_channels: int = 128, groupnorm_groups: int = 32)`
:   Configuration for the frozen-DINOv3 + deconvolution heatmap model.

    ### Instance variables

    `deconv1_channels: int`
    :   Output channels for the first transpose-convolution block.

    `deconv2_channels: int`
    :   Output channels for the second transpose-convolution block.

    `dinov3_ckpt: pathlib.Path`
    :   Path to serialized DINOv3 Equinox checkpoint.

    `groupnorm_groups: int`
    :   Number of groups for GroupNorm in each decoder block.

    `heatmap_size: int`
    :   Output heatmap side length in pixels.

    `out_channels: int`
    :   Number of endpoint heatmaps `[width_p0, width_p1, length_p0, length_p1]`.

`Model(cfg: btx.modeling.heatmap.Heatmap, *, key: jax.Array)`
:   Heatmap decoder head on top of a frozen DINOv3 ViT backbone.
    
    Initialize the heatmap decoder and load frozen DINOv3 weights.
    
    Args:
        cfg: Heatmap model configuration.
        key: JAX PRNG key used to initialize decoder layers.

    ### Ancestors (in MRO)

    * equinox._module._module.Module
    * collections.abc.Hashable

    ### Instance variables

    `deconv1: equinox.nn._conv.ConvTranspose2d`
    :   First upsampling block (`embed_dim -> deconv1_channels`, stride 2).

    `deconv2: equinox.nn._conv.ConvTranspose2d`
    :   Second upsampling block (`deconv1_channels -> deconv2_channels`, stride 2).

    `gn1: equinox.nn._normalisation.GroupNorm`
    :   GroupNorm after first upsampling block.

    `gn2: equinox.nn._normalisation.GroupNorm`
    :   GroupNorm after second upsampling block.

    `heatmap_size: int`
    :   Expected side length for decoder output heatmaps.

    `out_conv: equinox.nn._conv.Conv2d`
    :   Final `1x1` projection from decoder channels to endpoint heatmaps.

    `vit: btx.modeling.dinov3.VisionTransformer`
    :   Frozen DINOv3 vision transformer.

    ### Methods

    `extract_features(self, x_hwc: jaxtyping.Float[Array, 'h w c']) ‑> tuple[jaxtyping.Float[Array, 'channels height width'], jaxtyping.Float[Array, 'embed_dim']]`
    :   Forward pass returning both heatmap logits and the CLS embedding.
        
        Args:
            x_hwc: Input image tensor in `HWC` format.
        
        Returns:
            Tuple of (logits_chw, cls_d) where cls_d is the ViT CLS token.