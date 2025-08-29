Module btx.modeling.dinov3
==========================

Functions
---------

`apply_rope(x: jaxtyping.Float[Array, 'n_heads n d_head'], sin: jaxtyping.Float[Array, 'n d_head'], cos: jaxtyping.Float[Array, 'n d_head']) ‑> jaxtyping.Float[Array, 'n_heads n d_head']`
:   

`dump(model: btx.modeling.dinov3.VisionTransformer, fpath: str | pathlib.Path)`
:   

`load(fpath: str | pathlib.Path) ‑> btx.modeling.dinov3.VisionTransformer`
:   

`rope_fn(q_nhd: jaxtyping.Float[Array, 'n n_heads d_head'], k_nhd: jaxtyping.Float[Array, 'n n_heads d_head'], rope_2pd: jaxtyping.Float[Array, '2 n_pos d_head']) ‑> tuple[jaxtyping.Float[Array, 'n n_heads d_head'], jaxtyping.Float[Array, 'n n_heads d_head']]`
:   

`rope_rotate_half(x_hnd: jaxtyping.Float[Array, 'n_heads n d_head']) ‑> jaxtyping.Float[Array, 'n_heads n d_head']`
:   

Classes
-------

`Config(img_size: int = 224, patch_size: int = 16, in_chans: int = 3, pos_embed_rope_base: float = 100.0, pos_embed_rope_min_period: float | None = None, pos_embed_rope_max_period: float | None = None, pos_embed_rope_normalize_coords: Literal['min', 'max', 'separate'] = 'separate', pos_embed_rope_shift_coords: float | None = None, pos_embed_rope_jitter_coords: float | None = None, pos_embed_rope_rescale_coords: float | None = None, pos_embed_rope_dtype: str = 'bf16', embed_dim: int = 768, depth: int = 12, num_heads: int = 12, ffn_ratio: float = 4.0, qkv_bias: bool = True, drop_path_rate: float = 0.0, layerscale_init: float | None = None, norm_layer: str = 'layernorm', ffn_layer: str = 'mlp', ffn_bias: bool = True, proj_bias: bool = True, n_storage_tokens: int = 0, mask_k_bias: bool = False, untie_cls_and_patch_norms: bool = False, untie_global_and_local_cls_norm: bool = False, device: typing.Any | None = None)`
:   Config(img_size: int = 224, patch_size: int = 16, in_chans: int = 3, pos_embed_rope_base: float = 100.0, pos_embed_rope_min_period: float | None = None, pos_embed_rope_max_period: float | None = None, pos_embed_rope_normalize_coords: Literal['min', 'max', 'separate'] = 'separate', pos_embed_rope_shift_coords: float | None = None, pos_embed_rope_jitter_coords: float | None = None, pos_embed_rope_rescale_coords: float | None = None, pos_embed_rope_dtype: str = 'bf16', embed_dim: int = 768, depth: int = 12, num_heads: int = 12, ffn_ratio: float = 4.0, qkv_bias: bool = True, drop_path_rate: float = 0.0, layerscale_init: float | None = None, norm_layer: str = 'layernorm', ffn_layer: str = 'mlp', ffn_bias: bool = True, proj_bias: bool = True, n_storage_tokens: int = 0, mask_k_bias: bool = False, untie_cls_and_patch_norms: bool = False, untie_global_and_local_cls_norm: bool = False, device: typing.Any | None = None)

    ### Instance variables

    `depth: int`
    :   Number of transformer blocks.

    `device: typing.Any | None`
    :   Device for tensor operations.

    `drop_path_rate: float`
    :   Stochastic depth drop rate.

    `embed_dim: int`
    :   Embedding dimension for transformer.

    `ffn_bias: bool`
    :   Whether to use bias in feed-forward network.

    `ffn_layer: str`
    :   Type of feed-forward network layer.

    `ffn_ratio: float`
    :   Feed-forward network expansion ratio.

    `img_size: int`
    :   Image width and height in pixels.

    `in_chans: int`
    :   Number of input image channels.

    `layerscale_init: float | None`
    :   Initial value for layer scale.

    `mask_k_bias: bool`
    :   Whether to mask K bias in attention.

    `n_storage_tokens: int`
    :   Number of storage/register tokens.

    `norm_layer: str`
    :   Type of normalization layer to use.

    `num_heads: int`
    :   Number of attention heads.

    `patch_size: int`
    :   Size of each patch in pixels.

    `pos_embed_rope_base: float`
    :   Base frequency for RoPE positional encoding.

    `pos_embed_rope_dtype: str`
    :   Data type for RoPE positional encoding.

    `pos_embed_rope_jitter_coords: float | None`
    :   Jitter amount for RoPE coordinates.

    `pos_embed_rope_max_period: float | None`
    :   Maximum period for RoPE positional encoding.

    `pos_embed_rope_min_period: float | None`
    :   Minimum period for RoPE positional encoding.

    `pos_embed_rope_normalize_coords: Literal['min', 'max', 'separate']`
    :   Coordinate normalization method for RoPE encoding.

    `pos_embed_rope_rescale_coords: float | None`
    :   Rescaling factor for RoPE coordinates.

    `pos_embed_rope_shift_coords: float | None`
    :   Shift offset for RoPE coordinates.

    `proj_bias: bool`
    :   Whether to use bias in output projection.

    `qkv_bias: bool`
    :   Whether to use bias in QKV projection.

    `untie_cls_and_patch_norms: bool`
    :   Whether to use separate norms for CLS and patch tokens.

    `untie_global_and_local_cls_norm: bool`
    :   Whether to use separate norms for global and local CLS tokens.

`LayerScale(dim: int, *, key: jax.Array)`
:   LayerScale(dim: int, *, key: jax.Array)

    ### Ancestors (in MRO)

    * equinox._module._module.Module
    * collections.abc.Hashable

    ### Instance variables

    `gamma: jaxtyping.Float[Array, 'dim']`
    :

`LinearKMaskedBias(*args, **kwargs)`
:   LinearKMaskedBias(*args, **kwargs)
    
    **Arguments:**
    
    - `in_features`: The input size. The input to the layer should be a vector of
        shape `(in_features,)`
    - `out_features`: The output size. The output from the layer will be a vector
        of shape `(out_features,)`.
    - `use_bias`: Whether to add on a bias as well.
    - `dtype`: The dtype to use for the weight and the bias in this layer.
        Defaults to either `jax.numpy.float32` or `jax.numpy.float64` depending
        on whether JAX is in 64-bit mode.
    - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
        initialisation. (Keyword only argument.)
    
    Note that `in_features` also supports the string `"scalar"` as a special value.
    In this case the input to the layer should be of shape `()`.
    
    Likewise `out_features` can also be a string `"scalar"`, in which case the
    output from the layer will have shape `()`.

    ### Ancestors (in MRO)

    * equinox.nn._linear.Linear
    * equinox._module._module.Module
    * collections.abc.Hashable

    ### Instance variables

    `bias_mask: jaxtyping.Float[Array, 'd_out'] | None`
    :

    ### Methods

    `forward(self, x: jaxtyping.Float[Array, 'd_in']) ‑> jaxtyping.Float[Array, 'd_out']`
    :

`Mlp(in_features: int, hidden_features: int | None, out_features: int | None, act_fn: str, *, key: jax.Array)`
:   Mlp(in_features: int, hidden_features: int | None, out_features: int | None, act_fn: str, *, key: jax.Array)

    ### Ancestors (in MRO)

    * equinox._module._module.Module
    * collections.abc.Hashable

    ### Instance variables

    `act: <class 'collections.abc.Callable'>`
    :

    `fc1: equinox.nn._linear.Linear`
    :

    `fc2: equinox.nn._linear.Linear`
    :

    `hidden_features: int`
    :

    `in_features: int`
    :

    `out_features: int`
    :

`PatchEmbed(img_size: int, patch_size: int, in_chans: int, embed_dim: int, key: jax.Array)`
:   2D image to patch embedding: (C,H,W) -> (N,D)
    
    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.

    ### Ancestors (in MRO)

    * equinox._module._module.Module
    * collections.abc.Hashable

    ### Instance variables

    `embed_dim: int`
    :

    `img_size: tuple[int, int]`
    :

    `in_chans: int`
    :

    `patch_size: tuple[int, int]`
    :

    `proj: equinox.nn._conv.Conv2d`
    :

`RopePositionEmbedding(embed_dim: int, *, num_heads: int, base: float | None, min_period: float | None, max_period: float | None, normalize_coords: Literal['min', 'max', 'separate'], dtype: numpy.dtype)`
:   RoPE positional embedding with no mixing of coordinates (axial) and no learnable weights. Supports two parametrizations of the rope parameters: either using `base` or `min_period` and `max_period`.

    ### Ancestors (in MRO)

    * equinox._module._module.Module
    * collections.abc.Hashable

    ### Instance variables

    `base: float | None`
    :

    `d_head: int`
    :

    `dtype: numpy.dtype`
    :

    `max_period: float | None`
    :

    `min_period: float | None`
    :

    `normalize_coords: Literal['min', 'max', 'separate']`
    :

    `periods: jaxtyping.Float[Array, 'd_period']`
    :

`SelfAttention(cfg: btx.modeling.dinov3.Config, *, key: jax.Array)`
:   SelfAttention(cfg: btx.modeling.dinov3.Config, *, key: jax.Array)

    ### Ancestors (in MRO)

    * equinox._module._module.Module
    * collections.abc.Hashable

    ### Instance variables

    `cfg: btx.modeling.dinov3.Config`
    :

    `proj: equinox.nn._linear.Linear`
    :

    `qkv: equinox.nn._linear.Linear | btx.modeling.dinov3.LinearKMaskedBias`
    :

    `scale: float`
    :

`SelfAttentionBlock(cfg: btx.modeling.dinov3.Config, key: jax.Array)`
:   SelfAttentionBlock(cfg: btx.modeling.dinov3.Config, key: jax.Array)

    ### Ancestors (in MRO)

    * equinox._module._module.Module
    * collections.abc.Hashable

    ### Instance variables

    `attn: btx.modeling.dinov3.SelfAttention`
    :

    `cfg: btx.modeling.dinov3.Config`
    :

    `ls1: btx.modeling.dinov3.LayerScale`
    :

    `ls2: btx.modeling.dinov3.LayerScale`
    :

    `mlp: btx.modeling.dinov3.Mlp`
    :

    `norm1: equinox._module._module.Module`
    :

    `norm2: equinox._module._module.Module`
    :

`VisionTransformer(cfg: btx.modeling.dinov3.Config, key: jax.Array)`
:   VisionTransformer(cfg: btx.modeling.dinov3.Config, key: jax.Array)

    ### Ancestors (in MRO)

    * equinox._module._module.Module
    * collections.abc.Hashable

    ### Instance variables

    `blocks: list[btx.modeling.dinov3.SelfAttentionBlock]`
    :

    `cfg: btx.modeling.dinov3.Config`
    :

    `cls_token: jaxtyping.Float[Array, 'dim']`
    :

    `mask_token: jaxtyping.Float[Array, 'dim']`
    :

    `norm: equinox._module._module.Module`
    :

    `patch_embed: btx.modeling.dinov3.PatchEmbed`
    :

    `rope_embed: btx.modeling.dinov3.RopePositionEmbedding`
    :

    `storage_tokens: jaxtyping.Float[Array, 'n_storage dim']`
    :