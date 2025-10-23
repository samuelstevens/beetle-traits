# /src/btx/modeling/dinov3.py
import dataclasses
import functools
import json
import math
import pathlib
import typing as tp
from collections.abc import Callable

import beartype
import chex
import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PyTree, jaxtyped


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    img_size: int = 224
    """Image width and height in pixels."""
    patch_size: int = 16
    """Size of each patch in pixels."""
    in_chans: int = 3
    """Number of input image channels."""
    pos_embed_rope_base: float = 100.0
    """Base frequency for RoPE positional encoding."""
    pos_embed_rope_min_period: float | None = None
    """Minimum period for RoPE positional encoding."""
    pos_embed_rope_max_period: float | None = None
    """Maximum period for RoPE positional encoding."""
    pos_embed_rope_normalize_coords: tp.Literal["min", "max", "separate"] = "separate"
    """Coordinate normalization method for RoPE encoding."""
    pos_embed_rope_shift_coords: float | None = None
    """Shift offset for RoPE coordinates."""
    pos_embed_rope_jitter_coords: float | None = None
    """Jitter amount for RoPE coordinates."""
    pos_embed_rope_rescale_coords: float | None = None
    """Rescaling factor for RoPE coordinates."""
    pos_embed_rope_dtype: str = "bf16"
    """Data type for RoPE positional encoding."""
    embed_dim: int = 768
    """Embedding dimension for transformer."""
    depth: int = 12
    """Number of transformer blocks."""
    num_heads: int = 12
    """Number of attention heads."""
    ffn_ratio: float = 4.0
    """Feed-forward network expansion ratio."""
    qkv_bias: bool = True
    """Whether to use bias in QKV projection."""
    drop_path_rate: float = 0.0
    """Stochastic depth drop rate."""
    layerscale_init: float | None = None
    """Initial value for layer scale."""
    norm_layer: str = "layernorm"
    """Type of normalization layer to use."""
    ffn_layer: str = "mlp"
    """Type of feed-forward network layer."""
    ffn_bias: bool = True
    """Whether to use bias in feed-forward network."""
    proj_bias: bool = True
    """Whether to use bias in output projection."""
    n_storage_tokens: int = 0
    """Number of storage/register tokens."""
    mask_k_bias: bool = False
    """Whether to mask K bias in attention."""
    untie_cls_and_patch_norms: bool = False
    """Whether to use separate norms for CLS and patch tokens."""
    untie_global_and_local_cls_norm: bool = False
    """Whether to use separate norms for global and local CLS tokens."""
    device: tp.Any | None = None
    """Device for tensor operations."""


_norm_layer_lookup = {
    "layernorm": functools.partial(eqx.nn.LayerNorm, eps=1e-6),
    "layernormbf16": functools.partial(eqx.nn.LayerNorm, eps=1e-5),
}

_dtype_lookup = {
    "fp32": jnp.dtype("float32"),
}

_act_fn_lookup = {"gelu": functools.partial(jax.nn.gelu, approximate=False)}


@jaxtyped(typechecker=beartype.beartype)
def rope_rotate_half(
    x_hnd: Float[Array, "n_heads n d_head"],
) -> Float[Array, "n_heads n d_head"]:
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x3 -x4 -x5  x0  x1  x2]
    x1, x2 = jnp.split(x_hnd, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)


@jaxtyped(typechecker=beartype.beartype)
def apply_rope(
    x: Float[Array, "n_heads n d_head"],
    sin: Float[Array, "n d_head"],
    cos: Float[Array, "n d_head"],
) -> Float[Array, "n_heads n d_head"]:
    # x:   [..., D], eg [x0,     x1,   x2,   x3,   x4,   x5]
    # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
    # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2]
    return (x * cos) + (rope_rotate_half(x) * sin)


@jaxtyped(typechecker=beartype.beartype)
def rope_fn(
    q_nhd: Float[Array, "n n_heads d_head"],
    k_nhd: Float[Array, "n n_heads d_head"],
    rope_2pd: Float[Array, "2 n_pos d_head"],
) -> tuple[Float[Array, "n n_heads d_head"], Float[Array, "n n_heads d_head"]]:
    sin_pd, cos_pd = rope_2pd

    n, n_heads, d_head = q_nhd.shape
    n_pos, d_head = sin_pd.shape

    prefix = n - n_pos
    assert prefix >= 0, f"Got {n} residual streams but only {n_pos} patches."

    q_prefix_hd = q_nhd[:prefix]
    q_hpd = einops.rearrange(
        q_nhd[prefix:], "n_pos n_heads d_head -> n_heads n_pos d_head"
    )
    q_phd = einops.rearrange(
        apply_rope(q_hpd, sin_pd, cos_pd),
        "n_heads n_pos d_head -> n_pos n_heads d_head",
    )
    q_nhd = jnp.concatenate((q_prefix_hd, q_phd), axis=0)
    k_prefix_hd = k_nhd[:prefix]
    k_hpd = einops.rearrange(
        k_nhd[prefix:], "n_pos n_heads d_head -> n_heads n_pos d_head"
    )
    k_phd = einops.rearrange(
        apply_rope(k_hpd, sin_pd, cos_pd),
        "n_heads n_pos d_head -> n_pos n_heads d_head",
    )
    k_nhd = jnp.concatenate((k_prefix_hd, k_phd), axis=0)
    return q_nhd, k_nhd


@jaxtyped(typechecker=beartype.beartype)
class PatchEmbed(eqx.Module):
    """
    2D image to patch embedding: (C,H,W) -> (N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
    """

    img_size: tuple[int, int]
    patch_size: tuple[int, int]
    in_chans: int
    embed_dim: int
    proj: eqx.nn.Conv2d

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        embed_dim: int,
        key: chex.PRNGKey,
    ) -> None:
        image_HW = (img_size, img_size)
        patch_HW = (patch_size, patch_size)

        self.img_size = image_HW
        self.patch_size = patch_HW

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = eqx.nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW, key=key
        )

    def __call__(
        self, x_chw: Float[Array, "channnels height width"]
    ) -> Float[Array, "h w dim"]:
        _, H, W = x_chw.shape
        # patch_H, patch_W = self.patch_size
        # assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        # assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x_dhw = self.proj(x_chw)
        _, h, w = x_dhw.shape
        x_nd = einops.rearrange(x_dhw, "d h w -> (h w) d")
        x = einops.rearrange(x_nd, "(h w) d -> h w d", h=h, w=w)
        return x


@jaxtyped(typechecker=beartype.beartype)
class LayerScale(eqx.Module):
    gamma: Float[Array, " dim"]

    def __init__(self, dim: int, *, key: chex.PRNGKey):
        # TODO: initialize with key
        self.gamma = jnp.zeros(dim)

    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        return x * self.gamma


@jaxtyped(typechecker=beartype.beartype)
class RopePositionEmbedding(eqx.Module):
    """RoPE positional embedding with no mixing of coordinates (axial) and no learnable weights. Supports two parametrizations of the rope parameters: either using `base` or `min_period` and `max_period`."""

    d_head: int
    base: float | None
    min_period: float | None
    max_period: float | None
    normalize_coords: tp.Literal["min", "max", "separate"]
    dtype: jnp.dtype

    periods: Float[Array, " d_period"]

    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None,
        min_period: float | None,
        max_period: float | None,
        normalize_coords: tp.Literal["min", "max", "separate"],
        dtype: jnp.dtype,
    ):
        assert embed_dim % (4 * num_heads) == 0
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError(
                "Either `base` or `min_period`+`max_period` must be provided."
            )

        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.d_head = embed_dim // num_heads
        self.normalize_coords = normalize_coords
        self.dtype = dtype

        if self.base is not None:
            periods = self.base ** (
                2 * jnp.arange(self.d_head // 4, dtype=dtype) / (self.d_head // 2)
            )
        else:
            base = self.max_period / self.min_period
            exponents = jnp.linspace(0, 1, self.d_head // 4, dtype=dtype)
            periods = base**exponents  # range [1, max_period / min_period]
            periods = periods / base  # range [min_period / max_period, 1]
            periods = periods * self.max_period  # range [min_period, max_period]

        self.periods = periods

    def __call__(
        self, *, h: int, w: int
    ) -> tuple[Float[Array, "{h*w} d_head"], Float[Array, "{h*w} d_head"]]:
        # Prepare coords in range [-1, +1]
        if self.normalize_coords == "max":
            max_hw = max(h, w)
            coords_h = jnp.arange(0.5, h, dtype=self.dtype) / max_hw  # [H]
            coords_w = jnp.arange(0.5, w, dtype=self.dtype) / max_hw  # [W]
        elif self.normalize_coords == "min":
            min_hw = min(h, w)
            coords_h = jnp.arange(0.5, h, dtype=self.dtype) / min_hw  # [H]
            coords_w = jnp.arange(0.5, w, dtype=self.dtype) / min_hw  # [W]
        elif self.normalize_coords == "separate":
            coords_h = jnp.arange(0.5, h, dtype=self.dtype) / h  # [H]
            coords_w = jnp.arange(0.5, w, dtype=self.dtype) / w  # [W]
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")

        coords_hw2 = jnp.stack(jnp.meshgrid(coords_h, coords_w, indexing="ij"), axis=-1)
        coords_n2 = einops.rearrange(
            coords_hw2, "height width ndim -> (height width) ndim"
        )
        coords_n2 = 2.0 * coords_n2 - 1.0  # Shift range [0, 1] to [-1, +1]

        # Prepare angles and sin/cos
        angles = (
            2 * math.pi * coords_n2[:, :, None] / self.periods[None, None, :]
        )  # [HW, 2, D//4]
        angles = einops.rearrange(
            angles, "n ndim subhead -> n (ndim subhead)"
        )  # [HW, D//2]
        angles = jnp.tile(angles, (1, 2))  # [HW, D]
        cos = jnp.cos(angles)  # [HW, D]
        sin = jnp.sin(angles)  # [HW, D]

        return (sin, cos)  # 2 * [HW, D]


@beartype.beartype
class LinearKMaskedBias(eqx.nn.Linear):
    bias_mask: Float[Array, " d_out"] | None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.out_features % 3 == 0

        if self.bias is not None:
            # This is a bug: https://github.com/facebookresearch/dinov3/issues/58#issuecomment-3204669831. But since it doesn't affect the released models (they all have 0 in their bias_mask), we don't worry about it.
            self.bias_mask = jnp.zeros_like(self.bias)

    def forward(self, x: Float[Array, " d_in"]) -> Float[Array, " d_out"]:
        x = self.weight @ x
        if self.bias is not None:
            masked_bias = self.bias * self.bias_mask.to(self.bias.dtype)
            x = x + masked_bias
        return x


@beartype.beartype
class Mlp(eqx.Module):
    in_features: int
    hidden_features: int
    out_features: int
    fc1: eqx.nn.Linear
    act: Callable
    fc2: eqx.nn.Linear

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None,
        out_features: int | None,
        act_fn: str,
        *,
        key: chex.PRNGKey,
    ):
        self.in_features = in_features
        self.hidden_features = hidden_features or in_features
        self.out_features = out_features or in_features

        k1, k2 = jax.random.split(key, 2)
        self.fc1 = eqx.nn.Linear(
            self.in_features, self.hidden_features, use_bias=True, key=k1
        )
        self.fc2 = eqx.nn.Linear(
            self.hidden_features, self.in_features, use_bias=True, key=k2
        )
        self.act = _act_fn_lookup[act_fn]

    def __call__(self, x: Float[Array, " d"]) -> Float[Array, " d"]:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


@beartype.beartype
class SelfAttention(eqx.Module):
    cfg: Config
    scale: float
    qkv: eqx.nn.Linear | LinearKMaskedBias
    proj: eqx.nn.Linear

    def __init__(self, cfg: Config, *, key: chex.PRNGKey):
        self.cfg = cfg
        head_dim = cfg.embed_dim // cfg.num_heads
        self.scale = head_dim**-0.5

        k1, k2 = jax.random.split(key, 2)

        linear_class = LinearKMaskedBias if cfg.mask_k_bias else eqx.nn.Linear
        self.qkv = linear_class(
            cfg.embed_dim, cfg.embed_dim * 3, use_bias=cfg.qkv_bias, key=k1
        )
        self.proj = eqx.nn.Linear(
            cfg.embed_dim, cfg.embed_dim, use_bias=cfg.proj_bias, key=k2
        )

    def __call__(
        self, x_nd: Float[Array, "n d"], rope: Float[Array, "2 n_pos d_head"] | None
    ) -> Float[Array, "n d"]:
        n_tok, d = x_nd.shape

        qkv_3nd = einops.rearrange(
            jax.vmap(self.qkv)(x_nd),
            "n_tok (parts d) -> parts n_tok d",
            parts=3,  # [q, k, v] = 3 parts
            d=d,
        )
        qkv_3nhd = einops.rearrange(
            qkv_3nd,
            "parts n_tok (n_heads d_head) -> parts n_tok n_heads d_head",
            parts=3,
            n_heads=self.cfg.num_heads,
            d_head=d // self.cfg.num_heads,
        )
        q_nhd, k_nhd, v_nhd = jnp.unstack(qkv_3nhd, axis=0)
        if rope is not None:
            q_nhd, k_nhd = rope_fn(q_nhd, k_nhd, rope)
        x_nhd = jax.nn.dot_product_attention(q_nhd, k_nhd, v_nhd)

        x_nd = einops.rearrange(x_nhd, "n_tok n_heads d_head -> n_tok (n_heads d_head)")
        x_nd = jax.vmap(self.proj)(x_nd)
        return x_nd


@beartype.beartype
class SelfAttentionBlock(eqx.Module):
    cfg: Config
    ls1: LayerScale
    norm1: eqx.Module
    norm2: eqx.Module
    attn: SelfAttention
    mlp: Mlp
    ls2: LayerScale

    def __init__(self, cfg: Config, key: chex.PRNGKey):
        self.cfg = cfg
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.ls1 = LayerScale(cfg.embed_dim, key=k1)
        self.norm1 = _norm_layer_lookup[cfg.norm_layer](cfg.embed_dim)
        self.norm2 = _norm_layer_lookup[cfg.norm_layer](cfg.embed_dim)
        self.attn = SelfAttention(cfg, key=k2)
        ffn_hidden_dim = int(cfg.embed_dim * cfg.ffn_ratio)
        self.mlp = Mlp(cfg.embed_dim, ffn_hidden_dim, cfg.embed_dim, "gelu", key=k3)
        self.ls2 = LayerScale(cfg.embed_dim, key=k4)

    def __call__(
        self, x_nd: Float[Array, "n d"], rope: Float[Array, "2 n_pos d_head"]
    ) -> Float[Array, "n d"]:
        x_attn_nd = x_nd + self.ls1(self.attn(jax.vmap(self.norm1)(x_nd), rope=rope))
        x_ffn_nd = x_attn_nd + self.ls2(
            jax.vmap(self.mlp)(jax.vmap(self.norm2)(x_attn_nd))
        )
        return x_ffn_nd


# @beartype.beartype
# class VisionTransformer(eqx.Module):
#     cfg: Config
#     cls_token: Float[Array, " dim"]
#     storage_tokens: Float[Array, "n_storage dim"]
#     mask_token: Float[Array, " dim"]
#     patch_embed: PatchEmbed
#     rope_embed: RopePositionEmbedding
#     blocks: list[SelfAttentionBlock]
#     norm: eqx.Module

#     def __init__(self, cfg: Config, key: chex.PRNGKey):
#         self.cfg = cfg

#         assert not self.cfg.untie_cls_and_patch_norms, "Not supported"

#         k1, *keys = jax.random.split(key, (2 + cfg.depth))

#         self.cls_token = jnp.zeros((cfg.embed_dim,))
#         self.storage_tokens = jnp.zeros((cfg.n_storage_tokens, cfg.embed_dim))
#         self.mask_token = jnp.zeros((cfg.embed_dim,))

#         self.patch_embed = PatchEmbed(
#             cfg.img_size,
#             cfg.patch_size,
#             cfg.in_chans,
#             cfg.embed_dim,
#             k1,
#         )
#         self.rope_embed = RopePositionEmbedding(
#             cfg.embed_dim,
#             num_heads=cfg.num_heads,
#             base=cfg.pos_embed_rope_base,
#             min_period=cfg.pos_embed_rope_min_period,
#             max_period=cfg.pos_embed_rope_max_period,
#             normalize_coords=cfg.pos_embed_rope_normalize_coords,
#             dtype=_dtype_lookup[cfg.pos_embed_rope_dtype],
#         )
#         self.blocks = [SelfAttentionBlock(cfg, k) for k in keys]
#         self.norm = _norm_layer_lookup[cfg.norm_layer](cfg.embed_dim)

#     def __call__(self, x: Float[Array, "..."]):
#         x_hwd = self.patch_embed(x)
#         h, w, _ = x_hwd.shape
#         x_nd = einops.rearrange(x_hwd, "height width dim -> (height width) dim")
#         cls_1d = self.cls_token[None, :] + 0 * self.mask_token
#         storage_tokens_md = self.storage_tokens

#         x_nd = jnp.concatenate([cls_1d, storage_tokens_md, x_nd], axis=0)

#         rope_sincos = self.rope_embed(h=h, w=w)
#         rope_2nd = jnp.stack(rope_sincos, axis=0)
#         for block in self.blocks:
#             x_nd = block(x_nd, rope_2nd)

#         if self.cfg.untie_global_and_local_cls_norm:
#             x_cls_reg_nd = x_nd[: self.cfg.n_storage_tokens + 1]
#             x_norm_cls_reg = jax.vmap(self.norm)(x_cls_reg_nd)
#         else:
#             x_norm_nd = jax.vmap(self.norm)(x_nd)
#             x_norm_cls_reg = x_norm_nd[: self.cfg.n_storage_tokens + 1]

#         return x_norm_cls_reg[0]

@beartype.beartype
class VisionTransformer(eqx.Module):
    cfg: Config
    cls_token: Float[Array, " dim"]
    storage_tokens: Float[Array, "n_storage dim"]
    mask_token: Float[Array, " dim"]
    patch_embed: PatchEmbed
    rope_embed: RopePositionEmbedding
    blocks: list[SelfAttentionBlock]
    norm: eqx.Module

    def __init__(self, cfg: Config, key: chex.PRNGKey):
        self.cfg = cfg
        assert not self.cfg.untie_cls_and_patch_norms, "Not supported"

        k1, *keys = jax.random.split(key, (2 + cfg.depth))

        self.cls_token = jnp.zeros((cfg.embed_dim,))
        self.storage_tokens = jnp.zeros((cfg.n_storage_tokens, cfg.embed_dim))
        self.mask_token = jnp.zeros((cfg.embed_dim,))

        self.patch_embed = PatchEmbed(
            cfg.img_size, cfg.patch_size, cfg.in_chans, cfg.embed_dim, k1
        )
        self.rope_embed = RopePositionEmbedding(
            cfg.embed_dim,
            num_heads=cfg.num_heads,
            base=cfg.pos_embed_rope_base,
            min_period=cfg.pos_embed_rope_min_period,
            max_period=cfg.pos_embed_rope_max_period,
            normalize_coords=cfg.pos_embed_rope_normalize_coords,
            dtype=_dtype_lookup[cfg.pos_embed_rope_dtype],
        )
        self.blocks = [SelfAttentionBlock(cfg, k) for k in keys]
        self.norm = _norm_layer_lookup[cfg.norm_layer](cfg.embed_dim)

    # NEW: expose both CLS and patch tokens
    def tokens(self, x: Float[Array, "..."]):
        """
        Returns:
          cls_token: Float[Array, "dim"]
          patch_tokens: Float[Array, " (Hp*Wp) dim"]
          grid_hw: (Hp, Wp) spatial grid size of patches
        """
        # Patchify
        x_hwd = self.patch_embed(x)                 # (Hp, Wp, D) or (H, W, D) as implemented
        h, w, _ = x_hwd.shape
        x_nd = einops.rearrange(x_hwd, "h w d -> (h w) d")  # (N_patches, D)

        # Prep sequence: [CLS] + [storage tokens] + [patch tokens]
        cls_1d = self.cls_token[None, :] + 0 * self.mask_token
        x_nd = jnp.concatenate([cls_1d, self.storage_tokens, x_nd], axis=0)  # (1+S+N, D)

        # RoPE + blocks
        rope_2nd = jnp.stack(self.rope_embed(h=h, w=w), axis=0)
        for block in self.blocks:
            x_nd = block(x_nd, rope_2nd)

        # IMPORTANT: normalize the WHOLE sequence so we can return normalized patches
        x_norm_nd = jax.vmap(self.norm)(x_nd)  # (1+S+N, D)

        cls_token = x_norm_nd[0]
        patch_tokens = x_norm_nd[self.cfg.n_storage_tokens + 1 :]  # (N_patches, D) == (h*w, D)
        return cls_token, patch_tokens, (h, w)

    # Keep old API: return only CLS token (for the "frozen" model path)
    def __call__(self, x: Float[Array, "..."]):
        cls_token, _patch_tokens, _grid = self.tokens(x)
        return cls_token


@beartype.beartype
def load(fpath: str | pathlib.Path) -> VisionTransformer:
    with open(fpath, "rb") as fd:
        cfg_dict = json.loads(fd.readline())
        cfg = Config(**cfg_dict)
        model = VisionTransformer(cfg, key=jax.random.key(seed=0))
        return eqx.tree_deserialise_leaves(fd, model)


@beartype.beartype
def dump(model: VisionTransformer, fpath: str | pathlib.Path):
    with open(fpath, "wb") as fd:
        cfg_str = json.dumps(dataclasses.asdict(model.cfg))
        fd.write((cfg_str + "\n").encode("utf-8"))
        eqx.tree_serialise_leaves(fd, model)


_PRETRAINED_CFGS = {
    "dinov3_vits16": Config(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100.0,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2.0,
        pos_embed_rope_dtype="fp32",
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
    ),
    "dinov3_vits16plus": Config(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100.0,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2.0,
        pos_embed_rope_dtype="fp32",
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=6.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="swiglu",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
    ),
    "dinov3_vitb16": Config(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100.0,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2.0,
        pos_embed_rope_dtype="fp32",
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
    ),
    "dinov3_vitl16": Config(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100.0,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2.0,
        pos_embed_rope_dtype="fp32",
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        untie_global_and_local_cls_norm=False,
    ),
    "dinov3_vitl16plus": Config(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100.0,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2.0,
        pos_embed_rope_dtype="fp32",
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=6.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="swiglu",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
    ),
    "dinov3_vith16plus": Config(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100.0,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2.0,
        pos_embed_rope_dtype="fp32",
        embed_dim=1280,
        depth=32,
        num_heads=20,
        ffn_ratio=6.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="swiglu",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
    ),
    "dinov3_vit7b16": Config(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100.0,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2.0,
        pos_embed_rope_dtype="fp32",
        embed_dim=4096,
        depth=40,
        num_heads=32,
        ffn_ratio=3.0,
        qkv_bias=False,
        drop_path_rate=0.4,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="swiglu64",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        untie_global_and_local_cls_norm=True,
    ),
}


@beartype.beartype
def _parse_name_pt(dinov3_ckpt: pathlib.Path) -> str:
    name_ds, sha = dinov3_ckpt.stem.split("-")
    *name, pretrain, ds = name_ds.split("_")
    assert pretrain == "pretrain"
    return "_".join(name)


def _tokenize(path: str) -> list[str | int]:
    """Split 'blocks.0.mlp.fc1.weight' -> ['blocks', 0, 'mlp', 'fc1', 'weight']"""
    toks = []
    for part in path.split("."):
        if part.isdigit():
            toks.append(int(part))
        else:
            toks.append(part)
    return toks


def _make_where(key: str) -> Callable[[PyTree], tp.Any]:
    toks = _tokenize(key)

    def _where(tree: PyTree) -> tp.Any:
        for attr in toks:
            if isinstance(attr, int):
                tree = tree[attr]
            else:
                try:
                    tree = getattr(tree, attr)
                except AttributeError as err:
                    raise ValueError(f"Missing attr '{key}': {err}") from err
        return tree

    return _where


def _to_jax(x) -> jnp.ndarray:
    # x: torch.Tensor or np.ndarray
    import torch  # import here to keep this file importable without torch installed

    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        arr = x
    else:
        raise TypeError(f"Unsupported tensor type: {type(x)}")
    out = jnp.asarray(arr)
    return out


@jaxtyped(typechecker=beartype.beartype)
def _coerce_to_jax(
    key: str, value, update_dst: Float[Array, "*axes"]
) -> Float[Array, "*axes"]:
    import difflib

    import torch  # import here to keep this file importable without torch installed

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        value = value
    else:
        raise TypeError(f"Unsupported tensor type: {type(value)}")
    value = jnp.asarray(value)

    if value.shape != update_dst.shape:
        for line in difflib.ndiff(value.shape, update_dst.shape):
            op, _, *rest = line
            if op == " ":
                # Same shape
                continue
            shape = int("".join(rest))
            if shape == 1:
                # Can reshape easily
                continue

            raise ValueError(
                f"Can't automatically reshape '{key}': {value.shape} => {update_dst.shape}"
            )

        print(f"Reshaping '{key}' from {value.shape} to {update_dst.shape}.")
        value = value.reshape(update_dst.shape)

    return value


@beartype.beartype
def _convert(dinov3_ckpt: pathlib.Path, dump_to: pathlib.Path):
    """Convert DINOv3 checkpoints from PyTorch to Jax/Equinox.

    Run with `uv run --with torch src/btx/modeling/dinov3.py` to include torch.

    Args:
        dinov3_ckpt: The specific .pth checkpoint you want to convert.
        dump_to: Where to save the .eqx checkpoint file.
    """
    import tempfile

    import torch

    name = _parse_name_pt(dinov3_ckpt)
    if name not in _PRETRAINED_CFGS:
        raise ValueError(f"Name '{name}' not in {list(_PRETRAINED_CFGS)}.")

    vit_pt = torch.hub.load(
        "facebookresearch/dinov3", name, source="github", weights=str(dinov3_ckpt)
    )
    cfg = _PRETRAINED_CFGS[name]
    vit_eqx = VisionTransformer(cfg, key=jax.random.key(seed=0))

    for key, value in vit_pt.state_dict().items():
        where = _make_where(key)
        update_dst = where(vit_eqx)
        update_src = _coerce_to_jax(key, value, update_dst)
        vit_eqx = eqx.tree_at(where, vit_eqx, update_src)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        dummy_fpath = str(tmpdir / "dummy.eqx")
        dump(vit_eqx, dummy_fpath)
        load(dummy_fpath)

    dump_to.mkdir(parents=True, exist_ok=True)
    fpath = str(dump_to / f"{name}.eqx")
    dump(vit_eqx, fpath)
    print(f"Saved to '{fpath}'.")


if __name__ == "__main__":
    import tyro

    tyro.cli(_convert)
