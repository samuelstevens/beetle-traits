# src/btx/modeling/frozen_features.py
import dataclasses
import pathlib

import beartype
import chex
import einops
import equinox as eqx
import jax.nn as jnn
from jaxtyping import Array, Float, jaxtyped

from . import dinov3


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Frozen:
    dinov3_ckpt: pathlib.Path = pathlib.Path("models/dinov3_vitb16.eqx")


@jaxtyped(typechecker=beartype.beartype)
class Model(eqx.Module):
    vit: dinov3.VisionTransformer
    head: eqx.nn.MLP

    def __init__(self, cfg: Frozen, *, key: chex.PRNGKey):
        self.vit = dinov3.load(cfg.dinov3_ckpt)
        # Predict 8 coordinates: two measurements x (x, y) x 2 endpoints
        # MLP: embed_dim -> embed_dim (hidden) -> 8
        self.head = eqx.nn.MLP(
            in_size=self.vit.cfg.embed_dim,
            out_size=8,
            width_size=self.vit.cfg.embed_dim,
            depth=1,
            activation=jnn.gelu,
            key=key,
        )

    def __call__(
        self, x_hwc: Float[Array, "h w c"], *, key: chex.PRNGKey | None = None
    ) -> Float[Array, "2 2 2"]:
        x_cwh = einops.rearrange(x_hwc, "h w c -> c h w")
        out = self.vit(x_cwh)
        coords = self.head(out["cls"])
        return coords.reshape(2, 2, 2)
