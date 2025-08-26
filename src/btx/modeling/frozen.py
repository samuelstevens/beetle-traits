# src/btx/modeling/frozen_features.py
import dataclasses
import pathlib

import beartype
import chex
import einops
import equinox as eqx
from jaxtyping import Array, Float, jaxtyped

from . import dinov3


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Frozen:
    img_size: int = 256
    dinov3_ckpt: pathlib.Path = pathlib.Path("models/dinov3_vitb16.eqx")


@jaxtyped(typechecker=beartype.beartype)
class Model(eqx.Module):
    vit: dinov3.VisionTransformer
    head: eqx.nn.Linear

    def __init__(self, cfg: Frozen, *, key: chex.PRNGKey):
        self.vit = dinov3.load(cfg.dinov3_ckpt)
        # Predict 8 coordinates: two measurements x (x, y) x 2 endpoints
        self.head = eqx.nn.Linear(self.vit.cfg.embed_dim, 8, key=key)

    def __call__(
        self, x_whc: Float[Array, "w h c"], *, key: chex.PRNGKey | None = None
    ) -> Float[Array, "2 2 2"]:
        x_cwh = einops.rearrange(x_whc, "w h c -> c h w")
        x_d = self.vit(x_cwh)
        coords = self.head(x_d)
        return coords.reshape(2, 2, 2)
