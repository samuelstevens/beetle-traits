# src/btx/modeling/toy.py
import dataclasses

import beartype
import chex
import einops
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, jaxtyped


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    img_size: int = 256
    patch: int = 16
    d_model: int = 192


@jaxtyped(typechecker=beartype.beartype)
class PatchEmbed(eqx.Module):
    proj: eqx.nn.Conv2d  # conv with stride = patch
    pos: Float[Array, "n_patches+1 d_model"]

    def __init__(
        self,
        patch: int = 16,
        d_model: int = 192,
        img_size: int = 256,
        *,
        key: chex.PRNGKey,
    ):
        k1, k2 = jax.random.split(key)
        self.proj = eqx.nn.Conv2d(
            in_channels=3,
            out_channels=d_model,
            kernel_size=patch,
            stride=patch,
            padding=0,
            key=k1,
        )
        n_patches = (img_size // patch) ** 2
        self.pos = jax.random.normal(k2, (n_patches + 1, d_model)) * 0.02

    def __call__(self, x_cwh: Float[Array, "c w h"]):
        x_dwh = self.proj(x_cwh)  # (B,E,H',W')
        x_pd = einops.rearrange(x_dwh, "d w h -> (w h) d")
        cls_1d = einops.reduce(x_pd, "patches d -> 1 d", "mean")
        x_pd = jnp.concatenate([cls_1d, x_pd], axis=0) + self.pos
        return x_pd


@jaxtyped(typechecker=beartype.beartype)
class Model(eqx.Module):
    patch: PatchEmbed
    head: eqx.nn.Linear

    def __init__(self, cfg: Config, *, key: chex.PRNGKey):
        k1, k2 = jax.random.split(key)
        self.patch = PatchEmbed(cfg.patch, cfg.d_model, cfg.img_size, key=k1)
        # Predict 8 coordinates: two measurements x (x, y) x 2 endpoints
        self.head = eqx.nn.Linear(cfg.d_model, 8, key=k2)

    def __call__(
        self, x_whc: Float[Array, "w h c"], *, key: chex.PRNGKey | None = None
    ) -> Float[Array, "2 2 2"]:
        x_cwh = einops.rearrange(x_whc, "w h c -> c w h")
        x_pd = self.patch(x_cwh)
        cls_d = x_pd[0, :]
        coords = self.head(cls_d)
        return coords.reshape(2, 2, 2)
