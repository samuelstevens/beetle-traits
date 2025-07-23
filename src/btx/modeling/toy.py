# src/btx/modeling/toy.py
import dataclasses

import beartype
import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, jaxtyped


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    n_keypts: int = 2
    img_size: int = 128
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
        img_size: int = 128,
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

    def __call__(self, x):  # x: (B,3,H,W)
        x = self.proj(x)  # (B,E,H',W')
        x = x.reshape(x.shape[0], x.shape[1], -1)  # (B,E,P)
        x = x.transpose(0, 2, 1)  # (B,P,E)
        cls = jnp.mean(x, axis=1, keepdims=True)  # cheap “class token”
        x = jnp.concatenate([cls, x], axis=1) + self.pos
        return x  # (B,P+1,E)


@jaxtyped(typechecker=beartype.beartype)
class Model(eqx.Module):
    patch: PatchEmbed
    head: eqx.nn.Linear

    def __init__(self, cfg: Config, *, key: chex.PRNGKey):
        k1, k2 = jax.random.split(key)
        self.patch = PatchEmbed(cfg.patch, cfg.d_model, cfg.img_size, key=k1)
        self.head = eqx.nn.Linear(cfg.d_model, cfg.n_keypts * 2, key=k2)

    def __call__(
        self,
        x_cwh: Float[Array, "channels width height"],
        *,
        key: chex.PRNGKey | None = None,
    ):
        x_pd = self.patch(x_cwh)  # (B,P+1,E)
        breakpoint()
        cls = x[:, 0]  # take pooled token
        coords = jax.nn.sigmoid(self.head(cls))  # (B,2*K) ∈ [0,1]
        return coords.reshape(x.shape[0], -1, 2)
