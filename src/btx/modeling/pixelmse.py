import dataclasses
import pathlib
from typing import Tuple

import beartype
import chex
import einops
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, jaxtyped

from . import dinov3


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class PixelMse:
    img_size: int = 256
    patch_size: int = 16
    num_keypoints: int = 4
    dinov3_ckpt: pathlib.Path = pathlib.Path("models/dinov3_vitb16.eqx")


@jaxtyped(typechecker=beartype.beartype)
class Model(eqx.Module):
    vit: dinov3.VisionTransformer
    head: eqx.nn.Linear
    cfg: PixelMse

    def __init__(self, cfg: PixelMse, *, key: chex.PRNGKey):
        self.cfg = cfg
        self.vit = dinov3.load(cfg.dinov3_ckpt)
        out_dim = (cfg.patch_size * cfg.patch_size) * cfg.num_keypoints
        self.head = eqx.nn.Linear(self.vit.cfg.embed_dim, out_dim, key=key)


    def _get_patch_tokens(
        self, x_cwh: Float[Array, "c h w"]
    ) -> tuple[Float[Array, "n_tokens d"], tuple[int, int]]:
        """
        Preferred path: VisionTransformer.tokens(x) -> (cls, patches, (Hp, Wp)).
        Falls back to older styles if tokens() isn't present.
        """
        vit = self.vit

        if hasattr(vit, "tokens"):
            cls_tok, patch_toks, grid_hw = vit.tokens(x_cwh) 
            return patch_toks, grid_hw

        try:
            cls_tok, patch_toks = vit(x_cwh, return_tokens=True)
            ph = pw = self.cfg.patch_size
            H = W = self.cfg.img_size
            return patch_toks, (H // ph, W // pw)
        except TypeError:
            pass

        if hasattr(vit, "forward_tokens"):
            out = vit.forward_tokens(x_cwh)
            if isinstance(out, tuple) and len(out) == 2:
                _, patch_toks = out
                ph = pw = self.cfg.patch_size
                H = W = self.cfg.img_size
                return patch_toks, (H // ph, W // pw)
            if isinstance(out, dict) and "patches" in out:
                patch_toks = out["patches"]
                ph = pw = self.cfg.patch_size
                H = W = self.cfg.img_size
                return patch_toks, (H // ph, W // pw)

        if hasattr(vit, "forward_features"):
            feats = vit.forward_features(x_cwh)
            if isinstance(feats, dict) and "patches" in feats:
                patch_toks = feats["patches"]
                ph = pw = self.cfg.patch_size
                H = W = self.cfg.img_size
                return patch_toks, (H // ph, W // pw)

        raise RuntimeError(
            "Could not extract patch tokens from ViT. ")



    def __call__(
        self, x_hwc: Float[Array, "h w c"], *, key: chex.PRNGKey | None = None
    ) -> Float[Array, "k h w"]:
        cfg = self.cfg
        ph = pw = cfg.patch_size
        K = cfg.num_keypoints
        x_cwh = einops.rearrange(x_hwc, "h w c -> c h w")
        patch_tokens, (Hp, Wp) = self._get_patch_tokens(x_cwh)  
        logits = jax.vmap(self.head)(patch_tokens)
        probs_patch = einops.rearrange(
            logits,
            "(hp wp) (ph pw k) -> k hp ph wp pw",
            hp=Hp, wp=Wp, ph=ph, pw=pw, k=K,
        )
        probs_full = einops.rearrange(
            probs_patch, "k hp ph wp pw -> k (hp ph) (wp pw)"
        ).astype(jnp.float32)
        probs_full = jax.nn.sigmoid(probs_full)
        return probs_full

