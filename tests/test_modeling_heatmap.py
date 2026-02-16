import pathlib

import jax
import jax.numpy as jnp
import pytest

from btx.modeling import dinov3, heatmap


@pytest.mark.parametrize(
    ("embed_dim", "num_heads"),
    [
        (384, 6),
        (768, 12),
        (1024, 16),
    ],
)
def test_heatmap_model_forward_shape_is_stable_across_vit_sizes(
    monkeypatch: pytest.MonkeyPatch,
    embed_dim: int,
    num_heads: int,
):
    vit_cfg = dinov3.Config(
        img_size=256,
        patch_size=16,
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=0,
        pos_embed_rope_dtype="fp32",
    )

    def _load_stub(_fpath: pathlib.Path) -> dinov3.VisionTransformer:
        return dinov3.VisionTransformer(vit_cfg, key=jax.random.key(0))

    monkeypatch.setattr(dinov3, "load", _load_stub)
    cfg = heatmap.Heatmap(dinov3_ckpt=pathlib.Path("unused.eqx"))
    model = heatmap.Model(cfg, key=jax.random.key(1))

    img_hwc = jnp.zeros((256, 256, 3), dtype=jnp.float32)
    logits_chw = model(img_hwc)

    assert logits_chw.shape == (4, 64, 64)
    assert jnp.isfinite(logits_chw).all()
