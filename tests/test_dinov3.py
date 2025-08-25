# tests/test_dinov3.py

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from btx.modeling import dinov3


@pytest.fixture(scope="session")
def pt_ckpt_dir(pytestconfig):
    ckpt_dir = pytestconfig.getoption("--dinov3-pt-ckpts")
    if ckpt_dir is None:
        pytest.skip("--dinov3-pt-ckpts not supplied")
    return ckpt_dir


@pytest.fixture(scope="session")
def jax_ckpt_dir(pytestconfig):
    ckpt_dir = pytestconfig.getoption("--dinov3-jax-ckpts")
    if ckpt_dir is None:
        pytest.skip("--dinov3-jax-ckpts not supplied")
    return ckpt_dir


@pytest.fixture(scope="function")
def vits(vit_paths):
    (pt_path, pt_name), jax_path = vit_paths
    vit_pt = torch.hub.load(
        "facebookresearch/dinov3", pt_name, source="github", weights=pt_path
    ).eval()
    vit_eqx = dinov3.load(jax_path)

    return vit_pt, vit_eqx


def test_load_jax_ckpt(jax_path):
    # Parameterize this test with each filepath in whatever is in jax_ckpt_dir.
    dinov3.load(jax_path)


@pytest.mark.parametrize("h,w", [(2, 2), (4, 4), (16, 16), (8, 12)])
def test_rope_fwd(h, w, vits):
    vit_pt, vit_eqx = vits
    rope_pt = vit_pt.rope_embed
    rope_eqx = vit_eqx.rope_embed
    out_pt_h, out_pt_w = rope_pt(H=h, W=w)
    out_eqx_h, out_eqx_w = rope_eqx(h=h, w=w)

    # Check shape
    assert out_pt_h.shape == out_eqx_h.shape
    assert out_pt_w.shape == out_eqx_w.shape

    # Check values
    np.testing.assert_allclose(out_pt_h, out_eqx_h, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(out_pt_w, out_eqx_w, rtol=1e-6, atol=1e-6)


def test_patch_embed_fwd(vits):
    vit_pt, vit_eqx = vits
    torch.manual_seed(14)

    x_1chw = torch.rand((1, 3, vit_eqx.cfg.img_size, vit_eqx.cfg.img_size))
    in_eqx = jnp.asarray(x_1chw.detach().cpu().numpy())[0]
    out_eqx = vit_eqx.patch_embed(in_eqx)

    # PyTorch forward pass
    with torch.no_grad():
        out_pt = vit_pt.patch_embed(x_1chw)
    out_pt = jnp.asarray(out_pt.detach().cpu().numpy())[0]

    # Check shapes
    assert out_eqx.shape == out_pt.shape

    # Check values
    np.testing.assert_allclose(out_pt, out_eqx, rtol=1e-6, atol=1e-4)


def test_self_attn_fwd(vits):
    vit_pt, vit_eqx = vits
    torch.manual_seed(14)

    h = w = vit_eqx.cfg.img_size // vit_eqx.cfg.patch_size
    n_tok = h * w + vit_eqx.cfg.n_storage_tokens + 1
    x_1nd = torch.rand((1, n_tok, vit_eqx.cfg.embed_dim))

    rope_sincos_pt = vit_pt.rope_embed(H=h, W=w)
    rope_sincos_eqx = vit_eqx.rope_embed(h=h, w=w)
    rope_2pd_eqx = jnp.stack(rope_sincos_eqx, axis=0)

    out_eqx = vit_eqx.blocks[0].attn(
        jnp.asarray(x_1nd.detach().cpu().numpy())[0], rope_2pd_eqx
    )

    # PyTorch forward pass
    with torch.no_grad():
        out_pt = vit_pt.blocks[0].attn(x_1nd, rope=rope_sincos_pt)
    out_pt = jnp.asarray(out_pt.detach().cpu().numpy())[0]

    # Check shapes
    assert out_eqx.shape == out_pt.shape

    # Check values
    np.testing.assert_allclose(out_pt, out_eqx, rtol=1e-6, atol=1e-4)


def test_self_attn_no_rope_fwd(vits):
    vit_pt, vit_eqx = vits
    torch.manual_seed(12)

    h = w = vit_eqx.cfg.img_size // vit_eqx.cfg.patch_size
    n_tok = h * w + vit_eqx.cfg.n_storage_tokens + 1
    x_1nd = torch.rand((1, n_tok, vit_eqx.cfg.embed_dim))

    out_eqx = vit_eqx.blocks[0].attn(jnp.asarray(x_1nd.detach().cpu().numpy())[0], None)

    # PyTorch forward pass
    with torch.no_grad():
        out_pt = vit_pt.blocks[0].attn(x_1nd)
    out_pt = jnp.asarray(out_pt.detach().cpu().numpy())[0]

    # Check shapes
    assert out_eqx.shape == out_pt.shape

    # Check values
    np.testing.assert_allclose(out_pt, out_eqx, rtol=1e-6, atol=1e-4)


def test_mlp_fwd(vits):
    vit_pt, vit_eqx = vits
    torch.manual_seed(15)

    x_1d = torch.rand((1, vit_eqx.cfg.embed_dim))

    in_eqx = jnp.asarray(x_1d.detach().cpu().numpy())[0]
    out_eqx = vit_eqx.blocks[1].mlp(in_eqx)

    # PyTorch forward pass
    with torch.no_grad():
        out_pt = vit_pt.blocks[1].mlp(x_1d)
    out_pt = jnp.asarray(out_pt.detach().cpu().numpy())[0]

    # Check shapes
    assert out_eqx.shape == out_pt.shape

    # Check values
    np.testing.assert_allclose(out_pt, out_eqx, rtol=1e-6, atol=1e-4)


def test_self_attn_block_fwd(vits):
    vit_pt, vit_eqx = vits
    torch.manual_seed(14)

    h = w = vit_eqx.cfg.img_size // vit_eqx.cfg.patch_size
    n_tok = h * w + vit_eqx.cfg.n_storage_tokens + 1
    x_1nd = torch.rand((1, n_tok, vit_eqx.cfg.embed_dim))

    rope_sincos_pt = vit_pt.rope_embed(H=h, W=w)
    rope_sincos_eqx = vit_eqx.rope_embed(h=h, w=w)
    rope_2pd_eqx = jnp.stack(rope_sincos_eqx, axis=0)

    x_eqx = jnp.asarray(x_1nd.detach().cpu().numpy())[0]
    out_eqx = vit_eqx.blocks[0](x_eqx, rope_2pd_eqx)

    # PyTorch forward pass
    with torch.no_grad():
        out_pt = vit_pt.blocks[0](x_1nd, rope_sincos_pt)
    out_pt = jnp.asarray(out_pt.detach().cpu().numpy())[0]

    # Check shapes
    assert out_eqx.shape == out_pt.shape

    # Check values
    np.testing.assert_allclose(out_pt, out_eqx, rtol=1e-6, atol=1e-4)


def test_vit_fwd(vits):
    vit_pt, vit_eqx = vits

    torch.manual_seed(12)
    img_bchw = torch.rand((1, 3, 224, 224))

    # Get Jax forward pass. We vmap because the jax version is not batched by default.
    x_eqx = jnp.asarray(img_bchw.detach().cpu().numpy())[0]
    out_eqx = vit_eqx(x_eqx)

    # Get PT forward pass
    with torch.no_grad():
        out_pt = vit_pt(img_bchw)
    out_pt = jnp.asarray(out_pt.detach().cpu().numpy())[0]

    # Check shape
    assert out_pt.shape == out_eqx.shape

    # Check values
    np.testing.assert_allclose(out_pt, out_eqx, rtol=1e-6, atol=1e-4)
