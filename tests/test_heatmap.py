import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import btx.heatmap


@given(
    points_b2=hnp.arrays(
        dtype=np.float32,
        shape=(32, 2),
        elements=st.floats(
            min_value=-1024.0,
            max_value=1024.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
)
@settings(deadline=None)
def test_udp_roundtrip_recovers_original_points(points_b2: np.ndarray):
    cfg = btx.heatmap.Config(image_size=256, heatmap_size=64, sigma=2.0)
    points_hm_b2 = btx.heatmap.image_to_heatmap_udp(
        jnp.asarray(points_b2, dtype=jnp.float32), cfg=cfg
    )
    points_img_b2 = btx.heatmap.heatmap_to_image_udp(points_hm_b2, cfg=cfg)
    np.testing.assert_allclose(points_img_b2, points_b2, atol=1e-6)


@given(
    points_hm_b2=hnp.arrays(
        dtype=np.float32,
        shape=(32, 2),
        elements=st.floats(
            min_value=-32.0,
            max_value=96.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
)
@settings(deadline=None)
def test_udp_roundtrip_recovers_original_heatmap_points(points_hm_b2: np.ndarray):
    cfg = btx.heatmap.Config(image_size=256, heatmap_size=64, sigma=2.0)
    points_img_b2 = btx.heatmap.heatmap_to_image_udp(
        jnp.asarray(points_hm_b2, dtype=jnp.float32), cfg=cfg
    )
    points_hm2_b2 = btx.heatmap.image_to_heatmap_udp(points_img_b2, cfg=cfg)
    np.testing.assert_allclose(points_hm2_b2, points_hm_b2, atol=1e-6)


def test_udp_maps_image_borders_to_expected_heatmap_coords():
    cfg = btx.heatmap.Config(image_size=256, heatmap_size=64, sigma=2.0)
    points_b2 = jnp.array([[0.0, 0.0], [255.0, 255.0]], dtype=jnp.float32)

    points_hm_b2 = btx.heatmap.image_to_heatmap_udp(points_b2, cfg=cfg)
    expected_hm_b2 = jnp.array([[-0.375, -0.375], [63.375, 63.375]], dtype=jnp.float32)
    np.testing.assert_allclose(points_hm_b2, expected_hm_b2, atol=1e-6)


def test_make_targets_returns_four_endpoint_heatmaps():
    cfg = btx.heatmap.Config(image_size=256, heatmap_size=64, sigma=1.5)
    points_l22 = jnp.array(
        [
            [[1.5, 1.5], [5.5, 9.5]],
            [[13.5, 17.5], [21.5, 25.5]],
        ],
        dtype=jnp.float32,
    )

    heatmaps_chw = btx.heatmap.make_targets(points_l22, cfg=cfg)
    assert heatmaps_chw.shape == (4, 64, 64)
    np.testing.assert_allclose(jnp.max(heatmaps_chw, axis=(1, 2)), 1.0, atol=1e-7)


def test_make_targets_channel_order_matches_width_then_length_endpoints():
    cfg = btx.heatmap.Config(image_size=256, heatmap_size=64, sigma=0.5)
    points_l22 = jnp.array(
        [
            [[1.5, 1.5], [5.5, 9.5]],
            [[13.5, 17.5], [21.5, 25.5]],
        ],
        dtype=jnp.float32,
    )

    heatmaps_chw = btx.heatmap.make_targets(points_l22, cfg=cfg)
    argmax_hw = [
        np.unravel_index(int(np.argmax(np.asarray(heatmaps_chw[c]))), (64, 64))
        for c in range(4)
    ]
    assert argmax_hw == [
        (0, 0),  # width p0
        (2, 1),  # width p1
        (4, 3),  # length p0
        (6, 5),  # length p1
    ]


def test_heatmaps_to_coords_returns_line_endpoint_coordinates():
    cfg = btx.heatmap.Config(image_size=256, heatmap_size=64, sigma=2.0)
    logits_chw = jnp.zeros((4, 64, 64), dtype=jnp.float32)

    coords_l22 = btx.heatmap.heatmaps_to_coords(logits_chw, cfg=cfg)
    assert coords_l22.shape == (2, 2, 2)
    np.testing.assert_allclose(coords_l22, 127.5, atol=1e-4)


def test_heatmap_loss_zero_when_predictions_match_targets():
    cfg = btx.heatmap.Config(image_size=256, heatmap_size=64, sigma=2.0)
    tgt_chw = jnp.arange(4 * 8 * 8, dtype=jnp.float32).reshape(4, 8, 8) / 100.0
    pred_chw = tgt_chw.copy()
    loss_mask_l = jnp.array([1.0, 1.0], dtype=jnp.float32)

    loss = btx.heatmap.heatmap_loss(pred_chw, tgt_chw, loss_mask_l, cfg=cfg)
    np.testing.assert_allclose(loss, 0.0, atol=1e-8)


def test_heatmap_loss_masks_inactive_lines_from_denominator():
    cfg = btx.heatmap.Config(image_size=256, heatmap_size=64, sigma=2.0)
    pred_chw = jnp.zeros((4, 8, 8), dtype=jnp.float32)
    tgt_chw = jnp.zeros((4, 8, 8), dtype=jnp.float32)
    tgt_chw = tgt_chw.at[0].set(1.0)
    tgt_chw = tgt_chw.at[1].set(1.0)
    loss_mask_l = jnp.array([1.0, 0.0], dtype=jnp.float32)

    # Width has two active channels, each with per-pixel squared error 1.0.
    # Normalization by active elements (2 * H * W) should give exactly 1.0.
    loss = btx.heatmap.heatmap_loss(pred_chw, tgt_chw, loss_mask_l, cfg=cfg)
    np.testing.assert_allclose(loss, 1.0, atol=1e-8)


def test_heatmap_loss_is_permutation_invariant_per_line():
    cfg = btx.heatmap.Config(image_size=256, heatmap_size=64, sigma=2.0)
    pred_chw = jnp.zeros((4, 4, 4), dtype=jnp.float32)
    tgt_chw = jnp.zeros((4, 4, 4), dtype=jnp.float32)

    pred_chw = pred_chw.at[0, 0, 0].set(1.0)
    pred_chw = pred_chw.at[1, 1, 1].set(1.0)

    # Targets swapped relative to predicted width channels.
    tgt_chw = tgt_chw.at[0, 1, 1].set(1.0)
    tgt_chw = tgt_chw.at[1, 0, 0].set(1.0)
    loss_mask_l = jnp.array([1.0, 0.0], dtype=jnp.float32)

    loss = btx.heatmap.heatmap_loss(pred_chw, tgt_chw, loss_mask_l, cfg=cfg)
    np.testing.assert_allclose(loss, 0.0, atol=1e-8)
