import jax.numpy as jnp
import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from btx import metrics


def test_apply_affine_matches_manual_homogeneous_matmul():
    affine = jnp.array(
        [
            [
                [2.0, 0.0, 1.0],
                [0.0, 3.0, -2.0],
                [0.0, 0.0, 1.0],
            ]
        ],
        dtype=jnp.float32,
    )
    points = jnp.array(
        [
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        ],
        dtype=jnp.float32,
    )

    got = np.asarray(metrics.apply_affine(affine, points))

    flat = np.asarray(points).reshape(-1, 2)
    ones = np.ones((flat.shape[0], 1), dtype=np.float32)
    hom = np.concatenate([flat, ones], axis=1).T
    manual = (np.asarray(affine)[0] @ hom).T[:, :2].reshape(1, 2, 2, 2)
    np.testing.assert_allclose(got, manual, atol=1e-6)


def test_choose_endpoint_matching_swaps_when_swapped_pairing_is_better():
    pred = jnp.array(
        [
            [
                [[0.0, 0.0], [10.0, 0.0]],
                [[0.0, 1.0], [10.0, 1.0]],
            ]
        ],
        dtype=jnp.float32,
    )
    tgt = jnp.array(
        [
            [
                [[10.0, 0.0], [0.0, 0.0]],
                [[0.0, 1.0], [10.0, 1.0]],
            ]
        ],
        dtype=jnp.float32,
    )
    got = np.asarray(metrics.choose_endpoint_matching(pred, tgt))
    expected = np.array(
        [
            [
                [[0.0, 0.0], [10.0, 0.0]],
                [[0.0, 1.0], [10.0, 1.0]],
            ]
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_choose_endpoint_matching_keeps_direct_when_direct_is_better():
    pred = jnp.array(
        [
            [
                [[0.0, 0.0], [5.0, 0.0]],
                [[10.0, 1.0], [12.0, 1.0]],
            ]
        ],
        dtype=jnp.float32,
    )
    tgt = jnp.array(
        [
            [
                [[0.2, 0.0], [5.1, -0.1]],
                [[10.2, 1.0], [11.9, 1.1]],
            ]
        ],
        dtype=jnp.float32,
    )
    got = np.asarray(metrics.choose_endpoint_matching(pred, tgt))
    np.testing.assert_allclose(got, np.asarray(tgt), atol=1e-6)


_lines_strategy = hnp.arrays(
    dtype=np.float32,
    shape=(1, 2, 2, 2),
    elements=st.floats(
        min_value=-512.0,
        max_value=512.0,
        allow_nan=False,
        allow_infinity=False,
        width=32,
    ),
)


@given(pred=_lines_strategy, tgt=_lines_strategy)
def test_choose_endpoint_matching_never_worse_than_direct_or_swapped(
    pred: np.ndarray, tgt: np.ndarray
):
    got = np.asarray(
        metrics.choose_endpoint_matching(jnp.asarray(pred), jnp.asarray(tgt))
    )
    swapped = tgt[:, :, ::-1, :]

    got_cost = np.linalg.norm(pred - got, axis=-1).sum(axis=-1)
    direct_cost = np.linalg.norm(pred - tgt, axis=-1).sum(axis=-1)
    swapped_cost = np.linalg.norm(pred - swapped, axis=-1).sum(axis=-1)
    best_cost = np.minimum(direct_cost, swapped_cost)
    assert np.all(got_cost <= best_cost + 1e-5)


def test_scalebar_mask_rejects_nonfinite_or_tiny_scalebars():
    scalebar = jnp.array(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [jnp.nan, 1.0]],
        ],
        dtype=jnp.float32,
    )
    scalebar_valid = jnp.array([True, True, True])
    out_mask, px_per_cm = metrics.get_scalebar_mask(scalebar, scalebar_valid)

    np.testing.assert_array_equal(np.asarray(out_mask)[:2], np.array([True, False]))
    assert not out_mask[2]
    np.testing.assert_allclose(np.asarray(px_per_cm)[:2], np.array([1.0, 0.0]))
    assert np.isnan(float(px_per_cm[2]))


def test_scalebar_mask_respects_input_mask():
    scalebar = jnp.array(
        [
            [[0.0, 0.0], [5.0, 0.0]],
            [[0.0, 0.0], [7.0, 0.0]],
        ],
        dtype=jnp.float32,
    )
    scalebar_valid = jnp.array([False, True])
    out_mask, px_per_cm = metrics.get_scalebar_mask(scalebar, scalebar_valid)
    np.testing.assert_array_equal(np.asarray(out_mask), np.array([False, True]))
    np.testing.assert_allclose(np.asarray(px_per_cm), np.array([5.0, 7.0]), atol=1e-8)


def test_scalebar_mask_rejects_at_exact_min_threshold():
    scalebar = jnp.array(
        [
            [[0.0, 0.0], [1e-6, 0.0]],
            [[0.0, 0.0], [2e-6, 0.0]],
        ],
        dtype=jnp.float32,
    )
    scalebar_valid = jnp.array([True, True])
    out_mask, _ = metrics.get_scalebar_mask(scalebar, scalebar_valid)
    np.testing.assert_array_equal(np.asarray(out_mask), np.array([False, True]))
