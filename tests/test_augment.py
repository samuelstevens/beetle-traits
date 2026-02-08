import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from btx.data import augment


def _sample() -> dict[str, object]:
    return {
        "img": np.zeros((256, 256, 3), dtype=np.float32),
        "points_px": np.array(
            [
                [[10.0, 20.0], [30.0, 40.0]],
                [[100.0, 120.0], [140.0, 180.0]],
            ],
            dtype=np.float32,
        ),
        "scalebar_px": np.array([[5.0, 5.0], [25.0, 5.0]], dtype=np.float32),
        "loss_mask": np.array([1.0, 1.0], dtype=np.float32),
    }


def _spatial_sample() -> dict[str, object]:
    sample = _sample()
    sample["t_aug_from_orig"] = np.eye(3, dtype=np.float32)
    sample["metric_mask_cm"] = 1.0
    return sample


def test_augment_config_requires_fixed_size():
    with pytest.raises(AssertionError):
        augment.AugmentConfig(size=128)


def test_get_identity_affine_matches_eye():
    got = augment.get_identity_affine()
    np.testing.assert_allclose(got, np.eye(3), atol=1e-8)


def test_get_crop_resize_affine_matches_spec_formula():
    got = augment.get_crop_resize_affine(
        x0=10.0,
        y0=20.0,
        crop_w=128.0,
        crop_h=64.0,
        size=256,
    )
    expected = np.array([
        [2.0, 0.0, -20.0],
        [0.0, 4.0, -80.0],
        [0.0, 0.0, 1.0],
    ])
    np.testing.assert_allclose(got, expected, atol=1e-8)


def test_get_hflip_affine_matches_spec_formula():
    got = augment.get_hflip_affine(size=256)
    expected = np.array([
        [-1.0, 0.0, 255.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    np.testing.assert_allclose(got, expected, atol=1e-8)


def test_get_vflip_affine_matches_spec_formula():
    got = augment.get_vflip_affine(size=256)
    expected = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 255.0],
        [0.0, 0.0, 1.0],
    ])
    np.testing.assert_allclose(got, expected, atol=1e-8)


def test_get_rot90_affine_k1_matches_spec_formula():
    got = augment.get_rot90_affine(k=1, size=256)
    expected = np.array([
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 255.0],
        [0.0, 0.0, 1.0],
    ])
    np.testing.assert_allclose(got, expected, atol=1e-8)


@given(k=st.integers(min_value=0, max_value=3))
def test_rot90_inverse_property(k: int):
    rot = augment.get_rot90_affine(k=k, size=256)
    inv = augment.get_rot90_affine(k=(4 - k) % 4, size=256)
    np.testing.assert_allclose(inv @ rot, np.eye(3), atol=1e-8)


_pts_strategy = hnp.arrays(
    dtype=np.float64,
    shape=(2, 2, 2),
    elements=st.floats(
        min_value=-1024.0,
        max_value=1024.0,
        allow_nan=False,
        allow_infinity=False,
    ),
)


@given(points_l22=_pts_strategy)
def test_hflip_twice_returns_original(points_l22: np.ndarray):
    flip = augment.get_hflip_affine(size=256)
    once = augment.apply_affine_to_points(flip, points_l22)
    twice = augment.apply_affine_to_points(flip, once)
    np.testing.assert_allclose(twice, points_l22, atol=1e-6)


@given(points_l22=_pts_strategy)
def test_vflip_twice_returns_original(points_l22: np.ndarray):
    flip = augment.get_vflip_affine(size=256)
    once = augment.apply_affine_to_points(flip, points_l22)
    twice = augment.apply_affine_to_points(flip, once)
    np.testing.assert_allclose(twice, points_l22, atol=1e-6)


def test_apply_affine_to_points_matches_manual_homogeneous_matmul():
    points = np.array([
        [[2.0, 4.0], [10.0, 20.0]],
        [[-1.0, 3.0], [0.5, -2.0]],
    ])
    affine = np.array([
        [2.0, 0.0, 1.0],
        [0.0, 3.0, -2.0],
        [0.0, 0.0, 1.0],
    ])

    flat = points.reshape(-1, 2)
    ones = np.ones((flat.shape[0], 1))
    hom = np.concatenate([flat, ones], axis=1).T
    manual = (affine @ hom).T[:, :2].reshape(points.shape)

    got = augment.apply_affine_to_points(affine, points)
    np.testing.assert_allclose(got, manual, atol=1e-8)


def test_is_in_bounds_uses_half_open_interval():
    points = np.array([
        [[0.0, 0.0], [255.0, 255.0]],
        [[-1.0, 4.0], [128.0, 256.0]],
    ])
    got = augment.is_in_bounds(points, size=256)
    expected = np.array([
        [True, True],
        [False, False],
    ])
    np.testing.assert_array_equal(got, expected)


def test_init_aug_state_adds_identity_matrices_and_metric_mask():
    sample = _sample()
    out = augment.InitAugState(size=256).map(sample)
    np.testing.assert_allclose(out["t_aug_from_orig"], np.eye(3), atol=1e-8)
    np.testing.assert_allclose(out["t_orig_from_aug"], np.eye(3), atol=1e-8)
    assert float(out["metric_mask_cm"]) == 1.0


def test_init_aug_state_masks_cm_metrics_for_degenerate_scalebar():
    sample = _sample()
    sample["scalebar_px"] = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    out = augment.InitAugState(size=256, min_px_per_cm=1e-6).map(sample)
    assert float(out["metric_mask_cm"]) == 0.0


def test_finalize_targets_matches_affine_application_and_inverse_identity():
    sample = _sample()
    sample["metric_mask_cm"] = 1.0
    sample["t_aug_from_orig"] = np.array(
        [
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 255.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    out = augment.FinalizeTargets(cfg=augment.AugmentConfig()).map(sample)
    expected_tgt = augment.apply_affine_to_points(
        out["t_aug_from_orig"], sample["points_px"]
    )
    np.testing.assert_allclose(out["tgt"], expected_tgt, atol=1e-6)
    np.testing.assert_allclose(
        out["t_orig_from_aug"] @ out["t_aug_from_orig"],
        np.eye(3),
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("oob_policy", "expected"),
    [
        ("mask_any_oob", np.array([0.0, 0.0], dtype=np.float32)),
        ("mask_all_oob", np.array([1.0, 0.0], dtype=np.float32)),
        ("supervise_oob", np.array([1.0, 1.0], dtype=np.float32)),
    ],
)
def test_finalize_targets_applies_oob_policy_to_loss_mask(
    oob_policy: augment.OobPolicy, expected: np.ndarray
):
    sample = _spatial_sample()
    sample["points_px"] = np.array(
        [
            [[10.0, 20.0], [300.0, 40.0]],
            [[-5.0, 1.0], [-1.0, 4.0]],
        ],
        dtype=np.float32,
    )
    cfg = augment.AugmentConfig(oob_policy=oob_policy)
    out = augment.FinalizeTargets(cfg=cfg).map(sample)
    np.testing.assert_allclose(out["loss_mask"], expected, atol=1e-8)


def test_finalize_targets_asserts_when_affine_has_non_finite_values():
    sample = _spatial_sample()
    sample["t_aug_from_orig"] = np.array(
        [
            [1.0, 0.0, np.nan],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    with pytest.raises(AssertionError):
        _ = augment.FinalizeTargets(cfg=augment.AugmentConfig()).map(sample)


def test_affine_composition_order_is_not_commutative():
    points = np.array(
        [
            [[10.0, 20.0], [30.0, 40.0]],
            [[80.0, 100.0], [160.0, 190.0]],
        ],
        dtype=np.float32,
    )
    rot = augment.get_rot90_affine(k=1, size=256)
    flip = augment.get_hflip_affine(size=256)
    rot_then_flip = augment.apply_affine_to_points(flip @ rot, points)
    flip_then_rot = augment.apply_affine_to_points(rot @ flip, points)
    assert not np.allclose(rot_then_flip, flip_then_rot, atol=1e-6)


def test_random_resized_crop_is_identity_at_full_scale_square_ratio():
    cfg = augment.AugmentConfig(
        crop_scale_min=1.0,
        crop_scale_max=1.0,
        crop_ratio_min=1.0,
        crop_ratio_max=1.0,
    )
    sample = _spatial_sample()
    out = augment.RandomResizedCrop(cfg).random_map(
        sample,
        np.random.default_rng(seed=17),
    )
    np.testing.assert_allclose(out["t_aug_from_orig"], np.eye(3), atol=1e-6)


def test_random_flip_prob_zero_is_identity():
    cfg = augment.AugmentConfig(hflip_prob=0.0, vflip_prob=0.0)
    sample = _spatial_sample()
    out = augment.RandomFlip(cfg).random_map(sample, np.random.default_rng(seed=17))
    np.testing.assert_allclose(out["t_aug_from_orig"], np.eye(3), atol=1e-6)


def test_random_flip_prob_one_applies_hflip_when_vflip_disabled():
    cfg = augment.AugmentConfig(hflip_prob=1.0, vflip_prob=0.0)
    sample = _spatial_sample()
    out = augment.RandomFlip(cfg).random_map(sample, np.random.default_rng(seed=17))
    expected = augment.get_hflip_affine(size=256)
    np.testing.assert_allclose(out["t_aug_from_orig"], expected, atol=1e-6)


def test_random_rotation_prob_zero_is_identity():
    cfg = augment.AugmentConfig(rotation_prob=0.0)
    sample = _spatial_sample()
    out = augment.RandomRotation90(cfg).random_map(
        sample, np.random.default_rng(seed=17)
    )
    np.testing.assert_allclose(out["t_aug_from_orig"], np.eye(3), atol=1e-6)


def test_random_rotation_prob_one_is_non_identity():
    cfg = augment.AugmentConfig(rotation_prob=1.0)
    sample = _spatial_sample()
    out = augment.RandomRotation90(cfg).random_map(
        sample, np.random.default_rng(seed=17)
    )
    assert not np.allclose(out["t_aug_from_orig"], np.eye(3), atol=1e-6)


def test_color_jitter_zero_strength_is_noop():
    cfg = augment.AugmentConfig(
        brightness=0.0,
        contrast=0.0,
        saturation=0.0,
        hue=0.0,
    )
    sample = _sample()
    sample["img"] = np.full((256, 256, 3), 0.25, dtype=np.float32)
    out = augment.ColorJitter(cfg).random_map(sample, np.random.default_rng(seed=17))
    np.testing.assert_allclose(out["img"], sample["img"], atol=1e-6)
