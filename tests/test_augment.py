import hypothesis
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from PIL import Image

import btx.data.transforms


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
    sample["scalebar_valid"] = np.bool_(True)
    return sample


def test_augment_config_requires_fixed_size():
    with pytest.raises(AssertionError):
        btx.data.transforms.AugmentConfig(size=128)


@pytest.mark.parametrize("prob", [-0.1, 1.1])
def test_augment_config_requires_valid_color_jitter_prob(prob: float):
    with pytest.raises(AssertionError):
        btx.data.transforms.AugmentConfig(color_jitter_prob=prob)


def test_get_identity_affine_matches_eye():
    got = btx.data.transforms.get_identity_affine()
    np.testing.assert_allclose(got, np.eye(3), atol=1e-8)


def test_get_crop_resize_affine_matches_spec_formula():
    got = btx.data.transforms.get_crop_resize_affine(
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
    got = btx.data.transforms.get_hflip_affine(size=256)
    expected = np.array([
        [-1.0, 0.0, 255.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    np.testing.assert_allclose(got, expected, atol=1e-8)


def test_get_vflip_affine_matches_spec_formula():
    got = btx.data.transforms.get_vflip_affine(size=256)
    expected = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 255.0],
        [0.0, 0.0, 1.0],
    ])
    np.testing.assert_allclose(got, expected, atol=1e-8)


def test_get_rotation_affine_90_matches_manual():
    got = btx.data.transforms.get_rotation_affine(90.0, size=256)
    expected = np.array([
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 255.0],
        [0.0, 0.0, 1.0],
    ])
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_rotation_affine_identity_at_zero():
    got = btx.data.transforms.get_rotation_affine(0.0, size=256)
    np.testing.assert_allclose(got, np.eye(3), atol=1e-6)


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
    flip = btx.data.transforms.get_hflip_affine(size=256)
    once = btx.data.transforms.apply_affine_to_points(flip, points_l22)
    twice = btx.data.transforms.apply_affine_to_points(flip, once)
    np.testing.assert_allclose(twice, points_l22, atol=1e-6)


@given(points_l22=_pts_strategy)
def test_vflip_twice_returns_original(points_l22: np.ndarray):
    flip = btx.data.transforms.get_vflip_affine(size=256)
    once = btx.data.transforms.apply_affine_to_points(flip, points_l22)
    twice = btx.data.transforms.apply_affine_to_points(flip, once)
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

    got = btx.data.transforms.apply_affine_to_points(affine, points)
    np.testing.assert_allclose(got, manual, atol=1e-8)


def test_is_in_bounds_uses_half_open_interval():
    points = np.array([
        [[0.0, 0.0], [255.0, 255.0]],
        [[-1.0, 4.0], [128.0, 256.0]],
    ])
    got = btx.data.transforms.is_in_bounds(points, size=256)
    expected = np.array([
        [True, True],
        [False, False],
    ])
    np.testing.assert_array_equal(got, expected)


def test_init_aug_state_adds_identity_matrices_and_metric_mask():
    sample = _sample()
    out = btx.data.transforms.InitAugState(size=256).map(sample)
    np.testing.assert_allclose(out["t_aug_from_orig"], np.eye(3), atol=1e-8)
    np.testing.assert_allclose(out["t_orig_from_aug"], np.eye(3), atol=1e-8)
    assert out["scalebar_valid"]


def test_init_aug_state_masks_cm_metrics_for_degenerate_scalebar():
    sample = _sample()
    sample["scalebar_px"] = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    out = btx.data.transforms.InitAugState(size=256, min_px_per_cm=1e-6).map(sample)
    assert not out["scalebar_valid"]


def test_init_aug_state_preserves_existing_scalebar_valid_false():
    sample = _sample()
    sample["scalebar_valid"] = np.bool_(False)
    out = btx.data.transforms.InitAugState(size=256).map(sample)
    assert not out["scalebar_valid"]


def test_init_aug_state_converts_pil_image_to_float_array():
    sample = _sample()
    sample["img"] = Image.fromarray(np.full((256, 256, 3), 128, dtype=np.uint8))
    out = btx.data.transforms.InitAugState(size=256).map(sample)
    img = out["img"]
    assert isinstance(img, np.ndarray)
    assert img.dtype == np.float32
    np.testing.assert_allclose(img[0, 0, 0], np.float32(128.0 / 255.0), atol=1e-6)


def test_init_aug_state_converts_uint8_array_to_float_array():
    sample = _sample()
    sample["img"] = np.full((256, 256, 3), 64, dtype=np.uint8)
    out = btx.data.transforms.InitAugState(size=256).map(sample)
    img = out["img"]
    assert isinstance(img, np.ndarray)
    assert img.dtype == np.float32
    np.testing.assert_allclose(img[0, 0, 0], np.float32(64.0 / 255.0), atol=1e-6)


def test_finalize_targets_matches_affine_application_and_inverse_identity():
    sample = _sample()
    sample["scalebar_valid"] = np.bool_(True)
    sample["t_aug_from_orig"] = np.array(
        [
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 255.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    out = btx.data.transforms.FinalizeTargets(
        cfg=btx.data.transforms.AugmentConfig()
    ).map(sample)
    expected_tgt = btx.data.transforms.apply_affine_to_points(
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
    oob_policy: btx.data.transforms.OobPolicy, expected: np.ndarray
):
    sample = _spatial_sample()
    sample["points_px"] = np.array(
        [
            [[10.0, 20.0], [300.0, 40.0]],
            [[-5.0, 1.0], [-1.0, 4.0]],
        ],
        dtype=np.float32,
    )
    cfg = btx.data.transforms.AugmentConfig(oob_policy=oob_policy)
    out = btx.data.transforms.FinalizeTargets(cfg=cfg).map(sample)
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
        _ = btx.data.transforms.FinalizeTargets(
            cfg=btx.data.transforms.AugmentConfig()
        ).map(sample)


def test_finalize_targets_seed2_regression_does_not_fail_inverse_invariant():
    cfg = btx.data.transforms.AugmentConfig()
    sample = btx.data.transforms.InitAugState(
        size=cfg.size, min_px_per_cm=cfg.min_px_per_cm
    ).map(_sample())
    rng = np.random.default_rng(seed=2)
    sample = btx.data.transforms.RandomResizedCrop(cfg).random_map(sample, rng)
    sample = btx.data.transforms.RandomFlip(cfg).random_map(sample, rng)
    sample = btx.data.transforms.RandomRotation(cfg).random_map(sample, rng)
    out = btx.data.transforms.FinalizeTargets(cfg=cfg).map(sample)
    assert np.all(np.isfinite(out["t_aug_from_orig"]))
    assert np.all(np.isfinite(out["t_orig_from_aug"]))


def test_gaussian_heatmap_generates_endpoint_targets_with_expected_peaks():
    sample = _spatial_sample()
    sample["tgt"] = np.array(
        [
            [[1.5, 1.5], [5.5, 9.5]],
            [[13.5, 17.5], [21.5, 25.5]],
        ],
        dtype=np.float32,
    )
    out = btx.data.transforms.GaussianHeatmap(
        image_size=256, heatmap_size=64, sigma=0.5
    ).map(sample)
    heatmap = out["heatmap_tgt"]

    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (4, 64, 64)
    np.testing.assert_allclose(np.max(heatmap, axis=(1, 2)), 1.0, atol=1e-6)
    peak_hw = [
        np.unravel_index(int(np.argmax(heatmap[c])), heatmap[c].shape) for c in range(4)
    ]
    assert peak_hw == [
        (0, 0),  # width p0
        (2, 1),  # width p1
        (4, 3),  # length p0
        (6, 5),  # length p1
    ]


def test_gaussian_heatmap_init_asserts_on_nondivisible_size_ratio():
    with pytest.raises(AssertionError):
        _ = btx.data.transforms.GaussianHeatmap(image_size=255, heatmap_size=64)


def test_gaussian_heatmap_init_asserts_on_nonpositive_sigma():
    with pytest.raises(AssertionError):
        _ = btx.data.transforms.GaussianHeatmap(sigma=0.0)


def test_affine_composition_order_is_not_commutative():
    points = np.array(
        [
            [[10.0, 20.0], [30.0, 40.0]],
            [[80.0, 100.0], [160.0, 190.0]],
        ],
        dtype=np.float32,
    )
    rot = btx.data.transforms.get_rotation_affine(90.0, size=256)
    flip = btx.data.transforms.get_hflip_affine(size=256)
    rot_then_flip = btx.data.transforms.apply_affine_to_points(flip @ rot, points)
    flip_then_rot = btx.data.transforms.apply_affine_to_points(rot @ flip, points)
    assert not np.allclose(rot_then_flip, flip_then_rot, atol=1e-6)


@given(
    h=st.integers(min_value=64, max_value=512),
    w=st.integers(min_value=64, max_value=512),
    scale=st.floats(min_value=0.5, max_value=1.0),
    ratio=st.floats(min_value=0.75, max_value=1.333),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_random_resized_crop_respects_scale_and_ratio(h, w, scale, ratio, seed):
    hypothesis.assume(h != w)
    # Skip cases where the target crop can't physically fit in the image.
    area = float(h * w)
    target_w = int(round(np.sqrt(area * scale * ratio)))
    target_h = int(round(np.sqrt(area * scale / ratio)))
    hypothesis.assume(target_w <= w and target_h <= h)
    cfg = btx.data.transforms.AugmentConfig(
        crop_scale_min=scale,
        crop_scale_max=scale,
        crop_ratio_min=ratio,
        crop_ratio_max=ratio,
    )
    sample = _spatial_sample()
    sample["img"] = np.zeros((h, w, 3), dtype=np.float32)
    out = btx.data.transforms.RandomResizedCrop(cfg).random_map(
        sample, np.random.default_rng(seed=seed)
    )
    t = out["t_aug_from_orig"]
    crop_w = 256.0 / t[0, 0]
    crop_h = 256.0 / t[1, 1]
    actual_scale = (crop_w * crop_h) / (w * h)
    actual_ratio = crop_w / crop_h
    # Tolerance accounts for rounding crop dims to integers.
    assert abs(actual_scale - scale) < 0.05, (
        f"scale: expected {scale:.3f}, got {actual_scale:.3f} (image {w}x{h})"
    )
    assert abs(actual_ratio - ratio) < 0.05, (
        f"ratio: expected {ratio:.3f}, got {actual_ratio:.3f} (image {w}x{h})"
    )


def test_random_resized_crop_is_identity_at_full_scale_square_ratio():
    cfg = btx.data.transforms.AugmentConfig(
        crop_scale_min=1.0,
        crop_scale_max=1.0,
        crop_ratio_min=1.0,
        crop_ratio_max=1.0,
    )
    sample = _spatial_sample()
    out = btx.data.transforms.RandomResizedCrop(cfg).random_map(
        sample,
        np.random.default_rng(seed=17),
    )
    np.testing.assert_allclose(out["t_aug_from_orig"], np.eye(3), atol=1e-6)


def test_random_flip_prob_zero_is_identity():
    cfg = btx.data.transforms.AugmentConfig(hflip_prob=0.0, vflip_prob=0.0)
    sample = _spatial_sample()
    out = btx.data.transforms.RandomFlip(cfg).random_map(
        sample, np.random.default_rng(seed=17)
    )
    np.testing.assert_allclose(out["t_aug_from_orig"], np.eye(3), atol=1e-6)


def test_random_flip_prob_one_applies_hflip_when_vflip_disabled():
    cfg = btx.data.transforms.AugmentConfig(hflip_prob=1.0, vflip_prob=0.0)
    sample = _spatial_sample()
    out = btx.data.transforms.RandomFlip(cfg).random_map(
        sample, np.random.default_rng(seed=17)
    )
    expected = btx.data.transforms.get_hflip_affine(size=256)
    np.testing.assert_allclose(out["t_aug_from_orig"], expected, atol=1e-6)


def test_random_flip_prob_one_applies_vflip_when_hflip_disabled():
    cfg = btx.data.transforms.AugmentConfig(hflip_prob=0.0, vflip_prob=1.0)
    sample = _spatial_sample()
    out = btx.data.transforms.RandomFlip(cfg).random_map(
        sample, np.random.default_rng(seed=17)
    )
    expected = btx.data.transforms.get_vflip_affine(size=256)
    np.testing.assert_allclose(out["t_aug_from_orig"], expected, atol=1e-6)


def test_random_rotation_prob_zero_is_identity():
    cfg = btx.data.transforms.AugmentConfig(rotation_prob=0.0)
    sample = _spatial_sample()
    out = btx.data.transforms.RandomRotation(cfg).random_map(
        sample, np.random.default_rng(seed=17)
    )
    np.testing.assert_allclose(out["t_aug_from_orig"], np.eye(3), atol=1e-6)


def test_random_rotation_prob_one_is_non_identity():
    cfg = btx.data.transforms.AugmentConfig(rotation_prob=1.0)
    sample = _spatial_sample()
    out = btx.data.transforms.RandomRotation(cfg).random_map(
        sample, np.random.default_rng(seed=17)
    )
    assert not np.allclose(out["t_aug_from_orig"], np.eye(3), atol=1e-6)


def test_color_jitter_zero_strength_is_noop():
    cfg = btx.data.transforms.AugmentConfig(
        brightness=0.0,
        contrast=0.0,
        saturation=0.0,
        hue=0.0,
    )
    sample = _sample()
    sample["img"] = np.full((256, 256, 3), 0.25, dtype=np.float32)
    out = btx.data.transforms.ColorJitter(cfg).random_map(
        sample, np.random.default_rng(seed=17)
    )
    np.testing.assert_allclose(out["img"], sample["img"], atol=1e-6)


def test_color_jitter_prob_zero_is_noop_even_with_nonzero_strength():
    cfg = btx.data.transforms.AugmentConfig(
        brightness=0.5,
        contrast=0.4,
        saturation=0.4,
        hue=0.1,
        color_jitter_prob=0.0,
    )
    sample = _sample()
    sample["img"] = np.array(
        [
            [[0.8, 0.2, 0.1], [0.7, 0.3, 0.2]],
            [[0.6, 0.4, 0.2], [0.5, 0.4, 0.3]],
        ],
        dtype=np.float32,
    )
    out = btx.data.transforms.ColorJitter(cfg).random_map(
        sample, np.random.default_rng(seed=17)
    )
    np.testing.assert_allclose(out["img"], sample["img"], atol=1e-6)


def test_color_jitter_nonzero_strength_produces_valid_float_image():
    cfg = btx.data.transforms.AugmentConfig(
        brightness=0.5,
        contrast=0.4,
        saturation=0.4,
        hue=0.1,
    )
    sample = _sample()
    sample["img"] = np.array(
        [
            [[0.8, 0.2, 0.1], [0.7, 0.3, 0.2]],
            [[0.6, 0.4, 0.2], [0.5, 0.4, 0.3]],
        ],
        dtype=np.float32,
    )
    out = btx.data.transforms.ColorJitter(cfg).random_map(
        sample, np.random.default_rng(seed=17)
    )
    img = out["img"]
    assert isinstance(img, np.ndarray)
    assert img.dtype == np.float32
    assert img.shape == (2, 2, 3)
    assert np.all(np.isfinite(img))
    assert np.all(img >= 0.0)
    assert np.all(img <= 1.0)


def test_resize_tracks_affine_and_keeps_original_points():
    sample = {
        "img": np.zeros((100, 200, 3), dtype=np.float32),
        "points_px": np.array(
            [
                [[10.0, 20.0], [30.0, 40.0]],
                [[100.0, 60.0], [140.0, 80.0]],
            ],
            dtype=np.float32,
        ),
        "scalebar_px": np.array([[5.0, 5.0], [25.0, 5.0]], dtype=np.float32),
        "loss_mask": np.array([1.0, 1.0], dtype=np.float32),
    }
    out = btx.data.transforms.InitAugState(size=256).map(sample)
    points_before = np.array(out["points_px"], copy=True)
    out = btx.data.transforms.Resize(size=256).map(out)

    expected = btx.data.transforms.get_crop_resize_affine(
        x0=0.0,
        y0=0.0,
        crop_w=200.0,
        crop_h=100.0,
        size=256,
    )
    np.testing.assert_allclose(out["t_aug_from_orig"], expected, atol=1e-6)
    assert out["img"].shape == (256, 256, 3)
    np.testing.assert_allclose(out["points_px"], points_before, atol=1e-8)


def test_normalize_uses_imagenet_mean_std_by_default():
    sample = {"img": np.array([[[0.485, 0.456, 0.406]]], dtype=np.float32)}
    out = btx.data.transforms.Normalize().map(sample)
    np.testing.assert_allclose(out["img"], np.zeros((1, 1, 3), dtype=np.float32))
