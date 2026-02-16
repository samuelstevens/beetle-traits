import importlib.util
import pathlib

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import btx.data
import btx.data.transforms
import btx.heatmap

train_fpath = pathlib.Path(__file__).resolve().parents[1] / "train.py"
spec = importlib.util.spec_from_file_location("train_script", train_fpath)
assert spec is not None and spec.loader is not None
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)

COORD_OBJECTIVE = train.ObjectiveConfig(kind="coords")
HEATMAP_OBJECTIVE = train.ObjectiveConfig(
    kind="heatmap",
    heatmap_size=64,
    sigma=2.0,
    heatmap_loss="mse",
)


class ConstantModel(eqx.Module):
    pred_l22: jax.Array

    def __call__(self, _img_hwc):
        return self.pred_l22


class ConstantHeatmapModel(eqx.Module):
    pred_chw: jax.Array

    def __call__(self, _img_hwc):
        return self.pred_chw


class FrozenBackboneWithHead(eqx.Module):
    vit: eqx.nn.Linear
    head: eqx.nn.Linear


class FrozenBackboneWithDecoder(eqx.Module):
    vit: eqx.nn.Linear
    decoder: eqx.nn.Linear


def _partition_model(model: eqx.Module):
    spec = jax.tree_util.tree_map(eqx.is_array, model)
    return eqx.partition(model, spec)


def test_get_transforms_orders_finalize_before_normalize():
    cfg = btx.data.transforms.AugmentConfig()

    train_tfms = btx.data.transforms.make_transforms(cfg, is_train=True)
    eval_tfms = btx.data.transforms.make_transforms(cfg, is_train=False)

    train_names = [t.__class__.__name__ for t in train_tfms]
    eval_names = [t.__class__.__name__ for t in eval_tfms]

    assert train_names == [
        "DecodeRGB",
        "InitAugState",
        "RandomResizedCrop",
        "RandomFlip",
        "RandomRotation",
        "ColorJitter",
        "FinalizeTargets",
        "Normalize",
    ]
    assert eval_names == [
        "DecodeRGB",
        "InitAugState",
        "Resize",
        "FinalizeTargets",
        "Normalize",
    ]


def test_get_transforms_can_disable_normalization():
    cfg = btx.data.transforms.AugmentConfig(normalize=False)
    train_tfms = btx.data.transforms.make_transforms(cfg, is_train=True)
    eval_tfms = btx.data.transforms.make_transforms(cfg, is_train=False)

    train_names = [t.__class__.__name__ for t in train_tfms]
    eval_names = [t.__class__.__name__ for t in eval_tfms]

    assert train_names == [
        "DecodeRGB",
        "InitAugState",
        "RandomResizedCrop",
        "RandomFlip",
        "RandomRotation",
        "ColorJitter",
        "FinalizeTargets",
    ]
    assert eval_names == [
        "DecodeRGB",
        "InitAugState",
        "Resize",
        "FinalizeTargets",
    ]


def test_get_transforms_never_includes_gaussian_heatmap():
    cfg = btx.data.transforms.AugmentConfig()
    train_tfms = btx.data.transforms.make_transforms(cfg, is_train=True)
    eval_tfms = btx.data.transforms.make_transforms(cfg, is_train=False)

    train_names = [t.__class__.__name__ for t in train_tfms]
    eval_names = [t.__class__.__name__ for t in eval_tfms]
    assert "GaussianHeatmap" not in train_names
    assert "GaussianHeatmap" not in eval_names


def test_get_augment_for_dataset_returns_dataset_specific_configs():
    cfg = train.Config(
        aug_hawaii=btx.data.transforms.AugmentConfig(crop_scale_min=0.5),
        aug_beetlepalooza=btx.data.transforms.AugmentConfig(crop_scale_min=0.8),
        aug_biorepo=btx.data.transforms.AugmentConfig(crop_scale_min=0.9),
    )

    got_hawaii = train.get_aug_for_dataset(cfg, btx.data.HawaiiConfig())
    got_beetle = train.get_aug_for_dataset(cfg, btx.data.BeetlePaloozaConfig())
    got_biorepo = train.get_aug_for_dataset(cfg, btx.data.BioRepoConfig())

    assert got_hawaii.crop_scale_min == 0.5
    assert got_beetle.crop_scale_min == 0.8
    assert got_biorepo.crop_scale_min == 0.9


def test_objective_config_asserts_on_invalid_fields():
    with np.testing.assert_raises(Exception):
        _ = train.ObjectiveConfig(kind="bad")
    with np.testing.assert_raises_regex(AssertionError, "positive heatmap_size"):
        _ = train.ObjectiveConfig(kind="heatmap", heatmap_size=0)
    with np.testing.assert_raises_regex(AssertionError, "positive sigma"):
        _ = train.ObjectiveConfig(kind="heatmap", sigma=0.0)


def test_objective_config_defaults_heatmap_loss_to_ce():
    cfg = train.ObjectiveConfig(kind="heatmap")
    assert cfg.heatmap_loss == "ce"


def test_objective_config_asserts_on_invalid_heatmap_loss():
    with np.testing.assert_raises(Exception):
        _ = train.ObjectiveConfig(kind="heatmap", heatmap_loss="bad")


def test_loss_and_aux_asserts_on_missing_required_batch_key():
    model = ConstantModel(pred_l22=jnp.zeros((2, 2, 2), dtype=jnp.float32))
    diff_model, static_model = _partition_model(model)
    batch = {
        "img": jnp.zeros((1, 256, 256, 3), dtype=jnp.float32),
        "tgt": jnp.zeros((1, 2, 2, 2), dtype=jnp.float32),
        "scalebar_px": jnp.array([[[0.0, 0.0], [10.0, 0.0]]], dtype=jnp.float32),
        "loss_mask": jnp.ones((1, 2), dtype=jnp.float32),
        "scalebar_valid": jnp.array([False]),
        "t_orig_from_aug": jnp.eye(3, dtype=jnp.float32)[None, :, :],
        "oob_points_frac": jnp.array([0.25], dtype=jnp.float32),
    }

    with np.testing.assert_raises_regex(AssertionError, "points_px"):
        _ = train.loss_and_aux(
            diff_model, static_model, batch, objective_cfg=COORD_OBJECTIVE
        )


def test_loss_and_aux_masks_cm_metrics_and_reports_oob_points_frac():
    model = ConstantModel(pred_l22=jnp.zeros((2, 2, 2), dtype=jnp.float32))
    diff_model, static_model = _partition_model(model)
    batch = {
        "img": jnp.zeros((1, 256, 256, 3), dtype=jnp.float32),
        "tgt": jnp.zeros((1, 2, 2, 2), dtype=jnp.float32),
        "points_px": jnp.zeros((1, 2, 2, 2), dtype=jnp.float32),
        "scalebar_px": jnp.array([[[0.0, 0.0], [10.0, 0.0]]], dtype=jnp.float32),
        "loss_mask": jnp.ones((1, 2), dtype=jnp.float32),
        "scalebar_valid": jnp.array([False]),
        "t_orig_from_aug": jnp.eye(3, dtype=jnp.float32)[None, :, :],
        "oob_points_frac": jnp.array([0.25], dtype=jnp.float32),
    }

    loss, aux = train.loss_and_aux(
        diff_model, static_model, batch, objective_cfg=COORD_OBJECTIVE
    )

    assert float(loss) == 0.0
    assert np.isnan(np.asarray(aux.point_err_cm)).all()
    assert np.isnan(np.asarray(aux.line_err_cm)).all()
    np.testing.assert_allclose(np.asarray(aux.oob_points_frac), np.array(0.25))


def test_loss_and_aux_uses_order_invariant_endpoint_matching():
    pred = jnp.array(
        [
            [[10.0, 20.0], [30.0, 20.0]],
            [[50.0, 80.0], [90.0, 80.0]],
        ],
        dtype=jnp.float32,
    )
    model = ConstantModel(pred_l22=pred)
    diff_model, static_model = _partition_model(model)

    points_px = np.asarray(pred)[None, :, :, :].copy()
    points_px[:, 0] = points_px[:, 0, ::-1, :]
    batch = {
        "img": jnp.zeros((1, 256, 256, 3), dtype=jnp.float32),
        "tgt": pred[None, :, :, :],
        "points_px": jnp.asarray(points_px, dtype=jnp.float32),
        "scalebar_px": jnp.array([[[0.0, 0.0], [5.0, 0.0]]], dtype=jnp.float32),
        "loss_mask": jnp.ones((1, 2), dtype=jnp.float32),
        "scalebar_valid": jnp.array([True]),
        "t_orig_from_aug": jnp.eye(3, dtype=jnp.float32)[None, :, :],
        "oob_points_frac": jnp.array([0.0], dtype=jnp.float32),
    }

    _, aux = train.loss_and_aux(
        diff_model, static_model, batch, objective_cfg=COORD_OBJECTIVE
    )
    np.testing.assert_allclose(np.asarray(aux.point_err_cm), 0.0, atol=1e-7)
    np.testing.assert_allclose(np.asarray(aux.line_err_cm), 0.0, atol=1e-7)


def test_loss_and_aux_weights_global_loss_by_active_targets():
    model = ConstantModel(pred_l22=jnp.zeros((2, 2, 2), dtype=jnp.float32))
    diff_model, static_model = _partition_model(model)

    tgt = jnp.zeros((2, 2, 2, 2), dtype=jnp.float32)
    tgt = tgt.at[0].set(1.0)
    tgt = tgt.at[1, 0].set(2.0)
    batch = {
        "img": jnp.zeros((2, 256, 256, 3), dtype=jnp.float32),
        "tgt": tgt,
        "points_px": jnp.zeros((2, 2, 2, 2), dtype=jnp.float32),
        "scalebar_px": jnp.array(
            [[[0.0, 0.0], [10.0, 0.0]], [[0.0, 0.0], [10.0, 0.0]]],
            dtype=jnp.float32,
        ),
        "loss_mask": jnp.array([[1.0, 1.0], [1.0, 0.0]], dtype=jnp.float32),
        "scalebar_valid": jnp.array([False, False]),
        "t_orig_from_aug": jnp.tile(
            jnp.eye(3, dtype=jnp.float32)[None, :, :], (2, 1, 1)
        ),
        "oob_points_frac": jnp.array([0.0, 0.0], dtype=jnp.float32),
    }

    loss, aux = train.loss_and_aux(
        diff_model, static_model, batch, objective_cfg=COORD_OBJECTIVE
    )

    np.testing.assert_allclose(np.asarray(aux.sample_loss), np.array([1.0, 4.0]))
    np.testing.assert_allclose(np.asarray(loss), np.array(2.0))


def test_loss_and_aux_marks_sample_loss_nan_when_sample_is_fully_masked():
    model = ConstantModel(pred_l22=jnp.zeros((2, 2, 2), dtype=jnp.float32))
    diff_model, static_model = _partition_model(model)

    tgt = jnp.zeros((2, 2, 2, 2), dtype=jnp.float32)
    tgt = tgt.at[0].set(1.0)
    tgt = tgt.at[1].set(5.0)
    batch = {
        "img": jnp.zeros((2, 256, 256, 3), dtype=jnp.float32),
        "tgt": tgt,
        "points_px": jnp.zeros((2, 2, 2, 2), dtype=jnp.float32),
        "scalebar_px": jnp.array(
            [[[0.0, 0.0], [10.0, 0.0]], [[0.0, 0.0], [10.0, 0.0]]],
            dtype=jnp.float32,
        ),
        "loss_mask": jnp.array([[1.0, 1.0], [0.0, 0.0]], dtype=jnp.float32),
        "scalebar_valid": jnp.array([False, False]),
        "t_orig_from_aug": jnp.tile(
            jnp.eye(3, dtype=jnp.float32)[None, :, :], (2, 1, 1)
        ),
        "oob_points_frac": jnp.array([0.0, 0.0], dtype=jnp.float32),
    }

    loss, aux = train.loss_and_aux(
        diff_model, static_model, batch, objective_cfg=COORD_OBJECTIVE
    )

    sample_loss = np.asarray(aux.sample_loss)
    np.testing.assert_allclose(sample_loss[0], np.array(1.0))
    assert np.isnan(sample_loss[1])
    np.testing.assert_allclose(np.asarray(loss), np.array(1.0))


def test_loss_and_aux_returns_zero_loss_and_zero_grads_when_all_targets_masked():
    model = ConstantModel(pred_l22=jnp.ones((2, 2, 2), dtype=jnp.float32))
    diff_model, static_model = _partition_model(model)

    batch = {
        "img": jnp.zeros((2, 256, 256, 3), dtype=jnp.float32),
        "tgt": jnp.ones((2, 2, 2, 2), dtype=jnp.float32),
        "points_px": jnp.zeros((2, 2, 2, 2), dtype=jnp.float32),
        "scalebar_px": jnp.array(
            [[[0.0, 0.0], [10.0, 0.0]], [[0.0, 0.0], [10.0, 0.0]]],
            dtype=jnp.float32,
        ),
        "loss_mask": jnp.zeros((2, 2), dtype=jnp.float32),
        "scalebar_valid": jnp.array([False, False]),
        "t_orig_from_aug": jnp.tile(
            jnp.eye(3, dtype=jnp.float32)[None, :, :], (2, 1, 1)
        ),
        "oob_points_frac": jnp.array([0.0, 0.0], dtype=jnp.float32),
    }

    def loss_only(diff_model, static_model):
        loss, _ = train.loss_and_aux(
            diff_model, static_model, batch, objective_cfg=COORD_OBJECTIVE
        )
        return loss

    loss_grad_fn = eqx.filter_value_and_grad(loss_only)
    loss, grads = loss_grad_fn(diff_model, static_model)
    _, aux = train.loss_and_aux(
        diff_model, static_model, batch, objective_cfg=COORD_OBJECTIVE
    )

    np.testing.assert_allclose(np.asarray(loss), np.array(0.0))
    grad = np.asarray(grads.pred_l22)
    assert np.isfinite(grad).all()
    np.testing.assert_allclose(grad, 0.0)
    assert np.isnan(np.asarray(aux.sample_loss)).all()


def test_loss_and_aux_uses_generated_heatmap_targets_from_tgt():
    model = ConstantHeatmapModel(pred_chw=jnp.zeros((4, 64, 64), dtype=jnp.float32))
    diff_model, static_model = _partition_model(model)

    batch = {
        "img": jnp.zeros((1, 256, 256, 3), dtype=jnp.float32),
        "tgt": jnp.full((1, 2, 2, 2), 999.0, dtype=jnp.float32),
        "points_px": jnp.zeros((1, 2, 2, 2), dtype=jnp.float32),
        "scalebar_px": jnp.array([[[0.0, 0.0], [10.0, 0.0]]], dtype=jnp.float32),
        "loss_mask": jnp.ones((1, 2), dtype=jnp.float32),
        "scalebar_valid": jnp.array([False]),
        "t_orig_from_aug": jnp.eye(3, dtype=jnp.float32)[None, :, :],
        "oob_points_frac": jnp.array([0.0], dtype=jnp.float32),
    }

    loss, aux = train.loss_and_aux(
        diff_model, static_model, batch, objective_cfg=HEATMAP_OBJECTIVE
    )

    np.testing.assert_allclose(np.asarray(loss), np.array(0.0))
    np.testing.assert_allclose(np.asarray(aux.sample_loss), np.array([0.0]))
    assert aux.preds.shape == (1, 2, 2, 2)
    assert np.isfinite(np.asarray(aux.preds)).all()


def test_loss_and_aux_heatmap_reports_uniform_map_diagnostics():
    model = ConstantHeatmapModel(pred_chw=jnp.zeros((4, 64, 64), dtype=jnp.float32))
    diff_model, static_model = _partition_model(model)
    batch = {
        "img": jnp.zeros((1, 256, 256, 3), dtype=jnp.float32),
        "tgt": jnp.full((1, 2, 2, 2), 128.0, dtype=jnp.float32),
        "points_px": jnp.zeros((1, 2, 2, 2), dtype=jnp.float32),
        "scalebar_px": jnp.array([[[0.0, 0.0], [10.0, 0.0]]], dtype=jnp.float32),
        "loss_mask": jnp.ones((1, 2), dtype=jnp.float32),
        "scalebar_valid": jnp.array([False]),
        "t_orig_from_aug": jnp.eye(3, dtype=jnp.float32)[None, :, :],
        "oob_points_frac": jnp.array([0.0], dtype=jnp.float32),
    }
    _, aux = train.loss_and_aux(
        diff_model, static_model, batch, objective_cfg=HEATMAP_OBJECTIVE
    )

    assert aux.heatmap_max_logit.shape == (1, 4)
    assert aux.heatmap_entropy.shape == (1, 4)
    assert aux.heatmap_near_uniform.shape == (1, 4)
    np.testing.assert_allclose(np.asarray(aux.heatmap_max_logit), 0.0, atol=1e-7)
    np.testing.assert_allclose(
        np.asarray(aux.heatmap_entropy),
        np.log(np.array(64 * 64, dtype=np.float32)),
        atol=1e-3,
    )
    np.testing.assert_allclose(np.asarray(aux.heatmap_near_uniform), 1.0, atol=1e-7)

    metrics = aux.metrics()
    for channel_name in btx.heatmap.CHANNEL_NAMES:
        key = f"heatmap_near_uniform_frac_{channel_name}"
        assert key in metrics
        np.testing.assert_allclose(np.asarray(metrics[key]), 1.0, atol=1e-7)


def test_loss_and_aux_heatmap_reports_spiky_map_diagnostics():
    pred_chw = -10.0 * jnp.ones((4, 64, 64), dtype=jnp.float32)
    pred_chw = pred_chw.at[0, 10, 20].set(10.0)
    pred_chw = pred_chw.at[1, 11, 21].set(10.0)
    pred_chw = pred_chw.at[2, 12, 22].set(10.0)
    pred_chw = pred_chw.at[3, 13, 23].set(10.0)
    model = ConstantHeatmapModel(pred_chw=pred_chw)
    diff_model, static_model = _partition_model(model)
    batch = {
        "img": jnp.zeros((1, 256, 256, 3), dtype=jnp.float32),
        "tgt": jnp.full((1, 2, 2, 2), 128.0, dtype=jnp.float32),
        "points_px": jnp.zeros((1, 2, 2, 2), dtype=jnp.float32),
        "scalebar_px": jnp.array([[[0.0, 0.0], [10.0, 0.0]]], dtype=jnp.float32),
        "loss_mask": jnp.ones((1, 2), dtype=jnp.float32),
        "scalebar_valid": jnp.array([False]),
        "t_orig_from_aug": jnp.eye(3, dtype=jnp.float32)[None, :, :],
        "oob_points_frac": jnp.array([0.0], dtype=jnp.float32),
    }
    _, aux = train.loss_and_aux(
        diff_model, static_model, batch, objective_cfg=HEATMAP_OBJECTIVE
    )

    np.testing.assert_allclose(np.asarray(aux.heatmap_max_logit), 10.0, atol=1e-7)
    np.testing.assert_allclose(np.asarray(aux.heatmap_near_uniform), 0.0, atol=1e-7)
    assert np.all(np.asarray(aux.heatmap_entropy) < 1.0)

    metrics = aux.metrics()
    for channel_name in btx.heatmap.CHANNEL_NAMES:
        max_key = f"heatmap_max_logit_{channel_name}"
        ent_key = f"heatmap_entropy_{channel_name}"
        uni_key = f"heatmap_near_uniform_frac_{channel_name}"
        assert max_key in metrics
        assert ent_key in metrics
        assert uni_key in metrics
        np.testing.assert_allclose(np.asarray(metrics[max_key]), 10.0, atol=1e-7)
        np.testing.assert_allclose(np.asarray(metrics[uni_key]), 0.0, atol=1e-7)
        assert np.all(np.asarray(metrics[ent_key]) < 1.0)


def test_loss_and_aux_coords_sets_heatmap_diagnostics_to_nan():
    model = ConstantModel(pred_l22=jnp.zeros((2, 2, 2), dtype=jnp.float32))
    diff_model, static_model = _partition_model(model)
    batch = {
        "img": jnp.zeros((1, 256, 256, 3), dtype=jnp.float32),
        "tgt": jnp.zeros((1, 2, 2, 2), dtype=jnp.float32),
        "points_px": jnp.zeros((1, 2, 2, 2), dtype=jnp.float32),
        "scalebar_px": jnp.array([[[0.0, 0.0], [10.0, 0.0]]], dtype=jnp.float32),
        "loss_mask": jnp.ones((1, 2), dtype=jnp.float32),
        "scalebar_valid": jnp.array([False]),
        "t_orig_from_aug": jnp.eye(3, dtype=jnp.float32)[None, :, :],
        "oob_points_frac": jnp.array([0.0], dtype=jnp.float32),
    }
    _, aux = train.loss_and_aux(
        diff_model, static_model, batch, objective_cfg=COORD_OBJECTIVE
    )

    assert np.isnan(np.asarray(aux.heatmap_max_logit)).all()
    assert np.isnan(np.asarray(aux.heatmap_entropy)).all()
    assert np.isnan(np.asarray(aux.heatmap_near_uniform)).all()


def test_loss_and_aux_heatmap_weights_global_loss_by_active_elements():
    model = ConstantHeatmapModel(pred_chw=jnp.zeros((4, 64, 64), dtype=jnp.float32))
    diff_model, static_model = _partition_model(model)
    batch = {
        "img": jnp.zeros((2, 256, 256, 3), dtype=jnp.float32),
        "tgt": jnp.array(
            [
                [[[128.0, 128.0], [128.0, 128.0]], [[128.0, 128.0], [128.0, 128.0]]],
                [[[999.0, 999.0], [999.0, 999.0]], [[999.0, 999.0], [999.0, 999.0]]],
            ],
            dtype=jnp.float32,
        ),
        "points_px": jnp.zeros((2, 2, 2, 2), dtype=jnp.float32),
        "scalebar_px": jnp.array(
            [[[0.0, 0.0], [10.0, 0.0]], [[0.0, 0.0], [10.0, 0.0]]],
            dtype=jnp.float32,
        ),
        "loss_mask": jnp.array([[1.0, 1.0], [1.0, 0.0]], dtype=jnp.float32),
        "scalebar_valid": jnp.array([False, False]),
        "t_orig_from_aug": jnp.tile(
            jnp.eye(3, dtype=jnp.float32)[None, :, :], (2, 1, 1)
        ),
        "oob_points_frac": jnp.array([0.0, 0.0], dtype=jnp.float32),
    }
    loss, aux = train.loss_and_aux(
        diff_model, static_model, batch, objective_cfg=HEATMAP_OBJECTIVE
    )
    heatmap_cfg = btx.heatmap.Config(image_size=256, heatmap_size=64, sigma=2.0)
    pred0 = jnp.zeros((4, 64, 64), dtype=jnp.float32)
    tgt0 = btx.heatmap.make_targets(batch["tgt"][0], cfg=heatmap_cfg)
    tgt1 = btx.heatmap.make_targets(batch["tgt"][1], cfg=heatmap_cfg)
    loss0 = btx.heatmap.heatmap_loss(
        pred0, tgt0, batch["loss_mask"][0], cfg=heatmap_cfg
    )
    loss1 = btx.heatmap.heatmap_loss(
        pred0, tgt1, batch["loss_mask"][1], cfg=heatmap_cfg
    )
    active = np.array([2.0 * 64 * 64, 1.0 * 64 * 64], dtype=np.float32)
    expected_loss = (float(loss0) * active[0] + float(loss1) * active[1]) / float(
        active.sum()
    )

    np.testing.assert_allclose(np.asarray(aux.sample_loss), np.array([loss0, loss1]))
    np.testing.assert_allclose(np.asarray(loss), np.array(expected_loss), rtol=1e-5)


def test_loss_and_aux_heatmap_ce_weights_global_loss_by_active_lines():
    model = ConstantHeatmapModel(pred_chw=jnp.zeros((4, 64, 64), dtype=jnp.float32))
    diff_model, static_model = _partition_model(model)
    batch = {
        "img": jnp.zeros((2, 256, 256, 3), dtype=jnp.float32),
        "tgt": jnp.array(
            [
                [[[40.0, 40.0], [80.0, 80.0]], [[120.0, 120.0], [160.0, 160.0]]],
                [[[32.0, 64.0], [96.0, 64.0]], [[128.0, 192.0], [192.0, 192.0]]],
            ],
            dtype=jnp.float32,
        ),
        "points_px": jnp.zeros((2, 2, 2, 2), dtype=jnp.float32),
        "scalebar_px": jnp.array(
            [[[0.0, 0.0], [10.0, 0.0]], [[0.0, 0.0], [10.0, 0.0]]],
            dtype=jnp.float32,
        ),
        "loss_mask": jnp.array([[1.0, 1.0], [1.0, 0.0]], dtype=jnp.float32),
        "scalebar_valid": jnp.array([False, False]),
        "t_orig_from_aug": jnp.tile(
            jnp.eye(3, dtype=jnp.float32)[None, :, :], (2, 1, 1)
        ),
        "oob_points_frac": jnp.array([0.0, 0.0], dtype=jnp.float32),
    }
    objective_cfg = train.ObjectiveConfig(
        kind="heatmap",
        heatmap_size=64,
        sigma=2.0,
        heatmap_loss="ce",
    )

    loss, aux = train.loss_and_aux(
        diff_model, static_model, batch, objective_cfg=objective_cfg
    )
    heatmap_cfg = btx.heatmap.Config(image_size=256, heatmap_size=64, sigma=2.0)
    pred0 = jnp.zeros((4, 64, 64), dtype=jnp.float32)
    tgt0 = btx.heatmap.make_targets(batch["tgt"][0], cfg=heatmap_cfg)
    tgt1 = btx.heatmap.make_targets(batch["tgt"][1], cfg=heatmap_cfg)
    loss0 = btx.heatmap.heatmap_ce_loss(
        pred0, tgt0, batch["loss_mask"][0], cfg=heatmap_cfg
    )
    loss1 = btx.heatmap.heatmap_ce_loss(
        pred0, tgt1, batch["loss_mask"][1], cfg=heatmap_cfg
    )
    active = np.array([2.0, 1.0], dtype=np.float32)
    expected_loss = (float(loss0) * active[0] + float(loss1) * active[1]) / float(
        active.sum()
    )

    np.testing.assert_allclose(np.asarray(aux.sample_loss), np.array([loss0, loss1]))
    np.testing.assert_allclose(np.asarray(loss), np.array(expected_loss), rtol=1e-5)


def test_loss_and_aux_heatmap_rejects_precomputed_heatmap_targets_in_batch():
    model = ConstantHeatmapModel(pred_chw=jnp.zeros((4, 64, 64), dtype=jnp.float32))
    diff_model, static_model = _partition_model(model)
    batch = {
        "img": jnp.zeros((1, 256, 256, 3), dtype=jnp.float32),
        "tgt": jnp.full((1, 2, 2, 2), 999.0, dtype=jnp.float32),
        "heatmap_tgt": jnp.zeros((1, 4, 64, 64), dtype=jnp.float32),
        "points_px": jnp.zeros((1, 2, 2, 2), dtype=jnp.float32),
        "scalebar_px": jnp.array([[[0.0, 0.0], [10.0, 0.0]]], dtype=jnp.float32),
        "loss_mask": jnp.ones((1, 2), dtype=jnp.float32),
        "scalebar_valid": jnp.array([False]),
        "t_orig_from_aug": jnp.eye(3, dtype=jnp.float32)[None, :, :],
        "oob_points_frac": jnp.array([0.0], dtype=jnp.float32),
    }

    with np.testing.assert_raises_regex(
        AssertionError, "no precomputed heatmap targets"
    ):
        _ = train.loss_and_aux(
            diff_model, static_model, batch, objective_cfg=HEATMAP_OBJECTIVE
        )


def test_loss_and_aux_heatmap_checks_cfg_heatmap_size_matches_batch():
    model = ConstantHeatmapModel(pred_chw=jnp.zeros((4, 64, 64), dtype=jnp.float32))
    diff_model, static_model = _partition_model(model)
    batch = {
        "img": jnp.zeros((1, 256, 256, 3), dtype=jnp.float32),
        "tgt": jnp.full((1, 2, 2, 2), 999.0, dtype=jnp.float32),
        "points_px": jnp.zeros((1, 2, 2, 2), dtype=jnp.float32),
        "scalebar_px": jnp.array([[[0.0, 0.0], [10.0, 0.0]]], dtype=jnp.float32),
        "loss_mask": jnp.ones((1, 2), dtype=jnp.float32),
        "scalebar_valid": jnp.array([False]),
        "t_orig_from_aug": jnp.eye(3, dtype=jnp.float32)[None, :, :],
        "oob_points_frac": jnp.array([0.0], dtype=jnp.float32),
    }
    heatmap_cfg = train.ObjectiveConfig(kind="heatmap", heatmap_size=32, sigma=2.0)

    with np.testing.assert_raises_regex(AssertionError, "heatmap_size"):
        _ = train.loss_and_aux(
            diff_model, static_model, batch, objective_cfg=heatmap_cfg
        )


def test_get_trainable_filter_spec_freezes_vit_for_head_models():
    k1, k2 = jax.random.split(jax.random.key(seed=0))
    model = FrozenBackboneWithHead(
        vit=eqx.nn.Linear(4, 4, key=k1),
        head=eqx.nn.Linear(4, 2, key=k2),
    )

    spec = train.get_trainable_filter_spec(model)

    assert spec.vit.weight is False
    assert spec.vit.bias is False
    assert spec.head.weight is True
    assert spec.head.bias is True


def test_get_trainable_filter_spec_does_not_require_head_attribute():
    k1, k2 = jax.random.split(jax.random.key(seed=1))
    model = FrozenBackboneWithDecoder(
        vit=eqx.nn.Linear(4, 4, key=k1),
        decoder=eqx.nn.Linear(4, 2, key=k2),
    )

    spec = train.get_trainable_filter_spec(model)

    assert spec.vit.weight is False
    assert spec.vit.bias is False
    assert spec.decoder.weight is True
    assert spec.decoder.bias is True
