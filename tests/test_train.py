import importlib.util
import pathlib

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import btx.data
from btx.data import augment

train_fpath = pathlib.Path(__file__).resolve().parents[1] / "train.py"
spec = importlib.util.spec_from_file_location("train_script", train_fpath)
assert spec is not None and spec.loader is not None
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


class ConstantModel(eqx.Module):
    pred_l22: jax.Array

    def __call__(self, _img_hwc):
        return self.pred_l22


def _partition_model(model: eqx.Module):
    spec = jax.tree_util.tree_map(eqx.is_array, model)
    return eqx.partition(model, spec)


def test_get_transforms_orders_finalize_before_normalize():
    cfg = augment.AugmentConfig()

    train_tfms = train.get_transforms(cfg, train_split=True)
    eval_tfms = train.get_transforms(cfg, train_split=False)

    train_names = [t.__class__.__name__ for t in train_tfms]
    eval_names = [t.__class__.__name__ for t in eval_tfms]

    assert train_names == [
        "DecodeRGB",
        "InitAugState",
        "RandomResizedCrop",
        "RandomFlip",
        "RandomRotation90",
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
    cfg = augment.AugmentConfig(normalize=False)
    train_tfms = train.get_transforms(cfg, train_split=True)
    eval_tfms = train.get_transforms(cfg, train_split=False)

    train_names = [t.__class__.__name__ for t in train_tfms]
    eval_names = [t.__class__.__name__ for t in eval_tfms]

    assert train_names == [
        "DecodeRGB",
        "InitAugState",
        "RandomResizedCrop",
        "RandomFlip",
        "RandomRotation90",
        "ColorJitter",
        "FinalizeTargets",
    ]
    assert eval_names == [
        "DecodeRGB",
        "InitAugState",
        "Resize",
        "FinalizeTargets",
    ]


def test_get_augment_for_dataset_returns_dataset_specific_configs():
    cfg = train.Config(
        augment=augment.AugmentConfig(crop_scale_min=0.6),
        aug_hawaii=augment.AugmentConfig(crop_scale_min=0.5),
        aug_beetlepalooza=augment.AugmentConfig(crop_scale_min=0.8),
        aug_biorepo=augment.AugmentConfig(crop_scale_min=0.9),
    )

    got_hawaii = train.get_aug_for_dataset(cfg, btx.data.HawaiiConfig())
    got_beetle = train.get_aug_for_dataset(cfg, btx.data.BeetlePaloozaConfig())
    got_biorepo = train.get_aug_for_dataset(cfg, btx.data.BioRepoConfig())

    assert got_hawaii.crop_scale_min == 0.5
    assert got_beetle.crop_scale_min == 0.8
    assert got_biorepo.crop_scale_min == 0.9


def test_loss_and_aux_masks_cm_metrics_and_reports_oob_points_frac():
    model = ConstantModel(pred_l22=jnp.zeros((2, 2, 2), dtype=jnp.float32))
    diff_model, static_model = _partition_model(model)
    batch = {
        "img": jnp.zeros((1, 256, 256, 3), dtype=jnp.float32),
        "tgt": jnp.zeros((1, 2, 2, 2), dtype=jnp.float32),
        "points_px": jnp.zeros((1, 2, 2, 2), dtype=jnp.float32),
        "scalebar_px": jnp.array([[[0.0, 0.0], [10.0, 0.0]]], dtype=jnp.float32),
        "loss_mask": jnp.ones((1, 2), dtype=jnp.float32),
        "metric_mask_cm": jnp.zeros((1,), dtype=jnp.float32),
        "t_orig_from_aug": jnp.eye(3, dtype=jnp.float32)[None, :, :],
        "oob_points_frac": jnp.array([0.25], dtype=jnp.float32),
    }

    loss, aux = train.loss_and_aux(diff_model, static_model, batch, min_px_per_cm=1e-6)

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
        "metric_mask_cm": jnp.ones((1,), dtype=jnp.float32),
        "t_orig_from_aug": jnp.eye(3, dtype=jnp.float32)[None, :, :],
        "oob_points_frac": jnp.array([0.0], dtype=jnp.float32),
    }

    _, aux = train.loss_and_aux(diff_model, static_model, batch, min_px_per_cm=1e-6)
    np.testing.assert_allclose(np.asarray(aux.point_err_cm), 0.0, atol=1e-7)
    np.testing.assert_allclose(np.asarray(aux.line_err_cm), 0.0, atol=1e-7)
