"""Tests for split="all" and split="unlabeled" support in dataset configs."""

import pathlib

import numpy as np
import pytest

import btx.data
import btx.data.biorepo
import btx.data.hawaii
import btx.data.utils

# ---------------------------------------------------------------------------
# Paths (same as test_inference.py)
# ---------------------------------------------------------------------------
HAWAII_HF_ROOT = pathlib.Path("/fs/ess/PAS2136/samuelstevens/datasets/hawaii-beetles")
BIOREPO_ROOT = pathlib.Path("/fs/scratch/PAS2136/cain429/Subset-Exp")
BIOREPO_ANN_FPATH = pathlib.Path(
    "/fs/scratch/PAS2136/cain429/biorepo-formatted/annotations.json"
)
BIOREPO_UNLABELED_ANN_FPATH = pathlib.Path(
    "/fs/scratch/PAS2136/cain429/unlabeled_biorepo_annotations.csv"
)

# ---------------------------------------------------------------------------
# Unit tests (no data needed)
# ---------------------------------------------------------------------------


def test_sample_has_split_field():
    assert "split" in btx.data.utils.Sample.__annotations__


def test_hawaii_config_accepts_all():
    btx.data.hawaii.Config(split="all")


def test_biorepo_config_accepts_all():
    btx.data.biorepo.Config(split="all")


def test_biorepo_config_accepts_unlabeled():
    btx.data.biorepo.Config(split="unlabeled")


# ---------------------------------------------------------------------------
# Integration tests (need data on disk)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    not (HAWAII_HF_ROOT / "individual_specimens").is_dir(),
    reason="Hawaii image data not available",
)
def test_hawaii_all_has_more_samples():
    base = {"hf_root": HAWAII_HF_ROOT, "include_polylines": False}
    ds_all = btx.data.hawaii.Dataset(btx.data.hawaii.Config(**base, split="all"))
    ds_train = btx.data.hawaii.Dataset(btx.data.hawaii.Config(**base, split="train"))
    ds_val = btx.data.hawaii.Dataset(btx.data.hawaii.Config(**base, split="val"))
    assert len(ds_all) > len(ds_train)
    assert len(ds_all) > len(ds_val)


@pytest.mark.slow
@pytest.mark.skipif(
    not BIOREPO_ROOT.is_dir(), reason="BioRepo image data not available"
)
def test_biorepo_all_has_more_samples():
    base = {"root": BIOREPO_ROOT, "annotations": BIOREPO_ANN_FPATH}
    ds_all = btx.data.biorepo.Dataset(btx.data.biorepo.Config(**base, split="all"))
    ds_val = btx.data.biorepo.Dataset(btx.data.biorepo.Config(**base, split="val"))
    assert len(ds_all) > len(ds_val)


@pytest.mark.slow
@pytest.mark.skipif(
    not (HAWAII_HF_ROOT / "individual_specimens").is_dir(),
    reason="Hawaii image data not available",
)
def test_hawaii_all_samples_have_split():
    ds = btx.data.hawaii.Dataset(
        btx.data.hawaii.Config(
            hf_root=HAWAII_HF_ROOT, include_polylines=False, split="all"
        )
    )
    for i in range(len(ds)):
        sample = ds[i]
        assert sample["split"] in ("train", "val"), (
            f"Sample {i} has split={sample['split']!r}"
        )


@pytest.mark.slow
@pytest.mark.skipif(
    not BIOREPO_ROOT.is_dir(), reason="BioRepo image data not available"
)
def test_biorepo_all_samples_have_split():
    ds = btx.data.biorepo.Dataset(
        btx.data.biorepo.Config(
            root=BIOREPO_ROOT, annotations=BIOREPO_ANN_FPATH, split="all"
        )
    )
    for i in range(len(ds)):
        sample = ds[i]
        assert sample["split"] in ("train", "val"), (
            f"Sample {i} has split={sample['split']!r}"
        )


@pytest.mark.slow
@pytest.mark.skipif(
    not BIOREPO_UNLABELED_ANN_FPATH.is_file(),
    reason="Unlabeled BioRepo annotations CSV not available",
)
def test_biorepo_unlabeled_sample_has_placeholder_annotations():
    ds = btx.data.biorepo.Dataset(btx.data.biorepo.Config(split="unlabeled"))
    assert len(ds) > 0
    sample = ds[0]
    assert sample["split"] == "unlabeled"
    assert sample["points_px"].shape == (2, 2, 2)
    assert np.all(np.isnan(sample["points_px"]))
    assert sample["scalebar_px"].shape == (2, 2)
    assert np.all(np.isnan(sample["scalebar_px"]))
    assert sample["scalebar_valid"] == np.bool_(False)
    assert sample["loss_mask"].shape == (2,)
    assert np.all(sample["loss_mask"] == 0.0)
    assert isinstance(sample["img_fpath"], str)
    assert pathlib.Path(sample["img_fpath"]).is_file()
