"""Tests for split="all" support in dataset configs."""

import pathlib

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

# ---------------------------------------------------------------------------
# Unit tests (no data needed)
# ---------------------------------------------------------------------------


def test_sample_has_split_field():
    assert "split" in btx.data.utils.Sample.__annotations__


def test_hawaii_config_accepts_all():
    btx.data.hawaii.Config(split="all")


def test_biorepo_config_accepts_all():
    btx.data.biorepo.Config(split="all")


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
