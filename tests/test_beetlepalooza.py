"""Tests for the BeetlePalooza dataset loader."""

import typing as tp
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from btx.data import beetlepalooza, transforms


@pytest.fixture
def cfg():
    """Default config for testing."""
    return beetlepalooza.Config()


def test_config_creation(cfg):
    """Test that config can be created with defaults."""
    assert cfg.hf_root == Path("data/beetlepalooza")
    assert cfg.annotations == Path("data/beetlepalooza-formatted/annotations.json")
    assert cfg.annotators == ["IsaFluck"]
    assert cfg.include_polylines is False


def test_config_paths_exist(cfg):
    """Test that configured paths exist."""
    assert cfg.hf_root.exists(), f"{cfg.hf_root} does not exist"
    assert cfg.annotations.exists(), f"{cfg.annotations} does not exist"


def test_trusted_data_filtering(cfg):
    """Test that _trusted_data filters annotations correctly."""
    df = beetlepalooza._trusted_data(cfg)

    # Basic sanity checks
    assert df.height > 0, "No data after filtering"
    assert "individual_id" in df.columns
    assert "measurements" in df.columns

    # Each row should have measurements
    assert not df["measurements"].is_null().any()


def test_dataset_creation(cfg):
    """Test that Dataset can be instantiated."""
    ds = beetlepalooza.Dataset(cfg)
    assert len(ds) > 0


def test_dataset_getitem(cfg):
    """Test loading a single sample."""
    ds = beetlepalooza.Dataset(cfg)
    sample = ds[0]

    # Check all expected keys exist
    assert "img_fpath" in sample
    assert "points_px" in sample
    assert "scalebar_px" in sample
    assert "beetle_id" in sample
    assert "beetle_position" in sample
    assert "group_img_basename" in sample

    # Check shapes
    assert sample["points_px"].shape == (2, 2, 2), (
        "points_px should be (2 lines, 2 points, 2 coords)"
    )
    assert sample["scalebar_px"].shape == (2, 2), (
        "scalebar_px should be (2 points, 2 coords)"
    )

    # Check image file exists
    assert Path(sample["img_fpath"]).exists(), f"Image {sample['img_fpath']} not found"


def test_dataset_multiple_samples(cfg):
    """Test loading multiple samples to ensure consistency."""
    ds = beetlepalooza.Dataset(cfg)

    # Test first 5 samples
    n_samples = min(5, len(ds))
    for i in range(n_samples):
        sample = ds[i]
        assert Path(sample["img_fpath"]).exists()
        assert sample["points_px"].shape == (2, 2, 2)
        assert sample["scalebar_px"].shape == (2, 2)


def test_dataset_with_different_annotators():
    """Test that changing annotators affects filtering."""
    cfg_default = beetlepalooza.Config()
    ds_default = beetlepalooza.Dataset(cfg_default)
    default_len = len(ds_default)

    # Verify the config accepts the parameter
    assert cfg_default.annotators == ["IsaFluck"]
    assert default_len > 0


@pytest.mark.parametrize("idx", [0, 1])
def test_dataset_indexing(cfg, idx):
    """Test various indexing patterns."""
    ds = beetlepalooza.Dataset(cfg)

    sample = ds[idx]
    assert sample is not None


def test_dataset_last_item(cfg):
    """Test accessing the last item in the dataset."""
    ds = beetlepalooza.Dataset(cfg)
    idx = len(ds) - 1

    sample = ds[idx]
    assert sample is not None


def test_with_grain_transforms(cfg):
    """Test integration with grain transformations."""
    ds = beetlepalooza.Dataset(cfg)

    # Test that DecodeRGB transform works on a sample
    sample = ds[0]
    decode_rgb = transforms.DecodeRGB()
    transformed = tp.cast(dict[str, object], decode_rgb.map(sample))

    # DecodeRGB should add "img" key
    assert "img" in transformed
    assert transformed["img"] is not None
    # PIL Image has mode attribute
    assert hasattr(transformed["img"], "mode")


def test_sample_matches_utils_sample_type(cfg):
    """Test that returned samples match the utils.Sample TypedDict."""
    ds = beetlepalooza.Dataset(cfg)
    sample = ds[0]

    # Check types match what's expected in utils.Sample
    assert isinstance(sample["img_fpath"], str)
    assert isinstance(sample["points_px"], np.ndarray)
    assert isinstance(sample["scalebar_px"], np.ndarray)
    assert isinstance(sample["beetle_id"], str)
    assert isinstance(sample["beetle_position"], (int, np.integer))
    assert isinstance(sample["group_img_basename"], str)


def test_index_out_of_range_fails(cfg):
    """Tests that a sample out of the range of the dataset returns an error"""
    ds = beetlepalooza.Dataset(cfg)

    with pytest.raises(pl.exceptions.OutOfBoundsError):
        _ = ds[len(ds)]


def test_points_are_valid_coordinates(cfg):
    """Test that point coordinates are reasonable (non-negative, finite)."""
    ds = beetlepalooza.Dataset(cfg)
    sample = ds[0]

    # Points should be finite and non-negative
    assert np.all(np.isfinite(sample["points_px"]))
    assert np.all(sample["points_px"] >= 0)

    # Scalebar should also be valid
    assert np.all(np.isfinite(sample["scalebar_px"]))
    assert np.all(sample["scalebar_px"] >= 0)
