"""Sweep comparing biorepo as training data vs validation-only data.

Two runs (one per condition) each allocated 4 hours:
- biorepo-train: biorepo split="train" adds biorepo to the training set
- biorepo-val: biorepo split="val" (default) uses biorepo only for validation

Uses the best-known settings from exp-006 (cosine schedule, lr=3e-4).

Run with:
    uv run launch.py --sweep docs/experiments/007-biorepo-split/sweep.py
"""

import pathlib

import btx.modeling.heatmap
import btx.objectives

HAWAII_HF_ROOT = pathlib.Path("/fs/ess/PAS2136/samuelstevens/datasets/hawaii-beetles")
HAWAII_ANN_FPATH = pathlib.Path("data/hawaii-formatted/annotations.json")
BEETLEPALOOZA_HF_ROOT = pathlib.Path(
    "/fs/ess/PAS2136/samuelstevens/datasets/beetlepalooza-beetles"
)
BEETLEPALOOZA_ANN_FPATH = pathlib.Path("data/beetlepalooza-formatted/annotations.json")
BIOREP_HF_ROOT = pathlib.Path("/fs/scratch/PAS2136/cain429/Subset-Exp")
BIOREP_ANN_FPATH = pathlib.Path(
    "/fs/scratch/PAS2136/cain429/biorepo-formatted/annotations.json"
)

DINO_CKPT_FPATH = pathlib.Path(
    "/fs/ess/PAS2136/samuelstevens/models/dinov3-jax/dinov3_vits16.eqx"
)


def _make_run(biorepo_split: str, tag: str) -> dict:
    return {
        "seed": 17,
        "learning_rate": 3e-4,
        "schedule": "none",
        "n_steps": 20_000,
        "n_hours": 4.0,
        "warmup_steps": 2_000,
        "decay_steps": 18000,
        "tags": ["exp-007-biorepo-split", tag],
        "objective": btx.objectives.Heatmap(heatmap_size=64, sigma=1.0),
        "model": btx.modeling.heatmap.Heatmap(dinov3_ckpt=DINO_CKPT_FPATH),
        "hawaii": {
            "hf_root": HAWAII_HF_ROOT,
            "annotations": HAWAII_ANN_FPATH,
            "include_polylines": False,
        },
        "beetlepalooza": {
            "hf_root": BEETLEPALOOZA_HF_ROOT,
            "annotations": BEETLEPALOOZA_ANN_FPATH,
            "include_polylines": False,
        },
        "biorepo": {
            "root": BIOREP_HF_ROOT,
            "annotations": BIOREP_ANN_FPATH,
            "split": biorepo_split,
        },
        "aug_hawaii": {"go": True, "normalize": True, "crop": False},
        "aug_beetlepalooza": {"go": True, "normalize": True, "crop": False},
        "aug_biorepo": {"go": True, "normalize": True, "crop": False},
    }


def make_cfgs() -> list[dict]:
    return [
        _make_run("train", "biorepo-train"),
        _make_run("val", "biorepo-val"),
    ]
