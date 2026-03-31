"""
Run with:
    uv run launch.py --sweep docs/experiments/009-round2-traing/sweep.py
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
BIOREPO_UNLABELED_ANNOTATIONS = pathlib.Path(
        "/fs/scratch/PAS2136/cain429/unlabeled_biorepo_annotations.csv")
BIOREPO_ROUND2_ANN_FPATH = pathlib.Path(
    "/fs/scratch/PAS2136/cain429/active_learning_round2/round2/annotations.json")

DINO_CKPT_FPATH = pathlib.Path(
    "/fs/ess/PAS2136/samuelstevens/models/dinov3-jax/dinov3_vits16.eqx"
)
DINO_CKPT_FPATHBASE = pathlib.Path(
    "/fs/ess/PAS2136/samuelstevens/models/dinov3-jax/dinov3_vitb16.eqx"
)


def make_cfgs() -> list[dict]:
    cfgs = []
    cfgs.append(
    {
        "seed": 17,
        "learning_rate": 1e-2,
        "schedule": "wsd",
        "n_steps": 60_000,
        "n_hours": 9.0,
        "warmup_steps": 3000,
        "decay_steps": 6_000,
        "tags": ["exp-011-round2-results", "small"],
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
            "split": "train",
            "training_annotations": BIOREPO_ROUND2_ANN_FPATH
        },
        "aug_hawaii": {"go": True, "normalize": True, "crop": False},
        "aug_beetlepalooza": {"go": True, "normalize": True, "crop": False},
        "aug_biorepo": {"go": True, "normalize": True, "crop": False},
    })
    cfgs.append(
    {
        "seed": 17,
        "learning_rate": 1e-2,
        "schedule": "none",
        "n_steps": 60_000,
        "n_hours": 9.0,
        "warmup_steps": 3000,
        "decay_steps": 6_000,
        "tags": ["exp-011-round2-results", "base"],
        "objective": btx.objectives.Heatmap(heatmap_size=64, sigma=1.0),
        "model": btx.modeling.heatmap.Heatmap(dinov3_ckpt=DINO_CKPT_FPATHBASE),
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
            "split": "train",
            "training_annotations": BIOREPO_ROUND2_ANN_FPATH
        },
        "aug_hawaii": {"go": True, "normalize": True, "crop": False},
        "aug_beetlepalooza": {"go": True, "normalize": True, "crop": False},
        "aug_biorepo": {"go": True, "normalize": True, "crop": False},
    })
    return cfgs
