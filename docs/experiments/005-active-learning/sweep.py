"""Sweep for experiment 005 (active learning baseline checkpoints).

3 runs: sigma=1, LR in {0.03, 0.1, 0.3}. These bracket the BioRepo-optimal LR from exp 004. Each run saves a checkpoint at the end for inference.

Run with:
    uv run launch.py train --sweep docs/experiments/005-active-learning/sweep.py
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
BIOREPO_ROOT = pathlib.Path("/fs/scratch/PAS2136/cain429/Subset-Exp")
BIOREPO_ANN_FPATH = pathlib.Path(
    "/fs/scratch/PAS2136/cain429/biorepo-formatted/annotations.json"
)
DINO_CKPT_FPATH = pathlib.Path(
    "/fs/ess/PAS2136/samuelstevens/models/dinov3-jax/dinov3_vits16.eqx"
)

LRS = [0.03, 0.1, 0.3]


def _make_run(*, lr: float, run_i: int) -> dict:
    return {
        "seed": 5_000 + run_i,
        "batch_size": 256,
        "learning_rate": lr,
        "n_steps": 60_000,
        "n_hours": 12.0,
        "tags": ["exp-005-al"],
        "objective": btx.objectives.Heatmap(heatmap_size=64, sigma=1.0),
        "model": btx.modeling.heatmap.Heatmap(dinov3_ckpt=DINO_CKPT_FPATH),
        "ckpt_fpath": f"logs/exp005_lr{lr}_sigma1.eqx",
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
        "biorepo": {"root": BIOREPO_ROOT, "annotations": BIOREPO_ANN_FPATH},
        "aug_hawaii": {"go": True, "normalize": True, "crop": False},
        "aug_beetlepalooza": {"go": True, "normalize": True, "crop": False},
        "aug_biorepo": {"go": True, "normalize": True, "crop": False},
        "slurm_acct": "PAS2136",
        "slurm_partition": "nextgen",
    }


def make_cfgs() -> list[dict]:
    return [_make_run(lr=lr, run_i=i) for i, lr in enumerate(LRS)]
