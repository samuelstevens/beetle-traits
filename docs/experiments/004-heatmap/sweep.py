"""Twelve-run sweep for experiment 004 (heatmap regression).

Run with:
    uv run launch.py --sweep docs/experiments/004-heatmap/sweep.py

Notes:
1. This script encodes the `(sigma, learning_rate)` grid from the spec.
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

SIGMAS = [1.0, 2.0, 3.0]
LRS = [1e-3, 3e-3, 1e-2, 3e-2]


def _get_seed(run_i: int) -> int:
    return 4_000 + run_i


def _make_run(*, sigma: float, lr: float, run_i: int) -> dict:
    return {
        "seed": _get_seed(run_i),
        "batch_size": 256,
        "learning_rate": lr,
        "n_steps": 100_000,
        "n_hours": 12.0,
        "tags": ["exp-004-heatmap", "ce-fix"],
        "objective": btx.objectives.Heatmap(heatmap_size=64, sigma=sigma),
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
        "biorepo": {"root": BIOREPO_ROOT, "annotations": BIOREPO_ANN_FPATH},
        "aug_hawaii": {"go": True, "normalize": True, "crop": False},
        "aug_beetlepalooza": {"go": True, "normalize": True, "crop": False},
        "aug_biorepo": {"go": True, "normalize": True, "crop": False},
        "slurm_acct": "PAS2136",
        "slurm_partition": "nextgen",
    }


def make_cfgs() -> list[dict]:
    cfgs: list[dict] = []
    run_i = 0
    for sigma in SIGMAS:
        for lr in LRS:
            cfgs.append(_make_run(sigma=sigma, lr=lr, run_i=run_i))
            run_i += 1
    return cfgs
