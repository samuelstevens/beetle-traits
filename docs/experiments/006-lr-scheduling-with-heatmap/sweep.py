"""Quick-run sweep for learning rate scheduling experiment.

This sweep tests different LR schedules:
- Cosine schedule with warmup
- WSD (Warmup-Stable-Decay) schedule with different configurations

Run with:
    uv run launch.py --sweep docs/experiments/006-lr-scheduling/sweep.py
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
BIOREP_ANN_FPATH = pathlib.Path("/fs/scratch/PAS2136/cain429/biorepo-formatted/annotations.json")

DINO_CKPT_FPATH = pathlib.Path(
    "/fs/ess/PAS2136/samuelstevens/models/dinov3-jax/dinov3_vits16.eqx"
)

def _base_aug_cfg() -> dict:
    """Standard augmentation configuration."""
    return {
        "go": True,
        "normalize": True,
        "crop_scale_min": 0.9,
        "crop_scale_max": 1.0,
        "crop_ratio_min": 0.95,
        "crop_ratio_max": 1.067,
    }


def _make_run(
    seed: int,
    lr: float,
    schedule: str,
    warmup_steps: int,
    decay_steps: int,
    name: str,
) -> dict:
    aug_cfg = _base_aug_cfg()

    return {
        "seed": seed,
        "learning_rate": lr,
        "schedule": schedule,
        "n_steps": 80_000,
        "warmup_steps": warmup_steps,
        "decay_steps": decay_steps,
        "tags": [
            "exp-006-lr-scheduling-with-heatmap",
            name,
        ],
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
        "biorepo" : {
            "root": BIOREP_HF_ROOT,
            "annotations": BIOREP_ANN_FPATH,
        },


        "aug_hawaii": {"go": True, "normalize": True, "crop": False},
        "aug_beetlepalooza": {"go": True, "normalize": True, "crop": False},
        "aug_biorepo": {"go": True, "normalize": True, "crop": False},
    }


def make_cfgs() -> list[dict]:
    cfgs: list[dict] = []
    lr_specs = [
        1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-1, 1e-1,
    ]

    schedule_configs = [
        ("cosine", 4_000, 76_000, "cosine"),
        ("wsd", 4_000, 0, "wsd-no-decay"),
        ("wsd", 4_000, 8_000, "wsd"),
        ("none", 0, 0, "none")
    ]

    # Generate all combinations
    for lr in lr_specs:
        for schedule, warmup, decay, tag in schedule_configs:
            cfgs.append(_make_run(17, lr, schedule, warmup, decay, tag))

    return cfgs
