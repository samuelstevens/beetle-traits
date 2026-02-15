"""Quick-run sweep for augmentation experiment.

This sweep creates 10 runs:
- 5 paired (seed, learning-rate) settings
- 2 conditions (norm-only, full-augmentation with no crop)

Run with:
    uv run launch.py --sweep docs/experiments/002-augmentation/sweep.py
"""

import pathlib

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


def _norm_only_cfg() -> dict:
    return {"go": False, "normalize": True, "crop": False}


def _full_aug_cfg() -> dict:
    return {"go": True, "normalize": True, "crop": False}


def _make_run(seed: int, lr: float, name: str, aug_cfg: dict) -> dict:
    return {
        "seed": seed,
        "learning_rate": lr,
        "n_steps": 100_000,
        "tags": [
            "exp-002-augmentation",
            name,
        ],
        "model": {"dinov3_ckpt": DINO_CKPT_FPATH},
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
        "aug_hawaii": dict(aug_cfg),
        "aug_beetlepalooza": dict(aug_cfg),
        "aug_biorepo": dict(aug_cfg),
        "slurm_acct": "PAS2136",
        "slurm_partition": "nextgen",
    }


def make_cfgs() -> list[dict]:
    cfgs: list[dict] = []
    seed_lr_specs = [
        (47, 1e-3),
        (65, 3e-3),
        (72, 1e-2),
        (89, 3e-2),
        (91, 1e-1),
    ]
    conditions = [
        ("norm-only", _norm_only_cfg()),
        ("full-aug", _full_aug_cfg()),
    ]
    for seed, lr in seed_lr_specs:
        for name, aug_cfg in conditions:
            cfgs.append(_make_run(seed, lr, name, aug_cfg))
    return cfgs
