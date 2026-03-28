"""Inference sweep for experiment 005.

Runs inference on all three training checkpoints (LR=0.03, 0.1, 0.3) over Hawaii (all splits) + BioRepo (all splits). One Parquet per checkpoint.

Run with:
    uv run launch.py inference --sweep docs/experiments/005-active-learning/sweeps/inference.py
"""

import pathlib

HAWAII_HF_ROOT = pathlib.Path("/fs/ess/PAS2136/samuelstevens/datasets/hawaii-beetles")
HAWAII_ANN_FPATH = pathlib.Path("data/hawaii-formatted/annotations.json")
BIOREPO_ROOT = pathlib.Path("/fs/scratch/PAS2136/cain429/Subset-Exp")
BIOREPO_ANN_FPATH = pathlib.Path(
    "/fs/scratch/PAS2136/cain429/biorepo-formatted/annotations.json"
)

RUN_IDS = ["3e21o8cr"]

OUT_DPATH = pathlib.Path(
    "/fs/scratch/PAS2136/cain429/active_learning_round1/results"
)

SLURM = {
    "slurm_acct": "PAS2136",
    "slurm_partition": "nextgen",
    "n_hours": 1.0,
}


def make_cfgs() -> list[dict]:
    cfgs = []
    for run_id in RUN_IDS:
        # Labeled: Hawaii (all) + BioRepo (all)
        cfgs.append({
            "ckpt_fpath": f"checkpoints/{run_id}/model.eqx",
            "out_fpath": str(OUT_DPATH / f"{run_id}_labeled.parquet"),
            "hawaii": {
                "hf_root": HAWAII_HF_ROOT,
                "annotations": HAWAII_ANN_FPATH,
                "include_polylines": False,
                "split": "all",
            },
            "beetlepalooza": {"go": False},
            "biorepo": {
                "root": BIOREPO_ROOT,
                "annotations": BIOREPO_ANN_FPATH,
                "split": "all",
            },
            **SLURM,
        })
        # Unlabeled: BioRepo only
        cfgs.append({
            "ckpt_fpath": f"checkpoints/{run_id}/model.eqx",
            "out_fpath": str(OUT_DPATH / f"{run_id}_unlabeled.parquet"),
            "hawaii": {"go": False},
            "beetlepalooza": {"go": False},
            "biorepo": {
                "root": BIOREPO_ROOT,
                "annotations": BIOREPO_ANN_FPATH,
                "split": "unlabeled",
            },
            **SLURM,
        })

    return cfgs
