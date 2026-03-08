"""Inference sweep for experiment 007.

Runs inference on both training checkpoints over Hawaii (all splits) + BioRepo (all splits). One Parquet per checkpoint.

Run with:
    uv run launch.py inference --sweep docs/experiments/007-biorepo-split/sweeps/inference.py

Training run IDs (from exp-007-biorepo-split sweep):
    um3ukq2x  biorepo-train (biorepo split="train")
    16gsx5v7  biorepo-val   (biorepo split="val")
"""

import pathlib

HAWAII_HF_ROOT = pathlib.Path("/fs/ess/PAS2136/samuelstevens/datasets/hawaii-beetles")
HAWAII_ANN_FPATH = pathlib.Path("data/hawaii-formatted/annotations.json")
BIOREPO_ROOT = pathlib.Path("/fs/scratch/PAS2136/cain429/Subset-Exp")
BIOREPO_ANN_FPATH = pathlib.Path(
    "/fs/scratch/PAS2136/cain429/biorepo-formatted/annotations.json"
)

RUN_IDS = ["um3ukq2x", "16gsx5v7"]

OUT_DPATH = pathlib.Path("docs/experiments/007-biorepo-split/results")

SLURM = {
    "slurm_acct": "PAS2136",
    "slurm_partition": "nextgen",
    "n_hours": 1.0,
}


def make_cfgs() -> list[dict]:
    cfgs = []
    for run_id in RUN_IDS:
        cfgs.append({
            "ckpt_fpath": f"checkpoints/{run_id}/model.eqx",
            "out_fpath": str(OUT_DPATH / f"{run_id}.parquet"),
            "hawaii": {
                "go": False,
                "hf_root": HAWAII_HF_ROOT,
                "annotations": HAWAII_ANN_FPATH,
                "include_polylines": False,
                "split": "all",
            },
            "beetlepalooza": {"go": False},
            "biorepo": {
                "root": BIOREPO_ROOT,
                "annotations": BIOREPO_ANN_FPATH,
                "split": "val",
            },
            **SLURM,
        })

    return cfgs
