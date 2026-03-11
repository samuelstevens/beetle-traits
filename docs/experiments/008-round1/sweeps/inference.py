"""Inference sweep for experiment 008.

Runs inference on the biorepo-train checkpoint over BioRepo (val split). One Parquet per checkpoint.

Run with:
    uv run launch.py inference --sweep docs/experiments/008-round1/sweeps/inference.py

Training run IDs (from exp-008-round1 sweep):
    9a7jk34w  biorepo-train (biorepo split="train")
"""

import pathlib

BIOREPO_ROOT = pathlib.Path("/fs/scratch/PAS2136/cain429/Subset-Exp")
BIOREPO_ANN_FPATH = pathlib.Path(
    "/fs/scratch/PAS2136/cain429/biorepo-formatted/annotations.json"
)

RUN_IDS = ["9a7jk34w"]

OUT_DPATH = pathlib.Path("docs/experiments/008-round1/results")

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
            "hawaii": {"go": False},
            "beetlepalooza": {"go": False},
            "biorepo": {
                "root": BIOREPO_ROOT,
                "annotations": BIOREPO_ANN_FPATH,
                "split": "val",
            },
            **SLURM,
        })

    return cfgs
