"""Inference sweep for experiment 005.

Runs inference on all three training checkpoints (LR=0.03, 0.1, 0.3) over Hawaii + BioRepo. Outputs one Parquet per checkpoint.

Run with:
    uv run launch.py inference --sweep docs/experiments/005-active-learning/sweeps/inference.py

Training run IDs (from Slurm array 3776753):
    gxdlfrgd  LR=0.03
    egqr97d7  LR=0.1
    v1t5i5tq  LR=0.3
"""

import pathlib

HAWAII_HF_ROOT = pathlib.Path("/fs/ess/PAS2136/samuelstevens/datasets/hawaii-beetles")
HAWAII_ANN_FPATH = pathlib.Path("data/hawaii-formatted/annotations.json")
BIOREPO_ROOT = pathlib.Path("/fs/scratch/PAS2136/cain429/Subset-Exp")
BIOREPO_ANN_FPATH = pathlib.Path(
    "/fs/scratch/PAS2136/cain429/biorepo-formatted/annotations.json"
)

RUN_IDS = ["gxdlfrgd", "egqr97d7", "v1t5i5tq"]

OUT_DPATH = pathlib.Path("docs/experiments/005-active-learning/results")


def make_cfgs() -> list[dict]:
    cfgs = []
    for run_id in RUN_IDS:
        cfgs.append({
            "ckpt_fpath": f"checkpoints/exp005/{run_id}/model.eqx",
            "out_fpath": str(OUT_DPATH / f"{run_id}.parquet"),
            "hawaii": {
                "hf_root": HAWAII_HF_ROOT,
                "annotations": HAWAII_ANN_FPATH,
                "include_polylines": False,
            },
            "beetlepalooza": {"go": False},
            "biorepo": {"root": BIOREPO_ROOT, "annotations": BIOREPO_ANN_FPATH},
            "slurm_acct": "PAS2136",
            "slurm_partition": "nextgen",
            "n_hours": 1.0,
        })
    return cfgs
