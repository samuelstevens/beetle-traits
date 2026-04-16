"""Inference sweep for experiment 011.

Runs inference on the two round-2 checkpoints over BioRepo (val split). One Parquet per checkpoint.

Run with:
    uv run launch.py inference --sweep docs/experiments/011-round2-results/sweeps/inference.py

Training run IDs (from exp-011-round2-results sweep):
    1ykzeqat  small (ViT-S, wsd schedule)
    7wdeiefc  base  (ViT-B, no schedule)
"""

import pathlib

BIOREPO_ROOT = pathlib.Path("/fs/scratch/PAS2136/cain429/Subset-Exp")
BIOREPO_ANN_FPATH = pathlib.Path(
    "/fs/scratch/PAS2136/cain429/biorepo-formatted/annotations.json"
)
BIOREPO_UNLABELED_ANN_FPATH = pathlib.Path("/fs/scratch/PAS2136/cain429/unlabeled_with_scalebar.csv")

RUN_IDS = ["1ykzeqat"]

OUT_DPATH = pathlib.Path("/fs/scratch/PAS2136/cain429/beetle-traits/exp-12-outputs")

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
                "unlabeled_annotations": BIOREPO_UNLABELED_ANN_FPATH,
                "split": "unlabeled",
            },
            **SLURM,
        })

    return cfgs
