"""Inference sweep for experiment 010: annotation improvements.

One labeled inference job per training run (0, 5, 10, 15, all samples per genus).
Runs over BioRepo (all labeled splits) to measure per-genus percent error.

Run with:
    uv run launch.py inference --sweep docs/experiments/010-annotation-improvements/sweeps/inference.py --slurm_acct PAS2136 --slurm-partition nextgen --n_workers 5 --log_to /users/PAS2136/cain429/projects/beetle-traits/logs
"""

import pathlib

BIOREPO_ROOT = pathlib.Path("/fs/scratch/PAS2136/cain429/Subset-Exp")
BIOREPO_ANN_FPATH = pathlib.Path(
    "/fs/scratch/PAS2136/cain429/biorepo-formatted/annotations.json"
)

OUT_DPATH = pathlib.Path("/fs/scratch/PAS2136/cain429/beetle-traits-exp-10/results")

# Fill in run IDs after training completes (one per n_samples entry below).
RUN_ID_MAP: dict[str, str] = {
    "0": "nsonyiah",
    "5": "hl3qjpzy",
    "10": "2q9c6pae",
    "15": "7l8pqxme",
    "all": "hczddqw8",
}

SLURM = {
    "slurm_acct": "PAS2136",
    "slurm_partition": "nextgen",
    "n_hours": 1.0,
}


def make_cfgs() -> list[dict]:
    cfgs = []
    for n_samples, run_id in RUN_ID_MAP.items():
        cfgs.append({
            "ckpt_fpath": f"checkpoints/{run_id}/model.eqx",
            "out_fpath": str(OUT_DPATH / f"exp10_{n_samples}_labeled.parquet"),
            "hawaii": {"go": False},
            "beetlepalooza": {"go": False},
            "biorepo": {
                "root": BIOREPO_ROOT,
                "annotations": BIOREPO_ANN_FPATH,
                "split": "all",
            },
            **SLURM,
        })
    return cfgs
