"""Rank sweep for experiment 009: active learning round 2.

Run with:
    uv run launch.py rank --sweep docs/experiments/005-active-learning/sweeps/rank.py
"""

import pathlib

RESULTS_DPATH = pathlib.Path(
     "/fs/scratch/PAS2136/cain429/active_learning_round1/results"
)


SHARED = {
    "glob_pattern": str(RESULTS_DPATH / "*_unlabeled.parquet"),
    "k": 73,
    "n_avg": 10,
    "n_groups": 50,
    "slurm_acct": "PAS2136",
    "slurm_partition": "debug-nextgen",
    "n_hours": 0.5,
}


def make_cfgs() -> list[dict]:
    return [
        {**SHARED, "out_fpath": str(RESULTS_DPATH / "round1"), "cost_alpha": 1.0},
    ]
