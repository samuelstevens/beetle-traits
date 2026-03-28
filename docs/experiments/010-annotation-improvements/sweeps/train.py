"""Sweep testing annotation count per genus on BioRepo training.

Five runs, one per sample count (0, 5, 10, 15, all) from each of the 3 target
genera (Dicheirus, Discoderus, Metrius). All round-1 training annotations are
included in each run via the training_annotations path.

Run with:
    uv run launch.py --sweep docs/experiments/010-annotation-improvements/sweeps/train.py
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
BIOREP_ANN_FPATH = pathlib.Path(
    "/fs/scratch/PAS2136/cain429/biorepo-formatted/annotations.json"
)
EXP10_ANN_DPATH = pathlib.Path(
    "/fs/scratch/PAS2136/cain429/beetle-traits-exp-10/experiment-annotations"
)

DINO_CKPT_FPATH = pathlib.Path(
    "/fs/ess/PAS2136/samuelstevens/models/dinov3-jax/dinov3_vits16.eqx"
)


def _make_run(n_samples: int | str) -> dict:
    if n_samples == "all":
        ann_fname = "annotations-all-samples.json"
    else:
        ann_fname = f"annotations-{n_samples}-samples.json"
    return {
        "seed": 17,
        "learning_rate": 1e-2,
        "schedule": "wsd",
        "n_steps": 26_000,
        "n_hours": 4.0,
        "warmup_steps": 1_300,
        "decay_steps": 2_600,
        "tags": ["exp-010-annotation-improvements", f"n-samples-{n_samples}"],
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
        "biorepo": {
            "root": BIOREP_HF_ROOT,
            "annotations": BIOREP_ANN_FPATH,
            "split": "train",
            "training_annotations": EXP10_ANN_DPATH / ann_fname,
        },
        "aug_hawaii": {"go": True, "normalize": True, "crop": False},
        "aug_beetlepalooza": {"go": True, "normalize": True, "crop": False},
        "aug_biorepo": {"go": True, "normalize": True, "crop": False},
    }


def make_cfgs() -> list[dict]:
    return [_make_run(n) for n in (0, 5, 10, 15, "all")]
