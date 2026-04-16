"""
Scalebar localization training on bbox clips.

Instead of full group images, each sample is a tight crop around the scalebar
region detected from Scalebar/Plotted/ (see detect_scalebar_bboxes.py and
make_scalebar_clips.py). The scalebar fills most of the frame, making this a
much easier task than the original full-image approach. ~75 annotated clips
are available.

crop=False: random crop augmentation is disabled because the scalebar already
fills the clip and cropping could cut it out of frame.

Run with:
    uv run python train_scalebar.py \
        --sweep docs/experiments/013-scalebar-training/sweeps/train.py \
        --slurm-acct PAS2136 \
        --slurm-partition gpu
"""

import pathlib

import btx.modeling.frozen
import btx.objectives.coords

DINO_CKPT_FPATH = pathlib.Path(
    "/fs/ess/PAS2136/samuelstevens/models/dinov3-jax/dinov3_vits16.eqx"
)
CLIPS_JSON = pathlib.Path(
    "/fs/scratch/PAS2136/cain429/beetle-traits/scalebar_clips/clips.json"
)


def make_cfgs() -> list[dict]:
    configs = []

    configs.append({
        "ckpt_dpath": "checkpoints/exp013",
        "tags": ["exp-013-scalebar-training"],
        "batch_size": 16,
        "schedule": "wsd",
        "n_steps": 15_000,
        "n_hours": 1.0,
        "warmup_steps": 1000,
        "decay_steps": 1000,
        "learning_rate": 1e-3,
        "n_workers": 4,
        "scalebar": {"clips_json": CLIPS_JSON},
        "aug": {
            #"size": 512,
            "go": True,
            "crop": False,  # scalebar fills the clip; random crop would cut it out
        },
        "model": btx.modeling.frozen.Frozen(dinov3_ckpt=DINO_CKPT_FPATH),
        "objective": btx.objectives.coords.Coords(),
    })

    return configs
