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

import btx.modeling.heatmap
import btx.objectives

DINO_CKPT_FPATH = pathlib.Path(
    "/fs/ess/PAS2136/samuelstevens/models/dinov3-jax/dinov3_vits16.eqx"
)
CLIPS_JSON = pathlib.Path(
    "/fs/scratch/PAS2136/cain429/beetle-traits/scalebar_clips/clips.json"
)
LRS = [1e-3]


def make_cfgs() -> list[dict]:
    configs = []

    for LR in LRS:
        configs.append({
            "ckpt_dpath": "checkpoints/exp013",
            "tags": ["exp-013-scalebar-training"],
            "batch_size": 32,
            "schedule": "wsd",
            "n_steps": 5_000,
            "n_hours": 1.0,
            "warmup_steps": 1_000,
            "decay_steps": 1_000,
            "learning_rate": LR,
            "n_workers": 4,
            "scalebar": {"clips_json": CLIPS_JSON},
            "aug": {
                "size": 512,
                "go": True,
                "crop": False,  # scalebar fills the clip; random crop would cut it out
            },
            "model": btx.modeling.heatmap.Heatmap(
                dinov3_ckpt=DINO_CKPT_FPATH,
                heatmap_size=128,
            ),
            "objective": btx.objectives.Heatmap(image_size=512, heatmap_size=128),
        })

    return configs
