"""Six-job sweep: three learning rates x two data configurations."""

import pathlib

HAWAII_HF_ROOT = pathlib.Path("/fs/ess/PAS2136/samuelstevens/datasets/hawaii-beetles")
HAWAII_ANN_FPATH = pathlib.Path("data/hawaii-formatted/annotations.json")
BEETLEPALOOZA_HF_ROOT = pathlib.Path(
    "/fs/ess/PAS2136/samuelstevens/datasets/beetlepalooza-beetles"
)
BEETLEPALOOZA_ANN_FPATH = pathlib.Path("data/beetlepalooza-formatted/annotations.json")
DINO_CKPT_FPATH = pathlib.Path(
    "/fs/ess/PAS2136/samuelstevens/models/dinov3-jax/dinov3_vits16.eqx"
)


def make_cfgs() -> list[dict]:
    cfgs = []
    for lr in [1e-4, 3e-4, 1e-3]:
        cfgs.append({
            "seed": 17,
            "batch_size": 256,
            "learning_rate": lr,
            "tags": ["exp-001", "hawaii-only"],
            "model": {"dinov3_ckpt": DINO_CKPT_FPATH},
            "hawaii": {
                "hf_root": HAWAII_HF_ROOT,
                "annotations": HAWAII_ANN_FPATH,
                "include_polylines": False,
            },
            "beetlepalooza": {
                "go": False,
                "hf_root": BEETLEPALOOZA_HF_ROOT,
                "annotations": BEETLEPALOOZA_ANN_FPATH,
                "include_polylines": False,
            },
        })
        cfgs.append({
            "seed": 17,
            "batch_size": 256,
            "learning_rate": lr,
            "tags": ["exp-001", "extra-data"],
            "model": {"dinov3_ckpt": DINO_CKPT_FPATH},
            "hawaii": {
                "hf_root": HAWAII_HF_ROOT,
                "annotations": HAWAII_ANN_FPATH,
                "include_polylines": False,
            },
            "beetlepalooza": {
                "go": True,
                "hf_root": BEETLEPALOOZA_HF_ROOT,
                "annotations": BEETLEPALOOZA_ANN_FPATH,
                "include_polylines": False,
            },
        })

    return cfgs
