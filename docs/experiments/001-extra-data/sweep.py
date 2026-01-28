"""Six-job sweep: three seeds x two data configurations."""

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
    for seed in [17, 23, 29]:
        cfgs.append({
            "seed": seed,
            "batch_size": 256,
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
            "seed": seed,
            "batch_size": 256,
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
