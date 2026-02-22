"""End-to-end test: train 1 step, save checkpoint, run inference, verify Parquet."""

import importlib.util
import pathlib

import numpy as np
import polars as pl
import pytest

import btx.data
import btx.objectives
from btx.modeling import heatmap

inference_fpath = pathlib.Path(__file__).resolve().parents[1] / "inference.py"
spec = importlib.util.spec_from_file_location("inference_script", inference_fpath)
assert spec is not None and spec.loader is not None
inference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inference)

train_fpath = pathlib.Path(__file__).resolve().parents[1] / "train.py"
spec = importlib.util.spec_from_file_location("train_script", train_fpath)
assert spec is not None and spec.loader is not None
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)

HAWAII_HF_ROOT = pathlib.Path("/fs/ess/PAS2136/samuelstevens/datasets/hawaii-beetles")
BIOREPO_ROOT = pathlib.Path("/fs/scratch/PAS2136/cain429/Subset-Exp")
BIOREPO_ANN_FPATH = pathlib.Path(
    "/fs/scratch/PAS2136/cain429/biorepo-formatted/annotations.json"
)


@pytest.mark.slow
@pytest.mark.skipif(
    not (HAWAII_HF_ROOT / "individual_specimens").is_dir(),
    reason="Hawaii image data not available",
)
@pytest.mark.skipif(
    not BIOREPO_ROOT.is_dir(), reason="BioRepo image data not available"
)
def test_train_then_infer_end_to_end(
    dinov3_vits_fpath: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    ckpt_dpath = tmp_path / "checkpoints"

    # 1) Train for 1 step.
    train_cfg = train.Config(
        n_steps=1,
        batch_size=32,
        n_workers=0,
        model=heatmap.Heatmap(dinov3_ckpt=dinov3_vits_fpath),
        objective=btx.objectives.Heatmap(),
        hawaii=btx.data.HawaiiConfig(hf_root=HAWAII_HF_ROOT, include_polylines=False),
        beetlepalooza=btx.data.BeetlePaloozaConfig(go=False),
        biorepo=btx.data.BioRepoConfig(
            root=BIOREPO_ROOT, annotations=BIOREPO_ANN_FPATH
        ),
        val=train.ValConfig(every=1_000_000, n_fixed=1),
        save_every=1_000_000,
        log_every=1,
        ckpt_dpath=ckpt_dpath,
        wandb_project="test",
    )

    monkeypatch.setenv("WANDB_MODE", "disabled")
    train.train(train_cfg)

    # Find the checkpoint saved under {ckpt_dpath}/{run_id}/model.eqx.
    ckpt_fpaths = list(ckpt_dpath.glob("*/model.eqx"))
    assert len(ckpt_fpaths) == 1, f"Expected 1 checkpoint, found {ckpt_fpaths}"
    ckpt_fpath = ckpt_fpaths[0]

    # 2) Run inference on Hawaii val.
    out_fpath = tmp_path / "results.parquet"
    infer_cfg = inference.Config(
        ckpt_fpath=ckpt_fpath,
        hawaii=btx.data.HawaiiConfig(
            split="val", include_polylines=False, hf_root=HAWAII_HF_ROOT
        ),
        beetlepalooza=btx.data.BeetlePaloozaConfig(go=False),
        biorepo=btx.data.BioRepoConfig(go=False),
        batch_size=32,
        n_workers=0,
        out_fpath=out_fpath,
    )
    inference.infer(infer_cfg)

    # 3) Verify Parquet output.
    assert out_fpath.exists()
    df = pl.read_parquet(out_fpath)

    assert len(df) > 0
    assert df.schema == inference.SCHEMA
    assert df["dataset"].unique().to_list() == ["hawaii"]

    pred_coords = np.array(df["pred_coords_px"].to_list())
    assert np.isfinite(pred_coords).all()

    cls_embeddings = np.array(df["cls_embedding"].to_list())
    assert np.isfinite(cls_embeddings).all()
    assert cls_embeddings.shape == (len(df), 384)

    entropy = df["mean_entropy"].to_numpy()
    assert np.isfinite(entropy).all()
    assert (entropy > 0).all()
