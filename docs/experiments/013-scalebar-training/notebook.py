import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    from PIL import Image

    import btx.data as data

    return Image, data, mo, np, pl, plt


@app.cell
def _(mo):
    mo.md("""
    # Experiment 013: Scalebar training inputs

    The training sweep in `sweeps/train.py` uses `aug.size=512` and `aug.go=False`, so the model sees deterministic 512x512 bilinear resizes of the full group images before normalization. This notebook shows a few examples side by side: original image on the left, resized training input on the right.
    """)
    return


@app.cell
def _(mo):
    split = mo.ui.dropdown(["train", "val"], value="train", label="Split")
    n_show = mo.ui.slider(1, 8, value=4, label="Examples")
    seed = mo.ui.number(value=0, label="Sample seed")
    return n_show, seed, split


@app.cell
def _(pl):
    df = pl.read_csv("/fs/ess/PAS2136/CarabidImaging/allImages.csv", null_values=["NA"])
    df
    return


@app.cell
def _(data, split):
    size = 256
    cfg = data.ScalebarGroupConfig(
        split=split.value,
        cache=True,
        cache_size=size,
    )
    ds = data.ScalebarGroupDataset(cfg)
    assert len(ds) > 0, f"Empty scalebar split: {split.value}"
    return ds, size


@app.cell
def _(ds, n_show, np, seed):
    n = min(int(n_show.value), len(ds))
    rng = np.random.default_rng(int(seed.value))
    sample_indices = rng.choice(len(ds), size=n, replace=False)
    return (sample_indices,)


@app.cell
def _(Image, ds, mo, np, plt, sample_indices, size):
    rows = len(sample_indices)
    fig, axes = plt.subplots(
        rows,
        2,
        figsize=(10, 3.5 * rows),
        dpi=110,
        layout="constrained",
    )
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for ax_row, idx in zip(axes, sample_indices, strict=True):
        sample = ds[int(idx)]
        img_fpath = sample["img_fpath"]
        with Image.open(img_fpath) as im:
            rgb = im.convert("RGB")
            orig = np.asarray(rgb)
            resized = np.asarray(rgb.resize((size, size), Image.Resampling.BILINEAR))

        ax_orig, ax_resized = ax_row
        ax_orig.imshow(orig)
        ax_orig.set_title(
            f"Original: {sample['group_img_basename']} ({orig.shape[1]}x{orig.shape[0]})",
            fontsize=8,
        )
        ax_orig.set_xticks([])
        ax_orig.set_yticks([])
        ax_orig.spines[:].set_visible(False)

        ax_resized.imshow(resized)
        ax_resized.set_title(f"Training input: {size}x{size}", fontsize=8)
        ax_resized.set_xticks([])
        ax_resized.set_yticks([])
        ax_resized.spines[:].set_visible(False)

    fig.suptitle(
        f"Scalebar group images for split={ds.cfg.split!r} (left: original, right: resized training input)",
        fontsize=10,
    )
    mo.mpl.interactive(fig)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Clip images at 256x256

    Each sample is a tight bbox crop around the scalebar (from `make_scalebar_clips.py`), resized to 256x256 for training. The scalebar fills most of the frame.
    """)
    return


@app.cell
def _(data):
    clip_size = 256
    clip_cfg = data.ScalebarClipConfig(split="train", cache=True, cache_size=clip_size)
    clip_ds = data.ScalebarClipDataset(clip_cfg)
    assert len(clip_ds) > 0, "Empty clip dataset"
    return clip_ds, clip_size


@app.cell
def _(clip_ds, n_show, np, seed):
    clip_n = min(int(n_show.value), len(clip_ds))
    clip_rng = np.random.default_rng(int(seed.value))
    clip_indices = clip_rng.choice(len(clip_ds), size=clip_n, replace=False)
    return clip_indices, clip_n


@app.cell
def _(Image, clip_ds, clip_indices, clip_n, clip_size, mo, np, plt):
    _fig, _axes = plt.subplots(
        clip_n,
        2,
        figsize=(8, 3.5 * clip_n),
        dpi=110,
        layout="constrained",
    )
    if clip_n == 1:
        _axes = np.expand_dims(_axes, axis=0)

    for _ax_row, _idx in zip(_axes, clip_indices, strict=True):
        _sample = clip_ds[int(_idx)]
        with Image.open(_sample["img_fpath"]) as _im:
            _clip_orig = np.asarray(_im.convert("RGB"))
            _clip_256 = np.asarray(
                _im.resize((clip_size, clip_size), Image.Resampling.BILINEAR)
            )

        _pts = _sample["points_px"][0]  # scalebar endpoints, shape (2, 2)
        # Scale GT endpoints from cache_size coords to original clip coords for display
        _ch, _cw = _clip_orig.shape[:2]
        _pts_orig = _pts * np.array([_cw / clip_size, _ch / clip_size])

        _ax_orig, _ax_256 = _ax_row
        _ax_orig.imshow(_clip_orig)
        _ax_orig.plot(
            [_pts_orig[0, 0], _pts_orig[1, 0]],
            [_pts_orig[0, 1], _pts_orig[1, 1]],
            color="lime",
            linewidth=1.5,
            marker="x",
            markersize=5,
        )
        _ax_orig.set_title(
            f"Clip: {_sample['group_img_basename']} ({_cw}x{_ch})", fontsize=8
        )
        _ax_orig.set_xticks([])
        _ax_orig.set_yticks([])
        _ax_orig.spines[:].set_visible(False)

        _ax_256.imshow(_clip_256)
        _ax_256.plot(
            [_pts[0, 0], _pts[1, 0]],
            [_pts[0, 1], _pts[1, 1]],
            color="lime",
            linewidth=1.5,
            marker="x",
            markersize=5,
        )
        _ax_256.set_title(f"Training input: {clip_size}x{clip_size}", fontsize=8)
        _ax_256.set_xticks([])
        _ax_256.set_yticks([])
        _ax_256.spines[:].set_visible(False)

    _fig.suptitle(
        "Scalebar clips (left: original crop, right: 256x256 training input, green=GT scalebar)",
        fontsize=10,
    )
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Heatmap visualization

    Load a trained checkpoint and overlay the predicted endpoint probability heatmaps
    on val clips. Green line/+ marks the GT scalebar; the hot colormap shows where the
    model places probability mass for each endpoint channel (p0, p1).
    """)
    return


@app.cell
def _(mo):
    import pathlib as _pathlib

    _ckpt_dpath = _pathlib.Path("checkpoints/exp013")
    _run_ids = (
        sorted([
            d.name
            for d in _ckpt_dpath.iterdir()
            if d.is_dir() and (d / "model.eqx").exists()
        ])
        if _ckpt_dpath.exists()
        else []
    )

    ckpt_run_id = mo.ui.dropdown(
        _run_ids or ["(none)"],
        value=_run_ids[0] if _run_ids else "(none)",
        label="Checkpoint run",
    )
    n_viz = mo.ui.slider(1, 8, value=4, label="Clips")
    mo.hstack([ckpt_run_id, n_viz])
    return ckpt_run_id, n_viz


@app.cell
def _(ckpt_run_id, mo):
    import pathlib as _pathlib

    import btx.modeling

    _ckpt_fpath = _pathlib.Path(f"checkpoints/exp013/{ckpt_run_id.value}/model.eqx")
    mo.stop(not _ckpt_fpath.exists(), mo.md(f"Checkpoint not found: `{_ckpt_fpath}`"))
    viz_model, _viz_model_cfg, viz_obj_cfg = btx.modeling.load_ckpt(_ckpt_fpath)
    return viz_model, viz_obj_cfg


@app.cell
def _(Image, data, mo, n_viz, np, plt, viz_model, viz_obj_cfg):
    import typing as _tp

    import equinox as eqx
    import jax

    _val_cfg = data.ScalebarClipConfig(split="val", cache=False)
    _val_ds = data.ScalebarClipDataset(_val_cfg)
    _n = min(int(n_viz.value), len(_val_ds))

    _size = viz_obj_cfg.image_size
    _hm_size = viz_obj_cfg.heatmap_size
    _mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    @eqx.filter_jit
    def _forward(model, x):
        return _tp.cast(_tp.Callable, model)(x)

    def _softmax_2d(x):
        e = np.exp(x - x.max())
        return e / e.sum()

    def _upsample_heatmap(prob):
        return (
            np.asarray(
                Image.fromarray((prob * 255).astype(np.uint8)).resize(
                    (_size, _size), Image.Resampling.BILINEAR
                ),
                dtype=np.float32,
            )
            / 255.0
        )

    _fig, _axes = plt.subplots(
        _n, 3, figsize=(10, 3.2 * _n), dpi=110, layout="constrained", squeeze=False
    )
    _fig.suptitle(
        "Val clips: GT scalebar (green) and predicted endpoint heatmaps", fontsize=10
    )

    for _i in range(_n):
        _sample = _val_ds[_i]
        with Image.open(_sample["img_fpath"]) as _im:
            _clip_rgb = _im.convert("RGB")
            _clip_w, _clip_h = _clip_rgb.size
            _clip_u8 = np.asarray(
                _clip_rgb.resize((_size, _size), Image.Resampling.BILINEAR)
            )

        # Normalize and run forward pass -> (4, hm_size, hm_size) logits.
        _inp = (_clip_u8.astype(np.float32) / 255.0 - _mean) / _std
        _logits = np.asarray(_forward(viz_model, jax.device_put(_inp)))

        _prob0 = _softmax_2d(_logits[0])  # scalebar endpoint p0
        _prob1 = _softmax_2d(_logits[1])  # scalebar endpoint p1

        # GT endpoints scaled to _size x _size display space.
        _pts = _sample["points_px"][0]  # (2, 2): [[x0,y0],[x1,y1]]
        _scale = np.array([_size / _clip_w, _size / _clip_h])
        _pts_d = _pts * _scale

        _h0 = _upsample_heatmap(_prob0)
        _h1 = _upsample_heatmap(_prob1)

        # Column 0: clip with GT scalebar line.
        _ax = _axes[_i, 0]
        _ax.imshow(_clip_u8)
        _ax.plot(
            [_pts_d[0, 0], _pts_d[1, 0]],
            [_pts_d[0, 1], _pts_d[1, 1]],
            color="lime",
            linewidth=1.5,
        )
        _ax.plot(*_pts_d[0], "g+", ms=10, mew=2)
        _ax.plot(*_pts_d[1], "gs", ms=6)
        _ax.set_title(_sample["group_img_basename"], fontsize=7)
        _ax.axis("off")

        # Columns 1-2: heatmap overlays for each endpoint.
        for _col, (_h, _pt, _label) in enumerate(
            [(_h0, _pts_d[0], "p0"), (_h1, _pts_d[1], "p1")], start=1
        ):
            _ax = _axes[_i, _col]
            _ax.imshow(_clip_u8, alpha=0.45)
            _ax.imshow(_h, cmap="hot", alpha=0.65, vmin=0)
            _ax.plot(*_pt, "g+", ms=10, mew=2)
            _ax.set_title(f"heatmap {_label}", fontsize=8)
            _ax.axis("off")

    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Inference results: checkpoint fdpj3qnm

    Results from running `sweeps/infer.py` with checkpoint `checkpoints/exp013/fdpj3qnm/model.eqx`.
    Each entry has `px_per_cm` (Euclidean distance between predicted scalebar endpoints),
    `scalebar_pts` (two endpoints in original image coordinates), and `bbox` (the detected scalebar crop).
    """)
    return


@app.cell
def _(mo, np, pl):
    import json as _json
    import pathlib as _pathlib

    _results_fpath = _pathlib.Path(
        "docs/experiments/013-scalebar-training/results/fdpj3qnm_px_per_cm.json"
    )
    mo.stop(
        not _results_fpath.exists(),
        mo.md(f"Results not found: `{_results_fpath}`. Run `sweeps/infer.py` first."),
    )

    with _results_fpath.open() as _fd:
        _raw: dict = _json.load(_fd)

    _df = pl.DataFrame({
        "img_fpath": list(_raw.keys()),
        "px_per_cm": [v["px_per_cm"] for v in _raw.values()],
    })

    _vals = np.array(_df["px_per_cm"].to_list(), dtype=np.float32)
    infer_results = _raw
    mo.vstack([
        mo.md(
            f"**{len(_df)} images** | px/cm: mean={_vals.mean():.1f}, median={float(np.median(_vals)):.1f}, std={_vals.std():.1f}, min={_vals.min():.1f}, max={_vals.max():.1f}"
        ),
        mo.ui.table(_df.head(20)),
    ])
    return (infer_results,)


@app.cell
def _(Image, infer_results, mo, np, plt):
    _n_show = min(6, len(infer_results))
    _keys = sorted(infer_results.keys())[:_n_show]

    _fig, _axes = plt.subplots(
        _n_show,
        2,
        figsize=(10, 3.5 * _n_show),
        dpi=110,
        layout="constrained",
        squeeze=False,
    )
    _fig.suptitle(
        "Inference results (fdpj3qnm): bbox crop (left) and predicted scalebar (right)",
        fontsize=10,
    )

    for _i, _fpath in enumerate(_keys):
        _entry = infer_results[_fpath]
        _bbox = _entry["bbox"]
        _pts = np.array(_entry["scalebar_pts"])  # (2, 2): [[x0,y0],[x1,y1]]

        with Image.open(_fpath) as _im:
            _rgb = _im.convert("RGB")
            _bx, _by, _bw, _bh = _bbox["x"], _bbox["y"], _bbox["w"], _bbox["h"]
            _crop = np.asarray(_rgb.crop((_bx, _by, _bx + _bw, _by + _bh)))

        # Translate predicted pts into crop-local coordinates for display.
        _pts_crop = _pts - np.array([_bx, _by])

        _ax_full, _ax_crop = _axes[_i]

        _ax_full.imshow(np.asarray(_rgb))
        _ax_full.plot(_pts[:, 0], _pts[:, 1], "r-", linewidth=1.5)
        _ax_full.plot(*_pts[0], "r+", ms=8, mew=2)
        _ax_full.plot(*_pts[1], "rs", ms=5)
        _ax_full.set_title(_fpath.split("/")[-1], fontsize=7)
        _ax_full.axis("off")

        _ax_crop.imshow(_crop)
        _ax_crop.plot(_pts_crop[:, 0], _pts_crop[:, 1], "r-", linewidth=2)
        _ax_crop.plot(*_pts_crop[0], "r+", ms=10, mew=2)
        _ax_crop.plot(*_pts_crop[1], "rs", ms=6)
        _ax_crop.set_title(f"px/cm={_entry['px_per_cm']:.1f}", fontsize=8)
        _ax_crop.axis("off")

    mo.mpl.interactive(_fig)
    return


if __name__ == "__main__":
    app.run()
