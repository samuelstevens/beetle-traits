import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    import tempfile

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import wandb
    from PIL import Image

    return Image, mo, np, os, pl, plt, tempfile, wandb


@app.cell
def _(mo, np, pl, wandb):
    entity = "samuelstevens"
    project = "beetle-traits"
    heatmap_tag = "ce-fix"
    baseline_tag = "exp-002-augmentation"

    def normalize_cfg_value(value):
        if value is None:
            return None
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (bool, int, float, str)):
            return value
        return str(value)

    def flatten_cfg(dct: dict) -> dict[str, object]:
        flat = {}
        stack = [("", dct)]
        while stack:
            prefix, node = stack.pop()
            for key, value in node.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                if isinstance(value, dict):
                    stack.append((path, value))
                    continue
                flat[path] = normalize_cfg_value(value)
        return flat

    api = wandb.Api()

    def fetch_runs(tag: str, *, run_filter=None) -> pl.DataFrame:
        runs = list(
            api.runs(path=f"{entity}/{project}", filters={"tags": {"$in": [tag]}})
        )
        rows = []
        for run in mo.status.progress_bar(runs, title=tag):
            if run_filter is not None and not run_filter(run):
                continue
            cfg_flat = flatten_cfg(dict(run.config))
            run_meta = {
                "run_id": run.id,
                "name": run.name,
                "state": run.state,
                **cfg_flat,
            }
            for entry in run.scan_history():
                step = entry.get("_step", entry.get("step"))
                if step is None:
                    continue
                row = dict(run_meta)
                row["step"] = int(step)
                for key, value in entry.items():
                    if key in {"_step", "step"}:
                        continue
                    if isinstance(value, bool) or not isinstance(
                        value, (int, float, np.integer, np.floating)
                    ):
                        continue
                    if not np.isfinite(value):
                        continue
                    row[key.replace("/", "_")] = float(value)
                rows.append(row)
        return pl.DataFrame(rows) if rows else pl.DataFrame()

    # Only include exp-002 runs with per-dataset aug config (new sweep),
    # matching the filter in docs/experiments/002-augmentation/notebook.py.
    def _baseline_filter(run) -> bool:
        return "crop" in run.config.get("aug_hawaii", {})

    heatmap_df = fetch_runs(heatmap_tag)
    baseline_df = fetch_runs(baseline_tag, run_filter=_baseline_filter)
    return api, baseline_df, entity, heatmap_df, project


@app.cell
def _(heatmap_df, mo, pl, plt):
    mo.stop(heatmap_df.is_empty(), mo.md("No heatmap runs found."))

    _sigma_col = "objective.sigma"
    _sigma_colors = {1.0: "#1f77b4", 2.0: "#ff7f0e", 3.0: "#2ca02c"}
    _datasets = [
        ("val_hawaii_length_line_err_cm", "Hawaii"),
        ("val_biorepo_length_line_err_cm", "BioRepo"),
    ]

    _fig, _axes = plt.subplots(
        1, 2, figsize=(14, 4.5), dpi=140, sharey=True, layout="constrained"
    )

    for _col_i, (_metric, _ds_label) in enumerate(_datasets):
        _ax = _axes[_col_i]
        if _metric not in heatmap_df.columns:
            _ax.set_title(f"{_ds_label} (missing {_metric})")
            _ax.spines[["right", "top"]].set_visible(False)
            continue

        _hm = (
            heatmap_df
            .filter(pl.col(_metric).is_not_null())
            .sort(["run_id", "step"])
            .group_by("run_id")
            .agg(
                pl.col("learning_rate").first(),
                pl.col(_sigma_col).first().alias("sigma"),
                pl.col(_metric).min().alias("best"),
                pl.col(_metric).drop_nulls().last().alias("final"),
            )
        )

        for _s, _color in _sigma_colors.items():
            _sub = _hm.filter(pl.col("sigma") == _s).sort("learning_rate")
            if _sub.is_empty():
                continue
            _lrs = _sub.get_column("learning_rate").to_numpy()
            _ax.plot(_lrs, _sub.get_column("best").to_numpy(), "o-", color=_color, linewidth=1.8, markersize=6, label=f"best s={_s:g}")
            _ax.plot(_lrs, _sub.get_column("final").to_numpy(), "o--", color=_color, linewidth=1.2, markersize=4, label=f"final s={_s:g}")

        _ax.set_xscale("log")
        _ax.set_yscale("log")
        _ax.set_xlabel("Learning rate")
        _ax.set_ylabel("length_line_err_cm")
        _ax.set_title(f"LR vs val line error ({_ds_label})")
        _ax.legend(fontsize=7, frameon=False, ncol=2)
        _ax.grid(alpha=0.25)
        _ax.spines[["right", "top"]].set_visible(False)

    _fig
    return


@app.cell
def _(heatmap_df, mo, pl, plt):
    mo.stop(heatmap_df.is_empty(), mo.md("No heatmap runs found."))

    _sigma_col = "objective.sigma"
    _sigma_colors = {1.0: "#1f77b4", 2.0: "#ff7f0e", 3.0: "#2ca02c"}
    _datasets = [
        ("val_hawaii_loss", "Hawaii"),
        ("val_biorepo_loss", "BioRepo"),
    ]

    _fig, _axes = plt.subplots(
        1, 2, figsize=(14, 4.5), dpi=140, sharey=True, layout="constrained"
    )

    for _col_i, (_metric, _ds_label) in enumerate(_datasets):
        _ax = _axes[_col_i]
        if _metric not in heatmap_df.columns:
            _ax.set_title(f"{_ds_label} (missing {_metric})")
            _ax.spines[["right", "top"]].set_visible(False)
            continue

        _hm = (
            heatmap_df
            .filter(pl.col(_metric).is_not_null())
            .sort(["run_id", "step"])
            .group_by("run_id")
            .agg(
                pl.col("learning_rate").first(),
                pl.col(_sigma_col).first().alias("sigma"),
                pl.col(_metric).min().alias("best"),
                pl.col(_metric).drop_nulls().last().alias("final"),
            )
        )

        for _s, _color in _sigma_colors.items():
            _sub = _hm.filter(pl.col("sigma") == _s).sort("learning_rate")
            if _sub.is_empty():
                continue
            _lrs = _sub.get_column("learning_rate").to_numpy()
            _ax.plot(_lrs, _sub.get_column("best").to_numpy(), "o-", color=_color, linewidth=1.8, markersize=6, label=f"best s={_s:g}")
            _ax.plot(_lrs, _sub.get_column("final").to_numpy(), "o--", color=_color, linewidth=1.2, markersize=4, label=f"final s={_s:g}")

        _ax.set_xscale("log")
        _ax.set_yscale("log")
        _ax.set_xlabel("Learning rate")
        _ax.set_ylabel("val loss")
        _ax.set_title(f"LR vs val loss ({_ds_label})")
        _ax.legend(fontsize=7, frameon=False, ncol=2)
        _ax.grid(alpha=0.25)
        _ax.spines[["right", "top"]].set_visible(False)

    _fig
    return


@app.cell
def _(heatmap_df, mo, pl, plt):
    mo.stop(heatmap_df.is_empty(), mo.md("No heatmap runs found."))

    _sigma_col = "objective.sigma"
    _sigma_colors = {1.0: "#1f77b4", 2.0: "#ff7f0e", 3.0: "#2ca02c"}
    _datasets = [
        ("val_hawaii_point_err_cm", "Hawaii"),
        ("val_biorepo_point_err_cm", "BioRepo"),
    ]

    _fig, _axes = plt.subplots(
        1, 2, figsize=(14, 4.5), dpi=140, sharey=True, layout="constrained"
    )

    for _col_i, (_metric, _ds_label) in enumerate(_datasets):
        _ax = _axes[_col_i]
        if _metric not in heatmap_df.columns:
            _ax.set_title(f"{_ds_label} (missing {_metric})")
            _ax.spines[["right", "top"]].set_visible(False)
            continue

        _hm = (
            heatmap_df
            .filter(pl.col(_metric).is_not_null())
            .sort(["run_id", "step"])
            .group_by("run_id")
            .agg(
                pl.col("learning_rate").first(),
                pl.col(_sigma_col).first().alias("sigma"),
                pl.col(_metric).min().alias("best"),
                pl.col(_metric).drop_nulls().last().alias("final"),
            )
        )

        for _s, _color in _sigma_colors.items():
            _sub = _hm.filter(pl.col("sigma") == _s).sort("learning_rate")
            if _sub.is_empty():
                continue
            _lrs = _sub.get_column("learning_rate").to_numpy()
            _ax.plot(_lrs, _sub.get_column("best").to_numpy(), "o-", color=_color, linewidth=1.8, markersize=6, label=f"best s={_s:g}")
            _ax.plot(_lrs, _sub.get_column("final").to_numpy(), "o--", color=_color, linewidth=1.2, markersize=4, label=f"final s={_s:g}")

        _ax.set_xscale("log")
        _ax.set_yscale("log")
        _ax.set_xlabel("Learning rate")
        _ax.set_ylabel("point_err_cm")
        _ax.set_title(f"LR vs val point error ({_ds_label})")
        _ax.legend(fontsize=7, frameon=False, ncol=2)
        _ax.grid(alpha=0.25)
        _ax.spines[["right", "top"]].set_visible(False)

    _fig
    return


@app.cell
def _(heatmap_df, mo, pl):
    mo.stop(heatmap_df.is_empty(), mo.md("No heatmap data for CSV."))

    _sigma_col = "objective.sigma"

    _metrics = [
        ("val_hawaii_loss", "hawaii_loss"),
        ("val_biorepo_loss", "biorepo_loss"),
        ("val_hawaii_point_err_cm", "hawaii_point_err_cm"),
        ("val_biorepo_point_err_cm", "biorepo_point_err_cm"),
        ("val_hawaii_length_line_err_cm", "hawaii_line_err_cm"),
        ("val_biorepo_length_line_err_cm", "biorepo_line_err_cm"),
    ]

    _aggs = []
    for _col, _alias in _metrics:
        if _col not in heatmap_df.columns:
            continue
        _aggs.append(pl.col(_col).min().alias(f"best_{_alias}"))
        _aggs.append(pl.col(_col).drop_nulls().last().alias(f"final_{_alias}"))

    sweep_csv = (
        heatmap_df
        .sort(["run_id", "step"])
        .group_by("run_id")
        .agg(
            pl.col("learning_rate").first(),
            pl.col(_sigma_col).first().alias("sigma"),
            pl.col("step").max().alias("final_step"),
            *_aggs,
        )
        .sort(["sigma", "learning_rate"])
    )
    sweep_csv.write_csv("docs/experiments/004-heatmap/sweep_results.csv")
    sweep_csv
    return


@app.cell
def _(baseline_df, heatmap_df, mo, pl, plt):
    mo.stop(heatmap_df.is_empty(), mo.md("No heatmap runs found."))

    _datasets = [
        ("hawaii", "Hawaii"),
        ("biorepo", "BioRepo"),
    ]

    def _best_run_curve(df, rank_col):
        _vals = df.filter(pl.col(rank_col).is_not_null())
        if _vals.is_empty():
            return None, None
        _rid = _vals.sort(rank_col).head(1).get_column("run_id").to_list()[0]
        return _rid, df.filter(pl.col("run_id") == _rid).sort("step")

    _fig, _axes = plt.subplots(
        1, 2, figsize=(14, 4.5), dpi=140, sharey=True, layout="constrained"
    )

    for _col_i, (_ds, _ds_label) in enumerate(_datasets):
        _ax = _axes[_col_i]
        _mean_col = f"val_{_ds}_length_line_err_cm"
        _med_col = f"val_{_ds}_median_length_line_err_cm"

        # Heatmap lines (green).
        _hm_rid, _hm_df = _best_run_curve(heatmap_df, _mean_col)
        if _hm_df is not None:
            _meta = _hm_df.head(1).to_dicts()[0]
            _sigma = _meta.get("objective.sigma", "?")
            _lr = _meta.get("learning_rate", "?")
            _sfx = f"(s={_sigma}, lr={_lr})"
            for _col, _ls, _lbl in [
                (_mean_col, "-", f"heatmap mean {_sfx}"),
                (_med_col, "--", f"heatmap median {_sfx}"),
            ]:
                if _col not in _hm_df.columns:
                    continue
                _v = _hm_df.filter(pl.col(_col).is_not_null()).slice(1)
                if _v.is_empty():
                    continue
                _ax.plot(
                    _v.get_column("step").to_numpy(),
                    _v.get_column(_col).to_numpy(),
                    color="#2ca02c",
                    linestyle=_ls,
                    linewidth=1.8,
                    label=_lbl,
                )

        # MLP baseline lines (blue).
        if not baseline_df.is_empty():
            _bl_rid, _bl_df = _best_run_curve(baseline_df, _mean_col)
            if _bl_df is not None:
                _bl_lr = _bl_df.head(1).to_dicts()[0].get("learning_rate", "?")
                for _col, _ls, _lbl in [
                    (_mean_col, "-", f"MLP mean (lr={_bl_lr})"),
                    (_med_col, "--", f"MLP median (lr={_bl_lr})"),
                ]:
                    if _col not in _bl_df.columns:
                        continue
                    _v = _bl_df.filter(pl.col(_col).is_not_null()).slice(1)
                    if _v.is_empty():
                        continue
                    _ax.plot(
                        _v.get_column("step").to_numpy(),
                        _v.get_column(_col).to_numpy(),
                        color="#1f77b4",
                        linestyle=_ls,
                        linewidth=1.8,
                        label=_lbl,
                    )

        _ax.set_xlabel("Step")
        _ax.set_yscale("log")
        _ax.set_ylabel("length_line_err (cm)")
        _ax.set_title(f"Best heatmap vs best MLP ({_ds_label})")
        _ax.legend(fontsize=8, frameon=False)
        _ax.grid(alpha=0.25)
        _ax.spines[["right", "top"]].set_visible(False)

    _fig
    return


@app.cell
def _(
    Image,
    api,
    baseline_df,
    entity,
    heatmap_df,
    mo,
    os,
    pl,
    project,
    tempfile,
):
    mo.stop(heatmap_df.is_empty(), mo.md("No heatmap data."))

    _metric = "val_hawaii_length_line_err_cm"

    def _find_best_run_id(df):
        if df.is_empty() or _metric not in df.columns:
            return None
        _vals = df.filter(pl.col(_metric).is_not_null())
        if _vals.is_empty():
            return None
        return _vals.sort(_metric).head(1).get_column("run_id").to_list()[0]

    def _max_step(df, run_id):
        return int(
            df.filter(pl.col("run_id") == run_id).select(pl.col("step").max()).item()
        )

    def fetch_fixed_images(run_id, targets):
        """Download fixed val images at target training steps in one scan.

        Returns (targets, {target: {beetle_key: Image}}, {target: actual_step}).
        """
        _run = api.run(f"{entity}/{project}/{run_id}")

        _best = {s: (float("inf"), None, 0) for s in targets}
        for _row in _run.scan_history():
            _step = _row.get("_step", 0)
            _has_fixed = any(
                k.startswith("val/") and "/images/fixed/" in k
                for k in _row
                if isinstance(_row[k], dict)
            )
            if not _has_fixed:
                continue
            for _tgt in targets:
                _dist = abs(_step - _tgt)
                if _dist < _best[_tgt][0]:
                    _best[_tgt] = (_dist, dict(_row), _step)

        _tmpdir = tempfile.mkdtemp()
        _all_imgs = {}
        _actual_steps = {}
        for _tgt, (_dist, _row, _step) in _best.items():
            _actual_steps[_tgt] = _step
            _imgs = {}
            if _row is None:
                _all_imgs[_tgt] = _imgs
                continue
            for _k, _v in _row.items():
                if not _k.startswith("val/") or "/images/fixed/" not in _k:
                    continue
                if not isinstance(_v, dict) or _v.get("_type") != "image-file":
                    continue
                _path = _v["path"]
                _run.file(_path).download(root=_tmpdir, replace=True)
                _img = Image.open(os.path.join(_tmpdir, _path))
                _img.thumbnail((128, 128))
                _imgs[_k] = _img
            _all_imgs[_tgt] = _imgs
        return targets, _all_imgs, _actual_steps

    _hm_id = _find_best_run_id(heatmap_df)
    _bl_id = _find_best_run_id(baseline_df)

    hm_imgs, bl_imgs = None, None
    if _bl_id:
        _bl_max = _max_step(baseline_df, _bl_id)
        _bl_targets = [int(_bl_max * p) for p in (0.25, 0.50, 0.75, 1.0)]
        bl_imgs = fetch_fixed_images(_bl_id, _bl_targets)
    if _hm_id:
        _hm_max = _max_step(heatmap_df, _hm_id)
        _hm_targets = [_hm_max - 15000, _hm_max - 10000, _hm_max - 5000, _hm_max]
        hm_imgs = fetch_fixed_images(_hm_id, _hm_targets)
    return bl_imgs, hm_imgs


@app.cell
def _(mo):
    mo.md("""
    ### MLP baseline

    green=GT, red=pred, blue+=center
    """)
    return


@app.cell
def _(bl_imgs, mo, np, plt):
    mo.stop(bl_imgs is None, mo.md("No MLP baseline run found."))
    _targets, _all_imgs, _actual_steps = bl_imgs

    _beetle_keys = sorted({k for _imgs in _all_imgs.values() for k in _imgs})
    mo.stop(not _beetle_keys, mo.md("No MLP baseline images."))

    _fig, _axes = plt.subplots(
        len(_beetle_keys),
        len(_targets),
        figsize=(2.5 * len(_targets), 2.5 * len(_beetle_keys)),
        dpi=72,
        layout="constrained",
        squeeze=False,
    )
    for _c, _tgt in enumerate(_targets):
        _axes[0, _c].set_title(f"step {_actual_steps.get(_tgt, _tgt)}", fontsize=8)
    for _r, _bkey in enumerate(_beetle_keys):
        for _c, _tgt in enumerate(_targets):
            _ax = _axes[_r, _c]
            _ax.axis("off")
            _img = _all_imgs.get(_tgt, {}).get(_bkey)
            if _img is None:
                continue
            _arr = np.asarray(_img)
            _ax.imshow(_arr)
            _cy, _cx = _arr.shape[0] / 2, _arr.shape[1] / 2
            _ax.plot(_cx, _cy, "+", color="blue", markersize=8, markeredgewidth=1.5)
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ### Heatmap (CE)

    green=GT, red=pred, blue+=center
    """)
    return


@app.cell
def _(hm_imgs, mo, np, plt):
    mo.stop(hm_imgs is None, mo.md("No heatmap run found."))
    _targets, _all_imgs, _actual_steps = hm_imgs

    _beetle_keys = sorted({k for _imgs in _all_imgs.values() for k in _imgs})
    mo.stop(not _beetle_keys, mo.md("No heatmap images."))

    _fig, _axes = plt.subplots(
        len(_beetle_keys),
        len(_targets),
        figsize=(2.5 * len(_targets), 2.5 * len(_beetle_keys)),
        dpi=72,
        layout="constrained",
        squeeze=False,
    )
    for _c, _tgt in enumerate(_targets):
        _axes[0, _c].set_title(f"step {_actual_steps.get(_tgt, _tgt)}", fontsize=8)
    for _r, _bkey in enumerate(_beetle_keys):
        for _c, _tgt in enumerate(_targets):
            _ax = _axes[_r, _c]
            _ax.axis("off")
            _img = _all_imgs.get(_tgt, {}).get(_bkey)
            if _img is None:
                continue
            _arr = np.asarray(_img)
            _ax.imshow(_arr)
            _cy, _cx = _arr.shape[0] / 2, _arr.shape[1] / 2
            _ax.plot(_cx, _cy, "+", color="blue", markersize=8, markeredgewidth=1.5)
    _fig
    return


if __name__ == "__main__":
    app.run()
