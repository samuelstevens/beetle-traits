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
    heatmap_tag = "exp-004-heatmap"
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
    return baseline_df, heatmap_df


@app.cell
def _(baseline_df, heatmap_df, mo, pl, plt):
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

        # Best val metric per heatmap run.
        _hm = (
            heatmap_df
            .filter(pl.col(_metric).is_not_null())
            .sort(["run_id", _metric])
            .group_by("run_id")
            .agg(
                pl.col("learning_rate").first(),
                pl.col(_sigma_col).first().alias("sigma"),
                pl.col(_metric).first().alias("best_err"),
            )
        )

        for _s, _color in _sigma_colors.items():
            _sub = _hm.filter(pl.col("sigma") == _s).sort("learning_rate")
            if _sub.is_empty():
                continue
            _ax.plot(
                _sub.get_column("learning_rate").to_numpy(),
                _sub.get_column("best_err").to_numpy(),
                "o-",
                color=_color,
                linewidth=1.8,
                markersize=6,
                label=f"sigma={_s:g}",
            )

        # Best full-aug baseline.
        if not baseline_df.is_empty() and _metric in baseline_df.columns:
            _bl = baseline_df.filter(pl.col(_metric).is_not_null())
            if not _bl.is_empty():
                _bl_best = float(_bl.select(pl.col(_metric).min()).item())
                _ax.axhline(
                    _bl_best,
                    color="#888",
                    linestyle="--",
                    linewidth=1.2,
                    label=f"best MLP ({_bl_best:.3f})",
                )

        _ax.set_xscale("log")
        _ax.set_xlabel("Learning rate")
        _ax.set_ylabel("Best length_line_err_cm")
        _ax.set_title(f"LR vs best val error ({_ds_label})")
        _ax.legend(fontsize=8, frameon=False)
        _ax.grid(alpha=0.25)
        _ax.spines[["right", "top"]].set_visible(False)

    _fig
    return


@app.cell
def _(baseline_df, heatmap_df, mo, pl, plt):
    mo.stop(heatmap_df.is_empty(), mo.md("No heatmap runs found."))

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

        # Find the heatmap run with the lowest best val metric.
        _hm_val = heatmap_df.filter(pl.col(_metric).is_not_null())
        _hm_best_run = _hm_val.sort(_metric).head(1).get_column("run_id").to_list()[0]
        _hm_curve = (
            _hm_val
            .filter(pl.col("run_id") == _hm_best_run)
            .sort("step")
            .select("step", _metric)
        )
        _hm_meta = (
            heatmap_df.filter(pl.col("run_id") == _hm_best_run).head(1).to_dicts()[0]
        )
        _hm_sigma = _hm_meta.get("objective.sigma", "?")
        _hm_lr = _hm_meta.get("learning_rate", "?")

        _ax.plot(
            _hm_curve.get_column("step").to_numpy(),
            _hm_curve.get_column(_metric).to_numpy(),
            color="#2ca02c",
            linewidth=1.8,
            label=f"heatmap (sigma={_hm_sigma}, lr={_hm_lr})",
        )

        # Find the best baseline run (full-aug only).
        if not baseline_df.is_empty() and _metric in baseline_df.columns:
            _bl = baseline_df.filter(pl.col(_metric).is_not_null())
            if not _bl.is_empty():
                _bl_best_run = (
                    _bl.sort(_metric).head(1).get_column("run_id").to_list()[0]
                )
                _bl_curve = (
                    _bl
                    .filter(pl.col("run_id") == _bl_best_run)
                    .sort("step")
                    .select("step", _metric)
                )
                _bl_meta = (
                    baseline_df
                    .filter(pl.col("run_id") == _bl_best_run)
                    .head(1)
                    .to_dicts()[0]
                )
                _bl_lr = _bl_meta.get("learning_rate", "?")
                _ax.plot(
                    _bl_curve.get_column("step").to_numpy(),
                    _bl_curve.get_column(_metric).to_numpy(),
                    color="#1f77b4",
                    linewidth=1.8,
                    label=f"MLP baseline (lr={_bl_lr})",
                )

        _ax.set_xlabel("Step")
        _ax.set_ylabel("length_line_err_cm")
        _ax.set_title(f"Best heatmap vs best MLP ({_ds_label})")
        _ax.legend(fontsize=8, frameon=False)
        _ax.grid(alpha=0.25)
        _ax.spines[["right", "top"]].set_visible(False)

    _fig
    return


@app.cell
def _(Image, baseline_df, heatmap_df, mo, np, os, pl, plt, tempfile, wandb):
    mo.stop(heatmap_df.is_empty(), mo.md("No heatmap data."))

    _api = wandb.Api()
    _entity = "samuelstevens"
    _project = "beetle-traits"
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

    _pcts = [("25%", 0.25), ("50%", 0.50), ("75%", 0.75), ("100%", 1.0)]

    def _get_fixed_images_at_targets(run_id, max_step):
        """Download fixed val images at multiple training checkpoints in one scan.

        Returns {pct_label: {beetle_key: Image, ...}, ...} and {pct_label: actual_step}.
        """
        _run = _api.run(f"{_entity}/{_project}/{run_id}")
        _targets = {label: int(max_step * pct) for label, pct in _pcts}

        # Collect the best row (closest step) for each target.
        _best = {label: (float("inf"), None, 0) for label in _targets}
        for _row in _run.scan_history():
            _step = _row.get("_step", 0)
            _has_fixed = any(
                k.startswith("images/val/") and "/fixed/" in k
                for k in _row
                if isinstance(_row[k], dict)
            )
            if not _has_fixed:
                continue
            for _label, _tgt in _targets.items():
                _dist = abs(_step - _tgt)
                if _dist < _best[_label][0]:
                    _best[_label] = (_dist, dict(_row), _step)

        _tmpdir = tempfile.mkdtemp()
        _all_imgs = {}
        _actual_steps = {}
        for _label, (_dist, _row, _step) in _best.items():
            _actual_steps[_label] = _step
            _imgs = {}
            if _row is None:
                _all_imgs[_label] = _imgs
                continue
            for _k, _v in _row.items():
                if not _k.startswith("images/val/") or "/fixed/" not in _k:
                    continue
                if not isinstance(_v, dict) or _v.get("_type") != "image-file":
                    continue
                _path = _v["path"]
                _run.file(_path).download(root=_tmpdir, replace=True)
                _img = Image.open(os.path.join(_tmpdir, _path))
                _img.thumbnail((128, 128))
                _imgs[_k] = _img
            _all_imgs[_label] = _imgs
        return _all_imgs, _actual_steps

    _hm_id = _find_best_run_id(heatmap_df)
    _bl_id = _find_best_run_id(baseline_df)

    _runs = []
    if _hm_id:
        _runs.append(("Heatmap", _hm_id, _max_step(heatmap_df, _hm_id)))
    if _bl_id:
        _runs.append(("MLP baseline", _bl_id, _max_step(baseline_df, _bl_id)))

    mo.stop(not _runs, mo.md("No runs found."))

    # Build one figure per run: rows=beetles, cols=25%/50%/75%/100%.
    _figs = []
    for _label, _rid, _mstep in mo.status.progress_bar(
        _runs, title="Downloading images"
    ):
        _all_imgs, _actual_steps = _get_fixed_images_at_targets(_rid, _mstep)

        # Collect beetle keys across all time points.
        _beetle_keys = sorted({k for imgs in _all_imgs.values() for k in imgs})
        if not _beetle_keys:
            continue

        _n_rows = len(_beetle_keys)
        _n_cols = len(_pcts)
        _fig, _axes = plt.subplots(
            _n_rows,
            _n_cols,
            figsize=(2.5 * _n_cols, 2.5 * _n_rows),
            dpi=72,
            layout="constrained",
            squeeze=False,
        )

        # Column headers.
        for _c, (pct_label, _) in enumerate(_pcts):
            _step = _actual_steps.get(pct_label, 0)
            _axes[0, _c].set_title(f"{pct_label} (step {_step})", fontsize=8)

        for _r, _bkey in enumerate(_beetle_keys):
            for _c, (pct_label, _) in enumerate(_pcts):
                _ax = _axes[_r, _c]
                _ax.axis("off")
                _img = _all_imgs.get(pct_label, {}).get(_bkey)
                if _img is None:
                    continue
                _arr = np.asarray(_img)
                _ax.imshow(_arr)
                _cy, _cx = _arr.shape[0] / 2, _arr.shape[1] / 2
                _ax.plot(_cx, _cy, "+", color="blue", markersize=8, markeredgewidth=1.5)

        _fig.suptitle(
            f"{_label} (original images, predictions mapped back): green=GT, red=pred, blue+=center",
            fontsize=9,
        )
        _figs.append(_fig)

    mo.vstack([_f for _f in _figs])
    return


if __name__ == "__main__":
    app.run()
