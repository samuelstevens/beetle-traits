import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl

    import wandb
    return mo, np, pl, plt, wandb


@app.cell
def _(mo, np, pl, wandb):
    entity = "samuelstevens"
    project = "beetle-traits"
    tag = "exp-002-augmentation"

    def get_condition(tags: list[str]) -> str | None:
        for cond in ("baseline", "norm-only", "full-aug"):
            if cond in tags:
                return cond
        return None

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
    runs = list(api.runs(path=f"{entity}/{project}", filters={"tags": {"$in": [tag]}}))

    history_rows = []
    static_cols = {"run_id", "name", "state", "condition"}
    for run in mo.status.progress_bar(runs):
        tags = list(run.tags)
        cfg_flat = flatten_cfg(dict(run.config))
        run_meta = {
            "run_id": run.id,
            "name": run.name,
            "state": run.state,
            "condition": get_condition(tags),
            **cfg_flat,
        }
        static_cols.update(cfg_flat.keys())

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

            history_rows.append(row)

    history_df = pl.DataFrame(history_rows)

    if history_df.is_empty():
        steps_df = history_df
    else:
        id_cols = ["run_id", "step"]
        meta_cols = [
            col
            for col in sorted(static_cols)
            if col in history_df.columns and col not in id_cols
        ]
        metric_cols = [
            col for col in history_df.columns if col not in set(id_cols + meta_cols)
        ]
        steps_df = (
            history_df
            .group_by(id_cols, maintain_order=True)
            .agg(
                [pl.col(col).first().alias(col) for col in meta_cols]
                + [pl.col(col).max().alias(col) for col in metric_cols]
            )
            .sort([
                col
                for col in ["condition", "seed", "step"]
                if col in history_df.columns
            ])
        )
    return (steps_df,)


@app.cell
def _(steps_df):
    if steps_df.is_empty():
        preview_df = steps_df
    else:
        lr_col = "learning_rate" if "learning_rate" in steps_df.columns else None
        seed_col = "seed" if "seed" in steps_df.columns else None
        cols = ["run_id", "name", "condition"]
        if seed_col is not None:
            cols.append(seed_col)
        if lr_col is not None:
            cols.append(lr_col)
        cols.extend(["step", "train_loss", "val_loss"])
        preview_df = steps_df.select([
            col for col in cols if col in steps_df.columns
        ]).sort([
            col for col in ["condition", "seed", "step"] if col in steps_df.columns
        ])

    preview_df
    return


@app.cell
def _(mo, pl, steps_df):
    mo.stop(steps_df.is_empty(), mo.md("No rows in `steps_df`."))

    _val_cols = [
        col for col in ["val_loss", "val_length_line_err_cm"] if col in steps_df.columns
    ]
    mo.stop(not _val_cols, mo.md("No validation metric columns found in `steps_df`."))

    _latest_val_df = (
        steps_df
        .filter(pl.col(_val_cols[0]).is_not_null())
        .sort(["run_id", "step"])
        .group_by("run_id")
        .agg(
            pl.col("name").last().alias("name"),
            pl.col("condition").last().alias("condition"),
            pl.col("learning_rate").last().alias("learning_rate"),
            pl.col("step").last().alias("latest_val_step"),
            *[pl.col(col).last().alias(col) for col in _val_cols],
        )
        .sort(["learning_rate", "condition", "name"])
    )

    _latest_by_lr_df = (
        _latest_val_df
        .group_by("learning_rate")
        .agg(
            pl.len().alias("n_runs"),
            *[pl.col(col).mean().alias(f"mean_{col}") for col in _val_cols],
            *[pl.col(col).std().alias(f"std_{col}") for col in _val_cols],
        )
        .sort("learning_rate")
    )

    if "_runtime" not in steps_df.columns:
        _speed_by_run_df = pl.DataFrame()
        _speed_by_cond_df = pl.DataFrame()
        _speed_by_cond_workers_df = pl.DataFrame()
    else:
        _speed_pts_df = (
            steps_df
            .filter(pl.col("_runtime").is_not_null())
            .sort(["run_id", "step"])
            .with_columns(
                (pl.col("step") - pl.col("step").shift(1).over("run_id")).alias(
                    "dstep"
                ),
                (pl.col("_runtime") - pl.col("_runtime").shift(1).over("run_id")).alias(
                    "druntime"
                ),
            )
            .filter((pl.col("dstep") > 0) & (pl.col("druntime") > 0))
            .with_columns((pl.col("dstep") / pl.col("druntime")).alias("steps_per_sec"))
        )

        _speed_by_run_df = (
            _speed_pts_df
            .group_by("run_id")
            .agg(
                pl.col("name").last().alias("name"),
                pl.col("condition").last().alias("condition"),
                pl.col("n_workers").last().alias("n_workers"),
                pl.col("learning_rate").last().alias("learning_rate"),
                pl.col("steps_per_sec").mean().alias("mean_steps_per_sec"),
                pl.col("steps_per_sec").std().alias("std_steps_per_sec_within_run"),
                pl.len().alias("n_delta_points"),
            )
            .sort(["condition", "n_workers", "learning_rate", "name"])
        )
        _speed_by_cond_df = (
            _speed_by_run_df
            .group_by("condition")
            .agg(
                pl.len().alias("n_runs"),
                pl.col("mean_steps_per_sec").mean().alias("mean_steps_per_sec"),
                pl
                .col("mean_steps_per_sec")
                .std()
                .alias("std_steps_per_sec_across_runs"),
            )
            .sort("condition")
        )
        _speed_by_cond_workers_df = (
            _speed_by_run_df
            .group_by(["condition", "n_workers"])
            .agg(
                pl.len().alias("n_runs"),
                pl.col("mean_steps_per_sec").mean().alias("mean_steps_per_sec"),
                pl
                .col("mean_steps_per_sec")
                .std()
                .alias("std_steps_per_sec_across_runs"),
            )
            .sort(["condition", "n_workers"])
        )

    mo.vstack([
        mo.md("### Latest validation metrics per run"),
        _latest_val_df,
        mo.md("### Latest validation metrics grouped by learning rate"),
        _latest_by_lr_df,
        mo.md("### Throughput per run (`steps/sec`)"),
        _speed_by_run_df,
        mo.md("### Throughput grouped by augmentation group"),
        _speed_by_cond_df,
        mo.md("### Throughput grouped by augmentation group and n_workers"),
        _speed_by_cond_workers_df,
    ])
    return


@app.cell
def _(np, plt, steps_df):
    fig, (ax_train, ax_val, ax_gap) = plt.subplots(
        3,
        1,
        figsize=(10, 10),
        dpi=140,
        sharex=True,
        layout="constrained",
    )
    ax_val.sharey(ax_train)
    aug_style_map = {
        "baseline": "-",
        "norm-only": "--",
        "full-aug": ":",
    }
    lr_vals = (
        steps_df
        .select("learning_rate")
        .drop_nulls()
        .unique()
        .sort("learning_rate")
        .get_column("learning_rate")
        .to_list()
        if "learning_rate" in steps_df.columns
        else []
    )
    lr_vals = [float(lr) for lr in lr_vals if float(lr) > 0]
    lr_cmap = plt.get_cmap("viridis")
    if lr_vals:
        lr_log_vals = np.log10(np.array(lr_vals, dtype=float))
        lr_log_min = float(np.min(lr_log_vals))
        lr_log_max = float(np.max(lr_log_vals))
        if lr_log_max <= lr_log_min:
            lr_log_max = lr_log_min + 1.0
        lr_norm = plt.Normalize(vmin=lr_log_min, vmax=lr_log_max)
    else:
        lr_norm = None

    def _get_lr_color(lr: float | None):
        if lr is None or lr_norm is None:
            return "#7f7f7f"
        lr_float = float(lr)
        if lr_float <= 0:
            return "#7f7f7f"
        return lr_cmap(lr_norm(np.log10(lr_float)))

    ema_beta = 0.9

    def _ema_np(y: np.ndarray, *, beta: float) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if y.size == 0:
            return y
        y_ema = np.empty_like(y)
        y_ema[0] = y[0]
        for i in range(1, y.size):
            y_ema[i] = beta * y_ema[i - 1] + (1.0 - beta) * y[i]
        return y_ema

    def _plot_loss_metric(ax, metric: str, title: str) -> bool:
        if metric not in steps_df.columns:
            ax.set_title(f"{title} (no {metric} column)")
            ax.set_ylabel("Loss")
            ax.spines[["right", "top"]].set_visible(False)
            return False

        metric_df = steps_df.filter(steps_df[metric].is_not_null())
        if metric_df.is_empty():
            ax.set_title(f"{title} (no data)")
            ax.set_ylabel("Loss")
            ax.spines[["right", "top"]].set_visible(False)
            return False

        run_ids = metric_df.get_column("run_id").unique().to_list()
        for run_id in run_ids:
            run_df = metric_df.filter(metric_df["run_id"] == run_id).sort("step")
            first_row = run_df.head(1).to_dicts()[0]
            condition = first_row.get("condition")
            lr = first_row.get("learning_rate")
            color = _get_lr_color(lr)
            style = aug_style_map.get(condition, "-.")
            y_vals = run_df.get_column(metric).to_numpy()
            if metric.startswith("train_"):
                y_vals = _ema_np(y_vals, beta=ema_beta)

            ax.plot(
                run_df.get_column("step").to_numpy(),
                y_vals,
                color=color,
                linestyle=style,
                linewidth=1.8,
                alpha=0.9,
            )

        ax.set_title(title)
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.spines[["right", "top"]].set_visible(False)
        return True

    def _plot_loss_gap(
        ax,
        *,
        train_metric: str,
        val_metric: str,
        title: str,
    ) -> bool:
        if train_metric not in steps_df.columns or val_metric not in steps_df.columns:
            ax.set_title(f"{title} (missing {train_metric} or {val_metric})")
            ax.set_ylabel("Val - Train")
            ax.spines[["right", "top"]].set_visible(False)
            return False

        gap_df = steps_df.filter(
            steps_df[train_metric].is_not_null() & steps_df[val_metric].is_not_null()
        )
        if gap_df.is_empty():
            ax.set_title(f"{title} (no paired data)")
            ax.set_ylabel("Val - Train")
            ax.spines[["right", "top"]].set_visible(False)
            return False

        run_ids = gap_df.get_column("run_id").unique().to_list()
        for run_id in run_ids:
            run_df = gap_df.filter(gap_df["run_id"] == run_id).sort("step")
            first_row = run_df.head(1).to_dicts()[0]
            condition = first_row.get("condition")
            lr = first_row.get("learning_rate")
            color = _get_lr_color(lr)
            style = aug_style_map.get(condition, "-.")
            gap_vals = (
                run_df.get_column(val_metric).to_numpy()
                - run_df.get_column(train_metric).to_numpy()
            )
            gap_vals = _ema_np(gap_vals, beta=ema_beta)

            ax.plot(
                run_df.get_column("step").to_numpy(),
                gap_vals,
                color=color,
                linestyle=style,
                linewidth=1.8,
                alpha=0.9,
            )

        ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle=":")
        ax.text(
            0.01,
            0.96,
            "(val > train)",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            color="#444444",
        )
        ax.text(
            0.01,
            0.04,
            "(val < train)",
            transform=ax.transAxes,
            va="bottom",
            fontsize=8,
            color="#444444",
        )
        ax.set_title(title)
        ax.set_ylabel("Val - Train")
        ax.set_yscale("symlog", linthresh=1e0)
        ax.grid(alpha=0.25)
        ax.spines[["right", "top"]].set_visible(False)
        return True

    train_ok = _plot_loss_metric(ax_train, "train_loss", "Train loss vs step")
    val_ok = _plot_loss_metric(ax_val, "val_loss", "Val loss vs step")
    gap_ok = _plot_loss_gap(
        ax_gap,
        train_metric="train_loss",
        val_metric="val_loss",
        title="Generalization gap (val - train) vs step",
    )
    ax_train.text(
        0.99,
        0.03,
        f"EMA beta={ema_beta:g} (train/gap only)",
        transform=ax_train.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    ax_gap.set_xlabel("Step")

    if train_ok or val_ok or gap_ok:
        cond_handles = [
            ax_train.plot(
                [], [], color="#444444", linestyle=style, linewidth=2.0, label=cond
            )[0]
            for cond, style in aug_style_map.items()
        ]
        cond_legend = ax_train.legend(
            handles=cond_handles,
            title="Aug group",
            loc="upper right",
            fontsize=7,
            frameon=False,
        )
        ax_train.add_artist(cond_legend)
        if lr_norm is not None:
            lr_sm = plt.cm.ScalarMappable(norm=lr_norm, cmap=lr_cmap)
            lr_sm.set_array([])
            lr_cbar = fig.colorbar(lr_sm, ax=[ax_train, ax_val, ax_gap], pad=0.01)
            lr_tick_vals = np.array(lr_vals, dtype=float)
            lr_cbar.set_ticks(np.log10(lr_tick_vals))
            lr_cbar.set_ticklabels([f"{lr:g}" for lr in lr_tick_vals])
            lr_cbar.set_label("Learning rate")

    fig
    return


@app.cell
def _(np, pl, plt, steps_df):
    _lr_fig, _lr_ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(7.5, 4.5),
        dpi=140,
        layout="constrained",
    )
    _lr_marker_map = {
        "baseline": "o",
        "norm-only": "s",
        "full-aug": "^",
    }

    _lr_required_cols = {"run_id", "learning_rate", "step", "val_loss"}
    if not _lr_required_cols.issubset(steps_df.columns):
        _lr_ax.set_title("LR vs final val loss (missing required columns)")
        _lr_ax.set_xlabel("Learning rate")
        _lr_ax.set_ylabel("Final val loss")
        _lr_ax.spines[["right", "top"]].set_visible(False)
    else:
        _lr_plot_df = (
            steps_df
            .filter(
                pl.col("learning_rate").is_not_null()
                & pl.col("step").is_not_null()
                & pl.col("val_loss").is_not_null()
                & (pl.col("val_loss") > 0)
            )
            .sort(["run_id", "step"])
            .group_by("run_id")
            .agg(
                pl.col("name").last().alias("name"),
                pl.col("condition").last().alias("condition"),
                pl.col("learning_rate").last().alias("learning_rate"),
                pl.col("step").last().alias("final_step"),
                pl.col("val_loss").last().alias("final_val_loss"),
            )
            .sort(["learning_rate", "condition", "name"])
        )

        if _lr_plot_df.is_empty():
            _lr_ax.set_title("LR vs final val loss (no data)")
            _lr_ax.set_xlabel("Learning rate")
            _lr_ax.set_ylabel("Final val loss")
            _lr_ax.spines[["right", "top"]].set_visible(False)
        else:
            _lr_steps = _lr_plot_df.get_column("final_step").to_numpy()
            _lr_step_min = float(np.min(_lr_steps))
            _lr_step_max = float(np.max(_lr_steps))
            if _lr_step_max <= _lr_step_min:
                _lr_step_max = _lr_step_min + 1.0
            _lr_norm = plt.Normalize(vmin=_lr_step_min, vmax=_lr_step_max)
            _lr_cmap = plt.get_cmap("plasma")

            _lr_conditions = (
                _lr_plot_df
                .select("condition")
                .unique()
                .sort("condition")
                .get_column("condition")
                .to_list()
            )
            for _lr_condition in _lr_conditions:
                _lr_cond_df = _lr_plot_df.filter(
                    pl.col("condition") == _lr_condition
                ).sort("learning_rate")
                _lr_marker = _lr_marker_map.get(_lr_condition, "D")

                _lr_ax.scatter(
                    _lr_cond_df.get_column("learning_rate").to_numpy(),
                    _lr_cond_df.get_column("final_val_loss").to_numpy(),
                    c=_lr_cond_df.get_column("final_step").to_numpy(),
                    cmap=_lr_cmap,
                    norm=_lr_norm,
                    marker=_lr_marker,
                    alpha=0.5,
                    label=_lr_condition,
                )

            _lr_ax.set_title("LR vs final val loss (color=final step)")
            _lr_ax.set_xlabel("Learning rate")
            _lr_ax.set_ylabel("Final val loss")
            _lr_ax.set_xscale("log")
            _lr_ax.set_yscale("log")
            _lr_ax.grid(alpha=0.25)
            _lr_ax.spines[["right", "top"]].set_visible(False)
            _lr_ax.legend(title="Aug group", fontsize=7, frameon=False)

            _lr_sm = plt.cm.ScalarMappable(norm=_lr_norm, cmap=_lr_cmap)
            _lr_sm.set_array([])
            _lr_fig.colorbar(_lr_sm, ax=_lr_ax, label="Final step")

    _lr_fig
    return


@app.cell
def _(np, pl, plt, steps_df):
    _speed_fig, _speed_ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(10, 4.5),
        dpi=140,
        layout="constrained",
    )
    _speed_color_map = {
        "baseline": "#1f77b4",
        "norm-only": "#ff7f0e",
        "full-aug": "#2ca02c",
    }

    if "_runtime" not in steps_df.columns:
        _speed_ax.set_title("Steps/sec vs step (missing _runtime)")
        _speed_ax.set_xlabel("Step")
        _speed_ax.set_ylabel("steps/sec")
        _speed_ax.spines[["right", "top"]].set_visible(False)
    else:
        _speed_df = steps_df.filter(steps_df["_runtime"].is_not_null()).sort([
            "run_id",
            "step",
        ])
        if _speed_df.is_empty():
            _speed_ax.set_title("Steps/sec vs step (no runtime data)")
            _speed_ax.set_xlabel("Step")
            _speed_ax.set_ylabel("steps/sec")
            _speed_ax.spines[["right", "top"]].set_visible(False)
        else:
            _speed_rows: list[dict[str, object]] = []
            _speed_run_ids = _speed_df.get_column("run_id").unique().to_list()
            for _speed_run_id in _speed_run_ids:
                _speed_run_df = _speed_df.filter(
                    _speed_df["run_id"] == _speed_run_id
                ).sort("step")
                _speed_first_row = _speed_run_df.head(1).to_dicts()[0]
                _speed_condition = _speed_first_row.get("condition")
                if _speed_condition is None:
                    _speed_condition = "unknown"
                _speed_workers = _speed_first_row.get("n_workers")
                if _speed_workers is None:
                    _speed_workers = -1

                _speed_step_np = _speed_run_df.get_column("step").to_numpy()
                _speed_runtime_np = _speed_run_df.get_column("_runtime").to_numpy()
                _speed_dstep = np.diff(_speed_step_np)
                _speed_druntime = np.diff(_speed_runtime_np)
                _speed_keep = (_speed_dstep > 0) & (_speed_druntime > 0)
                if not np.any(_speed_keep):
                    continue
                _speed_x = _speed_step_np[1:][_speed_keep]
                _speed_y = _speed_dstep[_speed_keep] / _speed_druntime[_speed_keep]

                for _speed_step, _speed_val in zip(_speed_x, _speed_y, strict=False):
                    _speed_rows.append({
                        "condition": _speed_condition,
                        "n_workers": int(_speed_workers),
                        "step": int(_speed_step),
                        "steps_per_sec": float(_speed_val),
                    })

            if not _speed_rows:
                _speed_ax.set_title(
                    "Training throughput (steps/sec) vs step (no valid deltas)"
                )
                _speed_ax.set_xlabel("Step")
                _speed_ax.set_ylabel("steps/sec")
                _speed_ax.spines[["right", "top"]].set_visible(False)
            else:
                _speed_points_df = pl.DataFrame(_speed_rows)
                _speed_agg_df = (
                    _speed_points_df
                    .group_by(["condition", "n_workers", "step"], maintain_order=True)
                    .agg(
                        pl.col("steps_per_sec").mean().alias("steps_per_sec_mean"),
                        pl.col("steps_per_sec").std().alias("steps_per_sec_std"),
                    )
                    .sort(["condition", "n_workers", "step"])
                )
                _speed_workers_vals = (
                    _speed_agg_df
                    .select("n_workers")
                    .drop_nulls()
                    .unique()
                    .sort("n_workers")
                    .get_column("n_workers")
                    .to_list()
                )
                _speed_style_cycle = ["-", "--", ":", "-."]
                _speed_workers_style_map = {
                    int(_n_workers): _speed_style_cycle[i % len(_speed_style_cycle)]
                    for i, _n_workers in enumerate(_speed_workers_vals)
                }
                _speed_groups = (
                    _speed_agg_df
                    .select(["condition", "n_workers"])
                    .unique()
                    .sort(["condition", "n_workers"])
                    .to_dicts()
                )
                for _speed_group in _speed_groups:
                    _speed_cond = _speed_group["condition"]
                    _speed_workers = int(_speed_group["n_workers"])
                    _speed_cond_df = _speed_agg_df.filter(
                        (pl.col("condition") == _speed_cond)
                        & (pl.col("n_workers") == _speed_workers)
                    ).sort("step")
                    _speed_x = _speed_cond_df.get_column("step").to_numpy()
                    _speed_mean = _speed_cond_df.get_column(
                        "steps_per_sec_mean"
                    ).to_numpy()
                    _speed_std = _speed_cond_df.get_column(
                        "steps_per_sec_std"
                    ).to_numpy()
                    _speed_color = _speed_color_map.get(_speed_cond, "#7f7f7f")
                    _speed_style = _speed_workers_style_map.get(_speed_workers, "-")

                    _speed_ax.plot(
                        _speed_x,
                        _speed_mean,
                        color=_speed_color,
                        linestyle=_speed_style,
                        linewidth=2.0,
                        alpha=0.95,
                        label=f"{_speed_cond} (n_workers={_speed_workers})",
                    )

                    _speed_std_ok = np.isfinite(_speed_std) & (_speed_std > 0)
                    if np.any(_speed_std_ok):
                        _speed_ax.fill_between(
                            _speed_x[_speed_std_ok],
                            (_speed_mean - _speed_std)[_speed_std_ok],
                            (_speed_mean + _speed_std)[_speed_std_ok],
                            color=_speed_color,
                            alpha=0.18,
                            linewidth=0,
                        )

                _speed_ax.set_title(
                    "Training throughput (mean steps/sec +/- std) vs step"
                )
                _speed_ax.set_xlabel("Step")
                _speed_ax.set_ylabel("steps/sec")
                _speed_ax.grid(alpha=0.25)
                _speed_ax.spines[["right", "top"]].set_visible(False)
                _speed_cond_handles = [
                    _speed_ax.plot(
                        [], [], color=color, linestyle="-", linewidth=2.0, label=cond
                    )[0]
                    for cond, color in _speed_color_map.items()
                ]
                _speed_workers_handles = [
                    _speed_ax.plot(
                        [],
                        [],
                        color="#444444",
                        linestyle=_speed_workers_style_map[int(_n_workers)],
                        linewidth=2.0,
                        label=f"n_workers={int(_n_workers)}",
                    )[0]
                    for _n_workers in _speed_workers_vals
                ]
                _speed_cond_legend = _speed_ax.legend(
                    handles=_speed_cond_handles,
                    title="Aug group",
                    loc="upper right",
                    fontsize=7,
                    frameon=False,
                )
                _speed_ax.add_artist(_speed_cond_legend)
                if _speed_workers_handles:
                    _speed_ax.legend(
                        handles=_speed_workers_handles,
                        title="n_workers",
                        loc="upper center",
                        fontsize=7,
                        frameon=False,
                    )

    _speed_fig
    return


@app.cell
def _(np, plt, steps_df):
    _cm_fig, (_cm_ax_train, _cm_ax_val, _cm_ax_gap) = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(10, 9.5),
        dpi=140,
        layout="constrained",
        sharex=True,
    )
    _cm_ax_val.sharey(_cm_ax_train)
    _cm_aug_style_map = {
        "baseline": "-",
        "norm-only": "--",
        "full-aug": ":",
    }
    _cm_lr_vals = (
        steps_df
        .select("learning_rate")
        .drop_nulls()
        .unique()
        .sort("learning_rate")
        .get_column("learning_rate")
        .to_list()
        if "learning_rate" in steps_df.columns
        else []
    )
    _cm_lr_vals = [float(_cm_lr) for _cm_lr in _cm_lr_vals if float(_cm_lr) > 0]
    _cm_lr_cmap = plt.get_cmap("viridis")
    if _cm_lr_vals:
        _cm_lr_log_vals = np.log10(np.array(_cm_lr_vals, dtype=float))
        _cm_lr_log_min = float(np.min(_cm_lr_log_vals))
        _cm_lr_log_max = float(np.max(_cm_lr_log_vals))
        if _cm_lr_log_max <= _cm_lr_log_min:
            _cm_lr_log_max = _cm_lr_log_min + 1.0
        _cm_lr_norm = plt.Normalize(vmin=_cm_lr_log_min, vmax=_cm_lr_log_max)
    else:
        _cm_lr_norm = None

    def _get_cm_lr_color(_cm_lr: float | None):
        if _cm_lr is None or _cm_lr_norm is None:
            return "#7f7f7f"
        _cm_lr_float = float(_cm_lr)
        if _cm_lr_float <= 0:
            return "#7f7f7f"
        return _cm_lr_cmap(_cm_lr_norm(np.log10(_cm_lr_float)))

    _cm_ema_beta = 0.9

    def _ema_cm_np(_cm_y: np.ndarray, *, _cm_beta: float) -> np.ndarray:
        _cm_y = np.asarray(_cm_y, dtype=float)
        if _cm_y.size == 0:
            return _cm_y
        _cm_y_ema = np.empty_like(_cm_y)
        _cm_y_ema[0] = _cm_y[0]
        for _cm_i in range(1, _cm_y.size):
            _cm_y_ema[_cm_i] = (
                _cm_beta * _cm_y_ema[_cm_i - 1] + (1.0 - _cm_beta) * _cm_y[_cm_i]
            )
        return _cm_y_ema

    def _plot_cm_metric(_cm_ax, _cm_metric: str, _cm_title: str) -> bool:
        if _cm_metric not in steps_df.columns:
            _cm_cols = [col for col in steps_df.columns if col.endswith("_cm")]
            _cm_ax.set_title(f"{_cm_metric} not found")
            _cm_ax.set_ylabel("cm")
            _cm_ax.text(
                0.01,
                0.95,
                f"available cm columns: {_cm_cols}",
                transform=_cm_ax.transAxes,
                va="top",
                fontsize=8,
            )
            _cm_ax.spines[["right", "top"]].set_visible(False)
            return False

        _cm_metric_df = steps_df.filter(
            steps_df[_cm_metric].is_not_null() & (steps_df[_cm_metric] > 0)
        )
        if _cm_metric_df.is_empty():
            _cm_ax.set_title(f"{_cm_title} (no data)")
            _cm_ax.set_ylabel("cm")
            _cm_ax.spines[["right", "top"]].set_visible(False)
            return False

        _cm_run_ids = _cm_metric_df.get_column("run_id").unique().to_list()
        for _cm_run_id in _cm_run_ids:
            _cm_run_df = _cm_metric_df.filter(
                _cm_metric_df["run_id"] == _cm_run_id
            ).sort("step")
            _cm_first_row = _cm_run_df.head(1).to_dicts()[0]
            _cm_condition = _cm_first_row.get("condition")
            _cm_lr = _cm_first_row.get("learning_rate")
            _cm_color = _get_cm_lr_color(_cm_lr)
            _cm_style = _cm_aug_style_map.get(_cm_condition, "-.")
            _cm_y_vals = _cm_run_df.get_column(_cm_metric).to_numpy()
            if _cm_metric.startswith("train_"):
                _cm_y_vals = _ema_cm_np(_cm_y_vals, _cm_beta=_cm_ema_beta)
            _cm_ax.plot(
                _cm_run_df.get_column("step").to_numpy(),
                _cm_y_vals,
                color=_cm_color,
                linestyle=_cm_style,
                linewidth=1.8,
                alpha=0.9,
            )

        _cm_ax.set_title(_cm_title)
        _cm_ax.set_ylabel("cm")
        _cm_ax.set_yscale("log")
        _cm_ax.grid(alpha=0.25)
        _cm_ax.spines[["right", "top"]].set_visible(False)
        return True

    _cm_train_ok = _plot_cm_metric(
        _cm_ax_train, "train_length_line_err_cm", "Train length line err (cm) vs step"
    )
    _cm_val_ok = _plot_cm_metric(
        _cm_ax_val, "val_length_line_err_cm", "Val length line err (cm) vs step"
    )

    def _plot_cm_gap(
        _cm_gap_ax,
        *,
        _cm_train_metric: str,
        _cm_val_metric: str,
        _cm_title: str,
    ) -> bool:
        if (
            _cm_train_metric not in steps_df.columns
            or _cm_val_metric not in steps_df.columns
        ):
            _cm_gap_ax.set_title(
                f"{_cm_title} (missing {_cm_train_metric} or {_cm_val_metric})"
            )
            _cm_gap_ax.set_ylabel("val - train (cm)")
            _cm_gap_ax.spines[["right", "top"]].set_visible(False)
            return False

        _cm_gap_df = steps_df.filter(
            steps_df[_cm_train_metric].is_not_null()
            & steps_df[_cm_val_metric].is_not_null()
        )
        if _cm_gap_df.is_empty():
            _cm_gap_ax.set_title(f"{_cm_title} (no paired data)")
            _cm_gap_ax.set_ylabel("val - train (cm)")
            _cm_gap_ax.spines[["right", "top"]].set_visible(False)
            return False

        _cm_gap_run_ids = _cm_gap_df.get_column("run_id").unique().to_list()
        for _cm_gap_run_id in _cm_gap_run_ids:
            _cm_gap_run_df = _cm_gap_df.filter(
                _cm_gap_df["run_id"] == _cm_gap_run_id
            ).sort("step")
            _cm_gap_first_row = _cm_gap_run_df.head(1).to_dicts()[0]
            _cm_gap_condition = _cm_gap_first_row.get("condition")
            _cm_gap_lr = _cm_gap_first_row.get("learning_rate")
            _cm_gap_color = _get_cm_lr_color(_cm_gap_lr)
            _cm_gap_style = _cm_aug_style_map.get(_cm_gap_condition, "-.")
            _cm_gap_vals = (
                _cm_gap_run_df.get_column(_cm_val_metric).to_numpy()
                - _cm_gap_run_df.get_column(_cm_train_metric).to_numpy()
            )
            _cm_gap_vals = _ema_cm_np(_cm_gap_vals, _cm_beta=_cm_ema_beta)

            _cm_gap_ax.plot(
                _cm_gap_run_df.get_column("step").to_numpy(),
                _cm_gap_vals,
                color=_cm_gap_color,
                linestyle=_cm_gap_style,
                linewidth=1.8,
                alpha=0.9,
            )

        _cm_gap_ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle=":")
        _cm_gap_ax.text(
            0.01,
            0.96,
            "(val > train)",
            transform=_cm_gap_ax.transAxes,
            va="top",
            fontsize=8,
            color="#444444",
        )
        _cm_gap_ax.text(
            0.01,
            0.04,
            "(val < train)",
            transform=_cm_gap_ax.transAxes,
            va="bottom",
            fontsize=8,
            color="#444444",
        )
        _cm_gap_ax.set_title(_cm_title)
        _cm_gap_ax.set_ylabel("val - train (cm)")
        _cm_gap_ax.set_yscale("symlog", linthresh=1e-2)
        _cm_gap_ax.grid(alpha=0.25)
        _cm_gap_ax.spines[["right", "top"]].set_visible(False)
        return True

    _cm_gap_ok = _plot_cm_gap(
        _cm_ax_gap,
        _cm_train_metric="train_length_line_err_cm",
        _cm_val_metric="val_length_line_err_cm",
        _cm_title="Generalization gap (val - train) (cm) vs step",
    )
    _cm_ax_train.text(
        0.99,
        0.03,
        f"EMA beta={_cm_ema_beta:g} (train/gap only)",
        transform=_cm_ax_train.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    _cm_ax_gap.set_xlabel("Step")

    if _cm_train_ok or _cm_val_ok or _cm_gap_ok:
        _cm_cond_handles = [
            _cm_ax_train.plot(
                [], [], color="#444444", linestyle=style, linewidth=2.0, label=cond
            )[0]
            for cond, style in _cm_aug_style_map.items()
        ]
        _cm_cond_legend = _cm_ax_train.legend(
            handles=_cm_cond_handles,
            title="Aug group",
            loc="upper right",
            fontsize=7,
            frameon=False,
        )
        _cm_ax_train.add_artist(_cm_cond_legend)
        if _cm_lr_norm is not None:
            _cm_lr_sm = plt.cm.ScalarMappable(norm=_cm_lr_norm, cmap=_cm_lr_cmap)
            _cm_lr_sm.set_array([])
            _cm_lr_cbar = _cm_fig.colorbar(
                _cm_lr_sm, ax=[_cm_ax_train, _cm_ax_val, _cm_ax_gap], pad=0.01
            )
            _cm_lr_tick_vals = np.array(_cm_lr_vals, dtype=float)
            _cm_lr_cbar.set_ticks(np.log10(_cm_lr_tick_vals))
            _cm_lr_cbar.set_ticklabels([f"{_cm_lr:g}" for _cm_lr in _cm_lr_tick_vals])
            _cm_lr_cbar.set_label("Learning rate")

    _cm_fig
    return


if __name__ == "__main__":
    app.run()
