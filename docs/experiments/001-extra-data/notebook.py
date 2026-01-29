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
def _(mo):
    mo.md("""
    # Extra Data Sweep

    Plot final validation loss and width/length validation errors vs learning rate, colored by BeetlePalooza usage.
    """)
    return


@app.cell
def _(np, pl, wandb):
    entity = "samuelstevens"
    project = "beetle-traits"
    tag = "exp-001"

    def _get_nested(dct: dict, path: str):
        value = dct
        for part in path.split("."):
            if not isinstance(value, dict) or part not in value:
                return None
            value = value[part]
        return value

    def _get_bp_go(cfg: dict) -> bool | None:
        value = _get_nested(cfg, "beetlepalooza.go")
        if value is not None:
            return bool(value)

        value = cfg.get("beetlepalooza/go")
        if value is not None:
            return bool(value)

        value = cfg.get("beetlepalooza")
        if isinstance(value, dict):
            value = value.get("go")
        if value is not None:
            return bool(value)

        return None

    def _get_lr(cfg: dict) -> float | None:
        value = cfg.get("learning_rate")
        if value is None:
            value = cfg.get("learning-rate")
        if value is None:
            value = _get_nested(cfg, "learning_rate")
        if value is None:
            return None
        return float(value)

    def _get_split_metrics(summary: dict, split: str) -> dict[str, float]:
        metrics = {}
        for key, value in summary.items():
            if not key.startswith(split + "/"):
                continue
            if isinstance(value, bool) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                continue
            if not np.isfinite(value):
                continue
            metrics[key.replace("/", "_")] = float(value)
        return metrics

    api = wandb.Api()
    runs = api.runs(path=f"{entity}/{project}", filters={"tags": {"$in": [tag]}})

    rows = []
    for run in runs:
        lr = _get_lr(run.config)
        step = run.summary.get("step") or run.summary.get("_step")
        runtime_s = run.summary.get("_runtime") or run.summary.get("runtime")

        train_metrics = _get_split_metrics(run.summary, "train")
        val_metrics = _get_split_metrics(run.summary, "val")

        rows.append({
            "run_id": run.id,
            "name": run.name,
            "state": run.state,
            "learning_rate": lr,
            "step": int(step) if step is not None else None,
            "runtime_s": float(runtime_s) if runtime_s is not None else None,
            "runtime_hr": float(runtime_s) / 3600.0 if runtime_s is not None else None,
            "beetlepalooza": _get_bp_go(run.config),
            **train_metrics,
            **val_metrics,
        })

    runs_df = pl.DataFrame(rows)
    return (runs_df,)


@app.cell
def _(pl, runs_df):
    runs_df.filter(pl.col("step") > 15_000)
    return


@app.cell
def _(np, pl, plt, runs_df):
    plot_df = runs_df.filter(
        pl.col("learning_rate").is_not_null()
        & pl.col("beetlepalooza").is_not_null()
        & pl.col("step").is_not_null()
        & (pl.col("step") > 10_000)
        # & ~pl.col("beetlepalooza")
    )

    fig, axes = plt.subplots(
        4, 2, figsize=(8, 12), dpi=150, sharex=True, layout="constrained", sharey="row"
    )
    axes = axes.reshape(-1)

    labels = {True: "Hawaii + BeetlePalooza", False: "Hawaii only"}
    markers = {True: "o", False: "s"}
    metrics = [
        ("train_loss", "Train Loss"),
        ("val_loss", "Val Loss"),
        ("val_length_line_err_cm", "Length Line Error (mm)"),
        ("val_width_line_err_cm", "Width Line Error (mm)"),
        ("val_max_length_line_err_cm", "Max Length Line Error (mm)"),
        ("val_max_width_line_err_cm", "Max Width Line Error (mm)"),
        ("val_median_length_line_err_cm", "Median Length Line Error (mm)"),
        ("val_median_width_line_err_cm", "Median Width Line Error (mm)"),
    ]

    cmap = plt.get_cmap("plasma")
    step_values = plot_df.get_column("step").drop_nulls()
    step_min = float(step_values.min()) if len(step_values) else 0.0
    step_max = float(step_values.max()) if len(step_values) else 1.0
    norm = plt.Normalize(vmin=step_min, vmax=step_max)

    for ax, (metric, title) in zip(axes, metrics, strict=True):
        metric_df = plot_df.filter(pl.col(metric).is_not_null() & (pl.col(metric) > 0))
        for flag in [False, True]:
            subset = metric_df.filter(pl.col("beetlepalooza") == flag)
            steps = np.array(subset["step"].to_list(), dtype=float)
            scalar = 10 if "cm" in metric else 1
            ax.scatter(
                subset["learning_rate"].to_list(),
                subset[metric].to_numpy() * scalar,
                c=steps,
                cmap=cmap,
                norm=norm,
                marker=markers[flag],
                label=labels[flag],
                s=60,
                alpha=0.5,
            )
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("Learning rate")
        ax.grid(True, alpha=0.3)
        ax.spines[["right", "top"]].set_visible(False)

    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].legend()
    axes[-1].legend()

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label="Step", aspect=64.0)

    fig.savefig("docs/experiments/001-extra-data/fig1.png")
    fig
    return


if __name__ == "__main__":
    app.run()
