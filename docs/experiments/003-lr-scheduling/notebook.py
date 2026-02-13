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
    return mo, pl, plt, wandb


@app.cell
def _(mo):
    mo.md("""
    # LR Scheduling Sweep

    Training curves for each experiment: train loss, train error, BioRepo/Hawaii length and width errors over training steps.
    """)
    return


@app.cell
def _(pl, wandb):
    entity = "cain-429-the-ohio-state-university"
    project = "beetle-traits"
    tag = "exp-003-lr-scheduling"

    history_keys = [
        "train/loss",
        "train/line_err_cm",
        "val/biorepo/length_line_err_cm",
        "val/hawaii/length_line_err_cm",
        "val/biorepo/width_line_err_cm",
        "val/hawaii/width_line_err_cm",
    ]

    api = wandb.Api()
    runs = api.runs(path=f"{entity}/{project}", filters={"tags": {"$in": [tag]}})

    rows = []
    for _run in runs:
        _lr = _run.config.get("learning_rate") or _run.config.get("learning-rate")
        _schedule = _run.config.get("schedule")
        _label = f"{_schedule} lr={_lr}"

        for _row in _run.scan_history():
            _step = _row.get("_step")
            if _step is None:
                continue
            rows.append({
                "run_id": _run.id,
                "label": _label,
                "learning_rate": float(_lr) if _lr else None,
                "schedule": str(_schedule) if _schedule else None,
                "step": int(_step),
                **{k.replace("/", "_"): _row.get(k) for k in history_keys},
            })

    history_df = pl.DataFrame(rows, schema={
        "run_id": pl.Utf8,
        "label": pl.Utf8,
        "learning_rate": pl.Float64,
        "schedule": pl.Utf8,
        "step": pl.Int64,
        **{k.replace("/", "_"): pl.Float64 for k in history_keys},
    })
    history_df
    return (history_df,)


@app.cell
def _(history_df, pl, plt):
    metrics = [
        ("train_loss", "Train Loss"),
        ("train_line_err_cm", "Train Line Error (mm)"),
        ("val_biorepo_length_line_err_cm", "BioRepo Length Error (mm)"),
        ("val_hawaii_length_line_err_cm", "Hawaii Length Error (mm)"),
        ("val_biorepo_width_line_err_cm", "BioRepo Width Error (mm)"),
        ("val_hawaii_width_line_err_cm", "Hawaii Width Error (mm)"),
    ]

    fig, axes = plt.subplots(
        3, 2, figsize=(12, 10), dpi=150, sharex=True, layout="constrained"
    )
    axes = axes.reshape(-1)

    linestyles = {"cosine": "-", "wsd": "--"}
    lr_values = sorted(
        history_df.select("run_id", "learning_rate")
        .unique("run_id")
        .get_column("learning_rate")
        .drop_nulls()
        .to_list()
    )
    cmap = plt.get_cmap("tab10")
    lr_to_color = {
        lr: cmap(i / max(len(lr_values) - 1, 1)) for i, lr in enumerate(lr_values)
    }

    run_ids = history_df.get_column("run_id").unique().to_list()

    for _ax, (_metric, _title) in zip(axes, metrics, strict=True):
        for _run_id in run_ids:
            _run_data = history_df.filter(
                (pl.col("run_id") == _run_id) & pl.col(_metric).is_not_null()
            ).sort("step")
            if len(_run_data) == 0:
                continue

            _lr = _run_data.get_column("learning_rate")[0]
            _schedule = _run_data.get_column("schedule")[0]
            _label = _run_data.get_column("label")[0]

            _steps = _run_data.get_column("step").to_numpy()
            _values = _run_data.get_column(_metric).to_numpy()
            _scalar = 10 if "cm" in _metric else 1

            _ax.plot(
                _steps,
                _values * _scalar,
                color=lr_to_color.get(_lr, "gray"),
                linestyle=linestyles.get(_schedule, "-"),
                label=_label,
                alpha=0.7,
                linewidth=1.5,
            )

        _ax.set_title(_title)
        _ax.set_xlabel("Step")
        _ax.grid(True, alpha=0.3)
        _ax.spines[["right", "top"]].set_visible(False)

    axes[0].set_yscale("log")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right upper", fontsize="small")

    fig.savefig("docs/experiments/003-lr-scheduling/fig1.png")
    fig
    return


if __name__ == "__main__":
    app.run()
