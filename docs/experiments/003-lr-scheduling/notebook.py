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
    return pl, plt, wandb


@app.cell
def _(pl, wandb):
    entity = "cain-429-the-ohio-state-university"
    project = "beetle-traits"
    tag = "exp-003-lr-scheduling"

    history_keys = [
        "train/loss",
        "val/biorepo/loss",
        "val/hawaii/loss",
    ]

    api = wandb.Api()
    runs = api.runs(
        path=f"{entity}/{project}",
        filters={"tags": {"$in": [tag]}},
        order="-updated_at",
        per_page=20,
    )

    schedule_tags = {"cosine", "wsd", "wsd-no-decay", "none"}

    rows = []
    for _run in list(runs)[:20]:
        _lr = _run.config.get("learning_rate") or _run.config.get("learning-rate")
        _schedule_tags = [t for t in _run.tags if t in schedule_tags]
        assert len(_schedule_tags) == 1, f"run {_run.id} has tags {_run.tags}"
        _schedule = _schedule_tags[0]
        _label = f"{_schedule} lr={_lr}"

        for _row in _run.scan_history(keys=["_step", *history_keys]):
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

    history_df = pl.DataFrame(
        rows,
        schema={
            "run_id": pl.Utf8,
            "label": pl.Utf8,
            "learning_rate": pl.Float64,
            "schedule": pl.Utf8,
            "step": pl.Int64,
            **{k.replace("/", "_"): pl.Float64 for k in history_keys},
        },
    )
    return (history_df,)


@app.cell
def _(history_df):
    history_df["schedule"].unique()
    return


@app.cell
def _(history_df, pl, plt):
    val_metrics = [
        ("val_biorepo_loss", "BioRepo"),
        ("val_hawaii_loss", "Hawaii"),
    ]
    schedules = ["cosine", "none", "wsd", "wsd-no-decay"]
    color_map = {
        "cosine": "#1f77b4",
        "none": "#ff7f0e",
        "wsd": "#2ca02c",
        "wsd-no-decay": "#d62728",
    }

    def _make_summary(_df, _col):
        return (
            _df
            .filter(pl.col(_col).is_not_null())
            .group_by("run_id")
            .agg(
                pl.col("learning_rate").first(),
                pl.col("schedule").first(),
                pl.col(_col).min().alias("best"),
                pl.col(_col).sort_by("step").last().alias("final"),
            )
            .sort("learning_rate")
        )

    def _plot_schedule(_ax, _summary, _schedule):
        _subset = _summary.filter(pl.col("schedule") == _schedule)
        if len(_subset) == 0:
            return
        _lrs = _subset.get_column("learning_rate").to_numpy()
        _color = color_map[_schedule]
        _ax.plot(
            _lrs,
            _subset.get_column("best").to_numpy(),
            marker="o",
            linestyle="-",
            color=_color,
            label=f"{_schedule} best",
            alpha=0.7,
        )
        _ax.plot(
            _lrs,
            _subset.get_column("final").to_numpy(),
            marker="s",
            linestyle="--",
            color=_color,
            label=f"{_schedule} final",
            alpha=0.7,
        )

    def _style_ax(_ax, _title):
        _ax.set_xlabel("Learning Rate")
        _ax.set_ylabel("Loss")
        _ax.set_title(_title)
        _ax.set_xscale("log")
        _ax.set_yscale("log")
        _ax.legend(fontsize="small")
        _ax.grid(True, alpha=0.3)
        _ax.spines[["right", "top"]].set_visible(False)

    # Row 0: all schedules combined; rows 1-4: one schedule each
    _fig, _axes = plt.subplots(5, 2, figsize=(12, 20), dpi=150, layout="constrained")

    for _j, (_col, _dataset) in enumerate(val_metrics):
        _summary = _make_summary(history_df, _col)

        # Top row: all schedules
        for _s in schedules:
            _plot_schedule(_axes[0, _j], _summary, _s)
        _style_ax(_axes[0, _j], f"{_dataset} (all schedules)")

        # Rows 1-4: individual schedules
        for _i, _s in enumerate(schedules):
            _plot_schedule(_axes[_i + 1, _j], _summary, _s)
            _style_ax(_axes[_i + 1, _j], f"{_dataset} ({_s})")

    _fig.savefig("docs/experiments/003-lr-scheduling/fig1.png")
    _fig
    return


if __name__ == "__main__":
    app.run()
