import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
def _():
    import pathlib

    import altair as alt
    import matplotlib.pyplot as plt
    import polars as pl
    return alt, pathlib, pl, plt


@app.cell
def _():
    import marimo as mo

    mo.md("""
    # Experiment 011: Round 2 Results

    Two training runs on BioRepo (val split): `1ykzeqat` (ViT-S, wsd schedule) and `7wdeiefc` (ViT-B, no schedule). Box plots of length % error by genus, plus a 3-way comparison table against exp-008 round-1 (`9a7jk34w`).
    """)
    return (mo,)


@app.cell
def _(pathlib, pl):
    RESULTS_DPATH = pathlib.Path("docs/experiments/011-round2-results/results")
    EXP008_FPATH = pathlib.Path("docs/experiments/008-round1/results/9a7jk34w.parquet")

    RUNS = {
        "small (1ykzeqat)": RESULTS_DPATH / "1ykzeqat.parquet",
        "base (7wdeiefc)": RESULTS_DPATH / "7wdeiefc.parquet",
    }

    def load_run(fpath: pathlib.Path, label: str) -> pl.DataFrame:
        assert fpath.exists(), f"Results not found at '{fpath}'. Run inference first."
        return (
            pl
            .read_parquet(fpath)
            .with_columns(
                (pl.col("length_line_err_cm") / pl.col("gt_length_cm") * 100).alias(
                    "length_pct_err"
                ),
                (pl.col("width_line_err_cm") / pl.col("gt_width_cm") * 100).alias(
                    "width_pct_err"
                ),
                pl.col("scientific_name").str.split(" ").list.first().alias("genus"),
                pl.lit(label).alias("run"),
            )
            .filter(pl.col("length_pct_err").is_finite())
        )

    dfs = {label: load_run(fpath, label) for label, fpath in RUNS.items()}
    return EXP008_FPATH, dfs, load_run


@app.cell
def _(dfs, pl):
    dfs["base (7wdeiefc)"].filter(pl.col("scientific_name").str.contains("Agonum"))
    return


@app.cell
def _(dfs, pl):
    dfs["small (1ykzeqat)"].filter(pl.col("scientific_name").str.contains("Agonum"))
    return


@app.cell
def _(dfs, pl):
    dfs["small (1ykzeqat)"].group_by("genus").len().filter(pl.col("len") > 5).height
    return


@app.cell
def _(pathlib, pl):
    import btx.data.biorepo as _biorepo

    _cfg = _biorepo.Config(
        split="train",
        annotations=pathlib.Path(
            "/fs/scratch/PAS2136/cain429/biorepo-formatted/annotations.json"
        ),
        root=pathlib.Path("/fs/scratch/PAS2136/cain429/Subset-Exp"),
    )
    train_genera = frozenset(
        _biorepo
        .Dataset(_cfg)
        .df.with_columns(
            pl.col("scientific_name").str.split(" ").list.first().alias("genus")
        )
        .get_column("genus")
        .unique()
        .to_list()
    )
    return (train_genera,)


@app.cell
def _(train_genera):
    def label_genus(genus: str) -> str:
        return f"* {genus}" if genus in train_genera else genus
    return (label_genus,)


@app.cell
def _(dfs, label_genus, mo, pl):
    _items = []
    for _label, _df in dfs.items():
        _summary = (
            _df
            .group_by("genus")
            .agg(
                pl.col("length_pct_err").median().alias("median_length_pct_err"),
                pl.col("length_pct_err").mean().alias("mean_length_pct_err"),
                pl.len().alias("n_samples"),
            )
            .with_columns(
                pl
                .col("genus")
                .map_elements(label_genus, return_dtype=pl.String)
                .alias("genus")
            )
            .sort("median_length_pct_err", descending=True)
        )
        _items.append(mo.md(f"## Summary: {_label}"))
        _items.append(_summary)
    return


@app.cell
def _(mo):
    only_large_genera = mo.ui.switch(
        value=False, label="Only genera with >10 individuals"
    )
    only_large_genera
    return (only_large_genera,)


@app.cell
def _(EXP008_FPATH, alt, dfs, label_genus, load_run, only_large_genera, pl):
    _exp008 = load_run(EXP008_FPATH, "exp-008 (9a7jk34w)")
    _all_runs = {**dfs, "exp-008 (9a7jk34w)": _exp008}

    _combined = pl.concat(list(_all_runs.values()))
    _combined_counts = _combined.group_by("genus").agg(pl.len().alias("n"))

    _combined_stats = (
        _combined
        .group_by("genus")
        .agg(pl.col("length_pct_err").median().alias("median"))
        .join(_combined_counts, on="genus")
        .with_columns(
            pl
            .col("genus")
            .map_elements(label_genus, return_dtype=pl.String)
            .alias("genus")
        )
        .sort("median", descending=True)
    )
    _global_genus_order = _combined_stats.get_column("genus").to_list()

    _x_min = (
        _combined
        .get_column("length_pct_err")
        .filter(_combined.get_column("length_pct_err") > 0)
        .min()
    )
    _x_max = _combined.get_column("length_pct_err").max()
    _x_scale = alt.Scale(type="log", domain=[_x_min, _x_max])

    def _make_boxplot(df: pl.DataFrame, title: str):
        stats = (
            df
            .group_by("genus")
            .agg(
                pl.col("length_pct_err").min().alias("min"),
                pl.col("length_pct_err").quantile(0.25).alias("q1"),
                pl.col("length_pct_err").median().alias("median"),
                pl.col("length_pct_err").quantile(0.75).alias("q3"),
                pl.col("length_pct_err").max().alias("max"),
                pl.len().alias("n"),
            )
            .with_columns(
                pl
                .col("genus")
                .map_elements(label_genus, return_dtype=pl.String)
                .alias("genus")
            )
        )
        if only_large_genera.value:
            stats = stats.filter(pl.col("n") > 10)
        genus_order = [
            g
            for g in _global_genus_order
            if g in set(stats.get_column("genus").to_list())
        ]

        whiskers = (
            alt
            .Chart(stats)
            .mark_rule()
            .encode(
                y=alt.Y("genus:N", sort=genus_order, title="Genus (* = in training)"),
                x=alt.X("min:Q", scale=_x_scale, title="Length % Error"),
                x2=alt.X2("max:Q"),
                color=alt.Color("genus:N", legend=None),
            )
        )
        box = (
            alt
            .Chart(stats)
            .mark_bar(size=10)
            .encode(
                y=alt.Y("genus:N", sort=genus_order, title=None),
                x=alt.X("q1:Q", scale=_x_scale, title=None),
                x2=alt.X2("q3:Q"),
                color=alt.Color("genus:N", legend=None),
            )
        )
        median_tick = (
            alt
            .Chart(stats)
            .mark_tick(color="white", size=10, thickness=2)
            .encode(
                y=alt.Y("genus:N", sort=genus_order, title=None),
                x=alt.X("median:Q", scale=_x_scale, title=None),
            )
        )
        threshold = (
            alt
            .Chart(pl.DataFrame({"x": [0.3]}))
            .mark_rule(color="red", strokeDash=[4, 4])
            .encode(x=alt.datum(0.3))
        )
        return alt.layer(whiskers, box, median_tick, threshold).properties(
            title=title,
            width=600,
            height=len(genus_order) * 20,
        )

    alt.vconcat(*[
        _make_boxplot(
            df,
            f"{label}: BioRepo val length % error by genus (red = 0.3% target, * = in training)",
        )
        for label, df in _all_runs.items()
    ])
    return


@app.cell
def _(EXP008_FPATH, dfs, load_run, mo, pl):
    _exp008 = load_run(EXP008_FPATH, "exp-008 (9a7jk34w)")
    _all = pl.concat([*dfs.values(), _exp008])

    THRESHOLD = 0.3

    _summary = (
        _all
        .group_by("run")
        .agg(
            pl.col("length_pct_err").mean().alias("mean_%_err"),
            pl.col("length_pct_err").median().alias("median_%_err"),
            (pl.col("length_pct_err") < THRESHOLD).sum().alias(f"n_under_{THRESHOLD}%"),
            pl.len().alias("n_total"),
        )
        .with_columns(
            (pl.col(f"n_under_{THRESHOLD}%") / pl.col("n_total") * 100).alias(
                f"%_under_{THRESHOLD}%"
            ),
        )
        .sort("median_%_err")
    )
    mo.vstack([
        mo.md(
            f"## Overall comparison: mean/median length % error (threshold = {THRESHOLD}%)"
        ),
        _summary,
    ])
    return


@app.cell
def _(EXP008_FPATH, dfs, label_genus, load_run, mo, pl):
    _exp008 = load_run(EXP008_FPATH, "exp-008 (9a7jk34w)")
    _all = pl.concat([*dfs.values(), _exp008])

    _medians = (
        _all
        .group_by("run", "genus")
        .agg(pl.col("length_pct_err").median().alias("median_pct_err"))
        .pivot(on="run", index="genus", values="median_pct_err")
        .with_columns(
            pl
            .col("genus")
            .map_elements(label_genus, return_dtype=pl.String)
            .alias("genus")
        )
        .sort("genus")
    )
    mo.vstack([mo.md("## 3-way comparison: median length % error by genus"), _medians])
    return


@app.cell
def _(EXP008_FPATH, alt, dfs, label_genus, load_run, pl):
    _exp008 = load_run(EXP008_FPATH, "exp-008 (9a7jk34w)")
    _all = pl.concat([*dfs.values(), _exp008])

    _delta = (
        _all
        .group_by("run", "genus")
        .agg(pl.col("length_pct_err").median().alias("median"))
        .pivot(on="run", index="genus", values="median")
        .rename({
            "small (1ykzeqat)": "small",
            "base (7wdeiefc)": "base",
            "exp-008 (9a7jk34w)": "exp008",
        })
        .with_columns(
            (pl.col("small") - pl.col("exp008")).alias("delta_small"),
            (pl.col("base") - pl.col("exp008")).alias("delta_base"),
            pl
            .col("genus")
            .map_elements(label_genus, return_dtype=pl.String)
            .alias("genus"),
        )
        .sort("delta_small")
    )

    _order = _delta.get_column("genus").to_list()
    _zero = (
        alt
        .Chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(color="gray", strokeDash=[4, 4])
        .encode(x="x:Q")
    )

    def _bars(col: str, title: str):
        return (
            alt
            .Chart(_delta)
            .mark_bar()
            .encode(
                y=alt.Y("genus:N", sort=_order, title="Genus (* = in training)"),
                x=alt.X(f"{col}:Q", title=title),
                color=alt.condition(
                    f"datum.{col} < 0", alt.value("steelblue"), alt.value("orange")
                ),
                tooltip=[
                    "genus:N",
                    alt.Tooltip(f"{col}:Q", format=".2f"),
                    alt.Tooltip("small:Q", format=".2f"),
                    alt.Tooltip("base:Q", format=".2f"),
                    alt.Tooltip("exp008:Q", format=".2f"),
                ],
            )
        )

    alt.vconcat(
        alt.layer(
            _zero, _bars("delta_small", "Delta median length % error (small - exp-008)")
        ).properties(
            title="small (1ykzeqat) vs exp-008 (blue = 011 better, orange = 008 better)",
            width=600,
            height=len(_order) * 20,
        ),
        alt.layer(
            _zero, _bars("delta_base", "Delta median length % error (base - exp-008)")
        ).properties(
            title="base (7wdeiefc) vs exp-008 (blue = 011 better, orange = 008 better)",
            width=600,
            height=len(_order) * 20,
        ),
    )
    return


@app.cell
def _(dfs, mo, pl):
    _combined = pl.concat(list(dfs.values()))
    _genera = _combined.get_column("genus").unique().sort().to_list()
    genus_picker = mo.ui.dropdown(_genera, value=_genera[0], label="Genus")
    genus_picker
    return (genus_picker,)


@app.cell
def _(dfs, genus_picker, mo, pl, plt):
    import numpy as np
    from PIL import Image

    _figs = []
    for _label, _df in dfs.items():
        _genus_df = _df.filter(pl.col("genus") == genus_picker.value)
        _sample_df = _genus_df.sample(n=min(20, len(_genus_df)), seed=0)
        _n = len(_sample_df)
        _ncols = 4
        _nrows = (_n + _ncols - 1) // _ncols
        _fig, _axes = plt.subplots(
            _nrows,
            _ncols,
            figsize=(4 * _ncols, 4 * _nrows),
            layout="constrained",
            dpi=90,
        )
        _axes_flat = np.asarray(_axes).ravel()
        for _ax in _axes_flat[_n:]:
            _ax.set_visible(False)

        for _ax, _row in zip(_axes_flat, _sample_df.iter_rows(named=True)):
            with Image.open(_row["img_fpath"]) as _fd:
                _img = np.asarray(_fd.convert("RGB"))
            _ax.imshow(_img)

            _pred = np.asarray(_row["pred_coords_px"], dtype=np.float32).reshape(
                2, 2, 2
            )
            _gt = np.asarray(_row["gt_coords_px"], dtype=np.float32).reshape(2, 2, 2)

            for _line_i, (_gt_col, _pred_col) in enumerate([
                ("cyan", "lime"),
                ("magenta", "yellow"),
            ]):
                (_x0, _y0), (_x1, _y1) = _gt[_line_i]
                _ax.plot(
                    [_x0, _x1],
                    [_y0, _y1],
                    color=_gt_col,
                    linewidth=1.8,
                    marker="o",
                    markersize=4,
                )
                (_x0, _y0), (_x1, _y1) = _pred[_line_i]
                _ax.plot(
                    [_x0, _x1],
                    [_y0, _y1],
                    color=_pred_col,
                    linestyle="--",
                    linewidth=1.8,
                    marker="x",
                    markersize=5,
                )

            _err = _row["length_pct_err"]
            _ax.set_title(
                f"{_row['scientific_name']}\n{_err:.1f}% err"
                if np.isfinite(_err)
                else _row["scientific_name"],
                fontsize=8,
            )
            _ax.set_xticks([])
            _ax.set_yticks([])
            _ax.spines[:].set_visible(False)

        _fig.suptitle(
            f"{_label} -- {genus_picker.value}: cyan/magenta=GT, lime/yellow=pred",
            fontsize=10,
        )
        _figs.append(mo.mpl.interactive(_fig))

    mo.vstack(_figs)
    return


@app.cell
def _(alt, dfs, pl):
    _df = (
        dfs["small (1ykzeqat)"]
        .filter(pl.col("mean_entropy").is_finite())
        .sample(n=500, seed=0, shuffle=True)
    )
    alt.Chart(_df).mark_point(opacity=0.4, size=20).encode(
        x=alt.X("mean_entropy:Q", title="Mean Entropy"),
        y=alt.Y(
            "length_pct_err:Q", title="Length % Error", scale=alt.Scale(type="log")
        ),
        color=alt.Color("genus:N", legend=None),
        tooltip=[
            "scientific_name:N",
            alt.Tooltip("length_pct_err:Q", format=".2f"),
            alt.Tooltip("mean_entropy:Q", format=".3f"),
        ],
    ).properties(
        title="small (1ykzeqat): length % error vs mean entropy",
        width=600,
        height=400,
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
