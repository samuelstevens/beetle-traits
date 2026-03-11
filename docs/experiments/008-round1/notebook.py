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
    # Experiment 008: Round 1 BioRepo Results

    Single training run (`9a7jk34w`, biorepo-train). Box plots of length % error by genus on the BioRepo val split.
    """)
    return (mo,)


@app.cell
def _(pathlib, pl):
    run_id = "9a7jk34w"
    fpath = pathlib.Path("docs/experiments/008-round1/results") / f"{run_id}.parquet"
    assert fpath.exists(), f"Results not found at '{fpath}'. Run inference first."

    df = (
        pl
        .read_parquet(fpath)
        .with_columns(
            (pl.col("length_line_err_cm") / pl.col("gt_length_cm") * 100).alias(
                "length_pct_err"
            ),
            pl.col("scientific_name").str.split(" ").list.first().alias("genus"),
        )
        .filter(pl.col("length_pct_err").is_finite())
    )
    return (df,)


@app.cell
def _(pathlib, pl):
    import btx.data.biorepo as _biorepo

    _cfg = _biorepo.Config(
        split="train",
        annotations=pathlib.Path("data/biorepo-formatted/annotations.json"),
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
def _(df, pl, train_genera):
    def label_genus(genus: str) -> str:
        return f"* {genus}" if genus in train_genera else genus

    # Summary table
    (
        df
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
    return (label_genus,)


@app.cell
def _(alt, df, label_genus, pl):
    # Pre-aggregate to avoid embedding raw data in the Altair JSON.
    # Genera with biorepo training data are prefixed with "* ".
    stats = (
        df
        .group_by("genus")
        .agg(
            pl.col("length_pct_err").min().alias("min"),
            pl.col("length_pct_err").quantile(0.25).alias("q1"),
            pl.col("length_pct_err").median().alias("median"),
            pl.col("length_pct_err").quantile(0.75).alias("q3"),
            pl.col("length_pct_err").max().alias("max"),
        )
        .with_columns(
            pl
            .col("genus")
            .map_elements(label_genus, return_dtype=pl.String)
            .alias("genus")
        )
        .sort("median", descending=True)
    )

    genus_order = stats.get_column("genus").to_list()
    n_genera = len(genus_order)
    x_scale = alt.Scale(type="log")

    whiskers = (
        alt
        .Chart(stats)
        .mark_rule()
        .encode(
            y=alt.Y("genus:N", sort=genus_order, title="Genus (* = in training)"),
            x=alt.X("min:Q", scale=x_scale, title="Length % Error"),
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
            x=alt.X("q1:Q", scale=x_scale, title=None),
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
            x=alt.X("median:Q", scale=x_scale, title=None),
        )
    )
    threshold = (
        alt
        .Chart(pl.DataFrame({"x": [0.3]}))
        .mark_rule(color="red", strokeDash=[4, 4])
        .encode(x=alt.datum(0.3))
    )

    alt.layer(whiskers, box, median_tick, threshold).properties(
        title="BioRepo val: length % error by genus (red = 0.3% target, * = in training)",
        width=600,
        height=n_genera * 20,
    )
    return


@app.cell
def _(alt, label_genus, pathlib, pl):
    def _load(fpath: pathlib.Path, label: str) -> pl.DataFrame:
        assert fpath.exists(), f"Results not found at '{fpath}'. Run inference first."
        return (
            pl
            .read_parquet(fpath)
            .with_columns(
                (pl.col("length_line_err_cm") / pl.col("gt_length_cm") * 100).alias(
                    "length_pct_err"
                ),
                pl.col("scientific_name").str.split(" ").list.first().alias("genus"),
                pl.lit(label).alias("run"),
            )
            .filter(pl.col("length_pct_err").is_finite())
        )

    _both = pl.concat([
        _load(
            pathlib.Path("docs/experiments/008-round1/results/9a7jk34w.parquet"),
            "exp-008",
        ),
        _load(
            pathlib.Path("docs/experiments/007-biorepo-split/results/um3ukq2x.parquet"),
            "exp-007",
        ),
    ])

    _delta = (
        _both
        .group_by("run", "genus")
        .agg(pl.col("length_pct_err").median().alias("median"))
        .pivot(on="run", index="genus", values="median")
        .with_columns(
            (pl.col("exp-008") - pl.col("exp-007")).alias("delta"),
            pl
            .col("genus")
            .map_elements(label_genus, return_dtype=pl.String)
            .alias("genus"),
        )
        .sort("delta")
    )

    _order = _delta.get_column("genus").to_list()

    _zero = (
        alt
        .Chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(color="gray", strokeDash=[4, 4])
        .encode(x="x:Q")
    )
    _bars = (
        alt
        .Chart(_delta)
        .mark_bar()
        .encode(
            y=alt.Y("genus:N", sort=_order, title="Genus (* = in training)"),
            x=alt.X("delta:Q", title="Delta median length % error (exp-008 - exp-007)"),
            color=alt.condition(
                "datum.delta < 0", alt.value("steelblue"), alt.value("orange")
            ),
            tooltip=[
                "genus:N",
                alt.Tooltip("delta:Q", format=".2f"),
                alt.Tooltip("exp-008:Q", format=".2f"),
                alt.Tooltip("exp-007:Q", format=".2f"),
            ],
        )
    )

    alt.layer(_zero, _bars).properties(
        title="Median length % error delta by genus: exp-008 minus exp-007 (blue = 008 better, orange = 007 better)",
        width=600,
        height=len(_order) * 20,
    )
    return


@app.cell
def _(df, mo):
    _genera = df.get_column("genus").unique().sort().to_list()
    genus_picker = mo.ui.dropdown(_genera, value=_genera[0], label="Genus")
    genus_picker
    return (genus_picker,)


@app.cell
def _(df, genus_picker, mo, pl, plt):
    import numpy as np
    from PIL import Image

    _genus_df = df.filter(pl.col("genus") == genus_picker.value)
    _sample_df = _genus_df.sample(n=min(20, len(_genus_df)), seed=0)

    _n = len(_sample_df)
    _ncols = 4
    _nrows = (_n + _ncols - 1) // _ncols
    _fig, _axes = plt.subplots(
        _nrows, _ncols, figsize=(4 * _ncols, 4 * _nrows), layout="constrained", dpi=90
    )
    _axes_flat = np.asarray(_axes).ravel()
    for _ax in _axes_flat[_n:]:
        _ax.set_visible(False)

    for _ax, _row in zip(_axes_flat, _sample_df.iter_rows(named=True)):
        with Image.open(_row["img_fpath"]) as _fd:
            _img = np.asarray(_fd.convert("RGB"))
        _ax.imshow(_img)

        # pred_coords_px and gt_coords_px are flat [w0x,w0y,w1x,w1y,l0x,l0y,l1x,l1y]
        _pred = np.asarray(_row["pred_coords_px"], dtype=np.float32).reshape(2, 2, 2)
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
        f"{genus_picker.value}: cyan/magenta=GT, lime/yellow=pred", fontsize=10
    )
    mo.mpl.interactive(_fig)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
