import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
def _():
    import pathlib

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl
    return alt, mo, np, pathlib, pl


@app.cell
def _(mo):
    mo.md("""
    # Experiment 010: Annotation Improvements

    Five training runs (seed=17, LR=0.01, WSD, 26k steps) identical except for how many annotations are added per target genus (Dicheirus, Discoderus, Metrius): 0, 5, 10, 15, or all. All other training data (Hawaii, Beetlepalooza, BioRepo round-1) is held constant.

    **Hypothesis**: adding more annotations for a genus decreases its median percent error, and a small number of annotations (5-10) may have an outsized effect.

    **Goal metric**: percent error = |pred_length - gt_length| / gt_length * 100. Target: consistently below 0.3%.
    """)
    return


@app.cell
def _(pathlib, pl):
    results_dpath = pathlib.Path(
        "/fs/scratch/PAS2136/cain429/beetle-traits-exp-10/results"
    )

    n_samples_order = ["0", "5", "10", "15", "all"]

    # Drop heavy per-row columns (embeddings, raw coords, image paths) that are
    # not needed for summary charts. They are loaded separately in the example
    # predictions cell.
    _keep = [
        "dataset",
        "scientific_name",
        "beetle_id",
        "length_line_err_cm",
        "gt_length_cm",
        "width_line_err_cm",
        "gt_width_cm",
        "mean_entropy",
    ]

    _parts = []
    for _n in n_samples_order:
        _fpath = results_dpath / f"exp10_{_n}_labeled.parquet"
        _df = pl.read_parquet(_fpath, columns=_keep).with_columns(
            pl.lit(_n).alias("n_samples"),
            (pl.col("length_line_err_cm") / pl.col("gt_length_cm") * 100).alias(
                "length_pct_err"
            ),
            (pl.col("width_line_err_cm") / pl.col("gt_width_cm") * 100).alias(
                "width_pct_err"
            ),
            pl.col("scientific_name").str.split(" ").list.first().alias("genus"),
        )
        _parts.append(_df)

    all_df = pl.concat(_parts)
    target_genera = ["Dicheirus", "Discoderus", "Metrius"]
    return all_df, n_samples_order, target_genera


@app.cell
def _(mo):
    mo.md("""
    ## Per-run summary
    """)
    return


@app.cell
def _(all_df, pl):
    (
        all_df.group_by("n_samples", "dataset")
        .agg(
            pl.col("length_pct_err").median().alias("median_length_pct_err"),
            pl.col("length_pct_err").mean().alias("mean_length_pct_err"),
            pl.col("width_pct_err").median().alias("median_width_pct_err"),
            pl.col("length_line_err_cm").median().alias("median_length_err_cm"),
            pl.len().alias("n_samples_count"),
        )
        .sort("n_samples", "dataset")
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Hypothesis test: target genera error vs n_samples added

    Median length percent error for each target genus as more annotations are added. The key question is whether error drops monotonically and whether the first few samples have an outsized effect.
    """)
    return


@app.cell
def _(all_df, alt, mo, n_samples_order, pathlib, pl, target_genera):
    _target_df = (
        all_df.filter(
            pl.col("genus").is_in(target_genera) & pl.col("length_pct_err").is_finite()
        )
        .group_by("n_samples", "genus")
        .agg(
            pl.col("length_pct_err").median().alias("median_length_pct_err"),
            pl.col("length_pct_err").mean().alias("mean_length_pct_err"),
            pl.len().alias("n"),
        )
    )

    _threshold = (
        alt.Chart(pl.DataFrame({"y": [0.3]}))
        .mark_rule(color="red", strokeDash=[4, 4])
        .encode(y="y:Q")
    )

    _line = (
        alt.Chart(_target_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("n_samples:O", sort=n_samples_order, title="Samples added per genus"),
            y=alt.Y(
                "median_length_pct_err:Q",
                title="Median length percent error (%)",
                scale=alt.Scale(type="log"),
            ),
            color=alt.Color("genus:N", title="Genus"),
            tooltip=["genus:N", "n_samples:O", "median_length_pct_err:Q", "n:Q"],
        )
    )

    _chart = (_line + _threshold).properties(
        width=500,
        height=350,
        title="Target genus error vs annotations added (red = 0.3% target)",
    )
    _out = pathlib.Path("docs/experiments/010-annotation-improvements/results/target-genus-error-line.html")
    _out.parent.mkdir(parents=True, exist_ok=True)
    _chart.save(_out)
    mo.vstack([_chart, mo.md(f"Saved to `{_out}`")])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Target genera: error distribution by n_samples
    """)
    return


@app.cell
def _(all_df, alt, mo, n_samples_order, pathlib, pl, target_genera):
    _df = all_df.filter(
        pl.col("genus").is_in(target_genera) & pl.col("length_pct_err").is_finite()
    )

    _threshold = (
        alt.Chart(_df)
        .mark_rule(color="red", strokeDash=[4, 4])
        .encode(y=alt.datum(0.3))
    )

    _boxes = (
        alt.Chart(_df)
        .mark_boxplot(extent="min-max", opacity=0.7)
        .encode(
            x=alt.X("n_samples:O", sort=n_samples_order, title="Samples added per genus"),
            y=alt.Y(
                "length_pct_err:Q",
                title="Length percent error (%)",
                scale=alt.Scale(type="log"),
            ),
            color=alt.Color("n_samples:O", sort=n_samples_order, legend=None),
        )
    )

    _chart = alt.layer(_boxes, _threshold).facet(
        column=alt.Column("genus:N", title="Genus"),
    ).properties(title="Length percent error for target genera").resolve_scale(y="shared")
    _out = pathlib.Path("docs/experiments/010-annotation-improvements/results/target-genera-boxplot.html")
    _chart.save(_out)
    mo.vstack([_chart, mo.md(f"Saved to `{_out}`")])
    return


@app.cell
def _(mo):
    mo.md("""
    ## All genera: percent error distribution by n_samples
    """)
    return


@app.cell
def _(all_df, alt, mo, n_samples_order, pathlib, pl):
    _df = all_df.filter(pl.col("length_pct_err").is_finite())

    _genus_order = (
        _df.filter(pl.col("n_samples") == "0")
        .group_by("genus")
        .agg(pl.col("length_pct_err").median().alias("median_pct_err"))
        .sort("median_pct_err", descending=True)
        .get_column("genus")
        .to_list()
    )

    _threshold = (
        alt.Chart(_df)
        .mark_rule(color="red", strokeDash=[4, 4])
        .encode(y=alt.datum(0.3))
    )

    _boxes = (
        alt.Chart(_df)
        .mark_boxplot(opacity=0.5, ticks=True)
        .encode(
            x=alt.X("genus:N", sort=_genus_order, title="Genus"),
            y=alt.Y(
                "length_pct_err:Q",
                title="Length percent error (%)",
                scale=alt.Scale(type="log"),
            ),
            color=alt.Color("n_samples:O", sort=n_samples_order, title="N samples"),
        )
    )

    _chart = alt.layer(_boxes, _threshold).facet(
        row=alt.Row("n_samples:O", sort=n_samples_order, title="Samples added per genus"),
    ).properties(title="Length percent error by genus (red = 0.3% target)").resolve_scale(
        y="shared"
    )
    _out = pathlib.Path("docs/experiments/010-annotation-improvements/results/all-genera-boxplot.html")
    _chart.save(_out)
    mo.vstack([_chart, mo.md(f"Saved to `{_out}`")])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Per-species error table (all runs)
    """)
    return


@app.cell
def _(all_df, mo, pathlib, pl):
    _table = (
        all_df.filter(pl.col("length_pct_err").is_finite())
        .group_by("n_samples", "scientific_name", "dataset")
        .agg(
            pl.col("length_pct_err").median().alias("median_length_pct_err"),
            pl.col("length_pct_err").mean().alias("mean_length_pct_err"),
            pl.len().alias("n"),
        )
        .sort("n_samples", "median_length_pct_err", descending=[False, True])
    )
    _out = pathlib.Path("docs/experiments/010-annotation-improvements/results/per-species-error.csv")
    _table.write_csv(_out)
    mo.vstack([_table, mo.md(f"Saved to `{_out}`")])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Uncertainty vs percent error (target genera)
    """)
    return


@app.cell
def _(all_df, alt, mo, n_samples_order, pathlib, pl, target_genera):
    _chart = alt.Chart(
        all_df.filter(
            pl.col("genus").is_in(target_genera) & pl.col("length_pct_err").is_finite()
        ).select("mean_entropy", "length_pct_err", "n_samples", "genus")
    ).mark_circle(size=12, opacity=0.3).encode(
        x=alt.X("mean_entropy:Q", title="Mean heatmap entropy"),
        y=alt.Y(
            "length_pct_err:Q",
            title="Length percent error (%)",
            scale=alt.Scale(type="log"),
        ),
        color=alt.Color("n_samples:O", sort=n_samples_order, title="N samples"),
        column=alt.Column("genus:N"),
    ).properties(width=250, height=300)
    _out = pathlib.Path("docs/experiments/010-annotation-improvements/results/entropy-scatter.html")
    _chart.save(_out)
    mo.vstack([_chart, mo.md(f"Saved to `{_out}`")])
    return


@app.cell
def _(all_df, mo, pl):
    _runs = []
    for _n in ["0", "5", "10", "15", "all"]:
        _finite = all_df.filter(
            (pl.col("n_samples") == _n) & pl.col("length_pct_err").is_finite()
        )
        _n_total = len(_finite)
        _n_below = _finite.filter(pl.col("length_pct_err") < 0.3).height
        _pct = _n_below / _n_total * 100 if _n_total > 0 else 0
        _runs.append(f"| {_n} | {_n_below} / {_n_total} | {_pct:.1f}% |")

    mo.md(
        "## Target metric: fraction below 0.3% error\n\n"
        "| N samples | Below 0.3% | Fraction |\n"
        "|---|---|---|\n"
        + "\n".join(_runs)
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Example predictions: 0 vs all samples

    2 samples at each error quantile (0%, 25%, 50%, 75%, 100%) for the 0-sample and all-sample runs on BioRepo. Cyan/magenta = GT width/length, lime/yellow = predicted.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    from PIL import Image
    return Image, plt


@app.cell
def _(Image, np, pathlib, pl, plt):
    _results_dpath = pathlib.Path(
        "/fs/scratch/PAS2136/cain429/beetle-traits-exp-10/results"
    )
    _heavy_cols = [
        "dataset",
        "scientific_name",
        "beetle_id",
        "img_fpath",
        "gt_coords_px",
        "pred_coords_px",
        "length_line_err_cm",
        "gt_length_cm",
    ]

    def _load_run(n):
        return pl.read_parquet(
            _results_dpath / f"exp10_{n}_labeled.parquet", columns=_heavy_cols
        ).with_columns(
            (pl.col("length_line_err_cm") / pl.col("gt_length_cm") * 100).alias(
                "length_pct_err"
            )
        )

    _quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]
    _labels = ["0%", "25%", "50%", "75%", "100%"]

    def _pick_quantile_rows(df, quantiles):
        _sorted = df.filter(
            (pl.col("dataset") == "biorepo") & pl.col("length_pct_err").is_finite()
        ).sort("length_pct_err")
        _nd = len(_sorted)
        rows = []
        for _q in quantiles:
            _qi = min(round(_q * (_nd - 1)), _nd - 2)
            rows.append(_sorted.row(_qi, named=True))
            rows.append(_sorted.row(_qi + 1, named=True))
        return rows

    _rows_0 = _pick_quantile_rows(_load_run("0"), _quantiles)
    _rows_all = _pick_quantile_rows(_load_run("all"), _quantiles)

    _grid = []
    for _row_i in range(len(_quantiles)):
        _hi = _row_i * 2
        _grid.append((_rows_0[_hi], "0 samples"))
        _grid.append((_rows_0[_hi + 1], "0 samples"))
        _grid.append((_rows_all[_hi], "all samples"))
        _grid.append((_rows_all[_hi + 1], "all samples"))

    fig, axes = plt.subplots(5, 4, figsize=(18, 22), dpi=60, layout="constrained")

    for _idx, (ax, (row, run_label)) in enumerate(zip(axes.reshape(-1), _grid)):
        with Image.open(row["img_fpath"]) as fd:
            ax.imshow(np.array(fd.convert("RGB")))

        _gt = np.array(row["gt_coords_px"]).reshape(2, 2, 2)
        _pred = np.array(row["pred_coords_px"]).reshape(2, 2, 2)

        for line_i, (gt_color, pred_color) in enumerate([
            ("cyan", "lime"),
            ("magenta", "yellow"),
        ]):
            (gx0, gy0), (gx1, gy1) = _gt[line_i]
            ax.plot([gx0, gx1], [gy0, gy1], "o-", color=gt_color, linewidth=2, markersize=4)
            (px0, py0), (px1, py1) = _pred[line_i]
            ax.plot(
                [px0, px1], [py0, py1], "x--", color=pred_color, linewidth=2, markersize=5
            )

        ax.set_title(
            f"{row['scientific_name']} ({row['beetle_id']})\n{run_label} | {row['length_pct_err']:.2f}%",
            fontsize=8,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        if _idx % 4 == 0:
            ax.set_ylabel(_labels[_idx // 4], fontsize=12, rotation=0, labelpad=30)

    fig.suptitle(
        "Left 2 cols: 0 samples | Right 2 cols: all samples\ncyan/magenta = GT, lime/yellow = predicted",
        fontsize=11,
    )
    _out = pathlib.Path("docs/experiments/010-annotation-improvements/results/example-predictions.png")
    fig.savefig(_out, dpi=100, bbox_inches="tight")
    fig
    return


if __name__ == "__main__":
    app.run()
