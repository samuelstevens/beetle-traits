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
    # Experiment 005: Active Learning Analysis

    Three training runs with sigma=1, LR in {0.03, 0.1, 0.3}. Inference over Hawaii (train split) + BioRepo.

    **Goal metric**: percent error = |pred_length - gt_length| / gt_length * 100. Target: consistently below 0.3% (collaborator's CV^2 threshold). This ensures prediction error is proportional to body size.

    **Known issue**: `gt_coords_px` is all NaN in the current inference output. We need to fix inference.py to save ground truth coordinates and add `gt_width_cm` / `gt_length_cm` columns to compute percent error.
    """)
    return


@app.cell
def _(pathlib, pl):
    results_dpath = pathlib.Path("results")

    run_ids = {
        "gxdlfrgd": 0.03,
        "egqr97d7": 0.1,
        "v1t5i5tq": 0.3,
    }

    dfs = []
    for run_id, lr in run_ids.items():
        fpath = results_dpath / f"{run_id}.parquet"
        df = pl.read_parquet(fpath).with_columns(
            pl.lit(lr).alias("learning_rate"),
            pl.lit(run_id).alias("run_id"),
        )
        dfs.append(df)

    all_df = pl.concat(dfs)
    return all_df, run_ids


@app.cell
def _(mo):
    mo.md("## Per-run, per-dataset summary")
    return


@app.cell
def _(all_df, pl):
    summary = (
        all_df.group_by("learning_rate", "dataset")
        .agg(
            pl.col("width_line_err_cm").median().alias("median_width_err_cm"),
            pl.col("length_line_err_cm").median().alias("median_length_err_cm"),
            pl.col("width_line_err_cm").mean().alias("mean_width_err_cm"),
            pl.col("length_line_err_cm").mean().alias("mean_length_err_cm"),
            pl.col("mean_entropy").median().alias("median_entropy"),
            pl.len().alias("n_samples"),
        )
        .sort("dataset", "learning_rate")
    )
    summary
    return


@app.cell
def _(mo):
    mo.md("## Error distributions by dataset and learning rate")
    return


@app.cell
def _(all_df, alt, pl):
    err_df = all_df.select(
        "dataset", "learning_rate", "width_line_err_cm", "length_line_err_cm"
    ).unpivot(
        on=["width_line_err_cm", "length_line_err_cm"],
        index=["dataset", "learning_rate"],
        variable_name="metric",
        value_name="error_cm",
    ).with_columns(
        pl.col("metric").str.replace("_line_err_cm", ""),
        pl.col("learning_rate").cast(pl.String),
    )

    alt.Chart(err_df).mark_boxplot(extent="min-max").encode(
        x=alt.X("learning_rate:N", title="Learning Rate"),
        y=alt.Y("error_cm:Q", title="Line Error (cm)", scale=alt.Scale(type="log")),
        color="learning_rate:N",
        column=alt.Column("dataset:N"),
        row=alt.Row("metric:N"),
    ).properties(width=250, height=200)
    return


@app.cell
def _(mo):
    mo.md("## Per-species error (all runs combined)")
    return


@app.cell
def _(all_df, pl):
    species_err = (
        all_df.group_by("scientific_name", "dataset")
        .agg(
            pl.col("length_line_err_cm").median().alias("median_length_err_cm"),
            pl.col("length_line_err_cm").mean().alias("mean_length_err_cm"),
            pl.col("length_line_err_cm").max().alias("max_length_err_cm"),
            pl.col("width_line_err_cm").median().alias("median_width_err_cm"),
            pl.col("mean_entropy").median().alias("median_entropy"),
            pl.len().alias("n_samples"),
        )
        .sort("median_length_err_cm", descending=True)
    )
    species_err
    return


@app.cell
def _(mo):
    mo.md("## Uncertainty vs error")
    return


@app.cell
def _(all_df, alt, pl):
    scatter_df = all_df.select(
        "mean_entropy", "length_line_err_cm", "dataset", "learning_rate"
    ).with_columns(pl.col("learning_rate").cast(pl.String))

    alt.Chart(scatter_df).mark_circle(size=10, opacity=0.3).encode(
        x=alt.X("mean_entropy:Q", title="Mean Heatmap Entropy"),
        y=alt.Y("length_line_err_cm:Q", title="Length Line Error (cm)", scale=alt.Scale(type="log")),
        color="dataset:N",
        column=alt.Column("learning_rate:N"),
    ).properties(width=300, height=300)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Ground truth coords check

    Verify whether `gt_coords_px` contains valid data. If all NaN, percent error cannot be computed from the current Parquet files.
    """)
    return


@app.cell
def _(all_df, mo, np):
    gt = np.array(all_df["gt_coords_px"].to_list())
    n_finite = np.isfinite(gt).any(axis=1).sum()
    n_total = len(gt)

    mo.md(f"""
    - Samples with any finite gt_coords: **{n_finite} / {n_total}**
    - {"gt_coords_px is all NaN -- inference.py needs to be fixed to save ground truth coordinates." if n_finite == 0 else f"{n_finite} samples have ground truth coordinates."}
    - **To compute percent error**, we also need `gt_width_cm` and `gt_length_cm` columns in the inference output.
    """)
    return


if __name__ == "__main__":
    app.run()
