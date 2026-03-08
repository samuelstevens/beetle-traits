import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
def _():
    import pathlib

    import altair as alt
    import marimo as mo
    import polars as pl
    return alt, mo, pathlib, pl


@app.cell
def _(mo):
    mo.md("""
    # Experiment 007: BioRepo Split Analysis

    Two training runs comparing biorepo as training data vs validation-only data. Uses best-known settings from exp-006 (cosine schedule, lr=3e-4).

    **Conditions**:
    - `biorepo-train` (um3ukq2x): biorepo split="train" adds biorepo to training set
    - `biorepo-val` (16gsx5v7): biorepo split="val" (default) uses biorepo only for validation

    **Goal**: does adding biorepo to training reduce length percent error on biorepo specimens?
    """)
    return


@app.cell
def _(pathlib, pl):
    results_dpath = pathlib.Path("docs/experiments/007-biorepo-split/results")

    run_ids = {
        "um3ukq2x": "biorepo-train",
        "16gsx5v7": "biorepo-val",
    }

    dfs = []
    for run_id, condition in run_ids.items():
        fpath = results_dpath / f"{run_id}.parquet"
        if not fpath.exists():
            continue
        df = pl.read_parquet(fpath).with_columns(
            pl.lit(condition).alias("condition"),
            pl.lit(run_id).alias("run_id"),
        )
        dfs.append(df)

    assert dfs, f"No result parquets found in '{results_dpath}'. Run inference first."
    all_df = pl.concat(dfs).with_columns(
        (pl.col("length_line_err_cm") / pl.col("gt_length_cm") * 100).alias("length_pct_err"),
    )
    return (all_df,)


@app.cell
def _(mo):
    mo.md("""
    ## Summary: median length % error by condition and dataset
    """)
    return


@app.cell
def _(all_df, pl):
    (
        all_df
        .group_by("condition", "dataset")
        .agg(
            pl.col("length_pct_err").median().alias("median_length_pct_err"),
            pl.col("length_pct_err").mean().alias("mean_length_pct_err"),
            pl.len().alias("n_samples"),
        )
        .sort("dataset", "condition")
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Length % error by genus

    Dumbbell chart: each genus is one row, two dots per condition. Green line = biorepo-train is better, orange = biorepo-val is better. Genera marked with `*` had samples in the biorepo training split.
    """)
    return


@app.cell
def _(all_df, alt, pl):
    # Genera that had samples in the biorepo training split (biorepo-train condition only)
    _train_genera = (
        all_df
        .filter(
            pl.col("dataset") == "biorepo",
            pl.col("condition") == "biorepo-train",
            pl.col("split") == "train",
        )
        .with_columns(pl.col("scientific_name").str.split(" ").list.first().alias("genus"))
        .get_column("genus")
        .unique()
        .to_list()
    )

    _genus_med = (
        all_df
        .filter(pl.col("length_pct_err").is_finite(), pl.col("dataset") == "biorepo")
        .with_columns(pl.col("scientific_name").str.split(" ").list.first().alias("genus"))
        .group_by("genus", "condition")
        .agg(pl.col("length_pct_err").median().alias("median_pct_err"))
        # Prepend "* " to genus label if it was in the biorepo training split
        .with_columns(
            pl.when(pl.col("genus").is_in(_train_genera))
            .then(pl.concat_str(pl.lit("* "), pl.col("genus")))
            .otherwise(pl.col("genus"))
            .alias("genus")
        )
    )

    # Sort by biorepo-val error (baseline, worst first)
    genus_order = (
        _genus_med
        .filter(pl.col("condition") == "biorepo-val")
        .sort("median_pct_err", descending=True)
        .get_column("genus")
        .to_list()
    )

    _w = _genus_med.pivot(on="condition", index="genus", values="median_pct_err")
    _m = _genus_med
    n_genera = _genus_med["genus"].n_unique()

    _lines = (
        alt.Chart(_w)
        .mark_rule(strokeWidth=2)
        .encode(
            y=alt.Y("genus:N", sort=genus_order, title="Genus"),
            x=alt.X("biorepo-val:Q", scale=alt.Scale(type="log"), title="Median Length % Error (val split)"),
            x2=alt.X2("biorepo-train:Q"),
            color=alt.condition(
                "datum['biorepo-train'] < datum['biorepo-val']",
                alt.value("green"),
                alt.value("orange"),
            ),
        )
    )
    _dots = (
        alt.Chart(_m)
        .mark_circle(size=60)
        .encode(
            y=alt.Y("genus:N", sort=genus_order, title=None),
            x=alt.X("median_pct_err:Q", scale=alt.Scale(type="log"), title=None),
            color=alt.Color("condition:N"),
        )
    )
    _threshold = (
        alt.Chart(_m)
        .mark_rule(color="red", strokeDash=[4, 4])
        .encode(x=alt.datum(0.3))
    )
    alt.layer(_lines, _dots, _threshold).properties(
        title="BioRepo: median length % error by genus (green = training helps, red = 0.3% target)",
        width=600,
        height=n_genera * 20,
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Per-genus median difference (train - val)

    Negative = biorepo-train is better.
    """)
    return


@app.cell
def _(all_df, pl):
    genus_med = (
        all_df
        .filter(pl.col("length_pct_err").is_finite())
        .with_columns(pl.col("scientific_name").str.split(" ").list.first().alias("genus"))
        .group_by("genus", "dataset", "condition")
        .agg(pl.col("length_pct_err").median().alias("median_pct_err"))
    )

    (
        genus_med
        .pivot(on="condition", index=["genus", "dataset"], values="median_pct_err")
        .with_columns(
            (pl.col("biorepo-train") - pl.col("biorepo-val")).alias("delta")
        )
        .sort("delta")
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Per-genus boxplot
    """)
    return


@app.cell
def _(all_df, mo, pl):
    _genera = (
        all_df
        .filter(pl.col("length_pct_err").is_finite(), pl.col("dataset") == "biorepo")
        .with_columns(pl.col("scientific_name").str.split(" ").list.first().alias("genus"))
        .get_column("genus")
        .unique()
        .sort()
        .to_list()
    )
    genus_dropdown = mo.ui.dropdown(_genera, value=_genera[0], label="Genus")
    genus_dropdown


@app.cell
def _(all_df, alt, genus_dropdown, pl):
    _genus_df = (
        all_df
        .filter(pl.col("length_pct_err").is_finite(), pl.col("dataset") == "biorepo")
        .with_columns(pl.col("scientific_name").str.split(" ").list.first().alias("genus"))
        .filter(pl.col("genus") == genus_dropdown.value)
        .select("condition", "length_pct_err")
    )

    alt.Chart(_genus_df).mark_boxplot(extent="min-max").encode(
        x=alt.X("condition:N", title="Condition"),
        y=alt.Y("length_pct_err:Q", title="Length % Error", scale=alt.Scale(type="log")),
        color="condition:N",
    ).properties(width=300, height=300, title=f"{genus_dropdown.value}: length % error by condition")


@app.cell
def _(mo):
    mo.md("""
    ## Data sparsity: training samples vs val error

    Each point is a genus. X = number of biorepo training samples, Y = median length % error on val split (biorepo-val condition as baseline).
    """)
    return


@app.cell
def _(all_df, alt, pathlib, pl):
    import btx.data.biorepo as biorepo

    _cfg = biorepo.Config(
        split="train",
        annotations=pathlib.Path("data/biorepo-formatted/annotations.json"),
        root=pathlib.Path("/fs/scratch/PAS2136/cain429/Subset-Exp"),
    )
    _train_counts = (
        biorepo.Dataset(_cfg).df
        .with_columns(pl.col("scientific_name").str.split(" ").list.first().alias("genus"))
        .group_by("genus")
        .agg(pl.len().alias("n_train"))
    )

    _delta = (
        all_df
        .filter(pl.col("length_pct_err").is_finite(), pl.col("dataset") == "biorepo")
        .with_columns(pl.col("scientific_name").str.split(" ").list.first().alias("genus"))
        .group_by("genus", "condition")
        .agg(pl.col("length_pct_err").median().alias("median_pct_err"))
        .pivot(on="condition", index="genus", values="median_pct_err")
        .with_columns((pl.col("biorepo-train") - pl.col("biorepo-val")).alias("delta"))
        .join(_train_counts, on="genus", how="left")
        .fill_null(0)
    )

    zero_line = alt.Chart(pl.DataFrame({"y": [0.0]})).mark_rule(color="gray", strokeDash=[4, 4]).encode(y="y:Q")

    scatter = alt.Chart(_delta).mark_circle(size=60, opacity=0.7).encode(
        x=alt.X("n_train:O", title="# BioRepo training samples"),
        y=alt.Y("delta:Q", title="Delta median length % error (train - val)"),
        tooltip=["genus:N", "n_train:Q", "delta:Q"],
    )

    alt.layer(zero_line, scatter).properties(width=500, height=400, title="Training samples vs error difference (negative = biorepo-train is better)")


if __name__ == "__main__":
    app.run()
