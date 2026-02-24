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
    """)
    return


@app.cell
def _(pathlib, pl):
    results_dpath = pathlib.Path("docs/experiments/005-active-learning/results")

    run_ids = {
        "gxdlfrgd": 0.03,
        "egqr97d7": 0.1,
        "v1t5i5tq": 0.3,
    }

    dfs = []
    for run_id, lr in run_ids.items():
        for split in ["train", "val"]:
            fpath = results_dpath / f"{run_id}_{split}.parquet"
            if not fpath.exists():
                continue
            df = pl.read_parquet(fpath).with_columns(
                pl.lit(lr).alias("learning_rate"),
                pl.lit(run_id).alias("run_id"),
                pl.lit(split).alias("split"),
            )
            dfs.append(df)

    all_df = pl.concat(dfs).with_columns(
        (pl.col("length_line_err_cm") / pl.col("gt_length_cm") * 100).alias(
            "length_pct_err"
        ),
        (pl.col("width_line_err_cm") / pl.col("gt_width_cm") * 100).alias(
            "width_pct_err"
        ),
    )
    return (all_df,)


@app.cell
def _(mo):
    mo.md("""
    ## Per-run, per-dataset summary
    """)
    return


@app.cell
def _(all_df, pl):
    summary = (
        all_df
        .group_by("learning_rate", "dataset")
        .agg(
            pl.col("length_pct_err").median().alias("median_length_pct_err"),
            pl.col("length_pct_err").mean().alias("mean_length_pct_err"),
            pl.col("width_pct_err").median().alias("median_width_pct_err"),
            pl.col("length_line_err_cm").median().alias("median_length_err_cm"),
            pl.col("width_line_err_cm").median().alias("median_width_err_cm"),
            pl.col("mean_entropy").median().alias("median_entropy"),
            pl.len().alias("n_samples"),
        )
        .sort("dataset", "learning_rate")
    )
    summary
    return


@app.cell
def _(mo):
    mo.md("""
    ## Percent error distribution by dataset and learning rate
    """)
    return


@app.cell
def _(all_df, alt, pl):
    pct_df = (
        all_df
        .select("dataset", "learning_rate", "length_pct_err", "width_pct_err")
        .unpivot(
            on=["length_pct_err", "width_pct_err"],
            index=["dataset", "learning_rate"],
            variable_name="metric",
            value_name="pct_err",
        )
        .with_columns(
            pl.col("metric").str.replace("_pct_err", ""),
            pl.col("learning_rate").cast(pl.String),
        )
    )

    alt.Chart(pct_df).mark_boxplot(extent="min-max").encode(
        x=alt.X("learning_rate:N", title="Learning Rate"),
        y=alt.Y("pct_err:Q", title="Percent Error (%)", scale=alt.Scale(type="log")),
        color="learning_rate:N",
        column=alt.Column("dataset:N"),
        row=alt.Row("metric:N"),
    ).properties(width=250, height=200)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Absolute error distributions by dataset and learning rate
    """)
    return


@app.cell
def _(all_df, alt, pl):
    err_df = (
        all_df
        .select("dataset", "learning_rate", "width_line_err_cm", "length_line_err_cm")
        .unpivot(
            on=["width_line_err_cm", "length_line_err_cm"],
            index=["dataset", "learning_rate"],
            variable_name="metric",
            value_name="error_cm",
        )
        .with_columns(
            pl.col("metric").str.replace("_line_err_cm", ""),
            pl.col("learning_rate").cast(pl.String),
        )
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
    mo.md("""
    ## Per-species percent error (all runs combined)
    """)
    return


@app.cell
def _(all_df, pl):
    species_err = (
        all_df
        .group_by("scientific_name", "dataset")
        .agg(
            pl.col("length_pct_err").median().alias("median_length_pct_err"),
            pl.col("length_pct_err").mean().alias("mean_length_pct_err"),
            pl.col("length_pct_err").max().alias("max_length_pct_err"),
            pl.col("gt_length_cm").median().alias("median_gt_length_cm"),
            pl.col("mean_entropy").median().alias("median_entropy"),
            pl.len().alias("n_samples"),
        )
        .sort("median_length_pct_err", descending=True)
    )
    species_err
    return


@app.cell
def _(mo):
    mo.md("""
    ## Uncertainty vs percent error
    """)
    return


@app.cell
def _(all_df, alt, pl):
    scatter_df = all_df.select(
        "mean_entropy", "length_pct_err", "dataset", "learning_rate"
    ).with_columns(pl.col("learning_rate").cast(pl.String))

    alt.Chart(scatter_df).mark_circle(size=10, opacity=0.3).encode(
        x=alt.X("mean_entropy:Q", title="Mean Heatmap Entropy"),
        y=alt.Y(
            "length_pct_err:Q",
            title="Length Percent Error (%)",
            scale=alt.Scale(type="log"),
        ),
        color="dataset:N",
        column=alt.Column("learning_rate:N"),
    ).properties(width=300, height=300)
    return


@app.cell
def _(all_df, mo, np, pl):
    finite = all_df.filter(pl.col("length_pct_err").is_finite())
    n_total = len(finite)
    n_below = finite.filter(pl.col("length_pct_err") < 0.3).height
    pct_below = n_below / n_total * 100 if n_total > 0 else 0

    gt = np.array(all_df["gt_coords_px"].to_list())
    n_finite_gt = np.isfinite(gt).any(axis=1).sum()

    mo.md(f"""
    ## Target metric check

    - Samples with finite gt_coords: **{n_finite_gt} / {len(gt)}**
    - Samples with length percent error < 0.3%: **{n_below} / {n_total}** ({pct_below:.1f}%)
    """)
    return


@app.cell
def _(all_df, alt, pl):
    genus_df = (
        all_df
        .filter(pl.col("length_pct_err").is_finite())
        .with_columns(
            pl.col("scientific_name").str.split(" ").list.first().alias("genus")
        )
        .select("genus", "dataset", "length_pct_err")
    )

    genus_order = (
        genus_df
        .group_by("genus")
        .agg(pl.col("length_pct_err").median().alias("median_pct_err"))
        .sort("median_pct_err", descending=True)
        .get_column("genus")
        .to_list()
    )

    threshold = (
        alt
        .Chart(pl.DataFrame({"y": [0.3]}))
        .mark_rule(color="red", strokeDash=[4, 4])
        .encode(y="y:Q")
    )

    boxes = (
        alt
        .Chart(genus_df)
        .mark_boxplot(opacity=0.5, ticks=True)
        .encode(
            x=alt.X("genus:N", sort=genus_order, title="Genus"),
            y=alt.Y(
                "length_pct_err:Q",
                title="Length Percent Error (%)",
                scale=alt.Scale(type="log"),
            ),
            color="dataset:N",
        )
    )

    (boxes + threshold).properties(
        width=800, height=400, title="Length percent error by genus (red = 0.3% target)"
    )
    return


@app.cell
def _(all_df, np, pl):
    _parts = []
    for _ds in all_df["dataset"].unique().to_list():
        _sub = all_df.filter(pl.col("dataset") == _ds)
        _emb = np.array(_sub["cls_embedding"].to_list())
        _centered = _emb - _emb.mean(axis=0)
        _, _, _vt = np.linalg.svd(_centered, full_matrices=False)
        _proj = _centered @ _vt[:2].T
        _parts.append(
            _sub.select(
                "dataset", "length_pct_err", "mean_entropy", "scientific_name"
            ).with_columns(
                pl.Series("pc1", _proj[:, 0]),
                pl.Series("pc2", _proj[:, 1]),
                pl.col("scientific_name").str.split(" ").list.first().alias("genus"),
            )
        )
    pca_df = pl.concat(_parts)
    return (pca_df,)


@app.cell
def _(mo):
    mo.md("""
    ## CLS embedding PCA

    Rows: dataset. Columns: percent error, heatmap entropy, genus.
    """)
    return


@app.cell
def _(alt, pca_df):
    _base = (
        alt
        .Chart(pca_df)
        .mark_circle(size=12, opacity=0.4)
        .encode(
            x=alt.X("pc1:Q", title="PC1"),
            y=alt.Y("pc2:Q", title="PC2"),
            tooltip=["genus:N", "dataset:N", "length_pct_err:Q", "mean_entropy:Q"],
        )
    )

    _err = (
        _base
        .encode(
            color=alt.Color(
                "length_pct_err:Q",
                scale=alt.Scale(scheme="inferno", type="log"),
                title="% Error",
            ),
            row=alt.Row("dataset:N"),
        )
        .properties(title="Percent error")
        .resolve_scale(x="independent", y="independent")
    )

    _entropy = (
        _base
        .encode(
            color=alt.Color(
                "mean_entropy:Q", scale=alt.Scale(scheme="viridis"), title="Entropy"
            ),
            row=alt.Row("dataset:N"),
        )
        .properties(title="Heatmap entropy")
        .resolve_scale(x="independent", y="independent")
    )

    _genus = (
        _base
        .encode(
            color=alt.Color("genus:N", legend=None),
            row=alt.Row("dataset:N"),
        )
        .properties(title="Genus")
        .resolve_scale(x="independent", y="independent")
    )

    _err | _entropy | _genus
    return


@app.cell
def _(mo):
    mo.md("""
    ## Example predictions

    2 Hawaii + 2 BioRepo samples at each error quantile (0%, 25%, 50%, 75%, 100%) from the LR=0.1 run. Cyan/magenta = GT width/length, lime/yellow = predicted width/length.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    from PIL import Image

    return Image, plt


@app.cell
def _(Image, all_df, np, pl, plt):
    # Use one run to avoid duplicate beetle_ids. Pick 2 hawaii + 2 biorepo per quantile.
    _base = all_df.filter(
        pl.col("length_pct_err").is_finite(), pl.col("run_id") == "egqr97d7"
    )
    _quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]
    _labels = ["0%", "25%", "50%", "75%", "100%"]
    _picks = []
    for _ds in ["hawaii", "biorepo"]:
        _sorted = _base.filter(pl.col("dataset") == _ds).sort("length_pct_err")
        _nd = len(_sorted)
        for _q in _quantiles:
            _qi = min(round(_q * (_nd - 1)), _nd - 2)
            _picks.append(_sorted.row(_qi, named=True))
            _picks.append(_sorted.row(_qi + 1, named=True))

    # Interleave so each row has [hawaii_0, hawaii_1, biorepo_0, biorepo_1].
    _n_per_ds = len(_quantiles) * 2
    _grid = []
    for _row_i in range(len(_quantiles)):
        _hi = _row_i * 2
        _grid.append(_picks[_hi])
        _grid.append(_picks[_hi + 1])
        _grid.append(_picks[_n_per_ds + _hi])
        _grid.append(_picks[_n_per_ds + _hi + 1])

    fig, axes = plt.subplots(5, 4, figsize=(18, 22), dpi=60, layout="constrained")

    for _idx, (ax, row) in enumerate(zip(axes.reshape(-1), _grid)):
        with Image.open(row["img_fpath"]) as fd:
            ax.imshow(np.array(fd.convert("RGB")))

        _gt = np.array(row["gt_coords_px"]).reshape(2, 2, 2)
        _pred = np.array(row["pred_coords_px"]).reshape(2, 2, 2)

        for line_i, (gt_color, pred_color) in enumerate([
            ("cyan", "lime"),
            ("magenta", "yellow"),
        ]):
            (gx0, gy0), (gx1, gy1) = _gt[line_i]
            ax.plot(
                [gx0, gx1], [gy0, gy1], "o-", color=gt_color, linewidth=2, markersize=4
            )
            (px0, py0), (px1, py1) = _pred[line_i]
            ax.plot(
                [px0, px1],
                [py0, py1],
                "x--",
                color=pred_color,
                linewidth=2,
                markersize=5,
            )

        ax.set_title(
            f"{row['scientific_name']} ({row['beetle_id']})\n{row['dataset']} | {row['length_pct_err']:.2f}%",
            fontsize=8,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        if _idx % 4 == 0:
            ax.set_ylabel(_labels[_idx // 4], fontsize=12, rotation=0, labelpad=30)

    fig.suptitle("cyan/magenta = GT width/length, lime/yellow = predicted", fontsize=11)
    fig
    return


@app.cell
def _(all_df, mo, np, pl):
    _bug = all_df.filter(
        pl.col("beetle_id") == "NEON.BET.D20.003085",
        pl.col("run_id") == "egqr97d7",
    )
    _rows = []
    for _i in range(_bug.height):
        _r = _bug.row(_i, named=True)
        _gt = np.array(_r["gt_coords_px"]).reshape(2, 2, 2)
        _pred = np.array(_r["pred_coords_px"]).reshape(2, 2, 2)
        _gt_w_len = np.linalg.norm(_gt[0, 0] - _gt[0, 1])
        _gt_l_len = np.linalg.norm(_gt[1, 0] - _gt[1, 1])
        _pred_w_len = np.linalg.norm(_pred[0, 0] - _pred[0, 1])
        _pred_l_len = np.linalg.norm(_pred[1, 0] - _pred[1, 1])
        _rows.append(f"""
**split={_r["split"]}**

| | GT | Pred | Abs Err (px) | Recomputed % Err |
|---|---|---|---|---|
| width endpoints | {_gt[0].tolist()} | {_pred[0].tolist()} | | |
| width line len (px) | {_gt_w_len:.1f} | {_pred_w_len:.1f} | {abs(_pred_w_len - _gt_w_len):.1f} | {abs(_pred_w_len - _gt_w_len) / _gt_w_len * 100:.2f}% |
| length endpoints | {_gt[1].tolist()} | {_pred[1].tolist()} | | |
| length line len (px) | {_gt_l_len:.1f} | {_pred_l_len:.1f} | {abs(_pred_l_len - _gt_l_len):.1f} | {abs(_pred_l_len - _gt_l_len) / _gt_l_len * 100:.2f}% |
| **Parquet values** | gt_length_cm={_r["gt_length_cm"]:.4f} | | length_line_err_cm={_r["length_line_err_cm"]:.4f} | length_pct_err={_r["length_line_err_cm"] / _r["gt_length_cm"] * 100:.2f}% |
| | gt_width_cm={_r["gt_width_cm"]:.4f} | | width_line_err_cm={_r["width_line_err_cm"]:.4f} | width_pct_err={_r["width_line_err_cm"] / _r["gt_width_cm"] * 100:.2f}% |
""")

    mo.md(f"""
## Debug: NEON.BET.D20.003085

{"".join(_rows)}
""")
    return


if __name__ == "__main__":
    app.run()
