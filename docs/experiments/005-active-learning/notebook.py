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

    Three training runs with sigma=1, LR in {0.03, 0.1, 0.3}. Inference over Hawaii (all splits) + BioRepo (all splits).

    **Goal metric**: percent error = |pred_length - gt_length| / gt_length * 100. Target: consistently below 0.3% (collaborator's CV^2 threshold). This ensures prediction error is proportional to body size.
    """)
    return


@app.cell
def _(pathlib, pl):
    results_dpath = pathlib.Path(
        "/fs/ess/PAS2136/samuelstevens/beetle-traits/005-active-learning/results"
    )

    run_ids = {
        "gxdlfrgd": 0.03,
        "egqr97d7": 0.1,
        "v1t5i5tq": 0.3,
    }

    dfs = []
    for run_id, lr in run_ids.items():
        fpath = results_dpath / f"{run_id}_labeled.parquet"
        if not fpath.exists():
            continue
        df = pl.read_parquet(fpath).with_columns(
            pl.lit(lr).alias("learning_rate"),
            pl.lit(run_id).alias("run_id"),
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
def _(pathlib, pl):
    _results_dpath = pathlib.Path(
        "/fs/ess/PAS2136/samuelstevens/beetle-traits/005-active-learning/results"
    )

    _run_ids = {
        "gxdlfrgd": 0.03,
        "egqr97d7": 0.1,
        "v1t5i5tq": 0.3,
    }

    _dfs = []
    for _run_id, _lr in _run_ids.items():
        _fpath = _results_dpath / f"{_run_id}_unlabeled.parquet"
        if not _fpath.exists():
            continue
        _df = pl.read_parquet(_fpath).with_columns(
            pl.lit(_lr).alias("learning_rate"),
            pl.lit(_run_id).alias("run_id"),
        )
        _dfs.append(_df)

    unlabeled_df = pl.concat(_dfs)
    return (unlabeled_df,)


@app.cell
def _(mo, unlabeled_df):
    mo.md(f"""
    ## Unlabeled BioRepo data

    **{unlabeled_df.height:,}** total rows ({unlabeled_df.n_unique("beetle_id"):,} unique beetles x {unlabeled_df.n_unique("run_id")} runs).
    """)
    return


@app.cell
def _(all_df, alt, pl, unlabeled_df):
    _labeled = all_df.select("mean_entropy", "dataset", "learning_rate").with_columns(
        pl.lit("labeled").alias("source")
    )
    _unlabeled = (
        unlabeled_df
        .select("mean_entropy", "dataset", "learning_rate")
        .sample(n=min(5000, unlabeled_df.height), seed=0)
        .with_columns(pl.lit("unlabeled").alias("source"))
    )
    _ent_df = pl.concat([_labeled, _unlabeled]).with_columns(
        pl.col("learning_rate").cast(pl.String)
    )

    alt.Chart(_ent_df).mark_boxplot(extent="min-max").encode(
        x=alt.X("source:N", title=""),
        y=alt.Y("mean_entropy:Q", title="Mean Heatmap Entropy"),
        color="source:N",
        column=alt.Column("learning_rate:N", title="Learning Rate"),
    ).properties(width=200, height=300, title="Entropy: labeled vs unlabeled")
    return


@app.cell
def _(all_df, alt, pl, unlabeled_df):
    _labeled_ent = all_df.select("mean_entropy").with_columns(
        pl.lit("labeled").alias("source")
    )
    _unlabeled_ent = (
        unlabeled_df
        .filter(pl.col("run_id") == "egqr97d7")
        .select("mean_entropy")
        .sample(n=min(10000, unlabeled_df.height), seed=0)
        .with_columns(pl.lit("unlabeled").alias("source"))
    )
    _hist_df = pl.concat([_labeled_ent, _unlabeled_ent])

    alt.Chart(_hist_df).mark_bar(opacity=0.6).encode(
        x=alt.X(
            "mean_entropy:Q", bin=alt.Bin(maxbins=60), title="Mean Heatmap Entropy"
        ),
        y=alt.Y("count():Q", stack=None, title="Count"),
        color="source:N",
    ).properties(
        width=600, height=300, title="Entropy histogram (LR=0.1 run for unlabeled)"
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Combined embedding PCA (labeled + unlabeled)

    Project labeled and unlabeled BioRepo embeddings into a shared PCA space (LR=0.1 run). Color by source and by entropy.
    """)
    return


@app.cell
def _(all_df, alt, np, pl, unlabeled_df):
    _lr_run = "egqr97d7"
    _labeled_bio = all_df.filter(
        pl.col("dataset") == "biorepo", pl.col("run_id") == _lr_run
    )
    _unlabeled_bio = unlabeled_df.filter(pl.col("run_id") == _lr_run)

    # Sample unlabeled for plotting, but fit PCA on all data.
    _unlabeled_sample = _unlabeled_bio.sample(
        n=min(10000, _unlabeled_bio.height), seed=0
    )

    _emb_l = np.array(_labeled_bio["cls_embedding"].to_list())
    _emb_u = np.array(_unlabeled_sample["cls_embedding"].to_list())
    _emb_all = np.vstack([_emb_l, _emb_u])

    _centered = _emb_all - _emb_all.mean(axis=0)
    _, _, _vt = np.linalg.svd(_centered, full_matrices=False)
    _proj = _centered @ _vt[:2].T

    _combined = pl.concat([
        _labeled_bio.select("mean_entropy", "scientific_name").with_columns(
            pl.lit("labeled").alias("source")
        ),
        _unlabeled_sample.select("mean_entropy", "scientific_name").with_columns(
            pl.lit("unlabeled").alias("source")
        ),
    ]).with_columns(
        pl.Series("pc1", _proj[:, 0]),
        pl.Series("pc2", _proj[:, 1]),
        pl.col("scientific_name").str.split(" ").list.first().alias("genus"),
    )

    _base = (
        alt
        .Chart(_combined)
        .mark_circle(size=8, opacity=0.3)
        .encode(
            x=alt.X("pc1:Q", title="PC1"),
            y=alt.Y("pc2:Q", title="PC2"),
            tooltip=["genus:N", "source:N", "mean_entropy:Q"],
        )
    )

    _by_source = _base.encode(color="source:N").properties(
        width=400, height=400, title="Labeled vs unlabeled"
    )
    _by_entropy = _base.encode(
        color=alt.Color(
            "mean_entropy:Q", scale=alt.Scale(scheme="viridis"), title="Entropy"
        ),
    ).properties(width=400, height=400, title="Heatmap entropy")

    _by_genus = _base.encode(color=alt.Color("genus:N", legend=None)).properties(
        width=400, height=400, title="Genus"
    )

    _by_source | _by_entropy | _by_genus
    return


@app.cell
def _(mo, np, pl, unlabeled_df):
    _run = unlabeled_df.filter(pl.col("run_id") == "egqr97d7")
    _ent = _run["mean_entropy"].to_numpy()
    _q75 = np.percentile(_ent, 75)
    _q90 = np.percentile(_ent, 90)
    _q95 = np.percentile(_ent, 95)

    mo.md(f"""
    ## Unlabeled entropy summary (LR=0.1)

    | Statistic | Value |
    |---|---|
    | Min | {_ent.min():.3f} |
    | Median | {np.median(_ent):.3f} |
    | Mean | {_ent.mean():.3f} |
    | 75th pctl | {_q75:.3f} |
    | 90th pctl | {_q90:.3f} |
    | 95th pctl | {_q95:.3f} |
    | Max | {_ent.max():.3f} |

    High-entropy samples are candidates for active learning annotation.
    """)
    return


@app.cell
def _(alt, mo, np, pl, unlabeled_df):
    def cross_run_entropy_corr():
        from scipy.stats import rankdata

        run_ids = ["gxdlfrgd", "egqr97d7", "v1t5i5tq"]
        # Avoid '=' in column names -- Altair parses it as shorthand.
        lr_labels = {"gxdlfrgd": "lr003", "egqr97d7": "lr01", "v1t5i5tq": "lr03"}
        lr_titles = {"lr003": "LR=0.03", "lr01": "LR=0.1", "lr03": "LR=0.3"}

        # Only need beetle_id + mean_entropy per run. Drop heavy columns early.
        cols: dict[str, pl.Series] = {}
        beetle_ids = None
        for rid in mo.status.progress_bar(run_ids, title="Normalizing entropy"):
            run = unlabeled_df.filter(pl.col("run_id") == rid)
            if beetle_ids is None:
                beetle_ids = run["beetle_id"]
            ent = run["mean_entropy"].to_numpy()
            col = lr_labels[rid]
            cols[col] = pl.Series(col, rankdata(ent, method="average") / len(ent))

        # Build wide dataframe without joins -- all runs share the same beetle_id order.
        wide = pl.DataFrame({"beetle_id": beetle_ids, **cols})

        pairs = [("lr003", "lr01"), ("lr003", "lr03"), ("lr01", "lr03")]

        sample = wide.sample(n=min(5000, wide.height), seed=42)

        charts = []
        for xa, ya in mo.status.progress_bar(pairs, title="Building scatter plots"):
            r = np.corrcoef(wide[xa].to_numpy(), wide[ya].to_numpy())[0, 1]
            tx, ty = lr_titles[xa], lr_titles[ya]
            diag = (
                alt
                .Chart(pl.DataFrame({"x": [0, 1], "y": [0, 1]}))
                .mark_line(color="red", strokeDash=[4, 4])
                .encode(x="x:Q", y="y:Q")
            )
            scatter = (
                alt
                .Chart(sample)
                .mark_circle(size=6, opacity=0.2)
                .encode(
                    x=alt.X(f"{xa}:Q", title=tx, scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y(f"{ya}:Q", title=ty, scale=alt.Scale(domain=[0, 1])),
                )
                .properties(width=300, height=300, title=f"{tx} vs {ty} (r={r:.3f})")
            )
            charts.append(scatter + diag)

        return charts[0] | charts[1] | charts[2]

    mo.vstack([
        mo.md(r"""
    ## Cross-run entropy correlation

    Pairwise scatter plots of normalized entropy (percentile rank) across the three training runs for all unlabeled beetles. Points should cluster along the diagonal if runs agree on which beetles are uncertain. Off-diagonal scatter means one run's entropy signal is decorrelated, which would make our normalize-then-min aggregation noisy.
    """),
        cross_run_entropy_corr(),
    ])
    return


@app.cell
def _(mo, np, pathlib, pl, unlabeled_df):
    def embedding_umap_selection_overlay():
        results_dpath = pathlib.Path(
            "/fs/ess/PAS2136/samuelstevens/beetle-traits/005-active-learning/results"
        )

        # Read rank.py outputs: exact priority set and selected groups.
        all_unlabeled = pl.read_csv(results_dpath / "round1_all_unlabeled.csv")
        selected_groups = set(
            pl.read_csv(results_dpath / "round1_groups.csv")[
                "group_img_basename"
            ].to_list()
        )

        is_priority = all_unlabeled["is_priority"].to_numpy()
        group_basenames = all_unlabeled["group_img_basename"].to_list()
        n = len(all_unlabeled)

        # 2x2: priority yes/no x in-selected-group yes/no.
        categories = np.array([
            "priority+selected"
            if pri and sel
            else "priority+not selected"
            if pri
            else "not priority+selected"
            if sel
            else "other"
            for pri, sel in mo.status.progress_bar(
                zip(is_priority, (g in selected_groups for g in group_basenames)),
                title="Categorizing",
                total=n,
            )
        ])

        # Subsample to ~10K total. Keep all non-"other", subsample "other".
        rng = np.random.default_rng(42)
        highlight_i = np.where(categories != "other")[0]
        other_i = np.where(categories == "other")[0]
        n_oth = min(len(other_i), 10_000 - len(highlight_i))
        keep_i = np.sort(
            np.concatenate([
                highlight_i,
                rng.choice(other_i, size=n_oth, replace=False),
            ])
        )

        # CLS embeddings from one run (frozen backbone, all identical).
        ref_run = unlabeled_df.filter(pl.col("run_id") == "gxdlfrgd")
        emb = np.array(ref_run["cls_embedding"].to_list(), dtype=np.float32)[keep_i]
        keep_categories = categories[keep_i]

        # UMAP to 2 dims (t-SNE hangs on this node due to BLAS threading issues).
        import umap

        proj = umap.UMAP(n_components=2, random_state=42, n_neighbors=30).fit_transform(
            emb
        )

        import matplotlib.pyplot as plt

        draw_order = [
            "other",
            "not priority+selected",
            "priority+not selected",
            "priority+selected",
        ]
        cat_colors = {
            "other": "#cccccc",
            "priority+not selected": "#ff8c00",
            "not priority+selected": "#6baed6",
            "priority+selected": "#e00000",
        }
        cat_sizes = {
            "other": 4,
            "priority+not selected": 8,
            "not priority+selected": 8,
            "priority+selected": 8,
        }

        def plot_scatter(ax, x, y, cats, title):
            for cat in draw_order:
                mask = cats == cat
                if not mask.any():
                    continue
                ax.scatter(
                    x[mask],
                    y[mask],
                    s=cat_sizes[cat],
                    c=cat_colors[cat],
                    alpha=0.3,
                    label=f"{cat} ({mask.sum()})",
                )
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.set_title(title)
            ax.legend(markerscale=3, fontsize=8)

        fig, (ax_full, ax_zoom) = plt.subplots(
            1, 2, figsize=(16, 8), dpi=100, layout="constrained"
        )

        # Left: full view.
        plot_scatter(ax_full, proj[:, 0], proj[:, 1], keep_categories, "Full embedding")

        # Right: re-run UMAP on the dense central cluster.
        zoom_mask = (
            (proj[:, 0] > 4) & (proj[:, 0] < 13) & (proj[:, 1] > -3) & (proj[:, 1] < 5)
        )
        zoom_emb = emb[zoom_mask]
        zoom_cats = keep_categories[zoom_mask]
        zoom_proj = umap.UMAP(
            n_components=2, random_state=42, n_neighbors=15
        ).fit_transform(zoom_emb)
        plot_scatter(
            ax_zoom,
            zoom_proj[:, 0],
            zoom_proj[:, 1],
            zoom_cats,
            f"Dense cluster ({zoom_mask.sum()} pts)",
        )

        out_fpath = pathlib.Path(
            "docs/experiments/005-active-learning/results/umap-selection-overlay.png"
        )
        out_fpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_fpath)

        n_pri = int(is_priority.sum())
        caption = mo.md(f"""
    ## Embedding UMAP with selection overlay

    Using exact priority set from rank.py ({n_pri} beetles). Gray = not priority, orange = priority but not selected, blue = not priority but in selected group (free riders), red = priority and selected.

    Saved to `{out_fpath}`.
    """)

        return mo.vstack([caption, fig])

    embedding_umap_selection_overlay()
    return


@app.cell
def _(alt, mo, np, pathlib, pl):
    def cluster_coverage():
        results_dpath = pathlib.Path(
            "/fs/ess/PAS2136/samuelstevens/beetle-traits/005-active-learning/results"
        )

        # Read rank.py outputs (no K-means or entropy needed in notebook).
        all_unlabeled = pl.read_csv(results_dpath / "round1_all_unlabeled.csv")
        selected_groups = set(
            pl.read_csv(results_dpath / "round1_groups.csv")[
                "group_img_basename"
            ].to_list()
        )

        is_priority = all_unlabeled["is_priority"].to_numpy()
        group_basenames = all_unlabeled["group_img_basename"].to_list()

        # Use first run's cluster labels (frozen backbone, all runs identical).
        labels = np.array([
            int(s.split(",")[0]) for s in all_unlabeled["cluster_ids"].to_list()
        ])
        k = labels.max() + 1

        # Per-cluster: fraction of priority beetles that ended up in selected groups.
        rows = []
        for cid in range(k):
            in_cluster = labels == cid
            pri_in_cluster = in_cluster & is_priority
            n_pri = pri_in_cluster.sum()
            if n_pri == 0:
                rows.append({
                    "cluster": cid,
                    "n_priority": 0,
                    "n_selected": 0,
                    "frac_selected": 0.0,
                    "cluster_size": int(in_cluster.sum()),
                })
                continue
            n_sel = sum(
                1
                for i in np.where(pri_in_cluster)[0]
                if group_basenames[i] in selected_groups
            )
            rows.append({
                "cluster": cid,
                "n_priority": int(n_pri),
                "n_selected": n_sel,
                "frac_selected": n_sel / n_pri,
                "cluster_size": int(in_cluster.sum()),
            })

        df = pl.DataFrame(rows).sort("frac_selected", descending=True)

        chart = (
            alt
            .Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("cluster:O", sort="-y", title="Cluster ID"),
                y=alt.Y(
                    "frac_selected:Q",
                    title="Fraction of priority beetles in selected groups",
                ),
                color=alt.Color(
                    "n_priority:Q",
                    scale=alt.Scale(scheme="oranges"),
                    title="# priority",
                ),
                tooltip=[
                    "cluster:O",
                    "n_priority:Q",
                    "n_selected:Q",
                    "frac_selected:Q",
                    "cluster_size:Q",
                ],
            )
            .properties(
                width=700,
                height=300,
                title="Cluster coverage: priority beetles selected per cluster",
            )
        )

        n_zero = df.filter(pl.col("frac_selected") == 0).height
        n_full = df.filter(pl.col("frac_selected") == 1.0).height
        median_frac = df.filter(pl.col("n_priority") > 0)["frac_selected"].median()

        caption = mo.md(f"""
    ## Cluster coverage

    For each of {k} K-means clusters (from rank.py): what fraction of that cluster's priority beetles ended up in selected groups? Uniform coverage (~10-30% per cluster) means diversity is working. Clusters with 0% coverage are being ignored; 100% means over-concentration.

    - Clusters with 0% coverage: **{n_zero}**
    - Clusters with 100% coverage: **{n_full}**
    - Median coverage (non-empty clusters): **{median_frac:.1%}**
    """)

        return mo.vstack([caption, chart])

    cluster_coverage()
    return


if __name__ == "__main__":
    app.run()
