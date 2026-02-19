import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
def _():
    import copy
    import pathlib

    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    import btx
    import btx.data
    import btx.objectives.heatmap
    return Image, btx, copy, jnp, np, pathlib, plt


@app.cell
def _(Image, btx, np, pathlib):
    def load_samples(cfg, *, limit: int = 6):
        """Load raw samples from a dataset config, skipping broken ones."""
        try:
            ds = cfg.dataset(cfg)
        except (AssertionError, FileNotFoundError, NotImplementedError) as err:
            print(err)
            return []

        samples = []
        for idx in range(len(ds)):
            if len(samples) >= limit:
                break
            try:
                sample = dict(ds[idx])
                with Image.open(sample["img_fpath"]) as fd:
                    sample["img"] = fd.convert("RGB")
                points_px = sample["points_px"]
                msg = f"Expected points_px shape (2, 2, 2), got {np.asarray(points_px).shape}"
                assert np.asarray(points_px).shape == (2, 2, 2), msg
            except (AssertionError, FileNotFoundError, OSError):
                continue
            samples.append(sample)
        return samples

    HAWAII_HF_ROOT = pathlib.Path(
        "/fs/ess/PAS2136/samuelstevens/datasets/hawaii-beetles"
    )
    BIOREPO_ROOT = pathlib.Path("/fs/scratch/PAS2136/cain429/Subset-Exp")
    BIOREPO_ANN = pathlib.Path(
        "/fs/scratch/PAS2136/cain429/biorepo-formatted/annotations.json"
    )

    dataset_rows = [
        (
            "Hawaii",
            load_samples(
                btx.data.HawaiiConfig(
                    split="train",
                    include_polylines=False,
                    hf_root=HAWAII_HF_ROOT,
                )
            ),
        ),
        ("BeetlePalooza", load_samples(btx.data.BeetlePaloozaConfig())),
        (
            "BioRepo",
            load_samples(
                btx.data.BioRepoConfig(
                    split="val",
                    root=BIOREPO_ROOT,
                    annotations=BIOREPO_ANN,
                )
            ),
        ),
    ]
    return (dataset_rows,)


@app.cell
def _(btx, copy, jnp, np):
    def run_pre_heatmap_pipeline(raw_sample: dict[str, object]) -> dict[str, object]:
        """Create model-space image and point targets before heatmap generation."""
        aug_cfg = btx.data.transforms.AugmentConfig(
            go=False, crop=False, normalize=False
        )
        tfms = btx.data.transforms.make_transforms(aug_cfg, is_train=True)
        sample = copy.deepcopy(raw_sample)
        for tfm in tfms:
            sample = tfm.map(sample)
        msg = "Expected keys 'img' and 'tgt' after pre-heatmap pipeline."
        assert "img" in sample and "tgt" in sample, msg
        msg = f"Expected tgt shape (2, 2, 2), got {np.asarray(sample['tgt']).shape}"
        assert np.asarray(sample["tgt"]).shape == (2, 2, 2), msg
        return sample

    def make_heatmap_logits(
        tgt_l22: np.ndarray, *, sigma: float
    ) -> tuple[btx.objectives.heatmap.Config, jnp.ndarray, jnp.ndarray]:
        """Build heatmap config, targets, and stable logits from endpoint coords."""
        heatmap_cfg = btx.objectives.heatmap.Config(
            image_size=256, heatmap_size=64, sigma=sigma
        )
        tgt_j = jnp.asarray(tgt_l22, dtype=jnp.float32)
        msg = f"Expected tgt shape (2, 2, 2), got {tgt_j.shape}"
        assert tgt_j.shape == (2, 2, 2), msg
        heatmap_tgt = btx.objectives.heatmap.make_targets(tgt_j, cfg=heatmap_cfg)
        msg = f"Expected heatmap target shape (4, 64, 64), got {heatmap_tgt.shape}"
        assert heatmap_tgt.shape == (4, 64, 64), msg
        logits_chw = jnp.log(jnp.maximum(heatmap_tgt, heatmap_cfg.eps))
        return heatmap_cfg, heatmap_tgt, logits_chw

    def decode_argmax(logits_chw: jnp.ndarray, *, cfg) -> np.ndarray:
        """Decode endpoint coordinates by hard argmax in heatmap space."""
        flat_i = jnp.argmax(jnp.reshape(logits_chw, (4, -1)), axis=1)
        y_i = flat_i // cfg.heatmap_size
        x_i = flat_i % cfg.heatmap_size
        points_hm_n2 = jnp.stack(
            [x_i.astype(jnp.float32), y_i.astype(jnp.float32)],
            axis=1,
        )
        points_img_n2 = btx.objectives.heatmap.heatmap_to_image_udp(
            points_hm_n2, cfg=cfg
        )
        return np.asarray(jnp.reshape(points_img_n2, (2, 2, 2)))

    def get_heatmap_views(
        prepped_sample: dict[str, object], *, sigma: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate heatmaps from `tgt` and decode points back to image space."""
        tgt_l22 = np.asarray(prepped_sample["tgt"], dtype=np.float32)
        heatmap_cfg, heatmap_tgt, logits_chw = make_heatmap_logits(tgt_l22, sigma=sigma)
        pred_l22 = btx.objectives.heatmap.heatmaps_to_coords(
            logits_chw, cfg=heatmap_cfg
        )
        msg = f"Expected decoded points shape (2, 2, 2), got {pred_l22.shape}"
        assert pred_l22.shape == (2, 2, 2), msg
        return np.asarray(heatmap_tgt), np.asarray(pred_l22)

    def plot_lines(
        ax,
        points_l22: np.ndarray,
        *,
        width_color: str = "cyan",
        length_color: str = "magenta",
        marker: str = "o",
        linestyle: str = "-",
        linewidth: float = 1.8,
        alpha: float = 1.0,
    ) -> None:
        """Plot width and length endpoint lines on an axis."""
        (x0, y0), (x1, y1) = points_l22[0]
        ax.plot(
            [x0, x1],
            [y0, y1],
            marker=marker,
            linestyle=linestyle,
            color=width_color,
            linewidth=linewidth,
            markersize=4,
            alpha=alpha,
        )
        (x0, y0), (x1, y1) = points_l22[1]
        ax.plot(
            [x0, x1],
            [y0, y1],
            marker=marker,
            linestyle=linestyle,
            color=length_color,
            linewidth=linewidth,
            markersize=4,
            alpha=alpha,
        )
    return (
        decode_argmax,
        get_heatmap_views,
        make_heatmap_logits,
        plot_lines,
        run_pre_heatmap_pipeline,
    )


@app.cell
def _(
    dataset_rows,
    get_heatmap_views,
    np,
    plot_lines,
    plt,
    run_pre_heatmap_pipeline,
):
    sigma_values = [1.0, 2.0, 3.0]
    n_per_dataset = 2

    rows = []
    for dataset_name, samples in dataset_rows:
        for sample in samples[:n_per_dataset]:
            rows.append((dataset_name, sample))

    assert rows, "No samples were available from the configured datasets."

    n_rows = len(rows)
    n_cols = 2 + len(sigma_values)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.2 * n_cols, 2.9 * n_rows),
        layout="constrained",
        dpi=80,
    )
    if n_rows == 1:
        axes = axes[None, :]

    axes[0, 0].set_title("raw + points_px", fontsize=10)
    axes[0, 1].set_title("before heatmap: img + tgt", fontsize=10)
    for col_i, sigma in enumerate(sigma_values, start=2):
        axes[0, col_i].set_title(f"after heatmap (sigma={sigma:g})", fontsize=10)

    prev_dataset = None
    for row_i, (dataset_name, raw_sample) in enumerate(rows):
        raw_img = np.asarray(raw_sample["img"], dtype=np.float32) / 255.0
        raw_points = np.asarray(raw_sample["points_px"], dtype=np.float32)
        prepped = run_pre_heatmap_pipeline(raw_sample)
        pre_img = np.asarray(prepped["img"], dtype=np.float32)
        tgt = np.asarray(prepped["tgt"], dtype=np.float32)

        ax = axes[row_i, 0]
        ax.imshow(raw_img)
        plot_lines(ax, raw_points)
        if dataset_name != prev_dataset:
            ax.set_ylabel(dataset_name, fontsize=10)
            prev_dataset = dataset_name
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)

        ax = axes[row_i, 1]
        ax.imshow(np.clip(pre_img, 0.0, 1.0))
        plot_lines(ax, tgt)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)

        for col_i, sigma in enumerate(sigma_values, start=2):
            heatmap_tgt, pred_l22 = get_heatmap_views(prepped, sigma=sigma)
            heat_max = np.max(heatmap_tgt, axis=0)
            mean_err_px = float(np.linalg.norm(pred_l22 - tgt, axis=2).mean())

            ax = axes[row_i, col_i]
            ax.imshow(np.clip(pre_img, 0.0, 1.0))
            ax.imshow(
                heat_max,
                cmap="magma",
                alpha=0.55,
                interpolation="bilinear",
                extent=(-0.5, 255.5, 255.5, -0.5),
            )
            plot_lines(
                ax,
                pred_l22,
                width_color="lime",
                length_color="yellow",
                marker="x",
                linestyle="--",
                linewidth=1.4,
            )
            if row_i > 0:
                ax.set_title(f"decode err={mean_err_px:.2f}px", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[:].set_visible(False)

    fig.suptitle(
        "Heatmap Objective Demo: cyan/magenta=GT lines, lime/yellow=decoded from target heatmaps",
        fontsize=12,
    )
    fig
    return


@app.cell
def _():
    import marimo as _mo

    _mo.md(
        """
        ## Color Legend

        - `cyan`: width ground-truth line/endpoints
        - `magenta`: length ground-truth line/endpoints
        - `lime`: width softargmax prediction
        - `yellow`: length softargmax prediction
        - `orange`: width hard argmax prediction
        - `red`: length hard argmax prediction
        - `white` connector segment: distance between softargmax and hard argmax endpoint predictions
        - `magma` heat overlay: maximum intensity over the 4 target heatmap channels
        """
    )
    return


@app.cell
def _(
    btx,
    dataset_rows,
    decode_argmax,
    jnp,
    make_heatmap_logits,
    np,
    plt,
    run_pre_heatmap_pipeline,
):
    _cmp_sigma_large = 6.0
    _cmp_n_per_dataset = 2

    _cmp_rows = []
    for _cmp_dataset_name, _cmp_samples in dataset_rows:
        for _cmp_sample in _cmp_samples[:_cmp_n_per_dataset]:
            _cmp_rows.append((_cmp_dataset_name, _cmp_sample))
    assert _cmp_rows, "No samples were available from the configured datasets."

    _cmp_endpoint_soft_px = []
    _cmp_endpoint_hard_px = []
    _cmp_endpoint_soft_cm = []
    _cmp_endpoint_hard_cm = []
    _cmp_endpoint_edge_dist_hm = []
    _cmp_sample_records = []

    for _cmp_dataset_name, _cmp_raw_sample in _cmp_rows:
        _cmp_prepped = run_pre_heatmap_pipeline(_cmp_raw_sample)
        _cmp_tgt_l22 = np.asarray(_cmp_prepped["tgt"], dtype=np.float32)
        (
            _cmp_heatmap_cfg,
            _,
            _cmp_logits_chw,
        ) = make_heatmap_logits(_cmp_tgt_l22, sigma=_cmp_sigma_large)
        _cmp_pred_soft_l22 = np.asarray(
            btx.objectives.heatmap.heatmaps_to_coords(
                _cmp_logits_chw, cfg=_cmp_heatmap_cfg
            )
        )
        _cmp_pred_hard_l22 = decode_argmax(_cmp_logits_chw, cfg=_cmp_heatmap_cfg)

        _cmp_err_soft_l2 = np.linalg.norm(_cmp_pred_soft_l22 - _cmp_tgt_l22, axis=2)
        _cmp_err_hard_l2 = np.linalg.norm(_cmp_pred_hard_l22 - _cmp_tgt_l22, axis=2)
        _cmp_endpoint_soft_px.extend(_cmp_err_soft_l2.reshape(-1).tolist())
        _cmp_endpoint_hard_px.extend(_cmp_err_hard_l2.reshape(-1).tolist())

        _cmp_tgt_n2 = jnp.asarray(_cmp_tgt_l22.reshape(4, 2), dtype=jnp.float32)
        _cmp_tgt_hm_n2 = np.asarray(
            btx.objectives.heatmap.image_to_heatmap_udp(
                _cmp_tgt_n2, cfg=_cmp_heatmap_cfg
            )
        )
        _cmp_edge_dist_hm = np.minimum.reduce([
            _cmp_tgt_hm_n2[:, 0],
            _cmp_tgt_hm_n2[:, 1],
            (_cmp_heatmap_cfg.heatmap_size - 1.0) - _cmp_tgt_hm_n2[:, 0],
            (_cmp_heatmap_cfg.heatmap_size - 1.0) - _cmp_tgt_hm_n2[:, 1],
        ])
        _cmp_endpoint_edge_dist_hm.extend(_cmp_edge_dist_hm.tolist())

        _cmp_scalebar = np.asarray(_cmp_prepped["scalebar_px"], dtype=np.float32)
        _cmp_scalebar_valid = bool(np.asarray(_cmp_prepped["scalebar_valid"]))
        _cmp_px_per_cm_f = float(np.linalg.norm(_cmp_scalebar[1] - _cmp_scalebar[0]))
        _cmp_scalebar_mask = (
            _cmp_scalebar_valid
            and np.isfinite(_cmp_px_per_cm_f)
            and (_cmp_px_per_cm_f > 1e-6)
        )
        if _cmp_scalebar_mask:
            _cmp_endpoint_soft_cm.extend(
                (_cmp_err_soft_l2.reshape(-1) / _cmp_px_per_cm_f).tolist()
            )
            _cmp_endpoint_hard_cm.extend(
                (_cmp_err_hard_l2.reshape(-1) / _cmp_px_per_cm_f).tolist()
            )

        _cmp_sample_records.append((
            _cmp_dataset_name,
            float(_cmp_err_soft_l2.mean()),
            float(_cmp_err_hard_l2.mean()),
        ))

    _cmp_soft_px = np.asarray(_cmp_endpoint_soft_px, dtype=np.float64)
    _cmp_hard_px = np.asarray(_cmp_endpoint_hard_px, dtype=np.float64)
    _cmp_edge_hm = np.asarray(_cmp_endpoint_edge_dist_hm, dtype=np.float64)
    _cmp_soft_cm = np.asarray(_cmp_endpoint_soft_cm, dtype=np.float64)
    _cmp_hard_cm = np.asarray(_cmp_endpoint_hard_cm, dtype=np.float64)

    _cmp_fig, _cmp_axes = plt.subplots(
        1, 3, figsize=(16, 4.6), layout="constrained", dpi=100
    )

    _cmp_bins = np.linspace(0.0, max(float(np.max(_cmp_hard_px)), 1e-6), 18)
    _cmp_axes[0].hist(
        _cmp_soft_px, bins=_cmp_bins, alpha=0.7, label="softargmax", color="#2ca02c"
    )
    _cmp_axes[0].hist(
        _cmp_hard_px, bins=_cmp_bins, alpha=0.55, label="argmax", color="#1f77b4"
    )
    _cmp_axes[0].set_xlabel("Endpoint decode error (px)")
    _cmp_axes[0].set_ylabel("Count")
    _cmp_axes[0].set_title("Endpoint Error Distribution")
    _cmp_axes[0].legend(frameon=False)
    _cmp_axes[0].grid(alpha=0.25)
    _cmp_axes[0].spines[["right", "top"]].set_visible(False)

    _cmp_axes[1].scatter(
        _cmp_edge_hm,
        _cmp_soft_px,
        s=22,
        alpha=0.8,
        label="softargmax",
        color="#2ca02c",
    )
    _cmp_axes[1].scatter(
        _cmp_edge_hm,
        _cmp_hard_px,
        s=22,
        alpha=0.8,
        label="argmax",
        color="#1f77b4",
    )
    _cmp_axes[1].set_xlabel("Distance to heatmap border (pixels)")
    _cmp_axes[1].set_ylabel("Endpoint decode error (px)")
    _cmp_axes[1].set_title("Edge Effects at Large Sigma")
    _cmp_axes[1].legend(frameon=False)
    _cmp_axes[1].grid(alpha=0.25)
    _cmp_axes[1].spines[["right", "top"]].set_visible(False)

    _cmp_summary_lines = [
        f"sigma={_cmp_sigma_large:g}, n_samples={len(_cmp_sample_records)}, n_endpoints={_cmp_soft_px.size}",
        f"softargmax px mean={_cmp_soft_px.mean():.3f}, median={np.median(_cmp_soft_px):.3f}, p95={np.quantile(_cmp_soft_px, 0.95):.3f}",
        f"argmax px mean={_cmp_hard_px.mean():.3f}, median={np.median(_cmp_hard_px):.3f}, p95={np.quantile(_cmp_hard_px, 0.95):.3f}",
    ]
    if _cmp_soft_cm.size > 0 and _cmp_hard_cm.size > 0:
        _cmp_summary_lines.extend([
            f"softargmax cm mean={_cmp_soft_cm.mean():.5f}, p95={np.quantile(_cmp_soft_cm, 0.95):.5f}",
            f"argmax cm mean={_cmp_hard_cm.mean():.5f}, p95={np.quantile(_cmp_hard_cm, 0.95):.5f}",
        ])
    _cmp_summary_lines.append("")
    _cmp_summary_lines.append("Per-sample mean decode error (px):")
    for _cmp_ds_name, _cmp_soft_mean, _cmp_hard_mean in _cmp_sample_records:
        _cmp_summary_lines.append(
            f"- {_cmp_ds_name:13s} soft={_cmp_soft_mean:.3f} | argmax={_cmp_hard_mean:.3f}"
        )
    _cmp_axes[2].axis("off")
    _cmp_axes[2].text(
        0.0,
        1.0,
        "\n".join(_cmp_summary_lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=9,
    )

    _cmp_fig.suptitle(
        "Decode Comparison at Large Sigma: softargmax vs argmax",
        fontsize=12,
    )
    _cmp_fig
    return


@app.cell
def _(
    btx,
    dataset_rows,
    decode_argmax,
    make_heatmap_logits,
    np,
    plt,
    run_pre_heatmap_pipeline,
):
    _viz_sigma_large = 8.0
    _viz_top_k = 4
    _viz_n_candidates_per_dataset = 6

    _viz_rows = []
    for _viz_dataset_name, _viz_samples in dataset_rows:
        for _viz_sample in _viz_samples[:_viz_n_candidates_per_dataset]:
            _viz_rows.append((_viz_dataset_name, _viz_sample))
    assert _viz_rows, "No samples were available from the configured datasets."

    _viz_records = []
    for _viz_dataset_name, _viz_raw_sample in _viz_rows:
        _viz_prepped = run_pre_heatmap_pipeline(_viz_raw_sample)
        _viz_img = np.asarray(_viz_prepped["img"], dtype=np.float32)
        _viz_tgt = np.asarray(_viz_prepped["tgt"], dtype=np.float32)
        _viz_cfg, _viz_heatmap_tgt, _viz_logits = make_heatmap_logits(
            _viz_tgt, sigma=_viz_sigma_large
        )
        _viz_pred_soft = np.asarray(
            btx.objectives.heatmap.heatmaps_to_coords(_viz_logits, cfg=_viz_cfg)
        )
        _viz_pred_hard = decode_argmax(_viz_logits, cfg=_viz_cfg)
        _viz_soft_hard_px = np.linalg.norm(_viz_pred_soft - _viz_pred_hard, axis=2)
        _viz_soft_err_px = np.linalg.norm(_viz_pred_soft - _viz_tgt, axis=2)
        _viz_hard_err_px = np.linalg.norm(_viz_pred_hard - _viz_tgt, axis=2)
        _viz_records.append({
            "dataset_name": _viz_dataset_name,
            "img": _viz_img,
            "tgt": _viz_tgt,
            "pred_soft": _viz_pred_soft,
            "pred_hard": _viz_pred_hard,
            "heat_max": np.max(np.asarray(_viz_heatmap_tgt), axis=0),
            "soft_hard_mean_px": float(_viz_soft_hard_px.mean()),
            "soft_err_mean_px": float(_viz_soft_err_px.mean()),
            "hard_err_mean_px": float(_viz_hard_err_px.mean()),
        })

    _viz_records = sorted(
        _viz_records, key=lambda _viz_r: _viz_r["soft_hard_mean_px"], reverse=True
    )
    _viz_show = _viz_records[:_viz_top_k]
    assert _viz_show, "No candidates to visualize."

    _viz_n_cols = 2
    _viz_n_rows = int(np.ceil(len(_viz_show) / _viz_n_cols))
    _viz_fig, _viz_axes = plt.subplots(
        _viz_n_rows,
        _viz_n_cols,
        figsize=(7.2 * _viz_n_cols, 5.8 * _viz_n_rows),
        layout="constrained",
        dpi=110,
    )
    _viz_axes = np.asarray(_viz_axes).reshape(-1)

    for _viz_ax, _viz_item in zip(_viz_axes, _viz_show):
        _viz_ax.imshow(np.clip(_viz_item["img"], 0.0, 1.0))
        _viz_ax.imshow(
            _viz_item["heat_max"],
            cmap="magma",
            alpha=0.45,
            interpolation="bilinear",
            extent=(-0.5, 255.5, 255.5, -0.5),
        )

        for _viz_line_i in range(2):
            _viz_gt = _viz_item["tgt"][_viz_line_i]
            _viz_soft = _viz_item["pred_soft"][_viz_line_i]
            _viz_hard = _viz_item["pred_hard"][_viz_line_i]
            _viz_gt_color = "cyan" if _viz_line_i == 0 else "magenta"
            _viz_soft_color = "lime" if _viz_line_i == 0 else "yellow"
            _viz_hard_color = "orange" if _viz_line_i == 0 else "red"

            _viz_ax.plot(
                _viz_gt[:, 0],
                _viz_gt[:, 1],
                color=_viz_gt_color,
                marker="o",
                linewidth=2.0,
                markersize=4,
                alpha=0.95,
            )
            _viz_ax.plot(
                _viz_soft[:, 0],
                _viz_soft[:, 1],
                color=_viz_soft_color,
                marker="x",
                linestyle="--",
                linewidth=1.8,
                markersize=5,
                alpha=0.95,
            )
            _viz_ax.plot(
                _viz_hard[:, 0],
                _viz_hard[:, 1],
                color=_viz_hard_color,
                marker="s",
                linestyle=":",
                linewidth=1.8,
                markersize=4,
                alpha=0.95,
            )

            for _viz_pt_i in range(2):
                _viz_ax.plot(
                    [_viz_soft[_viz_pt_i, 0], _viz_hard[_viz_pt_i, 0]],
                    [_viz_soft[_viz_pt_i, 1], _viz_hard[_viz_pt_i, 1]],
                    color="white",
                    linewidth=1.3,
                    alpha=0.9,
                )

        _viz_ax.set_title(
            f"{_viz_item['dataset_name']} | soft-hard={_viz_item['soft_hard_mean_px']:.2f}px | soft_err={_viz_item['soft_err_mean_px']:.2f}px | hard_err={_viz_item['hard_err_mean_px']:.2f}px",
            fontsize=9,
        )
        _viz_ax.set_xticks([])
        _viz_ax.set_yticks([])
        _viz_ax.spines[:].set_visible(False)

    for _viz_ax in _viz_axes[len(_viz_show) :]:
        _viz_ax.axis("off")

    _viz_fig.suptitle(
        f"Large-Sigma Examples (sigma={_viz_sigma_large:g}): softargmax vs argmax",
        fontsize=12,
    )
    _viz_fig
    return


if __name__ == "__main__":
    app.run()
