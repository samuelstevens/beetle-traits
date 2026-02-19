import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
def _():
    import copy
    import pathlib

    import grain.transforms
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    import btx
    import btx.data.transforms
    return Image, btx, copy, grain, mo, np, pathlib, plt


@app.cell
def _(Image, btx, pathlib):
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

    beetlepalooza_samples = load_samples(btx.data.BeetlePaloozaConfig())
    hawaii_samples = load_samples(
        btx.data.HawaiiConfig(
            split="train",
            include_polylines=False,
            hf_root=HAWAII_HF_ROOT,
        )
    )
    biorepo_samples = load_samples(
        btx.data.BioRepoConfig(
            split="val",
            root=BIOREPO_ROOT,
            annotations=BIOREPO_ANN,
        )
    )
    return beetlepalooza_samples, biorepo_samples, hawaii_samples


@app.cell
def _(
    beetlepalooza_samples,
    biorepo_samples,
    btx,
    copy,
    grain,
    hawaii_samples,
    np,
    plt,
):
    def _run_aug(raw_sample, aug_cfg, *, seed):
        """Run the full augmentation pipeline on a raw sample."""
        tfms = btx.data.transforms.make_transforms(aug_cfg, is_train=True)
        rng = np.random.default_rng(seed=seed)
        sample = copy.deepcopy(raw_sample)
        for tfm in tfms:
            if isinstance(tfm, grain.transforms.RandomMap):
                sample = tfm.random_map(sample, rng=rng)
            else:
                sample = tfm.map(sample)
        return sample

    def _plot_kp(ax, pts):
        """Plot width (cyan) and length (magenta) keypoints."""
        (x1, y1), (x2, y2) = pts[0]
        ax.plot([x1, x2], [y1, y2], "o-", color="cyan", markersize=4, linewidth=1.5)
        (x1, y1), (x2, y2) = pts[1]
        ax.plot([x1, x2], [y1, y2], "o-", color="magenta", markersize=4, linewidth=1.5)

    _default = btx.data.transforms.AugmentConfig()
    _alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    _n_per_ds = 3
    _datasets = [
        ("Hawaii", hawaii_samples),
        ("BeetlePalooza", beetlepalooza_samples),
        ("BioRepo", biorepo_samples),
    ]

    _rows = []
    for _ds_name, _ds_samples in _datasets:
        for _s in _ds_samples[:_n_per_ds]:
            _rows.append((_ds_name, _s))

    _n_rows = len(_rows)
    # Layout: original | gap | crop nocrop | gap | crop nocrop | ...
    # Width ratios: 1 for image columns, 0.15 for gap columns.
    _widths = [1.0]  # original
    for _ci in range(len(_alphas)):
        _widths.append(0.15)  # gap
        _widths.extend([1.0, 1.0])  # crop, no-crop
    _n_grid_cols = len(_widths)
    # Map from (alpha_idx, crop_idx) to grid column.
    _img_col = {}
    for _ci in range(len(_alphas)):
        _img_col[(_ci, 0)] = 1 + _ci * 3 + 1  # crop
        _img_col[(_ci, 1)] = 1 + _ci * 3 + 2  # no-crop
    _gap_cols = {1 + _ci * 3 for _ci in range(len(_alphas))}

    if _n_rows == 0:
        _fig, _ax = plt.subplots(figsize=(8, 2))
        _ax.text(0.5, 0.5, "No samples available", ha="center", va="center")
        _ax.axis("off")
    else:
        _fig, _axes = plt.subplots(
            _n_rows,
            _n_grid_cols,
            figsize=(2.2 * (1 + 2 * len(_alphas)), 2.2 * _n_rows),
            layout="constrained",
            dpi=50,
            gridspec_kw={"width_ratios": _widths},
        )
        if _n_rows == 1:
            _axes = _axes[None, :]

        _fig.suptitle("Augmentation Sweep (width=cyan, length=magenta)")

        # Hide gap columns.
        for _ri in range(_n_rows):
            for _gc in _gap_cols:
                _axes[_ri, _gc].set_visible(False)

        # Column headers (first row only)
        _axes[0, 0].set_title("original", fontsize=9)
        for _ci, _alpha in enumerate(_alphas):
            _axes[0, _img_col[(_ci, 0)]].set_title(f"a={_alpha:.2f}\ncrop", fontsize=8)
            _axes[0, _img_col[(_ci, 1)]].set_title(
                f"a={_alpha:.2f}\nno crop", fontsize=8
            )

        _prev_ds = None
        for _ri, (_ds_name, _raw) in enumerate(_rows):
            # Original column
            _ax = _axes[_ri, 0]
            _ax.imshow(np.asarray(_raw["img"], dtype=np.float32) / 255.0)
            _plot_kp(_ax, _raw["points_px"])
            if _ds_name != _prev_ds:
                _ax.set_ylabel(_ds_name, fontsize=10, rotation=90)
                _prev_ds = _ds_name
            _ax.set_xticks([])
            _ax.set_yticks([])
            _ax.spines[:].set_visible(False)

            # Paired columns for each alpha
            for _ci, _alpha in enumerate(_alphas):
                _scale_min = float(1.0 - _alpha * (1.0 - _default.crop_scale_min))
                _ratio_min = float(1.0 - _alpha * (1.0 - _default.crop_ratio_min))
                _ratio_max = float(1.0 + _alpha * (_default.crop_ratio_max - 1.0))
                _base = dict(
                    crop_scale_min=_scale_min,
                    crop_scale_max=1.0,
                    crop_ratio_min=_ratio_min,
                    crop_ratio_max=_ratio_max,
                    normalize=False,
                )
                for _j, _do_crop in enumerate([True, False]):
                    _cfg = btx.data.transforms.AugmentConfig(crop=_do_crop, **_base)
                    _out = _run_aug(_raw, _cfg, seed=7000 + _ri * 31 + _ci)
                    _ax = _axes[_ri, _img_col[(_ci, _j)]]
                    _ax.imshow(np.clip(_out["img"], 0.0, 1.0))
                    _plot_kp(_ax, _out["tgt"])
                    _oob = float(_out.get("oob_points_frac", 0.0))
                    if _ri > 0:
                        _ax.set_title(f"oob={_oob:.2f}", fontsize=8)
                    _ax.set_xticks([])
                    _ax.set_yticks([])
                    _ax.spines[:].set_visible(False)

    _fig
    return


@app.cell
def _(btx, grain, mo, np, pathlib, plt):
    oob_default_cfg = btx.data.transforms.AugmentConfig()

    def make_crop_cfg(alpha: float, *, crop: bool = True):
        scale_min = float(1.0 - alpha * (1.0 - oob_default_cfg.crop_scale_min))
        ratio_min = float(1.0 - alpha * (1.0 - oob_default_cfg.crop_ratio_min))
        ratio_max = float(1.0 + alpha * (oob_default_cfg.crop_ratio_max - 1.0))
        return btx.data.transforms.AugmentConfig(
            crop=crop,
            crop_scale_min=scale_min,
            crop_scale_max=1.0,
            crop_ratio_min=ratio_min,
            crop_ratio_max=ratio_max,
        )

    def mean_oob_for_cfg(dataset_cfg, aug_cfg, *, seed: int, n_samples: int):
        source = dataset_cfg.dataset(dataset_cfg)
        assert isinstance(source, grain.sources.RandomAccessDataSource)
        n_eval = min(n_samples, len(source))
        assert n_eval > 0, "Dataset has no samples."

        ds = grain.MapDataset.source(source).seed(seed).shuffle()
        tfms = btx.data.transforms.make_transforms(aug_cfg, is_train=True)
        for tfm_i, tfm in enumerate(tfms):
            if isinstance(tfm, grain.transforms.RandomMap):
                ds = ds.random_map(tfm, seed=seed + tfm_i)
            else:
                ds = ds.map(tfm)

        fracs = []
        for idx in range(n_eval):
            sample = ds[idx]
            frac = float(np.asarray(sample["oob_points_frac"], dtype=np.float32))
            fracs.append(frac)
        return float(np.mean(fracs)), n_eval

    def try_dataset(cfg):
        try:
            source = cfg.dataset(cfg)
            if len(source) == 0:
                return None
            # Try loading one sample to check images exist on disk.
            source[0]
            return cfg
        except (AssertionError, FileNotFoundError, NotImplementedError, OSError):
            return None

    all_datasets = [
        ("BeetlePalooza", "tab:orange", btx.data.BeetlePaloozaConfig()),
        (
            "Hawaii",
            "tab:blue",
            btx.data.HawaiiConfig(
                split="train",
                include_polylines=False,
                hf_root=pathlib.Path(
                    "/fs/ess/PAS2136/samuelstevens/datasets/hawaii-beetles"
                ),
            ),
        ),
        (
            "BioRepo",
            "tab:green",
            btx.data.BioRepoConfig(
                split="val",
                root=pathlib.Path("/fs/scratch/PAS2136/cain429/Subset-Exp"),
                annotations=pathlib.Path(
                    "/fs/scratch/PAS2136/cain429/biorepo-formatted/annotations.json"
                ),
            ),
        ),
    ]
    _datasets = [
        (_name, _color, _cfg)
        for _name, _color, _cfg in all_datasets
        if try_dataset(_cfg) is not None
    ]

    alphas = np.linspace(0.0, 1.0, 6)
    n_samples = 128
    oob_crop: dict[str, list[float]] = {_name: [] for _name, _, _ in _datasets}
    oob_nocrop: dict[str, list[float]] = {_name: [] for _name, _, _ in _datasets}
    n_per_dataset: dict[str, int] = {_name: 0 for _name, _, _ in _datasets}

    for _, alpha in mo.status.progress_bar(
        enumerate(alphas), total=len(alphas), title="OOB sweep over alphas"
    ):
        _a = float(alpha)
        _crop_cfg = make_crop_cfg(_a)
        _nocrop_cfg = make_crop_cfg(_a, crop=False)
        for _name, _, _ds_cfg in _datasets:
            _mean, _n = mean_oob_for_cfg(
                _ds_cfg,
                _crop_cfg,
                seed=17,
                n_samples=n_samples,
            )
            oob_crop[_name].append(_mean)
            n_per_dataset[_name] = _n
            _mean_nc, _ = mean_oob_for_cfg(
                _ds_cfg,
                _nocrop_cfg,
                seed=17,
                n_samples=n_samples,
            )
            oob_nocrop[_name].append(_mean_nc)

    fig_oob, ax_oob = plt.subplots(figsize=(9, 5), layout="constrained")
    for _name, _color, _ in _datasets:
        ax_oob.plot(
            alphas,
            oob_crop[_name],
            color=_color,
            linewidth=2,
            label=f"{_name} crop",
        )
        ax_oob.plot(
            alphas,
            oob_nocrop[_name],
            color=_color,
            linewidth=2,
            linestyle="--",
            label=f"{_name} no crop",
        )
    ax_oob.set_xlabel("Augmentation Level Alpha (0=Resize baseline, 1=target params)")
    ax_oob.set_ylabel("Mean OOB Points Fraction")
    ax_oob.set_title(
        f"OOB Fraction vs Crop Augmentation Level (n={n_samples} per dataset)"
    )
    ax_oob.grid(alpha=0.25)
    ax_oob.legend()
    ax_oob.spines[["top", "right"]].set_visible(False)
    ax_oob.set_yscale("symlog", linthresh=0.01)
    fig_oob
    return


if __name__ == "__main__":
    app.run()
