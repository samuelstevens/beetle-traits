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
    import btx.data.augment

    return Image, btx, copy, grain, mo, np, pathlib, plt


@app.cell
def _(btx, grain):
    cfg = btx.data.augment.AugmentConfig()

    tfms: list[grain.transforms.Map | grain.transforms.RandomMap] = [
        btx.data.utils.DecodeRGB(),
        btx.data.augment.InitAugState(size=cfg.size, min_px_per_cm=cfg.min_px_per_cm),
        btx.data.augment.RandomResizedCrop(cfg=cfg),
        btx.data.augment.RandomFlip(cfg=cfg),
        btx.data.augment.RandomRotation90(cfg=cfg),
        btx.data.augment.ColorJitter(cfg=cfg),
        btx.data.augment.FinalizeTargets(cfg=cfg),
        btx.data.utils.Normalize(),
    ]
    tfms
    return (tfms,)


@app.cell
def _(Image, btx, pathlib):
    def get_orig_sample():
        candidates = [
            btx.data.BeetlePaloozaConfig(),
            btx.data.HawaiiConfig(
                split="train",
                include_polylines=False,
                hf_root=pathlib.Path(
                    "/fs/ess/PAS2136/samuelstevens/datasets/hawaii-beetles"
                ),
            ),
        ]
        sample = None
        for ds_cfg in candidates:
            try:
                ds = ds_cfg.dataset(ds_cfg)
            except (AssertionError, FileNotFoundError, NotImplementedError) as err:
                print(err)
                continue

            n = min(len(ds), 128)
            for idx in range(n):
                try:
                    print(ds)
                    sample = dict(ds[idx])

                    with Image.open(sample["img_fpath"]) as fd:
                        sample["img"] = fd.convert("RGB")
                except (AssertionError, FileNotFoundError, OSError) as err:
                    print(f"Tried {idx}: {err}")
                    continue
                break

            if sample is not None:
                break

        assert sample is not None, "No beetle samples found in configured datasets."
        return sample

    def get_beetlepalooza_samples(limit: int = 6):
        bp_cfg = btx.data.BeetlePaloozaConfig()
        ds = bp_cfg.dataset(bp_cfg)
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

    def get_hawaii_samples(limit: int = 6):
        hi_cfg = btx.data.HawaiiConfig(
            split="train",
            include_polylines=False,
            hf_root=pathlib.Path(
                "/fs/ess/PAS2136/samuelstevens/datasets/hawaii-beetles"
            ),
        )
        try:
            ds = hi_cfg.dataset(hi_cfg)
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

    def get_biorepo_samples(limit: int = 6):
        br_cfg = btx.data.BioRepoConfig(split="train")
        try:
            ds = br_cfg.dataset(br_cfg)
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

    orig_sample = get_orig_sample()
    beetlepalooza_samples = get_beetlepalooza_samples()
    hawaii_samples = get_hawaii_samples()
    biorepo_samples = get_biorepo_samples()
    orig_sample
    return beetlepalooza_samples, biorepo_samples, hawaii_samples, orig_sample


@app.cell
def _(
    copy,
    grain,
    np,
    orig_sample,
    tfms: "list[grain.transforms.Map | grain.transforms.RandomMap]",
):
    samples = []
    for seed in range(4):
        rng = np.random.default_rng(seed=seed + 45)
        sample = copy.deepcopy(orig_sample)

        for tfm in tfms:
            if isinstance(tfm, grain.transforms.RandomMap):
                sample = tfm.random_map(sample, rng=rng)
            else:
                sample = tfm.map(sample)

        samples.append(sample)
    samples
    return (samples,)


@app.cell
def _(np, orig_sample, plt, samples):
    def to_display_img(img):
        arr_raw = np.asarray(img)
        if np.issubdtype(arr_raw.dtype, np.integer):
            arr = arr_raw.astype(np.float32) / 255.0
        else:
            arr = arr_raw.astype(np.float32, copy=False)

        if arr.min() < -0.01 or arr.max() > 1.01:
            mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
            std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
            arr = arr * std[None, None, :] + mean[None, None, :]

        return np.clip(arr, 0.0, 1.0)

    def show():
        fig, axes = plt.subplots(
            ncols=len(samples) + 1, sharex=True, sharey=True, layout="constrained"
        )
        axes = axes.reshape(-1)

        for sample, ax in zip([orig_sample] + samples, axes, strict=True):
            ax.imshow(to_display_img(sample["img"]))
            # ax.scatter()
            if "tgt" in sample:
                width, length = sample["tgt"]
            else:
                width, length = sample["points_px"]
            (x1, y1), (x2, y2) = width
            ax.scatter([x1, x2], [y1, y2])
            (x1, y1), (x2, y2) = length
            ax.scatter([x1, x2], [y1, y2])

            ax.spines[:].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])

        return fig

    show()
    return


@app.cell
def _(beetlepalooza_samples, btx, copy, np, plt):
    _default_cfg = btx.data.augment.AugmentConfig()
    _alphas = np.linspace(0.0, 1.0, 5)
    presets = []
    for _alpha in _alphas:
        _params = dict(
            brightness=float(_alpha * _default_cfg.brightness),
            contrast=float(_alpha * _default_cfg.contrast),
            saturation=float(_alpha * _default_cfg.saturation),
            hue=float(_alpha * _default_cfg.hue),
        )
        presets.append((f"alpha={_alpha:.2f}", _params))

    n_rows = min(len(beetlepalooza_samples), 6)
    n_cols = len(presets)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2 * n_cols, 2 * n_rows),
        layout="constrained",
        sharex=True,
        sharey=True,
    )
    if n_rows == 1:
        axes = axes[None, :]

    for row_i, base_sample in enumerate(beetlepalooza_samples[:n_rows]):
        for col_i, (name, params) in enumerate(presets):
            _sample = {"img": np.asarray(base_sample["img"], dtype=np.float32) / 255.0}
            _cfg = btx.data.augment.AugmentConfig(**params)
            jitter = btx.data.augment.ColorJitter(cfg=_cfg)
            _rng = np.random.default_rng(seed=1000 + row_i * 17 + col_i)
            _sample = jitter.random_map(copy.deepcopy(_sample), rng=_rng)

            ax = axes[row_i, col_i]
            ax.imshow(np.clip(_sample["img"], 0.0, 1.0))
            if row_i == 0:
                ax.set_title(name)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[:].set_visible(False)

    fig
    return


@app.cell
def _(beetlepalooza_samples, biorepo_samples, btx, copy, hawaii_samples, np, plt):
    def make_crop_presets(
        *,
        target_scale_min: float,
        target_ratio_min: float,
        target_ratio_max: float,
        steps: int = 5,
    ):
        alphas = np.linspace(0.0, 1.0, steps)
        presets = []
        for alpha in alphas:
            scale_min = float(1.0 - alpha * (1.0 - target_scale_min))
            ratio_min = float(1.0 - alpha * (1.0 - target_ratio_min))
            ratio_max = float(1.0 + alpha * (target_ratio_max - 1.0))
            presets.append((
                f"a={alpha:.2f}\ns=({scale_min:.2f},1.00)\nr=({ratio_min:.2f},{ratio_max:.2f})",
                dict(
                    crop_scale_min=scale_min,
                    crop_scale_max=1.0,
                    crop_ratio_min=ratio_min,
                    crop_ratio_max=ratio_max,
                ),
            ))
        return presets

    def show_crop_sweep(samples, presets, title: str):
        n_rows = min(len(samples), 6)
        if n_rows == 0:
            fig, ax = plt.subplots(figsize=(8, 2), layout="constrained")
            ax.text(
                0.5, 0.5, f"No samples available: {title}", ha="center", va="center"
            )
            ax.axis("off")
            return fig

        n_cols = len(presets)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(2.4 * n_cols, 2.2 * n_rows),
            layout="constrained",
            sharex=True,
            sharey=True,
        )
        if n_rows == 1:
            axes = axes[None, :]

        fig.suptitle(title)
        for row_i, base_sample in enumerate(samples[:n_rows]):
            base_img = np.asarray(base_sample["img"], dtype=np.float32) / 255.0
            for col_i, (name, params) in enumerate(presets):
                cfg = btx.data.augment.AugmentConfig(
                    brightness=0.0,
                    contrast=0.0,
                    saturation=0.0,
                    hue=0.0,
                    **params,
                )
                crop = btx.data.augment.RandomResizedCrop(cfg=cfg)
                sample = {
                    "img": copy.deepcopy(base_img),
                    "t_aug_from_orig": np.eye(3, dtype=np.float64),
                }
                rng = np.random.default_rng(seed=4000 + row_i * 29 + col_i)
                out = crop.random_map(sample, rng=rng)

                ax = axes[row_i, col_i]
                ax.imshow(np.clip(out["img"], 0.0, 1.0))
                if row_i == 0:
                    ax.set_title(name, fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines[:].set_visible(False)
        return fig

    default_cfg = btx.data.augment.AugmentConfig()
    hawaii_presets = make_crop_presets(
        target_scale_min=default_cfg.crop_scale_min,
        target_ratio_min=default_cfg.crop_ratio_min,
        target_ratio_max=default_cfg.crop_ratio_max,
    )
    beetlepalooza_presets = make_crop_presets(
        target_scale_min=0.75,
        target_ratio_min=0.9,
        target_ratio_max=1.111,
    )
    biorepo_presets = make_crop_presets(
        target_scale_min=default_cfg.crop_scale_min,
        target_ratio_min=default_cfg.crop_ratio_min,
        target_ratio_max=default_cfg.crop_ratio_max,
    )

    fig_hawaii = show_crop_sweep(
        hawaii_samples,
        hawaii_presets,
        title="Hawaii Crop Sweep",
    )
    fig_beetlepalooza = show_crop_sweep(
        beetlepalooza_samples,
        beetlepalooza_presets,
        title="BeetlePalooza Crop Sweep",
    )
    fig_biorepo = show_crop_sweep(
        biorepo_samples,
        biorepo_presets,
        title="BioRepo Crop Sweep",
    )
    fig_hawaii
    fig_beetlepalooza
    fig_biorepo
    return


@app.cell
def _(btx, grain, mo, np, pathlib, plt):
    oob_default_cfg = btx.data.augment.AugmentConfig()

    def make_crop_cfg(alpha: float):
        scale_min = float(1.0 - alpha * (1.0 - oob_default_cfg.crop_scale_min))
        ratio_min = float(1.0 - alpha * (1.0 - oob_default_cfg.crop_ratio_min))
        ratio_max = float(1.0 + alpha * (oob_default_cfg.crop_ratio_max - 1.0))
        return btx.data.augment.AugmentConfig(
            crop_scale_min=scale_min,
            crop_scale_max=1.0,
            crop_ratio_min=ratio_min,
            crop_ratio_max=ratio_max,
        )

    def get_train_tfms(cfg, *, use_resize_baseline: bool):
        tfms = [
            btx.data.utils.DecodeRGB(),
            btx.data.augment.InitAugState(
                size=cfg.size, min_px_per_cm=cfg.min_px_per_cm
            ),
        ]
        if use_resize_baseline:
            tfms.append(btx.data.augment.Resize(size=cfg.size))
        else:
            tfms.append(btx.data.augment.RandomResizedCrop(cfg=cfg))
        tfms.extend([
            btx.data.augment.RandomFlip(cfg=cfg),
            btx.data.augment.RandomRotation90(cfg=cfg),
            btx.data.augment.ColorJitter(cfg=cfg),
            btx.data.augment.FinalizeTargets(cfg=cfg),
            btx.data.utils.Normalize(),
        ])
        return tfms

    def mean_oob_for_cfg(
        dataset_cfg, aug_cfg, *, seed: int, n_samples: int, use_resize_baseline: bool
    ):
        source = dataset_cfg.dataset(dataset_cfg)
        assert isinstance(source, grain.sources.RandomAccessDataSource)
        n_eval = min(n_samples, len(source))
        assert n_eval > 0, "Dataset has no samples."

        ds = grain.MapDataset.source(source).seed(seed).shuffle()
        tfms = get_train_tfms(aug_cfg, use_resize_baseline=use_resize_baseline)
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

    beetle_cfg = btx.data.BeetlePaloozaConfig()
    hawaii_cfg = btx.data.HawaiiConfig(
        split="train",
        include_polylines=False,
        hf_root=pathlib.Path("/fs/ess/PAS2136/samuelstevens/datasets/hawaii-beetles"),
    )
    biorepo_cfg = btx.data.BioRepoConfig(split="train")

    alphas = np.linspace(0.0, 1.0, 11)
    beetle_oob = []
    hawaii_oob = []
    biorepo_oob = []
    n_beetle = 0
    n_hawaii = 0
    n_biorepo = 0

    for _, alpha in mo.status.progress_bar(
        enumerate(alphas), total=len(alphas), title="OOB sweep over alphas"
    ):
        aug_cfg = make_crop_cfg(float(alpha))
        use_resize_baseline = bool(np.isclose(alpha, 0.0))
        beetle_mean, n_beetle = mean_oob_for_cfg(
            beetle_cfg,
            aug_cfg,
            seed=17,
            n_samples=128,
            use_resize_baseline=use_resize_baseline,
        )
        hawaii_mean, n_hawaii = mean_oob_for_cfg(
            hawaii_cfg,
            aug_cfg,
            seed=17,
            n_samples=128,
            use_resize_baseline=use_resize_baseline,
        )
        biorepo_mean, n_biorepo = mean_oob_for_cfg(
            biorepo_cfg,
            aug_cfg,
            seed=17,
            n_samples=128,
            use_resize_baseline=use_resize_baseline,
        )
        beetle_oob.append(beetle_mean)
        hawaii_oob.append(hawaii_mean)
        biorepo_oob.append(biorepo_mean)

    fig_oob, ax_oob = plt.subplots(figsize=(9, 5), layout="constrained")
    ax_oob.scatter(
        alphas,
        beetle_oob,
        color="tab:orange",
        s=36,
        label=f"BeetlePalooza (n={n_beetle})",
    )
    ax_oob.plot(alphas, beetle_oob, color="tab:orange", linewidth=2)
    ax_oob.scatter(
        alphas,
        hawaii_oob,
        color="tab:blue",
        s=36,
        label=f"Hawaii (n={n_hawaii})",
    )
    ax_oob.plot(alphas, hawaii_oob, color="tab:blue", linewidth=2)
    ax_oob.scatter(
        alphas,
        biorepo_oob,
        color="tab:green",
        s=36,
        label=f"BioRepo (n={n_biorepo})",
    )
    ax_oob.plot(alphas, biorepo_oob, color="tab:green", linewidth=2)
    ax_oob.set_xlabel("Augmentation Level Alpha (0=Resize baseline, 1=target params)")
    ax_oob.set_ylabel("Mean OOB Points Fraction")
    ax_oob.set_title("OOB Fraction vs Crop Augmentation Level")
    ax_oob.grid(alpha=0.25)
    ax_oob.legend()
    fig_oob
    return


if __name__ == "__main__":
    app.run()
