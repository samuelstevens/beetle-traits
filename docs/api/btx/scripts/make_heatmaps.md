Module btx.scripts.make_heatmaps
================================
Generate heatmap overlay images at entropy percentiles for unlabeled beetles.

Picks unlabeled beetles at entropy percentiles (p5, p25, p50, p75, p95, p99), runs forward passes through a trained model, and saves predicted heatmap overlays as a single PNG grid.

Run with:
    uv run python -m btx.scripts.make_heatmaps --parquet-fpath results/run_unlabeled.parquet --ckpt-fpath checkpoints/model.eqx --out-dpath results/heatmaps/ --slurm-acct PAS2136

Functions
---------

`load_and_preprocess(img_fpath: str) ‑> tuple[numpy.ndarray, numpy.ndarray]`
:   Load image, resize to 256x256. Returns (display_hwc, normalized_hwc).

`main(cfg: Annotated[btx.scripts.make_heatmaps.Config, _ArgConfig(name='', metavar=None, help=None, help_behavior_hint=None, aliases=None, prefix_name=None, constructor_factory=None, default=<NonpropagatingMissingType id='22484197865904'>)], sweep: pathlib.Path | None = None)`
:   

`run(cfg: btx.scripts.make_heatmaps.Config)`
:   Generate heatmap overlays at entropy percentiles. Runs on the worker node.

Classes
-------

`Config(parquet_fpath: pathlib.Path = PosixPath('results/unlabeled.parquet'), ckpt_fpath: pathlib.Path = PosixPath('checkpoints/model.eqx'), out_dpath: pathlib.Path = PosixPath('results/heatmaps'), n_per_pct: int = 4, slurm_acct: str = '', slurm_partition: str = 'debug-nextgen', n_hours: float = 0.5, log_to: pathlib.Path = PosixPath('logs'))`
:   Config(parquet_fpath: pathlib.Path = PosixPath('results/unlabeled.parquet'), ckpt_fpath: pathlib.Path = PosixPath('checkpoints/model.eqx'), out_dpath: pathlib.Path = PosixPath('results/heatmaps'), n_per_pct: int = 4, slurm_acct: str = '', slurm_partition: str = 'debug-nextgen', n_hours: float = 0.5, log_to: pathlib.Path = PosixPath('logs'))

    ### Instance variables

    `ckpt_fpath: pathlib.Path`
    :   Path to saved model checkpoint (.eqx file).

    `log_to: pathlib.Path`
    :   Where to save submitit/slurm logs.

    `n_hours: float`
    :   Slurm job length in hours.

    `n_per_pct: int`
    :   Number of images per entropy percentile.

    `out_dpath: pathlib.Path`
    :   Output directory for heatmap PNG files.

    `parquet_fpath: pathlib.Path`
    :   Path to an *_unlabeled.parquet from inference.py.

    `slurm_acct: str`
    :   Slurm account. If empty, uses DebugExecutor (local execution).

    `slurm_partition: str`
    :   Slurm partition.