# train_scalebar.py
"""Launcher for scalebar localization training.

All training logic lives in btx.scalebar_trainer (separate file required
because submitit and jaxtyping cannot coexist in the same module).

USAGE:
------
  uv run python train_scalebar.py

  # With Slurm:
  uv run python train_scalebar.py --slurm-acct PAS2136 --slurm-partition gpu

  # With a sweep:
  uv run python train_scalebar.py --sweep docs/experiments/013-scalebar-training/sweeps/train.py --slurm-acct PAS2136 --slurm-partition gpu
"""

import logging
import pathlib
import typing as tp

import beartype
import tyro

import btx.configs
from btx.scalebar_trainer import Config, train

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)

logger = logging.getLogger("train_scalebar.py")


@beartype.beartype
def main(
    cfg: tp.Annotated[Config, tyro.conf.arg(name="")],
    sweep: pathlib.Path | None = None,
) -> None:
    if sweep is None:
        cfgs = [cfg]
    else:
        sweep_dcts = btx.configs.load_sweep(sweep)
        if not sweep_dcts:
            logger.error("No valid sweeps found in '%s'.", sweep)
            return

        cfgs, errs = btx.configs.load_cfgs(cfg, default=Config(), sweep_dcts=sweep_dcts)
        if errs:
            for err in errs:
                logger.warning("Error in sweep config: %s", err)
            return

    base = cfgs[0]
    for c in cfgs[1:]:
        msg = "Sweep configs must share slurm_acct, slurm_partition, n_hours, n_workers, and log_to."
        assert c.slurm_acct == base.slurm_acct, msg
        assert c.slurm_partition == base.slurm_partition, msg
        assert c.n_hours == base.n_hours, msg
        assert c.n_workers == base.n_workers, msg
        assert c.log_to == base.log_to, msg

    if base.slurm_acct:
        import submitit

        executor = submitit.SlurmExecutor(folder=base.log_to)
        executor.update_parameters(
            job_name="beetle-scalebar",
            time=int(base.n_hours * 60),
            partition=base.slurm_partition,
            gpus_per_node=1,
            ntasks_per_node=1,
            cpus_per_task=base.n_workers + 1,
            account=base.slurm_acct,
            mem="32GB",
            stderr_to_stdout=True,
            setup=["unset SLURM_CPUS_PER_TASK", "export CUDA_VISIBLE_DEVICES=0", "export XLA_PYTHON_CLIENT_PREALLOCATE=false"],
        )
    else:
        import submitit

        executor = submitit.DebugExecutor(folder=base.log_to)

    with executor.batch():
        jobs = [executor.submit(train, job_cfg) for job_cfg in cfgs]

    for job in jobs:
        logger.info("Running job %s.", job.job_id)

    for job in jobs:
        job.result()


if __name__ == "__main__":
    tyro.cli(main)
