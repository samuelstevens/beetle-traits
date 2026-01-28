import logging
import pathlib
import sys

import beartype
import submitit
import tyro

import btx.configs
import train

logger = logging.getLogger(__name__)


@beartype.beartype
def _assert_executor_compatible(cfgs: list[train.Config]) -> train.Config:
    msg = "Expected at least one config to launch."
    assert cfgs, msg

    base = cfgs[0]
    for cfg in cfgs[1:]:
        msg = "Sweep configs must share slurm_acct, slurm_partition, n_hours, n_workers, and log_to."
        assert cfg.slurm_acct == base.slurm_acct, msg
        assert cfg.slurm_partition == base.slurm_partition, msg
        assert cfg.n_hours == base.n_hours, msg
        assert cfg.n_workers == base.n_workers, msg
        assert cfg.log_to == base.log_to, msg

    return base


@beartype.beartype
def main(cfg: train.Config, sweep: pathlib.Path | None = None):
    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)

    if sweep is None:
        cfgs = [cfg]
    else:
        sweep_dcts = btx.configs.load_sweep(sweep)
        if not sweep_dcts:
            logger.error("No valid sweeps found in '%s'.", sweep)
            sys.exit(1)

        cfgs, errs = btx.configs.load_cfgs(
            cfg, default=train.Config(), sweep_dcts=sweep_dcts
        )
        if errs:
            for err in errs:
                logger.warning("Error in config: %s", err)
            sys.exit(1)

    cfg = _assert_executor_compatible(cfgs)

    if cfg.slurm_acct:
        executor = submitit.SlurmExecutor(folder=cfg.log_to)
        executor.update_parameters(
            time=int(cfg.n_hours * 60),
            partition=cfg.slurm_partition,
            gpus_per_node=1,
            ntasks_per_node=1,
            cpus_per_task=cfg.n_workers + 4,
            stderr_to_stdout=True,
            account=cfg.slurm_acct,
        )
    else:
        executor = submitit.DebugExecutor(folder=cfg.log_to)

    with executor.batch():
        jobs = [executor.submit(train.train, job_cfg) for job_cfg in cfgs]

    for job in jobs:
        logger.info("Running job %s", job.job_id)

    for job in jobs:
        job.result()


if __name__ == "__main__":
    tyro.cli(main)
