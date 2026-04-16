"""Scalebar inference for experiment 013, checkpoint fdpj3qnm.

Run with:
    uv run python docs/experiments/013-scalebar-training/sweeps/infer.py

    # With Slurm:
    uv run python docs/experiments/013-scalebar-training/sweeps/infer.py \
        --slurm-acct PAS2136 \
        --slurm-partition gpu
"""

import logging
import pathlib

import beartype
import tyro

import btx.scalebar_infer

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("infer-013")

_CKPT_FPATH = pathlib.Path("/users/PAS2136/cain429/projects/beetle-traits/checkpoints/exp013/u39iaei2/model.eqx")
_OUT_FPATH = pathlib.Path(
    "docs/experiments/013-scalebar-training/results/fdpj3qnm_px_per_cm.json"
)

@beartype.beartype
def main(slurm_acct: str = "", slurm_partition: str = "", n_hours: float = 1.0) -> None:
    cfg = btx.scalebar_infer.Config(
        ckpt_fpath=_CKPT_FPATH,
        out_fpath=_OUT_FPATH,
        slurm_acct=slurm_acct,
        slurm_partition=slurm_partition,
        n_hours=n_hours,
    )

    import submitit

    if slurm_acct:
        executor = submitit.SlurmExecutor(folder=cfg.log_to)
        executor.update_parameters(
            job_name="beetle-scalebar-infer",
            time=int(n_hours * 60),
            partition=slurm_partition,
            gpus_per_node=1,
            ntasks_per_node=1,
            cpus_per_task=4,
            account=slurm_acct,
            mem="16GB",
            stderr_to_stdout=True,
            setup=[
                "unset SLURM_CPUS_PER_TASK",
                "export CUDA_VISIBLE_DEVICES=0",
                "export XLA_PYTHON_CLIENT_PREALLOCATE=false",
            ],
        )
    else:
        executor = submitit.DebugExecutor(folder=cfg.log_to)

    job = executor.submit(btx.scalebar_infer.run, cfg)
    logger.info("Submitted job %s.", job.job_id)
    job.result()


if __name__ == "__main__":
    tyro.cli(main)
