import beartype
import submitit
import tyro
import os
import train


@beartype.beartype
def main(cfg: train.Config):
    # specifying wandb acct to log to
    os.environ["WANDB_API_KEY"] = os.environ.get("WANDB_API_KEY")
    os.environ["WANDB_MODE"] = "online"
    os.environ["WANDB_ENTITY"] = os.environ.get("WANDB_ENTITY")
    os.environ["WANDB_PROJECT"] = "beetle-traits"
    if cfg.slurm_acct:
        executor = submitit.SlurmExecutor(folder=cfg.log_to)
        executor.update_parameters(
            time=int(cfg.n_hours * 60),
            partition=cfg.slurm_partition,
            gpus_per_node=1,
            ntasks_per_node=1,
            cpus_per_task=cfg.data.n_workers + 4,
            stderr_to_stdout=True,
            account=cfg.slurm_acct,
        )

    else:
        executor = submitit.DebugExecutor(folder=cfg.log_to)

    job = executor.submit(train.train, cfg)
    print(f"Running job {job.job_id}")
    job.result()


if __name__ == "__main__":
    main(tyro.cli(train.Config))
