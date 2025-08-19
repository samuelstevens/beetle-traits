# src/btx/scripts/merge_to_imagefolder.py
import concurrent.futures
import dataclasses
import logging
import pathlib
import shutil

import beartype
import polars as pl
import tyro

from btx import helpers

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    bp_root: pathlib.Path = pathlib.Path("./data/beetlepalooza")
    """Where you dumped data when using download_beetlepalooza.py."""
    hi_root: pathlib.Path = pathlib.Path("./data/hawaii")
    """Where you dumped data when using download_hawaii.py."""
    br_root: pathlib.Path = pathlib.Path("./data/biorepo")
    """Where you dumped data when using download_biorepo.py."""
    dump_to: pathlib.Path = pathlib.Path("./data/imagefolder")
    """Where to write the new image directories."""
    n_threads: int = 16
    """Number of concurrent threads for copying files on disk."""
    job_size: int = 256
    """Number of images to copy per job."""


@beartype.beartype
def _move_bp(cfg: Config, start: int, end: int):
    bp_df = pl.read_csv(cfg.bp_root / "individual_specimens.csv")

    for (img_fpath,) in (
        bp_df.select("individualImageFilePath").slice(start, end - start).iter_rows()
    ):
        src_fpath = cfg.bp_root / img_fpath
        img_fname = pathlib.Path(img_fpath).name
        dst_fpath = cfg.dump_to / "beetlepalooza" / img_fname

        if not src_fpath.exists():
            continue

        if dst_fpath.exists():
            continue

        shutil.copy2(src_fpath, dst_fpath)


@beartype.beartype
def _move_hi(cfg: Config, start: int, end: int):
    hi_df = pl.read_csv(cfg.hi_root / "images_metadata.csv")
    for (img_fpath,) in (
        hi_df.select("individualImageFilePath").slice(start, end - start).iter_rows()
    ):
        src_fpath = cfg.hi_root / img_fpath
        img_fname = pathlib.Path(img_fpath).name
        dst_fpath = cfg.dump_to / "hawaii" / img_fname

        if not src_fpath.exists():
            continue

        if dst_fpath.exists():
            continue

        shutil.copy2(src_fpath, dst_fpath)
    pass


@beartype.beartype
def _move_br(cfg: Config):
    pass


@beartype.beartype
def main(cfg: Config):
    """
    Converts all the imageomics beetle datasets into one merged root/class/example.png format for use with ImageFolder dataloaders.
    """
    logger = logging.getLogger("merge")

    (cfg.dump_to / "beetlepalooza").mkdir(exist_ok=True, parents=True)
    (cfg.dump_to / "hawaii").mkdir(exist_ok=True, parents=True)
    (cfg.dump_to / "biorepo").mkdir(exist_ok=True, parents=True)

    bp_df = pl.read_csv(cfg.bp_root / "individual_specimens.csv")
    hi_df = pl.read_csv(cfg.hi_root / "images_metadata.csv")

    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.n_threads) as pool:
        futs = []

        # BeetlePalooza
        futs.extend([
            pool.submit(_move_bp, cfg, start, end)
            for start, end in helpers.batched_idx(bp_df.height, cfg.job_size)
        ])

        # Hawaii
        futs.extend([
            pool.submit(_move_hi, cfg, start, end)
            for start, end in helpers.batched_idx(hi_df.height, cfg.job_size)
        ])

        for fut in helpers.progress(
            concurrent.futures.as_completed(futs), total=len(futs), desc="copying"
        ):
            if err := fut.exception():
                logger.warning("Exception: %s", err)


if __name__ == "__main__":
    try:
        raise SystemExit(main(tyro.cli(Config)))
    except KeyboardInterrupt:
        print("Interrupted.")
        raise SystemExit(130)
