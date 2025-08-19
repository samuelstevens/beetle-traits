import dataclasses
import pathlib
import typing as tp

import beartype
import polars as pl
import tyro


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    hf_root: pathlib.Path = pathlib.Path("./data/beetlepalooza")
    """Where you dumped data when using download_beetlepalooza.py."""


@beartype.beartype
def main(cfg: Config):
    pass


if __name__ == "__main__":
    try:
        raise SystemExit(main(tyro.cli(Config)))
    except KeyboardInterrupt:
        print("Interrupted.")
        raise SystemExit(130)
