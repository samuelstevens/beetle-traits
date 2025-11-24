import dataclasses
import typing as tp

import beartype
import grain

from . import utils


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config(utils.Config):
    split: tp.Literal["train", "val"] = "train"
    """Which split."""
    # Split-related configuration.
    seed: int = 0
    """Random seed for split."""

    @property
    def dataset(self):
        return Dataset


@beartype.beartype
class Dataset(grain.sources.RandomAccessDataSource):
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def __len__(self) -> int:
        return 0
