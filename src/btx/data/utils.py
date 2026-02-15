import abc
import typing as tp

import beartype
import grain
import numpy as np
from jaxtyping import Float, jaxtyped


@beartype.beartype
class Config(abc.ABC):
    @property
    @abc.abstractmethod
    def key(self) -> str: ...

    @property
    @abc.abstractmethod
    def dataset(self) -> type["Dataset"]: ...


@beartype.beartype
class Dataset(grain.sources.RandomAccessDataSource, abc.ABC):
    @property
    @abc.abstractmethod
    def cfg(self) -> Config: ...


@jaxtyped(typechecker=beartype.beartype)
class Sample(tp.TypedDict):
    img_fpath: str

    points_px: Float[np.ndarray, "lines 2 2"]
    """{width, length} x two points x {x, y}."""
    scalebar_px: Float[np.ndarray, "2 2"]
    """two points x {x, y}."""
    loss_mask: Float[np.ndarray, "2"]
    """Mask for {width, length} indicating which measurements to train on. 1.0 = train, 0.0 = skip."""

    # Metadata
    beetle_id: str
    beetle_position: int
    group_img_basename: str
    scientific_name: str
