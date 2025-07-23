import dataclasses

import beartype
import equinox as eqx


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    pass


@beartype.beartype
class Model(eqx.Module):
    def __init__(self, cfg: Config):
        pass
