import typing as tp

import beartype
import equinox as eqx

from . import continuous

Config = continuous.Config


@beartype.beartype
def make(cfg: Config) -> eqx.Module:
    if isinstance(cfg, continuous.Config):
        return continuous.Model(cfg)
    else:
        tp.assert_never(cfg)
