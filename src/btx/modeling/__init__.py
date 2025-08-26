import typing as tp

import beartype
import chex
import equinox as eqx

from . import frozen, toy

Config = toy.Toy | frozen.Frozen


@beartype.beartype
def make(cfg: Config, key: chex.PRNGKey) -> eqx.Module:
    if isinstance(cfg, toy.Toy):
        return toy.Model(cfg, key=key)
    elif isinstance(cfg, frozen.Frozen):
        return frozen.Model(cfg, key=key)
    else:
        tp.assert_never(cfg)
