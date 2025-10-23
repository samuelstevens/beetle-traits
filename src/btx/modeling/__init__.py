import typing as tp

import beartype
import chex
import equinox as eqx

from . import frozen, pixelmse, toy

Config = toy.Toy | frozen.Frozen | pixelmse.PixelMse


@beartype.beartype
def make(cfg: Config, key: chex.PRNGKey) -> eqx.Module:
    if isinstance(cfg, toy.Toy):
        return toy.Model(cfg, key=key)
    elif isinstance(cfg, frozen.Frozen):
        return frozen.Model(cfg, key=key)
    elif isinstance(cfg, pixelmse.PixelMse): 
        return pixelmse.Model(cfg, key=key)
    else:
        tp.assert_never(cfg)
