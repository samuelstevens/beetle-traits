import typing as tp

import beartype
import chex
import equinox as eqx

from . import toy

Config = toy.Config


@beartype.beartype
def make(cfg: Config, key: chex.PRNGKey) -> eqx.Module:
    if isinstance(cfg, toy.Config):
        return toy.Model(cfg, key=key)
    else:
        tp.assert_never(cfg)
