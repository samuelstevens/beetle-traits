import typing as tp

import beartype
import chex
import equinox as eqx

from . import frozen, heatmap, toy

Config = toy.Toy | frozen.Frozen | heatmap.Heatmap


@beartype.beartype
def make(cfg: Config, key: chex.PRNGKey) -> eqx.Module:
    """Construct a model from configuration.

    Args:
        cfg: Model configuration variant.
        key: Random key used for initializing trainable layers.

    Returns:
        Instantiated Equinox model.
    """
    if isinstance(cfg, toy.Toy):
        return toy.Model(cfg, key=key)
    elif isinstance(cfg, frozen.Frozen):
        return frozen.Model(cfg, key=key)
    elif isinstance(cfg, heatmap.Heatmap):
        return heatmap.Model(cfg, key=key)
    else:
        tp.assert_never(cfg)
