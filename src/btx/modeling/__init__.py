"""Model construction and checkpoint save/load.

Checkpoint file format (following the dinov3.dump/load pattern):
  Line 1: JSON dict with 'model_cfg' and 'objective_cfg'
  Remaining bytes: eqx.tree_serialise_leaves binary data
"""

import dataclasses
import json
import pathlib
import typing as tp

import beartype
import chex
import equinox as eqx
import jax

import btx.objectives
import btx.objectives.heatmap

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


@beartype.beartype
def save_ckpt(
    model: eqx.Module,
    model_cfg: Config,
    objective_cfg: btx.objectives.Config,
    fpath: pathlib.Path,
):
    """Save a trained model checkpoint with embedded config.

    Args:
        model: Trained model to serialize.
        model_cfg: Model architecture config used to construct the model.
        objective_cfg: Objective config needed for inference decoding.
        fpath: Output file path.
    """
    fpath.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "model_cfg_type": type(model_cfg).__qualname__,
        "model_cfg": dataclasses.asdict(model_cfg),
        "objective_cfg_type": type(objective_cfg).__qualname__,
        "objective_cfg": dataclasses.asdict(objective_cfg),
    }
    with open(fpath, "wb") as fd:
        fd.write((json.dumps(header, default=str) + "\n").encode("utf-8"))
        eqx.tree_serialise_leaves(fd, model)


def _parse_model_cfg(header: dict) -> Config:
    cfg_type = header["model_cfg_type"]
    cfg_dict = header["model_cfg"]
    if cfg_type == "Heatmap":
        # Convert dinov3_ckpt string back to Path.
        cfg_dict["dinov3_ckpt"] = pathlib.Path(cfg_dict["dinov3_ckpt"])
        return heatmap.Heatmap(**cfg_dict)
    msg = f"Unknown model config type: '{cfg_type}'"
    raise ValueError(msg)


def _parse_objective_cfg(header: dict) -> btx.objectives.Config:
    cfg_type = header["objective_cfg_type"]
    cfg_dict = header["objective_cfg"]
    if cfg_type == "Config":
        return btx.objectives.heatmap.Config(**cfg_dict)
    if cfg_type == "Coords":
        return btx.objectives.Coords(**cfg_dict)
    msg = f"Unknown objective config type: '{cfg_type}'"
    raise ValueError(msg)


@beartype.beartype
def load_ckpt(
    fpath: pathlib.Path, *, key: chex.PRNGKey | None = None
) -> tuple[eqx.Module, Config, btx.objectives.Config]:
    """Load a trained model checkpoint.

    Args:
        fpath: Path to checkpoint file.
        key: Optional PRNG key for model construction. Weights are overwritten by checkpoint, so the key value doesn't matter.

    Returns:
        Tuple of (model, model_cfg, objective_cfg).
    """
    if key is None:
        key = jax.random.key(seed=0)
    with open(fpath, "rb") as fd:
        header = json.loads(fd.readline())
        model_cfg = _parse_model_cfg(header)
        objective_cfg = _parse_objective_cfg(header)
        model = make(model_cfg, key)
        model = eqx.tree_deserialise_leaves(fd, model)
    return model, model_cfg, objective_cfg
