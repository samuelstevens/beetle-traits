Module btx.modeling
===================
Model construction and checkpoint save/load.

Checkpoint file format (following the dinov3.dump/load pattern):
  Line 1: JSON dict with 'model_cfg' and 'objective_cfg'
  Remaining bytes: eqx.tree_serialise_leaves binary data

Sub-modules
-----------
* btx.modeling.dinov3
* btx.modeling.frozen
* btx.modeling.heatmap
* btx.modeling.toy

Functions
---------

`load_ckpt(fpath: pathlib.Path, *, key: jax.Array | None = None) ‑> tuple[equinox._module._module.Module, btx.modeling.toy.Toy | btx.modeling.frozen.Frozen | btx.modeling.heatmap.Heatmap, btx.objectives.coords.Coords | btx.objectives.heatmap.Config]`
:   Load a trained model checkpoint.
    
    Args:
        fpath: Path to checkpoint file.
        key: Optional PRNG key for model construction. Weights are overwritten by checkpoint, so the key value doesn't matter.
    
    Returns:
        Tuple of (model, model_cfg, objective_cfg).

`make(cfg: btx.modeling.toy.Toy | btx.modeling.frozen.Frozen | btx.modeling.heatmap.Heatmap, key: jax.Array) ‑> equinox._module._module.Module`
:   Construct a model from configuration.
    
    Args:
        cfg: Model configuration variant.
        key: Random key used for initializing trainable layers.
    
    Returns:
        Instantiated Equinox model.

`save_ckpt(model: equinox._module._module.Module, model_cfg: btx.modeling.toy.Toy | btx.modeling.frozen.Frozen | btx.modeling.heatmap.Heatmap, objective_cfg: btx.objectives.coords.Coords | btx.objectives.heatmap.Config, fpath: pathlib.Path)`
:   Save a trained model checkpoint with embedded config.
    
    Args:
        model: Trained model to serialize.
        model_cfg: Model architecture config used to construct the model.
        objective_cfg: Objective config needed for inference decoding.
        fpath: Output file path.