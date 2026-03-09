Module btx.objectives
=====================

Sub-modules
-----------
* btx.objectives.coords
* btx.objectives.heatmap

Classes
-------

`Heatmap(image_size: int = 256, heatmap_size: int = 64, sigma: float = 2.0, eps: float = 1e-08)`
:   Configuration for heatmap target generation and coordinate decoding.

    ### Instance variables

    `downsample: int`
    :   Integer image-to-heatmap stride (`image_size // heatmap_size`).

    `eps: float`
    :   Small positive constant used for safe normalizations.

    `heatmap_size: int`
    :   Square heatmap size in pixels used for supervision and decoding.

    `image_size: int`
    :   Square image size in pixels used by the model input pipeline.

    `sigma: float`
    :   Gaussian standard deviation in heatmap pixels.

    ### Methods

    `get_obj(self) ‑> btx.objectives.Obj`
    :

`Coords()`
:   Direct coordinate regression objective (pointwise MSE).

    ### Methods

    `get_obj(self) ‑> btx.objectives.Obj`
    :

`CoordsObj(cfg: btx.objectives.coords.Coords)`
:   CoordsObj(cfg: btx.objectives.coords.Coords)

    ### Ancestors (in MRO)

    * btx.objectives.Obj
    * abc.ABC

    ### Instance variables

    `cfg: btx.objectives.coords.Coords`
    :

`HeatmapObj(cfg: btx.objectives.heatmap.Config)`
:   HeatmapObj(cfg: btx.objectives.heatmap.Config)

    ### Ancestors (in MRO)

    * btx.objectives.Obj
    * abc.ABC

    ### Instance variables

    `cfg: btx.objectives.heatmap.Config`
    :

`LossAux(loss: jaxtyping.Float[Array, ''], sample_loss: jaxtyping.Float[Array, 'batch'], preds: jaxtyping.Float[Array, 'batch 2 2 2'], metrics: dict[str, jaxtyping.Float[Array, 'batch']])`
:   Objective-level outputs in augmented-image space.
    
    `metrics` contains objective-specific per-sample diagnostics keyed by stable metric names. Each metric value has shape `[batch]`.

    ### Instance variables

    `loss: jaxtyping.Float[Array, '']`
    :

    `metrics: dict[str, jaxtyping.Float[Array, 'batch']]`
    :

    `preds: jaxtyping.Float[Array, 'batch 2 2 2']`
    :

    `sample_loss: jaxtyping.Float[Array, 'batch']`
    :

`Obj()`
:   Runtime objective interface used by the training step.

    ### Ancestors (in MRO)

    * abc.ABC

    ### Descendants

    * btx.objectives.coords.CoordsObj
    * btx.objectives.heatmap.HeatmapObj

    ### Instance variables

    `key: str`
    :   Stable key used to namespace objective-specific metrics.

    ### Methods

    `get_loss_aux(self, *, preds_raw: jax.Array, batch: dict[str, jax.Array], mask_line: jax.Array) ‑> btx.objectives.LossAux`
    :   Compute objective loss plus objective-specific diagnostics.