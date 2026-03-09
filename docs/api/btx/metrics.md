Module btx.metrics
==================

Functions
---------

`apply_affine(affine_b33: jaxtyping.Float[Array, 'batch 3 3'], points_bl22: jaxtyping.Float[Array, 'batch lines 2 2']) ‑> jaxtyping.Float[Array, 'batch lines 2 2']`
:   

`choose_endpoint_matching(pred_bl22: jaxtyping.Float[Array, 'batch lines 2 2'], tgt_bl22: jaxtyping.Float[Array, 'batch lines 2 2']) ‑> jaxtyping.Float[Array, 'batch lines 2 2']`
:   

`get_scalebar_mask(scalebar_b22: jaxtyping.Float[Array, 'batch 2 2'], scalebar_valid_b: jaxtyping.Bool[Array, 'batch']) ‑> tuple[jaxtyping.Bool[Array, 'batch'], jaxtyping.Float[Array, 'batch']]`
: