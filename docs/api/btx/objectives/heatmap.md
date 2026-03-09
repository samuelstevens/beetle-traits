Module btx.objectives.heatmap
=============================
Heatmap target/loss/decoding utilities for the current line-endpoint task.

Channel contract is intentionally fixed to four endpoint channels:
`[width_p0, width_p1, length_p0, length_p1]`.
If we later switch to a different annotation shape (for example, polylines with
more than two endpoints), update `CHANNEL_NAMES` and the reshape assumptions in
`make_targets`/`heatmaps_to_coords`, plus their tests.

Functions
---------

`get_diagnostics(logits_bchw: jaxtyping.Float[Array, 'batch channels height width'], *, cfg: btx.objectives.heatmap.Config) ‑> tuple[jaxtyping.Float[Array, 'batch channels'], jaxtyping.Float[Array, 'batch channels'], jaxtyping.Float[Array, 'batch channels']]`
:   Compute collapse diagnostics from predicted heatmap logits.
    
    Args:
        logits_bchw: Predicted heatmap logits with shape `[batch, channels, H, W]`.
        cfg: Heatmap geometry and numerical configuration.
    
    Returns:
        Tuple `(max_logit_bc, entropy_bc, near_uniform_bc)` where:
        - `max_logit_bc` is max logit per sample/channel.
        - `entropy_bc` is spatial softmax entropy per sample/channel.
        - `near_uniform_bc` is 1.0 when normalized entropy exceeds threshold, else 0.0.

`heatmap_ce_loss(pred_chw: jaxtyping.Float[Array, 'channels height width'], tgt_chw: jaxtyping.Float[Array, 'channels height width'], loss_mask_l: jaxtyping.Float[Array, '2'], *, cfg: btx.objectives.heatmap.Config) ‑> jaxtyping.Float[Array, '']`
:   Compute permutation-invariant masked CE for endpoint heatmaps.

`heatmap_loss(pred_chw: jaxtyping.Float[Array, 'channels height width'], tgt_chw: jaxtyping.Float[Array, 'channels height width'], loss_mask_l: jaxtyping.Float[Array, '2'], *, cfg: btx.objectives.heatmap.Config) ‑> jaxtyping.Float[Array, '']`
:   Compute permutation-invariant masked MSE for endpoint heatmaps.
    
    Args:
        pred_chw: Predicted heatmap logits with channel order `[width_p0, width_p1, length_p0, length_p1]`.
        tgt_chw: Target endpoint heatmaps with the same channel order as `pred_chw`.
        loss_mask_l: Per-line supervision mask for `[width, length]`.
        cfg: Heatmap geometry and numerical configuration.
    
    Returns:
        Scalar masked MSE value.

`heatmap_to_image_udp(points_b2: jaxtyping.Float[Array, '*batch 2'], *, cfg: btx.objectives.heatmap.Config) ‑> jaxtyping.Float[Array, '*batch 2']`
:   Map heatmap-space coordinates back to image-space coordinates using UDP.
    
    Args:
        points_b2: Coordinates in heatmap space with trailing `[..., 2]` order `(hx, hy)`.
        cfg: Heatmap geometry and numerical configuration.
    
    Returns:
        Coordinates in image space with trailing `[..., 2]` order `(x, y)`.

`heatmaps_to_coords(logits_chw: jaxtyping.Float[Array, 'channels height width'], *, cfg: btx.objectives.heatmap.Config) ‑> jaxtyping.Float[Array, '2 2 2']`
:   Decode four endpoint heatmaps into line-endpoint coordinates.
    
    Args:
        logits_chw: Predicted endpoint logits with channel order `[width_p0, width_p1, length_p0, length_p1]`.
        cfg: Heatmap geometry and numerical configuration.
    
    Returns:
        Coordinates in image space with shape `[2, 2, 2]` and order `[line, endpoint, (x, y)]`.

`image_to_heatmap_udp(points_b2: jaxtyping.Float[Array, '*batch 2'], *, cfg: btx.objectives.heatmap.Config) ‑> jaxtyping.Float[Array, '*batch 2']`
:   Map image-space coordinates to heatmap-space coordinates using UDP.
    
    Args:
        points_b2: Coordinates in image space with trailing `[..., 2]` order `(x, y)`.
        cfg: Heatmap geometry and numerical configuration.
    
    Returns:
        Coordinates in heatmap space with trailing `[..., 2]` order `(hx, hy)`.

`make_targets(points_l22: jaxtyping.Float[Array, '2 2 2'], *, cfg: btx.objectives.heatmap.Config) ‑> jaxtyping.Float[Array, '4 height width']`
:   Build endpoint heatmap targets from line-endpoint coordinates.
    
    Args:
        points_l22: Line-endpoint coordinates in image space with shape `[line, endpoint, (x, y)]`.
        cfg: Heatmap geometry and numerical configuration.
    
    Returns:
        Endpoint heatmaps with channel order
        `[width_p0, width_p1, length_p0, length_p1]`.

Classes
-------

`Config(image_size: int = 256, heatmap_size: int = 64, sigma: float = 2.0, eps: float = 1e-08)`
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

`HeatmapObj(cfg: btx.objectives.heatmap.Config)`
:   HeatmapObj(cfg: btx.objectives.heatmap.Config)

    ### Ancestors (in MRO)

    * btx.objectives.Obj
    * abc.ABC

    ### Instance variables

    `cfg: btx.objectives.heatmap.Config`
    :