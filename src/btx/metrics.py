import beartype
import jax.numpy as jnp
from jaxtyping import Array, Float, jaxtyped

# Minimum valid pixels-per-centimeter for cm metrics. Smaller or non-finite scalebars are masked out.
MIN_PX_PER_CM: float = 1e-6


@jaxtyped(typechecker=beartype.beartype)
def apply_affine(
    affine_b33: Float[Array, "batch 3 3"], points_bl22: Float[Array, "batch lines 2 2"]
) -> Float[Array, "batch lines 2 2"]:
    b, lines, points, coords = points_bl22.shape
    msg = f"Expected points shape [batch, lines, 2, 2], got {points_bl22.shape}"
    assert points == 2 and coords == 2, msg
    msg = f"Expected affine shape [batch, 3, 3], got {affine_b33.shape}"
    assert affine_b33.shape == (b, 3, 3), msg

    pts = jnp.reshape(points_bl22, (b, lines * points, 2))
    ones = jnp.ones((b, lines * points, 1), dtype=points_bl22.dtype)
    hom = jnp.concatenate([pts, ones], axis=-1)
    out_h = jnp.einsum("bij,bpj->bpi", affine_b33, hom)
    return jnp.reshape(out_h[:, :, :2], points_bl22.shape)


@jaxtyped(typechecker=beartype.beartype)
def choose_endpoint_matching(
    pred_bl22: Float[Array, "batch lines 2 2"],
    tgt_bl22: Float[Array, "batch lines 2 2"],
) -> Float[Array, "batch lines 2 2"]:
    msg = f"Expected matching shapes, got {pred_bl22.shape} and {tgt_bl22.shape}"
    assert pred_bl22.shape == tgt_bl22.shape, msg
    swapped = tgt_bl22[:, :, ::-1, :]

    direct_cost = jnp.linalg.norm(pred_bl22 - tgt_bl22, axis=-1).sum(axis=-1)
    swapped_cost = jnp.linalg.norm(pred_bl22 - swapped, axis=-1).sum(axis=-1)
    use_swapped = swapped_cost < direct_cost
    use_swapped = use_swapped[:, :, None, None]
    return jnp.where(use_swapped, swapped, tgt_bl22)


@jaxtyped(typechecker=beartype.beartype)
def get_metric_mask_cm(
    scalebar_b22: Float[Array, "batch 2 2"], metric_mask_cm_b: Float[Array, " batch"]
) -> tuple[Float[Array, " batch"], Float[Array, " batch"]]:
    msg = f"Expected scalebar shape [batch, 2, 2], got {scalebar_b22.shape}"
    assert scalebar_b22.ndim == 3 and scalebar_b22.shape[1:] == (2, 2), msg
    msg = (
        "Expected metric mask shape [batch], got "
        f"{metric_mask_cm_b.shape} with scalebar {scalebar_b22.shape}"
    )
    assert (
        metric_mask_cm_b.ndim == 1
        and metric_mask_cm_b.shape[0] == scalebar_b22.shape[0]
    ), msg

    p0 = scalebar_b22[:, 0, :]
    p1 = scalebar_b22[:, 1, :]
    px_per_cm = jnp.linalg.norm(p1 - p0, axis=-1)
    is_valid = jnp.isfinite(px_per_cm) & (px_per_cm > MIN_PX_PER_CM)
    metric_mask_cm = metric_mask_cm_b * is_valid.astype(metric_mask_cm_b.dtype)
    return metric_mask_cm, px_per_cm
