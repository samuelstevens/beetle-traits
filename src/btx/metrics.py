import beartype
from jaxtyping import Array, Float, jaxtyped


@jaxtyped(typechecker=beartype.beartype)
def apply_affine_jax(
    affine_b33: Float[Array, "batch 3 3"],
    points_bl22: Float[Array, "batch lines 2 2"],
) -> Float[Array, "batch lines 2 2"]:
    raise NotImplementedError()


@jaxtyped(typechecker=beartype.beartype)
def choose_endpoint_matching(
    pred_bl22: Float[Array, "batch lines 2 2"],
    tgt_bl22: Float[Array, "batch lines 2 2"],
) -> Float[Array, "batch lines 2 2"]:
    raise NotImplementedError()


@jaxtyped(typechecker=beartype.beartype)
def get_metric_mask_cm(
    scalebar_b22: Float[Array, "batch 2 2"],
    metric_mask_cm_b: Float[Array, " batch"],
    *,
    min_px_per_cm: float,
) -> tuple[Float[Array, " batch"], Float[Array, " batch"]]:
    raise NotImplementedError()
