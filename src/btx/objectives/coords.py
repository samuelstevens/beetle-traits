"""Direct coordinate regression objective (pointwise MSE)."""

import dataclasses

import beartype
import jax
import jax.numpy as jnp

from btx.objectives import LossAux, Obj


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Coords:
    """Direct coordinate regression objective (pointwise MSE)."""

    def get_obj(self) -> Obj:
        return CoordsObj(self)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class CoordsObj(Obj):
    cfg: Coords

    @property
    def key(self) -> str:
        return "coords"

    def get_loss_aux(
        self, *, preds_raw: jax.Array, batch: dict[str, jax.Array], mask_line: jax.Array
    ) -> LossAux:
        msg = (
            "Expected coordinate predictions with shape [batch, 2, 2, 2], got "
            f"{preds_raw.shape}"
        )
        assert preds_raw.shape[1:] == (2, 2, 2), msg
        preds = preds_raw

        squared_error = (preds - batch["tgt"]) ** 2
        mask = jnp.reshape(mask_line, (mask_line.shape[0], 2, 1, 1))
        masked_error = squared_error * mask
        active_values = jnp.sum(mask) * 4
        active_values_safe = jnp.maximum(active_values, 1.0)
        loss = jnp.sum(masked_error) / active_values_safe

        sample_active_values = jnp.sum(mask, axis=(1, 2, 3)) * 4
        sample_loss = jnp.where(
            sample_active_values > 0,
            jnp.sum(masked_error, axis=(1, 2, 3)) / sample_active_values,
            jnp.nan,
        )

        return LossAux(
            loss=loss,
            sample_loss=sample_loss,
            preds=preds,
            metrics={},
        )
