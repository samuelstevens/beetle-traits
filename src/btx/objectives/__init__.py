import abc
import dataclasses

import beartype
import jax
from jaxtyping import Array, Float, jaxtyped


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class LossAux:
    """Objective-level outputs in augmented-image space.

    `metrics` contains objective-specific per-sample diagnostics keyed by stable metric names. Each metric value has shape `[batch]`.
    """

    loss: Float[Array, ""]
    sample_loss: Float[Array, " batch"]
    preds: Float[Array, "batch 2 2 2"]
    metrics: dict[str, Float[Array, " batch"]]


class Obj(abc.ABC):
    """Runtime objective interface used by the training step."""

    @property
    @abc.abstractmethod
    def key(self) -> str:
        """Stable key used to namespace objective-specific metrics."""
        pass

    @abc.abstractmethod
    def get_loss_aux(
        self,
        *,
        preds_raw: jax.Array,
        batch: dict[str, jax.Array],
        mask_line: jax.Array,
    ) -> LossAux:
        """Compute objective loss plus objective-specific diagnostics."""
        pass


from btx.objectives.coords import Coords, CoordsObj
from btx.objectives.heatmap import Config as Heatmap
from btx.objectives.heatmap import HeatmapObj

Config = Coords | Heatmap

__all__ = ["Config", "Coords", "CoordsObj", "Heatmap", "HeatmapObj", "LossAux", "Obj"]
