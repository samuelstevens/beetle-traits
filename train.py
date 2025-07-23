# train.py
import dataclasses

import beartype
import jax
import tyro

import btx.data
import btx.modeling


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    seed: int = 17
    """Random seed."""
    model: btx.modeling.Config = btx.modeling.Config()
    """Neural network config."""
    data: btx.data.HawaiiConfig = btx.data.HawaiiConfig()
    """Data config."""
    tags: list[str] = dataclasses.field(default_factory=list)
    """List of wandb tags to include."""


@beartype.beartype
def main(cfg: Config):
    key = jax.random.key(seed=cfg.seed)
    dataset = btx.data.HawaiiDataset(cfg.data)
    breakpoint()
    model = btx.modeling.make(cfg.model, key)


if __name__ == "__main__":
    main(tyro.cli(Config))
