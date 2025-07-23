# train.py
import dataclasses

import beartype
import tyro

import beetra.modeling


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    model: beetra.modeling.Config = beetra.modeling.Config()
    tags: list[str] = dataclasses.field(default_factory=list)
    """List of wandb tags to include."""


@beartype.beartype
def main(cfg: Config):
    model = beetra.modeling.make(cfg.model)


if __name__ == "__main__":
    main(tyro.cli(Config))
