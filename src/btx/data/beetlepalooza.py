"""BeetlePalooza dataset

Context: This dataset will be mixed with Hawaii and BioRepo via grain.MapDataset, so we need consistent per-beetle samples matching utils.Sample.
Problem - annotator variation: Labels come from multiple annotators, so targets may be inconsistent across copies.
Why fix it: Mixed-quality labels would leak noise into the shared training pipeline and destabilize convergence.
Proposed solution: Filter rows to a trusted annotator list (currently ['isa fluck']) before producing samples.

Problem - elytra width quality: Elytra width measurements are known junk.
Why fix it: Downstream models still expect a fixed target shape, so dropping them would break shapes while keeping them may mislead metrics.
Proposed solution: Keep widths in the sample for shape compatibility but note they are low-trust until a better handling strategy exists.

Problem - annotation count drift: Some images have 2-6 annotations instead of exactly 2.
Why fix it: Variable annotation counts would break batching and target shapes.
Proposed solution: Filtering by annotator may collapse this to 2; otherwise pick a deterministic rule (e.g., first two or majority vote) to enforce exactly two annotations before batching.
"""

import dataclasses
import pathlib

import beartype
import grain

from . import utils


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config(utils.Config):
    hf_root: pathlib.Path = pathlib.Path("data/beetlepalooza")
    """Path to the dataset root downloaded from HuggingFace."""
    annotations: pathlib.Path = pathlib.Path(
        "data/beetlepalooza-formatted/annotations.json"
    )
    """Path to the annotations.json file."""

    include_polylines: bool = False
    """Whether to include polylines (lines with more than 2 points)."""
    annotators: list[str] = dataclasses.field(default_factory=lambda: ["isa fluck"])
    """According to Aly, we need to filter by `annotator="isa fluck"`. See https://hdrimageomics.slack.com/archives/C08T6MCFME1/p1763993618130059?thread_ts=1763935266.449869&cid=C08T6MCFME1 for more context."""

    @property
    def dataset(self):
        return Dataset

    def __post_init__(self):
        # TODO: Check that hf_root exists and is a directory
        # TODO: Check that annotations.json exists and is a file.
        pass


@beartype.beartype
class Dataset(grain.sources.RandomAccessDataSource):
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> utils.Sample:
        """Load image and annotations for given index."""
