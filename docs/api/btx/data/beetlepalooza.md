Module btx.data.beetlepalooza
=============================
BeetlePalooza dataset

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

Classes
-------

`Config(go: bool = True, hf_root: pathlib.Path = PosixPath('data/beetlepalooza'), annotations: pathlib.Path = PosixPath('data/beetlepalooza-formatted/annotations.json'), include_polylines: bool = False, annotators: list[str] = <factory>)`
:   Config(go: bool = True, hf_root: pathlib.Path = PosixPath('data/beetlepalooza'), annotations: pathlib.Path = PosixPath('data/beetlepalooza-formatted/annotations.json'), include_polylines: bool = False, annotators: list[str] = <factory>)

    ### Ancestors (in MRO)

    * btx.data.utils.Config
    * abc.ABC

    ### Instance variables

    `annotations: pathlib.Path`
    :   Path to the annotations.json file.

    `annotators: list[str]`
    :   According to Aly, we need to filter by `annotator="IsaFluck"`. See https://hdrimageomics.slack.com/archives/C08T6MCFME1/p1763993618130059?thread_ts=1763935266.449869&cid=C08T6MCFME1 for more context.

    `dataset`
    :

    `go: bool`
    :   Whether to include this dataset in training.

    `hf_root: pathlib.Path`
    :   Path to the dataset root downloaded from HuggingFace.

    `include_polylines: bool`
    :   Whether to include polylines (lines with more than 2 points).

    `key: str`
    :

`Dataset(cfg: btx.data.beetlepalooza.Config)`
:   Interface for datasources where storage supports efficient random access.
    
    Note that `__repr__` has to be additionally implemented to make checkpointing
    work with this source.

    ### Ancestors (in MRO)

    * btx.data.utils.Dataset
    * grain._src.python.data_sources.RandomAccessDataSource
    * typing.Protocol
    * typing.Generic
    * abc.ABC

    ### Instance variables

    `cfg: btx.data.beetlepalooza.Config`
    :