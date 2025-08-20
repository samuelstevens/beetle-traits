Module btx.data
===============

Sub-modules
-----------
* btx.data.beetlepalooza
* btx.data.biorepo
* btx.data.hawaii

Classes
-------

`BioRepoConfig()`
:   

`HawaiiConfig(hf_root: pathlib.Path = PosixPath('data/hawaii'), annotations: pathlib.Path = PosixPath('data/hawaii-formatted/annotations.json'), include_polylines: bool = True, split: Literal['train', 'val'] = 'train', seed: int = 0)`
:   Config(hf_root: pathlib.Path = PosixPath('data/hawaii'), annotations: pathlib.Path = PosixPath('data/hawaii-formatted/annotations.json'), include_polylines: bool = True, split: Literal['train', 'val'] = 'train', seed: int = 0)

    ### Instance variables

    `annotations: pathlib.Path`
    :   Path to the annotations.json file made by running format_hawaii.py.

    `hf_root: pathlib.Path`
    :   Path to the dataset root downloaded from HuggingFace.

    `include_polylines: bool`
    :   Whether to include polylines (lines with more than 2 points).

    `seed: int`
    :

    `split: Literal['train', 'val']`
    :   Which split.

`BeetlePaloozaConfig()`
:   

`BioRepoDataset()`
:   

`HawaiiDataset(cfg: btx.data.hawaii.Config)`
:   Interface for datasources where storage supports efficient random access.
    
    Note that `__repr__` has to be additionally implemented to make checkpointing
    work with this source.

    ### Ancestors (in MRO)

    * grain._src.python.data_sources.RandomAccessDataSource
    * typing.Protocol
    * typing.Generic

`BeetlePaloozaDataset()`
: