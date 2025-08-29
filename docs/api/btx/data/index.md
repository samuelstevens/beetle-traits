Module btx.data
===============

Sub-modules
-----------
* btx.data.beetlepalooza
* btx.data.biorepo
* btx.data.hawaii
* btx.data.utils

Classes
-------

`BioRepoConfig()`
:   

`HawaiiConfig(hf_root: pathlib.Path = PosixPath('data/hawaii'), annotations: pathlib.Path = PosixPath('data/hawaii-formatted/annotations.json'), include_polylines: bool = True, split: Literal['train', 'val'] = 'train', seed: int = 0, min_val_groups: int = 2, min_val_beetles: int = 20, n_workers: int = 4, batch_size: int = 16)`
:   Config(hf_root: pathlib.Path = PosixPath('data/hawaii'), annotations: pathlib.Path = PosixPath('data/hawaii-formatted/annotations.json'), include_polylines: bool = True, split: Literal['train', 'val'] = 'train', seed: int = 0, min_val_groups: int = 2, min_val_beetles: int = 20, n_workers: int = 4, batch_size: int = 16)

    ### Instance variables

    `annotations: pathlib.Path`
    :   Path to the annotations.json file made by running format_hawaii.py.

    `batch_size: int`
    :

    `hf_root: pathlib.Path`
    :   Path to the dataset root downloaded from HuggingFace.

    `include_polylines: bool`
    :   Whether to include polylines (lines with more than 2 points).

    `min_val_beetles: int`
    :   Minimum beetles per species in validation.

    `min_val_groups: int`
    :   Minimum group images per species in validation.

    `n_workers: int`
    :

    `seed: int`
    :   Random seed for split.

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