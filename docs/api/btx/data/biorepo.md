Module btx.data.biorepo
=======================

Classes
-------

`Config(go: bool = True, root: pathlib.Path = PosixPath('data/biorepo'), annotations: pathlib.Path = PosixPath('data/biorepo-formatted/annotations.json'), unlabeled_annotations: pathlib.Path = PosixPath('/fs/scratch/PAS2136/cain429/unlabeled_biorepo_annotations.csv'), split: Literal['train', 'val', 'test', 'all', 'unlabeled'] = 'val')`
:   Config(go: bool = True, root: pathlib.Path = PosixPath('data/biorepo'), annotations: pathlib.Path = PosixPath('data/biorepo-formatted/annotations.json'), unlabeled_annotations: pathlib.Path = PosixPath('/fs/scratch/PAS2136/cain429/unlabeled_biorepo_annotations.csv'), split: Literal['train', 'val', 'test', 'all', 'unlabeled'] = 'val')

    ### Ancestors (in MRO)

    * btx.data.utils.Config
    * abc.ABC

    ### Instance variables

    `annotations: pathlib.Path`
    :   Path to the annotations.json file made by running format_biorepo.py.

    `dataset`
    :

    `go: bool`
    :   Whether to include this dataset.

    `key: str`
    :

    `root: pathlib.Path`
    :   Path to the dataset root

    `split: Literal['train', 'val', 'test', 'all', 'unlabeled']`
    :   Which split.

    `unlabeled_annotations: pathlib.Path`
    :

`Dataset(cfg: btx.data.biorepo.Config)`
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

    `cfg: btx.data.biorepo.Config`
    :