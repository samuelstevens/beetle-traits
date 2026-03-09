Module btx.data
===============

Sub-modules
-----------
* btx.data.beetlepalooza
* btx.data.biorepo
* btx.data.hawaii
* btx.data.transforms
* btx.data.utils

Functions
---------

`make_dataloader(dss: list[btx.data.utils.Dataset], aug_cfgs: list[btx.data.transforms.AugmentConfig], *, seed: int, batch_size: int, n_workers: int, shuffle: bool, finite: bool, is_train: bool)`
:   Build a mixed Grain dataloader from one or more dataset sources.
    
    Args:
        dss: Dataset sources to include in this loader.
        aug_cfgs: Per-dataset augmentation configs, same length as dss.
        seed: Random seed for shuffling and random transforms.
        batch_size: Batch size.
        n_workers: Number of dataloader workers. 0 means no multiprocessing.
        shuffle: Whether to shuffle each source dataset before transforms.
        finite: Whether the iterator should stop after one epoch (True) or repeat forever (False).
        is_train: Whether to build the train transform pipeline (True) or eval pipeline (False).
    
    Returns:
        Grain iterable dataset yielding transformed, batched samples.

Classes
-------

`AugmentConfig(go: bool = True, size: int = 256, crop: bool = True, crop_scale_min: float = 0.5, crop_scale_max: float = 1.0, crop_ratio_min: float = 0.75, crop_ratio_max: float = 1.333, hflip_prob: float = 0.5, vflip_prob: float = 0.5, rotation_prob: float = 0.75, brightness: float = 0.2, contrast: float = 0.2, saturation: float = 0.2, hue: float = 0.1, color_jitter_prob: float = 1.0, normalize: bool = True, oob_policy: Literal['mask_any_oob', 'mask_all_oob', 'supervise_oob'] = 'supervise_oob', min_px_per_cm: float = 1e-06)`
:   Configuration for train-time augmentation and cm-metric masking behavior.

    ### Instance variables

    `brightness: float`
    :   Color jitter brightness strength.

    `color_jitter_prob: float`
    :   Probability of applying color jitter when any jitter strength is nonzero.

    `contrast: float`
    :   Color jitter contrast strength.

    `crop: bool`
    :   Whether to use RandomResizedCrop (True) or plain Resize (False) during training. Set to False for tightly-cropped datasets where random cropping would lose too much content.

    `crop_ratio_max: float`
    :   Maximum random crop aspect ratio for RandomResizedCrop.

    `crop_ratio_min: float`
    :   Minimum random crop aspect ratio for RandomResizedCrop.

    `crop_scale_max: float`
    :   Maximum random crop area scale for RandomResizedCrop.

    `crop_scale_min: float`
    :   Minimum random crop area scale for RandomResizedCrop.

    `go: bool`
    :   Whether to enable the augmentation pipeline.

    `hflip_prob: float`
    :   Probability of applying horizontal flip.

    `hue: float`
    :   Color jitter hue strength.

    `min_px_per_cm: float`
    :   Minimum valid scalebar length in pixels for cm metrics.

    `normalize: bool`
    :   Whether to apply ImageNet normalization at the end of the transform pipeline.

    `oob_policy: Literal['mask_any_oob', 'mask_all_oob', 'supervise_oob']`
    :   Out-of-bounds supervision policy.

    `rotation_prob: float`
    :   Probability of applying a random rotation (uniform 0-360 degrees).

    `saturation: float`
    :   Color jitter saturation strength.

    `size: int`
    :   Output image side length in pixels. Fixed to 256 for this experiment.

    `vflip_prob: float`
    :   Probability of applying vertical flip.

`Config()`
:   Helper class that provides a standard way to create an ABC using
    inheritance.

    ### Ancestors (in MRO)

    * abc.ABC

    ### Descendants

    * btx.data.beetlepalooza.Config
    * btx.data.biorepo.Config
    * btx.data.hawaii.Config

    ### Instance variables

    `dataset: type['Dataset']`
    :

    `key: str`
    :

`BioRepoConfig(go: bool = True, root: pathlib.Path = PosixPath('data/biorepo'), annotations: pathlib.Path = PosixPath('data/biorepo-formatted/annotations.json'), unlabeled_annotations: pathlib.Path = PosixPath('/fs/scratch/PAS2136/cain429/unlabeled_biorepo_annotations.csv'), split: Literal['train', 'val', 'test', 'all', 'unlabeled'] = 'val')`
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

`HawaiiConfig(go: bool = True, hf_root: pathlib.Path = PosixPath('data/hawaii'), annotations: pathlib.Path = PosixPath('data/hawaii-formatted/annotations.json'), include_polylines: bool = True, split: Literal['train', 'val', 'all'] = 'train', seed: int = 0, min_val_groups: int = 2, min_val_beetles: int = 20)`
:   Config(go: bool = True, hf_root: pathlib.Path = PosixPath('data/hawaii'), annotations: pathlib.Path = PosixPath('data/hawaii-formatted/annotations.json'), include_polylines: bool = True, split: Literal['train', 'val', 'all'] = 'train', seed: int = 0, min_val_groups: int = 2, min_val_beetles: int = 20)

    ### Ancestors (in MRO)

    * btx.data.utils.Config
    * abc.ABC

    ### Instance variables

    `annotations: pathlib.Path`
    :   Path to the annotations.json file made by running format_hawaii.py.

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

    `min_val_beetles: int`
    :   Minimum beetles per species in validation.

    `min_val_groups: int`
    :   Minimum group images per species in validation.

    `seed: int`
    :   Random seed for split.

    `split: Literal['train', 'val', 'all']`
    :   Which split.

`BeetlePaloozaConfig(go: bool = True, hf_root: pathlib.Path = PosixPath('data/beetlepalooza'), annotations: pathlib.Path = PosixPath('data/beetlepalooza-formatted/annotations.json'), include_polylines: bool = False, annotators: list[str] = <factory>)`
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

`Dataset(*args, **kwargs)`
:   Interface for datasources where storage supports efficient random access.
    
    Note that `__repr__` has to be additionally implemented to make checkpointing
    work with this source.

    ### Ancestors (in MRO)

    * grain._src.python.data_sources.RandomAccessDataSource
    * typing.Protocol
    * typing.Generic
    * abc.ABC

    ### Descendants

    * btx.data.beetlepalooza.Dataset
    * btx.data.biorepo.Dataset
    * btx.data.hawaii.Dataset

    ### Instance variables

    `cfg: btx.data.utils.Config`
    :

`BioRepoDataset(cfg: btx.data.biorepo.Config)`
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

`HawaiiDataset(cfg: btx.data.hawaii.Config)`
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

    `cfg: btx.data.hawaii.Config`
    :

`BeetlePaloozaDataset(cfg: btx.data.beetlepalooza.Config)`
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