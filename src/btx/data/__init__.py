import typing as tp

import beartype
import grain

from . import transforms
from .beetlepalooza import Config as BeetlePaloozaConfig
from .beetlepalooza import Dataset as BeetlePaloozaDataset
from .biorepo import Config as BioRepoConfig
from .biorepo import Dataset as BioRepoDataset
from .hawaii import Config as HawaiiConfig
from .hawaii import Dataset as HawaiiDataset
from .transforms import AugmentConfig
from .utils import Config, Dataset

__all__ = [
    "Config",
    "Dataset",
    "AugmentConfig",
    "BioRepoConfig",
    "BioRepoDataset",
    "HawaiiConfig",
    "HawaiiDataset",
    "BeetlePaloozaConfig",
    "BeetlePaloozaDataset",
    "make_dataloader",
]


@beartype.beartype
def make_dataloader(
    dss: list[Dataset],
    aug_cfgs: list[AugmentConfig],
    *,
    seed: int,
    batch_size: int,
    n_workers: int,
    shuffle: bool,
    finite: bool,
    is_train: bool,
):
    """Build a mixed Grain dataloader from one or more dataset sources.

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
    """

    msg = f"Expected matching lengths: {len(dss)} datasets, {len(aug_cfgs)} aug configs"
    assert len(dss) == len(aug_cfgs), msg

    datasets = []
    weights = []

    for ds, aug_cfg in zip(dss, aug_cfgs):
        source = tp.cast(tp.Sequence[object], ds)
        mapped_ds = grain.MapDataset.source(source).seed(seed)
        if shuffle:
            mapped_ds = mapped_ds.shuffle()

        for i, tfm in enumerate(transforms.make_transforms(aug_cfg, is_train=is_train)):
            if isinstance(tfm, grain.transforms.RandomMap):
                mapped_ds = mapped_ds.random_map(tfm, seed=seed + i)
            else:
                mapped_ds = mapped_ds.map(tfm)

        datasets.append(mapped_ds)
        weights.append(len(ds))

    assert datasets, "No datasets provided."

    if len(datasets) == 1:
        mixed = datasets[0]
    else:
        total = sum(weights)
        assert total > 0, "All datasets are empty."
        mix_weights = [w / total for w in weights]
        mixed = grain.MapDataset.mix(datasets, weights=mix_weights)

    mixed = mixed.repeat(num_epochs=None if not finite else 1)
    mixed = mixed.batch(batch_size=batch_size, drop_remainder=False)

    iter_ds = mixed.to_iter_dataset(
        read_options=grain.ReadOptions(num_threads=2, prefetch_buffer_size=8)
    )

    if n_workers > 0:
        iter_ds = iter_ds.mp_prefetch(
            grain.multiprocessing.MultiprocessingOptions(
                num_workers=n_workers, per_worker_buffer_size=2
            )
        )

    return iter_ds
