Module btx.scripts.merge_to_imagefolder
=======================================

Functions
---------

`main(cfg: btx.scripts.merge_to_imagefolder.Config)`
:   Converts all the imageomics beetle datasets into one merged root/class/example.png format for use with ImageFolder dataloaders.

Classes
-------

`Config(bp_root: pathlib.Path = PosixPath('data/beetlepalooza'), hi_root: pathlib.Path = PosixPath('data/hawaii'), br_root: pathlib.Path = PosixPath('data/biorepo'), dump_to: pathlib.Path = PosixPath('data/imagefolder'), n_threads: int = 16, job_size: int = 256)`
:   Config(bp_root: pathlib.Path = PosixPath('data/beetlepalooza'), hi_root: pathlib.Path = PosixPath('data/hawaii'), br_root: pathlib.Path = PosixPath('data/biorepo'), dump_to: pathlib.Path = PosixPath('data/imagefolder'), n_threads: int = 16, job_size: int = 256)

    ### Instance variables

    `bp_root: pathlib.Path`
    :   Where you dumped data when using download_beetlepalooza.py.

    `br_root: pathlib.Path`
    :   Where you dumped data when using download_biorepo.py.

    `dump_to: pathlib.Path`
    :   Where to write the new image directories.

    `hi_root: pathlib.Path`
    :   Where you dumped data when using download_hawaii.py.

    `job_size: int`
    :   Number of images to copy per job.

    `n_threads: int`
    :   Number of concurrent threads for copying files on disk.