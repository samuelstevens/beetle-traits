Module btx.scripts.visualize_individual_annotations
===================================================
Visualize measurements on a specific individual beetle image.

This script loads annotations.json, finds a specific beetle by individual_id,
and draws all measurements on the individual beetle image.

Functions
---------

`main(cfg: btx.scripts.visualize_individual_annotations.Config) ‑> int`
:   

`visualize_individual(annotation: dict, cfg: btx.scripts.visualize_individual_annotations.Config, logger: logging.Logger) ‑> None`
:   Draw measurements on the individual beetle image.

Classes
-------

`Config(annotations_file: pathlib.Path = PosixPath('data/beetlepalooza-formatted/annotations.json'), output_dir: pathlib.Path = PosixPath('data/beetlepalooza-formatted/individual-visualization'), individual_id: str = 'A00000046137_9')`
:   Config(annotations_file: pathlib.Path = PosixPath('data/beetlepalooza-formatted/annotations.json'), output_dir: pathlib.Path = PosixPath('data/beetlepalooza-formatted/individual-visualization'), individual_id: str = 'A00000046137_9')

    ### Instance variables

    `annotations_file: pathlib.Path`
    :   Path to the annotations.json file.

    `individual_id: str`
    :   The individual_id to visualize.

    `output_dir: pathlib.Path`
    :   Where to save visualization images.