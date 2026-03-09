Module btx.scripts.format_biorepo_v2
====================================
Append new TORAS annotations to an existing annotations file.

This script processes new beetle group images (not yet annotated) by:
1. Loading existing annotations from a previous run (optional)
2. Determining which group images have TORAS annotations not yet in the formatted data
3. Template matching and renaming individual beetles for the new group images only
4. Creating annotations for the new group images from TORAS data
5. Validating and appending new annotations to the existing file

INPUTS:
- New group image names (optional, via --new-group-images; auto-detected from TORAS if omitted)
- Existing annotations.json (optional, from a previous run)
- TORAS annotations (JSON)
- Beetle metadata (CSV)
- Individual beetle images in per-group subdirectories under --individuals-dpath

OUTPUTS:
- annotations.json: Combined existing + new valid annotations
- validation_stats.json: Statistics for the newly added annotations

USAGE:
------
Auto-detect new images from TORAS (compares against existing annotations):
  python -m btx.scripts.format_biorepo_v2 \
    --existing-annotations-fpath ./data/biorepo-formatted/annotations.json

Explicit image list (first run, no existing annotations):
  python -m btx.scripts.format_biorepo_v2 --new-group-images img1.png img2.png

Append new images to existing annotations:
  python -m btx.scripts.format_biorepo_v2 \
    --new-group-images img3.png img4.png \
    --existing-annotations-fpath ./data/biorepo-formatted/annotations.json

With Slurm:
  python -m btx.scripts.format_biorepo_v2 \
    --slurm-acct=YOUR_ACCOUNT --slurm-partition=parallel

Functions
---------

`clean_toras_path(toras_path: str) ‑> str`
:   Convert a TORAS image path to a clean group image filename.
    
    Strips leading slash, removes Windows duplicate-file suffixes like ' (1)',
    and converts .jpg to .png.

`create_new_measurements_annotations(cfg: btx.scripts.format_biorepo_v2.Config, template_match_df: polars.dataframe.frame.DataFrame, beetle_metadata_df: polars.dataframe.frame.DataFrame, new_group_images: set[str]) ‑> list[dict]`
:   Create measurement annotations for the new group images only.

`main(cfg: btx.scripts.format_biorepo_v2.Config) ‑> int`
:   

`process_single_image(group_image_name: str, cfg: btx.scripts.format_biorepo_v2.Config) ‑> tuple[list[btx.scripts.format_biorepo.RenameOperation], polars.dataframe.frame.DataFrame]`
:   Process a single group image: template match, infer positions, prepare renames.

`validate_and_append_annotations(cfg: btx.scripts.format_biorepo_v2.Config, new_annotations: list[dict], existing_annotations: list[dict]) ‑> list[dict]`
:   Validate new annotations and append valid ones to existing, then save.

`visualize_results(group_image_path: pathlib.Path, position_map: dict[int, btx.scripts.format_biorepo.BeetleMatch], output_path: pathlib.Path, cfg: btx.scripts.format_biorepo_v2.Config)`
:   Draw bounding boxes and position labels on the group image.

`worker_fn(cfg: btx.scripts.format_biorepo_v2.Config, group_image_names: list[str]) ‑> tuple[list[btx.scripts.format_biorepo.RenameOperation], list[polars.dataframe.frame.DataFrame]]`
:   Worker function for parallel processing of group images.

Classes
-------

`Config(group_images_dpath: pathlib.Path = PosixPath('/fs/ess/PAS2136/CarabidImaging/Output/plotted_trays'), individuals_dpath: pathlib.Path = PosixPath('/fs/ess/PAS2136/CarabidImaging/Output/cropped_images'), dump_to: pathlib.Path = PosixPath('data/biorepo-formatted'), toras_anns: pathlib.Path = PosixPath('/fs/scratch/PAS2136/cain429/coco_anns.json'), metadata_csv: pathlib.Path = PosixPath('/fs/ess/PAS2136/CarabidImaging/allIndividuals.csv'), new_group_images: list[str] = <factory>, existing_annotations_fpath: pathlib.Path = PosixPath('/fs/scratch/PAS2136/cain429/biorepo-formatted/annotations.json'), viz_output_dpath: pathlib.Path = PosixPath('/fs/scratch/PAS2136/cain429/biorepo-formatted/template_match'), row_center_tolerance_ratio: float = 0.22583559168925021, create_template_match_images: bool = True, slurm_acct: str = '', slurm_partition: str = 'nextgen', log_to: pathlib.Path = PosixPath('logs'), n_hours: float = 2.0, groups_per_job: int = 4, bbox_color: tuple[int, int, int] = (255, 0, 0), bbox_width: int = 4, text_color: tuple[int, int, int] = (255, 255, 255))`
:   Config(group_images_dpath: pathlib.Path = PosixPath('/fs/ess/PAS2136/CarabidImaging/Output/plotted_trays'), individuals_dpath: pathlib.Path = PosixPath('/fs/ess/PAS2136/CarabidImaging/Output/cropped_images'), dump_to: pathlib.Path = PosixPath('data/biorepo-formatted'), toras_anns: pathlib.Path = PosixPath('/fs/scratch/PAS2136/cain429/coco_anns.json'), metadata_csv: pathlib.Path = PosixPath('/fs/ess/PAS2136/CarabidImaging/allIndividuals.csv'), new_group_images: list[str] = <factory>, existing_annotations_fpath: pathlib.Path = PosixPath('/fs/scratch/PAS2136/cain429/biorepo-formatted/annotations.json'), viz_output_dpath: pathlib.Path = PosixPath('/fs/scratch/PAS2136/cain429/biorepo-formatted/template_match'), row_center_tolerance_ratio: float = 0.22583559168925021, create_template_match_images: bool = True, slurm_acct: str = '', slurm_partition: str = 'nextgen', log_to: pathlib.Path = PosixPath('logs'), n_hours: float = 2.0, groups_per_job: int = 4, bbox_color: tuple[int, int, int] = (255, 0, 0), bbox_width: int = 4, text_color: tuple[int, int, int] = (255, 255, 255))

    ### Instance variables

    `bbox_color: tuple[int, int, int]`
    :

    `bbox_width: int`
    :

    `create_template_match_images: bool`
    :   Whether to create template matching visualization images.

    `dump_to: pathlib.Path`
    :   Where to save formatted data.

    `existing_annotations_fpath: pathlib.Path`
    :

    `group_images_dpath: pathlib.Path`
    :   Directory containing group images (PNG).

    `groups_per_job: int`
    :   Number of group images to process per job.

    `individuals_dpath: pathlib.Path`
    :   Directory containing per-group subdirectories of individual beetle images.

    `log_to: pathlib.Path`
    :   Where to save submitit/slurm logs.

    `metadata_csv: pathlib.Path`
    :   Beetle metadata CSV file.

    `n_hours: float`
    :   Number of hours to request for each job.

    `new_group_images: list[str]`
    :   List of new group image filenames to process (e.g. img1.png img2.png).

    `row_center_tolerance_ratio: float`
    :   Ratio of tolerance to average beetle height for row grouping.

    `slurm_acct: str`
    :   Slurm account to use. If empty, uses DebugExecutor (local execution).

    `slurm_partition: str`
    :   Slurm partition to use.

    `text_color: tuple[int, int, int]`
    :

    `toras_anns: pathlib.Path`
    :   Path to TORAS annotations JSON file.

    `viz_output_dpath: pathlib.Path`
    :   Where to save template matching visualizations.