Module btx.scripts.format_biorepo
=================================
Format and validate beetle position annotations from biorepo group images.

This script processes beetle collections by:
1. Template matching individual beetles on group images to find their spatial locations
2. Grouping beetles into rows and inferring correct positions (top-to-bottom, left-to-right)
3. Renaming individual beetle files to match inferred positions
4. Creating measurement annotations with taxon info and measurement coordinates
5. Validating all annotations ensure coordinates stay within image bounds
6. Generating validation statistics and outputting validated annotations

INPUTS:
- Group images (PNG): Original images with multiple beetles
- Individual beetles (PNG): Pre-extracted individual beetle images
- Beetle metadata (CSV): Scientific names and taxon IDs
- TORAS annotations (JSON): Manual measurements with coordinates

OUTPUTS:
- annotations.json: Validated beetle annotations with measurements and metadata
- validation_stats.json: Statistics on validation results and data quality

USAGE:
------
Local execution:
  python -m btx.scripts.format_biorepo

With Slurm (parallel processing):
  python -m btx.scripts.format_biorepo --slurm-acct=YOUR_ACCOUNT --slurm-partition=parallel

Custom paths:
  python -m btx.scripts.format_biorepo --biorepo-dir=./data/biorepo --dump-to=./data/biorepo-formatted

All parameters are configurable via command line (see Config class for available options).

Functions
---------

`beetles_on_same_row(beetle1: btx.scripts.format_biorepo.BeetleMatch, beetle2: btx.scripts.format_biorepo.BeetleMatch, tolerance: float = 30) ‑> bool`
:   Check if two beetles are on the same row based on their vertical alignment.
    
    Checks if centers, tops, OR bottoms are within tolerance. This handles cases where
    beetles are at different heights but clearly on the same row.
    
    Args:
        beetle1: First beetle
        beetle2: Second beetle
        tolerance: Vertical tolerance in pixels
    
    Returns:
        True if any of (centers, tops, or bottoms) are within tolerance

`create_measurements_annotations(cfg: btx.scripts.format_biorepo.Config, template_match_df: polars.dataframe.frame.DataFrame, beetle_metadata_df: polars.dataframe.frame.DataFrame) ‑> list[dict]`
:   

`execute_renames(rename_ops: list[btx.scripts.format_biorepo.RenameOperation]) ‑> tuple[int, int, list[btx.scripts.format_biorepo.RenameOperation]]`
:   Execute file rename operations using a two-phase approach to avoid conflicts.
    
    Args:
        rename_ops: List of rename operations to perform
    
    Returns:
        Tuple of (successful_renames, failed_renames, completed_operations)

`group_beetles_into_rows(matches: list[btx.scripts.format_biorepo.BeetleMatch], center_tolerance: float = 30) ‑> list[list[btx.scripts.format_biorepo.BeetleMatch]]`
:   Group beetles into rows based on their vertical centers.
    
    Args:
        matches: List of BeetleMatch objects
        center_tolerance: Max distance between vertical centers for beetles to be in same row
    
    Returns:
        List of rows, where each row is a list of BeetleMatch objects sorted left-to-right

`infer_positions(matches: list[btx.scripts.format_biorepo.BeetleMatch], row_tolerance: float) ‑> dict[int, btx.scripts.format_biorepo.BeetleMatch]`
:   Infer correct beetle positions based on row grouping and spatial layout.
    
    Args:
        matches: List of BeetleMatch objects
        row_tolerance: Tolerance for grouping beetles into same row (pixels)
    
    Returns:
        Dictionary mapping inferred position (1, 2, 3...) to BeetleMatch

`main(cfg: btx.scripts.format_biorepo.Config) ‑> tuple[int, polars.dataframe.frame.DataFrame]`
:   

`offset_polyline(polyline: list, offset_x: float, offset_y: float) ‑> list`
:   Offset all points in a polyline by the given x and y offsets.
    
    Input polyline from polars .to_list() is: [[[x1, y1, x2, y2, ...]]]
    Output is in paired format: [[x1+ox, y1+oy], [x2+ox, y2+oy], ...]

`process_single_image(group_image_name: str, cfg: btx.scripts.format_biorepo.Config) ‑> tuple[list[btx.scripts.format_biorepo.RenameOperation], polars.dataframe.frame.DataFrame]`
:   Process a single group image.
    
    Returns:
        Tuple of (list of RenameOperation objects, DataFrame with beetle match info)

`template_match_beetles(group_image_path: pathlib.Path, individual_paths: list[pathlib.Path]) ‑> list[btx.scripts.format_biorepo.BeetleMatch]`
:   Template match all individual beetles on the group image.
    
    Args:
        group_image_path: Path to group image
        individual_paths: List of individual beetle image paths
    
    Returns:
        List of BeetleMatch objects with coordinates and scores

`validate_annotations(cfg: btx.scripts.format_biorepo.Config, annotations: list[dict]) ‑> list[dict]`
:   Validate annotations ensuring coordinates are within individual image bounds.
    
    Args:
        cfg: Configuration object
        annotations: List of beetle annotation dictionaries
    
    Returns:
        List of valid annotations (invalid ones are removed)

`visualize_results(group_image_path: pathlib.Path, position_map: dict[int, btx.scripts.format_biorepo.BeetleMatch], output_path: pathlib.Path, cfg: btx.scripts.format_biorepo.Config)`
:   Draw bounding boxes on the group image.

`worker_fn(cfg: btx.scripts.format_biorepo.Config, group_image_names: list[str]) ‑> tuple[list[btx.scripts.format_biorepo.RenameOperation], list[polars.dataframe.frame.DataFrame]]`
:   Worker function for parallel processing of group images.
    
    Processes a batch of group images and returns rename operations and match dataframes.
    This function is designed to be run in parallel via Slurm.
    
    Args:
        cfg: Configuration object
        group_image_names: List of group image filenames to process
    
    Returns:
        Tuple of (list of RenameOperation, list of match DataFrames)

Classes
-------

`BeetleMatch(x: float, y: float, width: int, height: int, ncc: float, individual_path: pathlib.Path, filename_position: int)`
:   Result of template matching for one beetle.

    ### Instance variables

    `filename_position: int`
    :

    `height: int`
    :

    `individual_path: pathlib.Path`
    :

    `ncc: float`
    :

    `width: int`
    :

    `x: float`
    :

    `y: float`
    :

`Config(biorepo_dir: pathlib.Path = PosixPath('data/biorepo'), dump_to: pathlib.Path = PosixPath('data/biorepo-formatted'), toras_anns: pathlib.Path = PosixPath('data/biorepo/completed_annotations'), row_center_tolerance_ratio: float = 0.22583559168925021, create_template_match_images: bool = True, slurm_acct: str = '', slurm_partition: str = 'parallel', log_to: pathlib.Path = PosixPath('logs'), n_hours: float = 2.0, groups_per_job: int = 4, bbox_color: tuple[int, int, int] = (255, 0, 0), bbox_width: int = 4, text_color: tuple[int, int, int] = (255, 255, 255))`
:   Config(biorepo_dir: pathlib.Path = PosixPath('data/biorepo'), dump_to: pathlib.Path = PosixPath('data/biorepo-formatted'), toras_anns: pathlib.Path = PosixPath('data/biorepo/completed_annotations'), row_center_tolerance_ratio: float = 0.22583559168925021, create_template_match_images: bool = True, slurm_acct: str = '', slurm_partition: str = 'parallel', log_to: pathlib.Path = PosixPath('logs'), n_hours: float = 2.0, groups_per_job: int = 4, bbox_color: tuple[int, int, int] = (255, 0, 0), bbox_width: int = 4, text_color: tuple[int, int, int] = (255, 255, 255))

    ### Instance variables

    `bbox_color: tuple[int, int, int]`
    :   Red for bounding boxes.

    `bbox_width: int`
    :   Width of bounding box lines.

    `biorepo_dir: pathlib.Path`
    :   Where biorepo data is stored.

    `create_template_match_images: bool`
    :   Whether to create template matching visualization images.

    `dump_to: pathlib.Path`
    :   Where to save formatted data.

    `groups_per_job: int`
    :   Number of group images to process per job.

    `images_dir: pathlib.Path`
    :   Where group images are stored.

    `log_to: pathlib.Path`
    :   Where to save submitit/slurm logs.

    `metadata_csv: pathlib.Path`
    :   Metadata CSV file.

    `n_hours: float`
    :   Number of hours to request for each job.

    `output_dir: pathlib.Path`
    :   Where individual beetle images are stored.

    `row_center_tolerance_ratio: float`
    :   Ratio of tolerance to average beetle height for row grouping.

    `slurm_acct: str`
    :   Slurm account to use. If empty, uses DebugExecutor (local execution).

    `slurm_partition: str`
    :   Slurm partition to use.

    `text_color: tuple[int, int, int]`
    :   White for text labels.

    `toras_anns: pathlib.Path`
    :   Where the toras annotations are stored

    `viz_output_dir: pathlib.Path`
    :   Where to save visualizations.

`RenameOperation(old_path: pathlib.Path, new_path: pathlib.Path, group_image: str, old_position: int, new_position: int)`
:   Information about a file rename operation.

    ### Instance variables

    `group_image: str`
    :

    `new_path: pathlib.Path`
    :

    `new_position: int`
    :

    `old_path: pathlib.Path`
    :

    `old_position: int`
    :