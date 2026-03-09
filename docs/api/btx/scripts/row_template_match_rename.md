Module btx.scripts.row_template_match_rename
============================================
Test template matching and position inference for beetle images.

This script:
1. Template matches individual beetles on group images
2. Infers correct beetle positions based on spatial layout and scale bar
3. Identifies files with incorrect position labels
4. Renames files to correct position numbers (with dry-run option)
5. Creates visualizations and rename logs

USAGE:
------
1. First run in DRY_RUN mode (DRY_RUN = True) to see what would be renamed
2. Review the visualizations in ./template_match_output/
3. Check the rename log CSV
4. If satisfied, set DRY_RUN = False to actually rename files

CONFIGURATION:
--------------
- PROCESS_ALL_IMAGES: True = process all images in metadata, False = use TEST_GROUP_IMAGES list
- DRY_RUN: True = show what would happen, False = actually rename files
- CREATE_RENAME_LOG: True = save CSV log of all renames
- ROW_CENTER_TOLERANCE_RATIO: Ratio of row tolerance to average beetle height (0.226 = 50px @ 221.4px avg height)
- SCALEBAR_ROW_TOLERANCE_RATIO: Ratio for scale bar row detection (dynamic, scales with beetle size)

Visualization color key:
- Cyan: Scale bar location and horizontal reference line
- Red boxes: Beetle bounding boxes
- Magenta lines: Beetle center Y (thick)
- Light blue lines: Beetle top Y (thin)
- Orange lines: Beetle bottom Y (thin)
- Green lines: ElytraLength measurements
- Yellow lines: ElytraWidth measurements
- White labels: Inferred beetle positions (#1, #2, etc.)

Row detection: Beetles are grouped into rows if their centers, tops, OR bottoms
are within a dynamic tolerance that scales with beetle size. The tolerance is calculated
as (average_beetle_height * ROW_CENTER_TOLERANCE_RATIO), ensuring consistent grouping
across images with different-sized beetles.

Functions
---------

`beetles_on_same_row(beetle1: btx.scripts.row_template_match_rename.BeetleMatch, beetle2: btx.scripts.row_template_match_rename.BeetleMatch, tolerance: float = 30) ‑> bool`
:   Check if two beetles are on the same row based on their vertical alignment.
    
    Checks if centers, tops, OR bottoms are within tolerance. This handles cases where
    beetles are at different heights but clearly on the same row.
    
    Args:
        beetle1: First beetle
        beetle2: Second beetle
        tolerance: Vertical tolerance in pixels
    
    Returns:
        True if any of (centers, tops, or bottoms) are within tolerance

`create_new_filename(old_path: str, new_position: int) ‑> str`
:   Create new filename with corrected position number.
    
    Args:
        old_path: Original path like "part_000/A00000001831_specimen_5.png"
        new_position: New position number
    
    Returns:
        New path like "part_000/A00000001831_specimen_3.png"

`execute_renames(rename_ops: list[btx.scripts.row_template_match_rename.RenameOperation], dry_run: bool = True) ‑> tuple[int, int, list[btx.scripts.row_template_match_rename.RenameOperation]]`
:   Execute file rename operations using a two-phase approach to avoid conflicts.
    
    Phase 1: Rename all files to temporary names (with .tmp_rename suffix)
    Phase 2: Rename from temporary names to final names
    
    Args:
        rename_ops: List of rename operations to perform
        dry_run: If True, only print what would happen
    
    Returns:
        Tuple of (successful_renames, failed_renames, successfully_renamed_operations)

`get_measurements(measurements_df: polars.dataframe.frame.DataFrame, picture_id: str, inferred_position: int) ‑> dict[str, dict]`
:   Get measurement data for a specific beetle position.
    
    Returns:
        Dict with 'ElytraLength' and 'ElytraWidth' keys containing coordinate dicts

`group_beetles_into_rows(matches: list[btx.scripts.row_template_match_rename.BeetleMatch], center_tolerance: float = 30) ‑> list[list[btx.scripts.row_template_match_rename.BeetleMatch]]`
:   Group beetles into rows based on their vertical centers.
    
    Args:
        matches: List of BeetleMatch objects
        center_tolerance: Max distance between vertical centers for beetles to be in same row
    
    Returns:
        List of rows, where each row is a list of BeetleMatch objects sorted left-to-right

`infer_positions(matches: list[btx.scripts.row_template_match_rename.BeetleMatch], scale_bar: btx.scripts.row_template_match_rename.ScaleBar, row_tolerance: float, scalebar_tolerance: float) ‑> dict[int, btx.scripts.row_template_match_rename.BeetleMatch]`
:   Infer correct beetle positions based on spatial layout using row detection.
    
    Args:
        matches: List of BeetleMatch objects
        scale_bar: ScaleBar object with position information
        row_tolerance: Tolerance for grouping beetles into same row (pixels)
        scalebar_tolerance: Tolerance for detecting scale bar row (pixels)
    
    Logic:
    1. Group beetles into rows based on vertical overlap of bounding boxes
    2. If a row contains the scale bar: that row comes first, ordered left-to-right
    3. Other rows ordered top-to-bottom, within each row ordered left-to-right
    
    Returns:
        Dictionary mapping inferred position (1, 2, 3...) to BeetleMatch

`is_row_near_scalebar(row: list[btx.scripts.row_template_match_rename.BeetleMatch], scalebar_y: float, tolerance: float = 50) ‑> bool`
:   Check if a row of beetles is on the same row as the scale bar.
    
    Args:
        row: List of BeetleMatch objects in the same row
        scalebar_y: Y coordinate of scale bar center
        tolerance: Vertical tolerance in pixels

`load_scale_bar(measurements_df: polars.dataframe.frame.DataFrame, picture_id: str) ‑> btx.scripts.row_template_match_rename.ScaleBar`
:   Extract scale bar coordinates for a given picture.

`main(cfg: btx.scripts.row_template_match_rename.Config) ‑> int`
:   

`parse_coords(coords_str)`
:   Parse coordinate string from CSV.

`process_single_image(test_group_image: str, measurements_df: polars.dataframe.frame.DataFrame, metadata_df: polars.dataframe.frame.DataFrame, cfg: btx.scripts.row_template_match_rename.Config) ‑> list[btx.scripts.row_template_match_rename.RenameOperation]`
:   Process a single group image.
    
    Returns:
        List of RenameOperation objects for files that need to be renamed

`template_match_beetles(group_image_path: pathlib.Path, individual_paths: list[tuple[str, int]], cfg: btx.scripts.row_template_match_rename.Config) ‑> list[btx.scripts.row_template_match_rename.BeetleMatch]`
:   Template match all individual beetles on the group image.
    
    Args:
        group_image_path: Path to group image
        individual_paths: List of (individual_image_path, filename_position) tuples
        cfg: Configuration object
    
    Returns:
        List of BeetleMatch objects with coordinates and scores

`update_metadata_csv(rename_ops: list[btx.scripts.row_template_match_rename.RenameOperation], cfg: btx.scripts.row_template_match_rename.Config) ‑> tuple[int, int]`
:   Update metadata.csv to reflect renamed files.
    
    Args:
        rename_ops: List of successfully completed rename operations
        cfg: Configuration object
    
    Returns:
        Tuple of (successful_updates, failed_updates)

`visualize_results(group_image_path: pathlib.Path, position_map: dict[int, btx.scripts.row_template_match_rename.BeetleMatch], measurements_df: polars.dataframe.frame.DataFrame, picture_id: str, scale_bar: btx.scripts.row_template_match_rename.ScaleBar, output_path: pathlib.Path, cfg: btx.scripts.row_template_match_rename.Config)`
:   Draw bounding boxes and measurements on the group image.

Classes
-------

`BeetleMatch(x: float, y: float, width: int, height: int, ncc: float, individual_path: str, filename_position: int)`
:   Result of template matching for one beetle.

    ### Ancestors (in MRO)

    * builtins.tuple

    ### Instance variables

    `filename_position: int`
    :   Alias for field number 6

    `height: int`
    :   Alias for field number 3

    `individual_path: str`
    :   Alias for field number 5

    `ncc: float`
    :   Alias for field number 4

    `width: int`
    :   Alias for field number 2

    `x: float`
    :   Alias for field number 0

    `y: float`
    :   Alias for field number 1

`Config(hf_root: pathlib.Path = PosixPath('data/beetlepalooza/individual_specimens'), resized_root: pathlib.Path = PosixPath('data/beetlepalooza/group_images_resized'), output_dir: pathlib.Path = PosixPath('data/beetlepalooza/template_match_output'), process_all_images: bool = True, test_group_images: list[str] = <factory>, dry_run: bool = True, create_rename_log: bool = True, row_center_tolerance_ratio: float = 0.22583559168925021, scalebar_row_tolerance_ratio: float = 0.22583559168925021, bbox_color: tuple[int, int, int] = (255, 0, 0), bbox_width: int = 4, elytra_length_color: tuple[int, int, int] = (0, 255, 0), elytra_width_color: tuple[int, int, int] = (255, 255, 0), measurement_width: int = 3, text_color: tuple[int, int, int] = (255, 255, 255))`
:   Config(hf_root: pathlib.Path = PosixPath('data/beetlepalooza/individual_specimens'), resized_root: pathlib.Path = PosixPath('data/beetlepalooza/group_images_resized'), output_dir: pathlib.Path = PosixPath('data/beetlepalooza/template_match_output'), process_all_images: bool = True, test_group_images: list[str] = <factory>, dry_run: bool = True, create_rename_log: bool = True, row_center_tolerance_ratio: float = 0.22583559168925021, scalebar_row_tolerance_ratio: float = 0.22583559168925021, bbox_color: tuple[int, int, int] = (255, 0, 0), bbox_width: int = 4, elytra_length_color: tuple[int, int, int] = (0, 255, 0), elytra_width_color: tuple[int, int, int] = (255, 255, 0), measurement_width: int = 3, text_color: tuple[int, int, int] = (255, 255, 255))

    ### Instance variables

    `bbox_color: tuple[int, int, int]`
    :   Red for bounding boxes.

    `bbox_width: int`
    :   Width of bounding box lines.

    `create_rename_log: bool`
    :   Create a log file of all renames.

    `dry_run: bool`
    :   Set to False to actually rename files (True = just show what would happen).

    `elytra_length_color: tuple[int, int, int]`
    :   Green for elytra length measurements.

    `elytra_width_color: tuple[int, int, int]`
    :   Yellow for elytra width measurements.

    `hf_root: pathlib.Path`
    :   Where individual specimen images are stored (individual_specimens directory).

    `measurement_width: int`
    :   Width of measurement lines.

    `output_dir: pathlib.Path`
    :   Where to save visualizations and rename logs.

    `process_all_images: bool`
    :   Set to True to process all images in metadata, False to use test_group_images list.

    `resized_root: pathlib.Path`
    :   Where resized group images are stored.

    `row_center_tolerance_ratio: float`
    :   Ratio of tolerance to average beetle height.

    `scalebar_row_tolerance_ratio: float`
    :   Ratio for scale bar row detection.

    `test_group_images: list[str]`
    :   Only used if process_all_images = False.

    `text_color: tuple[int, int, int]`
    :   White for text labels.

`RenameOperation(old_path: pathlib.Path, new_path: pathlib.Path, old_relative: str, new_relative: str, group_image: str, old_position: int, new_position: int)`
:   Information about a file rename operation.

    ### Ancestors (in MRO)

    * builtins.tuple

    ### Instance variables

    `group_image: str`
    :   Alias for field number 4

    `new_path: pathlib.Path`
    :   Alias for field number 1

    `new_position: int`
    :   Alias for field number 6

    `new_relative: str`
    :   Alias for field number 3

    `old_path: pathlib.Path`
    :   Alias for field number 0

    `old_position: int`
    :   Alias for field number 5

    `old_relative: str`
    :   Alias for field number 2

`ScaleBar(x1: float, y1: float, x2: float, y2: float)`
:   Scale bar position.

    ### Ancestors (in MRO)

    * builtins.tuple

    ### Instance variables

    `center_y: float`
    :   Y coordinate of scale bar center.

    `x1: float`
    :   Alias for field number 0

    `x2: float`
    :   Alias for field number 2

    `y1: float`
    :   Alias for field number 1

    `y2: float`
    :   Alias for field number 3