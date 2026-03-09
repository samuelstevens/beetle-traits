Module btx.scripts.apply_rename_log
===================================
Apply renames from a rename log CSV file.

This script reads a rename log CSV (created by test_template_match.py) and applies
the file renames. Useful for re-running renames or batch processing.

USAGE:
------
1. Set RENAME_LOG_PATH to point to your rename log CSV file
2. First run with DRY_RUN = True to preview changes
3. If satisfied, set DRY_RUN = False to actually rename files

Variables
---------

`CREATE_BACKUP_LIST`
:   Create a CSV file listing all renames for potential rollback.

`DRY_RUN`
:   If True, only show what would happen. If False, actually rename files.

`INDIVIDUAL_IMAGES_DIR`
:   Base directory for individual images.

`RENAME_LOG_PATH`
:   Path to the rename log CSV file to apply.

Functions
---------

`apply_renames(operations: list[dict], base_dir: pathlib.Path, dry_run: bool = True) ‑> tuple[int, int]`
:   Apply rename operations using a two-phase approach to avoid conflicts.
    
    Phase 1: Rename all files to temporary names (with .tmp_rename suffix)
    Phase 2: Rename from temporary names to final names
    
    Args:
        operations: List of rename operations
        base_dir: Base directory for paths
        dry_run: If True, only show what would happen
    
    Returns:
        Tuple of (successful_count, failed_count)

`check_conflicts(operations: list[dict], base_dir: pathlib.Path) ‑> list[str]`
:   Check for potential conflicts (target files that already exist).
    
    Returns:
        List of conflict messages

`create_backup_list(operations: list[dict], output_dir: pathlib.Path, base_dir: pathlib.Path) ‑> pathlib.Path`
:   Create a backup list that can be used to reverse renames if needed.
    
    Returns:
        Path to the backup list file

`load_rename_operations(log_path: pathlib.Path) ‑> list[dict]`
:   Load rename operations from CSV file.
    
    Returns:
        List of dictionaries with rename information

`main()`
:   

`normalize_path(path_str: str, base_dir: pathlib.Path) ‑> pathlib.Path`
:   Normalize a path from the CSV - handle both relative and absolute paths.
    
    If the path already contains the base directory, use it as-is.
    Otherwise, join with base_dir.

`validate_rename_log(log_path: pathlib.Path) ‑> bool`
:   Validate that the rename log file exists and has the correct format.
    
    Returns:
        True if valid, False otherwise