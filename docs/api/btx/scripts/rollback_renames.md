Module btx.scripts.rollback_renames
===================================
Rollback renames using a backup list CSV file.

This script reverses renames that were applied using apply_rename_log.py.

USAGE:
------
1. Set BACKUP_LIST_PATH to point to your backup CSV file (rename_backup_*.csv)
2. First run with DRY_RUN = True to preview rollback
3. If satisfied, set DRY_RUN = False to actually rollback

Variables
---------

`BACKUP_LIST_PATH`
:   Path to the backup list CSV file.

`DRY_RUN`
:   If True, only show what would happen. If False, actually rename files.

`INDIVIDUAL_IMAGES_DIR`
:   Base directory for individual images.

Functions
---------

`apply_rollback(operations: list[dict], base_dir: pathlib.Path, dry_run: bool = True) ‑> tuple[int, int]`
:   Apply rollback operations.
    
    Returns:
        Tuple of (successful_count, failed_count)

`load_rollback_operations(backup_path: pathlib.Path) ‑> list[dict]`
:   Load rollback operations from backup CSV file.
    
    Returns:
        List of dictionaries with rollback information

`main()`
:   

`normalize_path(path_str: str, base_dir: pathlib.Path) ‑> pathlib.Path`
:   Normalize a path from the CSV - handle both relative and absolute paths.
    
    If the path already contains the base directory, use it as-is.
    Otherwise, join with base_dir.