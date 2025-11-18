"""
Rollback renames using a backup list CSV file.

This script reverses renames that were applied using apply_rename_log.py.

USAGE:
------
1. Set BACKUP_LIST_PATH to point to your backup CSV file (rename_backup_*.csv)
2. First run with DRY_RUN = True to preview rollback
3. If satisfied, set DRY_RUN = False to actually rollback
"""

import csv
import pathlib


# CONFIGURATION
BACKUP_LIST_PATH = pathlib.Path("./test_template_match_output/rename_backup_20250113_123456.csv")
"""Path to the backup list CSV file."""

DRY_RUN = True
"""If True, only show what would happen. If False, actually rename files."""

INDIVIDUAL_IMAGES_DIR = pathlib.Path("./data/beetlepalooza/individual_specimens")
"""Base directory for individual images."""


def normalize_path(path_str: str, base_dir: pathlib.Path) -> pathlib.Path:
    """
    Normalize a path from the CSV - handle both relative and absolute paths.

    If the path already contains the base directory, use it as-is.
    Otherwise, join with base_dir.
    """
    path_obj = pathlib.Path(path_str)

    # Check if this is already a full path that starts with base_dir
    # Convert to string and normalize separators for comparison
    path_str_normalized = str(path_obj).replace('\\', '/')
    base_str_normalized = str(base_dir).replace('\\', '/')

    if base_str_normalized in path_str_normalized:
        # Path already contains base dir, use as-is
        return path_obj
    else:
        # Relative path, join with base_dir
        return base_dir / path_obj


def load_rollback_operations(backup_path: pathlib.Path) -> list[dict]:
    """
    Load rollback operations from backup CSV file.

    Returns:
        List of dictionaries with rollback information
    """
    if not backup_path.exists():
        raise FileNotFoundError(f"Backup file not found: {backup_path}")

    operations = []

    with open(backup_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            operations.append({
                'group_image': row['group_image'],
                'from_path': row['rollback_from'],
                'to_path': row['rollback_to'],
                'position_before': row['applied_old_position'],
                'position_after': row['applied_new_position'],
            })

    return operations


def apply_rollback(operations: list[dict], base_dir: pathlib.Path, dry_run: bool = True) -> tuple[int, int]:
    """
    Apply rollback operations.

    Returns:
        Tuple of (successful_count, failed_count)
    """
    successful = 0
    failed = 0

    for idx, op in enumerate(operations, 1):
        from_full = normalize_path(op['from_path'], base_dir)
        to_full = normalize_path(op['to_path'], base_dir)

        print(f"\n[{idx}/{len(operations)}] {op['group_image']}: position {op['position_after']} -> {op['position_before']}")

        if dry_run:
            print(f"  [DRY RUN] Would rollback:")
            print(f"    FROM: {op['from_path']}")
            print(f"    TO:   {op['to_path']}")

            if not from_full.exists():
                print(f"    ⚠ WARNING: Source file not found!")
                failed += 1
            elif to_full.exists() and from_full != to_full:
                print(f"    ⚠ WARNING: Target already exists!")
                failed += 1
            else:
                successful += 1
        else:
            try:
                # Make sure parent directory exists
                to_full.parent.mkdir(parents=True, exist_ok=True)

                # Rename the file back
                from_full.rename(to_full)
                print(f"  ✓ Rolled back successfully")
                successful += 1
            except Exception as e:
                print(f"  ✗ FAILED: {e}")
                failed += 1

    return successful, failed


def main():
    print("="*80)
    print("ROLLBACK RENAMES")
    print("="*80)
    print(f"Backup file: {BACKUP_LIST_PATH}")
    print(f"Mode: {'DRY RUN' if DRY_RUN else 'EXECUTE ROLLBACK'}")
    print("="*80)

    # Load operations
    try:
        print("\nLoading rollback operations...")
        operations = load_rollback_operations(BACKUP_LIST_PATH)
        print(f"Loaded {len(operations)} rollback operations")
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    # Confirm if not dry run
    if not DRY_RUN:
        print("\n⚠ WARNING: This will reverse previously applied renames!")
        response = input("Are you sure you want to continue? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Aborted.")
            return 0

    # Apply rollback
    print("\n" + "="*80)
    print("ROLLING BACK")
    print("="*80)

    successful, failed = apply_rollback(operations, INDIVIDUAL_IMAGES_DIR, dry_run=DRY_RUN)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total operations: {len(operations)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if DRY_RUN:
        print("\nThis was a DRY RUN - no files were changed.")
        print("Set DRY_RUN = False to actually rollback renames.")
    else:
        print("\nRollback completed!")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        exit(130)
