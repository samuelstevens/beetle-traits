"""
Apply renames from a rename log CSV file.

This script reads a rename log CSV (created by test_template_match.py) and applies
the file renames. Useful for re-running renames or batch processing.

USAGE:
------
1. Set RENAME_LOG_PATH to point to your rename log CSV file
2. First run with DRY_RUN = True to preview changes
3. If satisfied, set DRY_RUN = False to actually rename files
"""

import csv
import pathlib
from datetime import datetime

# CONFIGURATION
RENAME_LOG_PATH = pathlib.Path(
    "./test_template_match_output/rename_log_20251112_233103.csv"
)
"""Path to the rename log CSV file to apply."""

DRY_RUN = False
"""If True, only show what would happen. If False, actually rename files."""

INDIVIDUAL_IMAGES_DIR = pathlib.Path("./data/beetlepalooza/individual_specimens")
"""Base directory for individual images."""

CREATE_BACKUP_LIST = True
"""Create a CSV file listing all renames for potential rollback."""


def validate_rename_log(log_path: pathlib.Path) -> bool:
    """
    Validate that the rename log file exists and has the correct format.

    Returns:
        True if valid, False otherwise
    """
    if not log_path.exists():
        print(f"ERROR: Rename log not found at: {log_path}")
        return False

    try:
        with open(log_path, "r") as f:
            reader = csv.DictReader(f)
            required_fields = {
                "group_image",
                "old_position",
                "new_position",
                "old_path",
                "new_path",
            }
            if not required_fields.issubset(reader.fieldnames):
                print("ERROR: Rename log missing required fields.")
                print(f"  Required: {required_fields}")
                print(f"  Found: {reader.fieldnames}")
                return False
    except Exception as e:
        print(f"ERROR: Failed to read rename log: {e}")
        return False

    return True


def load_rename_operations(log_path: pathlib.Path) -> list[dict]:
    """
    Load rename operations from CSV file.

    Returns:
        List of dictionaries with rename information
    """
    operations = []

    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            operations.append({
                "group_image": row["group_image"],
                "old_position": int(row["old_position"]),
                "new_position": int(row["new_position"]),
                "old_path": row["old_path"],
                "new_path": row["new_path"],
            })

    return operations


def normalize_path(path_str: str, base_dir: pathlib.Path) -> pathlib.Path:
    """
    Normalize a path from the CSV - handle both relative and absolute paths.

    If the path already contains the base directory, use it as-is.
    Otherwise, join with base_dir.
    """
    path_obj = pathlib.Path(path_str)

    # Check if this is already a full path that starts with base_dir
    # Convert to string and normalize separators for comparison
    path_str_normalized = str(path_obj).replace("\\", "/")
    base_str_normalized = str(base_dir).replace("\\", "/")

    if base_str_normalized in path_str_normalized:
        # Path already contains base dir, use as-is
        return path_obj
    else:
        # Relative path, join with base_dir
        return base_dir / path_obj


def check_conflicts(operations: list[dict], base_dir: pathlib.Path) -> list[str]:
    """
    Check for potential conflicts (target files that already exist).

    Returns:
        List of conflict messages
    """
    conflicts = []

    for op in operations:
        old_full = normalize_path(op["old_path"], base_dir)
        new_full = normalize_path(op["new_path"], base_dir)

        # Check if source file exists
        if not old_full.exists():
            conflicts.append(f"Source file missing: {op['old_path']}")

        # Check if target already exists (and is different from source)
        if new_full.exists() and old_full != new_full:
            conflicts.append(f"Target already exists: {op['new_path']}")

    return conflicts


def apply_renames(
    operations: list[dict], base_dir: pathlib.Path, dry_run: bool = True
) -> tuple[int, int]:
    """
    Apply rename operations using a two-phase approach to avoid conflicts.

    Phase 1: Rename all files to temporary names (with .tmp_rename suffix)
    Phase 2: Rename from temporary names to final names

    Args:
        operations: List of rename operations
        base_dir: Base directory for paths
        dry_run: If True, only show what would happen

    Returns:
        Tuple of (successful_count, failed_count)
    """
    successful = 0
    failed = 0

    if dry_run:
        # In dry run mode, just show what would happen
        for idx, op in enumerate(operations, 1):
            old_full = normalize_path(op["old_path"], base_dir)
            new_full = normalize_path(op["new_path"], base_dir)

            print(
                f"\n[{idx}/{len(operations)}] {op['group_image']}: position {op['old_position']} -> {op['new_position']}"
            )
            print("  [DRY RUN] Would rename:")
            print(f"    FROM: {op['old_path']}")
            print(f"    TO:   {op['new_path']}")

            if not old_full.exists():
                print("    ⚠ WARNING: Source file not found!")
                failed += 1
            else:
                successful += 1
    else:
        # Two-phase rename to avoid conflicts
        temp_mappings = []  # List of (temp_path, final_path, operation) tuples

        print("\n" + "=" * 60)
        print("PHASE 1: Renaming to temporary names")
        print("=" * 60)

        # Phase 1: Rename to temporary names
        for idx, op in enumerate(operations, 1):
            old_full = normalize_path(op["old_path"], base_dir)
            new_full = normalize_path(op["new_path"], base_dir)

            # Create temporary name (add .tmp_rename suffix)
            temp_full = old_full.with_suffix(old_full.suffix + f".tmp_rename_{idx}")

            print(f"\n[{idx}/{len(operations)}] {op['group_image']}: Temp rename phase")

            try:
                if not old_full.exists():
                    print("  ✗ FAILED: Source file not found!")
                    print(f"    {op['old_path']}")
                    failed += 1
                    continue

                # Rename to temporary name
                old_full.rename(temp_full)
                temp_mappings.append((temp_full, new_full, op))
                print(f"  ✓ Phase 1 OK: {old_full.name} -> {temp_full.name}")

            except Exception as e:
                print(f"  ✗ FAILED: {e}")
                print(f"    FROM: {op['old_path']}")
                failed += 1

        print("\n" + "=" * 60)
        print("PHASE 2: Renaming to final names")
        print("=" * 60)

        # Phase 2: Rename from temporary to final names
        for idx, (temp_full, new_full, op) in enumerate(temp_mappings, 1):
            print(
                f"\n[{idx}/{len(temp_mappings)}] {op['group_image']}: position {op['old_position']} -> {op['new_position']}"
            )

            try:
                # Make sure parent directory exists
                new_full.parent.mkdir(parents=True, exist_ok=True)

                # Rename from temp to final
                temp_full.rename(new_full)
                print("  ✓ Renamed successfully")
                print(f"    FROM: {op['old_path']}")
                print(f"    TO:   {op['new_path']}")
                successful += 1

            except Exception as e:
                print(f"  ✗ FAILED: {e}")
                print(f"    Temp was: {temp_full}")
                print(f"    Target: {op['new_path']}")
                failed += 1

                # Try to restore from temp name on failure
                try:
                    original_path = normalize_path(op["old_path"], base_dir)
                    temp_full.rename(original_path)
                    print(f"    → Restored to original: {op['old_path']}")
                except Exception as restore_error:
                    print(
                        f"    ⚠ WARNING: Could not restore temp file: {restore_error}"
                    )

    return successful, failed


def create_backup_list(
    operations: list[dict], output_dir: pathlib.Path, base_dir: pathlib.Path
) -> pathlib.Path:
    """
    Create a backup list that can be used to reverse renames if needed.

    Returns:
        Path to the backup list file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = output_dir / f"rename_backup_{timestamp}.csv"

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(backup_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "group_image",
            "applied_old_position",
            "applied_new_position",
            "applied_old_path",
            "applied_new_path",
            "rollback_from",
            "rollback_to",
        ])

        for op in operations:
            # Normalize paths to get actual file locations
            old_full = normalize_path(op["old_path"], base_dir)
            new_full = normalize_path(op["new_path"], base_dir)

            # Get relative paths for the backup file
            try:
                old_rel = old_full.relative_to(base_dir)
                new_rel = new_full.relative_to(base_dir)
            except ValueError:
                # If relative_to fails, use the path as-is
                old_rel = old_full
                new_rel = new_full

            writer.writerow([
                op["group_image"],
                op["old_position"],
                op["new_position"],
                str(old_rel),
                str(new_rel),
                str(new_rel),  # To rollback, rename FROM new_path
                str(old_rel),  # back TO old_path
            ])

    return backup_file


def main():
    print("=" * 80)
    print("APPLY RENAME LOG")
    print("=" * 80)
    print(f"Log file: {RENAME_LOG_PATH}")
    print(f"Mode: {'DRY RUN' if DRY_RUN else 'EXECUTE RENAMES'}")
    print("=" * 80)

    # Validate log file
    if not validate_rename_log(RENAME_LOG_PATH):
        return 1

    # Load operations
    print("\nLoading rename operations...")
    operations = load_rename_operations(RENAME_LOG_PATH)
    print(f"Loaded {len(operations)} rename operations")

    # Check for conflicts
    print("\nChecking for conflicts...")
    conflicts = check_conflicts(operations, INDIVIDUAL_IMAGES_DIR)

    if conflicts:
        print(f"\n⚠ WARNING: Found {len(conflicts)} potential issues:")
        for conflict in conflicts[:10]:  # Show first 10
            print(f"  - {conflict}")
        if len(conflicts) > 10:
            print(f"  ... and {len(conflicts) - 10} more")

        if not DRY_RUN:
            response = input("\nContinue anyway? (yes/no): ")
            if response.lower() not in ["yes", "y"]:
                print("Aborted.")
                return 1
    else:
        print("No conflicts found.")

    # Create backup list if requested
    if CREATE_BACKUP_LIST and not DRY_RUN:
        backup_file = create_backup_list(
            operations, RENAME_LOG_PATH.parent, INDIVIDUAL_IMAGES_DIR
        )
        print(f"\nBackup list created: {backup_file}")
        print("  (This can be used to reverse renames if needed)")

    # Apply renames
    print("\n" + "=" * 80)
    print("APPLYING RENAMES")
    print("=" * 80)

    successful, failed = apply_renames(
        operations, INDIVIDUAL_IMAGES_DIR, dry_run=DRY_RUN
    )

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total operations: {len(operations)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if DRY_RUN:
        print("\nThis was a DRY RUN - no files were changed.")
        print("Set DRY_RUN = False to actually rename files.")
    else:
        print("\nRenames completed!")
        if CREATE_BACKUP_LIST:
            print("Backup list saved for potential rollback.")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        exit(130)
