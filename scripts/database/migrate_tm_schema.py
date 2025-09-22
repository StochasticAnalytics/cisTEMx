#!/usr/bin/env python3
"""
Template Match Database Schema Migration Helper
================================================
Migrates scripts from cisTEM database schema v2 to v3

Usage:
  python migrate_tm_schema.py input_script.py output_script_v3.py [--dry-run]

IMPORTANT: Output filename must be different from input filename

Column Mappings:
================
Simple Renames in TEMPLATE_MATCH_LIST:
  TEMPLATE_MATCH_JOB_ID → SEARCH_ID
  INPUT_TEMPLATE_MATCH_ID → PARENT_SEARCH_ID
  (TEMPLATE_MATCH_ID remains unchanged as primary key)

Moved to TEMPLATE_MATCH_QUEUE (access via JOIN on SEARCH_ID):
  USED_SYMMETRY → SYMMETRY
  USED_PIXEL_SIZE → PIXEL_SIZE
  USED_VOLTAGE → VOLTAGE
  USED_SPHERICAL_ABERRATION → SPHERICAL_ABERRATION
  USED_AMPLITUDE_CONTRAST → AMPLITUDE_CONTRAST
  USED_DEFOCUS1 → DEFOCUS1
  USED_DEFOCUS2 → DEFOCUS2
  USED_DEFOCUS_ANGLE → DEFOCUS_ANGLE
  USED_PHASE_SHIFT → PHASE_SHIFT
  USED_LOW_RESOLUTION_LIMIT → LOW_RESOLUTION_LIMIT
  USED_HIGH_RESOLUTION_LIMIT → HIGH_RESOLUTION_LIMIT
  USED_OUT_OF_PLANE_ANGULAR_STEP → OUT_OF_PLANE_ANGULAR_STEP
  USED_IN_PLANE_ANGULAR_STEP → IN_PLANE_ANGULAR_STEP
  USED_DEFOCUS_SEARCH_RANGE → DEFOCUS_SEARCH_RANGE
  USED_DEFOCUS_STEP → DEFOCUS_STEP
  USED_PIXEL_SIZE_SEARCH_RANGE → PIXEL_SIZE_SEARCH_RANGE
  USED_PIXEL_SIZE_STEP → PIXEL_SIZE_STEP
  USED_REFINEMENT_THRESHOLD → REFINEMENT_THRESHOLD
  USED_REF_BOX_SIZE_IN_ANGSTROMS → REF_BOX_SIZE_IN_ANGSTROMS
  USED_MASK_RADIUS → MASK_RADIUS
  USED_MIN_PEAK_RADIUS → MIN_PEAK_RADIUS
  USED_XY_CHANGE_THRESHOLD → XY_CHANGE_THRESHOLD
  USED_EXCLUDE_ABOVE_XY_THRESHOLD → EXCLUDE_ABOVE_XY_THRESHOLD

Example query migration:
  OLD: SELECT USED_PIXEL_SIZE FROM TEMPLATE_MATCH_LIST WHERE ID = ?
  NEW: SELECT q.PIXEL_SIZE FROM TEMPLATE_MATCH_LIST list
       JOIN TEMPLATE_MATCH_QUEUE q ON list.SEARCH_ID = q.SEARCH_ID
       WHERE list.ID = ?
"""

import sys
import re
import os
from typing import List, Tuple, Dict

# Column mappings
SIMPLE_RENAMES = {
    'TEMPLATE_MATCH_JOB_ID': 'SEARCH_ID',
    'INPUT_TEMPLATE_MATCH_ID': 'PARENT_SEARCH_ID'
}

MOVED_TO_QUEUE = {
    'USED_SYMMETRY': 'SYMMETRY',
    'USED_PIXEL_SIZE': 'PIXEL_SIZE',
    'USED_VOLTAGE': 'VOLTAGE',
    'USED_SPHERICAL_ABERRATION': 'SPHERICAL_ABERRATION',
    'USED_AMPLITUDE_CONTRAST': 'AMPLITUDE_CONTRAST',
    'USED_DEFOCUS1': 'DEFOCUS1',
    'USED_DEFOCUS2': 'DEFOCUS2',
    'USED_DEFOCUS_ANGLE': 'DEFOCUS_ANGLE',
    'USED_PHASE_SHIFT': 'PHASE_SHIFT',
    'USED_LOW_RESOLUTION_LIMIT': 'LOW_RESOLUTION_LIMIT',
    'USED_HIGH_RESOLUTION_LIMIT': 'HIGH_RESOLUTION_LIMIT',
    'USED_OUT_OF_PLANE_ANGULAR_STEP': 'OUT_OF_PLANE_ANGULAR_STEP',
    'USED_IN_PLANE_ANGULAR_STEP': 'IN_PLANE_ANGULAR_STEP',
    'USED_DEFOCUS_SEARCH_RANGE': 'DEFOCUS_SEARCH_RANGE',
    'USED_DEFOCUS_STEP': 'DEFOCUS_STEP',
    'USED_PIXEL_SIZE_SEARCH_RANGE': 'PIXEL_SIZE_SEARCH_RANGE',
    'USED_PIXEL_SIZE_STEP': 'PIXEL_SIZE_STEP',
    'USED_REFINEMENT_THRESHOLD': 'REFINEMENT_THRESHOLD',
    'USED_REF_BOX_SIZE_IN_ANGSTROMS': 'REF_BOX_SIZE_IN_ANGSTROMS',
    'USED_MASK_RADIUS': 'MASK_RADIUS',
    'USED_MIN_PEAK_RADIUS': 'MIN_PEAK_RADIUS',
    'USED_XY_CHANGE_THRESHOLD': 'XY_CHANGE_THRESHOLD',
    'USED_EXCLUDE_ABOVE_XY_THRESHOLD': 'EXCLUDE_ABOVE_XY_THRESHOLD'
}


class SQLMigrator:
    def __init__(self):
        self.changes_made = []
        self.manual_review_needed = []

    def migrate_line(self, line: str, line_num: int) -> str:
        """Migrate a single line of code."""
        original_line = line
        modified = False

        # Check if this line contains SQL-like patterns
        if not self._contains_sql_pattern(line):
            return line

        # Apply simple renames
        for old_col, new_col in SIMPLE_RENAMES.items():
            pattern = r'\b' + old_col + r'\b'
            if re.search(pattern, line, re.IGNORECASE):
                line = re.sub(pattern, new_col, line, flags=re.IGNORECASE)
                self.changes_made.append(f"Line {line_num}: {old_col} → {new_col}")
                modified = True

        # Check for moved columns
        for old_col, new_col in MOVED_TO_QUEUE.items():
            pattern = r'\b' + old_col + r'\b'
            if re.search(pattern, line, re.IGNORECASE):
                self.manual_review_needed.append(
                    f"Line {line_num}: {old_col} moved to TEMPLATE_MATCH_QUEUE table. "
                    f"Requires JOIN on SEARCH_ID to access as q.{new_col}"
                )
                # Add migration comment
                if modified or original_line != line:
                    return f"# MIGRATION: {old_col} moved to TEMPLATE_MATCH_QUEUE.{new_col} - requires JOIN\n{line}"

        return line

    def _contains_sql_pattern(self, line: str) -> bool:
        """Check if line likely contains SQL."""
        sql_keywords = [
            'SELECT', 'FROM', 'WHERE', 'INSERT', 'UPDATE', 'DELETE',
            'TEMPLATE_MATCH_LIST', 'TEMPLATE_MATCH_QUEUE',
            'select', 'from', 'where', 'insert', 'update', 'delete'
        ]
        return any(keyword in line for keyword in sql_keywords)

    def migrate_file(self, input_path: str, output_path: str, dry_run: bool = False) -> None:
        """Migrate an entire file."""
        if not os.path.exists(input_path):
            print(f"Error: Input file '{input_path}' not found")
            sys.exit(1)

        if os.path.abspath(input_path) == os.path.abspath(output_path):
            print("Error: Output filename must be different from input filename")
            sys.exit(1)

        print(f"Migrating {input_path} → {output_path}")
        if dry_run:
            print("DRY RUN - no files will be written")

        self.changes_made = []
        self.manual_review_needed = []

        with open(input_path, 'r') as f:
            lines = f.readlines()

        migrated_lines = []
        for i, line in enumerate(lines, 1):
            migrated_lines.append(self.migrate_line(line, i))

        if not dry_run:
            with open(output_path, 'w') as f:
                # Add header comment
                f.write("# Migrated from cisTEM database schema v2 to v3\n")
                f.write(f"# Original file: {input_path}\n")
                f.write(f"# Migration date: {__import__('datetime').datetime.now()}\n\n")
                f.writelines(migrated_lines)

        # Print summary
        print("\n" + "="*60)
        print("MIGRATION SUMMARY")
        print("="*60)

        if self.changes_made:
            print(f"\n✓ Automatic changes made: {len(self.changes_made)}")
            for change in self.changes_made[:10]:  # Show first 10
                print(f"  • {change}")
            if len(self.changes_made) > 10:
                print(f"  ... and {len(self.changes_made) - 10} more")
        else:
            print("\n✓ No automatic changes needed")

        if self.manual_review_needed:
            print(f"\n⚠ Manual review needed: {len(self.manual_review_needed)} items")
            print("\nThe following require manual updates to add JOIN clauses:")
            for item in self.manual_review_needed[:10]:  # Show first 10
                print(f"  • {item}")
            if len(self.manual_review_needed) > 10:
                print(f"  ... and {len(self.manual_review_needed) - 10} more")

            print("\nExample JOIN pattern for accessing moved columns:")
            print("  SELECT q.PIXEL_SIZE")
            print("  FROM TEMPLATE_MATCH_LIST list")
            print("  JOIN TEMPLATE_MATCH_QUEUE q ON list.SEARCH_ID = q.SEARCH_ID")
            print("  WHERE list.ID = ?")
        else:
            print("\n✓ No manual review needed")

        if not dry_run:
            print(f"\n✓ Output written to: {output_path}")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    dry_run = '--dry-run' in sys.argv

    migrator = SQLMigrator()
    migrator.migrate_file(input_file, output_file, dry_run)


if __name__ == '__main__':
    main()