#!/usr/bin/env python3
"""
Sync Validation Tool for cisTEMx

Validates that hardcoded values across the codebase stay synchronized with their
source of truth. Uses a JSON rules database that describes WHERE to find values
and HOW to extract them, but does NOT store the values themselves.

The validator:
1. Reads the source of truth file and extracts the current value
2. For each target location, extracts the value found there
3. Compares: does target match source?
4. Reports any mismatches and exits with error code

Usage:
    python validate_sync.py [--rules-file PATH]

Exit codes:
    0 - All values in sync
    1 - One or more values out of sync
    2 - Configuration or file read error
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class SyncValidator:
    """Validates synchronization of hardcoded values across the codebase."""

    def __init__(self, rules_file: Path, project_root: Path):
        self.rules_file = rules_file
        self.project_root = project_root
        self.rules = []
        self.errors = []

    def load_rules(self) -> bool:
        """Load validation rules from JSON file."""
        try:
            with open(self.rules_file, 'r') as f:
                data = json.load(f)
                self.rules = data.get('rules', [])
                return True
        except FileNotFoundError:
            print(f"ERROR: Rules file not found: {self.rules_file}", file=sys.stderr)
            return False
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON in rules file: {e}", file=sys.stderr)
            return False

    def extract_value(self, file_path: Path, pattern: str, capture_group: int) -> Optional[Tuple[str, int]]:
        """
        Extract a value from a file using regex pattern.

        Returns:
            Tuple of (extracted_value, line_number) or None if not found
        """
        try:
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, start=1):
                    match = re.search(pattern, line)
                    if match:
                        try:
                            value = match.group(capture_group)
                            return (value.strip(), line_num)
                        except IndexError:
                            print(f"WARNING: Capture group {capture_group} not found in pattern", file=sys.stderr)
                            continue
            return None
        except FileNotFoundError:
            print(f"ERROR: File not found: {file_path}", file=sys.stderr)
            return None
        except Exception as e:
            print(f"ERROR reading {file_path}: {e}", file=sys.stderr)
            return None

    def validate_rule(self, rule: Dict) -> bool:
        """
        Validate a single synchronization rule.

        Returns:
            True if all targets match source, False otherwise
        """
        rule_name = rule.get('name', 'unknown')
        description = rule.get('description', '')

        print(f"\nValidating: {rule_name}")
        if description:
            print(f"  {description}")

        # Extract source value
        source_config = rule.get('source', {})
        source_file = self.project_root / source_config.get('file', '')
        source_pattern = source_config.get('pattern', '')
        source_capture_group = source_config.get('capture_group', 1)

        source_result = self.extract_value(source_file, source_pattern, source_capture_group)
        if source_result is None:
            error_msg = f"  ✗ Failed to extract source value from {source_file}"
            print(error_msg)
            self.errors.append(error_msg)
            return False

        source_value, source_line = source_result
        print(f"  Source: {source_file}:{source_line} = '{source_value}'")

        # Check each target
        all_match = True
        targets = rule.get('targets', [])

        for target in targets:
            target_file = self.project_root / target.get('file', '')
            target_pattern = target.get('pattern', '')
            target_capture_group = target.get('capture_group', 1)
            context = target.get('context', 'unknown')

            target_result = self.extract_value(target_file, target_pattern, target_capture_group)

            if target_result is None:
                error_msg = f"    ✗ {target_file}: pattern not found"
                print(error_msg)
                self.errors.append(error_msg)
                all_match = False
                continue

            target_value, target_line = target_result

            if target_value == source_value:
                print(f"    ✓ {target_file}:{target_line} = '{target_value}'")
            else:
                error_msg = f"    ✗ {target_file}:{target_line} = '{target_value}' (expected '{source_value}')"
                print(error_msg)
                print(f"       Context: {context}")
                self.errors.append(error_msg)
                all_match = False

        return all_match

    def validate_all(self) -> bool:
        """
        Validate all rules.

        Returns:
            True if all rules pass, False otherwise
        """
        if not self.rules:
            print("WARNING: No rules to validate", file=sys.stderr)
            return True

        print("=" * 70)
        print("Sync Validation")
        print("=" * 70)

        all_passed = True
        for rule in self.rules:
            if not self.validate_rule(rule):
                all_passed = False

        print("\n" + "=" * 70)
        if all_passed:
            print("✓ All synchronization checks passed")
            print("=" * 70)
            return True
        else:
            print("✗ Synchronization validation FAILED")
            print("=" * 70)
            print("\nErrors found:")
            for error in self.errors:
                print(error)
            print("\nPlease update the out-of-sync files to match their source of truth.")
            return False


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate synchronization of hardcoded values")
    parser.add_argument(
        '--rules-file',
        type=Path,
        help='Path to rules JSON file (default: .github/scripts/sync_validation_rules.json)'
    )
    parser.add_argument(
        '--project-root',
        type=Path,
        help='Project root directory (default: git root or current directory)'
    )

    args = parser.parse_args()

    # Determine project root
    if args.project_root:
        project_root = args.project_root.resolve()
    else:
        # Try to find git root
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', '--show-toplevel'],
                capture_output=True,
                text=True,
                check=True
            )
            project_root = Path(result.stdout.strip())
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fall back to current directory
            project_root = Path.cwd()

    # Determine rules file
    if args.rules_file:
        rules_file = args.rules_file
    else:
        rules_file = project_root / '.github' / 'scripts' / 'sync_validation_rules.json'

    # Validate
    validator = SyncValidator(rules_file, project_root)

    if not validator.load_rules():
        sys.exit(2)

    if validator.validate_all():
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
