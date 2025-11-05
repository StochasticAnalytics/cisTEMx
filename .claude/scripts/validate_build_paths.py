#!/usr/bin/env python3
"""
Validate that all build directories from tasks.json are in .gitignore.

Prevents git pollution from typos or misparsed build directory paths.
Run this script manually or as a pre-commit hook to catch potential issues.

Usage:
    python .claude/scripts/validate_build_paths.py
    python .claude/scripts/validate_build_paths.py --fix  # Add missing patterns to .gitignore
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple

# Import shared utilities
sys.path.insert(0, str(Path(__file__).parent / "common"))
from path_validation import find_git_root, validate_build_dir_in_gitignore


def extract_build_directories_from_tasks(tasks_json_path: Path) -> Set[str]:
    """
    Extract all build subdirectories mentioned in tasks.json.

    Args:
        tasks_json_path: Path to .vscode/tasks.json

    Returns:
        Set of build subdirectory names (e.g., {"debug-build", "release-build"})
    """
    try:
        with open(tasks_json_path, 'r') as f:
            # Remove comments for JSON parsing
            content = f.read()
            content = re.sub(r'//.*', '', content)
            tasks = json.loads(content)
    except Exception as e:
        print(f"Error reading tasks.json: {e}", file=sys.stderr)
        return set()

    build_dirs = set()

    # Look for BUILD tasks with pattern: cd ${build_dir}/SUBDIR && make
    pattern = r'cd\s+\$\{build_dir\}/([^\s&]+)'

    for task in tasks.get('tasks', []):
        label = task.get('label', '')
        command = task.get('command', '')

        if 'BUILD' in label:
            match = re.search(pattern, command)
            if match:
                build_subdir = match.group(1)
                build_dirs.add(build_subdir)

    return build_dirs


def validate_all_build_directories(
    git_root: Path,
    build_dirs: Set[str]
) -> Tuple[List[Tuple[Path, str]], List[Path]]:
    """
    Validate all build directories against .gitignore.

    Args:
        git_root: Project root
        build_dirs: Set of build subdirectory names

    Returns:
        Tuple of (uncovered_dirs, covered_dirs)
        - uncovered_dirs: List of (path, warning_message) for directories not in .gitignore
        - covered_dirs: List of paths that are properly covered
    """
    uncovered = []
    covered = []

    for build_subdir in build_dirs:
        build_path = git_root / "build" / build_subdir
        is_covered, warning = validate_build_dir_in_gitignore(build_path, git_root)

        if is_covered:
            covered.append(build_path)
        else:
            uncovered.append((build_path, warning))

    return uncovered, covered


def suggest_gitignore_additions(uncovered_dirs: List[Tuple[Path, str]], git_root: Path) -> List[str]:
    """
    Suggest patterns to add to .gitignore.

    Args:
        uncovered_dirs: List of (path, warning) for uncovered directories
        git_root: Project root

    Returns:
        List of suggested .gitignore patterns
    """
    suggestions = []

    # Extract just the first component of build paths
    first_components = set()
    for build_path, _ in uncovered_dirs:
        try:
            rel_path = build_path.relative_to(git_root)
            first_component = str(rel_path).split('/')[0]
            first_components.add(first_component)
        except ValueError:
            continue

    # If all uncovered dirs are under 'build/', suggest 'build/'
    if first_components == {'build'}:
        suggestions.append('build/')
    else:
        # Suggest specific patterns
        for build_path, _ in uncovered_dirs:
            try:
                rel_path = build_path.relative_to(git_root)
                suggestions.append(f'{rel_path}/')
            except ValueError:
                continue

    return suggestions


def add_to_gitignore(gitignore_path: Path, patterns: List[str]) -> bool:
    """
    Add patterns to .gitignore file.

    Args:
        gitignore_path: Path to .gitignore
        patterns: List of patterns to add

    Returns:
        True if successful, False otherwise
    """
    try:
        # Read existing content
        if gitignore_path.exists():
            with open(gitignore_path, 'r') as f:
                existing = f.read()
        else:
            existing = ""

        # Add new patterns with header
        new_content = existing
        if existing and not existing.endswith('\n'):
            new_content += '\n'

        new_content += '\n# Build directories (added by validate_build_paths.py)\n'
        for pattern in patterns:
            new_content += f'{pattern}\n'

        # Write back
        with open(gitignore_path, 'w') as f:
            f.write(new_content)

        return True

    except Exception as e:
        print(f"Error updating .gitignore: {e}", file=sys.stderr)
        return False


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Validate build directories are in .gitignore',
        epilog='Run this before committing to catch git pollution issues.'
    )
    parser.add_argument(
        '--fix',
        action='store_true',
        help='Automatically add missing patterns to .gitignore'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Only output errors, not successes'
    )

    args = parser.parse_args()

    # Find git root
    git_root = find_git_root()
    if not git_root:
        print("Error: Not in a git repository", file=sys.stderr)
        return 1

    # Locate tasks.json
    tasks_json = git_root / ".vscode" / "tasks.json"
    if not tasks_json.exists():
        print(f"Error: {tasks_json} not found", file=sys.stderr)
        return 1

    # Extract build directories
    build_dirs = extract_build_directories_from_tasks(tasks_json)
    if not build_dirs:
        print("Warning: No build directories found in tasks.json", file=sys.stderr)
        return 0

    if not args.quiet:
        print(f"Found {len(build_dirs)} build directory configuration(s) in tasks.json")
        for bd in sorted(build_dirs):
            print(f"  - build/{bd}")
        print()

    # Validate all build directories
    uncovered, covered = validate_all_build_directories(git_root, build_dirs)

    # Report covered directories
    if covered and not args.quiet:
        print(f"✓ {len(covered)} build director(ies) properly in .gitignore:")
        for path in covered:
            rel_path = path.relative_to(git_root)
            print(f"  - {rel_path}")
        print()

    # Report uncovered directories
    if uncovered:
        print(f"✗ {len(uncovered)} build director(ies) NOT in .gitignore:", file=sys.stderr)
        for path, warning in uncovered:
            rel_path = path.relative_to(git_root)
            print(f"  - {rel_path}", file=sys.stderr)
        print(file=sys.stderr)
        print("This could lead to accidentally committing build artifacts!", file=sys.stderr)
        print(file=sys.stderr)

        # Suggest fixes
        suggestions = suggest_gitignore_additions(uncovered, git_root)
        print("Suggested additions to .gitignore:", file=sys.stderr)
        for pattern in suggestions:
            print(f"  {pattern}", file=sys.stderr)
        print(file=sys.stderr)

        if args.fix:
            gitignore_path = git_root / ".gitignore"
            if add_to_gitignore(gitignore_path, suggestions):
                print(f"✓ Added {len(suggestions)} pattern(s) to .gitignore")
                return 0
            else:
                print("✗ Failed to update .gitignore", file=sys.stderr)
                return 1
        else:
            print("Run with --fix to automatically add these patterns", file=sys.stderr)
            return 1

    if not args.quiet:
        print("✓ All build directories are properly covered by .gitignore")

    return 0


if __name__ == "__main__":
    sys.exit(main())
