#!/usr/bin/env python3
"""
Search quick notes for keywords or patterns.

Useful for finding past observations, skill usage patterns,
or specific technical details across your lab notebook.
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


def find_git_root() -> Optional[Path]:
    """Find the git project root."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True
        )
        return Path(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def parse_note_timestamp(filename: str) -> Optional[datetime]:
    """Extract timestamp from lab note filename."""
    if not filename.startswith('lab_note_') or not filename.endswith('.md'):
        return None

    try:
        timestamp_str = filename[9:-3]
        return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def search_note(
    filepath: Path,
    pattern: str,
    case_sensitive: bool = False,
    section: Optional[str] = None
) -> List[Tuple[int, str]]:
    """
    Search a single note for pattern matches.

    Args:
        filepath: Path to note file
        pattern: Search pattern (regex supported)
        case_sensitive: Whether search is case-sensitive
        section: Limit search to specific section (Task, Intent, Observation, etc.)

    Returns:
        List of (line_number, line_content) tuples
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        flags = 0 if case_sensitive else re.IGNORECASE
        regex = re.compile(pattern, flags)

        matches = []
        current_section = None
        in_target_section = (section is None)  # If no section filter, search all

        for i, line in enumerate(lines, 1):
            # Track current section
            if line.startswith('## '):
                current_section = line[3:].strip()
                in_target_section = (section is None) or (current_section == section)
                continue

            # Search within target section
            if in_target_section and regex.search(line):
                matches.append((i, line.rstrip()))

        return matches

    except Exception as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
        return []


def search_all_notes(
    cache_dir: Path,
    pattern: str,
    case_sensitive: bool = False,
    section: Optional[str] = None
) -> List[Tuple[Path, datetime, List[Tuple[int, str]]]]:
    """
    Search all notes in cache directory.

    Returns:
        List of (filepath, timestamp, matches) tuples
    """
    if not cache_dir.exists():
        return []

    results = []

    for filepath in cache_dir.glob('lab_note_*.md'):
        timestamp = parse_note_timestamp(filepath.name)
        if not timestamp:
            continue

        matches = search_note(filepath, pattern, case_sensitive, section)
        if matches:
            results.append((filepath, timestamp, matches))

    # Sort by timestamp, newest first
    results.sort(key=lambda x: x[1], reverse=True)

    return results


def format_search_results(
    results: List[Tuple[Path, datetime, List[Tuple[int, str]]]],
    show_context: bool = True
) -> str:
    """Format search results for display."""
    if not results:
        return "No matches found."

    lines = []
    total_matches = 0

    for filepath, timestamp, matches in results:
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"\n**{timestamp_str}** - `{filepath.name}` ({len(matches)} match(es))")

        if show_context:
            for line_num, line_content in matches:
                lines.append(f"  Line {line_num}: {line_content.strip()}")

        total_matches += len(matches)

    summary = f"\n\nFound {total_matches} match(es) across {len(results)} note(s)"
    return '\n'.join(lines) + summary


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Search quick notes for keywords or patterns',
        epilog='Supports regex patterns. Examples:\n'
               '  %(prog)s --keyword "linker error"\n'
               '  %(prog)s --keyword "skill.*compile" --regex\n'
               '  %(prog)s --keyword "template" --section Observation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--keyword',
        required=True,
        help='Search keyword or pattern'
    )
    parser.add_argument(
        '--regex',
        action='store_true',
        help='Treat keyword as regex pattern'
    )
    parser.add_argument(
        '--case-sensitive',
        action='store_true',
        help='Make search case-sensitive'
    )
    parser.add_argument(
        '--section',
        choices=['Task', 'Intent', 'Observation', 'Skills Used', 'Context'],
        help='Limit search to specific section'
    )
    parser.add_argument(
        '--no-context',
        action='store_true',
        help='Show only filenames, not matching lines'
    )
    parser.add_argument(
        '--cache-dir',
        type=Path,
        help='Override cache directory'
    )

    args = parser.parse_args()

    # Determine cache directory
    if args.cache_dir:
        cache_dir = args.cache_dir
    else:
        git_root = find_git_root()
        if not git_root:
            print("Error: Not in a git repository", file=sys.stderr)
            return 1
        cache_dir = git_root / ".claude" / "cache"

    # Prepare pattern
    pattern = args.keyword
    if not args.regex:
        # Escape special regex characters for literal search
        pattern = re.escape(pattern)

    # Search notes
    try:
        results = search_all_notes(
            cache_dir,
            pattern,
            case_sensitive=args.case_sensitive,
            section=args.section
        )
    except re.error as e:
        print(f"Error: Invalid regex pattern: {e}", file=sys.stderr)
        return 1

    # Format and display
    output = format_search_results(results, show_context=not args.no_context)
    print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
