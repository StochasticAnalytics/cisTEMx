#!/usr/bin/env python3
"""
List recent quick notes from lab notebook.

Shows most recent notes by default, with options to filter by date
or limit the number shown.
"""

import argparse
import sys
from datetime import datetime, timedelta
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
    # Format: lab_note_YYYYMMDD_HHMMSS.md
    if not filename.startswith('lab_note_') or not filename.endswith('.md'):
        return None

    try:
        timestamp_str = filename[9:-3]  # Remove 'lab_note_' and '.md'
        return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def extract_note_preview(filepath: Path, max_lines: int = 3) -> str:
    """Extract first few lines of observation section."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        # Find observation section
        lines = content.split('\n')
        in_observation = False
        observation_lines = []

        for line in lines:
            if line.strip() == '## Observation':
                in_observation = True
                continue
            elif in_observation:
                if line.startswith('##'):  # Next section
                    break
                if line.strip():  # Non-empty line
                    observation_lines.append(line.strip())
                    if len(observation_lines) >= max_lines:
                        break

        preview = ' '.join(observation_lines)
        if len(preview) > 100:
            preview = preview[:97] + '...'

        return preview if preview else '(empty)'

    except Exception:
        return '(could not read)'


def list_notes(
    cache_dir: Path,
    last_n: Optional[int] = None,
    since_date: Optional[datetime] = None
) -> List[Tuple[Path, datetime]]:
    """
    List lab notes from cache directory.

    Args:
        cache_dir: Directory containing lab notes
        last_n: Limit to last N notes
        since_date: Only show notes since this date

    Returns:
        List of (filepath, timestamp) tuples, sorted newest first
    """
    if not cache_dir.exists():
        return []

    # Find all lab note files
    notes = []
    for filepath in cache_dir.glob('lab_note_*.md'):
        timestamp = parse_note_timestamp(filepath.name)
        if timestamp:
            # Filter by date if specified
            if since_date and timestamp < since_date:
                continue
            notes.append((filepath, timestamp))

    # Sort by timestamp, newest first
    notes.sort(key=lambda x: x[1], reverse=True)

    # Limit if requested
    if last_n is not None:
        notes = notes[:last_n]

    return notes


def format_note_list(notes: List[Tuple[Path, datetime]], show_preview: bool = True) -> str:
    """Format notes list for display."""
    if not notes:
        return "No notes found."

    lines = []
    for filepath, timestamp in notes:
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"\n**{timestamp_str}** - `{filepath.name}`")

        if show_preview:
            preview = extract_note_preview(filepath)
            lines.append(f"  {preview}")

    return '\n'.join(lines)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='List recent quick notes from lab notebook'
    )

    parser.add_argument(
        '--last',
        type=int,
        default=10,
        help='Show last N notes (default: 10)'
    )
    parser.add_argument(
        '--since',
        help='Show notes since date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--today',
        action='store_true',
        help='Show only notes from today'
    )
    parser.add_argument(
        '--no-preview',
        action='store_true',
        help='Hide observation previews'
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

    # Parse date filter
    since_date = None
    if args.today:
        since_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    elif args.since:
        try:
            since_date = datetime.strptime(args.since, "%Y-%m-%d")
        except ValueError:
            print(f"Error: Invalid date format: {args.since}", file=sys.stderr)
            print("Expected: YYYY-MM-DD", file=sys.stderr)
            return 1

    # List notes
    notes = list_notes(cache_dir, last_n=args.last, since_date=since_date)

    # Format and display
    output = format_note_list(notes, show_preview=not args.no_preview)
    print(output)

    if notes:
        print(f"\nTotal: {len(notes)} note(s)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
