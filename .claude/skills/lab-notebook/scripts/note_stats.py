#!/usr/bin/env python3
"""
Show statistics about lab notes.

Useful for determining when to create daily summaries,
understanding note-taking patterns, and tracking productivity.
"""

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import path validation utilities (via symlink to shared common module)
from path_validation import validate_path_within_project, find_git_root as git_root_from_validation


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


def extract_skills_from_note(filepath: Path) -> List[str]:
    """Extract skill names mentioned in note."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        skills = []
        in_skills_section = False

        for line in content.split('\n'):
            if line.strip() == '## Skills Used':
                in_skills_section = True
                continue
            elif line.startswith('##'):
                in_skills_section = False

            if in_skills_section and '**Name**:' in line:
                # Extract skill name from: - **Name**: `skill-name`
                parts = line.split('`')
                if len(parts) >= 2:
                    skills.append(parts[1])

        return skills

    except Exception:
        return []


def gather_statistics(
    cache_dir: Path,
    since_date: Optional[datetime] = None
) -> Dict:
    """
    Gather statistics about lab notes.

    Args:
        cache_dir: Directory containing notes
        since_date: Only include notes since this date

    Returns:
        Dictionary of statistics
    """
    if not cache_dir.exists():
        return {
            'total_notes': 0,
            'by_date': {},
            'by_hour': defaultdict(int),
            'skills_used': defaultdict(int),
            'date_range': None
        }

    notes_by_date = defaultdict(list)
    notes_by_hour = defaultdict(int)
    skills_counter = defaultdict(int)
    all_timestamps = []

    for filepath in cache_dir.glob('lab_note_*.md'):
        timestamp = parse_note_timestamp(filepath.name)
        if not timestamp:
            continue

        # Filter by date if specified
        if since_date and timestamp < since_date:
            continue

        all_timestamps.append(timestamp)

        # Track by date
        date_key = timestamp.strftime("%Y-%m-%d")
        notes_by_date[date_key].append(timestamp)

        # Track by hour of day
        hour = timestamp.hour
        notes_by_hour[hour] += 1

        # Track skills used
        skills = extract_skills_from_note(filepath)
        for skill in skills:
            skills_counter[skill] += 1

    # Calculate date range
    date_range = None
    if all_timestamps:
        date_range = (min(all_timestamps), max(all_timestamps))

    return {
        'total_notes': len(all_timestamps),
        'by_date': dict(notes_by_date),
        'by_hour': dict(notes_by_hour),
        'skills_used': dict(skills_counter),
        'date_range': date_range
    }


def format_statistics(stats: Dict, show_detailed: bool = True) -> str:
    """Format statistics for display."""
    lines = []

    # Overall summary
    lines.append("# Lab Notebook Statistics")
    lines.append("")
    lines.append(f"**Total notes**: {stats['total_notes']}")

    if stats['date_range']:
        start, end = stats['date_range']
        lines.append(f"**Date range**: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    lines.append("")

    # Notes by date
    if stats['by_date'] and show_detailed:
        lines.append("## Notes by Date")
        for date_str in sorted(stats['by_date'].keys(), reverse=True):
            count = len(stats['by_date'][date_str])
            lines.append(f"- **{date_str}**: {count} note(s)")
        lines.append("")

    # Most productive hours
    if stats['by_hour'] and show_detailed:
        lines.append("## Most Productive Hours")
        sorted_hours = sorted(stats['by_hour'].items(), key=lambda x: x[1], reverse=True)
        for hour, count in sorted_hours[:5]:  # Top 5
            time_str = f"{hour:02d}:00-{hour:02d}:59"
            lines.append(f"- **{time_str}**: {count} note(s)")
        lines.append("")

    # Skills usage
    if stats['skills_used'] and show_detailed:
        lines.append("## Skills Used")
        sorted_skills = sorted(stats['skills_used'].items(), key=lambda x: x[1], reverse=True)
        for skill, count in sorted_skills:
            lines.append(f"- **{skill}**: {count} time(s)")
        lines.append("")

    return '\n'.join(lines)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Show statistics about lab notes'
    )

    parser.add_argument(
        '--today',
        action='store_true',
        help='Show only statistics for today'
    )
    parser.add_argument(
        '--since',
        help='Show statistics since date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Show only summary (no detailed breakdowns)'
    )
    parser.add_argument(
        '--cache-dir',
        type=Path,
        help='Override cache directory'
    )

    args = parser.parse_args()

    # Determine cache directory
    git_root = find_git_root()
    if not git_root:
        print("Error: Not in a git repository", file=sys.stderr)
        return 1

    if args.cache_dir:
        cache_dir = args.cache_dir
        # Validate custom cache directory is within project
        valid, error = validate_path_within_project(cache_dir, git_root, "Cache directory")
        if not valid:
            print(error, file=sys.stderr)
            return 1
    else:
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

    # Gather and display statistics
    stats = gather_statistics(cache_dir, since_date)
    output = format_statistics(stats, show_detailed=not args.summary_only)
    print(output)

    # Return exit code indicating if notes exist for today
    if args.today:
        return 0 if stats['total_notes'] > 0 else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
