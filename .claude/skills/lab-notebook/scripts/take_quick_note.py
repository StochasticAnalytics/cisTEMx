#!/usr/bin/env python3
"""
Record quick notes during work sessions for later distillation.

Creates timestamped markdown files in .claude/cache/ capturing:
- Task and intent
- Observations and learnings
- Skills usage patterns
- Context metrics (tokens, tools)

These notes form the raw material for daily summaries.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


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


def format_skills_section(skills_json: Optional[str]) -> str:
    """Format skills usage information."""
    if not skills_json:
        return "*(No skills logged)*\n"

    try:
        skills = json.loads(skills_json)

        # Handle both single skill dict and list of skills
        if isinstance(skills, dict):
            skills = [skills]

        output = []
        for skill in skills:
            output.append(f"- **Name**: `{skill.get('name', 'unknown')}`")
            output.append(f"  - **Invoked by**: {skill.get('invoker', 'unknown')}")

            scripts = skill.get('scripts', [])
            if scripts:
                output.append(f"  - **Scripts used**: {', '.join(f'`{s}`' for s in scripts)}")

            refs = skill.get('refs', [])
            if refs:
                output.append(f"  - **References used**: {', '.join(f'`{r}`' for r in refs)}")

            refs_not = skill.get('refs_not_used', [])
            if refs_not:
                output.append(f"  - **References available but not used**: {', '.join(f'`{r}`' for r in refs_not)}")

        return '\n'.join(output) + '\n'

    except json.JSONDecodeError as e:
        return f"*(Invalid JSON: {e})*\n"


def format_context_section(tokens: Optional[str], tools: Optional[int]) -> str:
    """Format context metrics."""
    lines = []

    if tokens:
        # Parse "used/total" format
        if '/' in tokens:
            used, total = tokens.split('/')
            try:
                used_int = int(used.strip())
                total_int = int(total.strip())
                pct = (used_int / total_int) * 100
                lines.append(f"- **Tokens**: {used_int:,}/{total_int:,} ({pct:.1f}%)")
            except (ValueError, ZeroDivisionError):
                lines.append(f"- **Tokens**: {tokens}")
        else:
            lines.append(f"- **Tokens**: {tokens}")

    if tools is not None:
        lines.append(f"- **Tools used this session**: {tools}")

    return '\n'.join(lines) + '\n' if lines else "*(No context metrics provided)*\n"


def create_note(
    task: str,
    intent: str,
    observation: str,
    skills_used: Optional[str] = None,
    tokens_used: Optional[str] = None,
    tools_count: Optional[int] = None,
    output_dir: Optional[Path] = None
) -> Path:
    """
    Create a timestamped quick note.

    Args:
        task: What you're working on
        intent: Why you're doing it
        observation: What happened, what you learned
        skills_used: JSON string of skills usage
        tokens_used: Token usage in "used/total" format
        tools_count: Number of tools used
        output_dir: Override default cache directory

    Returns:
        Path to created note file
    """
    # Determine output directory
    if output_dir is None:
        git_root = find_git_root()
        if not git_root:
            print("Error: Not in a git repository", file=sys.stderr)
            sys.exit(1)
        output_dir = git_root / ".claude" / "cache"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp and filename
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    filename = f"lab_note_{timestamp}.md"
    filepath = output_dir / filename

    # Format timestamp for display
    display_time = now.strftime("%Y-%m-%d %H:%M:%S")

    # Build note content
    content_lines = [
        f"# Quick Note - {display_time}",
        "",
        "## Task",
        task,
        "",
        "## Intent",
        intent,
        "",
        "## Observation",
        observation,
        "",
        "## Skills Used",
        format_skills_section(skills_used),
        "## Context",
        format_context_section(tokens_used, tools_count),
        "---",
        "",
        "*This note will be distilled into daily summary at end of day*",
        ""
    ]

    content = '\n'.join(content_lines)

    # Write file
    with open(filepath, 'w') as f:
        f.write(content)

    return filepath


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Record quick notes during work sessions',
        epilog='Example: %(prog)s --task "Implement compile skill" '
               '--intent "Replace cpp-build-expert agent" '
               '--observation "Template errors excluded - wait for C++20"'
    )

    parser.add_argument(
        '--task',
        required=True,
        help='What you are working on'
    )
    parser.add_argument(
        '--intent',
        required=True,
        help='Why you are doing this task'
    )
    parser.add_argument(
        '--observation',
        required=True,
        help='What you observed, learned, or discovered'
    )
    parser.add_argument(
        '--skills-used',
        help='JSON string describing skills used (see documentation for format)'
    )
    parser.add_argument(
        '--tokens-used',
        help='Token usage in "used/total" format (e.g., "50000/200000")'
    )
    parser.add_argument(
        '--tools-count',
        type=int,
        help='Number of tools used in this work session'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Override default output directory (.claude/cache/)'
    )

    args = parser.parse_args()

    # Create the note
    filepath = create_note(
        task=args.task,
        intent=args.intent,
        observation=args.observation,
        skills_used=args.skills_used,
        tokens_used=args.tokens_used,
        tools_count=args.tools_count,
        output_dir=args.output_dir
    )

    print(f"✓ Note recorded: {filepath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
