# !/usr/bin/env python3
from pathlib import Path
import sys
import os

"""
Skill Loading Validator

Runs on SessionStart to verify that Claude Code has loaded all expected skills into memory.
Writes expected skills to /tmp (Claude can't auto-read) and prompts user to manually verify.
"""


def get_expected_skills():
    """List all skills that should be available to Claude."""
    skills_dir = Path("/workspaces/cisTEMx/.claude/skills")

    if not skills_dir.exists():
        return []

    skills = []
    for item in skills_dir.iterdir():
        if item.is_dir() and not item.name.startswith("__"):
            skill_file = item / "SKILL.md"
            if skill_file.exists():
                skills.append(item.name)

    return sorted(skills)


def main():
    """Generate validation output and user prompt."""
    skills = get_expected_skills()

    if not skills:
        print("WARNING: No skills found in .claude/skills/")
        return

    # Write expected skills to /tmp (hidden from Claude)
    output_file = "/tmp/expected_skills.txt"
    with open(output_file, "w") as f:
        for skill in skills:
            f.write(f"{skill}\n")

    # Print direct prompt to Claude (visible in normal view via stderr + exit code 1)
    print(file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print("SKILL LOADING VALIDATION", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(file=sys.stderr)
    print(f"Expected skills written to: {output_file}", file=sys.stderr)
    print(file=sys.stderr)
    print("Opened in verbose mode to make this message prominent. use /config to turn off if wanted", file=sys.stderr)
    print(file=sys.stderr)
    print("Give this prompt to claude and compare to the expected output", file=sys.stderr)
    print(file=sys.stderr)
    print("Claude: Please check your memory and list what skills you see loaded.", file=sys.stderr)
    print("Do not access the file system or use any other tools.", file=sys.stderr)
    print("Just report the skill names from your memory.", file=sys.stderr)
    print(file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    # Exit with code 1 to make stderr visible to user in normal view
    sys.exit(1)


if __name__ == "__main__":
    main()
