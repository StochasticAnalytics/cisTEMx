# Code Formatting Workflow

How clang-format integrates with the git commit workflow in cisTEMx.

## Overview

cisTEMx uses **clang-format-14** to maintain consistent C++/CUDA code style across the project. Formatting is enforced via a pre-commit hook and can be applied manually via a standalone script.

## Two Complementary Mechanisms

### 1. Pre-commit Hook (Automatic Check)

**Location**: `.git/hooks/pre-commit`

**Purpose**: Prevents commits with improperly formatted code

**How it works**:
1. Triggered automatically on `git commit`
2. Checks all staged C++/CUDA files against `.clang-format` config
3. **Blocks commit** if formatting issues found
4. Generates a fix script at `/tmp/fix_formatting_*.sh`

**Exclusions** (automatically skipped):
- `include/` - Third-party headers
- `src/gui/wxformbuilder/` - Form builder input files
- `*ProjectX_gui*` - Generated GUI code
- Files with "DO NOT EDIT" header warnings

### 2. Format-and-Stage Script (Manual Fix)

**Location**: `.claude/skills/version-control/scripts/format_and_stage.py`

**Purpose**: One-command formatting and staging

**Usage**:
```bash
python .claude/skills/version-control/scripts/format_and_stage.py
```

**How it works**:
1. Finds all staged C++/CUDA files
2. Applies same exclusion rules as pre-commit hook
3. Runs `clang-format-14 -i` to format in-place
4. Automatically stages formatted changes with `git add`
5. Reports summary of actions taken

**Advantages**:
- Single command to fix and stage
- No context usage (standalone script)
- Same exclusion logic as hook (stays in sync)
- Safe to run multiple times (idempotent)

## Recommended Workflow

### For Claude Code

When preparing commits:

1. **Stage changes**: `git add <files>`
2. **Format and re-stage**: `python .claude/skills/version-control/scripts/format_and_stage.py`
3. **Commit**: `git commit -m "message"`

If pre-commit hook fails (edge case):
- Run generated `/tmp/fix_formatting_*.sh` script
- Or manually format: `clang-format-14 -i <file> && git add <file>`

### For Human Developers

1. **Make changes**
2. **Stage**: `git add <files>`
3. **Attempt commit**: `git commit -m "message"`
4. **If blocked**: Run suggested `/tmp/fix_formatting_*.sh`
5. **Retry commit**

## Why Two Mechanisms?

**Pre-commit hook** = **Safety net**
- Catches formatting issues before they enter history
- Ensures all commits compile and follow standards
- Required by CLAUDE.md standards

**format_and_stage.py** = **Convenience tool**
- Proactively formats before hook runs
- Reduces approval round-trips for Claude Code
- Allows one-shot formatting without context usage

## Configuration

**Style config**: `.clang-format` (project root)

**Formatter**: `clang-format-14` (must be in PATH)

**Exclusion rules**: Defined identically in both:
- Pre-commit hook: `.git/hooks/pre-commit:should_exclude_file()`
- Python script: `format_and_stage.py:should_exclude_file()`

## Maintaining Sync

When updating exclusion rules, modify **both**:
1. `.git/hooks/pre-commit` (lines 23-50)
2. `.claude/skills/version-control/scripts/format_and_stage.py` (lines 43-78)

This ensures formatting behavior is consistent across automatic checks and manual application.

## Troubleshooting

**"clang-format-14 not found"**
- Install: `sudo apt install clang-format-14`
- Or add to PATH

**"File X keeps failing format check"**
- Check if it should be excluded (generated file?)
- Verify `.clang-format` config is valid
- Format manually and inspect diff

**"Pre-commit hook bypassed my commit"**
- Never use `git commit --no-verify` unless explicitly authorized
- Fix formatting issues properly instead of bypassing checks
