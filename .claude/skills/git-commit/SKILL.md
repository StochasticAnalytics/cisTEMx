---
name: git-commit
description: Create properly formatted git commits. Structure commit message and ensure code formatting prior to commit. Use before analyzing changes with other git tools like diff, log, status. Handles staging of changes git worktree, commit message formatting, git commit message templates, and pre-commit hook integration.
allowed-tools:  Bash(git status:*),Bash(git diff:*),Bash(git log:*),Bash(git branch:*),Bash(git show-branch:*),Bash(git merge-base:*), Bash(git ls-tree:*),Bash(git add:*), Bash(git commit -m:*)Read, Grep, Glob
---

# Git Commit

Guidance for creating well-formatted commits in cisTEMx, emphasizing commit discipline and code formatting workflow.

## Quick Reference

### Format and Stage

Before committing C++/CUDA code:

```bash
python .claude/skills/git-commit/scripts/format_and_stage.py
```

This script:

- Formats all staged C++/CUDA files with clang-format-14
- Automatically re-stages formatted changes
- Applies same exclusion rules as pre-commit hook
- Single approval, no context usage

See `resources/formatting_workflow.md` for complete formatting integration details.

### Commit Message Structure

```
<type>: <summary in ≤50 chars>

[Optional body explaining why]
```

**Types**: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `build`, `style`, `chore`

**Key principles**:

- Imperative mood ("Add" not "Added")
- Explain **why**, not what (diff shows what)
- Concise summary (≤50 chars)
- Frequent, focused commits

## Commit Workflow

1. **Make changes** to code
2. **Stage**: `git add <files>`
3. **Format and re-stage**: `python .claude/skills/git-commit/scripts/format_and_stage.py`
4. **Commit**: `git commit -m "type: summary"`
5. **If pre-commit hook fails**: Run generated fix script, retry

## Resources

### Detailed Guidance

- `resources/commit_best_practices.md` - Comprehensive commit message guidelines synthesized from recent research
- `resources/formatting_workflow.md` - How clang-format integrates with git workflow
- `templates/commit_examples.md` - Real-world examples for cisTEMx commits

### Scripts

- `scripts/format_and_stage.py` - One-shot formatting and staging for C++/CUDA files

## cisTEMx Standards

**From CLAUDE.md**:

- **Every commit must compile** - Non-negotiable for clean history
- **Commit frequently** - One discrete task per commit
- **Clean up debug code** - Remove `// revert` markers before committing
- **Never bypass hooks** - Don't use `--no-verify` without explicit authorization

## Common Patterns

### Simple Bug Fix

```bash
git add src/core/image.cpp
python .claude/skills/git-commit/scripts/format_and_stage.py
git commit -m "fix: Prevent crash on empty image stack"
```

### New Feature with Tests

```bash
git add src/core/new_feature.cpp tests/test_new_feature.cpp
python .claude/skills/git-commit/scripts/format_and_stage.py
git commit -m "feat: Add GPU-accelerated correlation function"
```

### Multiple Related Changes

Commit each logical change separately:

```bash
# First commit: implementation
git add src/core/feature.cpp
python .claude/skills/git-commit/scripts/format_and_stage.py
git commit -m "feat: Add new particle picker algorithm"

# Second commit: tests
git add tests/test_picker.cpp
python .claude/skills/git-commit/scripts/format_and_stage.py
git commit -m "test: Add unit tests for particle picker"

# Third commit: documentation
git add docs/particle_picking.md
git commit -m "docs: Document new picker algorithm parameters"
```

## Troubleshooting

**Pre-commit hook blocks commit**:

- Check generated `/tmp/fix_formatting_*.sh` script
- Or re-run `format_and_stage.py`

**Unsure about commit message type**:

- See `templates/commit_examples.md` for similar cases
- When in doubt, use `feat` for new functionality, `fix` for corrections

**Multiple unrelated changes staged**:

- Unstage: `git reset HEAD <file>`
- Commit related changes separately
- Keeps history clean and reviewable

## Integration with Pre-commit Hook

The pre-commit hook (`.git/hooks/pre-commit`) and `format_and_stage.py` share identical exclusion logic:

**Excluded from formatting**:

- `include/` - Third-party headers
- `src/gui/wxformbuilder/` - Form builder files
- `*ProjectX_gui*` - Generated GUI code
- Files with "DO NOT EDIT" warnings in header

When updating exclusions, modify both locations to maintain consistency.
