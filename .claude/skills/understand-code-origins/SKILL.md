---
name: understand-code-origins
description: Trace code back to its origins to understand why it exists and what problem it solved. Use when investigating legacy code, understanding design decisions, or finding original context lost through refactorings. Combines git blame, pickaxe, and commit mining.
---

# Understand Code Origins

Use this skill when you need to **understand why code exists** by tracing it back through history to its original intent and context.

## When to Use

✅ **Use this when:**
- Investigating unfamiliar or legacy code
- Need to understand a design decision
- Code behavior seems strange and unclear
- Planning refactoring (need to understand intent first)
- Looking for related code or patterns

❌ **Don't use when:**
- Just need to find a bug introduction (use `find-bug-introduction`)
- Need refactoring priorities (use `identify-refactoring-targets`)
- Code is self-documenting with clear purpose

## Quick Start

### Find Who Last Modified Code

```bash
# Basic blame
git blame file.cpp

# Ignore whitespace, detect moves/copies
git blame -wCCC file.cpp
```

### Find When Code Was Added

```bash
# Search for string (when count changed)
git log -S"functionName" --oneline

# Search for pattern (any line change)
git log -G"regex.*pattern" --oneline
```

### Trace Function Evolution

```bash
# See complete history of a function
git log -L :functionName:file.cpp
```

**See** `templates/investigation_workflow.md` for systematic process.

## Core Techniques

### 1. Git Blame (Who/When)
**Purpose**: Find who last modified each line and when

- **Basic**: See current authors
- **With -M**: Detect moves within file
- **With -C/-CC/-CCC**: Detect copies across files
- **With -w**: Ignore whitespace changes

**See** `resources/git_blame_guide.md` for complete reference.

### 2. Git Pickaxe (What Changed)
**Purpose**: Find when specific code was added/removed/modified

- **-S option**: Find when string count changed (faster)
- **-G option**: Find when pattern appeared in diff (more thorough)
- **Difference**: -S = additions/removals, -G = modifications too

**See** `resources/pickaxe_techniques.md` for detailed comparison.

### 3. Line-Level History
**Purpose**: Trace specific lines or functions through time

```bash
# Function evolution
git log -L :functionName:file.cpp

# Line range evolution
git log -L 10,20:file.cpp
```

**See** `resources/line_history.md` for advanced usage.

### 4. Commit Message Mining
**Purpose**: Find context from structured metadata

- Search trailers (Fixes:, Reviewed-by:, Co-authored-by:)
- Link to issue trackers
- Find related discussions

**See** `resources/commit_mining.md` for trailer patterns.

## Common Scenarios

### Scenario 1: "Why Does This Code Do That?"

```bash
# 1. Find who wrote it
git blame -wCCC file.cpp | grep "suspicious_line"

# 2. See full commit
git show <commit-hash>

# 3. Find related discussion
git log --grep="<keyword>" --oneline
```

### Scenario 2: Code Moved Through Refactorings

Use **John Firebaugh's workflow**:

1. Start with `git blame -wCCC file.cpp`
2. If lands on refactoring commit, use `git gui blame file.cpp`
3. Right-click → "Blame Parent Commit" to skip refactoring
4. Repeat until find original introduction

**See** `resources/expert_workflows.md` § John Firebaugh.

### Scenario 3: Function Renamed/Moved

```bash
# Find when it disappeared from old location
git log -S"oldFunctionName" old/path/file.cpp

# Find when it appeared in new location
git log -S"newFunctionName" new/path/file.cpp

# Track across file renames
git log --follow --all -S"functionName"
```

## Progressive Resources

Start here, go deeper as needed:

1. **`resources/git_blame_guide.md`** - Complete blame reference (-w, -M, -C, -CC, -CCC)
2. **`resources/pickaxe_techniques.md`** - When to use -S vs -G, examples
3. **`resources/line_history.md`** - git log -L for function tracking
4. **`resources/commit_mining.md`** - Trailers, issue linking, metadata
5. **`resources/expert_workflows.md`** - Proven investigation patterns

## Tools

### Interactive GUI (Recommended)

```bash
# git gui blame - interactive drilling through history
git gui blame file.cpp
# Right-click → "Blame Parent Commit" to skip refactorings
```

**See** `resources/expert_workflows.md` § Interactive Tools.

### Command Line Workflow

**See** `scripts/investigate_code.sh` for automated investigation template.

## Key Best Practices

✅ **Do:**
- Start with blame, use pickaxe if needed
- Always use -w (ignore whitespace)
- Use -CCC for aggressive refactoring detection
- Read commit messages for context
- Check for file renames with --follow

❌ **Don't:**
- Trust blame alone (may show refactoring commit)
- Forget to check for moves/copies across files
- Skip reading full commit messages
- Assume first commit is the "real" origin (may be migrated code)

## Integration with Other Skills

**Before this skill**:
- May have used `find-bug-introduction` to identify culprit
- May have used `identify-refactoring-targets` to find hotspots

**After this skill**:
- Better informed for refactoring decisions
- Context for code review discussions
- Historical understanding for documentation

## Troubleshooting

**"Blame shows refactoring commit"**: See `resources/expert_workflows.md` § Skipping Refactorings

**"Can't find when code was added"**: See `resources/pickaxe_techniques.md` § -S vs -G

**"Function moved across files"**: See `resources/git_blame_guide.md` § Cross-File Detection

**"File was renamed"**: See `resources/line_history.md` § --follow Flag

## Citations

All sources documented in `resources/citations.md`.
