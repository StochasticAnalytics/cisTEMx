# Git Pickaxe Techniques: -S vs -G

The "pickaxe" refers to git's -S and -G options for searching commits that changed specific code.

## Quick Comparison

| Feature | -S | -G |
|---------|----|----|
| Matches | Literal string (or regex with --pickaxe-regex) | Always regex |
| Triggers on | **Count** changed | **Any line** change |
| Detects in-file moves | No | Yes |
| Performance | Faster | Slower (2x diff + grep) |
| Use case | When code added/removed | When code modified |

## -S Option (String Count Search)

### How It Works

Finds commits where the **number of occurrences** changed.

```bash
git log -S"functionName" --oneline
```

**Triggers when**: String appears more or fewer times (added or removed).

**Does NOT trigger when**: String moved within file (count stays same).

### Basic Usage

```bash
# Find when "calculateTotal" was added or removed
git log -S"calculateTotal" --oneline

# Show full diffs
git log -S"calculateTotal" -p

# Search all branches
git log --all -S"API_KEY"

# Specific file
git log -S"functionName" -- path/to/file.cpp
```

### With Regex (--pickaxe-regex)

```bash
# Use regex with -S
git log -S'frotz\(nitfol' --pickaxe-regex

# Character class
git log -S'function[0-9]+' --pickaxe-regex
```

### When to Use -S

✅ **Use -S when:**
- Finding when function was introduced
- Finding when function was removed
- Finding when API changed
- Performance matters (faster than -G)
- Looking for additions/deletions

## -G Option (Regex Line Search)

### How It Works

Finds commits where regex pattern appears in an **added or removed line**.

```bash
git log -G"function.*initialize" --oneline
```

**Triggers when**: Pattern in any added/removed line, even if count unchanged.

**Always uses regex** (no --pickaxe-regex flag needed).

### Basic Usage

```bash
# Find any change to lines containing pattern
git log -G"function.*initialize" --oneline

# With patches
git log -G"TODO.*critical" -p

# Specific file
git log -G"if\s*\(\s*DEBUG\s*\)" -- file.cpp
```

### When to Use -G

✅ **Use -G when:**
- Finding modifications (not just add/remove)
- Finding in-file moves
- Need regex matching
- Looking for subtle changes
- -S returns too few results

## Concrete Example: -S vs -G

**Commit changes**: `frotz(nitfol` → `frotz(nitfol, 42)`

```bash
# -G will show this commit (line changed)
git log -G'frotz\(nitfol' --oneline
# Output: abc1234 Add parameter to frotz

# -S will NOT show (count didn't change - still appears once)
git log -S'frotz\(nitfol' --pickaxe-regex --oneline
# Output: (empty)
```

## Advanced Usage

### Show All Files in Matching Commits

By default, only shows matching files. Use --pickaxe-all to see entire changeset:

```bash
git log -S"API_KEY" --pickaxe-all --stat
```

Useful for understanding context of change.

### Time Range

```bash
# Last 6 months
git log -S"functionName" --since="6.month.ago" --oneline

# Specific date range
git log -G"pattern" --since="2024-01-01" --until="2024-12-31"
```

### Author Filter

```bash
# Changes by specific author
git log -S"functionName" --author="Alice" --oneline

# Exclude author
git log -G"pattern" --author="^((?!Bob).)*$" --perl-regexp --oneline
```

### With Patches and Context

```bash
# Show 5 lines of context around changes
git log -S"function" -p -U5

# Show function names
git log -G"pattern" -p --function-context
```

## Performance Optimization

### -S is Faster

```bash
# Fast: Count-based
git log -S"text" --oneline

# Slower: Line-based with regex
git log -G"text" --oneline
```

**When performance matters**, start with -S. Escalate to -G if needed.

### Limit Scope

```bash
# Only specific paths
git log -S"text" -- src/

# Only specific branch
git log -S"text" main

# Limited time range
git log -S"text" --since="1.year.ago"
```

### Avoid --all Unless Necessary

```bash
# ❌ Slow: Searches all branches
git log --all -S"text"

# ✅ Fast: Search only main
git log -S"text" main
```

## Decision Tree: Which to Use?

**Start here**:
1. **Finding introduction/removal?** → Use -S
2. **Finding modifications?** → Use -G
3. **-S returns nothing?** → Try -G (may have been modified, not added/removed)
4. **-G too noisy?** → Try -S with more specific string

## Common Workflows

### Workflow 1: When Was Function Introduced?

```bash
# Find introduction
git log -S"renderParticle" --diff-filter=A --oneline
# A = Added (first appearance)

# Show the commit
git log -S"renderParticle" --diff-filter=A -p | head -50
```

### Workflow 2: When Did API Change?

```bash
# Find when old API disappeared
git log -S"oldAPIcall" --oneline

# Find when new API appeared
git log -S"newAPIcall" --oneline

# Compare contexts
git show <old-api-commit>
git show <new-api-commit>
```

### Workflow 3: Track TODO Comments

```bash
# Find all commits adding/removing TODOs
git log -G"TODO" --oneline

# Find specific TODO
git log -G"TODO.*performance" --oneline

# With context
git log -G"TODO" -p | grep -A 3 "TODO"
```

### Workflow 4: Find When Code Moved

```bash
# -S won't find it (count same)
git log -S"codeBlock" --oneline  # Empty

# -G will find it (line changed)
git log -G"codeBlock" --oneline  # Shows move commit
```

## Integration with Other Tools

### Combine with Blame

```bash
# Find when code introduced
git log -S"functionName" --diff-filter=A --format="%H"

# Then blame at that point
git blame <commit>^ -- file.cpp | grep "functionName"
```

### Combine with Bisect

```bash
# If you know feature appeared between commits
git bisect start HEAD v1.0
git bisect run sh -c "git log -S'featureName' --oneline | grep -q ."
```

### Pipe to Other Commands

```bash
# Count how many times function was modified
git log -G"functionName" --oneline | wc -l

# Extract commit hashes
git log -S"pattern" --format="%H" > commits.txt
```

## Common Pitfalls

❌ **Using -S for modifications**: Won't catch changes that don't affect count
❌ **Using -G everywhere**: Slower, may be noisy
❌ **Forgetting --pickaxe-regex with -S**: Treats regex as literal string
❌ **Not limiting scope**: Searches everything, very slow
❌ **Assuming first match is origin**: Code may have been moved/renamed earlier

## Troubleshooting

**"Can't find when code was added"**:
- Try both -S and -G
- Check if file was renamed (use --follow)
- Search all branches (--all)
- Code may have been migrated from another repo

**"-G returns too many results"**:
- Make pattern more specific
- Use -S instead if appropriate
- Limit time range or paths
- Combine with other filters (--author, --since)

**"Performance is too slow"**:
- Use -S instead of -G
- Limit scope (paths, branches, dates)
- Avoid --all
- Consider commit-graph optimization (see performance optimization resources)

## Next Steps

- **Need line-level history?** See `line_history.md`
- **Want to see who wrote code?** See `git_blame_guide.md`
- **Interactive workflow?** See `expert_workflows.md`
- **Systematic investigation?** See `../templates/investigation_workflow.md`
