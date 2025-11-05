# Git Blame Complete Reference

Git blame shows who last modified each line of a file and when. Its real power lies in detecting code movement through refactorings.

## Basic Usage

```bash
# Basic blame
git blame file.cpp

# With full commit hashes and short date format
git blame -l --date=short file.cpp
```

**Output format**:
```
commit_hash (Author Name 2024-11-03 10:23:45 line_num) code content
```

## Essential Options

### -w (Ignore Whitespace)

**Always use this** - filters out formatting commits.

```bash
git blame -w file.cpp
```

Without -w, blame shows whoever reformatted the code, not who wrote it.

### -M (Move Detection Within File)

Detects lines moved or reorganized **within a single file**.

```bash
# Default threshold: 20 alphanumeric characters
git blame -M file.cpp

# Custom threshold
git blame -M30 file.cpp  # Require 30 characters minimum
```

**Use case**: Functions reordered during refactoring.

**Example**: Code moved from bottom to top of file - blame credits original author, not person who moved it.

### -C (Cross-File Copy Detection)

Detects lines moved or copied **from other files**.

Has three intensity levels:

#### -C (once): Same Commit

Searches files **modified in same commit**.

```bash
git blame -C file.cpp
```

**Use case**: Code extracted to new file in same commit.

#### -CC (twice): Created Files

Additionally searches files **created in that commit**.

```bash
git blame -CC file.cpp
```

**Use case**: File was split, code moved to newly created file.

#### -CCC (three times): Any Commit

Searches for copies from **any commit in history** (expensive!).

```bash
git blame -CCC file.cpp
```

**Use case**: Code copied from unrelated file at different time.

**Threshold**: Default 40 alphanumeric characters. Adjust with:
```bash
git blame -CCC50 file.cpp  # Require 50 characters
```

## Recommended Combination

**For code archaeology, use**:

```bash
git blame -wCCC file.cpp
```

This combination:
- Ignores whitespace (-w)
- Detects moves and copies aggressively (-CCC)
- Filters out most refactoring noise

## Additional Useful Options

### Line Range

```bash
# Specific line range
git blame -L 10,20 file.cpp

# Single line
git blame -L 42,42 file.cpp

# By function name
git blame -L :functionName file.cpp
```

### Show Email

```bash
git blame -e file.cpp
```

### Show Line Numbers

```bash
git blame -n file.cpp
```

### Show Original Line Numbers

Shows where code came from (useful with -M/-C):

```bash
git blame -f file.cpp
```

### Porcelain Format (For Parsing)

Machine-readable format:

```bash
git blame --porcelain file.cpp
```

## Investigating Code Origins

### Step 1: Find Author

```bash
git blame -wCCC file.cpp | grep "interesting_code"
```

Get commit hash from output.

### Step 2: See Full Commit

```bash
git show <commit-hash>
```

### Step 3: See Context Around That Time

```bash
# Related commits
git log --oneline --since="<commit-date>" --until="1.day.later" -- file.cpp

# What else changed in that commit
git show --stat <commit-hash>
```

## Handling Refactorings

### Problem: Blame Lands on Refactoring

**Symptom**: Blame shows commit like "Rename variables" or "Extract method"

**Solution**: Use interactive git gui blame

```bash
git gui blame file.cpp
```

**Workflow**:
1. Open file with git gui blame
2. Right-click line showing refactoring
3. Select "Blame Parent Commit"
4. Git re-blames from before refactoring
5. Repeat until find real origin

**See** `expert_workflows.md` for John Firebaugh's complete workflow.

## Performance Considerations

### Speed vs Thoroughness

| Option | Speed | Thoroughness |
|--------|-------|--------------|
| Basic blame | Fast | Low |
| -w | Fast | Medium |
| -M | Moderate | Medium |
| -C | Moderate | High |
| -CC | Slow | Higher |
| -CCC | Very Slow | Highest |

**Recommendation**:
- Start with `-wM` (fast, catches most cases)
- Escalate to `-wCCC` if needed (thorough but slow)

### Large Files

For large files, limit to relevant sections:

```bash
# Only blame specific function
git blame -wCCC -L :functionName large_file.cpp

# Only blame recent history
git blame -wCCC --since="1.year.ago" file.cpp
```

## Threshold Tuning

**Default thresholds**:
- -M: 20 characters
- -C/-CC/-CCC: 40 characters

**When to increase**:
- Too many false positives (unrelated code flagged as moved)
- Want only high-confidence moves

**When to decrease**:
- Missing legitimate small function moves
- Code is very terse (short lines)

**Examples**:
```bash
# Conservative (only obvious moves)
git blame -M50 -CCC60 file.cpp

# Aggressive (catch small moves)
git blame -M10 -CCC20 file.cpp
```

## Common Pitfalls

❌ **Not using -w**: Shows reformatter, not original author
❌ **Assuming blame is definitive**: May show refactoring commit
❌ **Not checking -CCC**: Misses cross-file copies
❌ **Forgetting thresholds**: Too strict or too loose
❌ **Not reading commit messages**: Missing context

## Integration with Other Tools

### Combine with git show

```bash
# Find commit
COMMIT=$(git blame -wCCC file.cpp | grep "code" | awk '{print $1}')

# See full commit
git show $COMMIT
```

### Combine with git log

```bash
# Find when line was introduced
git log -L 42,42:file.cpp

# Cross-reference with blame
git blame -L 42,42 file.cpp
```

## Real-World Examples

### Example 1: Who Wrote This Function?

```bash
git blame -wCCC src/renderer.cpp | grep "renderParticle"
```

Shows: `abc1234 (Alice 2023-05-15) void renderParticle(...)`

```bash
git show abc1234
```

See full commit context.

### Example 2: Code Moved from Another File

```bash
# Blame shows recent "Extract method" commit
git blame -wCCC src/new_file.cpp

# Shows: def5678 (Bob 2024-11-01) extracted from calculations.cpp
```

The -CCC detected cross-file copy. Check original:

```bash
git log -S"extractedFunction" src/calculations.cpp
```

### Example 3: Function Renamed

```bash
# Can't find old name with blame
git log -S"oldFunctionName" --oneline
# Shows: ghi9012 Rename oldFunctionName to newFunctionName

git show ghi9012
```

## Next Steps

- **Need to find when code changed?** See `pickaxe_techniques.md`
- **Want function evolution?** See `line_history.md`
- **Interactive workflow?** See `expert_workflows.md`
- **Systematic investigation?** See `../templates/investigation_workflow.md`
