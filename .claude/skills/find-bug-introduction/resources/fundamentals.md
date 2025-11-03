# Git Bisect Fundamentals

## What Is Git Bisect?

Git bisect performs **binary search** through commit history to find the specific commit that introduced a bug or changed behavior.

### How It Works

1. You mark a **bad** commit (bug exists)
2. You mark a **good** commit (bug doesn't exist)
3. Git checks out the **middle** commit
4. You test and mark as good or bad
5. Repeat until Git identifies the culprit

### Time Complexity

**O(log₂ n)** - logarithmic time

| Commits to Search | Tests Required |
|------------------|----------------|
| 100 | 7 |
| 1,024 | 10 |
| 10,000 | 14 |
| 20,000 | 15 |

**Example**: Finding a bug in the Linux kernel (750,000+ commits) takes ~20 tests.

## When to Use Bisect

### ✅ Good Use Cases

**Reproducible regressions**:
- "Feature X worked in v1.0, broken in v2.0"
- "Performance was good 6 months ago, slow now"
- "Tests passed last week, failing now"

**Requirements**:
- Automated test (script/command that exits 0 or non-zero)
- Known good and bad commits
- Reasonable test speed (runs O(log n) times)

### ❌ Don't Use When

**Not reproducible**:
- Heisenbug (disappears when investigated)
- Race condition without consistent reproduction
- Environmental issues (specific hardware/config)

**Not efficient**:
- Only 2-3 commits to check (just test them)
- Test takes hours (impractical to run 10-15 times)
- Already know the culprit (use `git show` instead)

**Not automatable**:
- Requires manual verification (UI issues, visual bugs)
- No clear pass/fail criteria
- However: Manual bisect is still possible, just slower

## Real-World Impact

### Case Study: Andreas Ericsson's Team

**Before git bisect**:
- Average bug resolution: 142.6 hours
- Manual commit review

**After git bisect**:
- Average bug resolution: 16.2 hours
- **88.6% reduction** in debugging time

### Case Study: Linux Kernel Development

**Ingo Molnar's automated bisect**:
- Fully automated bootup-hang detection
- Serial log monitoring with power cycling
- Runs autonomously without human intervention
- Critical tool for kernel stability

## Basic Workflow

### Manual Bisect

```bash
# 1. Start bisecting
git bisect start

# 2. Mark current commit as bad
git bisect bad

# 3. Mark a known good commit
git bisect good v1.0.0

# Git checks out middle commit
# Test it manually, then:

# If bug exists:
git bisect bad

# If bug doesn't exist:
git bisect good

# Repeat until Git identifies culprit
# Git will say: "<commit-hash> is the first bad commit"

# 4. Clean up
git bisect reset
```

### Automated Bisect (Recommended)

```bash
# 1. Start bisect
git bisect start HEAD v1.0.0

# 2. Let Git automate the testing
git bisect run ./test_script.sh

# Git will:
# - Check out each commit
# - Run your test script
# - Mark as good/bad based on exit code
# - Identify the culprit
# - Reset automatically
```

**See** `../scripts/bisect_template.sh` for test script template.

## Output Interpretation

When bisect completes, Git shows:

```
<commit-hash> is the first bad commit
commit <full-hash>
Author: Name <email>
Date:   Date

    Commit message

:100644 100644 <old> <new> M  path/to/file.cpp
```

### Next Steps

1. **Examine the commit**:
   ```bash
   git show <commit-hash>
   ```

2. **Verify it's the culprit**:
   - Revert the commit: does bug disappear?
   - Cherry-pick to earlier version: does bug appear?

3. **Understand the fix**:
   - Why did this change introduce the bug?
   - What needs to be changed?
   - Are there related issues?

4. **Check blame for context**:
   ```bash
   git blame <file>
   git log -p <commit-hash>
   ```

## Terminology

**good/bad**: Default terms
- `git bisect good` - bug not present
- `git bisect bad` - bug present

**Custom terms**: Use contextually appropriate language
```bash
git bisect start --term-old fast --term-new slow
git bisect slow  # For performance regressions
git bisect fast
```

Other examples:
- `broken/fixed` - feature regressions
- `old/new` - API changes
- `working/broken` - general failures

## Limitations

**What bisect CAN'T do**:
- Fix the bug (only finds it)
- Handle non-deterministic bugs reliably
- Work on shallow clones (need full history)
- Bisect multiple independent bugs simultaneously

**What bisect CAN do**:
- Find exact commit that introduced issue
- Work with any testable property (correctness, performance, behavior)
- Handle merge commits (with `--first-parent`)
- Skip untestable commits
- Save/replay sessions

## Performance Characteristics

### Time Savings

**Manual approach** (linear search):
- Check each commit: O(n) tests
- 1,000 commits = 1,000 tests (or educated guessing)

**Bisect approach**:
- Binary search: O(log n) tests
- 1,000 commits = 10 tests
- **100x fewer tests**

### When Bisect Is Fast

- Fast tests (seconds, not minutes)
- Every commit builds successfully
- Test is deterministic (always same result)

### When Bisect Is Slow

- Slow tests (minutes each)
- Many commits don't build (need skip logic)
- Flaky tests (need retry logic)
- Very large repositories (though still faster than linear)

**Solution**: See `edge_cases.md` for handling slow scenarios.

## Next Steps

- **Ready to bisect?** See `../templates/workflow_checklist.md`
- **Need to write test script?** See `automated_bisecting.md`
- **Hit edge cases?** See `edge_cases.md`
- **Want optimization?** See `advanced_techniques.md`
