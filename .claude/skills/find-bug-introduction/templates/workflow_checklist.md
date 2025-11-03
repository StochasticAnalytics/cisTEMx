# Git Bisect Workflow Checklist

Step-by-step process for finding bug introduction. Follow this checklist for systematic bisecting.

## Phase 1: Preparation

### ☐ Verify Bug is Reproducible

```bash
# Test on current (bad) commit
./run_test
echo $?  # Should be non-zero (failure)
```

**If not reproducible**: Fix test first, or bisect won't work.

### ☐ Find a Known Good Commit

```bash
# Try recent releases
git checkout v1.0
./run_test
echo $?  # Should be 0 (success)
```

**Strategies**:
- Try last stable release
- Try commit from 1 week/month ago
- Binary search manually to find approximate good commit

### ☐ Verify Build System Works

```bash
# On both good and bad commits
git checkout <bad-commit>
make clean && make  # Should succeed

git checkout <good-commit>
make clean && make  # Should succeed
```

**If builds fail**: You'll need exit 125 handling in test script.

### ☐ Estimate Time Required

Calculate: `log₂(commits) × test_time`

```bash
# Count commits between good and bad
commits=$(git rev-list --count <good-commit>..<bad-commit>)
echo "Commits to search: $commits"
echo "Estimated tests: $(echo "l($commits)/l(2)" | bc -l | xargs printf "%.0f")"

# Time per test (including build)
time (make clean && make && ./run_test)
# Multiply by estimated tests for total time
```

**Acceptable**: < 1 hour total
**Long**: 1-4 hours (consider pathspecs or parallel)
**Too long**: > 4 hours (optimize test or narrow scope)

---

## Phase 2: Create Test Script

### ☐ Choose Template

- **Simple test**: Use `../scripts/bisect_template.sh`
- **Flaky test**: Use `../scripts/bisect_with_retry.sh`
- **Custom**: See below

### ☐ Write Test Script

```bash
#!/bin/bash

# 1. Build (exit 125 if fails)
make clean && make || exit 125

# 2. Test (exit 0 if good, 1 if bad)
./run_specific_test

# Exit code is automatically used
```

**Key points**:
- Make executable: `chmod +x test_script.sh`
- Test on good commit (should exit 0)
- Test on bad commit (should exit non-zero, not 125)

### ☐ Validate Test Script

```bash
# Test on known good commit
git checkout <good-commit>
./test_script.sh
echo $?  # Should be 0

# Test on known bad commit
git checkout <bad-commit>
./test_script.sh
echo $?  # Should be 1 (or 2-124, not 0 or 125)

# Test on commit that won't build (if applicable)
git checkout <broken-commit>
./test_script.sh
echo $?  # Should be 125
```

**If any test fails**: Fix script before proceeding.

---

## Phase 3: Run Bisect

### ☐ Start Bisect

```bash
# Basic start
git bisect start <bad-commit> <good-commit>

# With pathspecs (if scope known)
git bisect start <bad-commit> <good-commit> -- src/specific/

# With first-parent (if merge-heavy repo)
git bisect start --first-parent <bad-commit> <good-commit>
```

### ☐ Run Automated Bisect

```bash
git bisect run ./test_script.sh
```

**What happens**:
- Git checks out commits in binary search order
- Runs your test script
- Interprets exit codes
- Continues until culprit found

**Monitor progress**: In another terminal:
```bash
watch -n 5 'git bisect log | tail -20'
```

### ☐ Handle Interruptions (if needed)

**If bisect stops for any reason**:
```bash
# Check current state
git bisect log

# Resume if needed
git bisect run ./test_script.sh
```

**If you need to pause**:
```bash
# Save state
git bisect log > bisect-session-$(date +%Y%m%d).txt

# Later, resume
git bisect replay bisect-session-*.txt
git bisect run ./test_script.sh
```

---

## Phase 4: Analyze Results

### ☐ Review Culprit Commit

```bash
# Git shows: "<hash> is the first bad commit"
CULPRIT=<hash-from-output>

# View full commit
git show $CULPRIT

# View in context
git log --oneline --graph $CULPRIT~5..$CULPRIT^+5
```

### ☐ Verify the Result

**Test 1: Revert and check**
```bash
git checkout <bad-commit>
git revert $CULPRIT --no-commit
./run_test  # Should pass

git reset --hard  # Cleanup
```

**Test 2: Cherry-pick to good commit**
```bash
git checkout <good-commit>
git cherry-pick $CULPRIT
./run_test  # Should fail

git reset --hard  # Cleanup
```

**If verification fails**: Bisect may have found a symptom, not root cause. Consider wider search.

### ☐ Understand the Bug

**Questions to answer**:
- What changed in this commit?
- Why did this change introduce the bug?
- Are there related changes nearby in history?
- Who authored it? (Context from commit message)

```bash
# Check commit details
git show --stat $CULPRIT
git show --format=fuller $CULPRIT

# Find related commits
git log --author="$(git show -s --format=%an $CULPRIT)" \
  --since="$(git show -s --format=%ai $CULPRIT | cut -d' ' -f1)" \
  --until="1 week later" --oneline

# Check what else changed same files
git log --oneline -- $(git show --name-only --format= $CULPRIT)
```

---

## Phase 5: Cleanup and Document

### ☐ End Bisect Session

```bash
git bisect reset
```

**This returns you to the commit you were on before bisecting.**

### ☐ Save Bisect Log

```bash
# Before resetting, optionally save
git bisect log > doc/bisect-investigation-$(date +%Y%m%d).txt
```

### ☐ Create Regression Test

**Convert your bisect test to permanent regression test**:

```bash
# 1. Copy test script
cp test_script.sh tests/regression_test_issue_<number>.sh

# 2. Adapt for CI (remove bisect-specific code)
# 3. Add to test suite
# 4. Ensure it runs in CI
```

**Why**: Prevent this bug from returning.

### ☐ Document Findings

**Create issue or commit message documenting**:
- What the bug was
- When it was introduced (commit hash)
- Why it was introduced (intent vs. effect)
- How to fix it
- Bisect log (optional, for complex cases)

**Example commit message**:
```
Fix regression introduced in abc1234

Bug introduced in commit abc1234 (2024-10-15) when refactoring
the authentication module. The change inadvertently removed a
null check, causing crashes when user session expires.

Found via git bisect in 12 automated tests over 15 minutes.

Fixes: #123
```

---

## Troubleshooting During Workflow

### Problem: Test script exits with wrong code

**Solution**: Validate script (Phase 2, step 3)

### Problem: Bisect takes too long

**Options**:
- Narrow scope with pathspecs
- Optimize test (use specific test, not full suite)
- Use parallel bisecting (git-pisect)

### Problem: Bisect identifies wrong commit

**Causes**:
- Flaky test (use retry logic)
- Build failures not handled (use exit 125)
- External dependencies (fix test to be isolated)

### Problem: Many commits won't build

**Solution**: Ensure test script uses `exit 125` for build failures

### Problem: Interrupted mid-bisect

**Solution**: `git bisect log > file.txt`, then `git bisect replay file.txt` later

---

## Quick Reference Commands

```bash
# Start bisect
git bisect start <bad> <good>

# Run automated bisect
git bisect run ./test_script.sh

# Check progress
git bisect log
git bisect visualize

# Save state
git bisect log > session.txt

# Resume
git bisect replay session.txt

# End bisect
git bisect reset
```

---

## Success Criteria

✅ **Bisect successful when**:
- Culprit commit identified
- Verification confirms it's the right commit
- You understand why the commit introduced the bug
- Regression test created to prevent recurrence
- Findings documented

✅ **Additional success indicators**:
- Took < 1 hour
- Automated (not manual)
- Reproducible (saved bisect log)
- Team can learn from your process

---

## Next Steps After Finding Bug

1. **Immediate**: Fix the bug
2. **Short-term**: Create regression test
3. **Long-term**: Consider related issues
   - Are other parts of code affected?
   - Does this indicate a pattern?
   - Should we refactor this area?

---

**See also**:
- `../resources/fundamentals.md` - Understanding bisect
- `../resources/automated_bisecting.md` - Writing better test scripts
- `../resources/edge_cases.md` - Handling problems
- `../resources/advanced_techniques.md` - Optimization
