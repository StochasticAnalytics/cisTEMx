# Handling Edge Cases in Git Bisect

Real-world bisecting often encounters complications. This guide covers how to handle them systematically.

## Build Failures

### Problem

Not every commit in history builds successfully:
- Work-in-progress commits
- Broken CI states
- Missing dependencies
- Platform-specific issues

### Solution: Exit Code 125

Always use **exit 125** to skip untestable commits:

```bash
#!/bin/bash
# Skip commits that won't build

make clean && make || exit 125  # If build fails, skip
./run_test                       # Normal test
```

### Advanced: Blacklist Known Bad Commits

If certain commits are known to be unbuildable:

Create `bisect.blacklist`:
```bash
# Commits that won't build
git bisect start
git bisect skip bef63087981a1033239f664e6772f86080bdec44
git bisect skip 72d1195b9b3919d1468d688909985b8f398c7e70
```

Then use in bisect:
```bash
git bisect replay bisect.blacklist
git bisect run ./test_script.sh
```

### Skip Algorithm

**How Git handles skips**: Uses PRNG with 1.5 power bias to avoid testing nearby commits.

**Why**: If commit N doesn't build, commits N±1 probably also don't build. Bias avoids clusters of untestable commits.

---

## Flaky Tests

### Problem

Tests pass/fail randomly:
- Race conditions
- Timing-dependent behavior
- External dependencies (network, filesystem)
- Uninitialized memory

**Critical**: Flaky tests produce incorrect bisect results.

### Solution 1: Retry Logic (Majority Voting)

Run test multiple times, require consistent failures:

```bash
#!/bin/bash
# bisect_with_retry.sh

# Build
make clean && make || exit 125

# Run test 10 times
failures=0
for i in {1..10}; do
    ./run_test || ((failures++))
done

# Require majority (>5) failures to mark as bad
if [ $failures -gt 5 ]; then
    echo "Test failed $failures/10 times - marking BAD"
    exit 1  # Bad
else
    echo "Test passed $((10 - failures))/10 times - marking GOOD"
    exit 0  # Good
fi
```

**Adjust threshold** based on flakiness:
- Very stable: 2/3 runs
- Moderate flakiness: 6/10 runs
- Very flaky: 8/10 runs (or fix the test!)

### Solution 2: Fix the Test

**Better approach**: Make the test deterministic.

Common fixes:
```bash
# 1. Seed random number generators
export RANDOM_SEED=12345

# 2. Disable timing-sensitive checks
export DISABLE_TIMEOUT_TESTS=1

# 3. Increase timeouts
export TEST_TIMEOUT=60

# 4. Mock external dependencies
export USE_MOCK_NETWORK=1
```

### Detection: Is My Test Flaky?

```bash
# Run test 100 times on same commit
for i in {1..100}; do
    ./run_test || echo "FAIL $i"
done | tee flakiness_check.txt

# Count failures
failures=$(grep -c FAIL flakiness_check.txt)
echo "Flakiness: $failures/100"
```

**Thresholds**:
- 0/100: Perfect, use directly
- 1-5/100: Minor flakiness, use retry logic
- 5-20/100: Significant flakiness, must fix or use high retry count
- >20/100: Too flaky, fix the test first

---

## Submodules

### Problem

Submodules may be at wrong version after checkout:

```bash
git bisect start HEAD v1.0
git bisect run ./test_script.sh
# Test fails because submodule not updated!
```

### Solution: Update Submodules in Test Script

```bash
#!/bin/bash
# bisect_with_submodules.sh

# Update submodules to match current commit
git submodule update --init --recursive || exit 125

# Build (may fail if submodule combination is broken)
make clean && make || exit 125

# Test
./run_test
```

### Alternative: --no-checkout with Worktrees

Avoid submodule issues by testing without checking out:

```bash
# Use worktree for bisect
git worktree add ../bisect-worktree HEAD
cd ../bisect-worktree

git bisect start HEAD v1.0 --no-checkout

# Test script works with BISECT_HEAD
cat > test.sh << 'EOF'
#!/bin/bash
commit=$(git rev-parse BISECT_HEAD)
git checkout $commit
git submodule update --init --recursive || exit 125
make && ./run_test
EOF

git bisect run ./test.sh
```

---

## Merge Commits

### Problem

Merge commits can have issues:
- Merged broken code
- Merge conflict resolution introduced bug
- Feature branch had issue before merge

### Solution: --first-parent (Git 2.29+)

**Follow only the main branch**, skip feature branch commits:

```bash
git bisect start --first-parent HEAD v1.0
git bisect run ./test_script.sh
```

**When to use**:
- Gitflow or feature branch workflow
- Only main/master should be stable
- Feature branches may have WIP commits

**Effect**: Bisects only merge commits and direct commits to main.

---

## Worktrees (Advanced)

### Problem

Bisect disrupts your current work:
- Checking out commits modifies working directory
- Can't continue development during bisect
- Risk of losing uncommitted changes

### Solution: Use Worktrees

Create separate worktree for bisecting:

```bash
# 1. Create worktree
git worktree add ../bisect-workspace HEAD

# 2. Bisect in worktree
cd ../bisect-workspace
git bisect start HEAD v1.0
git bisect run ./test_script.sh

# 3. Continue work in original directory
cd /original/path
# Your work is undisturbed!

# 4. Cleanup when done
git worktree remove ../bisect-workspace
```

---

## Long-Running Tests

### Problem

Each test takes minutes/hours:
- Integration tests
- Performance benchmarks
- End-to-end workflows

**Example**: 15 tests × 10 minutes each = 2.5 hours

### Solution 1: Narrow the Scope

Use pathspecs to reduce commits tested:

```bash
# Only bisect commits touching specific files
git bisect start HEAD v1.0 -- src/problematic/
```

**Effect**: May skip commits not touching those files, faster bisect.

### Solution 2: Faster Tests

Replace slow test with faster proxy:

```bash
# ❌ Slow: Full integration test
./run_full_integration_test  # 10 minutes

# ✅ Fast: Unit test that catches same issue
./run_specific_unit_test     # 30 seconds
```

### Solution 3: Parallel Bisecting

Use **git-pisect** for parallel testing:

```bash
# Install: https://hoelz.ro/blog/git-pisect
git pisect start HEAD v1.0
git pisect run -j 8 ./test_script.sh  # 8 parallel jobs
```

**Performance** (1,000 commits, 10s tests):
- Sequential: 1m42s
- Parallel (8 jobs): 38s (2.7x faster)

**Trade-off**: ~2.5x more tests, but executed in parallel.

**See** `advanced_techniques.md` for details.

---

## Platform-Specific Issues

### Problem

Bug only occurs on specific platform:
- Linux vs macOS vs Windows
- Specific architecture (ARM vs x86)
- Specific compiler version

### Solution: Test on Target Platform

```bash
# Option 1: Cross-compile in test script
./configure --target=arm-linux || exit 125
make && ./run_test

# Option 2: Remote testing via SSH
ssh target-machine "cd $PWD && make && ./run_test"

# Option 3: Container-based testing
docker run --rm -v $PWD:/src platform-image \
  bash -c "cd /src && make && ./run_test"
```

---

## Recovering from Mistakes

### Problem

You marked a commit incorrectly (good as bad, or vice versa).

### Solution 1: Edit Bisect Log

```bash
# 1. Save current state
git bisect log > bisect.txt

# 2. Edit bisect.txt to remove incorrect entry
vim bisect.txt

# 3. Reset and replay
git bisect reset
git bisect replay bisect.txt
```

### Solution 2: Direct Ref Manipulation

```bash
# If you know the correct good/bad commits
git bisect reset
git bisect start <correct-bad> <correct-good>
git bisect run ./test_script.sh
```

---

## Non-Compiling Commits

### Problem

Many historical commits don't compile (common in old repos).

### Solution: Maintain Skip List

Create comprehensive blacklist:

```bash
# Generate list of non-compiling commits
git rev-list HEAD..v1.0 | while read commit; do
    git checkout $commit 2>/dev/null
    make clean && make 2>/dev/null || echo "git bisect skip $commit"
done > skip_list.txt

# Use in bisect
git bisect start
git bisect bad HEAD
git bisect good v1.0
source skip_list.txt
git bisect run ./test_script.sh
```

**Alternative**: Use exit 125 in test script (simpler).

---

## Timeout Handling

### Problem

Test hangs indefinitely on certain commits.

### Solution: Timeout Wrapper

```bash
#!/bin/bash
# bisect_with_timeout.sh

# Build
make clean && make || exit 125

# Run test with timeout (GNU timeout or coreutils)
timeout 60 ./run_test
result=$?

# Exit code 124 means timeout
if [ $result -eq 124 ]; then
    echo "Test timed out - marking as BAD"
    exit 1  # Treat timeout as failure
fi

exit $result
```

---

## Binary Files / Non-Code Changes

### Problem

Bug introduced by:
- Asset changes (images, models, data files)
- Configuration changes
- Build system changes

### Solution: Test the Right Thing

```bash
#!/bin/bash
# Test output of build, not just code

# Build
make clean && make || exit 125

# Test resulting binary behavior
./binary_output > result.txt

# Compare with expected
if diff result.txt expected.txt > /dev/null; then
    exit 0  # Good
else
    exit 1  # Bad
fi
```

---

## Summary: Edge Case Decision Tree

```
Can't build commit?
  → exit 125

Test is flaky?
  → Use retry logic (majority voting)
  → OR fix the test first

Has submodules?
  → git submodule update in test script

Using feature branches?
  → git bisect start --first-parent

Tests take too long?
  → Narrow scope with pathspecs
  → OR use faster proxy test
  → OR parallel bisecting

Wrong platform?
  → Test on target (SSH, Docker, cross-compile)

Made a mistake?
  → Edit and replay bisect log

Test hangs?
  → Use timeout wrapper

Many non-compiling commits?
  → exit 125 or maintain skip list
```

---

## Best Practices

✅ **Always**:
- Use exit 125 for untestable commits
- Test your test script on known good/bad commits first
- Log bisect progress (git bisect log > file.txt)
- Verify results by examining culprit commit

✅ **For Flaky Tests**:
- Detect flakiness first (run 100 times)
- Use retry logic with appropriate threshold
- Better: Fix the test to be deterministic

✅ **For Complex Repos**:
- Update submodules in test script
- Use worktrees to avoid disrupting work
- Consider --first-parent for merge-heavy repos

❌ **Never**:
- Trust flaky tests without retry logic
- Forget to handle build failures (causes false positives/negatives)
- Skip validation (always examine the culprit commit)
- Bisect on shallow clones (need full history)

---

## Troubleshooting Reference

| Symptom | Cause | Solution |
|---------|-------|----------|
| Bisect identifies wrong commit | Flaky test | Add retry logic |
| Bisect marks good as bad | Build failed but returned exit 1 | Use exit 125 for build failures |
| Bisect never finishes | Many untestable commits | Check skip algorithm working |
| Test fails on all commits | Test setup issue | Verify test works manually |
| Bisect disrupts work | Working tree modified | Use worktrees |
| Slow bisect | Long tests | Narrow scope or parallelize |

---

## Next Steps

- **Need performance optimization?** See `advanced_techniques.md`
- **Ready to bisect?** See `../templates/workflow_checklist.md`
- **Want script templates?** See `../scripts/`
