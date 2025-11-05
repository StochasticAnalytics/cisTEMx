# Automated Bisecting with git bisect run

Automated bisecting lets Git handle the binary search while you provide a test script. This is the **recommended approach** for all bisect operations.

## Exit Code Conventions

Your test script **must** exit with specific codes to tell Git how to proceed:

| Exit Code | Meaning | Action |
|-----------|---------|--------|
| **0** | Good (bug not present) | Mark commit as good, continue bisecting |
| **1-124** | Bad (bug present) | Mark commit as bad, continue bisecting |
| **125** | Untestable (can't determine) | Skip this commit, try another |
| **126-127** | Reserved | Shell execution errors |
| **128-255** | Special | Abort bisection (Git signals) |

### Why Exit Code 125?

**Specifically chosen** as the highest practical value to avoid conflicts:
- Below 126 (shell reserved)
- Avoids POSIX signal range (128+)
- Unambiguous "skip" meaning

**Common uses for 125**:
- Commit won't compile
- Missing dependencies
- Broken test infrastructure
- Known bad commits (blacklisted)

## Basic Test Script Pattern

```bash
#!/bin/bash

# 1. Try to build (exit 125 if fails)
make clean && make || exit 125

# 2. Run test
# - Exit 0 if bug NOT present (good)
# - Exit non-zero if bug IS present (bad)
./run_test
```

**How it works**:
- `make || exit 125`: If make fails, skip this commit
- `./run_test`: Normal exit code determines good/bad

## Complete Examples

### Example 1: Unit Test Regression

**Problem**: Unit test passed before, fails now

```bash
#!/bin/bash
# test_regression.sh

# Build project
cmake -B build && cmake --build build || exit 125

# Run specific test
# Pass (exit 0) = bug not present (good)
# Fail (exit non-zero) = bug present (bad)
cd build && ctest -R TestName --output-on-failure
```

**Usage**:
```bash
git bisect start HEAD v1.0
git bisect run ./test_regression.sh
```

### Example 2: Performance Regression

**Problem**: Code was fast, now slow

```bash
#!/bin/bash
# test_performance.sh

# Build
make clean && make || exit 125

# Run benchmark, capture time
time_output=$(./benchmark 2>&1 | grep "Time:")
time_ms=$(echo "$time_output" | awk '{print $2}')

# Threshold: 100ms
if [ "$time_ms" -lt 100 ]; then
    exit 0  # Fast = good
else
    exit 1  # Slow = bad
fi
```

### Example 3: API Breaking Change

**Problem**: API signature changed, need to find when

```bash
#!/bin/bash
# test_api.sh

# Build
make clean && make || exit 125

# Try to compile test that uses old API
gcc -c api_test.c -o api_test.o 2>/dev/null

# If compiles, old API still exists (good)
# If fails, API changed (bad)
if [ $? -eq 0 ]; then
    exit 0  # Old API exists
else
    exit 1  # API changed
fi
```

### Example 4: Integration Test

**Problem**: End-to-end workflow broken

```bash
#!/bin/bash
# test_e2e.sh

# Build and deploy
./build.sh || exit 125
./deploy_test_env.sh || exit 125

# Run end-to-end test
./e2e_test.sh
result=$?

# Cleanup
./teardown_test_env.sh

# Exit with test result
exit $result
```

## Using git bisect run

### Basic Invocation

```bash
git bisect start <bad-commit> <good-commit>
git bisect run ./test_script.sh
```

**What happens**:
1. Git checks out middle commit
2. Runs `./test_script.sh`
3. Interprets exit code (0=good, 1=bad, 125=skip)
4. Repeats until culprit found
5. Shows result and resets

### Passing Arguments to Test Script

```bash
# Pass arguments to your script
git bisect run ./test_script.sh --verbose --timeout=30

# Use inline script
git bisect run sh -c "make && ./run_test --specific-flag"

# With environment variables
export TEST_ENV=production
git bisect run ./test_script.sh
```

### Real-World Example (Lazygit)

Jesse Duffield's actual bisect command:

```bash
git bisect run sh -c "(go build -o /dev/null || exit 125) && go test ./pkg/gui -run /$3"
```

**Breakdown**:
- `go build -o /dev/null || exit 125`: Build, skip if fails
- `&& go test ./pkg/gui -run /$3`: Run specific test
- Result: ~100 manual checks → ~7 automated tests

## Optimizing Test Scripts

### 1. Make Tests Fast

**Each test runs O(log n) times**, so speed matters:

```bash
# ❌ Slow (full test suite)
make test-all  # 10 minutes × 15 bisect steps = 2.5 hours

# ✅ Fast (specific test)
make test-unit TestSpecificFailure  # 30 seconds × 15 = 7.5 minutes
```

### 2. Fail Fast

```bash
# ❌ Continue on failure
make test-unit
make test-integration  # Runs even if first fails

# ✅ Fail immediately
make test-unit || exit 1  # Stop on first failure
```

### 3. Cache Build Artifacts

```bash
# Use ccache, incremental builds
export CCACHE_DIR=/tmp/bisect-ccache
make CC="ccache gcc"
```

### 4. Parallel Testing

```bash
# If test suite supports parallelism
make test -j$(nproc)
```

## Handling Test Output

### Capture Output for Debugging

```bash
#!/bin/bash
# test_with_logging.sh

LOG_FILE="bisect_test_$(git rev-parse --short HEAD).log"

# Build
make clean && make > "$LOG_FILE" 2>&1 || exit 125

# Test and log
./run_test >> "$LOG_FILE" 2>&1
result=$?

# Show summary
if [ $result -eq 0 ]; then
    echo "✓ Commit $(git rev-parse --short HEAD): GOOD" | tee -a "$LOG_FILE"
else
    echo "✗ Commit $(git rev-parse --short HEAD): BAD" | tee -a "$LOG_FILE"
fi

exit $result
```

### Quiet Mode

```bash
# Suppress verbose output
make > /dev/null 2>&1 || exit 125
./run_test > /dev/null 2>&1
```

## Session Management

### View Progress

While bisect is running:

```bash
# In another terminal
git bisect log        # See all good/bad marks
git bisect visualize  # See remaining commits (graphical)
```

### Save Session

```bash
# Save bisect state
git bisect log > bisect_session.txt

# Later, resume
git bisect replay bisect_session.txt
```

### Manual Intervention

If bisect pauses (e.g., after marking skip):

```bash
# Check current state
git bisect log

# Continue manually if needed
git bisect bad
# or
git bisect good
# or
git bisect skip

# Or resume automation
git bisect run ./test_script.sh
```

## Converting Bisect Tests to Regression Tests

Once you find the bug, **preserve the bisect test** as a permanent regression test:

```bash
# 1. Copy bisect script
cp bisect_test.sh tests/regression_test_issue_123.sh

# 2. Adapt for CI (remove git bisect specific code)
# 3. Add to test suite
# 4. Ensure it runs in CI

# Now future regressions are caught immediately
```

This is a **best practice**: Every bisect represents a bug that should never return.

## Best Practices Summary

✅ **Do**:
- Use `git bisect run` for automation (don't manual bisect)
- Make tests fast (run O(log n) times)
- Always use exit 125 for untestable commits
- Capture logs for debugging
- Convert bisect tests to regression tests
- Test your test script on both good and bad commits first

❌ **Don't**:
- Use slow tests (makes bisect impractical)
- Forget to handle build failures (causes false results)
- Assume tests are deterministic (check for flakiness)
- Leave bisect running indefinitely (set timeouts)
- Skip validation of results (always verify culprit)

## Troubleshooting

**"Test script exits incorrectly"**:
```bash
# Test your script first
git checkout <known-good>
./test_script.sh
echo $?  # Should be 0

git checkout <known-bad>
./test_script.sh
echo $?  # Should be non-zero (not 125)
```

**"Bisect gives wrong result"**:
- Check test determinism (run multiple times)
- Verify exit codes (0, 1-124, 125)
- Check for external dependencies (network, time, random)

**"Bisect takes forever"**:
- Profile test script speed
- Use more specific tests (not full suite)
- Consider parallel bisecting (see `advanced_techniques.md`)

## Next Steps

- **Hit edge cases?** See `edge_cases.md` for build failures, flaky tests
- **Need performance?** See `advanced_techniques.md` for optimization
- **Ready to bisect?** See `../templates/workflow_checklist.md` for step-by-step
