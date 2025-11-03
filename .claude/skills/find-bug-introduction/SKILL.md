---
name: find-bug-introduction
description: Find which commit introduced a bug using git bisect binary search. Use when you have a reproducible bug that worked in the past. Automates testing of commits to identify culprit in O(log n) time. Handles build failures, flaky tests, and complex repositories.
---

# Find Bug Introduction

Use this skill when you need to find **which commit introduced a bug** through automated binary search.

## When to Use

✅ **Use bisect when:**
- Bug is reproducible with automated test
- You know a good commit (bug didn't exist)
- You know a bad commit (bug exists)
- Testing is faster than manual investigation

❌ **Don't use bisect when:**
- Bug isn't reproducible
- No automated test possible
- Only 2-3 commits to check (just test them)
- Already know the culprit

## Quick Start

```bash
# 1. Start bisect
git bisect start <bad-commit> <good-commit>

# 2. Create test script that exits:
#    - 0 if bug NOT present (good)
#    - 1-124 if bug IS present (bad)
#    - 125 if commit can't be tested (skip)

# 3. Run automated bisect
git bisect run ./test_script.sh

# 4. Git identifies culprit, then cleanup
git bisect reset
```

**See** `templates/workflow_checklist.md` for step-by-step process.

## Efficiency

Git bisect uses binary search: **O(log₂ n)**

- 1,024 commits → 10 tests
- 10,000 commits → 14 tests
- 20,000 commits → 15 tests

**Real impact**: 88.6% reduction in bug resolution time (142.6 hours → 16.2 hours, production data)

## Common Scenarios

### Scenario 1: Simple Regression

```bash
# Bug exists now, didn't exist at v1.0
git bisect start HEAD v1.0
git bisect run make test
```

### Scenario 2: Build Failures

Use exit 125 to skip untestable commits:

```bash
#!/bin/bash
make || exit 125  # Skip if won't build
./run_tests       # Exit 0 (good) or non-zero (bad)
```

**See** `scripts/bisect_template.sh` for template.

### Scenario 3: Flaky Tests

**See** `scripts/bisect_with_retry.sh` for retry logic template.

## Progressive Resources

Start here, go deeper as needed:

1. **`resources/fundamentals.md`** - Understand what bisect is, when to use
2. **`resources/automated_bisecting.md`** - Exit codes, writing test scripts
3. **`resources/edge_cases.md`** - Handle builds failures, flaky tests, submodules
4. **`resources/advanced_techniques.md`** - Performance options, pathspecs, parallel bisecting

## Key Best Practices

✅ **Do:**
- Write fast tests (run O(log n) times)
- Use exit 125 for untestable commits
- Automate with `git bisect run`
- Validate results by examining culprit commit

❌ **Don't:**
- Manually mark commits unless automation impossible
- Use slow tests (makes bisect impractical)
- Forget to `git bisect reset` when done
- Trust flaky tests (use retry logic)

## Troubleshooting

**"My test is flaky"**: See `resources/edge_cases.md` § Flaky Tests

**"Commits won't build"**: See `resources/edge_cases.md` § Build Failures

**"Need to skip multiple commits"**: See `resources/advanced_techniques.md` § Session Management

**"Bisecting takes too long"**: See `resources/advanced_techniques.md` § Performance

## Citations

All sources documented in `resources/citations.md`.
