# Advanced Git Bisect Techniques

Once you're comfortable with basic bisect, these techniques optimize for specific scenarios.

## Pathspec Filtering

**When to use**: You know the bug is in specific files/directories.

**Benefit**: Drastically reduces commits to test.

```bash
# Only bisect commits that modified src/core/
git bisect start HEAD v1.0 -- src/core/

# Multiple paths
git bisect start HEAD v1.0 -- src/auth/ src/crypto/

# File patterns
git bisect start HEAD v1.0 -- "*.cpp" "*.h"

# Exclude paths
git bisect start HEAD v1.0 -- . ":(exclude)test/" ":(exclude)vendor/"
```

**Trade-off**: May miss commits that affected behavior without touching those files (e.g., build system changes).

## Custom Terminology

Replace "good/bad" with contextually appropriate terms:

### Performance Regression

```bash
git bisect start --term-old fast --term-new slow

git bisect slow HEAD
git bisect fast v1.0

git bisect run ./benchmark.sh
```

### Feature Changes

```bash
git bisect start --term-old broken --term-new fixed

git bisect fixed HEAD    # Feature works now
git bisect broken v1.0   # Feature was broken
```

### API Changes

```bash
git bisect start --term-old old-api --term-new new-api

# Find when API changed
git bisect new-api HEAD
git bisect old-api v1.0
```

**Benefit**: Makes bisect log more readable, clarifies intent.

## Session Management

### Save and Replay

```bash
# Save bisect state
git bisect log > bisect-session-2025-11-03.txt

# Later, resume from exactly where you left off
git bisect replay bisect-session-2025-11-03.txt
git bisect run ./test_script.sh
```

**Use cases**:
- Long-running bisect interrupted
- Share bisect state with team
- Document investigation process

### Visualize Progress

```bash
# See remaining commits graphically
git bisect visualize

# Or using gitk
git bisect visualize --oneline --graph

# Text-based visualization
git log --graph --oneline $(git bisect bad)...$(git bisect good)
```

### Manual Override

```bash
# Check current bisect state
git bisect log

# Manually mark current commit if needed
git bisect good
# or
git bisect bad
# or
git bisect skip

# Resume automation
git bisect run ./test_script.sh
```

## Parallel Bisecting with git-pisect

**Problem**: Bisecting takes too long even with fast tests.

**Solution**: Test multiple commits in parallel.

### Installation

```bash
# From: https://hoelz.ro/blog/git-pisect
git clone https://github.com/hoelzro/git-pisect.git
cd git-pisect
cpanm --installdeps .
perl Makefile.PL
make
make install
```

### Usage

```bash
# Like git bisect, but parallel
git pisect start HEAD v1.0
git pisect run -j 8 ./test_script.sh  # 8 parallel jobs
```

### Performance

**Example** (1,000-commit repo, 10s tests):

| Jobs | Time | Speedup |
|------|------|---------|
| 1 (sequential) | 1m42s | 1.0x |
| 4 | 45s | 2.3x |
| 8 | 38s | 2.7x |

**Trade-off**: ~2.5x more tests executed, but in parallel.

**When to use**:
- Many CPUs available
- Test is CPU-bound (not I/O-bound)
- Each test is isolated (no shared state)

## First-Parent Bisecting

**Problem**: Feature branches have broken commits, but merges to main are stable.

**Solution**: Only bisect along the main branch.

```bash
# Git 2.29+
git bisect start --first-parent HEAD v1.0
git bisect run ./test_script.sh
```

**Effect**: Skips commits in feature branches, only tests:
- Direct commits to main
- Merge commits

**Use cases**:
- Gitflow or GitHub Flow workflows
- Only main branch guaranteed stable
- Feature branches have WIP commits

**Example**:

```
main:    A---B---C---M1---D---M2---E  (bisect these)
                   /           /
feature1:     F1--F2        /        (skip these)
                           /
feature2:              G1--G2        (skip these)
```

## Multiple Good Commits

**When to use**: Uncertain about exact good commit, have several candidates.

```bash
# Provide multiple known-good commits
git bisect start HEAD
git bisect bad  # Current is bad
git bisect good commit1
git bisect good commit2
git bisect good commit3

# Git finds optimal search space
git bisect run ./test_script.sh
```

**Benefit**: Narrows search space more quickly.

## Skip Ranges

**Problem**: You know a range of commits is untestable.

```bash
# Skip all commits between v1.5 and v2.0
git bisect skip v1.5..v2.0

# Skip multiple ranges
git bisect skip commit1..commit2
git bisect skip commit3..commit4
```

**Alternative**: Use blacklist file (see `edge_cases.md`).

## No-Checkout Bisecting

**Problem**: Need to test without disrupting working directory.

**Solution**: Use `--no-checkout` with BISECT_HEAD reference.

```bash
git bisect start --no-checkout HEAD v1.0

# Test script uses BISECT_HEAD
cat > test.sh << 'EOF'
#!/bin/bash
commit=$(git rev-parse BISECT_HEAD)

# Extract file at commit
git show $commit:path/to/file.cpp > /tmp/test_file.cpp

# Compile and test without checkout
gcc /tmp/test_file.cpp && ./a.out
EOF

git bisect run ./test.sh
```

**Use cases**:
- Testing doesn't require full checkout
- Want to preserve working directory state
- Using worktrees instead

## Bisect with Submodule Pinning

**Problem**: Need specific submodule versions during bisect.

**Solution**: Pin submodule to known-good version.

```bash
#!/bin/bash
# bisect_pinned_submodule.sh

# Always use v1.0 of submodule (known good)
git submodule update --init external/lib || exit 125
cd external/lib
git checkout v1.0
cd ../..

# Now build and test with pinned submodule
make clean && make || exit 125
./run_test
```

**When to use**: Isolate bug to main repo vs. submodule.

## Bisect Across Repositories

**Problem**: Bug involves multiple repositories.

**Solution**: Coordinate bisect in both repos.

```bash
#!/bin/bash
# bisect_multi_repo.sh

# Bisect main repo as usual
# But test with specific version of dependency

DEPENDENCY_COMMIT="abc1234"  # Known good dependency

# Clone dependency at specific commit
rm -rf /tmp/dep
git clone https://example.com/dependency.git /tmp/dep
cd /tmp/dep
git checkout $DEPENDENCY_COMMIT
make && make install
cd -

# Now test main repo
make clean && make || exit 125
./run_test
```

## Post-Bisect Analysis

### Examine Culprit Commit

```bash
# After bisect identifies culprit
git bisect log | grep "first bad commit"

# Show full commit details
git show <commit-hash>

# See commit in context
git log --oneline --graph <commit>~5..<commit>^+5

# Check if it's a merge commit
git show --pretty=fuller <commit>

# See what else changed nearby
git log --oneline --since="<commit-date>" --until="1 day later"
```

### Verify the Culprit

**Test 1: Revert**
```bash
git checkout HEAD
git revert <culprit-commit> --no-commit
./run_test  # Should pass if truly the culprit
git reset --hard  # Cleanup
```

**Test 2: Cherry-pick to good commit**
```bash
git checkout <known-good-commit>
git cherry-pick <culprit-commit>
./run_test  # Should fail if truly the culprit
git reset --hard  # Cleanup
```

### Find Related Commits

```bash
# Commits by same author around same time
git log --author="<author>" \
  --since="<commit-date>" \
  --until="1 week later" \
  --oneline

# Commits touching same files
git log --oneline -- <files-changed-by-culprit>

# Commits with similar message
git log --grep="<keyword>" --oneline
```

## Integration with CI/CD

### Automated Bisect on CI Failure

```yaml
# .github/workflows/bisect-on-failure.yml
name: Auto-Bisect on Failure

on:
  workflow_dispatch:
    inputs:
      last_good_commit:
        description: 'Last known good commit'
        required: true

jobs:
  bisect:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
        with:
          fetch-depth: 0  # Need full history

      - name: Run bisect
        run: |
          git bisect start HEAD ${{ github.event.inputs.last_good_commit }}
          git bisect run make test

      - name: Report culprit
        if: always()
        run: |
          git bisect log | grep "first bad commit" | \
            gh issue comment ${{ github.event.issue.number }} \
            --body "Bisect identified: $(cat)"
```

### Nightly Bisect for Regressions

```bash
#!/bin/bash
# nightly-bisect.sh

# If today's build fails but yesterday's passed
if ! make test; then
    YESTERDAY=$(git rev-parse HEAD~1)
    git bisect start HEAD $YESTERDAY
    git bisect run make test

    # Email results
    git bisect log | mail -s "Regression found" team@example.com
    git bisect reset
fi
```

## Performance Optimization

### Use ccache for Faster Builds

```bash
export CCACHE_DIR=/tmp/bisect-ccache
export CC="ccache gcc"
export CXX="ccache g++"

git bisect run make test
```

**Benefit**: Incremental builds during bisect can be 10x faster.

### Parallel Make

```bash
#!/bin/bash
# Use all CPUs for building
make -j$(nproc) || exit 125
./run_test
```

### Skip Expensive Operations

```bash
#!/bin/bash
# Skip documentation generation during bisect
make NO_DOCS=1 || exit 125
./run_test
```

## Best Practices Summary

✅ **Do**:
- Use pathspecs when scope is known
- Use --first-parent for merge-heavy repos
- Save bisect logs for documentation
- Visualize progress for long bisects
- Verify culprit with revert/cherry-pick
- Convert bisect tests to regression tests
- Use parallel bisecting for CPU-bound tests

✅ **Advanced**:
- Custom terminology for clarity
- No-checkout for non-intrusive testing
- Coordinate multi-repo bisects
- Integrate with CI/CD for automation
- Use ccache/parallel builds for performance

❌ **Avoid**:
- Bisecting with shallow clones (need full history)
- Manual bisecting when automation possible
- Trusting bisect results without verification
- Forgetting to `git bisect reset` when done
- Using pathspecs that miss indirect causes

## Troubleshooting

**"Bisect identified wrong commit"**:
- Verify test determinism (run 100x on same commit)
- Check for external dependencies (time, network, random)
- Ensure proper exit codes (0, 1-124, 125)

**"Bisect takes too long"**:
- Profile test script execution time
- Use pathspecs to narrow scope
- Consider parallel bisecting with git-pisect
- Optimize build process (ccache, parallel make)

**"Need to bisect multiple bugs"**:
- Bisect them separately (one at a time)
- Or: Create test that checks for all bugs (harder to interpret)

## Next Steps

- **Review fundamentals?** See `fundamentals.md`
- **Need edge case handling?** See `edge_cases.md`
- **Ready to bisect?** See `../templates/workflow_checklist.md`
- **Want script templates?** See `../scripts/`
