# Git Analysis Patterns for Test Coverage Gaps

## Purpose

Collection of git commands and patterns for identifying test coverage gaps through version control history analysis.

## When You Need This

- Finding files changed without test updates
- Analyzing test-to-production change ratios
- Identifying high-churn code lacking tests
- Custom gap analysis beyond diff-cover capabilities

---

## Pattern 1: Test-to-Production Change Ratio

### Overview

Analyze whether test code is keeping pace with production code growth.

### Basic Ratio Analysis

**Production Changes (Last Month)**:
```bash
git log --stat --since="1 month ago" -- src/ | \
  grep -E "^ (.*)\|" | \
  awk '{files++; added+=$(NF-3); deleted+=$(NF-1)} \
  END {printf "Production: %d files, +%d -%d lines (net: %+d)\n", files, added, deleted, added-deleted}'
```

**Test Changes (Last Month)**:
```bash
git log --stat --since="1 month ago" -- test/ | \
  grep -E "^ (.*)\|" | \
  awk '{files++; added+=$(NF-3); deleted+=$(NF-1)} \
  END {printf "Tests: %d files, +%d -%d lines (net: %+d)\n", files, added, deleted, added-deleted}'
```

**Calculate Ratio**:
```bash
#!/bin/bash
# Calculate test:production ratio for a time period

TIME_PERIOD="${1:-1 month ago}"

prod_net=$(git log --stat --since="$TIME_PERIOD" -- src/ | \
  grep -E "^ (.*)\|" | \
  awk '{added+=$(NF-3); deleted+=$(NF-1)} END {print added-deleted}')

test_net=$(git log --stat --since="$TIME_PERIOD" -- test/ | \
  grep -E "^ (.*)\|" | \
  awk '{added+=$(NF-3); deleted+=$(NF-1)} END {print added-deleted}')

if [ "$prod_net" -gt 0 ]; then
    ratio=$(echo "scale=2; $test_net / $prod_net" | bc)
    echo "Ratio: 1:$ratio (test:production)"

    if (( $(echo "$ratio >= 1.0" | bc -l) )); then
        echo "Status: ✓ Healthy (≥1:1)"
    elif (( $(echo "$ratio >= 0.8" | bc -l) )); then
        echo "Status: ⚠ Acceptable (≥1:0.8)"
    else
        echo "Status: ✗ Warning (<1:0.8)"
    fi
else
    echo "No net production changes in period"
fi
```

### Ratio Over Time (Trend Analysis)

```bash
#!/bin/bash
# Show ratio trend over last 6 months

echo "Month       | Production | Tests  | Ratio  | Status"
echo "------------|------------|--------|--------|--------"

for i in {0..5}; do
    start_date=$(date -d "$i month ago" +%Y-%m-01)
    end_date=$(date -d "$((i-1)) month ago" +%Y-%m-01)

    prod=$(git log --stat --since="$start_date" --until="$end_date" -- src/ | \
      grep -E "^ (.*)\|" | \
      awk '{added+=$(NF-3); deleted+=$(NF-1)} END {print added-deleted}')

    test=$(git log --stat --since="$start_date" --until="$end_date" -- test/ | \
      grep -E "^ (.*)\|" | \
      awk '{added+=$(NF-3); deleted+=$(NF-1)} END {print added-deleted}')

    prod=${prod:-0}
    test=${test:-0}

    if [ "$prod" -gt 0 ]; then
        ratio=$(echo "scale=2; $test / $prod" | bc)
        status=$(echo "$ratio >= 0.8" | bc -l)
        [ "$status" -eq 1 ] && status="✓" || status="✗"
    else
        ratio="N/A"
        status="-"
    fi

    printf "%-11s | +%-9d | +%-5d | %-6s | %s\n" \
      "$(date -d "$start_date" +%Y-%m)" "$prod" "$test" "$ratio" "$status"
done
```

---

## Pattern 2: Commits Changing Production Without Tests

### Overview

Find commits that modified production code but didn't touch test files.

### Basic Detection

```bash
#!/bin/bash
# Find commits changing src/ but not test/

echo "Commits with production changes but no test changes:"
echo "================================================================"

git log --name-only --format="%H|%an|%ad|%s" --since="1 month ago" | \
awk '
  BEGIN { commit=""; has_prod=0; has_test=0; }

  /\|/ {
    # New commit - check previous
    if (commit != "" && has_prod && !has_test) {
      print commit
      print ""
    }
    # Reset for this commit
    commit = $0
    has_prod = 0
    has_test = 0
    next
  }

  /^src\// { has_prod = 1 }
  /^test\// { has_test = 1 }

  /^$/ { next }

  END {
    # Check last commit
    if (commit != "" && has_prod && !has_test) {
      print commit
    }
  }
'
```

### With File Details

```bash
#!/bin/bash
# Show which production files changed without test updates

git log --no-merges --since="1 month ago" --format="%H" | while read commit; do
    # Get changed files
    prod_files=$(git show --name-only --format= "$commit" | grep "^src/" | grep -v "test")
    test_files=$(git show --name-only --format= "$commit" | grep "^test/")

    if [ -n "$prod_files" ] && [ -z "$test_files" ]; then
        echo "Commit: $commit"
        git show --no-patch --format="Author: %an <%ae>%nDate:   %ad%nSubject: %s%n" "$commit"
        echo "Production files changed:"
        echo "$prod_files" | sed 's/^/  - /'
        echo ""
    fi
done
```

### Branch-Level Analysis

```bash
#!/bin/bash
# Check if feature branch has adequate test coverage

BRANCH="${1:-HEAD}"
BASE="${2:-main}"

echo "Analyzing: $BRANCH vs $BASE"
echo ""

# Files changed in production
prod_files=$(git diff --name-only "$BASE"..."$BRANCH" | grep "^src/" | grep -v "test")
prod_count=$(echo "$prod_files" | grep -v "^$" | wc -l)

# Files changed in tests
test_files=$(git diff --name-only "$BASE"..."$BRANCH" | grep "^test/")
test_count=$(echo "$test_files" | grep -v "^$" | wc -l)

echo "Production files changed: $prod_count"
echo "Test files changed: $test_count"
echo ""

if [ "$prod_count" -gt 0 ] && [ "$test_count" -eq 0 ]; then
    echo "⚠ WARNING: Production code changed but NO test files modified"
    echo ""
    echo "Production files:"
    echo "$prod_files" | sed 's/^/  - /'
elif [ "$prod_count" -gt 0 ]; then
    ratio=$(echo "scale=2; $test_count / $prod_count" | bc)
    echo "Test:Production file ratio: $ratio"

    if (( $(echo "$ratio >= 0.5" | bc -l) )); then
        echo "Status: ✓ Reasonable test activity"
    else
        echo "Status: ⚠ Low test activity for production changes"
    fi
fi
```

---

## Pattern 3: Files Never Associated with Tests

### Overview

Find production files that have never been modified in the same commit as test files.

### Never Tested in Commit History

```bash
#!/bin/bash
# Find production files never committed alongside test changes

echo "Files never committed with test changes:"
echo "========================================"

# Get all current production files
find src/ -type f \( -name "*.cpp" -o -name "*.c" -o -name "*.py" \) | while read prod_file; do
    # Get all commits that modified this file
    commits=$(git log --all --format=%H -- "$prod_file")

    never_with_tests=true
    for commit in $commits; do
        # Check if this commit also touched test files
        if git show --name-only --format= "$commit" | grep -q "^test/"; then
            never_with_tests=false
            break
        fi
    done

    if $never_with_tests && [ -n "$commits" ]; then
        # Count commits and age
        commit_count=$(echo "$commits" | wc -l)
        first_commit=$(echo "$commits" | tail -1)
        age_days=$(( ($(date +%s) - $(git show -s --format=%ct "$first_commit")) / 86400 ))

        echo "$prod_file"
        echo "  Commits: $commit_count | Age: ${age_days} days"
    fi
done
```

### Cross-Reference with Coverage Data

```bash
#!/bin/bash
# Find files with ZERO coverage that also never had test commits

# Requires: lcov coverage report

if [ ! -f "coverage.info" ]; then
    echo "Error: coverage.info not found"
    echo "Run: lcov --capture --directory . --output-file coverage.info"
    exit 1
fi

echo "Files with zero coverage AND never tested in git history:"
echo "========================================================="

# Get zero-coverage files
lcov --list coverage.info | awk '$2 == "0.0%" {print $1}' | while read file; do
    # Check if ever committed with tests
    commits=$(git log --all --format=%H -- "$file" 2>/dev/null)

    if [ -z "$commits" ]; then
        continue  # File not in git (generated?)
    fi

    never_with_tests=true
    for commit in $commits; do
        if git show --name-only --format= "$commit" | grep -q "^test/"; then
            never_with_tests=false
            break
        fi
    done

    if $never_with_tests; then
        echo "⚠ $file"

        # Get file stats
        if [ -f "$file" ]; then
            loc=$(wc -l < "$file")
            echo "  Lines: $loc"
        fi

        # Get age
        first_commit=$(echo "$commits" | tail -1)
        age_days=$(( ($(date +%s) - $(git show -s --format=%ct "$first_commit")) / 86400 ))
        echo "  Age: ${age_days} days"
        echo ""
    fi
done
```

---

## Pattern 4: High-Churn Files (Frequent Changes)

### Overview

Identify files changed frequently, which represent higher risk if untested.

### Top Changed Files

```bash
#!/bin/bash
# List most frequently changed files

SINCE="${1:-3 months ago}"
LIMIT="${2:-20}"

echo "Top $LIMIT most changed files since $SINCE:"
echo "============================================"

git log --no-merges --since="$SINCE" --name-only --format='' | \
  sort | uniq -c | sort -r -k1 -n | head -n "$LIMIT" | \
  while read count file; do
      printf "%3d changes | %s\n" "$count" "$file"
  done
```

### High-Churn Production Files (Excluding Tests)

```bash
#!/bin/bash
# High-churn production code (potential risk areas)

SINCE="${1:-3 months ago}"

echo "High-churn production files (>= 5 changes since $SINCE):"
echo "========================================================"

git log --no-merges --since="$SINCE" --name-only --format='' | \
  grep "^src/" | grep -v "test" | \
  sort | uniq -c | sort -r -k1 -n | \
  awk '$1 >= 5 {printf "%3d changes | %s\n", $1, $2}'
```

### Churn with Lines Changed

```bash
#!/bin/bash
# Show churn with total lines added/deleted

echo "File                          | Commits | Added | Deleted | Net   "
echo "------------------------------|---------|-------|---------|-------"

git log --all --numstat --format='%H' -- src/*.cpp | \
  awk '
    /^[0-9]/ {
      file=$3
      added[file] += $1
      deleted[file] += $2
      commits[file]++
    }
    END {
      for (file in commits) {
        net = added[file] - deleted[file]
        printf "%-30s| %7d | %5d | %7d | %+6d\n",
          file, commits[file], added[file], deleted[file], net
      }
    }
  ' | sort -t'|' -k2 -rn | head -20
```

---

## Pattern 5: Files with Repeated Bug Fixes

### Overview

Files that repeatedly have bug-fix commits indicate insufficient test coverage.

### Bug-Fix Frequency

```bash
#!/bin/bash
# Find files most frequently associated with bug fixes

SINCE="${1:-6 months ago}"

echo "Files most frequently associated with bug fixes:"
echo "================================================"

# Find bug-fix commits (by keyword in message)
git log --no-merges --since="$SINCE" --format="%H|%s" | \
  grep -iE "fix|bug|issue|defect" | cut -d'|' -f1 | \
  while read commit; do
      git show --name-only --format= "$commit"
  done | \
  grep "^src/" | grep -v "test" | \
  sort | uniq -c | sort -rn | \
  head -20 | \
  while read count file; do
      printf "%3d bug fixes | %s\n" "$count" "$file"
  done
```

### Bug Fixes with Coverage Correlation

```bash
#!/bin/bash
# Files with multiple bugs AND low coverage

SINCE="${1:-6 months ago}"
COV_THRESHOLD="${2:-50}"  # Coverage % threshold

if [ ! -f "coverage.info" ]; then
    echo "Warning: coverage.info not found, skipping coverage correlation"
    exit 1
fi

echo "Files with ≥3 bug fixes and <${COV_THRESHOLD}% coverage:"
echo "========================================================"

# Get bug-fix counts
git log --no-merges --since="$SINCE" --format="%H|%s" | \
  grep -iE "fix|bug|issue|defect" | cut -d'|' -f1 | \
  while read commit; do
      git show --name-only --format= "$commit"
  done | \
  grep "^src/" | grep -v "test" | \
  sort | uniq -c | sort -rn | \
  awk '$1 >= 3 {print $2, $1}' | \
  while read file bug_count; do
      # Get coverage for this file
      cov=$(lcov --list coverage.info | grep "$file" | awk '{print $2}' | tr -d '%')

      if [ -n "$cov" ]; then
          if (( $(echo "$cov < $COV_THRESHOLD" | bc -l) )); then
              printf "%-50s | %2d bugs | %5.1f%% coverage\n" "$file" "$bug_count" "$cov"
          fi
      fi
  done
```

---

## Pattern 6: Combining Metrics (Risk Scoring)

### Overview

Combine multiple signals (churn, coverage, bugs) to identify highest-risk files.

### Risk Score Calculation

```bash
#!/bin/bash
# Calculate risk scores for production files
# Risk = (churn × 10) + (bug_fixes × 20) + ((100 - coverage) × 2)

SINCE="${1:-6 months ago}"

# Temporary files
CHURN_FILE=$(mktemp)
BUG_FILE=$(mktemp)
COV_FILE=$(mktemp)

trap "rm -f $CHURN_FILE $BUG_FILE $COV_FILE" EXIT

# Get churn data
git log --no-merges --since="$SINCE" --name-only --format='' | \
  grep "^src/" | grep -v "test" | \
  sort | uniq -c | \
  awk '{print $2, $1}' > "$CHURN_FILE"

# Get bug-fix data
git log --no-merges --since="$SINCE" --format="%H|%s" | \
  grep -iE "fix|bug|issue|defect" | cut -d'|' -f1 | \
  while read commit; do
      git show --name-only --format= "$commit"
  done | \
  grep "^src/" | grep -v "test" | \
  sort | uniq -c | \
  awk '{print $2, $1}' > "$BUG_FILE"

# Get coverage data
if [ -f "coverage.info" ]; then
    lcov --list coverage.info | \
      awk 'NR>3 && $2 ~ /%/ {print $1, $2}' | \
      tr -d '%' > "$COV_FILE"
fi

# Combine and calculate risk
echo "Risk Score | File                                      | Churn | Bugs | Coverage"
echo "-----------|-------------------------------------------|-------|------|----------"

find src/ -type f \( -name "*.cpp" -o -name "*.c" -o -name "*.py" \) | while read file; do
    churn=$(grep -F "$file" "$CHURN_FILE" | awk '{print $2}')
    bugs=$(grep -F "$file" "$BUG_FILE" | awk '{print $2}')
    coverage=$(grep -F "$file" "$COV_FILE" | awk '{print $2}')

    churn=${churn:-0}
    bugs=${bugs:-0}
    coverage=${coverage:-0}

    # Risk formula
    risk=$(echo "($churn * 10) + ($bugs * 20) + ((100 - $coverage) * 2)" | bc)

    # Only show files with significant risk
    if [ "$risk" -gt 100 ]; then
        printf "%10d | %-42s | %5d | %4d | %7.1f%%\n" \
          "$risk" "${file:0:42}" "$churn" "$bugs" "$coverage"
    fi
done | sort -rn | head -30
```

---

## Pattern 7: Test File Naming Convention Validation

### Overview

Verify that production files have corresponding test files following naming conventions.

### Basic Convention Check

```bash
#!/bin/bash
# Check for missing test files (assumes test_*.cpp or *_test.cpp convention)

echo "Production files without corresponding test files:"
echo "=================================================="

find src/ -name "*.cpp" | while read prod_file; do
    base=$(basename "$prod_file" .cpp)
    dir=$(dirname "$prod_file")

    # Look for test_base.cpp or base_test.cpp
    if [ ! -f "test/test_${base}.cpp" ] && \
       [ ! -f "test/${base}_test.cpp" ] && \
       [ ! -f "${dir}/test_${base}.cpp" ] && \
       [ ! -f "${dir}/${base}_test.cpp" ]; then

        # Double-check if file is mentioned in ANY test file
        if ! grep -r -l "$base" test/ >/dev/null 2>&1; then
            echo "$prod_file"

            # Show file age and churn
            commits=$(git log --oneline -- "$prod_file" | wc -l)
            echo "  Commits: $commits"
        fi
    fi
done
```

---

## Integration with diff-cover

### Pre-diff-cover Analysis

Run git analysis before diff-cover to identify patterns:

```bash
#!/bin/bash
# Comprehensive pre-diff-cover analysis

echo "=== Git-Based Test Coverage Gap Analysis ==="
echo ""

echo "1. Test-to-Production Ratio (Last Month)"
./scripts/test_prod_ratio.sh "1 month ago"
echo ""

echo "2. Recent Commits Without Test Changes"
./scripts/find_untested_commits.sh | head -5
echo ""

echo "3. High-Churn Files (Top 10)"
./scripts/high_churn_files.sh "3 months ago" 10
echo ""

echo "4. Now running diff-cover for detailed line-level analysis..."
diff-cover coverage.xml --compare-branch=origin/main
```

---

## Best Practices

### 1. Automate Regular Analysis

Run these queries weekly or monthly:
```bash
# Cron job: weekly test gap report
0 9 * * MON /path/to/scripts/weekly_gap_report.sh | mail -s "Test Gap Report" team@example.com
```

### 2. Focus on Trends, Not Snapshots

Track metrics over time to see if improving or degrading.

### 3. Combine with Code Review

Use these patterns during PR review:
- "This file has 10 bug fixes in 6 months. Are new tests sufficient?"
- "High churn + low coverage = high risk. Consider refactoring."

### 4. Prioritize Actionable Insights

Don't try to test everything. Focus on:
1. High risk (churn + bugs + complexity + low coverage)
2. Critical paths (auth, data integrity, financial)
3. Recently changed code (easier to test, fresh in mind)

---

## Related Resources

- **`fundamentals.md`**: Understanding why these patterns matter
- **`diff_cover_workflow.md`**: Line-level coverage analysis with diff-cover
- **`prioritization_strategies.md`**: Deciding which gaps to address first
- **`tooling_integration.md`**: Getting coverage data for correlation
