# Code Churn Analysis

## Purpose

This resource provides detailed techniques for measuring code churn (change frequency) using git history to identify volatile code areas.

## When You Need This

- Identifying files that change most frequently
- Understanding code volatility patterns
- Building foundation for hotspot analysis
- Investigating instability in specific modules

---

## What is Code Churn?

**Definition**: The frequency of changes made to a specific area of the codebase over time.

**Two flavors**:
1. **Change frequency**: Number of commits touching a file
2. **Line churn**: Total lines added + deleted

**Key finding**: Pre-release code churn is a strong predictor of post-release defect density.

---

## Basic Churn: Most-Changed Files

### Top 50 Files (Past Year)

```bash
git log --format=format: --name-only --since=12.month \
  | egrep -v '^$' \
  | sort \
  | uniq -c \
  | sort -nr \
  | head -50
```

**Output**:
```
    87 src/core/processor.cpp
    64 src/ui/main_panel.cpp
    52 src/database/query.cpp
    ...
```

**Interpretation**: Number of commits touching each file, sorted descending.

### Variations

**Exclude file types**:
```bash
git log --format=format: --name-only --since=12.month \
  | egrep -v '^$' \
  | egrep -v '\.(json|md|txt)$' \
  | sort \
  | uniq -c \
  | sort -nr \
  | head -50
```

**Filter by directory**:
```bash
git log --format=format: --name-only --since=12.month -- src/core/ \
  | egrep -v '^$' \
  | sort \
  | uniq -c \
  | sort -nr \
  | head -20
```

**Specific date range**:
```bash
git log --format=format: --name-only \
  --after=2024-01-01 --before=2024-06-30 \
  | egrep -v '^$' \
  | sort \
  | uniq -c \
  | sort -nr
```

---

## Detailed Churn: Lines Added/Deleted

### Per-Commit Line Statistics

```bash
git log --pretty=format:'[%h] %aN · %s' --date=short --numstat --after=2024-01-01
```

**Output**:
```
[a3b2c1] John Doe · Refactor authentication
45      23      src/auth/login.cpp
12      8       src/auth/session.cpp

[b4c3d2] Jane Smith · Add caching layer
156     0       src/cache/manager.cpp
```

**Fields**:
- Lines added
- Lines deleted
- File path

### Absolute Churn Over Time

```bash
git log --all --numstat --date=short --pretty=format:'%ad' --no-renames \
  | awk '
    /^[0-9]/ {date=$1}
    /^[0-9]+\t[0-9]+\t/ {
      added+=$1
      deleted+=$2
      print date, added, deleted
    }
  '
```

**Output**: Cumulative lines added/deleted by date.

**Use case**: Plot in spreadsheet to visualize churn trends over time.

---

## Time-Based Analysis

### Churn by Month

```bash
git log --format=format:%ad --date=format:%Y-%m --since=12.month \
  | sort \
  | uniq -c
```

**Output**:
```
    134 2024-01
    156 2024-02
    142 2024-03
```

**Interpretation**: Commit volume by month. Spikes indicate intensive development periods.

### Churn by Day of Week

```bash
git log --format=format:%ad --date=format:%A --since=6.month \
  | sort \
  | uniq -c
```

**Output**: Identifies development patterns (e.g., Friday deploys).

### Activity Heatmap

```bash
git log --format=format:%ad --date=format:%Y-%m-%d --since=6.month \
  | sort \
  | uniq -c \
  | awk '{print $2","$1}' \
  > activity.csv
```

Import into Google Sheets/Excel to visualize daily activity.

---

## Interpreting Churn Data

### Healthy Patterns

**Feature development burst**:
```
Commits: ████████░░░░░░
         ↑
         Initial development, then stable
```

**Occasional maintenance**:
```
Commits: █░░░░█░░░░█░░░
         ↑    ↑    ↑
         Bug fixes, updates
```

### Unhealthy Patterns

**Never stabilizes**:
```
Commits: ████████████████
         Sustained high frequency
```

**Reasons**:
- Repeated bug fixes → quality issue
- Incremental feature additions → missing abstraction
- Configuration tweaks → poor separation of concerns
- Multiple developers → communication issues

**Oscillating complexity** (code grows then shrinks repeatedly):
```
LOC: ╱╲╱╲╱╲╱╲
     Code added, removed, added again
```

**Indicates**: Unclear requirements, design uncertainty, or refactoring without direction.

**Clustered changes** (many files change together):
```
10 files change in commit A
12 files change in commit B
15 files change in commit C
```

**Indicates**: Architectural coupling, missing abstractions, or ripple effects from changes.

---

## Thresholds and Red Flags

### Change Frequency Thresholds

| Commits/Year | Interpretation | Action |
|--------------|----------------|--------|
| <5 | Stable | OK, leave alone |
| 5-20 | Moderate activity | Normal development |
| 20-50 | High activity | Investigate patterns |
| >50 | Very high activity | Red flag - investigate why |

**Context matters**: Configuration files may legitimately have high churn. Core algorithms should stabilize.

### Investigation Questions

When you find high churn (>20 commits in 3 months):

1. **Why so many changes?**
   - Review commit messages
   - Identify patterns (fixes vs. features)

2. **Who's changing it?**
   - Single developer (feature work) vs. multiple (communication issues)
   ```bash
   git shortlog -sn -- path/to/file.cpp
   ```

3. **What's actually changing?**
   - Same lines repeatedly (bug fixes)
   - Different areas (feature additions)
   ```bash
   git log -p --follow -- path/to/file.cpp
   ```

4. **When did it start?**
   - Recent (active development) vs. historical (persistent issues)
   ```bash
   git log --oneline --since=6.month -- path/to/file.cpp
   ```

---

## Filtering Noise

### Exclude Non-Code Files

**Problem**: Documentation, config, and test files inflate churn metrics.

**Solution**:
```bash
git log --format=format: --name-only --since=12.month \
  | egrep -v '^$' \
  | egrep -v '\.(md|txt|json|xml|yml)$' \
  | egrep -v '^test/' \
  | egrep -v '^docs/' \
  | sort \
  | uniq -c \
  | sort -nr
```

### Exclude Refactoring Commits

**Problem**: Large refactorings skew metrics.

**Solution**: Filter by commit message pattern.

```bash
git log --format=format: --name-only --since=12.month --grep="refactor" --invert-grep \
  | egrep -v '^$' \
  | sort \
  | uniq -c \
  | sort -nr
```

### Exclude Bulk Changes

**Problem**: Formatting changes, mass renames, or automated updates.

**Solution**: Filter by numstat threshold.

```bash
git log --numstat --format=format: --since=12.month \
  | awk '/^[0-9]/ {if ($1 + $2 < 500) print $3}' \
  | sort \
  | uniq -c \
  | sort -nr
```

Only counts commits with <500 lines changed.

---

## File-Specific Churn Analysis

### Deep Dive on One File

```bash
FILE="src/core/processor.cpp"

# Commit count
git log --oneline --follow -- "$FILE" | wc -l

# Detailed history
git log --oneline --follow -- "$FILE"

# Contributors
git shortlog -sn --follow -- "$FILE"

# Line churn
git log --numstat --follow -- "$FILE" \
  | awk '/^[0-9]/ {added+=$1; deleted+=$2} END {print "Added:", added, "Deleted:", deleted}'

# Recent changes (last 20 commits)
git log -p -20 --follow -- "$FILE"
```

### Compare Two Files

```bash
FILE_A="src/core/module_a.cpp"
FILE_B="src/ui/panel_b.cpp"

echo "=== Churn for $FILE_A ==="
git log --oneline --since=12.month -- "$FILE_A" | wc -l

echo "=== Churn for $FILE_B ==="
git log --oneline --since=12.month -- "$FILE_B" | wc -l
```

---

## Directory-Level Churn

### Component/Module Analysis

```bash
for dir in src/core src/ui src/database; do
  count=$(git log --format=format: --name-only --since=12.month -- "$dir" \
    | egrep -v '^$' \
    | wc -l)
  echo "$dir: $count changes"
done
```

**Output**:
```
src/core: 342 changes
src/ui: 567 changes
src/database: 123 changes
```

**Interpretation**: Identifies which components have highest maintenance burden.

---

## Author-Based Churn

### Churn by Developer

```bash
git log --author="John Doe" --name-only --pretty=format: --since=6.month \
  | egrep -v '^$' \
  | sort \
  | uniq -c \
  | sort -nr
```

**Use case**: Understand individual focus areas or coordination patterns.

### Developer Impact (Lines Changed)

```bash
git log --author="John Doe" --numstat --pretty=format: --since=6.month \
  | awk '{added+=$1; deleted+=$2} END {print "Added:", added, "Deleted:", deleted}'
```

---

## Saving and Processing Results

### Export to CSV

```bash
git log --format=format: --name-only --since=12.month \
  | egrep -v '^$' \
  | sort \
  | uniq -c \
  | sort -nr \
  | awk '{print $2","$1}' \
  > churn_data.csv
```

**CSV format**: `file_path,commit_count`

### Import to Spreadsheet

1. Open Google Sheets or Excel
2. Import `churn_data.csv`
3. Sort by commit count (descending)
4. Filter top 50 files
5. Add complexity data (from Lizard/Radon)
6. Create scatter plot for hotspot visualization

---

## Common Issues

### Issue: Too Many Results

**Problem**: Thousands of files listed.

**Solution**: Focus on top 20-50 files. The tail doesn't matter.

```bash
git log --format=format: --name-only --since=12.month \
  | egrep -v '^$' \
  | sort \
  | uniq -c \
  | sort -nr \
  | head -20
```

### Issue: Renamed Files Show Separately

**Problem**: Git tracks renames as separate files.

**Solution**: Use `--follow` for individual file analysis. For bulk analysis, accept limitation or use Code Maat.

### Issue: Large Commits Dominate

**Problem**: One massive commit (e.g., code generation) skews results.

**Solution**: Filter by commit size or review outliers manually.

---

## Advanced: Churn Trends Over Time

### Monthly Churn Snapshot

Track how churn changes over time:

```bash
for month in {1..12}; do
  date="2024-$(printf %02d $month)-01"
  next_date="2024-$(printf %02d $((month+1)))-01"

  count=$(git log --format=format: --name-only \
    --after="$date" --before="$next_date" \
    | egrep -v '^$' \
    | wc -l)

  echo "$date,$count"
done > churn_trend.csv
```

**Plot**: Visualize monthly churn to identify patterns (increasing, decreasing, stable).

---

## Related Resources

- **`fundamentals.md`**: Understand why churn matters
- **`hotspot_analysis.md`**: Combine churn with complexity
- **`temporal_coupling.md`**: Find files that change together
- **`code_maat_guide.md`**: Advanced churn analysis with Code Maat
- **`practical_workflow.md`**: Complete step-by-step workflow
