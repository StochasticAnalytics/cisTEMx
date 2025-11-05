# Practical Workflow Guide

## Purpose

This resource provides complete, step-by-step workflows for conducting refactoring target identification from start to finish, with concrete examples and troubleshooting.

## When You Need This

- First time conducting hotspot analysis
- Need complete end-to-end workflow
- Want concrete examples to follow
- Troubleshooting analysis issues

---

## Prerequisites

### Required Tools

**Git** (already have it)
```bash
git --version
```

**Lizard** (complexity analysis)
```bash
pip install lizard
lizard --version
```

**Code Maat** (optional, for advanced analysis)
```bash
wget https://github.com/adamtornhill/code-maat/releases/download/v1.0.4/code-maat-1.0.4-standalone.jar
java -jar code-maat-1.0.4-standalone.jar -v
```

**Python + Pandas** (optional, for data joining)
```bash
pip install pandas matplotlib
```

### Repository Requirements

- Git repository with 6+ months of history
- At least 100 commits (more is better)
- Production code to analyze

---

## Workflow 1: Quick Hotspot Check (15 minutes)

**Goal**: Identify top 10 hotspot files using basic git + complexity tools.

### Step 1: Find High-Churn Files (3 min)

```bash
cd /path/to/repository

# Top 20 most-changed files (past year)
git log --format=format: --name-only --since=12.month \
  | egrep -v '^$' \
  | egrep -v '\.(md|json|txt)$' \
  | egrep -v '^test/' \
  | sort \
  | uniq -c \
  | sort -nr \
  | head -20
```

**Output**:
```
    87 src/core/processor.cpp
    64 src/ui/main_panel.cpp
    52 src/database/query.cpp
    ...
```

**Save**: Copy top 10 file paths to text file.

### Step 2: Check Complexity (5 min)

```bash
# Analyze top files
lizard src/core/processor.cpp src/ui/main_panel.cpp src/database/query.cpp
```

**Output**:
```
  NLOC    CCN   token  parameter  file
   850     45    4234         12  src/core/processor.cpp
   423     28    2156          8  src/ui/main_panel.cpp
   312     15    1678          6  src/database/query.cpp
```

**Note**: Look for high CCN (>20) combined with high commit count.

### Step 3: Calculate Hotspot Scores (2 min)

Manual calculation:
```
processor.cpp:    87 commits × 45 CCN = 3,915 (HOTSPOT!)
main_panel.cpp:   64 commits × 28 CCN = 1,792
query.cpp:        52 commits × 15 CCN =   780
```

### Step 4: Investigate Top Hotspot (5 min)

```bash
FILE="src/core/processor.cpp"

# View recent changes
git log --oneline -20 -- "$FILE"

# Identify patterns in commit messages
git log --pretty=format:'%s' -- "$FILE" | sort | uniq -c | sort -nr | head -10

# Check for technical debt markers
git grep -n "TODO\|FIXME\|HACK" -- "$FILE"
```

**Questions**:
- Why so many commits? (Bug fixes? Features? Configuration?)
- Any obvious technical debt comments?
- Multiple contributors or single owner?

### Step 5: Document Findings (5 min)

Create `refactoring_targets.md`:
```markdown
# Refactoring Targets Analysis

**Date**: 2024-11-03
**Analysis Period**: Past 12 months

## Top Hotspots

1. **src/core/processor.cpp** (Score: 3,915)
   - 87 commits, CCN 45
   - Patterns: Repeated validation fixes, config updates
   - Action: Refactor validation logic, extract config

2. **src/ui/main_panel.cpp** (Score: 1,792)
   - 64 commits, CCN 28
   - Patterns: UI layout adjustments
   - Action: Simplify layout logic

3. **src/database/query.cpp** (Score: 780)
   - 52 commits, CCN 15
   - Patterns: Normal development
   - Action: Monitor, not urgent
```

---

## Workflow 2: Comprehensive Hotspot Analysis (2 hours)

**Goal**: Full data-driven hotspot identification with visualization.

### Step 1: Generate Git Log (5 min)

```bash
cd /path/to/repository

git log --all --numstat --date=short \
  --pretty=format:'--%h--%ad--%aN' \
  --no-renames \
  --after=2024-01-01 \
  -- src/ \
  > git.log

# Verify
wc -l git.log
head git.log
```

**Expected**: Thousands of lines (depends on repo size).

### Step 2: Calculate Change Frequency (10 min)

**Option A: Using Code Maat**
```bash
java -jar code-maat-1.0.4-standalone.jar \
  -l git.log \
  -c git2 \
  -a revisions \
  > revisions.csv

head -20 revisions.csv
```

**Option B: Using Git (if no Code Maat)**
```bash
git log --format=format: --name-only --since=12.month -- src/ \
  | egrep -v '^$' \
  | sort \
  | uniq -c \
  | sort -nr \
  | awk '{print $2","$1}' \
  > churn.csv

head -20 churn.csv
```

**Output** (`revisions.csv` or `churn.csv`):
```
entity,n-revs
src/core/processor.cpp,87
src/ui/main_panel.cpp,64
...
```

### Step 3: Calculate Complexity (15 min)

```bash
# For C++
lizard -l cpp --csv src/ > complexity.csv

# For Python
radon cc -a -s --json src/ > complexity.json
# Convert JSON to CSV (manual or script)

# For multiple languages
lizard --csv src/ > complexity.csv
```

**Verify**:
```bash
head -20 complexity.csv
wc -l complexity.csv
```

**Expected**: CSV with NLOC, CCN, file columns.

### Step 4: Join Churn + Complexity (20 min)

**Python script** (`join_hotspots.py`):
```python
#!/usr/bin/env python3
import pandas as pd
import sys

# Load data
churn = pd.read_csv('revisions.csv')  # or churn.csv
complexity = pd.read_csv('complexity.csv')

# Normalize paths (adjust for your project structure)
churn['file'] = churn['entity'].str.strip()
complexity['file'] = complexity['file'].str.strip()

# Join on file path
hotspots = pd.merge(
    churn,
    complexity[['file', 'NLOC', 'CCN']],
    on='file',
    how='inner'
)

# Calculate hotspot score
hotspots['hotspot_score'] = hotspots['n-revs'] * hotspots['CCN']

# Sort by score
hotspots = hotspots.sort_values('hotspot_score', ascending=False)

# Save
hotspots.to_csv('hotspots.csv', index=False)

# Display top 20
print("\n=== Top 20 Hotspots ===")
print(hotspots[['file', 'n-revs', 'CCN', 'NLOC', 'hotspot_score']].head(20).to_string(index=False))
```

**Run**:
```bash
python join_hotspots.py
```

**Output**: `hotspots.csv` with combined metrics.

### Step 5: Visualize (30 min)

**Python visualization** (`visualize_hotspots.py`):
```python
#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

# Load hotspots
hotspots = pd.read_csv('hotspots.csv')

# Create scatter plot
plt.figure(figsize=(14, 10))
scatter = plt.scatter(
    hotspots['n-revs'],
    hotspots['CCN'],
    s=hotspots['NLOC']/5,  # Bubble size = LOC
    alpha=0.5,
    c=hotspots['hotspot_score'],
    cmap='YlOrRd'
)

# Label top 10
top_10 = hotspots.head(10)
for idx, row in top_10.iterrows():
    filename = row['file'].split('/')[-1]  # Just filename
    plt.annotate(
        filename,
        (row['n-revs'], row['CCN']),
        fontsize=9,
        alpha=0.8,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3)
    )

# Quadrant lines (median split)
median_churn = hotspots['n-revs'].median()
median_ccn = hotspots['CCN'].median()
plt.axvline(median_churn, color='gray', linestyle='--', alpha=0.5, label='Median Churn')
plt.axhline(median_ccn, color='gray', linestyle='--', alpha=0.5, label='Median Complexity')

# Labels and formatting
plt.xlabel('Change Frequency (# commits)', fontsize=12)
plt.ylabel('Cyclomatic Complexity (CCN)', fontsize=12)
plt.title('Code Hotspots Analysis\n(Bubble size = Lines of Code)', fontsize=14)
plt.colorbar(scatter, label='Hotspot Score')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save
plt.savefig('hotspots.png', dpi=150)
print("Saved: hotspots.png")

# Show
plt.show()
```

**Run**:
```bash
python visualize_hotspots.py
```

**Result**: `hotspots.png` with scatter plot showing hotspots.

### Step 6: Investigate Top 10 Hotspots (30 min)

For each top hotspot file:

```bash
FILE="src/core/processor.cpp"

echo "=== Analyzing: $FILE ==="

# Commit history
git log --oneline --follow -- "$FILE" | wc -l
git log --oneline -10 --follow -- "$FILE"

# Commit message patterns
echo "--- Common commit patterns ---"
git log --pretty=format:'%s' --follow -- "$FILE" \
  | sed 's/[0-9]\+//g' \
  | sort \
  | uniq -c \
  | sort -nr \
  | head -10

# Contributors
echo "--- Contributors ---"
git shortlog -sn --follow -- "$FILE"

# Technical debt markers
echo "--- Technical debt ---"
git grep -n "TODO\|FIXME\|HACK" -- "$FILE" || echo "None found"

# Function-level complexity
echo "--- Function complexity ---"
lizard --verbose "$FILE" | head -20
```

**Document findings** for each hotspot.

### Step 7: Create Refactoring Backlog (15 min)

**Spreadsheet or markdown table**:

| Priority | File | Score | Commits | CCN | Issue | Est. Effort | ROI |
|----------|------|-------|---------|-----|-------|-------------|-----|
| P0 | processor.cpp | 3915 | 87 | 45 | Repeated validation fixes | 2 weeks | High |
| P1 | main_panel.cpp | 1792 | 64 | 28 | Complex layout logic | 1 week | Medium |
| P2 | auth_service.cpp | 1350 | 54 | 25 | Session handling | 1 week | Medium |
| P3 | query.cpp | 780 | 52 | 15 | Normal complexity | 3 days | Low |

**Prioritization formula**:
```
Priority = (Score × Business_Impact_Factor) / Effort_Weeks
```

---

## Workflow 3: Temporal Coupling Analysis (1 hour)

**Goal**: Identify architectural issues via coupling detection.

### Step 1: Generate Git Log (5 min)

```bash
git log --all --numstat --date=short \
  --pretty=format:'--%h--%ad--%aN' \
  --no-renames \
  --after=2024-01-01 \
  > git.log
```

### Step 2: Run Coupling Analysis (10 min)

```bash
java -jar code-maat-1.0.4-standalone.jar \
  -l git.log \
  -c git2 \
  -a coupling \
  --min-coupling 40 \
  --min-revs 5 \
  --temporal-period 1 \
  > coupling.csv

# View results
head -20 coupling.csv
```

### Step 3: Filter by Hotspot Files (5 min)

```bash
# Extract hotspot files from hotspots.csv
tail -n +2 hotspots.csv | cut -d',' -f1 | head -10 > hotspot_files.txt

# Find coupling for hotspot files
while read file; do
  grep "$file" coupling.csv
done < hotspot_files.txt > hotspot_coupling.csv
```

### Step 4: Identify Architectural Issues (20 min)

**Questions**:
1. Do UI files couple with database files? (Layer violation)
2. Do unrelated feature modules couple? (Missing abstraction)
3. Do test files couple with production code? (Expected, OK)

**Example finding**:
```
src/ui/settings_panel.cpp,src/database/config_db.cpp,82,15
```

**Interpretation**: UI directly coupled to database (82% degree). Architectural violation.

### Step 5: Plan Decoupling (20 min)

For each problematic coupling:

**Strategy**:
- Extract interface/service layer
- Introduce facade pattern
- Use dependency injection
- Implement event-driven architecture

**Document**:
```markdown
## Coupling Issue: UI → Database

**Files**:
- src/ui/settings_panel.cpp
- src/database/config_db.cpp

**Coupling Degree**: 82%

**Problem**: UI directly reads/writes database, violating layered architecture.

**Solution**: Introduce `ConfigService` layer:
1. Create `ConfigService` interface
2. Move database access to service
3. UI calls service, not database
4. Test with mocks

**Effort**: 1 week
**Priority**: High (architectural violation)
```

---

## Workflow 4: SATD Mining (30 minutes)

**Goal**: Find technical debt markers in hotspot files.

### Step 1: Baseline SATD Count (5 min)

```bash
git grep -n -E "TODO|FIXME|HACK|XXX" -- '*.cpp' '*.h' '*.py' \
  > satd_baseline.txt

# Count total
wc -l satd_baseline.txt

# Categorize
grep "TODO" satd_baseline.txt | wc -l
grep "FIXME" satd_baseline.txt | wc -l
grep "HACK" satd_baseline.txt | wc -l
```

### Step 2: SATD in Hotspot Files (10 min)

```bash
# Extract hotspot files
tail -n +2 hotspots.csv | cut -d',' -f1 | head -20 > hotspot_files.txt

# Find SATD in hotspots
while read file; do
  grep "$file" satd_baseline.txt
done < hotspot_files.txt > satd_in_hotspots.txt

# Count
wc -l satd_in_hotspots.txt
```

### Step 3: Prioritize SATD Resolution (15 min)

Focus on SATD in:
1. Hotspot files (high churn + complexity)
2. Files with coupling issues
3. Files blocking new features

**Output**:
```markdown
## Technical Debt in Hotspots

1. **src/core/processor.cpp** (Score: 3915)
   - 5 TODO comments (validation refactoring)
   - 2 FIXME comments (memory leak)
   - Action: Address during refactoring

2. **src/ui/main_panel.cpp** (Score: 1792)
   - 3 HACK comments (layout workarounds)
   - Action: Clean up during simplification
```

---

## Workflow 5: Tracking Progress (Ongoing)

**Goal**: Measure refactoring impact over time.

### Before Refactoring

```bash
TAG="v2.0"
FILE="src/core/processor.cpp"

# Capture metrics
git checkout $TAG

# Complexity
lizard "$FILE" > before_complexity.txt

# Churn (past 3 months before refactoring)
git log --oneline --since=3.month --before="$(git log -1 --format=%ai $TAG)" -- "$FILE" \
  | wc -l > before_churn.txt
```

### After Refactoring

```bash
TAG="v2.1"
FILE="src/core/processor.cpp"

# Capture metrics
git checkout $TAG

# Complexity
lizard "$FILE" > after_complexity.txt

# Churn (3 months after refactoring)
git log --oneline --since="$(git log -1 --format=%ai v2.0)" --before=3.month.later -- "$FILE" \
  | wc -l > after_churn.txt
```

### Compare

```bash
echo "=== Before ==="
cat before_complexity.txt
cat before_churn.txt

echo "=== After ==="
cat after_complexity.txt
cat after_churn.txt
```

**Success metrics**:
- CCN reduced by 40-60%
- Churn reduced by 30-50%
- SATD comments resolved
- Fewer bug reports

---

## Troubleshooting

### Issue: "No hotspots found"

**Causes**:
1. Thresholds too high
2. Recent repository (not enough history)
3. Files filtered out incorrectly

**Solutions**:
- Lower complexity threshold (CCN >10 instead of >20)
- Use more history (--since=24.month)
- Check file filters (are you excluding production code?)

### Issue: "Git log is empty"

**Causes**:
1. Date range too narrow
2. No commits in specified paths
3. Wrong format

**Solutions**:
```bash
# Verify commits exist
git log --oneline --since=12.month

# Check specific path
git log --oneline -- src/

# Verify log format
head git.log
```

### Issue: "Complexity tool doesn't support my language"

**Solutions**:
- Use LOC as proxy complexity metric
- Find language-specific tool (see `hotspot_analysis.md` § Alternative Metrics)
- Use SonarQube for multi-language support

### Issue: "Paths don't match when joining"

**Causes**:
- Different path formats (relative vs. absolute)
- Whitespace in paths
- Case sensitivity

**Solutions**:
```python
# Normalize paths before joining
churn['file'] = churn['entity'].str.strip().str.lower()
complexity['file'] = complexity['file'].str.strip().str.lower()
```

---

## Complete Example: Real Analysis

**Scenario**: 75,000 line Python web application, development slowing down, unclear where to focus refactoring.

**Time**: 2 hours total

**Steps**:

1. **Quick churn check** (5 min)
   - Identified top 20 files
   - `views/dashboard.py` has 142 commits

2. **Complexity analysis** (10 min)
   - Radon: `dashboard.py` has CCN 65
   - Hotspot score: 142 × 65 = 9,230

3. **Investigation** (15 min)
   - 34 commits: "Fix dashboard loading"
   - 28 commits: "Update dashboard filters"
   - 12 TODOs in file

4. **Function-level X-ray** (10 min)
   - `load_dashboard_data()`: CCN 38 (of 65 total)
   - 420 lines in one function

5. **Coupling check** (15 min)
   - Dashboard couples with 12 other modules
   - High SoC score: 380

6. **Decision** (5 min)
   - Priority P0: Refactor `load_dashboard_data()`
   - Split into smaller functions
   - Extract data loading to service layer

7. **Refactoring** (2 weeks dev time)
   - Reduced CCN from 65 to 28
   - Split 420-line function into 8 smaller functions
   - Extracted `DashboardService` to decouple

8. **Results** (measured 3 months later)
   - Churn reduced by 58% (142 → 60 commits)
   - Zero dashboard loading bugs (was 34 before)
   - Feature velocity improved 40%
   - Team satisfaction: "Much easier to work with"

**ROI**: 2 weeks investment → ongoing productivity gains.

---

## Related Resources

- **`fundamentals.md`**: Core concepts and ROI framework
- **`churn_analysis.md`**: Detailed churn techniques
- **`hotspot_analysis.md`**: Hotspot identification methods
- **`temporal_coupling.md`**: Coupling detection and interpretation
- **`code_maat_guide.md`**: Code Maat tool reference
- **`templates/refactoring_assessment.md`**: Workflow checklist
