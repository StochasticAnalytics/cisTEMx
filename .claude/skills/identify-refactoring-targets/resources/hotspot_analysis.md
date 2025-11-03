# Hotspot Analysis

## Purpose

This resource provides step-by-step techniques for identifying code hotspots by combining change frequency (churn) with complexity metrics to find highest-priority refactoring targets.

## When You Need This

- Prioritizing which code to refactor first
- Building data-driven refactoring backlog
- Justifying refactoring ROI to stakeholders
- Focusing limited resources on highest-impact areas

---

## The Hotspot Concept

**Hotspot**: Code that is both **complex** AND **frequently modified**.

**Core principle**: "If the code never changes, it's not costing us money."

- Complex but stable code → Low priority
- Simple but frequently changed → Low priority
- **Complex AND frequently changed → HIGH PRIORITY**

---

## Step-by-Step Hotspot Analysis

### Step 1: Calculate Churn Score

Get the 50 most-changed files:

```bash
git log --format=format: --name-only --since=12.month \
  | egrep -v '^$' \
  | sort \
  | uniq -c \
  | sort -nr \
  | head -50 \
  > churn_data.txt
```

**Output format**:
```
    87 src/core/processor.cpp
    64 src/ui/main_panel.cpp
    52 src/database/query.cpp
```

**Export to CSV**:
```bash
cat churn_data.txt | awk '{print $2","$1}' > churn.csv
```

### Step 2: Calculate Complexity Score

**For C++**:
```bash
lizard -l cpp --csv src/ > complexity.csv
```

**For Python**:
```bash
radon cc -a -s --json src/ > complexity.json
```

**For Multiple Languages**:
```bash
lizard --csv src/ > complexity.csv
```

**For Ruby**:
```bash
flog --all --csv src/ > complexity.csv
```

**Lizard output fields**:
- `NLOC`: Lines of Code
- `CCN`: Cyclomatic Complexity Number
- `token`: Token count
- `parameter`: Parameter count
- `function`: Function name
- `file`: File path

### Step 3: Join Churn and Complexity Data

**Option A: Python Script**

```python
import pandas as pd

# Load data
churn = pd.read_csv('churn.csv', names=['file', 'commits'])
complexity = pd.read_csv('complexity.csv')

# Normalize paths (adjust as needed for your project)
churn['file'] = churn['file'].str.strip()
complexity['file'] = complexity['file'].str.strip()

# Join on file path
hotspots = pd.merge(
    churn,
    complexity[['file', 'NLOC', 'CCN']],
    on='file',
    how='inner'
)

# Calculate hotspot score (example: CCN × commits)
hotspots['hotspot_score'] = hotspots['CCN'] * hotspots['commits']

# Sort by score
hotspots = hotspots.sort_values('hotspot_score', ascending=False)

# Save
hotspots.to_csv('hotspots.csv', index=False)

# Display top 20
print(hotspots[['file', 'commits', 'CCN', 'NLOC', 'hotspot_score']].head(20))
```

**Option B: Spreadsheet Tool**

1. Import `churn.csv` into Sheet1
2. Import `complexity.csv` into Sheet2
3. Use VLOOKUP/INDEX-MATCH to join on file path
4. Create new column: `=CCN * commits` (or other scoring formula)
5. Sort by score descending

### Step 4: Visualize as Scatter Plot

**Using Python (matplotlib)**:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plt.scatter(hotspots['commits'], hotspots['CCN'], alpha=0.6, s=hotspots['NLOC']/10)

# Label top 10 hotspots
top_10 = hotspots.head(10)
for idx, row in top_10.iterrows():
    plt.annotate(
        row['file'].split('/')[-1],  # Just filename
        (row['commits'], row['CCN']),
        fontsize=8,
        alpha=0.8
    )

plt.xlabel('Change Frequency (# commits)')
plt.ylabel('Cyclomatic Complexity (CCN)')
plt.title('Code Hotspots Analysis')
plt.grid(True, alpha=0.3)
plt.savefig('hotspots.png', dpi=150)
plt.show()
```

**Using Google Sheets**:

1. Select data columns (commits, CCN)
2. Insert → Chart → Scatter chart
3. Customize:
   - X-axis: commits
   - Y-axis: CCN
   - Series: file names
   - Bubble size: NLOC (optional)
4. Identify top-right quadrant

### Step 5: Identify Top-Right Quadrant

Files in the **top-right quadrant** are your hotspots:
- High complexity (Y-axis)
- High change frequency (X-axis)

**Action**: These are your priority refactoring targets.

---

## Hotspot Scoring Formulas

Different formulas emphasize different aspects:

### Formula 1: Simple Product
```
hotspot_score = CCN × commits
```

**Pros**: Simple, intuitive
**Cons**: Doesn't account for file size

### Formula 2: Normalized by LOC
```
hotspot_score = (CCN / NLOC) × commits
```

**Pros**: Accounts for complexity density
**Cons**: Penalizes large files

### Formula 3: Weighted Sum
```
hotspot_score = (0.5 × CCN) + (0.3 × commits) + (0.2 × NLOC / 100)
```

**Pros**: Tunable weights
**Cons**: Weights are arbitrary

### Formula 4: Quadrant Position
```
hotspot_score = (CCN_percentile × commits_percentile) / 100
```

**Pros**: Relative ranking, no absolute thresholds
**Cons**: More complex calculation

**Recommendation**: Start with simple product (Formula 1). Adjust if needed.

---

## Multi-Level Hotspot Analysis

### Level 1: Architectural (Component/Directory)

Identify problematic components:

```bash
for dir in src/core src/ui src/database; do
  commits=$(git log --format=format: --name-only --since=12.month -- "$dir" \
    | egrep -v '^$' \
    | wc -l)
  loc=$(lizard --csv "$dir" | awk -F',' '{sum+=$2} END {print sum}')
  ccn=$(lizard --csv "$dir" | awk -F',' '{sum+=$3} END {print sum}')

  echo "$dir: $commits commits, $loc LOC, $ccn CCN"
done
```

**Result**: Identify which component(s) need attention.

### Level 2: File Level

Within problematic component, find specific files (Steps 1-5 above).

**Result**: Narrow to top 10-20 files.

### Level 3: Function/Method Level

Within hotspot files, identify exact functions:

```bash
# Function-level complexity
lizard --verbose src/core/hotspot_file.cpp
```

**Output**:
```
  NLOC    CCN   token  parameter  function@line
    145     32     856          5  processData@142
     87     18     432          3  validateInput@234
```

**Combine with git log -L** (function history):

```bash
git log -L :processData:src/core/hotspot_file.cpp --oneline
```

**Result**: Pinpoint exact functions needing refactoring (e.g., 186 lines out of 200,000 codebase).

---

## Alternative Complexity Metrics

### When Lizard Doesn't Support Your Language

**Option 1: Use LOC as Proxy**

```bash
# Count lines per file
find src/ -name "*.ext" -exec wc -l {} \; | awk '{print $2","$1}' > loc.csv
```

**Limitations**: LOC alone misses complexity, but combined with churn it's useful.

**Option 2: Language-Specific Tools**

| Language | Tool | Command |
|----------|------|---------|
| Python | Radon | `radon cc -a -s --json src/` |
| Ruby | Flog | `flog --all --csv src/` |
| JavaScript | ESLint complexity | `eslint --rule 'complexity: [error, 10]'` |
| Go | Gocyclo | `gocyclo -over 10 src/` |
| Java | PMD | `pmd -d src/ -R category/java/design.xml` |
| C/C++ | pmccabe | `pmccabe src/*.c` |

**Option 3: SonarQube**

Run SonarQube analysis and export metrics.

### Indentation Depth as Complexity Proxy

Deeply nested code is harder to understand:

```python
import re

def calculate_max_indent(file_path):
    with open(file_path, 'r') as f:
        max_indent = 0
        for line in f:
            if line.strip():  # Skip empty lines
                indent = len(line) - len(line.lstrip())
                max_indent = max(max_indent, indent)
    return max_indent

# Scan all files
for file in file_list:
    indent = calculate_max_indent(file)
    print(f"{file},{indent}")
```

**Threshold**: Max indent >6 levels is concerning.

---

## Thresholds and Interpretation

### Hotspot Classification

| CCN | Commits/Year | Classification | Action |
|-----|--------------|----------------|--------|
| <10 | <20 | Normal | OK |
| <10 | >20 | Active development | Monitor |
| >20 | <20 | Complex but stable | Leave alone unless modifying |
| >20 | >20 | **HOTSPOT** | **Priority refactoring** |
| >50 | >50 | **CRITICAL HOTSPOT** | **Urgent refactoring** |

### Complexity Grades (Radon)

| Grade | CCN Range | Interpretation |
|-------|-----------|----------------|
| A | 1-5 | Simple, low risk |
| B | 6-10 | Well-structured |
| C | 11-20 | Moderately complex |
| D | 21-50 | Complex, higher risk |
| E | 51-100 | Very complex, refactor |
| F | >100 | Extremely complex, urgent refactor |

---

## Investigating Hotspot Files

Once hotspots identified, investigate WHY:

```bash
FILE="src/core/hotspot_file.cpp"

# 1. View commit history
git log --oneline --follow -- "$FILE"

# 2. Analyze commit messages for patterns
git log --oneline --follow -- "$FILE" | awk '{$1=""; print}' | sort | uniq -c | sort -nr

# 3. Identify contributors
git shortlog -sn --follow -- "$FILE"

# 4. Check for technical debt markers
git grep -n "TODO\|FIXME\|HACK" -- "$FILE"

# 5. View detailed diffs (last 20 commits)
git log -p -20 --follow -- "$FILE"

# 6. Function-level complexity
lizard --verbose "$FILE"
```

**Questions to answer**:
- What's changing repeatedly? (Repeated bug fixes? Feature additions?)
- Who's changing it? (Single developer? Multiple teams?)
- Are there TODO/FIXME comments?
- What functions have highest complexity?

---

## Refactoring Prioritization

After identifying hotspots, prioritize by:

### Criteria

1. **Hotspot score** (CCN × commits) - highest first
2. **Business impact** - customer-facing features prioritized
3. **Team capacity** - realistic time estimates
4. **Dependencies** - can it be refactored in isolation?
5. **Test coverage** - easier to refactor with tests

### Decision Matrix

| File | Score | Impact | Effort | Priority |
|------|-------|--------|--------|----------|
| processor.cpp | 2784 | High | 3 weeks | P0 |
| main_panel.cpp | 1920 | Medium | 2 weeks | P1 |
| query.cpp | 1040 | Low | 1 week | P2 |

**Priority calculation**:
```
priority = (score × impact_factor) / effort_weeks
```

---

## Tracking Progress

### Before/After Comparison

**Before refactoring**:
```bash
git checkout v2.0
lizard --csv src/core/processor.cpp > before_complexity.csv
git log --oneline --since=12.month -- src/core/processor.cpp | wc -l > before_churn.txt
```

**After refactoring**:
```bash
git checkout v2.1
lizard --csv src/core/processor.cpp > after_complexity.csv
git log --oneline --since=3.month -- src/core/processor.cpp | wc -l > after_churn.txt
```

**Compare**:
- CCN reduction (target: 40-60%)
- Churn reduction (target: 30-50%)

### Complexity Trend

Track complexity over time:

```bash
for tag in v1.0 v1.1 v1.2 v2.0; do
  git checkout $tag
  ccn=$(lizard --csv src/core/processor.cpp | awk -F',' 'NR==2 {print $3}')
  echo "$tag,$ccn"
done > complexity_trend.csv
```

**Plot**: Visualize complexity evolution. Should decrease after refactoring.

---

## Common Pitfalls

### Pitfall 1: Refactoring Stable Code

**Problem**: High complexity but low churn.

**Solution**: Leave it alone unless you must modify it. "If it ain't broke, don't fix it."

### Pitfall 2: Ignoring Business Context

**Problem**: Refactoring internal utilities instead of customer-facing features.

**Solution**: Weight hotspot score by business impact.

### Pitfall 3: No Test Coverage

**Problem**: Refactoring without tests risks introducing bugs.

**Solution**: Add tests first, THEN refactor. Or write tests during refactoring (TDD).

### Pitfall 4: Scope Creep

**Problem**: Starting with one file, ending up refactoring entire component.

**Solution**: Set clear scope boundaries. Refactor incrementally.

### Pitfall 5: Not Measuring Results

**Problem**: Can't prove refactoring helped.

**Solution**: Track metrics before/after. Measure churn, complexity, velocity, defects.

---

## Example: Real Hotspot Analysis

**Scenario**: 50,000 line Python codebase

**Step 1: Churn**
```bash
git log --format=format: --name-only --since=12.month \
  | egrep -v '^$' \
  | egrep '\.py$' \
  | sort \
  | uniq -c \
  | sort -nr \
  | head -20 \
  > churn.txt
```

**Result**: `src/core/processor.py` has 87 commits.

**Step 2: Complexity**
```bash
radon cc -a -s --json src/ > complexity.json
```

**Result**: `processor.py` has CCN 45, NLOC 850.

**Step 3: Combine**
```
hotspot_score = 45 × 87 = 3,915
```

**Step 4: Investigate**
```bash
git log --oneline -- src/core/processor.py | head -20
```

**Findings**:
- 12 commits: "Fix bug in validation logic"
- 8 commits: "Update processor configuration"
- 6 commits: "Performance improvements"

**Conclusion**: Repeated bug fixes indicate quality issue. Validation logic likely needs refactoring.

**Step 5: X-Ray (Function-level)**
```bash
radon cc -s src/core/processor.py
```

**Result**: `validate_input()` has CCN 28 (out of total 45).

**Action**: Refactor `validate_input()` function (80 lines). Expected ROI: 60% reduction in bug fixes.

---

## Related Resources

- **`fundamentals.md`**: Understand hotspot concept and ROI framework
- **`churn_analysis.md`**: Detailed churn measurement techniques
- **`temporal_coupling.md`**: Find architectural issues via coupling
- **`code_maat_guide.md`**: Advanced analysis with Code Maat
- **`practical_workflow.md`**: Complete step-by-step workflow
