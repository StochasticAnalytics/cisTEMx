# Code Maat Guide

## Purpose

This resource provides comprehensive guide to using Code Maat, a command-line tool for mining version control data to analyze code evolution, coupling, ownership, and communication patterns.

## When You Need This

- Advanced version control analysis beyond basic git commands
- Temporal coupling detection
- Developer ownership and coordination analysis
- Automating behavioral code analysis workflows
- Research-grade VCS mining

---

## What is Code Maat?

**Code Maat**: Command-line tool for mining and analyzing version control data.

**Author**: Adam Tornhill (creator of behavioral code analysis methodology)

**Written in**: Clojure (runs on JVM)

**License**: Open source

**Supported VCS**: Git, Subversion, Mercurial, Perforce, TFS

**Key capability**: Transforms VCS logs into structured data (CSV) for analysis.

---

## Installation

### Option 1: Download Pre-Built JAR (Recommended)

```bash
# Download latest release
wget https://github.com/adamtornhill/code-maat/releases/download/v1.0.4/code-maat-1.0.4-standalone.jar

# Verify download
java -jar code-maat-1.0.4-standalone.jar -h
```

### Option 2: Build from Source

**Prerequisites**: Leiningen (Clojure build tool)

```bash
# Install Leiningen
curl https://raw.githubusercontent.com/technomancy/leiningen/stable/bin/lein > /usr/local/bin/lein
chmod +x /usr/local/bin/lein

# Clone and build
git clone https://github.com/adamtornhill/code-maat.git
cd code-maat
lein uberjar

# JAR location
ls target/code-maat-*-standalone.jar
```

### Verify Installation

```bash
java -jar code-maat-1.0.4-standalone.jar -v
```

**Output**: Version number and build info.

---

## Generating Input Logs

Code Maat requires VCS logs in specific format.

### Git Log Format (git2) - Recommended

```bash
git log --all --numstat --date=short \
  --pretty=format:'--%h--%ad--%aN' \
  --no-renames \
  --after=2024-01-01 \
  > git.log
```

**Format breakdown**:
- `--all`: All branches
- `--numstat`: Show line statistics (added/deleted)
- `--date=short`: YYYY-MM-DD format
- `--pretty=format:'--%h--%ad--%aN'`: Custom format
  - `--`: Separator
  - `%h`: Short commit hash
  - `--`: Separator
  - `%ad`: Author date
  - `--`: Separator
  - `%aN`: Author name
- `--no-renames`: Treat renamed files as separate
- `--after=2024-01-01`: Date filter

**Example output**:
```
--a3b2c1--2024-03-15--John Doe
45      23      src/core/processor.cpp
12      8       src/auth/login.cpp

--b4c3d2--2024-03-16--Jane Smith
0       0       README.md
```

### Filtering Strategies

**Exclude directories**:
```bash
git log --all --numstat --date=short \
  --pretty=format:'--%h--%ad--%aN' \
  --no-renames \
  --after=2024-01-01 \
  -- . ":(exclude)vendor/*" ":(exclude)test/*" \
  > git.log
```

**Include only production code**:
```bash
git log --all --numstat --date=short \
  --pretty=format:'--%h--%ad--%aN' \
  --no-renames \
  --after=2024-01-01 \
  -- src/ \
  > git.log
```

**Exclude specific file types**:
```bash
git log --all --numstat --date=short \
  --pretty=format:'--%h--%ad--%aN' \
  --no-renames \
  --after=2024-01-01 \
  | grep -v "\.json$" \
  | grep -v "\.md$" \
  > git.log
```

### Date Range Selection

**Recommendation**: 6-12 months of history.

**Why**:
- Too short (<3 months): Insufficient data, noise dominates
- Too long (>24 months): Outdated patterns, analysis slower

**Examples**:
```bash
# Last 6 months
--after=6.month.ago

# Last year
--after=12.month.ago

# Specific date range
--after=2024-01-01 --before=2024-06-30

# Relative to tag
--after="$(git log -1 --format=%ai v1.0)"
```

---

## Basic Usage

### Command Structure

```bash
java -jar code-maat-1.0.4-standalone.jar \
  -l <logfile> \
  -c <log-format> \
  -a <analysis-type> \
  [options] \
  > output.csv
```

**Required parameters**:
- `-l`: Path to git log file
- `-c`: Log format (`git2` for git)
- `-a`: Analysis type

**Optional parameters**: Vary by analysis type (covered below).

### Example: Revision Count

```bash
java -jar code-maat-1.0.4-standalone.jar \
  -l git.log \
  -c git2 \
  -a revisions \
  > revisions.csv
```

**Output** (`revisions.csv`):
```
entity,n-revs
src/core/processor.cpp,87
src/ui/main_panel.cpp,64
src/database/query.cpp,52
```

---

## Available Analyses

### 1. Summary

**Purpose**: Overview statistics for the repository.

```bash
java -jar code-maat-1.0.4-standalone.jar -l git.log -c git2 -a summary
```

**Output**:
```
statistic,value
number-of-commits,1234
number-of-entities,567
number-of-entities-changed,234
number-of-authors,12
```

### 2. Revisions

**Purpose**: Count how many times each file changed.

```bash
java -jar code-maat-1.0.4-standalone.jar -l git.log -c git2 -a revisions > revisions.csv
```

**Use case**: Identify high-churn files for hotspot analysis.

### 3. Authors

**Purpose**: Count developers per file/module.

```bash
java -jar code-maat-1.0.4-standalone.jar -l git.log -c git2 -a authors --min-revs 5 > authors.csv
```

**Options**:
- `--min-revs N`: Only show files with ≥N revisions

**Output**:
```
entity,n-authors,n-revs
src/core/processor.cpp,5,87
```

**Interpretation**: High author count indicates coordination complexity.

### 4. Entity Ownership

**Purpose**: Show contribution distribution per file.

```bash
java -jar code-maat-1.0.4-standalone.jar -l git.log -c git2 -a entity-ownership > ownership.csv
```

**Output**:
```
entity,author,added,deleted,commits
src/core/processor.cpp,John Doe,1234,567,45
src/core/processor.cpp,Jane Smith,890,234,32
```

**Use case**: Identify primary maintainers and knowledge silos.

### 5. Coupling

**Purpose**: Detect temporal coupling (files that change together).

```bash
java -jar code-maat-1.0.4-standalone.jar \
  -l git.log \
  -c git2 \
  -a coupling \
  --min-coupling 30 \
  --min-revs 5 \
  --temporal-period 1 \
  > coupling.csv
```

**Options**:
- `--min-coupling N`: Minimum coupling percentage (default: 30)
- `--min-revs N`: Minimum shared commits (default: 5)
- `--max-changeset-size N`: Maximum files per commit (default: 30)
- `--temporal-period N`: Aggregate commits within N days

**Output**:
```
entity,coupled,degree,average-revs
src/core/module_a.cpp,src/ui/panel_b.cpp,78,12
```

**Interpretation**: 78% of changes to `module_a.cpp` also touch `panel_b.cpp`.

### 6. Sum of Coupling (SoC)

**Purpose**: Identify files coupling with many others.

```bash
java -jar code-maat-1.0.4-standalone.jar -l git.log -c git2 -a soc > soc.csv
```

**Output**:
```
entity,soc
src/core/config.cpp,450
src/util/helper.cpp,380
```

**Interpretation**: `config.cpp` has highest total coupling. Likely architectural hub or coordination bottleneck.

### 7. Age

**Purpose**: Identify code stability by last modification date.

```bash
java -jar code-maat-1.0.4-standalone.jar -l git.log -c git2 -a age > age.csv
```

**Output**:
```
entity,age-months
src/legacy/old_module.cpp,48
src/core/processor.cpp,2
```

**Use case**: Combine with complexity to identify "ancient legacy code" vs. "active development."

### 8. Absolute Churn

**Purpose**: Total lines added/deleted per file.

```bash
java -jar code-maat-1.0.4-standalone.jar -l git.log -c git2 -a abs-churn > abs-churn.csv
```

**Output**:
```
entity,added,deleted,commits
src/core/processor.cpp,2345,1234,87
```

### 9. Entity Churn

**Purpose**: Pre-release defect predictor.

```bash
java -jar code-maat-1.0.4-standalone.jar -l git.log -c git2 -a entity-churn > entity-churn.csv
```

**Output**: Lines added/deleted/total per file.

**Research finding**: High churn predicts post-release defects.

### 10. Communication

**Purpose**: Identify developers who frequently work on same files.

```bash
java -jar code-maat-1.0.4-standalone.jar -l git.log -c git2 -a communication > communication.csv
```

**Output**:
```
author,peer,shared,average,strength
John Doe,Jane Smith,45,12,78
```

**Use case**: Conway's Law analysis, team coordination patterns.

### 11. Fragmentation

**Purpose**: Measure code ownership distribution.

```bash
java -jar code-maat-1.0.4-standalone.jar -l git.log -c git2 -a fragmentation > fragmentation.csv
```

**Output**:
```
entity,fragmentation
src/core/processor.cpp,0.85
```

**Interpretation**: High fragmentation (>0.7) means many developers contribute small amounts. Low ownership clarity.

### 12. Entity Effort

**Purpose**: Developer effort distribution per file.

```bash
java -jar code-maat-1.0.4-standalone.jar -l git.log -c git2 -a entity-effort > entity-effort.csv
```

**Output**: Author contribution ratios per file.

---

## Performance Tuning

### Increase Heap Memory

For large repositories (>100k commits):

```bash
java -Djava.awt.headless=true -Xmx4g \
  -jar code-maat-1.0.4-standalone.jar \
  -l git.log \
  -c git2 \
  -a coupling
```

**Options**:
- `-Xmx4g`: Set max heap to 4GB
- `-Djava.awt.headless=true`: Disable GUI (not needed)

### Limit Input Size

**Problem**: Code Maat processes logs in-memory.

**Solution**: Use sensible date ranges.

```bash
# Instead of entire history
git log --all --numstat --after=12.month.ago > git.log
```

### Parallelize Multiple Analyses

Run multiple analyses on same log:

```bash
LOG="git.log"

java -jar code-maat.jar -l $LOG -c git2 -a revisions > revisions.csv &
java -jar code-maat.jar -l $LOG -c git2 -a coupling > coupling.csv &
java -jar code-maat.jar -l $LOG -c git2 -a authors > authors.csv &

wait
```

---

## Workflow Integration

### Automated Analysis Script

```bash
#!/bin/bash
# analyze_codebase.sh

REPO_PATH="${1:-.}"
START_DATE="${2:-12.month.ago}"
OUTPUT_DIR="analysis_$(date +%Y%m%d)"
MAAT_JAR="code-maat-1.0.4-standalone.jar"

mkdir -p "$OUTPUT_DIR"
cd "$REPO_PATH"

echo "=== Generating git log ==="
git log --all --numstat --date=short \
  --pretty=format:'--%h--%ad--%aN' \
  --no-renames \
  --after="$START_DATE" \
  > "$OUTPUT_DIR/git.log"

LOG="$OUTPUT_DIR/git.log"

echo "=== Running analyses ==="
java -jar "$MAAT_JAR" -l "$LOG" -c git2 -a summary > "$OUTPUT_DIR/summary.csv"
java -jar "$MAAT_JAR" -l "$LOG" -c git2 -a revisions > "$OUTPUT_DIR/revisions.csv"
java -jar "$MAAT_JAR" -l "$LOG" -c git2 -a coupling --min-coupling 40 > "$OUTPUT_DIR/coupling.csv"
java -jar "$MAAT_JAR" -l "$LOG" -c git2 -a authors > "$OUTPUT_DIR/authors.csv"
java -jar "$MAAT_JAR" -l "$LOG" -c git2 -a soc > "$OUTPUT_DIR/soc.csv"

echo "=== Complete ==="
echo "Results in: $OUTPUT_DIR"
```

**Usage**:
```bash
chmod +x analyze_codebase.sh
./analyze_codebase.sh /path/to/repo 6.month.ago
```

---

## Combining with Complexity Data

### Hotspot Analysis Workflow

```bash
# 1. Generate git log
git log --all --numstat --date=short \
  --pretty=format:'--%h--%ad--%aN' \
  --no-renames --after=12.month.ago \
  > git.log

# 2. Get revisions (churn)
java -jar code-maat.jar -l git.log -c git2 -a revisions > revisions.csv

# 3. Get complexity
lizard --csv src/ > complexity.csv

# 4. Join in Python
python join_hotspots.py revisions.csv complexity.csv > hotspots.csv
```

**Python join script** (`join_hotspots.py`):
```python
import pandas as pd
import sys

revisions = pd.read_csv(sys.argv[1])
complexity = pd.read_csv(sys.argv[2])

hotspots = pd.merge(
    revisions,
    complexity[['file', 'NLOC', 'CCN']],
    left_on='entity',
    right_on='file',
    how='inner'
)

hotspots['score'] = hotspots['n-revs'] * hotspots['CCN']
hotspots = hotspots.sort_values('score', ascending=False)

hotspots.to_csv(sys.stdout, index=False)
```

---

## Common Issues

### Issue: Empty Output

**Symptom**: CSV has only headers, no data.

**Causes**:
1. Log format mismatch (check `-c git2`)
2. Thresholds too restrictive (`--min-coupling`, `--min-revs`)
3. Date range too narrow (no commits)

**Solutions**:
```bash
# Verify log format
head git.log

# Lower thresholds
--min-coupling 10 --min-revs 1

# Expand date range
--after=24.month.ago
```

### Issue: Analysis Too Slow

**Symptom**: Code Maat runs for minutes/hours.

**Causes**:
1. Too much history (>2 years)
2. Insufficient heap memory
3. Large repository (>500k commits)

**Solutions**:
```bash
# Limit date range
--after=6.month.ago

# Increase heap
java -Xmx8g -jar code-maat.jar ...

# Filter log before analysis
grep "^src/" git.log > git_filtered.log
```

### Issue: Out of Memory Error

**Symptom**: `java.lang.OutOfMemoryError: Java heap space`

**Solution**:
```bash
java -Xmx8g -jar code-maat-1.0.4-standalone.jar ...
```

If still failing, reduce input size (shorter date range or filter directories).

---

## Output Format Details

### CSV Structure

All outputs are CSV (comma-separated values):
```
header1,header2,header3
value1,value2,value3
```

**Import to spreadsheet**: Excel, Google Sheets, LibreOffice Calc

**Process with scripts**: Python pandas, R, awk

### Common Fields

- `entity`: File path
- `n-revs`: Number of revisions (commits)
- `n-authors`: Number of unique authors
- `added`: Lines added
- `deleted`: Lines deleted
- `degree`: Coupling percentage (0-100)
- `soc`: Sum of coupling score

---

## Advanced Usage

### Time-Travel Analysis

Compare coupling at different points in time:

```bash
# Coupling at v1.0
git log --all --numstat --date=short \
  --pretty=format:'--%h--%ad--%aN' \
  --no-renames \
  --before="$(git log -1 --format=%ai v1.0)" \
  > git_v1.log

java -jar code-maat.jar -l git_v1.log -c git2 -a coupling > coupling_v1.csv

# Coupling at v2.0
git log --all --numstat --date=short \
  --pretty=format:'--%h--%ad--%aN' \
  --no-renames \
  --before="$(git log -1 --format=%ai v2.0)" \
  > git_v2.log

java -jar code-maat.jar -l git_v2.log -c git2 -a coupling > coupling_v2.csv

# Compare
diff coupling_v1.csv coupling_v2.csv
```

### Module-Level Analysis

Aggregate file-level results to component level:

```python
import pandas as pd

revisions = pd.read_csv('revisions.csv')

# Extract component from path (e.g., "src/core/file.cpp" → "core")
revisions['component'] = revisions['entity'].str.split('/').str[1]

# Aggregate by component
component_churn = revisions.groupby('component')['n-revs'].sum().sort_values(ascending=False)

print(component_churn)
```

---

## Alternatives to Code Maat

**When Code Maat isn't suitable**:

- **GitNStats**: Simpler, pre-built binaries, less flexible
- **code-forensics**: Node.js toolset with visualizations (includes Code Maat internally)
- **git log + awk**: DIY for simple analyses
- **Custom scripts**: Python/Ruby for project-specific needs

**Code Maat advantages**:
- Research-grade algorithms
- Comprehensive analysis types
- CSV output (easy integration)
- Multi-VCS support

---

## Related Resources

- **`fundamentals.md`**: Why behavioral code analysis matters
- **`churn_analysis.md`**: Git-native churn techniques
- **`hotspot_analysis.md`**: Combining churn + complexity
- **`temporal_coupling.md`**: Deep dive on coupling analysis
- **`practical_workflow.md`**: Complete end-to-end workflow
