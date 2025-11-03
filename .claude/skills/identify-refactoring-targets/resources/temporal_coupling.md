# Temporal Coupling Analysis

## Purpose

This resource explains how to detect temporal coupling (files that change together) to reveal hidden dependencies, architectural issues, and missing abstractions.

## When You Need This

- Investigating architectural decay
- Understanding why changes ripple across modules
- Detecting hidden dependencies not visible in code
- Planning architectural improvements
- Identifying missing abstractions

---

## What is Temporal Coupling?

**Definition**: Modules that tend to change together over time, even if not directly dependent in code.

**Also called**: Logical coupling, evolutionary coupling, change coupling

**Example**: If `auth.cpp` and `database.cpp` change together in 75% of commits, they have 75% temporal coupling.

---

## Why It Matters

### Expected Coupling (Low Concern)

- UI file + corresponding CSS/template
- Header file + implementation file (C++)
- Test file + source file
- Database schema + migration script

**These are natural dependencies.**

### Unexpected Coupling (High Concern)

- Business logic + UI rendering
- Database layer + network protocol
- Unrelated feature modules
- Test utilities + production code

**These indicate architectural problems:**
- Missing abstractions
- Tangled responsibilities
- Copy-paste duplication
- Shared global state
- Breaking changes rippling through system

---

## Detecting Temporal Coupling

### Manual Approach: Multi-File Commits

```bash
# Find commits touching multiple files
git log --name-only --oneline --since=6.month \
  | awk '
    /^[0-9a-f]+ / {
      if (NR > 1 && count > 1) {
        print prev_commit ": " count " files"
        for (i in files) print "  " files[i]
      }
      commit = $0
      delete files
      count = 0
      next
    }
    NF > 0 {
      files[count++] = $0
    }
    {prev_commit = commit}
  '
```

**Output**:
```
a3b2c1 Fix authentication bug: 3 files
  src/auth/login.cpp
  src/database/user_query.cpp
  src/ui/login_panel.cpp
```

**Interpretation**: These 3 files change together. Investigate why.

### Simple Pairwise Analysis

```bash
# Files that change together with target file
TARGET="src/auth/login.cpp"

git log --name-only --oneline --since=6.month -- "$TARGET" \
  | grep -v "^[0-9a-f]" \
  | grep -v "^$" \
  | grep -v "$TARGET" \
  | sort \
  | uniq -c \
  | sort -nr \
  | head -20
```

**Output**:
```
   15 src/database/user_query.cpp
   12 src/ui/login_panel.cpp
    8 src/auth/session.cpp
```

**Interpretation**: `user_query.cpp` changes with `login.cpp` in 15 commits. Strong coupling.

---

## Using Code Maat for Coupling Analysis

### Step 1: Generate Git Log

```bash
git log --all --numstat --date=short \
  --pretty=format:'--%h--%ad--%aN' \
  --no-renames \
  --after=2024-01-01 \
  > git.log
```

**Format**: Code Maat requires specific log format (`git2`).

### Step 2: Run Coupling Analysis

```bash
java -jar code-maat-1.0.4-standalone.jar \
  -l git.log \
  -c git2 \
  -a coupling \
  --min-revs 5 \
  --min-coupling 30 \
  > coupling.csv
```

**Parameters**:
- `-a coupling`: Analysis type
- `--min-revs 5`: Minimum shared commits (default: 5)
- `--min-coupling 30`: Minimum coupling percentage (default: 30)
- `--max-changeset-size 30`: Maximum files in one commit (default: 30)

**Output** (`coupling.csv`):
```
entity,coupled,degree,average-revs
src/core/module_a.cpp,src/ui/panel_b.cpp,78,12
src/database/query.cpp,src/network/api.cpp,65,8
```

**Fields**:
- `entity`: First file
- `coupled`: Second file
- `degree`: Coupling percentage (0-100)
- `average-revs`: Average commits per shared change

### Step 3: Interpret Results

**Coupling degree interpretation**:

| Degree | Interpretation | Action |
|--------|----------------|--------|
| 20-40% | Weak coupling | Monitor, may be coincidental |
| 40-60% | Moderate coupling | Investigate relationship |
| 60-80% | Strong coupling | High priority for review |
| 80-100% | Very strong coupling | Immediate architectural review |

**Example**: 78% degree means each modification to `module_a.cpp` has a 78% chance of requiring changes to `panel_b.cpp`.

---

## Temporal Period Aggregation

**Problem**: Frequent incremental commits during active development inflate coupling scores.

**Solution**: Treat same-day commits as a single logical change.

```bash
java -jar code-maat-1.0.4-standalone.jar \
  -l git.log \
  -c git2 \
  -a coupling \
  --temporal-period 1 \
  --min-coupling 50
```

**`--temporal-period 1`**: Aggregate commits within 1 day.

**Effect**: Reduces noise from "checkpoint" commits during feature development.

---

## Sum of Coupling (SoC) Analysis

**Purpose**: Identify files that frequently couple with MANY other files.

**Indicates**: Architectural connectors, "God objects," or central coordination points.

```bash
java -jar code-maat-1.0.4-standalone.jar \
  -l git.log \
  -c git2 \
  -a soc \
  > soc.csv
```

**Output**:
```
entity,soc
src/core/config.cpp,450
src/util/helper.cpp,380
src/main.cpp,320
```

**Interpretation**: `config.cpp` has total coupling score of 450 across all relationships. Likely a coordination bottleneck or architectural hub.

**High SoC indicates**:
- Central configuration files (expected)
- Utility libraries used everywhere (often OK)
- God objects (problem)
- Missing abstraction layers (problem)

---

## Filtering and Focus

### Exclude Test Files

```bash
# Filter git log before analysis
grep -v "^.*test/" git.log > git_filtered.log

java -jar code-maat-1.0.4-standalone.jar \
  -l git_filtered.log \
  -c git2 \
  -a coupling
```

### Focus on Specific Directory

```bash
# Generate log for specific directory
git log --all --numstat --date=short \
  --pretty=format:'--%h--%ad--%aN' \
  --no-renames \
  --after=2024-01-01 \
  -- src/core/ \
  > git_core.log

java -jar code-maat-1.0.4-standalone.jar \
  -l git_core.log \
  -c git2 \
  -a coupling
```

### Adjust Changeset Size Threshold

**Problem**: Large commits (mass refactorings) skew coupling.

**Solution**: Limit maximum files per commit.

```bash
java -jar code-maat-1.0.4-standalone.jar \
  -l git.log \
  -c git2 \
  -a coupling \
  --max-changeset-size 10
```

**Effect**: Ignores commits touching >10 files (likely bulk operations).

---

## Investigating Coupled Files

Once coupling identified, investigate WHY:

```bash
FILE_A="src/core/module_a.cpp"
FILE_B="src/ui/panel_b.cpp"

# 1. Find commits touching both
git log --oneline --all -- "$FILE_A" "$FILE_B"

# 2. View detailed changes
git log -p --all -- "$FILE_A" "$FILE_B"

# 3. Review commit messages
git log --pretty=format:'%s' --all -- "$FILE_A" "$FILE_B" \
  | sort \
  | uniq -c \
  | sort -nr

# 4. Check for shared dependencies (grep for common imports/includes)
grep -h "^#include\|^import" "$FILE_A" "$FILE_B" | sort | uniq -c | sort -nr
```

**Questions to answer**:
- What's the shared responsibility?
- Is there a missing abstraction?
- Are they in correct architectural layers?
- Is this expected coupling or problematic?

---

## Coupling Across Architectural Boundaries

### Define Architectural Layers

**Example layered architecture**:
```
src/ui/           # Presentation layer
src/core/         # Business logic
src/database/     # Data access layer
src/network/      # External communication
```

**Expected coupling**:
- UI → Core (presentation depends on business logic)
- Core → Database (business logic queries data)
- Core → Network (business logic calls external APIs)

**Unexpected coupling**:
- UI → Database (presentation directly accessing data)
- Database → Network (data layer calling APIs)
- UI → Network (presentation directly calling APIs)

### Detect Cross-Boundary Coupling

```bash
# Filter coupling results for architectural violations
grep "src/ui" coupling.csv | grep "src/database"
```

**Example output**:
```
src/ui/user_panel.cpp,src/database/user_query.cpp,72,10
```

**Interpretation**: UI directly coupled to database (72% degree). Architectural violation. Should go through business logic layer.

**Action**: Introduce business logic interface to decouple UI from database.

---

## Conway's Law Analysis

**Conway's Law**: "Organizations which design systems are constrained to produce designs which are copies of the communication structures of these organizations."

**Implication**: Architectural coupling reflects team communication patterns.

### Step 1: Map Team Structure

Document team ownership:
```
Team A: src/frontend/
Team B: src/backend/api/
Team C: src/database/
```

### Step 2: Identify Cross-Team Coupling

```bash
# Find coupling across team boundaries
grep "src/frontend" coupling.csv | grep "src/database"
```

**High cross-team coupling indicates**:
- Missing architectural boundaries
- Need for better interfaces/contracts
- Potential team restructuring

### Step 3: Communication Analysis

Code Maat can analyze developer coupling:

```bash
java -jar code-maat-1.0.4-standalone.jar \
  -l git.log \
  -c git2 \
  -a communication \
  > communication.csv
```

**Output**: Developers who frequently commit to same files.

**Interpretation**: High cross-team developer coupling suggests coordination overhead.

### Resolving Misalignment

**Options**:
1. **Restructure teams**: Align with current architecture
2. **Refactor architecture**: Align with team boundaries
3. **Introduce interfaces**: Decouple teams via contracts

**Goal**: Minimize cross-team coordination.

---

## Decoupling Strategies

### Strategy 1: Extract Interface

**Problem**: Two modules directly depend on each other.

**Solution**: Define interface, hide implementation.

```cpp
// Before: Direct coupling
#include "database/user_query.h"  // Direct dependency

// After: Interface decoupling
#include "core/user_repository.h"  // Abstract interface
```

### Strategy 2: Introduce Facade

**Problem**: Multiple modules couple with complex subsystem.

**Solution**: Provide simple facade hiding complexity.

```cpp
// Instead of each module calling database, network, cache directly:
class DataManager {  // Facade
public:
    User getUser(int id);  // Hides database, cache, network complexity
};
```

### Strategy 3: Event-Driven Architecture

**Problem**: Changes ripple through multiple modules.

**Solution**: Use pub/sub to decouple.

```cpp
// Before: Module A directly calls Module B, C, D
moduleB.onUpdate();
moduleC.onUpdate();
moduleD.onUpdate();

// After: Event-driven
eventBus.publish(UpdateEvent());  // Modules subscribe independently
```

### Strategy 4: Dependency Injection

**Problem**: Hard-coded dependencies create tight coupling.

**Solution**: Pass dependencies, don't create them.

```cpp
// Before: Hard-coded dependency
class AuthService {
    DatabaseConnection db = new DatabaseConnection();  // Tight coupling
};

// After: Dependency injection
class AuthService {
    AuthService(DatabaseConnection db) : db_(db) {}  // Loose coupling
    DatabaseConnection db_;
};
```

---

## Coupling Trends Over Time

Track coupling evolution:

```bash
# Coupling at v1.0
git log --all --numstat --date=short \
  --pretty=format:'--%h--%ad--%aN' \
  --no-renames \
  --before=2024-01-01 \
  > git_v1.log

java -jar code-maat-1.0.4-standalone.jar -l git_v1.log -c git2 -a soc > soc_v1.csv

# Coupling at v2.0
git log --all --numstat --date=short \
  --pretty=format:'--%h--%ad--%aN' \
  --no-renames \
  --before=2024-06-01 \
  > git_v2.log

java -jar code-maat-1.0.4-standalone.jar -l git_v2.log -c git2 -a soc > soc_v2.csv

# Compare
diff soc_v1.csv soc_v2.csv
```

**Goal**: Sum of coupling should decrease after decoupling refactoring.

---

## Common Issues

### Issue: Too Many False Positives

**Problem**: Unrelated files show coupling.

**Cause**: Large commits touching many files (e.g., formatting changes).

**Solution**: Use `--max-changeset-size` to filter.

```bash
java -jar code-maat-1.0.4-standalone.jar \
  -a coupling \
  --max-changeset-size 10
```

### Issue: Expected Coupling Dominates Results

**Problem**: Test files coupling with source files clutters output.

**Cause**: Test-source pairs are expected.

**Solution**: Filter git log before analysis.

```bash
grep -v "test/" git.log > git_filtered.log
```

### Issue: No Coupling Detected

**Problem**: `--min-coupling` threshold too high.

**Solution**: Lower threshold.

```bash
java -jar code-maat-1.0.4-standalone.jar \
  -a coupling \
  --min-coupling 20  # Lower from default 30
```

---

## Example: Real Coupling Analysis

**Scenario**: 100,000 line C++ codebase, suspecting architectural issues.

**Step 1: Generate log**
```bash
git log --all --numstat --date=short \
  --pretty=format:'--%h--%ad--%aN' \
  --no-renames \
  --after=2024-01-01 \
  > git.log
```

**Step 2: Run coupling analysis**
```bash
java -jar code-maat-1.0.4-standalone.jar \
  -l git.log \
  -c git2 \
  -a coupling \
  --min-coupling 50 \
  --temporal-period 1 \
  > coupling.csv
```

**Step 3: Review results**
```bash
head -20 coupling.csv
```

**Finding**:
```
src/ui/settings_panel.cpp,src/database/config_db.cpp,82,15
```

**Interpretation**: UI directly coupled to database (82% degree, 15 shared commits). Architectural violation.

**Step 4: Investigate**
```bash
git log --oneline -- src/ui/settings_panel.cpp src/database/config_db.cpp
```

**Discovery**: Settings panel directly reads/writes database for config values.

**Action**: Introduce `ConfigManager` service layer to decouple UI from database.

**Step 5: Refactor & Re-measure**

After refactoring:
```bash
# Generate new log (post-refactor commits only)
git log --all --numstat --date=short \
  --pretty=format:'--%h--%ad--%aN' \
  --no-renames \
  --after=2024-06-01 \
  > git_new.log

java -jar code-maat-1.0.4-standalone.jar \
  -l git_new.log \
  -c git2 \
  -a coupling \
  --min-coupling 50
```

**Expected**: No coupling between `settings_panel.cpp` and `config_db.cpp`. Coupling now through `ConfigManager`.

---

## Related Resources

- **`fundamentals.md`**: Understanding why coupling matters
- **`hotspot_analysis.md`**: Combine coupling with hotspot analysis
- **`code_maat_guide.md`**: Complete Code Maat tool reference
- **`practical_workflow.md`**: Step-by-step workflow integrating coupling analysis
