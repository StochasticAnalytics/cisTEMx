# Test Coverage Gap Prioritization Strategies

## Purpose

Framework for deciding which test gaps to address first when resources are limited (they always are).

## When You Need This

- Faced with hundreds of untested files
- Need to justify test investment to stakeholders
- Planning sprint test improvement work
- Balancing test debt with feature delivery

---

## The Fundamental Problem

**You can't test everything at once.**

- Legacy codebases may have 50%+ untested code
- Writing tests takes time (30-50% of development effort)
- Test maintenance has ongoing cost
- Some code may never need tests (deprecated, low-risk)

**Solution**: Prioritize ruthlessly based on risk and value.

---

## Risk-Based Prioritization Framework

### Four Risk Dimensions

Prioritize files based on:

1. **Criticality**: Impact if code fails
2. **Complexity**: Difficulty of understanding/modifying
3. **Churn**: Frequency of changes
4. **Exposure**: Public API vs. internal implementation

### Risk Score Formula

```
Risk Score = (Criticality × 40) + (Complexity × 20) + (Churn × 30) + (Exposure × 10)
```

**Weights Explained**:
- Criticality (40%): Most important—critical code failure has severe consequences
- Churn (30%): High churn without tests = repeatedly introducing risk
- Complexity (20%): Complex code more likely to have bugs
- Exposure (10%): Public APIs affect more users, but internal code can be critical too

### Scoring Each Dimension

#### 1. Criticality (0-10)

| Score | Category | Examples |
|-------|----------|----------|
| 10 | Mission-Critical | Authentication, authorization, payment processing, data integrity |
| 8-9 | Core Business Logic | Order processing, inventory management, core algorithms |
| 6-7 | Important Features | User profiles, notifications, reporting |
| 4-5 | Secondary Features | UI enhancements, analytics, logging |
| 2-3 | Utilities | Helpers, formatters, converters |
| 0-1 | Trivial | Generated code, deprecated code, constants |

**How to Assess**:
- Ask: "What happens if this code breaks in production?"
- Consider: Data loss, security breach, revenue impact, user experience

#### 2. Complexity (0-10)

Use **cyclomatic complexity** or manual assessment:

| Score | Cyclomatic Complexity | Lines of Code | Characteristics |
|-------|----------------------|---------------|-----------------|
| 10 | >50 | >500 | Deeply nested, many paths, hard to understand |
| 8-9 | 25-50 | 300-500 | Multiple branches, loops, complex logic |
| 6-7 | 15-25 | 150-300 | Moderate branching, some complexity |
| 4-5 | 10-15 | 75-150 | Simple branching, straightforward |
| 2-3 | 5-10 | 25-75 | Linear logic, minimal branching |
| 0-1 | 1-5 | <25 | Trivial (getters, setters, simple functions) |

**Measuring Cyclomatic Complexity**:

```bash
# For C++ (requires lizard or similar)
pip install lizard
lizard src/ --CCN 15  # Show functions with complexity > 15

# For Python
pip install radon
radon cc src/ -s  # Show complexity scores
```

**Manual Assessment**:
- Count decision points: if, for, while, case, catch, &&, ||
- Complexity = decision points + 1

#### 3. Churn (0-10)

Based on number of commits modifying file in recent period (e.g., 6 months):

| Score | Commits (6mo) | Interpretation |
|-------|---------------|----------------|
| 10 | >30 | Extremely volatile, changed weekly |
| 8-9 | 20-30 | Very high churn, frequent changes |
| 6-7 | 10-20 | High churn, regular modifications |
| 4-5 | 5-10 | Moderate churn, occasional changes |
| 2-3 | 2-5 | Low churn, rarely changed |
| 0-1 | 0-1 | Stable, unchanged or nearly unchanged |

**Measuring Churn**:

```bash
# Commits per file in last 6 months
git log --since="6 months ago" --name-only --format='' | \
  sort | uniq -c | sort -rn
```

#### 4. Exposure (0-10)

Based on how many other components depend on this code:

| Score | Exposure Level | Examples |
|-------|----------------|----------|
| 10 | Public API | External APIs, published libraries |
| 8-9 | Internal API | Core libraries used by many modules |
| 6-7 | Module Public | Public within a module, used by multiple files |
| 4-5 | Package Public | Used within one package/namespace |
| 2-3 | File Private | Private methods, used within one file |
| 0-1 | Unused | Dead code, deprecated APIs |

**Measuring Exposure**:

```bash
# For C++: Count includes of this header
grep -r "#include \"myheader.h\"" src/ | wc -l

# For Python: Count imports (requires grep-ast or similar)
grep -r "from mymodule import" . | wc -l
```

---

## Prioritization Examples

### Example 1: High-Priority Gap

**File**: `src/auth/session_manager.cpp`

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Criticality | 10 | Authentication - security critical |
| Complexity | 7 | 300 lines, multiple auth flows |
| Churn | 8 | 22 commits in 6 months (frequent bugs) |
| Exposure | 9 | Used by all API endpoints |
| **Risk Score** | **870** | **Immediate priority** |

**Action**: Write comprehensive tests ASAP. This is a ticking time bomb.

### Example 2: Medium-Priority Gap

**File**: `src/utils/date_formatter.cpp`

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Criticality | 4 | Formatting utility, not critical |
| Complexity | 3 | Simple formatting logic |
| Churn | 6 | 12 commits (timezone bugs) |
| Exposure | 7 | Used by many modules |
| **Risk Score** | **350** | **Medium priority** |

**Action**: Add tests during next sprint. Not urgent but worth doing.

### Example 3: Low-Priority Gap

**File**: `src/legacy/old_parser.cpp`

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Criticality | 2 | Deprecated, being replaced |
| Complexity | 8 | Complex but rarely used |
| Churn | 1 | Unchanged for 2 years |
| Exposure | 2 | One legacy feature only |
| **Risk Score** | **180** | **Low priority** |

**Action**: Don't test. Schedule for deletion instead.

---

## Decision Matrix

Based on risk score:

| Risk Score | Priority | Action |
|------------|----------|--------|
| >700 | **CRITICAL** | Test immediately, block releases if needed |
| 500-700 | **High** | Test within current sprint |
| 300-500 | **Medium** | Backlog, test within 1-2 sprints |
| 150-300 | **Low** | Test opportunistically (when touching code) |
| <150 | **Defer** | Don't test unless circumstances change |

---

## Special Cases & Adjustments

### Adjustment 1: Recent Bugs

**If file has bug fixes in last 3 months:**
- Add +100 to risk score per recent bug
- Recent bugs indicate insufficient testing

### Adjustment 2: Code You're About to Change

**If file is on your current sprint plan:**
- Increase priority regardless of score
- Write tests BEFORE modifying (characterization tests)
- Prevents regressions from your changes

### Adjustment 3: Code You Just Wrote

**If file is <1 month old:**
- Increase priority by 2 levels
- Easiest to test while fresh in mind
- Establishes good habits

### Adjustment 4: Mutation Testing Results

**If mutation score is low (<60%) despite coverage:**
- Increase priority by 1 level
- Existing tests are weak, need improvement
- See `tooling_integration.md` for mutation testing

### Adjustment 5: Customer-Facing vs. Internal

**Customer-facing code:**
- Multiply criticality score by 1.5
- User-visible bugs have reputation impact

**Internal tooling:**
- Reduce criticality score by 25%
- Developer inconvenience < customer impact

---

## Prioritization Workflows

### Workflow 1: Sprint Planning

**Input**: List of all test gaps (from gap analysis)

**Process**:
1. Calculate risk scores for all gaps
2. Sort by risk score descending
3. Filter to top 20%
4. Estimate test effort for each
5. Select gaps that fit sprint capacity
6. Commit to completing selected gaps

**Output**: Sprint backlog with test improvement tasks

### Workflow 2: Opportunistic Testing

**Trigger**: Developer needs to modify file X

**Process**:
1. Check if file X has test coverage
2. If coverage < 60%:
   - Write characterization tests first (test current behavior)
   - Then make changes
   - Verify tests still pass (or update for intentional changes)
3. If coverage ≥ 60%:
   - Ensure diff-coverage for your changes meets threshold

**Output**: Incremental improvement with each feature

### Workflow 3: Bug-Driven Testing

**Trigger**: Bug reported in production

**Process**:
1. Write failing test reproducing bug
2. Fix bug, verify test passes
3. Analyze file's overall coverage
4. If coverage < 60%:
   - Calculate risk score
   - If high-risk: add to sprint backlog for comprehensive testing
   - If low-risk: mark as "has basic coverage, monitor"

**Output**: Regression prevention + strategic gap closure

### Workflow 4: Quarterly Cleanup

**Frequency**: Once per quarter

**Process**:
1. Re-run full gap analysis (scores may have changed)
2. Identify files that moved into high-risk category
3. Dedicate 1-2 sprint to test debt reduction
4. Focus on files crossing risk thresholds
5. Document progress and trend

**Output**: Prevent gap accumulation, strategic debt management

---

## Resource Allocation Strategies

### Strategy 1: Fixed Percentage

**Allocation**: 15-20% of sprint capacity to test improvement

**Pros**:
- Predictable, sustainable
- Prevents complete neglect
- Management can plan around it

**Cons**:
- May not address urgent gaps quickly
- Can feel arbitrary

### Strategy 2: Risk-Threshold Triggered

**Allocation**: Variable based on high-risk gap count

**Rules**:
- If 0-5 critical gaps: 10% capacity
- If 6-15 critical gaps: 20% capacity
- If 16+ critical gaps: 30% capacity (code red)

**Pros**:
- Responsive to actual risk
- Self-balancing (effort reduces gaps, reduces allocation)

**Cons**:
- Unpredictable sprint capacity

### Strategy 3: Boy Scout Rule

**Allocation**: No fixed capacity, embedded in feature work

**Rule**: "Leave code better than you found it"
- When touching file, add tests if missing
- Build test time into feature estimates
- No separate "test improvement" tasks

**Pros**:
- No explicit "test debt" work
- Tests aligned with active development

**Cons**:
- Doesn't address untouched high-risk code
- Requires discipline

**Recommendation**: Combine strategies
- Boy Scout Rule for opportunistic improvement
- 10% fixed capacity for strategic gap closure
- Risk-threshold for crisis response

---

## When NOT to Write Tests

### Acceptable Reasons to Skip Testing

1. **Code Will Be Deleted Soon**
   - Deprecated features scheduled for removal
   - Prototypes being replaced
   - Don't invest in dying code

2. **Code is Trivial and Stable**
   - Simple getters/setters
   - Constants and configuration
   - Hasn't changed in 2+ years and won't

3. **Testing Cost Exceeds Value**
   - Code is so poorly designed it would require full rewrite to test
   - Better to refactor or replace than test current version
   - Accept the risk for now, plan replacement

4. **Covered by Higher-Level Tests**
   - Integration tests already verify behavior
   - End-to-end tests cover the flow
   - Redundant unit tests add maintenance burden

5. **Generated or Template Code**
   - Auto-generated from tools
   - Can be regenerated at any time
   - Test the generator, not the generated code

### Documenting the Decision

When skipping tests on high-risk code:

```cpp
// TEST_EXEMPTION: No unit tests for this file
// Reason: Deprecated, scheduled for removal in v3.0
// Alternate coverage: Integration tests in test/e2e/legacy_flow_test.cpp
// Risk acceptance: Approved by Tech Lead (2025-11-03)
// Review date: 2025-12-01 (remove file or re-evaluate)
```

---

## Metrics for Tracking Progress

### Leading Indicators (Predict Future Quality)

1. **Diff Coverage Trend**: Increasing = preventing new gaps
2. **Test:Production Ratio**: ≥1:1 = sustainable
3. **High-Risk Gap Count**: Decreasing = addressing worst areas

### Lagging Indicators (Measure Past Outcomes)

1. **Production Bugs**: Decreasing = tests are effective
2. **Time to Fix Bugs**: Decreasing = easier debugging with tests
3. **Regression Rate**: Decreasing = tests prevent reintroduction

### Tracking Dashboard

```
Quarter: Q4 2025

Leading Indicators:
- Diff Coverage (3mo avg):     78% (↑ from 65%)
- Test:Prod Ratio (3mo):        1.2:1 (↑ from 0.9:1)
- Critical Gaps (>700 score):   3 (↓ from 12)

Lagging Indicators:
- Prod Bugs (last month):       4 (↓ from 9)
- Avg Time to Fix:              2.3 hrs (↓ from 5.1 hrs)
- Regressions (last month):     1 (↓ from 4)

Trend: ✓ Improving
```

---

## Common Pitfalls

### Pitfall 1: Testing for Coverage, Not Value

**Symptom**: 90% coverage but bugs still slip through

**Cause**: Tests execute code but don't assert meaningful behavior

**Solution**: Review test quality, use mutation testing

### Pitfall 2: Neglecting Integration Tests

**Symptom**: All units tested, but integration bugs in production

**Cause**: Only unit tests, missing integration/E2E coverage

**Solution**: Balance test pyramid (unit, integration, E2E)

### Pitfall 3: Analysis Paralysis

**Symptom**: Endless gap analysis, no tests written

**Cause**: Trying to perfect prioritization instead of starting

**Solution**: Use 80/20 rule—quickly identify top 20%, start testing

### Pitfall 4: Ignoring Maintenance Cost

**Symptom**: Test suite so large it slows development

**Cause**: Testing everything equally, including low-risk code

**Solution**: Periodically review and delete low-value tests

---

## Prioritization Script

```bash
#!/bin/bash
# Automated risk scoring for all production files

OUTPUT_FILE="test_gap_priority_report.csv"

echo "File,Criticality,Complexity,Churn,Exposure,Risk_Score" > "$OUTPUT_FILE"

find src/ -name "*.cpp" | while read file; do
    # Complexity (using lizard if available, else LOC proxy)
    if command -v lizard >/dev/null; then
        complexity=$(lizard "$file" -m | awk 'NR==3 {print $1}')
        complexity=$(echo "scale=0; $complexity / 5" | bc)  # Normalize to 0-10
    else
        loc=$(wc -l < "$file")
        complexity=$(echo "scale=0; $loc / 50" | bc)  # Rough proxy
    fi
    complexity=$(( complexity > 10 ? 10 : complexity ))

    # Churn (commits in last 6 months)
    commits=$(git log --since="6 months ago" --oneline -- "$file" | wc -l)
    churn=$(echo "scale=0; $commits / 3" | bc)  # 3 commits/mo = score of 2
    churn=$(( churn > 10 ? 10 : churn ))

    # Exposure (files that include this one)
    header=$(echo "$file" | sed 's/\.cpp$/.h/')
    if [ -f "$header" ]; then
        includes=$(grep -r "#include.*$(basename "$header")" src/ | wc -l)
        exposure=$(echo "scale=0; $includes / 2" | bc)
        exposure=$(( exposure > 10 ? 10 : exposure ))
    else
        exposure=3  # Default for .cpp without header
    fi

    # Criticality (requires manual tagging or keyword heuristics)
    # This is a placeholder - adapt to your codebase
    if echo "$file" | grep -qE "auth|security|payment|crypto"; then
        criticality=10
    elif echo "$file" | grep -qE "core|engine|processor"; then
        criticality=8
    elif echo "$file" | grep -qE "util|helper|common"; then
        criticality=3
    else
        criticality=5  # Default
    fi

    # Calculate risk score
    risk=$(( (criticality * 40) + (complexity * 20) + (churn * 30) + (exposure * 10) ))

    echo "$file,$criticality,$complexity,$churn,$exposure,$risk" >> "$OUTPUT_FILE"
done

# Sort by risk score and display top 20
echo ""
echo "Top 20 Highest Risk Files:"
sort -t',' -k6 -rn "$OUTPUT_FILE" | head -21 | column -t -s','
```

---

## Related Resources

- **`fundamentals.md`**: Why prioritization matters (limited resources)
- **`git_analysis_patterns.md`**: Measuring churn and bug frequency
- **`diff_cover_workflow.md`**: Preventing new gaps while addressing old ones
- **`tooling_integration.md`**: Measuring complexity and mutation scores
