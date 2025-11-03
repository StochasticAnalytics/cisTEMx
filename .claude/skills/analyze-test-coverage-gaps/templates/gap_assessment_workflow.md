# Test Coverage Gap Assessment Workflow

## Purpose

Step-by-step checklist for conducting systematic test coverage gap analysis. Use this workflow when planning test improvements, preparing for releases, or addressing quality concerns.

---

## When to Use This Workflow

- **Sprint Planning**: Quarterly or monthly gap assessment
- **Release Preparation**: Verify adequate coverage before major releases
- **Quality Review**: Investigating elevated defect rates
- **Onboarding**: Understanding test landscape in new codebase
- **Technical Debt**: Planning test improvement initiatives

---

## Prerequisites

- [ ] Access to git repository with commit history
- [ ] Coverage tool installed (gcov/lcov, coverage.py, JaCoCo, etc.)
- [ ] diff-cover installed (`pip install diff_cover`)
- [ ] Ability to build and run tests
- [ ] ~2-4 hours allocated for comprehensive analysis

---

## Phase 1: Preparation (15 minutes)

### 1.1 Define Scope

- [ ] **Time Period**: How far back to analyze?
  - Recommended: 3-6 months for ongoing projects
  - New teams: Start with 1 month
  - Legacy rescue: 6-12 months to see patterns

- [ ] **Directories**: Which code to include?
  - Production code: `src/`, `lib/`, `app/`
  - Test code: `test/`, `tests/`, `spec/`
  - Exclude: `vendor/`, `third_party/`, `generated/`

- [ ] **Metrics to Track**:
  - [ ] Test-to-production ratio
  - [ ] Commits without test changes
  - [ ] High-churn files
  - [ ] Files with repeated bugs
  - [ ] Zero/low coverage files

### 1.2 Document Current State

```
Analysis Date: [YYYY-MM-DD]
Period: [e.g., "Last 6 months"]
Analyst: [Your name]
Purpose: [e.g., "Q4 test debt assessment"]

Current Metrics (baseline):
- Overall coverage: [X%]
- Files with tests: [X / Y]
- Recent test commits: [X]
- Recent prod commits: [Y]
```

---

## Phase 2: Generate Coverage Data (30 minutes)

### 2.1 Build with Coverage

**For C++ (gcov/lcov)**:
```bash
# Clean previous coverage
find . -name "*.gcda" -delete

# Build with coverage
mkdir -p build-coverage
cd build-coverage
cmake -DENABLE_COVERAGE=ON ..
make -j$(nproc)

# Run tests
ctest --output-on-failure

# Generate coverage
lcov --capture --directory . --output-file coverage.info
lcov --remove coverage.info '/usr/*' '*/test/*' --output-file coverage_filtered.info
```

**For Python**:
```bash
coverage run -m pytest
coverage xml  # For diff-cover
coverage html # For human review
```

**For Java**:
```bash
mvn clean test
# Coverage at: target/site/jacoco/jacoco.xml
```

### 2.2 Verify Coverage Data

- [ ] Coverage file generated successfully
- [ ] File size > 0 (not empty)
- [ ] Contains expected source files
- [ ] Check sample: `lcov --list coverage.info | head -20`

---

## Phase 3: Automated Gap Analysis (30 minutes)

### 3.1 Run Analysis Scripts

**Master Analysis Script**:
```bash
cd /path/to/project
./scripts/analyze_coverage_gaps.sh "6 months ago" build-coverage/coverage.info
```

- [ ] Script completed without errors
- [ ] Review output sections:
  - Test-to-production ratio
  - Untested commits
  - High-churn files
  - Bug-fix frequency

### 3.2 Run Risk Scoring

```bash
python3 scripts/find_fragile_code.py build-coverage/coverage.info \
  --since "6 months ago" \
  --top 30 \
  --format table > risk_report.txt
```

- [ ] Risk report generated
- [ ] Review top 10 highest-risk files
- [ ] Note any surprises or expected entries

### 3.3 Run diff-cover (for recent changes)

```bash
diff-cover coverage.xml --compare-branch=origin/main --html-report=diff-coverage.html
```

- [ ] Diff coverage percentage calculated
- [ ] HTML report available for review
- [ ] Identify specific uncovered lines in recent changes

---

## Phase 4: Manual Investigation (45 minutes)

### 4.1 Review High-Risk Files

For each file in top 10 risk list:

**File**: `_________________________`

- [ ] **Coverage**: ______%
- [ ] **Bugs (6mo)**: ______
- [ ] **Churn (6mo)**: ______ commits
- [ ] **Criticality**: Low / Medium / High / Critical

**Questions**:
1. Why is coverage low?
   - [ ] Legacy code (pre-test era)
   - [ ] Difficult to test (high coupling)
   - [ ] Deemed low priority
   - [ ] Tests exist but aren't measured

2. Why repeated bugs?
   - [ ] Complex logic with edge cases
   - [ ] Tests don't cover failure modes
   - [ ] Integration issues not caught by unit tests
   - [ ] External dependencies

3. Action needed?
   - [ ] Write unit tests
   - [ ] Add integration tests
   - [ ] Refactor for testability
   - [ ] Accept risk (document why)
   - [ ] Schedule for replacement

### 4.2 Identify Patterns

Look for common themes across gaps:

- [ ] **Module-level gaps**: Entire modules under-tested?
- [ ] **Feature-level gaps**: Specific features lacking tests?
- [ ] **Author-level patterns**: Certain developers not writing tests?
- [ ] **Time-based patterns**: Coverage degraded after specific date?

**Patterns Observed**:
```
1. [e.g., "All auth-related modules have <40% coverage"]
2. [e.g., "Files touched by AuthorX rarely have test updates"]
3. [e.g., "Coverage declined after v2.0 release (deadline pressure)"]
```

### 4.3 Cross-Reference with Known Issues

- [ ] Check bug tracker for related issues
- [ ] Review recent production incidents
- [ ] Consult team about problem areas

**Correlation Found**:
```
- [e.g., "90% of P1 bugs in last quarter from files with <50% coverage"]
- [e.g., "Module X has 3 outages, all in untested code paths"]
```

---

## Phase 5: Prioritization & Planning (30 minutes)

### 5.1 Calculate Priority Scores

For each identified gap, use risk framework:

| File/Module | Criticality | Complexity | Churn | Exposure | Risk Score | Priority |
|-------------|-------------|------------|-------|----------|------------|----------|
| [example]   | 8           | 7          | 9     | 8        | 820        | High     |
|             |             |            |       |          |            |          |

**Priority Levels**:
- **Critical (>700)**: Address immediately
- **High (500-700)**: Current sprint
- **Medium (300-500)**: Next 1-2 sprints
- **Low (150-300)**: Backlog
- **Defer (<150)**: Don't test unless circumstances change

### 5.2 Create Action Plan

**Immediate Actions (This Week)**:
1. [ ] _______________________________________________
2. [ ] _______________________________________________
3. [ ] _______________________________________________

**Short-Term (This Sprint)**:
1. [ ] _______________________________________________
2. [ ] _______________________________________________
3. [ ] _______________________________________________

**Long-Term (Next Quarter)**:
1. [ ] _______________________________________________
2. [ ] _______________________________________________
3. [ ] _______________________________________________

### 5.3 Set Measurable Goals

**3-Month Goals**:
- [ ] Overall coverage: __% → __% (increase of __%)
- [ ] Critical gaps (>700 score): __ → 0
- [ ] Diff coverage on new PRs: ≥__% enforced
- [ ] Test-to-prod ratio: 1:__ → 1:__ (improve to ≥1:0.8)

**6-Month Goals**:
- [ ] Overall coverage: __% → __% (target 70-80%)
- [ ] Zero-coverage files: __% → <10%
- [ ] All critical paths (auth, payments, etc.) at ≥90%
- [ ] Mutation score for critical code: ≥70%

---

## Phase 6: Implementation Setup (30 minutes)

### 6.1 Enable Diff-Coverage Enforcement

- [ ] Install diff-cover in CI pipeline
- [ ] Configure threshold (start at 60%, increase to 80%)
- [ ] Add pre-commit hook (optional)
- [ ] Document process for team

**Configuration**:
```toml
# .diffcover.toml
[tool.diff_cover]
compare_branch = "origin/main"
fail_under = 70.0
html_report = "diff-coverage.html"
```

### 6.2 Set Up Monitoring

- [ ] Weekly/monthly automated gap analysis
- [ ] Coverage trend tracking
- [ ] Dashboard or reports for team visibility

**Cron Job Example**:
```bash
# Weekly gap report
0 9 * * MON /path/to/analyze_coverage_gaps.sh "1 week ago" | mail -s "Weekly Gap Report" team@example.com
```

### 6.3 Team Communication

- [ ] Share findings with team
- [ ] Explain prioritization rationale
- [ ] Set expectations for coverage targets
- [ ] Integrate test work into sprint planning

**Communication Points**:
- Why coverage matters (5x defect multiplier)
- Current state and goals
- Process changes (diff-coverage enforcement)
- Support available (resources, pair programming)

---

## Phase 7: Continuous Improvement (Ongoing)

### 7.1 Regular Reviews

**Weekly** (5 minutes):
- [ ] Check diff-coverage on merged PRs
- [ ] Celebrate improvements
- [ ] Address blockers

**Monthly** (30 minutes):
- [ ] Re-run gap analysis
- [ ] Update risk scores
- [ ] Adjust priorities based on changes

**Quarterly** (2 hours):
- [ ] Comprehensive gap assessment (repeat this workflow)
- [ ] Review progress toward goals
- [ ] Adjust thresholds and targets
- [ ] Update team on trends

### 7.2 Retrospective Questions

After 3 months, reflect:

1. **What worked well?**
   - __________________________________________
   - __________________________________________

2. **What didn't work?**
   - __________________________________________
   - __________________________________________

3. **What surprised us?**
   - __________________________________________
   - __________________________________________

4. **What should we change?**
   - __________________________________________
   - __________________________________________

---

## Deliverables Checklist

At end of workflow, you should have:

- [ ] **Coverage report** (HTML + data files)
- [ ] **Gap analysis report** (from scripts)
- [ ] **Risk-scored file list** (prioritized gaps)
- [ ] **Action plan** with owners and timelines
- [ ] **Configured CI** with diff-coverage enforcement
- [ ] **Team communication** (email, doc, presentation)
- [ ] **Tracking mechanism** (dashboard, spreadsheet, issues)

---

## Templates & Resources

### Gap Analysis Report Template

```markdown
# Test Coverage Gap Analysis Report

**Date**: [YYYY-MM-DD]
**Analyst**: [Name]
**Period**: [Time range analyzed]

## Executive Summary

- Overall coverage: [X%]
- Test-to-production ratio: 1:[X]
- Critical gaps identified: [X]
- High-priority files: [X]

**Key Finding**: [1-2 sentence summary]

## Detailed Findings

### 1. Coverage Metrics
[Paste metrics from analysis scripts]

### 2. High-Risk Files
[Top 10-20 from risk scoring]

### 3. Patterns Observed
[Themes identified in Phase 4]

## Recommendations

**Immediate** (This week):
1. [Action item]
2. [Action item]

**Short-term** (This sprint):
1. [Action item]
2. [Action item]

**Long-term** (This quarter):
1. [Action item]
2. [Action item]

## Appendix

- Full coverage report: [link]
- Risk score CSV: [link]
- diff-coverage report: [link]
```

---

## Success Metrics

You're making progress if:

- [ ] Coverage trend is upward (even slowly)
- [ ] Diff-coverage consistently ≥70% on new PRs
- [ ] High-risk file count decreasing
- [ ] Production bugs decreasing
- [ ] Team proactively writes tests without prompting
- [ ] Test maintenance burden is reasonable (not excessive)

---

## Common Pitfalls to Avoid

1. **Analysis Paralysis**: Don't spend months analyzing. One good analysis → start testing.

2. **Perfection Seeking**: Don't aim for 100%. Aim for steady improvement.

3. **Ignoring Context**: Not all gaps are equal. Prioritize ruthlessly.

4. **One-Time Exercise**: Gap analysis must be periodic, not one-and-done.

5. **Testing for Metrics**: Don't write hollow tests just for coverage numbers.

6. **Neglecting Maintenance**: Review and delete low-value tests periodically.

---

## Related Resources

- `../resources/fundamentals.md`: Why gaps matter
- `../resources/diff_cover_workflow.md`: Setting up incremental coverage
- `../resources/git_analysis_patterns.md`: Git queries for gap detection
- `../resources/prioritization_strategies.md`: Risk-based prioritization
- `../resources/tooling_integration.md`: Coverage tool setup
- `../scripts/analyze_coverage_gaps.sh`: Automated analysis
- `../scripts/find_fragile_code.py`: Risk scoring
