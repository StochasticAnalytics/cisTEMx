---
name: analyze-test-coverage-gaps
description: Identify test coverage gaps by analyzing git history, test-to-production ratios, untested changes, and files that break repeatedly. Use when assessing test debt, planning test improvements, or investigating quality issues. Combines git analysis with coverage tools (diff-cover, gcov/lcov, coverage.py) to find high-risk untested code.
tags: [testing, coverage, diff-cover, test-gaps, quality]
version: 1.0.0
author: Claude (cisTEMx)
status: BROKEN - DO NOT USE. Scripts have critical bugs including incorrect git log format parsing and hard-coded directory assumptions that don't match cisTEMx structure.
---

# Analyze Test Coverage Gaps

Systematic identification of test coverage gaps using git history analysis and coverage tooling.

## Overview

This skill helps identify where tests are missing or inadequate by combining:
- **Git history analysis**: Track test-to-production change ratios, files modified without tests
- **Coverage tools**: diff-cover (incremental), gcov/lcov (C++), coverage.py (Python)
- **Bug correlation**: Find files that break repeatedly despite having some tests
- **Prioritization frameworks**: Focus efforts on highest-risk gaps

**Key Insight**: Untested code changes are **5x more likely** to contain defects than tested code.

## When to Use This Skill

Use this skill when you need to:
- Assess overall test coverage health and identify gaps
- Find files changed without corresponding test updates
- Identify code that has never been tested
- Locate files with repeated bugs indicating insufficient coverage
- Plan test improvement initiatives
- Investigate quality issues or prepare for releases
- Set up diff-coverage enforcement in CI/CD

## Quick Start

### 1. Assess Test-to-Production Change Ratio
```bash
# See if tests are keeping pace with development
git log --stat --since="1 month ago" -- src/ | \
  grep -E "^ (.*)\|" | \
  awk '{added+=$(NF-3); deleted+=$(NF-1)} END {printf "Production: +%d -%d\n", added, deleted}'

git log --stat --since="1 month ago" -- test/ | \
  grep -E "^ (.*)\|" | \
  awk '{added+=$(NF-3); deleted+=$(NF-1)} END {printf "Tests: +%d -%d\n", added, deleted}'
```

**Healthy**: Test changes ≥ production changes (1:1 or better)
**Warning**: Ratio below 1:0.8
**Critical**: Ratio below 1:0.5

### 2. Find Recent Changes Without Test Updates
```bash
# List commits that changed src/ but not test/
git log --name-only --format="%H|%an|%ad|%s" --since="1 month ago" | \
awk '/\|/ {commit=$0; has_prod=0; has_test=0; next}
     /^src\// {has_prod=1}
     /^test\// {has_test=1}
     /^$/ {if (has_prod && !has_test) print commit; has_prod=0; has_test=0}'
```

### 3. Run diff-cover for Incremental Coverage
```bash
# Check coverage of changes vs. main branch
# (Requires coverage report - see tooling_integration.md)
diff-cover coverage.xml --compare-branch=origin/main --fail-under=80
```

## Core Gap Categories

1. **Ratio Imbalance**: Production code growing faster than tests
2. **Missing Test Updates**: Files changed without test modifications
3. **Never-Tested Files**: Code that has never had associated test commits
4. **Repeatedly Breaking Files**: Files with multiple bug fixes (weak tests)

## Available Resources

### Foundational Knowledge
- **`resources/fundamentals.md`**: Core concepts, terminology, why gaps matter
  - Code churn, test gap analysis, diff coverage
  - The 5x defect multiplier for untested changes
  - Test effectiveness vs. test existence

### Practical Workflows
- **`resources/diff_cover_workflow.md`**: Using diff-cover for incremental coverage
  - Installation and setup
  - Integration with C++/gcov, Python/coverage.py
  - CI/CD enforcement patterns
  - Real-world case: edX's 50% → 87% improvement

- **`resources/git_analysis_patterns.md`**: Git queries for finding gaps
  - Test-to-production ratio analysis
  - Finding untested commits and files
  - Identifying high-churn code
  - Cross-referencing with coverage data

- **`resources/prioritization_strategies.md`**: How to prioritize which gaps to address
  - Risk scoring: criticality × complexity × churn × exposure
  - Prioritization frameworks
  - When to write tests vs. refactor vs. accept risk

- **`resources/tooling_integration.md`**: Coverage tools and integration
  - C++: gcov + lcov workflow
  - Python: coverage.py
  - Java: JaCoCo
  - Mutation testing for test quality assessment

### Reference
- **`resources/citations.md`**: All sources, tools, and references

## Available Scripts

### Automated Analysis
- **`scripts/analyze_coverage_gaps.sh`**: Comprehensive gap analysis
  - Runs all four gap detection patterns
  - Generates prioritized report
  - Integrates with coverage tools if available

- **`scripts/find_fragile_code.py`**: Correlate coverage with bug history
  - Identifies files with low coverage AND high bug counts
  - Risk scoring algorithm
  - Produces prioritized remediation list

## Available Templates

- **`templates/gap_assessment_workflow.md`**: Step-by-step checklist
  - Systematic gap assessment process
  - Decision points and thresholds
  - Documentation template for findings

## Typical Workflow

1. **Run automated analysis** using `scripts/analyze_coverage_gaps.sh`
2. **Review four gap categories**: ratio, missing updates, never-tested, repeatedly breaking
3. **Prioritize** using risk scoring (see `resources/prioritization_strategies.md`)
4. **Set up diff-cover** for ongoing enforcement (see `resources/diff_cover_workflow.md`)
5. **Address gaps** incrementally with each feature/fix
6. **Monitor trends** over time to prevent regression

## Integration with Other Skills

- **unit-testing**: Use to write tests for identified gaps
- **find-bug-introduction**: Cross-reference bugs with coverage gaps
- **identify-refactoring-targets**: High-complexity, low-coverage code may need refactoring
- **compile-code**: Ensure tests compile and run after additions

## Key Metrics & Thresholds

From research synthesis:

- **Diff Coverage**: ≥80% on changes (strict), ≥60% (minimum viable)
- **Test-to-Production Ratio**: ≥1:1 (ideal), ≥1:0.8 (acceptable)
- **Overall Coverage**: 70-80% for mature projects, focus on critical paths
- **Zero Coverage Files**: Should be < 10% of codebase
- **Bug-to-Test Correlation**: Files with ≥3 bugs in 6 months need test review

## Getting Started

If you're new to test gap analysis:
1. Start with **`resources/fundamentals.md`** to understand core concepts
2. Run **`scripts/analyze_coverage_gaps.sh`** to get baseline assessment
3. Review **`resources/prioritization_strategies.md`** to plan improvements
4. Set up **diff-cover** using **`resources/diff_cover_workflow.md`**

If you know what you're looking for:
- Incremental coverage → `resources/diff_cover_workflow.md`
- Custom git queries → `resources/git_analysis_patterns.md`
- Tool setup → `resources/tooling_integration.md`
- Risk assessment → `scripts/find_fragile_code.py`

## Warning: Coverage is Necessary but Not Sufficient

**High coverage ≠ good tests.** Tests must:
- Assert meaningful behavior (not just execute code)
- Cover edge cases and error paths
- Test integration points
- Be maintained as code evolves

For test quality assessment beyond coverage, consider mutation testing (see `resources/tooling_integration.md`).
