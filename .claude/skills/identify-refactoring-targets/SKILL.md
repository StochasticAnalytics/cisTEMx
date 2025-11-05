---
name: identify-refactoring-targets
description: Identify code needing refactoring through churn analysis, hotspot detection, and temporal coupling. Use when prioritizing technical debt, planning refactoring sprints, or investigating maintenance burden. Combines git history with complexity metrics to find high-ROI refactoring targets using data-driven approach.
---

# Identify Refactoring Targets

Use this skill when you need to **identify which code should be refactored** based on empirical evidence from version control history.

## When to Use

✅ **Use this skill when:**
- Planning refactoring sprints or technical debt reduction
- Code quality issues but unclear where to focus
- Need to justify refactoring with data
- Investigating why development is slow
- Prioritizing maintenance work by ROI

❌ **Don't use this skill when:**
- Already know what needs refactoring
- No git history (new codebase)
- Code is stable and working fine
- Need immediate bug fixes (not strategic planning)

## Core Concept

**Not all code matters equally.** The Pareto principle applies: 20% of code causes 80% of problems.

**Hotspot**: Code that is both **complex** AND **frequently modified**. These represent highest-value refactoring targets.

```
High │         │ HOTSPOTS  │
Cmplx│  Skip   │ PRIORITY  │
     │         │  FOCUS    │
     ├─────────┼───────────┤
Low  │  OK     │  OK       │
Cmplx│         │           │
     └─────────┴───────────┘
       Low      High
      Change   Change
     Frequency Frequency
```

## Quick Start

```bash
# 1. Find most-changed files (past year)
git log --format=format: --name-only --since=12.month \
  | egrep -v '^$' | sort | uniq -c | sort -nr | head -20

# 2. Check complexity of top files
lizard --csv src/ > complexity.csv

# 3. Combine metrics to identify hotspots
# (high churn + high complexity = priority)
```

**See** `templates/refactoring_assessment.md` for complete workflow.

## Three Core Analyses

### 1. Churn Analysis
**What**: Identify files that change most frequently
**Why**: High churn indicates instability, poor design, or missing abstractions
**See**: `resources/churn_analysis.md`

### 2. Hotspot Analysis
**What**: Combine churn with complexity to find critical areas
**Why**: Complex code that changes often costs most money
**See**: `resources/hotspot_analysis.md`

### 3. Temporal Coupling
**What**: Find files that change together
**Why**: Reveals hidden dependencies and architectural issues
**See**: `resources/temporal_coupling.md`

## Progressive Resources

Start here, go deeper as needed:

1. **`resources/fundamentals.md`** - Core concepts: churn, hotspots, ROI framework
2. **`resources/churn_analysis.md`** - Measuring and interpreting code volatility
3. **`resources/hotspot_analysis.md`** - Combining churn + complexity for prioritization
4. **`resources/temporal_coupling.md`** - Finding hidden dependencies
5. **`resources/code_maat_guide.md`** - Using Code Maat tool for advanced analysis
6. **`resources/practical_workflow.md`** - Step-by-step implementation guide

## Common Scenarios

### Scenario 1: Quick Hotspot Check
```bash
# Run automated analysis
./scripts/analyze_churn.sh /path/to/repo 2024-01-01

# Review top 10 results
# Focus refactoring on highest scores
```

### Scenario 2: Comprehensive Analysis
```bash
# Generate full metrics
./scripts/identify_hotspots.sh /path/to/repo

# Creates: churn.csv, complexity.csv, hotspots.csv
# Visualize in spreadsheet tool
```

### Scenario 3: Architectural Review
```bash
# Find coupled modules
java -jar code-maat.jar -a coupling -l git.log

# Identify coupling across component boundaries
# Plan architectural improvements
```

**See** `resources/practical_workflow.md` for complete examples.

## Key Tools

- **Git log**: Built-in churn analysis
- **Code Maat**: Comprehensive VCS analysis (coupling, ownership, communication)
- **Lizard**: Multi-language complexity analysis
- **GitNStats**: Simple churn visualization

**See** `resources/code_maat_guide.md` for tool setup and usage.

## Key Best Practices

✅ **Do:**
- Use 6-12 months of history for analysis
- Combine multiple metrics (churn + complexity)
- Focus on top 10-20 hotspots
- Validate with developer experience
- Track metrics before/after refactoring

❌ **Don't:**
- Refactor stable code just because it's complex
- Refactor simple code just because it changes often
- Trust churn alone without complexity context
- Ignore business priorities for pure metrics
- Refactor without test coverage

## Expected Outcomes

After using this skill:
- Identify top 10-20 files causing most maintenance burden
- Quantify ROI for refactoring decisions
- Build data-driven refactoring backlog
- Align team on priorities with evidence
- Track improvement over time

## Troubleshooting

**"Git log shows too many changes"**: See `resources/churn_analysis.md` § Filtering Noise

**"Don't have Code Maat installed"**: See `resources/code_maat_guide.md` § Installation

**"Complexity tool doesn't support my language"**: See `resources/hotspot_analysis.md` § Alternative Metrics

**"Results don't match developer experience"**: See `resources/fundamentals.md` § Validation

## Citations

All sources documented in `resources/citations.md`.
