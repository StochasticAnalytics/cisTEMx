# analyze-test-coverage-gaps Skill

**Version**: 1.0.0
**Created**: November 3, 2025
**Status**: Production-ready

## Overview

This skill provides systematic identification of test coverage gaps using git history analysis and coverage tooling. It combines multiple gap detection strategies to help prioritize test improvements.

## Quick Start

1. **Read the main skill**: Start with `SKILL.md` for overview
2. **Run automated analysis**: Use `scripts/analyze_coverage_gaps.sh`
3. **Prioritize gaps**: Use `scripts/find_fragile_code.py` for risk scoring
4. **Follow workflow**: See `templates/gap_assessment_workflow.md` for comprehensive process

## Structure

```
analyze-test-coverage-gaps/
├── SKILL.md                          # Main entry point (179 lines)
├── resources/                        # Detailed documentation
│   ├── fundamentals.md              # Core concepts (319 lines)
│   ├── diff_cover_workflow.md       # Incremental coverage (619 lines)
│   ├── git_analysis_patterns.md     # Git queries (633 lines)
│   ├── prioritization_strategies.md # Risk-based prioritization (541 lines)
│   ├── tooling_integration.md       # Coverage tools setup (830 lines)
│   └── citations.md                 # All sources (370 lines)
├── scripts/                          # Automation
│   ├── analyze_coverage_gaps.sh     # Master analysis script (304 lines)
│   └── find_fragile_code.py         # Risk scoring (350 lines)
└── templates/                        # Workflows
    └── gap_assessment_workflow.md   # Step-by-step checklist (460 lines)

Total: 4,605 lines
```

## Key Features

### Four Gap Detection Categories

1. **Ratio Analysis**: Test-to-production change ratios
2. **Incremental Gaps**: Code changes without test updates
3. **Legacy Gaps**: Files never associated with tests
4. **Quality Gaps**: Files with repeated bugs despite tests

### Comprehensive Resources

- **fundamentals.md**: Core concepts, 5x defect multiplier, coverage vs. quality
- **diff_cover_workflow.md**: Using diff-cover for "if you touch it, test it"
- **git_analysis_patterns.md**: Git queries for finding untested code
- **prioritization_strategies.md**: Risk scoring framework (criticality × complexity × churn × exposure)
- **tooling_integration.md**: Setup for C++/gcov, Python/coverage.py, Java/JaCoCo, mutation testing
- **citations.md**: All 23 sources with access dates

### Practical Tools

- **analyze_coverage_gaps.sh**: Automated gap analysis across all categories
- **find_fragile_code.py**: Risk scoring combining coverage + bugs + churn
- **gap_assessment_workflow.md**: Complete workflow checklist for systematic analysis

## Key Insights

- **5x Defect Multiplier**: Untested code changes are 5× more likely to contain bugs (Teamscale)
- **Incremental Success**: edX went from <50% → 87% coverage in 10 months using diff-cover
- **Risk-Based**: Prioritize high-criticality, high-churn, low-coverage code first
- **Sustainable**: Enforce diff-coverage on new code while slowly improving legacy

## Usage Patterns

### Quick Gap Check
```bash
./scripts/analyze_coverage_gaps.sh "1 month ago"
```

### Comprehensive Analysis with Risk Scoring
```bash
# Generate coverage
lcov --capture --directory . --output-file coverage.info

# Run full analysis
./scripts/analyze_coverage_gaps.sh "3 months ago" coverage.info

# Get prioritized list
python3 scripts/find_fragile_code.py coverage.info --top 30
```

### Systematic Assessment
Follow `templates/gap_assessment_workflow.md` for complete process including:
- Coverage generation
- Automated analysis
- Manual investigation
- Prioritization
- Action planning
- CI/CD setup

## Integration with cisTEMx

This skill works with:
- **unit-testing**: Write tests for identified gaps
- **find-bug-introduction**: Cross-reference bugs with coverage
- **identify-refactoring-targets**: High-complexity + low-coverage candidates
- **compile-code**: Ensure coverage builds work
- **git-commit**: Commit discipline for test changes

## Research Foundation

Built on comprehensive research from:
- diff-cover (Bachmann1234)
- Open edX case study
- Teamscale test gap analysis
- Microsoft Azure DevOps churn analysis
- Jellyfish code churn metrics
- Martin Fowler on Test Impact Analysis
- gcov/lcov, coverage.py, JaCoCo documentation
- Mutation testing (mutmut, PIT, mull)

See `resources/citations.md` for complete source list.

## Maintenance

**Review Frequency**: Quarterly

**Check For**:
- diff-cover updates
- Coverage tool version changes
- New CI/CD platform features
- Emerging testing tools
- Updated research on test effectiveness

**Maintenance Owner**: lab-tech-blue (via skill-review)

## Success Metrics

The skill is working if:
- Gaps are identified systematically
- Prioritization is risk-based and actionable
- Team can set up diff-coverage enforcement
- Coverage trends improve over time
- High-risk areas are addressed first

## Version History

**v1.0.0** (2025-11-03)
- Initial creation using skill-builder methodology
- 6 comprehensive resources (3,312 lines)
- 2 automation scripts (654 lines)
- 1 workflow template (460 lines)
- 23 research sources integrated
- Production-ready for cisTEMx
