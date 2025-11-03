# Citations and References

## Purpose

Comprehensive list of sources consulted during skill creation. Enables future maintenance and currency checking.

---

## Primary Research Sources

### diff-cover (Core Tool)
**Source**: https://github.com/Bachmann1234/diff_cover
**Accessed**: November 3, 2025
**Relevant Learnings**:
- Gold standard for incremental coverage enforcement
- Supports multiple coverage formats (Cobertura, LCOV, JaCoCo, Clover)
- "If you touch it, test it" principle
- Configurable thresholds and output formats

**Linked Content**: `diff_cover_workflow.md` (entire resource)

### Open edX Case Study
**Source**: https://openedx.org/blog/diff-cover-test-coverage-git-commits/
**Accessed**: November 3, 2025
**Relevant Learnings**:
- Real-world success: 50% → 87% coverage in 10 months
- Diff-coverage enforcement more effective than global coverage targets
- Cultural shift: making coverage achievable within individual PRs
- Proof that incremental approach works

**Linked Content**:
- `fundamentals.md`: Real-world success story
- `diff_cover_workflow.md`: Case study section

### Teamscale Test Gap Analysis
**Source**: https://docs.teamscale.com/reference/test-gap-analysis/
**Accessed**: November 3, 2025
**Relevant Learnings**:
- **5x defect multiplier**: Untested code changes 5× more likely to contain bugs
- Method-level granularity for gap analysis
- Combines VCS changes with coverage profiling
- Visual treemaps for gap visualization

**Linked Content**:
- `fundamentals.md`: 5x defect multiplier statistic
- `SKILL.md`: Key insight highlighted
- `prioritization_strategies.md`: Risk scoring justification

---

## Code Churn & Quality Metrics

### Microsoft Azure DevOps - Code Churn Analysis
**Source**: https://learn.microsoft.com/en-us/previous-versions/azure/devops/report/sql-reports/perspective-code-analyze-report-code-churn-coverage
**Accessed**: November 3, 2025
**Relevant Learnings**:
- Code churn definition and measurement
- Correlation: high churn + low coverage = high defect risk
- Test-to-production ratio as health metric
- Positive correlation between test growth and code growth

**Linked Content**:
- `fundamentals.md`: Code churn definitions
- `git_analysis_patterns.md`: Ratio analysis patterns

### Jellyfish - Code Churn Library
**Source**: https://jellyfish.co/library/code-churn/
**Accessed**: November 3, 2025
**Relevant Learnings**:
- Lines Added, Deleted, Modified as churn metrics
- 3-week window for churn calculation
- High churn indicates unstable code
- Best practices for churn monitoring

**Linked Content**:
- `fundamentals.md`: Churn measurement details
- `prioritization_strategies.md`: Churn as risk dimension

### Atlassian - Code Coverage Best Practices
**Source**: https://www.atlassian.com/continuous-delivery/software-testing/code-coverage
**Accessed**: November 3, 2025
**Relevant Learnings**:
- Coverage thresholds: 70-80% for mature projects
- Coverage is necessary but not sufficient
- Diminishing returns above 80%
- Focus on critical paths over global percentage

**Linked Content**:
- `fundamentals.md`: Misconceptions section
- `prioritization_strategies.md`: Threshold recommendations

---

## Git Analysis Techniques

### Git Official Documentation
**Source**: https://git-scm.com/docs
**Accessed**: November 3, 2025
**Relevant Learnings**:
- `git log` filtering options
- `git diff` comparison techniques
- `git show` for commit inspection
- `git bisect` for regression analysis

**Linked Content**: `git_analysis_patterns.md` (all patterns)

### Stack Overflow - Git Log Queries
**Source**: Multiple threads
**Accessed**: November 3, 2025
**Relevant Learnings**:
- File change frequency queries
- Commit-by-commit analysis patterns
- Churn calculation with `--numstat`
- Branch comparison techniques

**Linked Content**: `git_analysis_patterns.md` (query examples)

---

## Coverage Tools & Documentation

### gcov/lcov (C/C++)

**gcov-example Repository**
**Source**: https://github.com/shenxianpeng/gcov-example
**Accessed**: November 3, 2025
**Relevant Learnings**:
- Complete C++ coverage workflow
- CMake integration patterns
- GitHub Actions integration
- LCOV report generation

**Linked Content**: `tooling_integration.md` (C++ section)

**Linux Test Project - LCOV Documentation**
**Source**: http://ltp.sourceforge.net/coverage/lcov.php
**Accessed**: November 3, 2025
**Relevant Learnings**:
- LCOV command-line options
- Report filtering techniques
- Branch coverage support
- HTML report generation

**Linked Content**: `tooling_integration.md` (C++ section)

### coverage.py (Python)

**Source**: https://coverage.readthedocs.io/
**Accessed**: November 3, 2025
**Relevant Learnings**:
- Configuration via .coveragerc or pyproject.toml
- Integration with pytest via pytest-cov
- Branch coverage options
- XML export for diff-cover

**Linked Content**: `tooling_integration.md` (Python section)

### JaCoCo (Java)

**Official Documentation**
**Source**: https://www.jacoco.org/
**Accessed**: November 3, 2025
**Relevant Learnings**:
- Maven and Gradle integration
- XML report format for diff-cover
- Threshold enforcement
- Branch and line coverage

**Linked Content**: `tooling_integration.md` (Java section)

---

## Test Impact Analysis

### Martin Fowler - Rise of Test Impact Analysis
**Source**: https://martinfowler.com/articles/rise-test-impact-analysis.html
**Accessed**: November 3, 2025
**Relevant Learnings**:
- TIA methodology and benefits
- Static analysis + dependency mapping
- Minimizing CI test execution time
- Trade-offs and limitations

**Linked Content**: `fundamentals.md` (TIA definition)

### Azure DevOps - Test Impact Analysis
**Source**: https://github.com/MicrosoftDocs/azure-devops-docs/blob/main/docs/pipelines/test/test-impact-analysis.md
**Accessed**: November 3, 2025
**Relevant Learnings**:
- TIA implementation in pipelines
- Identifying affected tests
- Performance benefits

**Linked Content**: `fundamentals.md` (TIA section)

### Datadog - Test Impact Analysis
**Source**: https://docs.datadoghq.com/tests/test_impact_analysis/
**Accessed**: November 3, 2025
**Relevant Learnings**:
- Modern TIA tooling
- Integration with CI/CD
- Coverage mapping techniques

**Linked Content**: `fundamentals.md` (TIA definition)

---

## Mutation Testing

### Pedro Rijo - Intro to Mutation Testing
**Source**: https://pedrorijo.com/blog/intro-mutation/
**Accessed**: November 3, 2025
**Relevant Learnings**:
- Why coverage alone is insufficient
- Mutation testing concept and value
- Mutation score calculation
- Tools by language (mutmut, PIT, etc.)

**Linked Content**:
- `fundamentals.md`: Coverage vs. test quality
- `tooling_integration.md`: Mutation testing section

### Codecov - Mutation Testing Guide
**Source**: https://about.codecov.io/ (mutation testing documentation)
**Accessed**: November 3, 2025
**Relevant Learnings**:
- Combining coverage with mutation testing
- When to use mutation testing
- Interpreting mutation scores

**Linked Content**: `tooling_integration.md` (mutation testing)

### Wikipedia - Mutation Testing
**Source**: https://en.wikipedia.org/wiki/Mutation_testing
**Accessed**: November 3, 2025
**Relevant Learnings**:
- Historical context and theory
- Mutation operators
- Equivalent mutants problem

**Linked Content**: `tooling_integration.md` (mutation testing background)

---

## Commercial Tools (Reference Only)

### SeaLights - Test Gap Analysis
**Source**: https://www.sealights.io/product/test-gap-analysis/
**Accessed**: November 3, 2025
**Relevant Learnings**:
- Enterprise gap analysis features
- Multi-stage coverage visibility
- Test type correlation (unit, integration, E2E)

**Linked Content**: `fundamentals.md` (test gap definition)

### Appsurify - Gap Analysis for QA
**Source**: https://appsurify.com/resources/use-gap-analysis-to-improve-testing/
**Accessed**: November 3, 2025
**Relevant Learnings**:
- Gap analysis methodologies
- Prioritization frameworks

**Linked Content**: `prioritization_strategies.md` (framework inspiration)

### Codecov & Coveralls (SaaS Platforms)
**Sources**:
- https://about.codecov.io/
- https://coveralls.io/

**Accessed**: November 3, 2025
**Relevant Learnings**:
- Hosted coverage tracking
- GitHub integration patterns
- Coverage badge generation
- Trend visualization

**Linked Content**: CI/CD examples in `diff_cover_workflow.md`

---

## CI/CD Integration References

### GitHub Actions Marketplace - JaCoCo Report
**Source**: https://github.com/marketplace/actions/jacoco-report
**Accessed**: November 3, 2025
**Relevant Learnings**:
- Action for Java coverage reporting
- PR comment integration
- Badge generation

**Linked Content**: `diff_cover_workflow.md` (GitHub Actions examples)

### GitLab Documentation - Code Coverage
**Source**: https://docs.gitlab.com/ci/testing/code_coverage/
**Accessed**: November 3, 2025
**Relevant Learnings**:
- GitLab CI coverage integration
- Coverage badge configuration
- Artifact management

**Linked Content**: `diff_cover_workflow.md` (GitLab CI examples)

---

## Supplementary Tools

### undercover (Ruby)
**Source**: https://github.com/grodowski/undercover
**Accessed**: November 3, 2025
**Relevant Learnings**:
- Ruby equivalent of diff-cover
- SimpleCov integration
- Changed method detection

**Linked Content**: `fundamentals.md` (tool landscape)

### SimpleCov (Ruby)
**Source**: https://github.com/simplecov-ruby/simplecov
**Accessed**: November 3, 2025
**Relevant Learnings**:
- Ruby coverage tool patterns
- Report formats

**Linked Content**: Referenced for comparison

---

## Pattern Evolution & Maintenance

### Currency Check Protocol

**Review Frequency**: Quarterly (every 3 months)

**Items to Check**:
1. **diff-cover updates**: New features, API changes, deprecated options
2. **Coverage tool versions**: gcov, lcov, coverage.py, JaCoCo updates
3. **CI/CD platform changes**: GitHub Actions, GitLab CI syntax updates
4. **New tools**: Emerging coverage or mutation testing tools
5. **Research updates**: New studies on test effectiveness, defect correlation

**Maintenance Owner**: lab-tech-blue (skill review)

**Change Log Location**: Track major updates in skill version history

---

## Related cisTEMx Skills

These skills reference or complement this one:

- **unit-testing**: Use to write tests for identified gaps
- **find-bug-introduction**: Cross-reference bugs with coverage gaps
- **identify-refactoring-targets**: High-complexity, low-coverage candidates
- **compile-code**: Ensure coverage builds compile successfully
- **git-version-control**: Commit discipline for test changes

---

## Version History

**v1.0.0** (November 3, 2025)
- Initial skill creation
- Comprehensive research synthesis
- All 23 primary sources integrated
- Resources, scripts, and templates created

---

**Note**: This skill synthesizes publicly available research, documentation, and best practices. All tools and methodologies referenced are either open-source or publicly documented. Commercial tools mentioned for reference/comparison only.
