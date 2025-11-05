# Citations and References

## Purpose

This document tracks all external sources, tools, and resources used in this skill to enable future maintenance and currency checks.

---

## Books

### Tornhill, Adam (2015). *Your Code as a Crime Scene*

**Publisher**: Pragmatic Bookshelf
**ISBN**: 978-1680500387
**Relevance**: First edition introducing behavioral code analysis, hotspots, coupling, churn
**Accessed**: November 3, 2025
**Linked Content**:
- `fundamentals.md` - Core concepts
- `churn_analysis.md` - Change frequency analysis
- `hotspot_analysis.md` - Hotspot methodology

### Tornhill, Adam (2018). *Software Design X-Rays*

**Publisher**: Pragmatic Bookshelf
**ISBN**: 978-1680502725
**Relevance**: Second edition with expanded techniques, X-Ray analysis, complexity trends, Conway's Law
**Accessed**: November 3, 2025
**Linked Content**:
- `fundamentals.md` - Three critical questions
- `hotspot_analysis.md` - Multi-level hotspot analysis
- `temporal_coupling.md` - Conway's Law analysis

### Tornhill, Adam (2022). *Your Code as a Crime Scene, Second Edition*

**Publisher**: Pragmatic Bookshelf
**ISBN**: 978-1680509786
**Relevance**: Updated with modern tooling and case studies
**Accessed**: November 3, 2025
**Linked Content**: All resources

---

## Tools

### Code Maat

**Repository**: https://github.com/adamtornhill/code-maat
**Version**: 1.0.4
**Author**: Adam Tornhill
**License**: Open source
**Accessed**: November 3, 2025
**Relevance**: Command-line tool for VCS data mining (coupling, churn, ownership)
**Linked Content**:
- `code_maat_guide.md` - Complete tool reference
- `temporal_coupling.md` - Coupling analysis
- `practical_workflow.md` - Integration examples

**Documentation**: README.md in repository
**Installation**: https://github.com/adamtornhill/code-maat/releases

### GitNStats

**Repository**: https://github.com/rubberduck203/GitNStats
**Releases**: https://github.com/rubberduck203/GitNStats/releases
**Author**: rubberduck203
**License**: Open source
**Accessed**: November 3, 2025
**Relevance**: Cross-platform git history analyzer for churn hotspots
**Linked Content**: `churn_analysis.md` - Alternative tools

### code-forensics

**Repository**: https://github.com/smontanari/code-forensics
**NPM**: https://www.npmjs.com/package/code-forensics
**Author**: Silvio Montanari
**License**: GPLv3
**Accessed**: November 3, 2025
**Relevance**: Node.js toolset with visualizations, builds on Code Maat
**Linked Content**: `code_maat_guide.md` - Alternatives section

### Lizard

**Repository**: https://github.com/terryyin/lizard
**PyPI**: https://pypi.org/project/lizard/
**Author**: Terry Yin
**License**: Open source (MIT)
**Accessed**: November 3, 2025
**Relevance**: Multi-language cyclomatic complexity analyzer (C++, Python, 20+ languages)
**Linked Content**:
- `hotspot_analysis.md` - Complexity measurement
- `practical_workflow.md` - Complexity analysis steps

**Documentation**: https://github.com/terryyin/lizard/blob/master/README.md

### Radon

**Repository**: https://github.com/rubik/radon
**PyPI**: https://pypi.org/project/radon/
**Documentation**: https://radon.readthedocs.io/
**Author**: Michele Lacchia
**License**: Open source (MIT)
**Accessed**: November 3, 2025
**Relevance**: Python-specific complexity analysis tool
**Linked Content**: `hotspot_analysis.md` - Python complexity

**Used by**: Codacy, CodeFactor for code quality analysis

### SonarQube

**Website**: https://www.sonarqube.org/
**Documentation**: https://docs.sonarqube.org/
**License**: Commercial + Community Edition
**Accessed**: November 3, 2025
**Relevance**: Static analysis + technical debt tracking for multiple languages
**Linked Content**: `hotspot_analysis.md` - Alternative metrics

---

## Articles and Blog Posts

### Tornhill, Adam (2018). "Software (r)Evolution — Part 2"

**URL**: https://adamtornhill.com/articles/software-revolution/part2/index.html
**Title**: "Novel Techniques to Prioritize Technical Debt"
**Accessed**: November 3, 2025
**Relevance**: Prioritization strategies, ROI framework
**Linked Content**: `fundamentals.md` - ROI framework

### Blain, Nicolas Espeon (2020). "Focus refactoring on what matters with Hotspots Analysis"

**URL**: https://understandlegacycode.com/blog/focus-refactoring-with-hotspots-analysis/
**Accessed**: November 3, 2025
**Relevance**: Practical hotspot analysis guide
**Linked Content**: `hotspot_analysis.md` - Methodology

### Blain, Nicolas Espeon (2018). "The key points of Software Design X-Rays"

**URL**: https://understandlegacycode.com/blog/key-points-of-software-design-x-rays/
**Accessed**: November 3, 2025
**Relevance**: Summary of Tornhill's methodology
**Linked Content**: `fundamentals.md` - Core concepts

### Embedded Artistry (2018). "GitNStats: A Git History Analyzer"

**URL**: https://embeddedartistry.com/blog/2018/06/21/gitnstats-a-git-history-analyzer-to-help-identify-code-hotspots/
**Accessed**: November 3, 2025
**Relevance**: GitNStats tool introduction and usage
**Linked Content**: `churn_analysis.md` - Tools section

---

## Official Documentation

### Git Documentation - Searching

**URL**: https://git-scm.com/book/en/v2/Git-Tools-Searching
**Accessed**: November 3, 2025
**Relevance**: Official documentation for git log -S, -G, -L options
**Linked Content**: `churn_analysis.md` - Advanced git techniques

### Git Blame Documentation

**URL**: https://git-scm.com/docs/git-blame
**Accessed**: November 3, 2025
**Relevance**: Official git blame documentation
**Linked Content**: `churn_analysis.md` - Authorship analysis

### Git Log Documentation

**URL**: https://git-scm.com/docs/git-log
**Accessed**: November 3, 2025
**Relevance**: Complete git log reference
**Linked Content**: All churn analysis commands

---

## Research Papers

### MIT and Harvard Business School: Conway's Law Study

**Finding**: "Strong evidence to support the mirroring hypothesis"
**Citation**: Referenced in multiple Conway's Law articles
**Accessed**: November 3, 2025
**Relevance**: Empirical validation of Conway's Law
**Linked Content**: `temporal_coupling.md` - Conway's Law analysis

### Pre-release Code Churn as Defect Predictor

**Finding**: Churn correlates with post-release defects
**Referenced in**: Code Maat documentation, academic literature
**Accessed**: November 3, 2025
**Relevance**: Justification for churn-based prioritization
**Linked Content**: `fundamentals.md` - Core metrics

---

## Community Resources

### Stack Overflow: "Code churn for git repository?"

**URL**: https://stackoverflow.com/questions/46612140/code-churn-for-git-repository
**Accessed**: November 3, 2025
**Relevance**: Practical git commands for churn analysis
**Linked Content**: `churn_analysis.md` - Command examples

### Stack Overflow: "How to grep committed code in Git history"

**URL**: https://stackoverflow.com/questions/2928584/how-to-grep-search-through-committed-code-in-the-git-history
**Accessed**: November 3, 2025
**Relevance**: Pickaxe techniques and examples
**Linked Content**: `churn_analysis.md` - Advanced techniques

### Martin Fowler's Bliki: "Conway's Law"

**URL**: https://martinfowler.com/bliki/ConwaysLaw.html
**Accessed**: November 3, 2025
**Relevance**: Authoritative explanation and architectural implications
**Linked Content**: `temporal_coupling.md` - Conway's Law section

---

## Platform Dependencies

### Python + Pandas

**Pandas Version**: 2.0+
**URL**: https://pandas.pydata.org/
**Relevance**: Data joining for hotspot analysis
**Linked Content**: `practical_workflow.md` - Data processing scripts

### Java Runtime (JRE)

**Version**: 8 or higher
**Relevance**: Required for Code Maat execution
**Linked Content**: `code_maat_guide.md` - Installation

### Matplotlib

**URL**: https://matplotlib.org/
**Relevance**: Hotspot visualization
**Linked Content**: `practical_workflow.md` - Visualization scripts

---

## Research Foundation

This skill is built upon comprehensive research documented in:
**`/workspaces/cisTEMx/.claude/cache/git_history_churn_research.md`**

**Research Date**: November 3, 2025
**Researcher**: Claude
**Scope**: Git history analysis, code churn, hotspots, refactoring prioritization

---

## Maintenance Notes

**Review Frequency**: Quarterly

**Update Triggers**:
- New Code Maat versions
- Git updates affecting log formats
- New complexity tools becoming available
- Changes to cited URLs (link rot)
- Academic research updates

**Last Reviewed**: November 3, 2025
**Next Review**: February 2026

---

## Version History

- **v1.0.0** (November 3, 2025): Initial skill creation
  - Complete resource suite
  - Code Maat v1.0.4
  - Lizard, Radon tool references
  - Comprehensive citations
