# Git History Analysis Skill

## Purpose

Master git history forensics to identify bugs, refactoring needs, test gaps, and code patterns through expert-level repository analysis. This skill provides techniques used by senior developers for code archaeology, technical debt identification, and development pattern recognition.

## When to Use This Skill

Use this skill when you need to:

- **Investigate bugs**: Find when/where a bug was introduced
- **Plan refactoring**: Identify problematic code hotspots needing attention
- **Assess test coverage**: Find untested or undertested code
- **Understand code evolution**: Track how code changed over time
- **Find code owners**: Identify experts for specific files/areas
- **Generate insights**: Understand team patterns and collaboration dynamics
- **Create release notes**: Extract meaningful changelogs from history

## Core Capabilities (8 Categories)

### 1. Bug Introduction Identification
**What**: Pinpoint the exact commit that introduced a bug or regression
**Primary tool**: `git bisect` (binary search through history)
**Use when**: Something worked before but is broken now
**Deep dive**: See `resources/bug_identification.md`

### 2. Code Churn Analysis
**What**: Identify files that change too frequently (design smell indicator)
**Primary tool**: `git log` with frequency analysis
**Use when**: Planning refactoring priorities
**Deep dive**: See `resources/churn_analysis.md`

### 3. Temporal Coupling Analysis
**What**: Find files that change together (hidden dependencies)
**Primary tool**: Code Maat coupling analysis
**Use when**: Investigating architectural decay
**Deep dive**: See `resources/coupling_analysis.md`

### 4. Test Coverage Gap Identification
**What**: Find code that changes frequently without corresponding tests
**Primary tool**: Production-to-test ratio analysis
**Use when**: Prioritizing test writing efforts
**Deep dive**: See `resources/test_coverage_gaps.md`

### 5. The Pickaxe (Code Search)
**What**: Find when specific code was added/removed
**Primary tool**: `git log -S` and `git log -G`
**Use when**: "When did this function appear?", code archaeology
**Deep dive**: See `resources/pickaxe_guide.md`

### 6. Advanced Blame Techniques
**What**: Track code through moves, copies, and refactorings
**Primary tool**: `git blame` with `-CCC` flags
**Use when**: Finding the TRUE author of code
**Deep dive**: See `resources/blame_techniques.md`

### 7. Author & Team Analysis
**What**: Understand code ownership and collaboration patterns
**Primary tool**: `git log` with author filtering and statistics
**Use when**: Finding reviewers, understanding team dynamics
**Deep dive**: See `resources/author_analysis.md`

### 8. Code Archaeology
**What**: Follow files through renames, find code origins
**Primary tool**: `git log --follow`, rename tracking
**Use when**: File history seems incomplete
**Deep dive**: See `resources/code_archaeology.md`

## Quick Reference - Common Scenarios

### "When was this bug introduced?"
```bash
# Binary search to find the bad commit
git bisect start
git bisect bad                    # Current state is broken
git bisect good v1.0             # Known good version
git bisect run ./test_script.sh  # Automate with test

# Or manually test and mark each commit
# git bisect good/bad
```
**See**: `resources/bug_identification.md` for advanced techniques

### "Which files need refactoring most urgently?"
```bash
# Find high-churn files (change frequently)
git log --format=format: --name-only --since=12.month \
  | egrep -v '^$' | sort | uniq -c | sort -nr | head -20
```
**Or use script**: `scripts/find_hotspots.sh`
**See**: `resources/churn_analysis.md` for hotspot analysis

### "Which files should have more tests?"
```bash
# Files that appear in "fix" commits repeatedly
git log --grep="fix|bug|crash" -i --name-only \
  | grep -E '\.(cpp|h)$' | sort | uniq -c | sort -nr | head -20
```
**Or use script**: `scripts/test_coverage_ratio.sh`
**See**: `resources/test_coverage_gaps.md` for comprehensive analysis

### "When did this function get added?"
```bash
# Find commits that added/removed specific code
git log -S"functionName" --source --all

# With actual diff shown
git log -S"functionName" -p

# Using regex pattern
git log -G"render.*Particle" --pickaxe-regex
```
**See**: `resources/pickaxe_guide.md` for advanced search patterns

### "Who should review changes to this file?"
```bash
# Find recent contributors (potential reviewers)
git log --format='%aN' --since="6.months" -- src/core/image.cpp \
  | sort | uniq -c | sort -nr | head -5
```
**Or use script**: `scripts/suggest_reviewers.sh path/to/file`
**See**: `resources/author_analysis.md` for team insights

### "Which files always change together?"
```bash
# Use Code Maat for temporal coupling analysis
git log --all --numstat --date=short \
  --pretty=format:'--%h--%ad--%aN' --no-renames > git.log
java -jar code-maat.jar -l git.log -c git2 -a coupling | sort -t',' -k3 -nr
```
**Or use script**: `scripts/coupling_analysis.sh`
**See**: `resources/coupling_analysis.md` for interpretation

### "Generate release notes from commits"
```bash
# List all commits since last release
git log v1.0..HEAD --pretty=format:"- %s" --no-merges

# Categorized by type (if using conventional commits)
git log v1.0..HEAD --grep="^feat:" --pretty=format:"- %s"
git log v1.0..HEAD --grep="^fix:" --pretty=format:"- %s"
```
**Or use script**: `scripts/generate_changelog.sh v1.0..HEAD`

## Utility Scripts

All scripts are located in `.claude/skills/git-history/scripts/`:

- **find_hotspots.sh** - Identify high-churn files (refactoring candidates)
- **coupling_analysis.sh** - Find files that change together (hidden dependencies)
- **suggest_reviewers.sh** - Find experts for specific files
- **test_coverage_ratio.sh** - Compare production vs test changes
- **generate_changelog.sh** - Create categorized release notes

Usage: `bash .claude/skills/git-history/scripts/<script_name> [args]`

## Report Templates

Templates for structured analysis output:

- **templates/refactoring_report.md** - Refactoring prioritization analysis
- **templates/hotspot_analysis.md** - Deep-dive on problem areas

## Progressive Disclosure

This SKILL.md provides quick reference and common scenarios. For deeper understanding:

1. **Start here** with the quick reference for your specific scenario
2. **Run utility scripts** for automated analysis
3. **Consult resource docs** for advanced techniques and edge cases
4. **Study workflows** in resource docs for complex investigations

## Key Principles

1. **Small, focused commits enable better history analysis** - Every technique works better with atomic commits
2. **Git history is a diagnostic tool** - Use it proactively, not just when problems arise
3. **Combine techniques** - Most investigations use multiple approaches
4. **Automate recurring analyses** - Scripts + regular checks reveal trends
5. **Document findings** - Use templates to capture and share insights

## Expert Workflows

Complex investigations typically combine multiple techniques:

### Investigating a Mysterious Bug
1. Use `git bisect` to find introducing commit
2. Use `git show <commit>` to see full context
3. Use `git blame -CCC` to find original author
4. Use `git log -S"function"` to see function's evolution
5. Check related changes with date-scoped `git log`

### Planning Major Refactoring
1. Run `find_hotspots.sh` to identify high-churn files
2. Run `coupling_analysis.sh` to find unexpected dependencies
3. Use `author_analysis.md` techniques to find code owners
4. Check for technical debt: `git log --grep="TODO|FIXME|HACK" -i`
5. Assess test coverage with `test_coverage_ratio.sh`
6. Generate report using `templates/refactoring_report.md`

### Preparing for Code Review
1. Run `suggest_reviewers.sh` on changed files
2. Check if changes touch hotspot files (extra scrutiny needed)
3. Verify test coverage for changed code
4. Document rationale for changes to frequently-modified code

## Common Gotchas

- **Merge commits skew statistics** - Use `--no-merges` flag
- **Rename detection** - Default 50% threshold; adjust with `-M<threshold>`
- **Git blame shows last edit** - Use `-CCC` to track through refactorings
- **Case sensitivity** - Most searches are case-sensitive; add `-i` for insensitive
- **Performance** - Limit scope with `--since`, `-- path/` for large repos

## External Tools (Optional)

These tools enhance git history analysis:

- **Code Maat** - Sophisticated forensic analysis (coupling, fragmentation, hotspots)
- **git-quick-stats** - Rich CLI statistics and visualizations
- **diff-cover** - Test coverage for changed lines (perfect for PRs)
- **GitNStats** - Cross-platform hotspot identification

## Further Reading

- Git official docs: https://git-scm.com/docs
- "Your Code as a Crime Scene" by Adam Tornhill (hotspot methodology)
- Code Maat: https://github.com/adamtornhill/code-maat
- Git forensics community practices on Stack Overflow

---

**Remember**: Git history is not just version control - it's a rich diagnostic database revealing code quality, team patterns, and technical debt. Master these techniques to become a more effective developer and architect.
