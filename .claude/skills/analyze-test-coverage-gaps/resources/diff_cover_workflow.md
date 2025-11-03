# diff-cover: Incremental Coverage Workflow

## Purpose

Practical guide to using diff-cover for enforcing test coverage on changes. The "if you touch it, test it" principle.

## When You Need This

- Setting up incremental coverage enforcement
- Want to improve coverage without testing entire legacy codebase
- Need CI/CD integration for coverage checks
- Addressing "tests don't keep up with code" problem

---

## What is diff-cover?

**diff-cover** is an open-source tool that compares your coverage report with `git diff` output to show coverage of **only the lines you changed**.

**Key Insight**: You can achieve 80%+ diff coverage on new code even if overall coverage is 40%.

**License**: Apache 2.0
**Language**: Python (but works with any language that generates coverage reports)
**Repository**: https://github.com/Bachmann1234/diff_cover

---

## Installation

### Basic Installation
```bash
pip install diff_cover
```

### With Development Tools
```bash
# Include quality checking tools
pip install diff_cover diff_quality
```

### Verify Installation
```bash
diff-cover --version
```

---

## Supported Coverage Formats

diff-cover works with multiple coverage report formats:

- **Cobertura XML** (Python, Java, C#, C++)
- **Clover XML** (Java, JavaScript)
- **JaCoCo XML** (Java)
- **LCOV** (C/C++, JavaScript)

**Tip**: Most coverage tools can export to one of these formats.

---

## Basic Workflow

### Step 1: Generate Coverage Report

**For C++ (using gcov/lcov)**:
```bash
# Clean previous coverage data
find . -name "*.gcda" -delete

# Compile with coverage flags
cmake -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_FLAGS="--coverage" \
      -DCMAKE_EXE_LINKER_FLAGS="--coverage" ..
make

# Run tests
./run_tests

# Generate LCOV report
lcov --capture --directory . --output-file coverage.info

# Optional: Generate HTML for human viewing
genhtml coverage.info --output-directory coverage_html
```

**For Python (using coverage.py)**:
```bash
# Run tests with coverage
coverage run -m pytest

# Generate XML report for diff-cover
coverage xml

# Optional: Generate HTML for human viewing
coverage html
```

**For Java (using JaCoCo with Maven)**:
```bash
# Run tests with coverage
mvn clean test

# Coverage report at: target/site/jacoco/jacoco.xml
```

### Step 2: Run diff-cover

**Compare Against Branch**:
```bash
# Compare your changes against main branch
diff-cover coverage.xml --compare-branch=origin/main
```

**Output Example**:
```
-------------
Diff Coverage
Diff: origin/main...HEAD, staged and unstaged changes
-------------
src/core/processor.cpp (88.2%)
src/utils/helper.cpp (75.0%)
-------------
Total:   82.5%
```

### Step 3: Enforce Threshold

**Fail Build if Below Threshold**:
```bash
diff-cover coverage.xml --compare-branch=origin/main --fail-under=80
```

**Exit codes**:
- `0`: Coverage meets threshold
- `1`: Coverage below threshold (fails CI)

---

## Advanced Usage

### Multiple Coverage Reports

Combine multiple coverage reports (e.g., unit + integration):
```bash
diff-cover unit_coverage.xml integration_coverage.xml \
  --compare-branch=origin/main
```

### Custom Diff Source

**Use Diff File Instead of Branch**:
```bash
# Generate diff file
git diff main..feature-branch > changes.diff

# Run diff-cover against diff file
diff-cover coverage.xml --diff-file=changes.diff
```

**Use Case**: Comparing against non-standard branches or analyzing historical changes.

### Path Filtering

**Include Only Specific Paths**:
```bash
# Only check coverage in src/core/
diff-cover coverage.xml \
  --compare-branch=origin/main \
  --include "src/core/**"
```

**Exclude Paths**:
```bash
# Exclude test files and generated code
diff-cover coverage.xml \
  --compare-branch=origin/main \
  --exclude "test_*.cpp" \
  --exclude "src/generated/**"
```

### Report Formats

**HTML Report**:
```bash
diff-cover coverage.xml \
  --compare-branch=origin/main \
  --html-report=diff-coverage.html
```

**JSON Report** (for programmatic consumption):
```bash
diff-cover coverage.xml \
  --compare-branch=origin/main \
  --json-report=diff-coverage.json
```

**Markdown Report** (for GitHub/GitLab comments):
```bash
diff-cover coverage.xml \
  --compare-branch=origin/main \
  --markdown-report=diff-coverage.md
```

### Quiet Mode

Suppress non-essential output:
```bash
diff-cover coverage.xml --compare-branch=origin/main -q
```

---

## Configuration File

Create `.diffcover.toml` in project root:

```toml
[tool.diff_cover]
compare_branch = "origin/develop"
fail_under = 80.0
html_report = "coverage_report.html"
json_report = "coverage_report.json"
exclude = [
    "setup.py",
    "*/migrations/*",
    "*/tests/*",
    "src/generated/**"
]
include = [
    "src/**"
]
quiet = false
```

Then run without arguments:
```bash
diff-cover coverage.xml  # Uses .diffcover.toml config
```

---

## CI/CD Integration

### GitHub Actions

```yaml
name: Diff Coverage Check

on:
  pull_request:
    branches: [main]

jobs:
  diff-coverage:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v3
        with:
          fetch-depth: 0  # Need full history for diff

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install pytest pytest-cov diff-cover

      - name: Run tests with coverage
        run: |
          pytest --cov --cov-report=xml

      - name: Check diff coverage
        run: |
          diff-cover coverage.xml \
            --compare-branch=origin/main \
            --fail-under=80 \
            --html-report=diff-coverage.html

      - name: Upload coverage report
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: diff-coverage-report
          path: diff-coverage.html

      - name: Comment PR with results
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const coverage = fs.readFileSync('diff-coverage.md', 'utf8');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: coverage
            });
```

### GitLab CI

```yaml
# .gitlab-ci.yml
stages:
  - test
  - coverage

test:
  stage: test
  script:
    - pip install pytest pytest-cov
    - pytest --cov --cov-report=xml
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml

diff-coverage:
  stage: coverage
  dependencies:
    - test
  script:
    - pip install diff-cover
    - diff-cover coverage.xml --compare-branch=origin/main --fail-under=80
  only:
    - merge_requests
```

### Jenkins

```groovy
pipeline {
    agent any

    stages {
        stage('Test') {
            steps {
                sh 'pytest --cov --cov-report=xml'
            }
        }

        stage('Diff Coverage') {
            steps {
                sh '''
                    pip install diff-cover
                    diff-cover coverage.xml \
                        --compare-branch=origin/main \
                        --fail-under=80 \
                        --html-report=diff-coverage.html
                '''
            }
        }
    }

    post {
        always {
            publishHTML([
                reportDir: '.',
                reportFiles: 'diff-coverage.html',
                reportName: 'Diff Coverage Report'
            ])
        }
    }
}
```

---

## Integration with cisTEMx

### For C++ Code with Catch2

**Assuming cisTEMx uses CMake + Catch2:**

```bash
#!/bin/bash
# scripts/check_diff_coverage.sh

set -e

echo "=== Building with Coverage ==="
mkdir -p build-coverage
cd build-coverage

cmake .. \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="--coverage" \
  -DCMAKE_EXE_LINKER_FLAGS="--coverage"

make -j$(nproc)

echo "=== Running Tests ==="
# Assuming tests are in build-coverage/bin/
find bin -name "test_*" -type f -executable -exec {} \;

echo "=== Generating Coverage Report ==="
lcov --capture --directory . --output-file coverage.info

# Filter out system headers and test files
lcov --remove coverage.info '/usr/*' '*/test/*' --output-file coverage_filtered.info

echo "=== Running diff-cover ==="
pip3 install diff-cover

diff-cover coverage_filtered.info \
  --compare-branch=origin/main \
  --fail-under=80 \
  --html-report=diff-coverage.html

echo "=== Results ==="
echo "HTML report: build-coverage/diff-coverage.html"
```

### Pre-commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit

# Check if this is a merge commit
if git rev-parse -q --verify MERGE_HEAD >/dev/null 2>&1; then
    exit 0  # Skip for merge commits
fi

# Check for production code changes
PROD_CHANGES=$(git diff --cached --name-only --diff-filter=AM | grep "^src/" | grep -v "test")

if [ -n "$PROD_CHANGES" ]; then
    echo "Production code changed. Checking diff coverage..."

    # Run quick coverage check
    # (Adjust based on your test setup)
    make test-coverage >/dev/null 2>&1

    if ! diff-cover coverage.xml --compare-branch=origin/main --fail-under=70 -q; then
        echo ""
        echo "ERROR: Diff coverage below 70%"
        echo "Run: diff-cover coverage.xml --compare-branch=origin/main"
        echo "to see uncovered lines."
        echo ""
        read -p "Commit anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

exit 0
```

---

## Real-World Case Study: edX

**Challenge**: Open edX codebase had <50% test coverage, hard to improve.

**Solution**: Enforced diff-coverage instead of overall coverage.

**Implementation**:
1. Required 80% diff-coverage on all PRs
2. Allowed legacy code to remain untested
3. Prevented new gaps from forming
4. Slowly improved overall coverage as old code was touched

**Results**:
- **10 months**: Coverage increased from <50% to 87%
- **Developer adoption**: High (achievable metrics)
- **Regression prevention**: Significant reduction in production bugs

**Key Lesson**: "If you touch it, test it" is more effective than "test everything".

---

## Troubleshooting

### Issue: "No coverage data for changed lines"

**Cause**: Coverage report doesn't include changed files (e.g., files weren't compiled with coverage flags).

**Solution**:
```bash
# Ensure ALL source files compiled with coverage
cmake -DCMAKE_CXX_FLAGS="--coverage" ..

# Verify .gcno files exist for changed files
find . -name "*.gcno"

# Run tests to generate .gcda files
./run_tests

# Verify .gcda files generated
find . -name "*.gcda"
```

### Issue: "diff-cover reports 0% for file I tested"

**Cause**: Coverage report path doesn't match git diff path.

**Solution**:
```bash
# Check coverage report paths
lcov --list coverage.info | head -20

# If paths are absolute, use --src-roots
diff-cover coverage.info \
  --compare-branch=origin/main \
  --src-roots=/path/to/project
```

### Issue: "fetch-depth: 0 required in CI"

**Cause**: GitHub Actions shallow clone doesn't have base branch for comparison.

**Solution**:
```yaml
- uses: actions/checkout@v3
  with:
    fetch-depth: 0  # Full history
```

Or manually fetch base branch:
```yaml
- name: Fetch base branch
  run: git fetch origin main:main
```

### Issue: "diff-cover too slow on large repos"

**Solution**: Use path filters to focus on relevant code:
```bash
diff-cover coverage.xml \
  --compare-branch=origin/main \
  --include "src/core/**" \
  --include "src/utils/**"
```

---

## Progressive Rollout Strategy

### Phase 1: Awareness (Week 1)
- Install diff-cover
- Run on recent PRs (no enforcement)
- Share reports in team meetings
- Discuss threshold targets

### Phase 2: Soft Enforcement (Weeks 2-4)
- Add diff-cover to CI (allowed to fail)
- Set threshold at 60%
- Require team to review failures (but allow merges)
- Build test-writing habits

### Phase 3: Hard Enforcement (Month 2+)
- Fail CI if diff-coverage < 70%
- Increase to 80% after 2 weeks
- Allow exceptions with explicit justification
- Track coverage trends over time

### Phase 4: Excellence (Month 3+)
- Maintain 80%+ diff-coverage
- Start addressing legacy gaps opportunistically
- Add mutation testing for critical paths
- Celebrate coverage milestones

---

## Best Practices

### 1. Focus on Diff Coverage, Not Overall Coverage

**Don't**: "We need to get from 40% to 80% coverage"
**Do**: "All new code must have 80% coverage"

### 2. Set Achievable Thresholds

**Start**: 60-70% diff-coverage
**Mature**: 80-90% diff-coverage
**Don't**: 100% (allows reasonable exceptions)

### 3. Combine with Code Review

During PR review, ask:
1. What changed? (git diff)
2. What's the diff-coverage? (diff-cover report)
3. Are uncovered lines intentional? (logging, impossible cases)
4. Are tests meaningful? (assertions, edge cases)

### 4. Make Reports Visible

- Post HTML reports in PR comments
- Track coverage trends over time
- Celebrate improvements
- Discuss failures constructively

### 5. Document Exceptions

When allowing <80% diff-coverage:
```python
# NOTE: Coverage exception approved in PR #1234
# Reason: Legacy API wrapper, will be deprecated in v3.0
# Owner: @username
# Review date: 2025-12-01
```

---

## Related Resources

- **`fundamentals.md`**: Why diff-coverage matters (5x defect multiplier)
- **`git_analysis_patterns.md`**: Combine diff-cover with git history analysis
- **`tooling_integration.md`**: Coverage tool setup for different languages
- **`prioritization_strategies.md`**: Prioritizing uncovered lines
