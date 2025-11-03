# Coverage Tooling Integration Guide

## Purpose

Practical guide to setting up and using coverage tools across different languages and integrating them with gap analysis workflows.

## When You Need This

- Setting up coverage infrastructure for first time
- Integrating coverage with CI/CD
- Troubleshooting coverage generation
- Evaluating test quality beyond coverage metrics

---

## Coverage Tools by Language

### C/C++: gcov + lcov

**gcov**: GCC's built-in coverage tool (comes with GCC)
**lcov**: Front-end for gcov, generates HTML reports

#### Installation

```bash
# gcov comes with GCC
gcc --version  # Verify GCC installed

# Install lcov
# Ubuntu/Debian
sudo apt-get install lcov

# macOS
brew install lcov

# Verify
lcov --version
```

#### Basic Workflow

**Step 1: Compile with Coverage Flags**

```bash
# Manual compilation
g++ -fprofile-arcs -ftest-coverage -O0 source.cpp -o program

# CMake approach (recommended)
cmake -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_FLAGS="--coverage" \
      -DCMAKE_EXE_LINKER_FLAGS="--coverage" \
      ..
make
```

**What these flags do**:
- `-fprofile-arcs`: Generate .gcda files with execution counts
- `-ftest-coverage`: Generate .gcno files with control flow graph
- `-O0`: Disable optimizations (prevents line merging)
- `--coverage`: Shorthand for both above + linking

**Step 2: Run Tests**

```bash
# Run your test executable(s)
./test_runner

# Or with Catch2
./build/bin/test_mymodule

# This generates .gcda files alongside .gcno files
```

**Step 3: Generate Coverage Report**

```bash
# Capture coverage data
lcov --capture --directory . --output-file coverage.info

# Filter out system headers and test files
lcov --remove coverage.info '/usr/*' '*/test/*' --output-file coverage_filtered.info

# Generate HTML report
genhtml coverage_filtered.info --output-directory coverage_html

# View report
open coverage_html/index.html  # macOS
xdg-open coverage_html/index.html  # Linux
```

**Step 4: View Summary**

```bash
# Terminal summary
lcov --list coverage_filtered.info

# Find zero-coverage files
lcov --list coverage_filtered.info | grep "0.0%"
```

#### CMake Integration

```cmake
# CMakeLists.txt - Add coverage option

option(ENABLE_COVERAGE "Enable coverage reporting" OFF)

if(ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        # Add coverage flags
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")

        # Add coverage target
        add_custom_target(coverage
            COMMAND ${CMAKE_COMMAND} -E make_directory coverage_html
            COMMAND lcov --capture --directory . --output-file coverage.info
            COMMAND lcov --remove coverage.info '/usr/*' '*/test/*' --output-file coverage_filtered.info
            COMMAND genhtml coverage_filtered.info --output-directory coverage_html
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Generating coverage report"
        )
    else()
        message(WARNING "Coverage only supported with GCC or Clang")
    endif()
endif()
```

**Usage**:
```bash
# Configure with coverage
cmake -DENABLE_COVERAGE=ON ..

# Build and test
make
make test

# Generate coverage
make coverage
```

#### Troubleshooting

**Issue**: "cannot open graph file"

**Solution**: .gcno files missing, recompile with `--coverage`.

**Issue**: No coverage data (.gcda files empty)

**Solution**: Tests didn't run or crashed. Verify tests complete successfully.

**Issue**: Coverage shows 0% for all files

**Solution**: Path mismatch. Use `lcov --rc lcov_branch_coverage=1` or check working directory.

---

### Python: coverage.py

**coverage.py**: Standard Python coverage tool

#### Installation

```bash
pip install coverage
```

#### Basic Workflow

**Step 1: Run Tests with Coverage**

```bash
# Using pytest
coverage run -m pytest

# Using unittest
coverage run -m unittest discover

# Custom script
coverage run myscript.py
```

**Step 2: View Results**

```bash
# Terminal report
coverage report

# Terminal with missing lines
coverage report -m

# HTML report
coverage html
open htmlcov/index.html
```

**Step 3: XML for diff-cover**

```bash
# Generate Cobertura XML
coverage xml

# Now use with diff-cover
diff-cover coverage.xml --compare-branch=origin/main
```

#### Configuration File

Create `.coveragerc`:

```ini
[run]
source = src/
omit =
    */tests/*
    */test_*.py
    */__init__.py
    */setup.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    if self.debug:
    if settings.DEBUG
    raise AssertionError
    raise NotImplementedError
    if 0:
    if __name__ == .__main__.:
    class .*\bProtocol\):
    @(abc\.)?abstractmethod

[html]
directory = htmlcov

[xml]
output = coverage.xml
```

**Usage**:
```bash
coverage run -m pytest
coverage report  # Uses .coveragerc config
```

#### pytest Integration

```bash
# Install pytest-cov
pip install pytest-cov

# Run with coverage
pytest --cov=src --cov-report=html --cov-report=xml

# With branch coverage
pytest --cov=src --cov-branch --cov-report=term-missing
```

**pyproject.toml** configuration:

```toml
[tool.coverage.run]
source = ["src"]
omit = ["*/tests/*", "*/__init__.py"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
]

[tool.pytest.ini_options]
addopts = "--cov=src --cov-report=html --cov-report=xml --cov-report=term-missing"
```

---

### Java: JaCoCo

**JaCoCo**: Java Code Coverage Library

#### Maven Integration

```xml
<!-- pom.xml -->
<project>
    <build>
        <plugins>
            <plugin>
                <groupId>org.jacoco</groupId>
                <artifactId>jacoco-maven-plugin</artifactId>
                <version>0.8.11</version>
                <executions>
                    <!-- Prepare agent -->
                    <execution>
                        <id>prepare-agent</id>
                        <goals>
                            <goal>prepare-agent</goal>
                        </goals>
                    </execution>

                    <!-- Generate report after tests -->
                    <execution>
                        <id>report</id>
                        <phase>test</phase>
                        <goals>
                            <goal>report</goal>
                        </goals>
                    </execution>

                    <!-- Enforce minimum coverage -->
                    <execution>
                        <id>check</id>
                        <goals>
                            <goal>check</goal>
                        </goals>
                        <configuration>
                            <rules>
                                <rule>
                                    <element>BUNDLE</element>
                                    <limits>
                                        <limit>
                                            <counter>LINE</counter>
                                            <value>COVEREDRATIO</value>
                                            <minimum>0.80</minimum>
                                        </limit>
                                    </limits>
                                </rule>
                            </rules>
                        </configuration>
                    </execution>
                </executions>
            </plugin>
        </plugins>
    </build>
</project>
```

**Usage**:
```bash
# Run tests with coverage
mvn clean test

# Reports generated at:
# - HTML: target/site/jacoco/index.html
# - XML: target/site/jacoco/jacoco.xml

# Use with diff-cover
diff-cover target/site/jacoco/jacoco.xml --compare-branch=origin/main
```

#### Gradle Integration

```gradle
// build.gradle
plugins {
    id 'jacoco'
}

jacoco {
    toolVersion = "0.8.11"
}

jacocoTestReport {
    reports {
        xml.enabled true
        html.enabled true
        csv.enabled false
    }
}

test {
    finalizedBy jacocoTestReport
}
```

**Usage**:
```bash
./gradlew test jacocoTestReport

# Reports at build/reports/jacoco/test/
```

---

## Branch Coverage vs. Line Coverage

### Line Coverage

**Definition**: Percentage of executable lines that were executed.

**Example**:
```cpp
if (condition) {
    doSomething();  // Line executed
}
```

If `condition` is true in test: 100% line coverage.

**Limitation**: Doesn't test all branches.

### Branch Coverage

**Definition**: Percentage of decision branches (true/false paths) executed.

**Example**:
```cpp
if (condition) {
    doSomething();
}
```

For 100% branch coverage:
- One test with `condition = true`
- One test with `condition = false`

### Enabling Branch Coverage

**C++ (lcov)**:
```bash
lcov --capture --directory . --output-file coverage.info --rc lcov_branch_coverage=1

genhtml coverage.info --output-directory coverage_html --branch-coverage
```

**Python (coverage.py)**:
```bash
coverage run --branch -m pytest
coverage report --branch
```

**Recommendation**: Use branch coverage for critical code, line coverage for broad metrics.

---

## Mutation Testing: Beyond Coverage

### What is Mutation Testing?

**Definition**: Introduce small changes (mutations) to code, verify tests catch them.

**Example Mutation**:
```cpp
// Original
if (x > 0) { ... }

// Mutant 1: Change > to >=
if (x >= 0) { ... }

// Mutant 2: Change > to <
if (x < 0) { ... }
```

**Killed Mutant**: Tests fail (good!)
**Survived Mutant**: Tests still pass (weak tests!)

**Mutation Score**: `killed / (killed + survived) × 100%`

### Why Mutation Testing?

**Scenario**: You have 100% coverage but...

```python
def calculate_discount(price, is_member):
    if is_member:
        return price * 0.9  # 10% discount
    return price

# Test with 100% line coverage
def test_calculate_discount():
    result = calculate_discount(100, True)
    # No assertion! Test always passes
```

**Coverage says**: ✓ 100%
**Mutation testing says**: ✗ 0% (mutants survive)

**Fix**:
```python
def test_calculate_discount():
    assert calculate_discount(100, True) == 90
    assert calculate_discount(100, False) == 100
```

### Tools by Language

#### C++: mull

```bash
# Install mull
brew install mull-project/mull/mull  # macOS

# Run mutation testing
mull-runner --ld-search-path=/usr/lib ./test_executable
```

#### Python: mutmut

```bash
# Install
pip install mutmut

# Run mutation testing
mutmut run

# Show results
mutmut results

# Show surviving mutants
mutmut show

# Apply mutation to see what changed
mutmut apply 1
```

**Configuration** (setup.cfg):
```ini
[mutmut]
paths_to_mutate=src/
tests_dir=tests/
runner=pytest
```

#### Java: PIT (Pitest)

```xml
<!-- pom.xml -->
<plugin>
    <groupId>org.pitest</groupId>
    <artifactId>pitest-maven</artifactId>
    <version>1.15.0</version>
    <configuration>
        <targetClasses>
            <param>com.example.*</param>
        </targetClasses>
        <targetTests>
            <param>com.example.*Test</param>
        </targetTests>
    </configuration>
</plugin>
```

```bash
mvn test-compile org.pitest:pitest-maven:mutationCoverage
```

### When to Use Mutation Testing

**Use For**:
- Critical paths (auth, payments, data integrity)
- Files with high coverage but repeated bugs
- Validating test quality improvements
- Code review: "Are these tests meaningful?"

**Don't Use For**:
- Initial coverage ramp-up (too slow)
- Low-risk code
- Every PR (too expensive)

**Strategy**: Run mutation testing on high-risk files quarterly.

---

## CI/CD Integration Patterns

### GitHub Actions

```yaml
name: Coverage Check

on:
  pull_request:
  push:
    branches: [main]

jobs:
  coverage:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0

      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y lcov

      - name: Build with coverage
        run: |
          mkdir build-coverage
          cd build-coverage
          cmake -DENABLE_COVERAGE=ON ..
          make -j$(nproc)

      - name: Run tests
        run: |
          cd build-coverage
          ctest --output-on-failure

      - name: Generate coverage
        run: |
          cd build-coverage
          lcov --capture --directory . --output-file coverage.info
          lcov --remove coverage.info '/usr/*' '*/test/*' --output-file coverage_filtered.info

      - name: Check diff coverage
        run: |
          pip install diff-cover
          cd build-coverage
          diff-cover coverage_filtered.info \
            --compare-branch=origin/main \
            --fail-under=80 \
            --html-report=diff-coverage.html

      - name: Upload coverage report
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: coverage-report
          path: build-coverage/coverage_html

      - name: Upload diff coverage
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: diff-coverage
          path: build-coverage/diff-coverage.html
```

### GitLab CI

```yaml
# .gitlab-ci.yml
stages:
  - build
  - test
  - coverage

build:
  stage: build
  script:
    - mkdir build && cd build
    - cmake -DENABLE_COVERAGE=ON ..
    - make -j$(nproc)
  artifacts:
    paths:
      - build/

test:
  stage: test
  dependencies:
    - build
  script:
    - cd build
    - ctest --output-on-failure

coverage:
  stage: coverage
  dependencies:
    - build
    - test
  script:
    - apt-get update && apt-get install -y lcov python3-pip
    - pip3 install diff-cover
    - cd build
    - lcov --capture --directory . --output-file coverage.info
    - lcov --remove coverage.info '/usr/*' '*/test/*' -o coverage_filtered.info
    - genhtml coverage_filtered.info --output-directory coverage_html
    - diff-cover coverage_filtered.info --compare-branch=origin/main --html-report=diff-coverage.html
  artifacts:
    paths:
      - build/coverage_html
      - build/diff-coverage.html
  coverage: '/lines\.*: \d+\.\d+%/'
```

---

## Combining Coverage with Gap Analysis

### Workflow: Comprehensive Gap Assessment

```bash
#!/bin/bash
# comprehensive_gap_analysis.sh

set -e

echo "=== Comprehensive Test Coverage Gap Analysis ==="
echo ""

# Step 1: Generate coverage data
echo "[1/5] Generating coverage data..."
mkdir -p build-coverage
cd build-coverage
cmake -DENABLE_COVERAGE=ON .. >/dev/null
make -j$(nproc) >/dev/null
ctest --output-on-failure >/dev/null
lcov --capture --directory . --output-file coverage.info 2>/dev/null
lcov --remove coverage.info '/usr/*' '*/test/*' -o coverage_filtered.info 2>/dev/null
cd ..

# Step 2: Run diff-cover
echo "[2/5] Checking diff coverage vs. main..."
diff-cover build-coverage/coverage_filtered.info --compare-branch=origin/main

# Step 3: Git-based gap analysis
echo "[3/5] Analyzing test-to-production ratio..."
prod_lines=$(git log --stat --since="1 month ago" -- src/ | \
  grep -E "^ (.*)\|" | awk '{s+=$(NF-3)} END {print s}')
test_lines=$(git log --stat --since="1 month ago" -- test/ | \
  grep -E "^ (.*)\|" | awk '{s+=$(NF-3)} END {print s}')
ratio=$(echo "scale=2; $test_lines / $prod_lines" | bc)
echo "Test:Production ratio = $ratio"

# Step 4: Find untested commits
echo "[4/5] Finding recent commits without test changes..."
git log --name-only --format="%H|%s" --since="1 month ago" | \
awk '/\|/ {commit=$0; has_prod=0; has_test=0; next}
     /^src\// {has_prod=1}
     /^test\// {has_test=1}
     /^$/ {if (has_prod && !has_test) print commit}' | \
head -5

# Step 5: Risk scoring
echo "[5/5] Calculating risk scores for high-churn files..."
python3 scripts/find_fragile_code.py build-coverage/coverage_filtered.info | head -10

echo ""
echo "=== Analysis Complete ==="
echo "Full coverage report: build-coverage/coverage_html/index.html"
```

---

## Best Practices

### 1. Separate Coverage Builds

**Don't**: Mix coverage and production builds

**Do**: Use separate build directories
```bash
mkdir build-release  # Production, optimized
mkdir build-debug    # Development, debuggable
mkdir build-coverage # Coverage, instrumented
```

### 2. Clean Before Coverage Runs

```bash
# Remove old .gcda files
find . -name "*.gcda" -delete

# Run tests
./run_tests

# Generate coverage (only from this run)
```

### 3. Filter Aggressively

```bash
lcov --remove coverage.info \
  '/usr/*' \           # System headers
  '*/test/*' \         # Test files
  '*/third_party/*' \  # External libs
  '*/generated/*' \    # Generated code
  -o coverage_filtered.info
```

### 4. Track Coverage Over Time

```bash
# Store coverage percentage in git
lcov --list coverage_filtered.info | \
  grep "Total:" | \
  awk '{print $2}' > .coverage_percentage

git add .coverage_percentage
git commit -m "Update coverage: $(cat .coverage_percentage)"
```

### 5. Make Coverage Reports Accessible

- Generate HTML reports in CI
- Upload as artifacts
- Post summary in PR comments
- Track trends in dashboards

---

## Troubleshooting Common Issues

### Issue: "gcov: cannot open source file"

**Cause**: Source file path in .gcno doesn't match current location

**Solution**: Build in same directory structure, or use `--source-prefix`

### Issue: Coverage slower than expected

**Cause**: Coverage instrumentation adds overhead

**Solution**: Normal. Use separate coverage builds, run in parallel CI jobs.

### Issue: Intermittent coverage results

**Cause**: Tests modifying global state, coverage data race conditions

**Solution**: Ensure tests are isolated, run sequentially for coverage builds.

### Issue: 100% coverage but tests don't assert anything

**Cause**: Tests execute code but don't verify behavior

**Solution**: Code review test quality, use mutation testing to validate.

---

## Related Resources

- **`diff_cover_workflow.md`**: Using diff-cover with these coverage reports
- **`git_analysis_patterns.md`**: Combining coverage data with git analysis
- **`prioritization_strategies.md`**: Using coverage metrics for prioritization
- **`fundamentals.md`**: Understanding coverage vs. test quality
