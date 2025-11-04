# Build System Integration

Guide for integrating unit tests into build systems and executing them.

## Overview

Unit tests must be integrated into the project's build system to:
- Ensure tests compile with the same flags as production code
- Enable automated test execution
- Support CI/CD pipelines
- Make testing part of the development workflow

## C++ with Autotools (cisTEMx)

cisTEMx uses GNU Autotools (autoconf/automake/libtool) as the primary build system.

### Build System Structure

```
cisTEMx/
├── configure.ac           # Autoconf configuration
├── Makefile.am            # Top-level automake file
└── src/
    ├── Makefile.am        # Source makefile (TESTS DEFINED HERE)
    ├── core/
    │   ├── matrix.cpp
    │   └── matrix.h
    └── test/
        ├── unit_test_runner.cpp    # Main test runner
        └── core/
            └── test_matrix.cpp      # Test files
```

### Adding Tests to Makefile.am

Edit `src/Makefile.am`:

```makefile
# Test programs
bin_PROGRAMS += unit_test_runner

# Main test runner
unit_test_runner_SOURCES = test/unit_test_runner.cpp

# Add each test file
unit_test_runner_SOURCES += test/core/test_matrix.cpp
unit_test_runner_SOURCES += test/core/test_non_wx_functions.cpp
unit_test_runner_SOURCES += test/core/test_curve.cpp

# Add new test file here
unit_test_runner_SOURCES += test/core/test_my_new_component.cpp

# Link against project libraries
unit_test_runner_LDADD = $(COMMON_LDADD)
```

### Build Process

After modifying `Makefile.am`:

```bash
# 1. Regenerate build system (if Makefile.am changed)
./autogen.sh

# 2. Configure with your flags
./configure --enable-cuda  # or other flags

# 3. Build tests
make unit_test_runner

# 4. Run tests
./build/src/unit_test_runner
```

### Test Runner Setup

The `unit_test_runner.cpp` file contains the Catch2 main:

```cpp
// src/test/unit_test_runner.cpp
#define CATCH_CONFIG_MAIN  // This tells Catch2 to provide main()
#include "../../include/catch2/catch.hpp"

// That's it! Individual test files just include catch.hpp
```

Individual test files only need:

```cpp
// src/test/core/test_my_component.cpp
#include "../../core/my_component.h"
#include "../../include/catch2/catch.hpp"

TEST_CASE("My component tests", "[component]") {
    // Tests here
}
```

### Conditional Compilation for GPU Tests

GPU tests should be conditionally compiled:

```makefile
# In Makefile.am
if ENABLE_CUDA
unit_test_runner_SOURCES += test/gpu/test_gpu_operations.cpp
endif
```

And in code:

```cpp
#ifdef cisTEM_USE_CUDA
TEST_CASE("GPU tests", "[gpu]") {
    // GPU test code
}
#endif
```

### Compiler Flags

Tests compile with same flags as production code:

```bash
./configure CXXFLAGS="-O2 -g -Wall"  # Tests use these flags too
```

### Quick Rebuild for Tests

When iterating on tests:

```bash
# Rebuild only test runner (faster)
make unit_test_runner

# Clean and rebuild if needed
make clean && make unit_test_runner
```

## Python with pytest

pytest integrates easily with most build systems through automatic discovery.

### Directory Structure

```
project/
├── src/
│   └── mymodule/
│       ├── __init__.py
│       └── core.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   └── test_core.py
└── setup.py  # or pyproject.toml
```

### Test Discovery

pytest automatically discovers:
- Files matching `test_*.py` or `*_test.py`
- Functions starting with `test_`
- Classes starting with `Test`

No build step required - just run:

```bash
pytest
```

### Integration with setup.py

```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="myproject",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # Production dependencies
    ],
    extras_require={
        "test": [
            "pytest>=7.0",
            "pytest-cov",
        ],
    },
)
```

Install and run tests:

```bash
pip install -e ".[test]"
pytest
```

### Integration with Makefile

For projects with Makefile:

```makefile
# Makefile
.PHONY: test
test:
\tpytest

.PHONY: test-verbose
test-verbose:
\tpytest -v

.PHONY: test-coverage
test-coverage:
\tpytest --cov=src --cov-report=html
```

Run with:

```bash
make test
make test-coverage
```

### CI Integration

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - run: pip install -e ".[test]"
      - run: pytest
```

## Bash with bats

bats tests are executed directly by the bats command.

### Directory Structure

```
project/
├── scripts/
│   ├── build_helper.sh
│   └── deploy.sh
└── tests/
    ├── test_build_helper.bats
    └── test_deploy.bats
```

### Running Tests

```bash
# Run all tests in directory
bats tests/

# Run specific test file
bats tests/test_build_helper.bats

# Parallel execution
bats -j 4 tests/  # 4 parallel jobs
```

### Integration with Makefile

```makefile
# Makefile
.PHONY: test
test:
\tbats tests/

.PHONY: test-verbose
test-verbose:
\tbats -t tests/  # Show timing
```

### CI Integration

```yaml
# .github/workflows/test.yml
name: Shell Script Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install bats
        run: |
          git clone https://github.com/bats-core/bats-core.git
          cd bats-core
          sudo ./install.sh /usr/local
      - name: Run tests
        run: bats tests/
```

## Running Tests

### C++ (Catch2)

```bash
# Run all tests
./unit_test_runner

# Run specific tag
./unit_test_runner "[core]"
./unit_test_runner "[matrix]"

# Exclude tags
./unit_test_runner "~[slow]"
./unit_test_runner "~[gpu]"

# Verbose output
./unit_test_runner -s  # Show successful tests
./unit_test_runner -d yes  # Show duration

# List tests without running
./unit_test_runner -l

# Run specific test
./unit_test_runner "Matrix multiplication"
```

### Python (pytest)

```bash
# Run all tests
pytest

# Run specific file
pytest tests/test_core.py

# Run specific test
pytest tests/test_core.py::test_function_name

# Run by marker
pytest -m slow
pytest -m "not slow"

# Verbose
pytest -v
pytest -vv

# Stop on first failure
pytest -x

# Coverage
pytest --cov=src
```

### Bash (bats)

```bash
# Run all tests
bats tests/

# Run specific file
bats tests/test_script.bats

# Show timing
bats -t tests/

# Count tests
bats -c tests/

# Parallel execution
bats -j 4 tests/
```

## CI/CD Integration

### General Principles

1. **Run tests on every commit** - Catch regressions early
2. **Run tests on pull requests** - Block merges of broken code
3. **Fast feedback** - Run fast tests first, slow tests later
4. **Clear reporting** - Make failures obvious
5. **Deterministic** - Tests must pass consistently

### GitHub Actions Example

```yaml
name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test-cpp:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y autoconf automake libtool
      - name: Build and test
        run: |
          ./autogen.sh
          ./configure
          make unit_test_runner
          ./build/src/unit_test_runner "~[slow]"

  test-python:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -e ".[test]"
      - name: Run tests
        run: pytest

  test-bash:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install bats
        run: |
          git clone https://github.com/bats-core/bats-core.git
          cd bats-core
          sudo ./install.sh /usr/local
      - name: Run tests
        run: bats scripts/tests/
```

## Pre-commit Hooks

Run tests before committing to catch issues early.

### Git Hook Example

```bash
#!/bin/bash
# .git/hooks/pre-commit

# Run fast unit tests before commit
echo "Running unit tests..."

# C++ tests (skip slow)
./build/src/unit_test_runner "~[slow]" || exit 1

# Python tests (skip slow)
pytest -m "not slow" || exit 1

echo "All tests passed!"
```

Make executable:

```bash
chmod +x .git/hooks/pre-commit
```

## Test Organization in Build

### Strategy 1: Single Test Binary

**cisTEMx uses this approach.**

All unit tests compile into one `unit_test_runner` binary:

**Pros:**
- Simple build configuration
- Easy to run all tests
- Shared test infrastructure

**Cons:**
- Longer compile times (all tests recompile together)
- All tests must link together

### Strategy 2: Multiple Test Binaries

Each component gets its own test binary:

```makefile
bin_PROGRAMS += test_matrix test_fft test_image

test_matrix_SOURCES = test/test_matrix.cpp
test_fft_SOURCES = test/test_fft.cpp
test_image_SOURCES = test/test_image.cpp
```

**Pros:**
- Faster incremental builds
- Tests can be run independently
- Easier to parallelize

**Cons:**
- More complex build configuration
- Need to track multiple binaries

### Strategy 3: Test per Source File

One test binary per source file (Google Test style):

```makefile
# Auto-generate test targets
TEST_SOURCES = $(wildcard test/*_test.cpp)
TEST_BINARIES = $(TEST_SOURCES:.cpp=)

$(TEST_BINARIES): %: %.cpp
\t$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)
```

**Pros:**
- Very fast incremental builds
- Clear test/source correspondence

**Cons:**
- Many small binaries
- Complex build rules

## Integration with compile-code Skill

When using the `compile-code` skill for builds:

```bash
# Typical workflow
1. Write test
2. Add test file to src/Makefile.am
3. Use compile-code skill to build
4. Run tests
5. Fix any failures
6. Repeat
```

The `compile-code` skill handles:
- Running autogen.sh/configure if needed
- Invoking make with correct targets
- Parsing build errors
- Suggesting fixes

## Best Practices

### 1. Keep Tests in Build

- Tests should always be buildable
- Don't commit tests that don't compile
- Tests are part of the codebase, not optional

### 2. Fast Default, Slow Optional

- Default build runs fast tests
- Tag slow tests: `[slow]`, `@pytest.mark.slow`
- Run slow tests separately or in CI only

### 3. Fail Fast

- Stop build on first test failure in development
- Run all tests in CI (to see all failures)

### 4. Clear Output

- Show which tests ran
- Clear failure messages
- Include test names in output

### 5. Consistent Environment

- Tests should pass in all supported environments
- Document any environment-specific requirements
- Skip tests gracefully when requirements missing

## Troubleshooting

### Tests Don't Build

**Problem:** Test files not added to Makefile.am

**Solution:**
```makefile
unit_test_runner_SOURCES += test/path/to/test_file.cpp
```
Then run `./autogen.sh && ./configure`

### Tests Can't Find Headers

**Problem:** Include paths not set correctly

**Solution:** Use relative paths from test file location:
```cpp
#include "../../include/catch2/catch.hpp"
#include "../../core/my_component.h"
```

### Tests Fail in CI but Pass Locally

**Problem:** Environment differences

**Solution:**
- Check PATH and environment variables
- Ensure dependencies installed
- Use same compiler/library versions
- Check for race conditions (may not appear locally)

### pytest Can't Find Modules

**Problem:** Python path not set correctly

**Solution:**
```bash
# Install in development mode
pip install -e .

# Or set PYTHONPATH
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
pytest
```

## Summary

**C++ (Autotools):**
- Add tests to `src/Makefile.am`
- Build with `make unit_test_runner`
- Run with `./build/src/unit_test_runner`

**Python (pytest):**
- Auto-discovery, no build step
- Run with `pytest`
- Install with `pip install -e ".[test]"`

**Bash (bats):**
- Run with `bats tests/`
- Install bats if not available
- Integrate with Makefile for convenience

**Best practices:**
- Integrate tests into build system
- Run tests automatically in CI
- Keep fast tests fast
- Fail fast in development
- Clear, actionable failure messages
