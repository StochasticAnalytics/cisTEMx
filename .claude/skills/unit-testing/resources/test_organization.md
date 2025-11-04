# Test Organization

Guide for organizing test files, naming conventions, and project structure patterns.

## Core Principle: Mirror Source Structure

**Test files should mirror the source file structure** for easy navigation and maintenance.

### Why Mirror Structure?

**Benefits:**
- Easy to find tests for any source file
- Clear one-to-one correspondence
- Scales naturally with project growth
- Reduces cognitive load
- Makes missing tests obvious

## C++ Test Organization (cisTEMx)

### Directory Structure

```
src/
├── core/
│   ├── matrix.cpp
│   ├── matrix.h
│   ├── image.cpp
│   ├── image.h
│   └── socket_communication_utils/
│       ├── job_packager.cpp
│       └── job_packager.h
├── gpu/
│   ├── gpu_core_headers/
│   │   └── device_functions.cuh
│   └── core_extensions/
│       └── data_views/
│           └── pointers.h
└── test/
    ├── unit_test_runner.cpp       # Main test runner
    ├── core/
    │   ├── test_matrix.cpp         # Tests for core/matrix.cpp
    │   ├── test_non_wx_functions.cpp
    │   └── socket_communication_utils/
    │       └── test_job_packager.cpp  # Tests for socket_communication_utils/job_packager.cpp
    └── gpu/
        ├── test_gpu.cpp
        ├── gpu_core_headers/
        │   └── test_gpu_core_headers.cpp
        └── core_extensions/
            └── data_views/
                └── test_pointers.cpp
```

### Naming Convention

**Pattern:** `test_<component_name>.cpp`

```
matrix.cpp → test_matrix.cpp
image.cpp → test_image.cpp
job_packager.cpp → test_job_packager.cpp
```

### Test File Structure

```cpp
/*
 * Copyright...
 */

// Include component under test
#include "../../core/matrix.h"

// Include test framework
#include "../../include/catch2/catch.hpp"

/**
 * @brief Unit tests for Matrix class
 *
 * Test coverage:
 * - Construction and initialization
 * - Matrix operations (addition, multiplication)
 * - Edge cases and error conditions
 * - Memory management
 */

// Helper functions (anonymous namespace)
namespace {
    Matrix CreateTestMatrix(int rows, int cols) {
        // ...
    }

    bool MatricesAreAlmostEqual(const Matrix& A, const Matrix& B) {
        // ...
    }
}

// Test cases grouped by functionality
TEST_CASE("Matrix construction", "[matrix][core]") {
    SECTION("default constructor") {
        // ...
    }
    SECTION("copy constructor") {
        // ...
    }
}

TEST_CASE("Matrix operations", "[matrix][core]") {
    SECTION("addition") {
        // ...
    }
    SECTION("multiplication") {
        // ...
    }
}

TEST_CASE("Matrix error handling", "[matrix][negative]") {
    // ...
}
```

## Python Test Organization

### Directory Structure

```
project/
├── src/
│   └── mymodule/
│       ├── __init__.py
│       ├── core.py
│       ├── utils.py
│       └── processing/
│           ├── __init__.py
│           └── filters.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Shared fixtures
│   ├── fixtures/                # Fixture modules
│   │   ├── database.py
│   │   └── mock_api.py
│   ├── test_core.py             # Tests for src/mymodule/core.py
│   ├── test_utils.py            # Tests for src/mymodule/utils.py
│   └── processing/
│       ├── __init__.py
│       └── test_filters.py      # Tests for src/mymodule/processing/filters.py
└── .claude/skills/
    └── my-skill/
        ├── scripts/
        │   ├── process_data.py
        │   └── validate_input.py
        └── tests/
            ├── test_process_data.py
            └── test_validate_input.py
```

### Naming Convention

**Pattern:** `test_<module_name>.py`

```
core.py → test_core.py
utils.py → test_utils.py
filters.py → test_filters.py
```

### Test File Structure

```python
"""Unit tests for mymodule.core

Test coverage:
- Function X: normal operation, edge cases
- Function Y: error handling
- Class Z: initialization, methods
"""

import pytest
from mymodule.core import function_x, function_y, ClassZ

# Fixtures specific to this module
@pytest.fixture
def sample_data():
    """Provide test data for core module"""
    return {"key": "value"}

# Test functions grouped by component
class TestFunctionX:
    """Tests for function_x"""

    def test_normal_operation(self):
        # ...

    def test_edge_case(self):
        # ...

    def test_error_handling(self):
        # ...

class TestClassZ:
    """Tests for ClassZ"""

    def test_initialization(self):
        # ...

    def test_method_a(self):
        # ...
```

### conftest.py Organization

```
tests/
├── conftest.py              # Global fixtures
├── unit/
│   ├── conftest.py          # Unit test fixtures
│   └── test_core.py
└── integration/
    ├── conftest.py          # Integration test fixtures
    └── test_workflows.py
```

**Fixture scope hierarchy:**
- Root `conftest.py`: Session/module-scope fixtures (database, expensive resources)
- Subdirectory `conftest.py`: Directory-specific fixtures
- Test file: Test-specific fixtures

## Bash Test Organization

### Directory Structure

```
scripts/
├── build/
│   ├── configure.sh
│   └── compile.sh
├── deploy/
│   ├── deploy.sh
│   └── rollback.sh
└── tests/
    ├── build/
    │   ├── test_configure.bats
    │   └── test_compile.bats
    └── deploy/
        ├── test_deploy.bats
        └── test_rollback.bats
```

### Naming Convention

**Pattern:** `test_<script_name>.bats`

```
configure.sh → test_configure.bats
deploy.sh → test_deploy.bats
```

### Test File Structure

```bash
#!/usr/bin/env bats

# Test suite for build/configure.sh
#
# Coverage:
# - Argument validation
# - Configuration file generation
# - Error handling

setup() {
    # Per-test setup
    TEST_TEMP_DIR="$(mktemp -d)"
}

teardown() {
    # Per-test cleanup
    rm -rf "$TEST_TEMP_DIR"
}

# Group related tests with descriptive names
@test "configure.sh requires config file argument" {
    # ...
}

@test "configure.sh validates config file exists" {
    # ...
}

@test "configure.sh generates correct output" {
    # ...
}

@test "configure.sh handles missing dependencies gracefully" {
    # ...
}
```

## Skill Script Testing

For scripts in `.claude/skills/`:

```
.claude/skills/
└── my-skill/
    ├── SKILL.md
    ├── scripts/
    │   ├── process_data.py
    │   ├── validate_input.sh
    │   └── helper_functions.py
    └── tests/
        ├── test_process_data.py
        ├── test_validate_input.bats
        └── test_helper_functions.py
```

**Key points:**
- Tests live in `tests/` subdirectory of the skill
- Mirror script organization
- Use appropriate test framework (pytest, bats)
- Test scripts independently of skill execution

## Test Grouping Strategies

### By Component (Recommended)

Group tests by the component they test:

```cpp
TEST_CASE("Matrix construction", "[matrix][core]") { }
TEST_CASE("Matrix operations", "[matrix][core]") { }
TEST_CASE("Matrix error handling", "[matrix][negative]") { }

TEST_CASE("Image loading", "[image][core]") { }
TEST_CASE("Image transformation", "[image][core]") { }
```

### By Feature

Group tests by feature or user story:

```python
class TestUserRegistration:
    def test_valid_registration(self): pass
    def test_duplicate_username_rejected(self): pass
    def test_invalid_email_rejected(self): pass

class TestUserAuthentication:
    def test_valid_login(self): pass
    def test_invalid_password(self): pass
    def test_locked_account(self): pass
```

### By Test Type

Separate unit, integration, and functional tests:

```
tests/
├── unit/           # Fast, isolated tests
│   ├── test_core.py
│   └── test_utils.py
├── integration/    # Tests with dependencies
│   ├── test_database.py
│   └── test_api.py
└── functional/     # End-to-end tests
    └── test_workflows.py
```

## Tagging and Marking

### C++ Catch2 Tags

Use tags for test categorization:

```cpp
TEST_CASE("Fast unit test", "[core][matrix]") { }
TEST_CASE("Slow integration", "[core][matrix][slow]") { }
TEST_CASE("GPU test", "[gpu][kernel]") { }
TEST_CASE("Network test", "[socket][network]") { }
```

**Common tag patterns:**
- Component: `[matrix]`, `[image]`, `[fft]`
- Subsystem: `[core]`, `[gpu]`, `[socket]`
- Speed: `[slow]`
- Type: `[negative]`, `[integration]`

Run specific tags:
```bash
./unit_test_runner "[core]"
./unit_test_runner "[matrix]"
./unit_test_runner "~[slow]"  # Exclude slow tests
```

### Python pytest Markers

```python
import pytest

@pytest.mark.slow
def test_expensive_operation():
    pass

@pytest.mark.integration
def test_database_integration():
    pass

@pytest.mark.gpu
def test_cuda_kernel():
    if not cuda_available():
        pytest.skip("CUDA not available")
```

Run specific markers:
```bash
pytest -m slow
pytest -m "not slow"
pytest -m "integration"
```

### Register custom markers

```python
# pytest.ini or pyproject.toml
[tool.pytest.ini_options]
markers =
    slow: marks tests as slow
    integration: marks tests as integration tests
    gpu: marks tests requiring GPU
```

## File Naming Best Practices

### Do: Clear, Descriptive Names

```
✓ test_matrix.cpp
✓ test_user_authentication.py
✓ test_build_helper.bats
✓ test_fft_implementation.cpp
```

### Don't: Vague or Ambiguous Names

```
✗ test1.cpp
✗ tests.py
✗ my_test.bats
✗ test_stuff.cpp
```

### Pattern Consistency

**C++:** `test_<component>.cpp`
**Python:** `test_<module>.py`
**Bash:** `test_<script>.bats`

## Test Function/Case Naming

### Descriptive Test Names

```cpp
// Good: Clear intent
TEST_CASE("Matrix multiplication with identity matrix returns original", "[matrix]")
TEST_CASE("Parser rejects malformed JSON", "[parser][negative]")

// Bad: Vague intent
TEST_CASE("Test 1", "[matrix]")
TEST_CASE("Check parser", "[parser]")
```

```python
# Good: Clear intent
def test_user_registration_with_valid_email_succeeds():
    pass

def test_duplicate_username_raises_validation_error():
    pass

# Bad: Vague intent
def test_user():
    pass

def test_registration():
    pass
```

### Naming Patterns

**Given-When-Then:**
```python
def test_given_empty_cart_when_add_item_then_cart_has_one_item():
    pass
```

**Should-behavior:**
```python
def test_should_reject_invalid_email_format():
    pass
```

**Component-action-outcome:**
```cpp
TEST_CASE("Parser parses valid JSON successfully", "[parser]")
TEST_CASE("Parser throws ParseError on invalid input", "[parser][negative]")
```

## Documentation in Tests

### File-Level Documentation

```cpp
/**
 * @file test_matrix.cpp
 * @brief Unit tests for Matrix class
 *
 * Test coverage:
 * - Construction: default, copy, move
 * - Operations: addition, subtraction, multiplication
 * - Properties: transpose, determinant, inverse
 * - Edge cases: empty matrices, dimension mismatches
 * - Memory: proper allocation/deallocation
 *
 * Known limitations:
 * - Does not test matrices larger than 1000x1000
 * - GPU tests require CUDA device
 */
```

```python
"""Unit tests for user authentication module

Test coverage:
- Valid login with correct credentials
- Invalid login attempts (wrong password, nonexistent user)
- Account lockout after failed attempts
- Password reset flow
- Session management

Fixtures:
- mock_database: In-memory database for testing
- sample_users: Pre-populated test users

Known issues:
- Tests require mock email service (not real SMTP)
"""
```

### Test-Level Documentation

```cpp
TEST_CASE("Matrix multiplication properties", "[matrix][properties]") {
    // Test mathematical properties that must hold:
    // 1. Associativity: (A*B)*C == A*(B*C)
    // 2. Distributivity: A*(B+C) == A*B + A*C
    // 3. Identity: A*I == A
    //
    // These properties provide stronger guarantees than
    // testing individual multiplication cases.

    SECTION("associativity") {
        // ...
    }
}
```

## Project-Wide Test Organization

### Small Projects

```
project/
├── src/
│   ├── main.cpp
│   ├── utils.cpp
│   └── utils.h
└── tests/
    └── test_utils.cpp
```

### Medium Projects (cisTEMx)

```
project/
├── src/
│   ├── core/
│   ├── gui/
│   ├── gpu/
│   └── test/
│       ├── unit_test_runner.cpp
│       ├── core/
│       ├── gui/
│       └── gpu/
└── scripts/
    └── tests/
```

### Large Projects

```
project/
├── src/
│   └── (source code)
├── tests/
│   ├── unit/
│   │   └── (unit tests mirroring src/)
│   ├── integration/
│   │   └── (integration tests)
│   ├── functional/
│   │   └── (end-to-end tests)
│   └── fixtures/
│       └── (shared test data)
└── scripts/
    └── tests/
```

## Avoiding Common Pitfalls

### Don't: Deeply Nested Test Directories

```
✗ tests/unit/core/math/matrix/operations/test_multiplication.cpp
  (Too deep - hard to navigate)

✓ tests/core/test_matrix.cpp
  (Mirrors src/core/matrix.cpp - easy to find)
```

### Don't: Mix Test Types

```
✗ tests/
    ├── test_unit_matrix.cpp
    ├── test_integration_database.cpp
    ├── test_functional_workflow.cpp
    └── test_unit_image.cpp
  (Mixed types in one directory - confusing)

✓ tests/
    ├── unit/
    │   ├── test_matrix.cpp
    │   └── test_image.cpp
    └── integration/
        └── test_database.cpp
  (Separated by type - clear)
```

### Don't: Inconsistent Naming

```
✗ tests/
    ├── test_matrix.cpp
    ├── ImageTests.cpp
    ├── fft_test.cpp
    └── check_parser.cpp
  (Inconsistent - hard to predict names)

✓ tests/
    ├── test_matrix.cpp
    ├── test_image.cpp
    ├── test_fft.cpp
    └── test_parser.cpp
  (Consistent test_* pattern)
```

## Summary

**Key principles:**
1. **Mirror source structure** - Easy to find tests
2. **Consistent naming** - Predictable, scalable
3. **Descriptive names** - Clear intent
4. **Logical grouping** - Related tests together
5. **Appropriate tags/markers** - Easy filtering

**Patterns:**
- C++: `src/core/matrix.cpp` → `src/test/core/test_matrix.cpp`
- Python: `src/module/core.py` → `tests/test_core.py`
- Bash: `scripts/build.sh` → `scripts/tests/test_build.bats`
- Skills: `.claude/skills/foo/scripts/bar.py` → `.claude/skills/foo/tests/test_bar.py`

**Benefits:**
- Easy navigation between source and tests
- Obvious when tests are missing
- Scales naturally with project growth
- Reduces cognitive load for developers
