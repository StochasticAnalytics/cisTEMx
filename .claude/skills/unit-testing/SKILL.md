---
name: unit-testing
description: Write unit tests for individual functions/methods in isolation across C++/Catch2, Python/pytest, Bash/bats, and CUDA. Use when testing single components, implementing TDD, or adding regression tests. For Claude's direct use. Covers testing frameworks, best practices, and code organization. NOT for integration/functional tests.
---

# Unit Testing

Write focused unit tests that verify individual functions or methods in isolation.

## Critical Scope Distinction

**This skill is ONLY for UNIT TESTING** - testing isolated functions/methods with minimal dependencies.

**NOT for:**
- Integration testing (testing multiple components together)
- Functional testing (end-to-end workflows like `console_tests.cpp`, `samples_functional_testing.cpp`)
- System testing (full application behavior)

If testing interactions between components or complex workflows, use different approaches.

## When to Use This Skill

- Testing individual functions or methods
- Verifying component behavior in isolation
- Implementing test-driven development (TDD)
- Adding regression tests for bug fixes
- Ensuring code maintainability through automated tests
- Writing tests for new code or refactored code

## Quick Start by Language

### C++ with Catch2
```cpp
#include "../../include/catch2/catch.hpp"

TEST_CASE("Matrix multiplication works correctly", "[matrix][core]") {
    SECTION("handles identity matrix") {
        Matrix A = CreateTestMatrix(3, 3);
        Matrix I = IdentityMatrix(3);
        Matrix result = A * I;
        REQUIRE(MatricesAreAlmostEqual(result, A));
    }
}
```
For comprehensive guide, see `resources/cpp_catch2.md`.

### Python with pytest
```python
def test_data_processor_handles_empty_input():
    # Arrange
    processor = DataProcessor()

    # Act
    result = processor.process([])

    # Assert
    assert result.status == "success"
    assert len(result.items) == 0
```
For comprehensive guide, see `resources/python_pytest.md`.

### Bash with bats
```bash
@test "script validates required argument" {
    run ./build_helper.sh
    [ "$status" -eq 1 ]
    [[ "$output" =~ "Error: Missing argument" ]]
}
```
For comprehensive guide, see `resources/bash_bats.md`.

### CUDA Testing
Use `__host__ __device__` functions for CPU+GPU testing:
```cpp
__host__ __device__ float complexCalculation(float a, float b) {
    return (a * a + b * b) / (a + b);
}

TEST_CASE("Complex calculation", "[cuda][math]") {
    REQUIRE(complexCalculation(3.0f, 4.0f) == Approx(25.0f/7.0f));
}
```
For comprehensive strategies, see `resources/cuda_testing.md`.

## Test File Organization

**Mirror source file structure:**
- `src/core/matrix.cpp` → `src/test/core/test_matrix.cpp`
- `.claude/skills/foo/scripts/bar.py` → `.claude/skills/foo/tests/test_bar.py`
- `scripts/build_helper.sh` → `scripts/tests/test_build_helper.bats`

For detailed organization patterns, see `resources/test_organization.md`.

## Testing Fundamentals

All unit tests should follow core principles:
- **AAA Pattern**: Arrange → Act → Assert
- **FIRST Principles**: Fast, Isolated, Repeatable, Self-validating, Timely
- **Test Isolation**: Each test runs independently with no shared state

For deep dive into methodology, see `resources/tdd_fundamentals.md`.

## Build Integration

Tests must be integrated into the build system:
- C++: Add to `src/Makefile.am`
- Python: pytest auto-discovers `test_*.py` files
- Bash: Run bats on `*.bats` files

For detailed integration steps, see `resources/build_integration.md`.

## Available Resources

- **`resources/cpp_catch2.md`** - C++ testing with Catch2 v3 framework
- **`resources/python_pytest.md`** - Python testing with pytest framework
- **`resources/bash_bats.md`** - Bash script testing with bats
- **`resources/cuda_testing.md`** - Strategies for testing CUDA code
- **`resources/tdd_fundamentals.md`** - TDD methodology, AAA pattern, FIRST principles
- **`resources/test_organization.md`** - File structure, naming conventions, discovery
- **`resources/build_integration.md`** - Build system integration and test execution
- **`resources/troubleshooting.md`** - Common issues and solutions
- **`resources/citations.md`** - Sources and references for maintenance

## Available Templates

- **`templates/catch2_test_template.cpp`** - C++ Catch2 test file template
- **`templates/pytest_test_template.py`** - Python pytest file template
- **`templates/bats_test_template.bats`** - Bash bats test file template
- **`templates/cuda_test_template.cpp`** - CUDA testable code pattern

## Helper Scripts

- **`scripts/create_test_file.py`** - Generate test file from template

## Integration with Other Skills

- Use **`compile-code`** skill to build and run C++ unit tests
- Document test creation decisions in **`lab-notebook`** skill
- Commit test changes using **`git-commit`** skill
