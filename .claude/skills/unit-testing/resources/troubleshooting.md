# Troubleshooting Unit Tests

Common problems, solutions, and debugging strategies for unit testing.

## C++ / Catch2 Issues

### Tests Don't Compile

**Problem: Undefined reference to test functions**

```
undefined reference to `TEST_CASE_...`
```

**Cause:** Test file not added to `Makefile.am`

**Solution:**
```makefile
# In src/Makefile.am
unit_test_runner_SOURCES += test/path/to/test_file.cpp
```

Then regenerate build:
```bash
./autogen.sh && ./configure && make
```

---

**Problem: Cannot find Catch2 header**

```
fatal error: catch.hpp: No such file or directory
```

**Cause:** Wrong include path

**Solution:** Use relative path from test file:
```cpp
#include "../../include/catch2/catch.hpp"  // Correct
// NOT: #include "catch.hpp"
```

---

**Problem: Multiple definition of main**

```
multiple definition of `main'
```

**Cause:** Multiple test files define `CATCH_CONFIG_MAIN`

**Solution:** Only define in `unit_test_runner.cpp`:
```cpp
// unit_test_runner.cpp (only here)
#define CATCH_CONFIG_MAIN
#include "../../include/catch2/catch.hpp"

// Other test files (no CATCH_CONFIG_MAIN)
#include "../../include/catch2/catch.hpp"
TEST_CASE(...) { }
```

### Tests Fail to Run

**Problem: Tests not discovered**

**Cause:** TEST_CASE not in global namespace or syntax error

**Solution:** Ensure TEST_CASE at file scope:
```cpp
// Good
TEST_CASE("Test name", "[tag]") { }

// Bad - inside namespace
namespace foo {
    TEST_CASE("Test name", "[tag]") { }  // Not discovered
}
```

---

**Problem: Segfault when running tests**

**Causes:**
- Dereferencing null pointer
- Buffer overflow
- Use after free
- Stack overflow (recursive test)

**Debug steps:**
1. Run with debugger:
```bash
gdb ./unit_test_runner
(gdb) run
(gdb) bt  # Backtrace to see where it crashed
```

2. Run specific test:
```bash
./unit_test_runner "Test name"  # Isolate failing test
```

3. Use valgrind:
```bash
valgrind --leak-check=full ./unit_test_runner
```

### GPU Test Issues

**Problem: GPU tests always skip**

**Causes:**
1. Not compiled with CUDA support
2. No CUDA device available
3. Runtime check fails

**Solutions:**

Check compilation:
```bash
# Ensure configured with CUDA
./configure --enable-cuda
```

Check device availability:
```bash
nvidia-smi  # Should show GPU
```

Verify gating:
```cpp
#ifdef cisTEM_USE_CUDA
TEST_CASE("GPU test", "[gpu]") {
    if (!cuda_device_available()) {
        SKIP("No CUDA device available");
    }
    // Test code
}
#endif
```

---

**Problem: GPU tests fail with CUDA errors**

**Cause:** CUDA API errors not checked

**Solution:** Check all CUDA calls:
```cpp
cudaError_t err = cudaMalloc(&ptr, size);
if (err != cudaSuccess) {
    // Handle error - print cudaGetErrorString(err)
}

// After kernel launch
cudaError_t launch_err = cudaGetLastError();
cudaError_t sync_err = cudaDeviceSynchronize();
```

### Assertion Failures

**Problem: Floating-point comparison fails unexpectedly**

```
FAILED: REQUIRE(result == 0.3)
  with expansion:  0.300000012 == 0.3
```

**Cause:** Floating-point precision

**Solution:** Use `Approx`:
```cpp
REQUIRE(result == Approx(0.3));
REQUIRE(result == Approx(0.3).epsilon(0.01));  // 1% tolerance
```

---

**Problem: Test passes when it should fail**

**Cause:** No assertion in test, or wrong assertion

**Solution:** Ensure every test has meaningful assertions:
```cpp
// Bad - no assertion
TEST_CASE("Test something") {
    doSomething();  // No REQUIRE - always passes
}

// Good
TEST_CASE("Test something") {
    auto result = doSomething();
    REQUIRE(result == expected);
}
```

## Python / pytest Issues

### Tests Not Discovered

**Problem: pytest doesn't find tests**

**Causes:**
1. Wrong file naming
2. Wrong function naming
3. Not in PYTHONPATH

**Solutions:**

Check naming:
```python
# Good
def test_something(): pass
class TestClass: pass

# Bad
def Something(): pass  # Missing test_ prefix
class MyTests: pass    # Missing Test prefix
```

Check file naming:
```
✓ test_module.py
✓ module_test.py
✗ tests.py
✗ testing_module.py
```

Fix PYTHONPATH:
```bash
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
pytest

# Or install in development mode
pip install -e .
```

### Import Errors

**Problem: Cannot import module being tested**

```
ModuleNotFoundError: No module named 'mymodule'
```

**Solutions:**

1. Install in development mode:
```bash
pip install -e .
```

2. Set PYTHONPATH:
```bash
export PYTHONPATH="${PWD}/src"
```

3. Add to sys.path in conftest.py:
```python
# tests/conftest.py
import sys
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))
```

### Fixture Issues

**Problem: Fixture not found**

```
fixture 'my_fixture' not found
```

**Causes:**
1. Fixture not in scope
2. Typo in fixture name
3. Fixture in wrong conftest.py

**Solutions:**

Check fixture scope:
```
tests/
├── conftest.py         # Available to all tests
├── unit/
│   ├── conftest.py     # Available to tests in unit/
│   └── test_foo.py     # Can use fixtures from both
└── integration/
    └── test_bar.py     # Only sees tests/conftest.py
```

List available fixtures:
```bash
pytest --fixtures
pytest --fixtures test_file.py  # For specific file
```

---

**Problem: Fixture runs too often**

**Cause:** Wrong fixture scope

**Solution:** Set appropriate scope:
```python
@pytest.fixture(scope="function")  # Default - per test
@pytest.fixture(scope="module")    # Per file
@pytest.fixture(scope="session")   # Once per test run
```

### Mocking Issues

**Problem: Mock not being called**

**Cause:** Wrong object being mocked

**Solution:** Mock where it's used, not where it's defined:
```python
# module.py
from external import api_call

def my_function():
    return api_call()

# test.py - Wrong
@patch('external.api_call')  # Doesn't work

# test.py - Correct
@patch('module.api_call')  # Mock where it's imported
```

---

**Problem: Mock returns Mock object instead of value**

**Cause:** Forgot to set return_value

**Solution:**
```python
mock_func = Mock()  # Returns Mock object by default
mock_func.return_value = 42  # Now returns 42
```

## Bash / bats Issues

### Tests Not Running

**Problem: bats not found**

```
bash: bats: command not found
```

**Solution:** Install bats:
```bash
git clone https://github.com/bats-core/bats-core.git
cd bats-core
sudo ./install.sh /usr/local
```

---

**Problem: Test file not executable**

**Cause:** Missing shebang or execute permission

**Solution:**
```bash
# Add shebang
echo '#!/usr/bin/env bats' | cat - test.bats > temp && mv temp test.bats

# Make executable
chmod +x test.bats
```

### Assertion Failures

**Problem: Test fails but output is unclear**

**Solution:** Add descriptive output:
```bash
@test "script processes file" {
    run ./script.sh input.txt

    echo "Status: $status"
    echo "Output: $output"

    [ "$status" -eq 0 ]
}
```

---

**Problem: Pattern matching doesn't work**

```
[[ "$output" =~ "pattern" ]]  # Fails unexpectedly
```

**Cause:** Special regex characters not escaped

**Solution:**
```bash
# Escape special characters
[[ "$output" =~ "file\.txt" ]]  # . is literal

# Or use glob pattern
[[ "$output" == *"substring"* ]]
```

### Cleanup Issues

**Problem: Tests leave files behind**

**Cause:** teardown() not cleaning up

**Solution:**
```bash
setup() {
    TEST_TEMP_DIR="$(mktemp -d)"
}

teardown() {
    # Always clean up, even if test fails
    [ -n "$TEST_TEMP_DIR" ] && rm -rf "$TEST_TEMP_DIR"
}
```

---

**Problem: teardown() not running**

**Cause:** Test exits early or crashes

**Solution:** Use trap for critical cleanup:
```bash
setup() {
    TEST_TEMP_DIR="$(mktemp -d)"
    trap "rm -rf '$TEST_TEMP_DIR'" EXIT
}
```

## Cross-Framework Issues

### Flaky Tests

**Problem: Tests pass sometimes, fail other times**

**Causes:**
1. Race conditions
2. Time dependencies
3. Randomness without fixed seed
4. Shared global state
5. External dependencies (network, filesystem)

**Solutions:**

Fix race conditions:
```cpp
// Add proper synchronization
std::lock_guard<std::mutex> lock(mutex);
```

Mock time:
```python
with patch('time.time', return_value=1234567890):
    # Test with fixed time
```

Use fixed seeds:
```cpp
std::mt19937 rng(42);  // Fixed seed
```

Isolate state:
```cpp
// Each test creates own state
TEST_CASE("Test") {
    LocalState state;  // Not global
}
```

### Slow Tests

**Problem: Tests take too long**

**Solutions:**

1. Tag slow tests:
```cpp
TEST_CASE("Slow test", "[slow]") { }
```

2. Run fast tests by default:
```bash
./unit_test_runner "~[slow]"
pytest -m "not slow"
```

3. Optimize test data:
```cpp
// Use small data
std::vector<float> data(100);  // Not 1000000
```

4. Mock expensive operations:
```python
@patch('expensive_operation')
def test_something(mock_op):
    mock_op.return_value = "instant"
```

### CI Failures

**Problem: Tests pass locally but fail in CI**

**Causes:**
1. Environment differences
2. Missing dependencies
3. Timing issues
4. File path issues

**Solutions:**

Match environments:
```bash
# Use same compiler version
# Use same library versions
# Check environment variables
```

Check dependencies:
```yaml
# In CI config
before_script:
  - apt-get install required-libs
```

Fix timing:
```python
# Increase timeouts for CI
if os.getenv('CI'):
    TIMEOUT = 60
else:
    TIMEOUT = 10
```

Use relative paths:
```cpp
// Not: "/home/user/project/file.txt"
// Use: relative path or __FILE__-based
```

### Debug Strategies

#### Isolate the Problem

1. Run specific test:
```bash
./unit_test_runner "Test name"
pytest tests/test_file.py::test_function
bats tests/test_file.bats
```

2. Add debug output:
```cpp
std::cout << "Debug: value = " << value << std::endl;
```

3. Use debugger:
```bash
gdb ./unit_test_runner
(gdb) break test_file.cpp:42
(gdb) run "Test name"
```

#### Verify Assumptions

1. Print intermediate values
2. Check assertions are meaningful
3. Verify test setup is correct

#### Simplify

1. Remove complexity from test
2. Test smaller pieces
3. Add helper tests

#### Check Documentation

1. Framework documentation
2. Project testing patterns
3. Similar tests in codebase

## Getting Help

### What Information to Provide

When asking for help:

1. **Exact error message**
2. **Minimal reproduction**
3. **Environment** (OS, compiler, versions)
4. **What you've tried**

### Where to Look

1. Framework documentation
2. Project test examples
3. Stack Overflow (check dates - recent answers)
4. GitHub issues for framework

## Summary

**Common patterns:**
- Most issues are environment, configuration, or naming
- Check the basics first (paths, naming, setup)
- Use verbose output for debugging
- Isolate failing tests
- Read error messages carefully

**Prevention:**
- Follow naming conventions
- Write clear, focused tests
- Keep tests isolated
- Check assertions are meaningful
- Test incrementally
