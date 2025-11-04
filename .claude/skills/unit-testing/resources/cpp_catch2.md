# C++ Unit Testing with Catch2

Comprehensive guide for writing C++ unit tests using Catch2 v3 framework in the cisTEMx project.

## Framework Overview

**Catch2** is a modern, header-only C++ testing framework with:
- Natural test case syntax using descriptive strings
- Section-based test organization
- Expression template decomposition for clear failures
- Tag-based test filtering for CI
- Minimal boilerplate

**cisTEMx uses Catch2** located at `/workspaces/cisTEMx/include/catch2/catch.hpp` (single-header version).

## Testing Philosophy

### Write Non-Trivial Tests Only

**Every test must have clear purpose.** No filler tests, no trivial assertions.

Ask yourself:
- What invariant does this test defend?
- What bug class could this catch?
- What edge case does this exercise?

If you can't answer these, don't write the test.

### Risk-Based Test Selection

Prioritize testing for:
- Binary parsers and serialization
- Socket I/O and network protocols
- Concurrency primitives (mutexes, queues, atomics)
- GPU staging buffers and memory transfers
- Kernel launch parameter validation
- Complex mathematical operations (FFT, matrix operations)

### Property-Based Thinking

Design tests around properties that must hold:
- **Serialization round-trip**: `deserialize(serialize(x)) == x`
- **Idempotence**: `f(f(x)) == f(x)`
- **Commutativity**: `f(a, b) == f(b, a)`
- **Invariant preservation**: `invariant(x) => invariant(transform(x))`

## cisTEMx Test Organization

### File Structure

Tests mirror source structure:
```
src/
├── core/
│   ├── matrix.cpp           # Source file
│   └── matrix.h
└── test/
    ├── unit_test_runner.cpp  # Main test runner
    └── core/
        └── test_matrix.cpp   # Test file
```

### Test Runner

The `unit_test_runner.cpp` file contains the main entry point:
```cpp
#define CATCH_CONFIG_MAIN
#include "../../include/catch2/catch.hpp"
```

This is compiled once; individual test files only need `#include "../../include/catch2/catch.hpp"`.

### Build Integration

Tests are defined in `src/Makefile.am`:
```makefile
bin_PROGRAMS = unit_test_runner

unit_test_runner_SOURCES = test/unit_test_runner.cpp
unit_test_runner_SOURCES += test/core/test_matrix.cpp
unit_test_runner_SOURCES += test/core/test_non_wx_functions.cpp
# Add your new test files here
```

After modifying `Makefile.am`, run:
```bash
./autogen.sh
./configure [your-flags]
make
```

## Catch2 v3 Best Practices

### Basic Test Structure

```cpp
#include "../../include/catch2/catch.hpp"
#include "../core/matrix.h"

TEST_CASE("Matrix multiplication works correctly", "[matrix][core]") {
    SECTION("handles identity matrix") {
        // Arrange
        Matrix A = CreateTestMatrix(3, 3);
        Matrix I = IdentityMatrix(3);

        // Act
        Matrix result = A * I;

        // Assert
        REQUIRE(MatricesAreAlmostEqual(result, A));
    }

    SECTION("handles zero matrix") {
        Matrix A = CreateTestMatrix(3, 3);
        Matrix Z = ZeroMatrix(3, 3);
        Matrix result = A * Z;
        REQUIRE(MatricesAreAlmostEqual(result, Z));
    }
}
```

### TEST_CASE and SECTION

**TEST_CASE**:
- Use descriptive strings (not C++ identifiers)
- One TEST_CASE per logical feature or behavior
- Can have multiple SECTION blocks within

**SECTION**:
- Each SECTION runs independently with fresh setup
- Enables alternative execution paths without test duplication
- Can nest SECTIONs for hierarchical testing

```cpp
TEST_CASE("Parser handles various inputs", "[parser]") {
    Parser p;  // Created once per SECTION

    SECTION("valid input") {
        // p is fresh
        REQUIRE(p.parse("valid") == SUCCESS);
    }

    SECTION("invalid input") {
        // p is fresh again
        REQUIRE_THROWS(p.parse("invalid"));
    }
}
```

### Assertions

**REQUIRE vs CHECK**:
- `REQUIRE(expr)` - Stops test on failure (critical assertion)
- `CHECK(expr)` - Continues on failure (non-critical)

**Common assertions**:
```cpp
REQUIRE(x == y);                        // Equality
REQUIRE(x != y);                        // Inequality
REQUIRE(x < y);                         // Comparison
REQUIRE_FALSE(condition);               // Negation
REQUIRE_THROWS(function());             // Any exception
REQUIRE_THROWS_AS(function(), Error);   // Specific exception
REQUIRE_NOTHROW(function());            // No exception
```

**Floating-point comparison**:
```cpp
#include <cmath>

REQUIRE(result == Approx(expected));
REQUIRE(result == Approx(expected).epsilon(0.01));  // 1% tolerance
REQUIRE(result == Approx(expected).margin(0.001));  // Absolute margin
```

**cisTEMx helpers** (from `test_matrix.cpp`):
```cpp
REQUIRE(FloatsAreAlmostTheSame(a, b));
REQUIRE(DoublesAreAlmostTheSame(a, b));
REQUIRE(MatricesAreAlmostEqual(A, B));
```

### Tags

Tags enable selective test execution:
```cpp
TEST_CASE("Fast unit test", "[core][matrix]") { }
TEST_CASE("Slow integration", "[slow][matrix]") { }
TEST_CASE("GPU test", "[gpu][kernel]") { }
```

Run specific tags:
```bash
./unit_test_runner "[core]"        # Run core tests
./unit_test_runner "[matrix]"      # Run matrix tests
./unit_test_runner "~[slow]"       # Skip slow tests
./unit_test_runner "[gpu][core]"   # Run tests with both tags
```

**Common cisTEMx tags**:
- `[core]` - Core functionality
- `[gpu]` - GPU-specific tests
- `[slow]` - Tests >200ms
- `[socket]` - Network tests
- `[parser]` - Parsing/serialization

### Helper Functions

Extract common test setup into helpers:
```cpp
namespace {
    Matrix CreateTestMatrix(int rows, int cols) {
        Matrix m(rows, cols);
        // Initialize with test data
        return m;
    }

    bool MatricesAreAlmostEqual(const Matrix& A, const Matrix& B,
                                double tolerance = 1e-6) {
        if (A.rows() != B.rows() || A.cols() != B.cols()) return false;
        for (int i = 0; i < A.size(); ++i) {
            if (std::abs(A[i] - B[i]) > tolerance) return false;
        }
        return true;
    }
}
```

See `src/test/core/test_matrix.cpp` for excellent examples.

### Parameterized Tests

Use `GENERATE` for data-driven tests:
```cpp
TEST_CASE("Function works with various inputs", "[core]") {
    auto input = GENERATE(1, 2, 3, 5, 8, 13);
    REQUIRE(fibonacci(input) > 0);
}

// Table-driven
TEST_CASE("Factorial calculation", "[math]") {
    auto [input, expected] = GENERATE(table<int, int>({
        {0, 1},
        {1, 1},
        {5, 120},
        {10, 3628800}
    }));

    REQUIRE(factorial(input) == expected);
}
```

**Use deterministic generators only** - no random values in tests.

### Type-Parameterized Tests

Test templates with multiple types:
```cpp
TEMPLATE_TEST_CASE("Vector works with numeric types", "[vector]",
                   int, float, double) {
    Vector<TestType> v(10);
    REQUIRE(v.size() == 10);
    REQUIRE(v[0] == TestType{});
}
```

### Fixtures (When Needed)

Catch2 is less fixture-heavy than other frameworks, but you can use them:
```cpp
struct DatabaseFixture {
    DatabaseFixture() {
        db.connect("test.db");
    }
    ~DatabaseFixture() {
        db.disconnect();
    }
    Database db;
};

TEST_CASE_METHOD(DatabaseFixture, "Database operations", "[db]") {
    REQUIRE(db.isConnected());
}
```

Often, SECTION blocks with local setup are clearer than fixtures.

## GPU Testing Patterns

### Compilation and Runtime Gating

**Always gate GPU tests properly:**
```cpp
#ifdef cisTEM_USE_CUDA
TEST_CASE("GPU kernel validation", "[gpu][kernel]") {
    if (!cuda_device_available()) {
        SKIP("No CUDA device available");
    }

    // Test implementation
}
#endif
```

This ensures:
- Tests don't compile on non-CUDA builds
- Tests skip gracefully when no GPU present
- Clear messaging for why tests are skipped

### GPU Test Patterns

**Pattern 1: Device memory operations**
```cpp
#ifdef cisTEM_USE_CUDA
TEST_CASE("Device pointer operations", "[gpu][memory]") {
    if (!cuda_device_available()) {
        SKIP("No CUDA device available");
    }

    DevicePointerArray<float2> testPtr;
    testPtr.resize(4);
    REQUIRE(testPtr.size() == 4);

    // Test device operations
    // Keep allocations small and time-bounded

    testPtr.Deallocate();
}
#endif
```

**Pattern 2: CPU reference comparison**
```cpp
#ifdef cisTEM_USE_CUDA
TEST_CASE("GPU kernel matches CPU reference", "[gpu][validation]") {
    if (!cuda_device_available()) {
        SKIP("No CUDA device available");
    }

    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};

    // CPU reference implementation
    std::vector<float> cpu_result = cpuImplementation(input);

    // GPU implementation
    std::vector<float> gpu_result = gpuKernel(input);

    // Compare results
    REQUIRE(cpu_result.size() == gpu_result.size());
    for (size_t i = 0; i < cpu_result.size(); ++i) {
        REQUIRE(cpu_result[i] == Approx(gpu_result[i]));
    }
}
#endif
```

**Pattern 3: Kernel launch parameter validation**
```cpp
#ifdef cisTEM_USE_CUDA
TEST_CASE("CUDA kernel parameter validation", "[gpu][validation]") {
    if (!cuda_device_available()) {
        SKIP("No CUDA device available");
    }

    SECTION("grid dimensions within limits") {
        KernelLaunchParams params;
        params.grid_dim = {65536, 65536, 1}; // At limit
        REQUIRE(params.validate());

        params.grid_dim = {65537, 1, 1}; // Over limit
        REQUIRE_FALSE(params.validate());
    }
}
#endif
```

### GPU Test Best Practices

- Keep device allocations small (< 1MB)
- Use bounded timeouts for kernel launches
- Always clean up device memory
- Tag with `[gpu]` for selective execution
- Target <200ms; tag longer tests with `[slow]`

## Data Realism Requirements

### Use Real Project Types

Don't test with toy examples. Use actual production types:
- Message structs with explicit endianness
- Binary protocol parsers
- `span` and buffer views
- Kernel launch validators
- Thread-safe queues
- Image data structures

### Realistic Test Data

```cpp
TEST_CASE("Message parser handles realistic payloads", "[parser]") {
    // Use realistic field ranges from production
    MessageHeader header;
    header.magic = 0x54454D43;  // 'TEMC' in hex
    header.version = 1;
    header.payload_size = 1024;
    header.checksum = compute_checksum(payload);

    // Not: magic = 1, version = 1, payload_size = 1
}
```

### Adversarial Inputs

**Every test suite must include at least one adversarial case:**
```cpp
TEST_CASE("Parser handles malformed input", "[parser][negative]") {
    SECTION("truncated header") {
        std::vector<uint8_t> truncated = {0x01, 0x02}; // Need 8 bytes
        REQUIRE_THROWS_AS(parse(truncated), ParseError);
    }

    SECTION("invalid magic number") {
        std::vector<uint8_t> bad_magic = create_frame_with_magic(0xDEADBEEF);
        REQUIRE_THROWS_AS(parse(bad_magic), InvalidMagicError);
    }

    SECTION("oversized payload") {
        std::vector<uint8_t> huge = create_frame_with_size(1ULL << 32);
        REQUIRE_THROWS_AS(parse(huge), OversizeError);
    }
}
```

### Endianness Testing

For binary protocols:
```cpp
TEST_CASE("Serialization handles endianness correctly", "[parser][endian]") {
    SECTION("little-endian round trip") {
        Message msg = create_test_message();
        auto bytes = serialize_le(msg);
        auto decoded = deserialize_le(bytes);
        REQUIRE(decoded == msg);
    }

    SECTION("big-endian round trip") {
        Message msg = create_test_message();
        auto bytes = serialize_be(msg);
        auto decoded = deserialize_be(bytes);
        REQUIRE(decoded == msg);
    }
}
```

## Test Isolation and Reproducibility

### Each Test Runs Independently

**Never rely on:**
- Global state
- Test execution order
- Previous test results
- External files or services (without explicit setup)

**Good pattern**:
```cpp
TEST_CASE("Independent tests", "[core]") {
    SECTION("test 1") {
        int x = 5;  // Local state
        REQUIRE(process(x) == 10);
    }

    SECTION("test 2") {
        int x = 7;  // Fresh local state
        REQUIRE(process(x) == 14);
    }
}
```

**Bad pattern**:
```cpp
static int global_counter = 0;  // DON'T DO THIS

TEST_CASE("Test A") {
    global_counter++;
    REQUIRE(global_counter == 1);  // Breaks if tests run in different order
}
```

### Filesystem Isolation

Use temporary directories:
```cpp
#include <cstdlib>
#include <filesystem>

TEST_CASE("File operations", "[io]") {
    // Create temp directory
    auto temp_dir = std::filesystem::temp_directory_path() / "test_XXXXXX";
    // ... create unique temp dir ...

    // Test operations in temp_dir

    // Clean up
    std::filesystem::remove_all(temp_dir);
}
```

### Network Isolation

For socket tests:
- Use loopback interface only (`127.0.0.1`)
- Use ephemeral ports (let OS assign)
- Set bounded timeouts
- Skip if network unavailable

```cpp
TEST_CASE("Socket communication", "[socket]") {
    Socket server(Socket::LOOPBACK, 0);  // Port 0 = ephemeral
    if (!server.bind()) {
        SKIP("Could not bind to loopback");
    }

    server.setTimeout(std::chrono::milliseconds(100));
    // Test with bounded timeout
}
```

### Deterministic Randomness

If tests need randomness, use deterministic seeds:
```cpp
TEST_CASE("Random data generation", "[random]") {
    std::mt19937 rng(42);  // Fixed seed
    auto value = generate_random(rng);
    // Test is reproducible
}
```

## Negative Testing Requirements

Every test suite must include tests for:

### 1. Malformed Input

```cpp
TEST_CASE("Parser rejects malformed input", "[parser][negative]") {
    Parser p;
    REQUIRE_THROWS(p.parse(""));
    REQUIRE_THROWS(p.parse("incomplete"));
    REQUIRE_THROWS(p.parse("\xFF\xFF\xFF\xFF"));
}
```

### 2. Boundary Conditions

```cpp
TEST_CASE("Buffer handles boundary conditions", "[buffer][negative]") {
    Buffer buf(10);

    SECTION("access at boundary") {
        REQUIRE_NOTHROW(buf[9]);  // Last valid index
    }

    SECTION("access beyond boundary") {
        REQUIRE_THROWS(buf[10]);  // One past end
    }

    SECTION("empty buffer") {
        Buffer empty(0);
        REQUIRE_THROWS(empty[0]);
    }
}
```

### 3. Resource Exhaustion

```cpp
TEST_CASE("Memory allocation handles exhaustion", "[memory][negative]") {
    SECTION("reasonable allocation succeeds") {
        REQUIRE_NOTHROW(allocate(1024));
    }

    SECTION("excessive allocation fails gracefully") {
        // Don't actually allocate, test the validation
        REQUIRE_THROWS_AS(validate_allocation(1ULL << 60), AllocationError);
    }
}
```

### 4. Race Conditions (for concurrent code)

```cpp
TEST_CASE("ThreadSafeQueue handles concurrent access", "[concurrent][queue]") {
    ThreadSafeQueue<int> queue;
    std::atomic<int> push_count{0};
    std::atomic<int> pop_count{0};

    // Launch multiple producer threads
    std::vector<std::thread> producers;
    for (int i = 0; i < 4; ++i) {
        producers.emplace_back([&]() {
            for (int j = 0; j < 100; ++j) {
                queue.push(j);
                push_count++;
            }
        });
    }

    // Launch multiple consumer threads
    std::vector<std::thread> consumers;
    for (int i = 0; i < 4; ++i) {
        consumers.emplace_back([&]() {
            while (pop_count < 400) {
                int val;
                if (queue.try_pop(val)) {
                    pop_count++;
                }
            }
        });
    }

    for (auto& t : producers) t.join();
    for (auto& t : consumers) t.join();

    REQUIRE(push_count == 400);
    REQUIRE(pop_count == 400);
    REQUIRE(queue.empty());
}
```

### 5. Invalid State Transitions

```cpp
TEST_CASE("State machine rejects invalid transitions", "[state][negative]") {
    StateMachine sm;

    sm.transition(State::INIT, State::READY);  // Valid
    REQUIRE_THROWS(sm.transition(State::READY, State::INIT));  // Invalid
}
```

## Performance Expectations

### Fast Tests by Default

- Target execution time: **<200ms per TEST_CASE**
- Tag slower tests: `[slow]`
- Minimize disk I/O, network operations, large allocations

### Tagging Slow Tests

```cpp
TEST_CASE("Fast unit test", "[core]") {
    // <200ms
}

TEST_CASE("Slower integration", "[core][slow]") {
    // >200ms but still unit-level
}
```

Run fast tests only:
```bash
./unit_test_runner "~[slow]"
```

## Example: Complete Test File

From `src/test/core/test_matrix.cpp` (simplified):

```cpp
/*
 * Copyright...
 */

#include "../../core/matrix.h"
#include "../../include/catch2/catch.hpp"

namespace {
    // Helper functions
    Matrix CreateTestMatrix(int rows, int cols) {
        Matrix m(rows, cols);
        for (int i = 0; i < rows * cols; ++i) {
            m[i] = static_cast<float>(i);
        }
        return m;
    }

    bool MatricesAreAlmostEqual(const Matrix& A, const Matrix& B,
                                double tolerance = 1e-6) {
        if (A.rows() != B.rows() || A.cols() != B.cols()) return false;
        for (int i = 0; i < A.size(); ++i) {
            if (std::abs(A[i] - B[i]) > tolerance) return false;
        }
        return true;
    }
}

TEST_CASE("Matrix construction", "[matrix][core]") {
    SECTION("default constructor creates zero matrix") {
        Matrix m(3, 3);
        REQUIRE(m.rows() == 3);
        REQUIRE(m.cols() == 3);
        for (int i = 0; i < 9; ++i) {
            REQUIRE(m[i] == 0.0f);
        }
    }

    SECTION("copy constructor creates independent copy") {
        Matrix A = CreateTestMatrix(2, 2);
        Matrix B(A);
        B[0] = 999.0f;
        REQUIRE(A[0] != 999.0f);  // A unchanged
    }
}

TEST_CASE("Matrix multiplication", "[matrix][core]") {
    SECTION("identity matrix property") {
        Matrix A = CreateTestMatrix(3, 3);
        Matrix I = Matrix::Identity(3);
        Matrix result = A * I;
        REQUIRE(MatricesAreAlmostEqual(result, A));
    }

    SECTION("zero matrix property") {
        Matrix A = CreateTestMatrix(3, 3);
        Matrix Z(3, 3);  // Zero matrix
        Matrix result = A * Z;
        REQUIRE(MatricesAreAlmostEqual(result, Z));
    }

    SECTION("throws on dimension mismatch") {
        Matrix A(2, 3);
        Matrix B(2, 2);  // Wrong dimensions
        REQUIRE_THROWS(A * B);
    }
}
```

## Running Tests

### Build and Run All Tests

```bash
cd /workspaces/cisTEMx/build
make unit_test_runner
./src/unit_test_runner
```

### Run Specific Tests

```bash
# By tag
./src/unit_test_runner "[core]"
./src/unit_test_runner "[matrix]"

# Exclude tags
./src/unit_test_runner "~[slow]"
./src/unit_test_runner "~[gpu]"

# By name (partial match)
./src/unit_test_runner "Matrix"

# Multiple filters
./src/unit_test_runner "[core]" "~[slow]"
```

### Verbose Output

```bash
./src/unit_test_runner -s  # Show successful tests
./src/unit_test_runner -d yes  # Show duration
```

## Quality Checklist

Before finalizing tests, verify:

- [ ] Every test has clear purpose (no filler tests)
- [ ] Test names clearly describe what is being tested
- [ ] Negative and boundary cases included
- [ ] Tests are deterministic and reproducible
- [ ] GPU tests properly gated with `#ifdef` and runtime checks
- [ ] Execution time <200ms (or tagged `[slow]`)
- [ ] No global state dependencies
- [ ] Proper cleanup of resources (memory, files, sockets)
- [ ] Clear comments explaining test intent
- [ ] Appropriate tags for CI selection (`[core]`, `[gpu]`, `[slow]`, etc.)
- [ ] Helper functions extracted for reusability
- [ ] Test file added to `src/Makefile.am`

## Troubleshooting

### Tests Don't Compile

1. Check include path: `#include "../../include/catch2/catch.hpp"`
2. Verify test file added to `src/Makefile.am`
3. Ensure `autogen.sh` and `configure` run after Makefile changes

### Tests Don't Link

1. Check that object files are linked correctly in Makefile.am
2. Verify all dependencies are listed
3. Ensure external libraries (MKL, CUDA) are available

### GPU Tests Always Skip

1. Verify `cisTEM_USE_CUDA` defined during compilation
2. Check CUDA device availability: `nvidia-smi`
3. Ensure `cuda_device_available()` function works

### Tests Are Flaky

1. Check for race conditions in concurrent tests
2. Verify no global state dependencies
3. Ensure deterministic random seeds
4. Check for proper resource cleanup

### Tests Are Slow

1. Profile with `./unit_test_runner -d yes`
2. Reduce test data sizes
3. Mock expensive operations
4. Tag slow tests with `[slow]`

## Related Resources

- `test_organization.md` - File naming and structure conventions
- `build_integration.md` - Detailed Autotools integration
- `tdd_fundamentals.md` - TDD methodology and AAA pattern
- `cuda_testing.md` - Advanced CUDA testing strategies
