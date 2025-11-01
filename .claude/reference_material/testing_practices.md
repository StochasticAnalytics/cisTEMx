# Testing Practices Reference
## Multi-Tiered Testing Approach for cisTEMx

### Testing Philosophy
cisTEMx employs a multi-tiered testing strategy to ensure correctness at different levels of complexity.

### Test Tiers

#### 1. Unit Tests
**Purpose**: Test individual methods and functions in isolation
**Runner**: `./unit_test_runner`
**Framework**: Catch2 v3
**Location**: `src/test/`

```bash
# Run all unit tests
./unit_test_runner

# Run specific test suite
./unit_test_runner "[socket]"
```

#### 2. Console Tests
**Purpose**: Mid-complexity tests of single methods with real data
**Runner**: `./console_test`
**Focus**: Core algorithms and data structures
**Location**: `src/programs/console_test/`

```bash
# Run console tests
./console_test
```

#### 3. Functional Tests
**Purpose**: Test complete workflows and image processing tasks
**Runner**: `./samples_functional_testing`
**Focus**: End-to-end validation of scientific computations
**Location**: `src/programs/samples_functional_testing/`

```bash
# Run functional tests
./samples_functional_testing
```

### Test Development Guidelines

#### When to Write Tests
- **Immediately after implementing new functionality**
- **When fixing bugs** (regression tests)
- **Before refactoring** (ensure behavior preservation)
- **For boundary conditions and edge cases**

#### Test Coverage Priorities
1. Public API methods
2. Complex algorithms
3. Error handling paths
4. GPU/CPU parity
5. Thread safety scenarios

### GPU Testing Considerations

#### GPU-Gated Tests
Tests requiring GPU should:
- Check for GPU availability
- Provide CPU fallback tests
- Mark clearly as GPU-required

```cpp
if (runtime.has_gpu) {
    // GPU-specific tests
} else {
    // CPU fallback or skip
}
```

### CI Testing Configuration

Tests run automatically on:
- Pull requests
- Commits to main branch
- Nightly builds

See `.github/workflows/` for CI test configurations.

### Debugging Test Failures

#### Console Test Failures
1. Run with verbose output
2. Check for memory issues with valgrind
3. Verify test data integrity

#### Functional Test Failures
1. Compare against reference outputs
2. Check numerical tolerances
3. Verify GPU/CPU consistency

### Test Data Management

- Small test data: Embedded in test files
- Large test data: Stored in `test_data/`
- Reference outputs: Version controlled for regression detection

### Performance Testing

While not part of regular CI:
- Benchmark critical algorithms
- Profile memory usage
- Monitor GPU utilization
- Track performance regressions

### Related Documentation
- See unit test architect agent for test creation
- Check gpu-test-debugger agent for debugging failures
- Review CI workflows in `.github/workflows/`