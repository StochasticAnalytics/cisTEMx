# Bash Unit Testing with bats

Comprehensive guide for writing Bash script tests using the Bash Automated Testing System (bats).

## Framework Overview

**bats (Bash Automated Testing System)** is a TAP-compliant testing framework for Bash 3.2+:
- Simple syntax matching Bash idioms
- TAP (Test Anything Protocol) output
- Setup/teardown support
- Test isolation
- Easy CI/CD integration

**Why test Bash scripts?**
- Prevent regressions from old bugs
- Ensure reliability of build/deployment scripts
- Improve code quality and maintainability
- Enable confident refactoring
- Essential for security-critical scripts

## Installation

### From GitHub

```bash
git clone https://github.com/bats-core/bats-core.git
cd bats-core
./install.sh /usr/local
```

### Verify installation

```bash
bats --version
```

### In cisTEMx

Check if available:
```bash
which bats
```

If not installed, tests will need to skip or install locally.

## Test Discovery

bats expects test files with `.bats` extension:

```
scripts/
├── build_helper.sh
└── tests/
    └── test_build_helper.bats
```

## Basic Test Structure

### Simple test

```bash
#!/usr/bin/env bats

@test "addition works" {
    result=$((2 + 2))
    [ "$result" -eq 4 ]
}
```

### Using the `run` helper

**The `run` helper** captures command output and exit status:

```bash
@test "script succeeds with valid input" {
    run ./my_script.sh --input test.txt

    [ "$status" -eq 0 ]
    [ "$output" = "Success" ]
}
```

**Variables after `run`:**
- `$status` - Exit code of the command
- `$output` - Combined stdout and stderr
- `${lines[@]}` - Array of output lines

### Test with AAA pattern

```bash
@test "script processes file correctly" {
    # Arrange
    input_file="test_input.txt"
    echo "test data" > "$input_file"

    # Act
    run ./process_file.sh "$input_file"

    # Assert
    [ "$status" -eq 0 ]
    [ -f "output.txt" ]
    [ "$(cat output.txt)" = "processed: test data" ]

    # Cleanup
    rm -f "$input_file" "output.txt"
}
```

## Assertions and Conditionals

### Exit status checks

```bash
@test "script exits successfully" {
    run ./script.sh
    [ "$status" -eq 0 ]
}

@test "script fails on invalid input" {
    run ./script.sh --invalid
    [ "$status" -eq 1 ]
}

@test "script fails with specific code" {
    run ./script.sh --missing-arg
    [ "$status" -eq 2 ]  # Specific error code
}
```

### Output pattern matching

```bash
@test "script outputs expected message" {
    run ./script.sh

    # Exact match
    [ "$output" = "Expected output" ]

    # Pattern matching (regex)
    [[ "$output" =~ ^Success ]]
    [[ "$output" =~ "Error: File not found" ]]

    # Substring check
    [[ "$output" == *"substring"* ]]
}
```

### Line-by-line checking

```bash
@test "script outputs multiple lines" {
    run ./script.sh

    [ "${lines[0]}" = "First line" ]
    [ "${lines[1]}" = "Second line" ]
    [ "${#lines[@]}" -eq 2 ]  # Exactly 2 lines
}
```

### File existence and properties

```bash
@test "script creates output file" {
    run ./script.sh

    [ -f "output.txt" ]      # File exists
    [ -d "output_dir" ]      # Directory exists
    [ -x "executable.sh" ]   # File is executable
    [ -s "output.txt" ]      # File is non-empty
}
```

### String comparisons

```bash
@test "string operations" {
    run ./script.sh

    [ "$output" = "exact match" ]      # Equality
    [ "$output" != "different" ]       # Inequality
    [ -z "$output" ]                   # Empty string
    [ -n "$output" ]                   # Non-empty string
}
```

### Numeric comparisons

```bash
@test "numeric checks" {
    count=5

    [ "$count" -eq 5 ]    # Equal
    [ "$count" -ne 0 ]    # Not equal
    [ "$count" -gt 3 ]    # Greater than
    [ "$count" -lt 10 ]   # Less than
    [ "$count" -ge 5 ]    # Greater than or equal
    [ "$count" -le 5 ]    # Less than or equal
}
```

## Setup and Teardown

### setup() - Before each test

Runs before every @test:

```bash
setup() {
    # Create temporary directory
    TEST_TEMP_DIR="$(mktemp -d)"
    export TEST_TEMP_DIR

    # Set up test files
    echo "test data" > "$TEST_TEMP_DIR/input.txt"
}

@test "test with setup" {
    # TEST_TEMP_DIR is available
    [ -d "$TEST_TEMP_DIR" ]
    [ -f "$TEST_TEMP_DIR/input.txt" ]
}
```

### teardown() - After each test

Runs after every @test:

```bash
teardown() {
    # Clean up temporary directory
    if [ -n "$TEST_TEMP_DIR" ] && [ -d "$TEST_TEMP_DIR" ]; then
        rm -rf "$TEST_TEMP_DIR"
    fi
}

@test "test with cleanup" {
    # Create files in TEST_TEMP_DIR
    touch "$TEST_TEMP_DIR/temp.txt"

    # teardown() will clean up after test
}
```

### setup_file() and teardown_file()

Run once per file (bats >= 1.5.0):

```bash
setup_file() {
    # Expensive setup once per file
    export SHARED_RESOURCE="expensive_computation_result"
}

teardown_file() {
    # Cleanup once per file
    unset SHARED_RESOURCE
}

@test "test 1" {
    [ -n "$SHARED_RESOURCE" ]
}

@test "test 2" {
    [ -n "$SHARED_RESOURCE" ]
}
```

## Test Isolation

### Create unique temporary directories

```bash
setup() {
    # Create unique temp dir per test
    TEST_TEMP_DIR="$(mktemp -d -t bats-test-XXXXXX)"
    cd "$TEST_TEMP_DIR"
}

teardown() {
    # Clean up
    if [ -n "$TEST_TEMP_DIR" ]; then
        rm -rf "$TEST_TEMP_DIR"
    fi
}

@test "isolated test 1" {
    # Each test gets fresh temp directory
    touch file1.txt
    [ -f file1.txt ]
}

@test "isolated test 2" {
    # file1.txt from test 1 doesn't exist here
    [ ! -f file1.txt ]
}
```

### Avoid global state

```bash
# Bad: Global variable shared between tests
COUNTER=0

@test "test 1" {
    COUNTER=$((COUNTER + 1))
    [ "$COUNTER" -eq 1 ]  # Breaks if tests run out of order
}

# Good: Local state
@test "test 1" {
    local counter=0
    counter=$((counter + 1))
    [ "$counter" -eq 1 ]
}
```

### Isolate environment variables

```bash
setup() {
    # Save original environment
    _OLD_PATH="$PATH"

    # Modify for test
    export PATH="/test/bin:$PATH"
}

teardown() {
    # Restore original environment
    export PATH="$_OLD_PATH"
}
```

## Common Patterns

### Testing script argument validation

```bash
@test "script requires input argument" {
    run ./script.sh

    [ "$status" -eq 1 ]
    [[ "$output" =~ "Error: Missing required argument" ]]
}

@test "script accepts valid argument" {
    run ./script.sh --input test.txt

    [ "$status" -eq 0 ]
}

@test "script rejects invalid flag" {
    run ./script.sh --invalid-flag

    [ "$status" -eq 1 ]
    [[ "$output" =~ "Unknown option" ]]
}
```

### Testing file processing

```bash
@test "script processes input file" {
    # Arrange
    input_file="$TEST_TEMP_DIR/input.txt"
    output_file="$TEST_TEMP_DIR/output.txt"
    echo "line 1" > "$input_file"
    echo "line 2" >> "$input_file"

    # Act
    run ./process.sh "$input_file" "$output_file"

    # Assert
    [ "$status" -eq 0 ]
    [ -f "$output_file" ]
    [ "$(wc -l < "$output_file")" -eq 2 ]
}
```

### Testing error conditions

```bash
@test "script handles missing file" {
    run ./script.sh nonexistent.txt

    [ "$status" -ne 0 ]
    [[ "$output" =~ "File not found" ]]
}

@test "script handles permission denied" {
    # Create read-only file
    touch "$TEST_TEMP_DIR/readonly.txt"
    chmod 000 "$TEST_TEMP_DIR/readonly.txt"

    run ./script.sh "$TEST_TEMP_DIR/readonly.txt"

    [ "$status" -ne 0 ]
    [[ "$output" =~ "Permission denied" ]]
}
```

### Testing environment variables

```bash
@test "script uses default config path" {
    run ./script.sh

    [[ "$output" =~ "/etc/default/config" ]]
}

@test "script respects CONFIG_PATH override" {
    export CONFIG_PATH="/custom/config"

    run ./script.sh

    [[ "$output" =~ "/custom/config" ]]
}
```

### Testing command substitution

```bash
@test "script captures command output" {
    # Script uses $(some_command) internally
    run ./script.sh

    [ "$status" -eq 0 ]
    [[ "$output" =~ "Captured output" ]]
}
```

### Skip tests conditionally

```bash
@test "GPU-dependent test" {
    if ! command -v nvidia-smi &> /dev/null; then
        skip "CUDA not available"
    fi

    run ./gpu_script.sh

    [ "$status" -eq 0 ]
}

@test "Linux-only test" {
    if [[ "$(uname)" != "Linux" ]]; then
        skip "Linux required"
    fi

    run ./linux_script.sh

    [ "$status" -eq 0 ]
}
```

## Testing Helper Functions

For complex scripts with helper functions:

### Load functions for testing

```bash
#!/usr/bin/env bats

# Load the script to test its functions
load ../build_helper.sh

@test "validate_input returns true for valid input" {
    run validate_input "valid_string"
    [ "$status" -eq 0 ]
}

@test "validate_input returns false for empty input" {
    run validate_input ""
    [ "$status" -eq 1 ]
}
```

### Test individual functions

```bash
# build_helper.sh
compute_checksum() {
    local file="$1"
    sha256sum "$file" | awk '{print $1}'
}

# test_build_helper.bats
@test "compute_checksum calculates SHA256" {
    echo "test data" > "$TEST_TEMP_DIR/test.txt"

    run compute_checksum "$TEST_TEMP_DIR/test.txt"

    [ "$status" -eq 0 ]
    [ ${#output} -eq 64 ]  # SHA256 is 64 hex chars
}
```

## Running Tests

### Run all tests in file

```bash
bats test/test_script.bats
```

### Run all tests in directory

```bash
bats test/
```

### Verbose output

```bash
bats -t test/test_script.bats  # Show timing
bats -p test/                  # Run tests in parallel
```

### TAP output

```bash
bats test/test_script.bats
# Output:
# 1..3
# ok 1 script succeeds with valid input
# ok 2 script fails on invalid input
# ok 3 script creates output file
```

### Count tests

```bash
bats -c test/test_script.bats
```

## CI/CD Integration

### GitHub Actions

```yaml
name: Test Bash Scripts

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
        run: bats scripts/tests/
```

### GitLab CI

```yaml
test:
  image: ubuntu:latest
  before_script:
    - apt-get update && apt-get install -y git
    - git clone https://github.com/bats-core/bats-core.git
    - cd bats-core && ./install.sh /usr/local && cd ..
  script:
    - bats scripts/tests/
```

## Best Practices

### 1. Clear Test Names

```bash
# Good: Descriptive test names
@test "script validates required --input argument"
@test "script creates output directory if missing"
@test "script handles permission denied gracefully"

# Bad: Vague test names
@test "test 1"
@test "check script"
```

### 2. Test One Thing Per Test

```bash
# Good: Focused test
@test "script exits with code 1 on missing file" {
    run ./script.sh nonexistent.txt
    [ "$status" -eq 1 ]
}

@test "script outputs error message on missing file" {
    run ./script.sh nonexistent.txt
    [[ "$output" =~ "File not found" ]]
}

# Bad: Tests too much
@test "script error handling" {
    run ./script.sh nonexistent.txt
    [ "$status" -eq 1 ]
    [[ "$output" =~ "File not found" ]]
    run ./script.sh --invalid-flag
    [ "$status" -eq 1 ]
    [[ "$output" =~ "Unknown option" ]]
    # What if one assertion fails? Which one?
}
```

### 3. Always Clean Up

```bash
teardown() {
    # Always clean up temp files/directories
    [ -n "$TEST_TEMP_DIR" ] && rm -rf "$TEST_TEMP_DIR"

    # Kill background processes
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null

    # Restore environment
    [ -n "$_OLD_PATH" ] && export PATH="$_OLD_PATH"
}
```

### 4. Use setup() for Common Arrange

```bash
setup() {
    # Common setup for all tests
    TEST_INPUT="$TEST_TEMP_DIR/input.txt"
    echo "test data" > "$TEST_INPUT"
}

@test "test 1" {
    # TEST_INPUT already set up
    run ./script.sh "$TEST_INPUT"
    [ "$status" -eq 0 ]
}

@test "test 2" {
    # TEST_INPUT available here too
    [ -f "$TEST_INPUT" ]
}
```

### 5. Deterministic Tests

```bash
# Good: Deterministic
@test "script output format" {
    echo "input" > input.txt

    run ./script.sh input.txt

    [ "$output" = "Processed: input" ]
}

# Bad: Non-deterministic
@test "script timestamp" {
    run ./script.sh

    # Output includes timestamp - different every run
    [[ "$output" =~ "2024-" ]]  # Brittle, will break
}
```

### 6. Test Error Paths

```bash
@test "script handles missing file" {
    run ./script.sh nonexistent.txt
    [ "$status" -ne 0 ]
}

@test "script handles empty input" {
    touch empty.txt
    run ./script.sh empty.txt
    [ "$status" -ne 0 ]
}

@test "script handles malformed input" {
    echo "invalid format" > bad.txt
    run ./script.sh bad.txt
    [ "$status" -ne 0 ]
}
```

## Example: Complete Test File

```bash
#!/usr/bin/env bats

# Test suite for build_helper.sh

setup() {
    # Create temporary test environment
    TEST_TEMP_DIR="$(mktemp -d -t bats-test-XXXXXX)"
    TEST_INPUT="$TEST_TEMP_DIR/input.txt"
    TEST_OUTPUT="$TEST_TEMP_DIR/output.txt"

    # Create test input
    echo "test data" > "$TEST_INPUT"
}

teardown() {
    # Clean up
    [ -n "$TEST_TEMP_DIR" ] && rm -rf "$TEST_TEMP_DIR"
}

@test "build_helper.sh exists and is executable" {
    [ -f "./build_helper.sh" ]
    [ -x "./build_helper.sh" ]
}

@test "script requires input argument" {
    run ./build_helper.sh

    [ "$status" -eq 1 ]
    [[ "$output" =~ "Error: Missing required argument" ]]
}

@test "script processes valid input file" {
    run ./build_helper.sh --input "$TEST_INPUT" --output "$TEST_OUTPUT"

    [ "$status" -eq 0 ]
    [ -f "$TEST_OUTPUT" ]
    [[ "$(cat "$TEST_OUTPUT")" == *"test data"* ]]
}

@test "script handles missing input file" {
    run ./build_helper.sh --input nonexistent.txt --output "$TEST_OUTPUT"

    [ "$status" -eq 1 ]
    [[ "$output" =~ "File not found" ]]
}

@test "script creates output directory if needed" {
    output_path="$TEST_TEMP_DIR/subdir/output.txt"

    run ./build_helper.sh --input "$TEST_INPUT" --output "$output_path"

    [ "$status" -eq 0 ]
    [ -f "$output_path" ]
}

@test "script respects VERBOSE environment variable" {
    export VERBOSE=1

    run ./build_helper.sh --input "$TEST_INPUT" --output "$TEST_OUTPUT"

    [ "$status" -eq 0 ]
    [[ "$output" =~ "Processing:" ]]  # Verbose output
}
```

## Troubleshooting

### Tests fail with "command not found"

**Problem:** Script under test not in PATH.

**Solution:** Use relative or absolute paths:
```bash
run ./script.sh  # Relative
run /full/path/to/script.sh  # Absolute
```

### Tests pass locally but fail in CI

**Problem:** Environment differences.

**Solution:**
- Check PATH and environment variables
- Ensure dependencies are installed
- Use absolute paths
- Clean up properly between tests

### Output matching fails unexpectedly

**Problem:** Unexpected whitespace or formatting.

**Solution:** Use pattern matching instead of exact match:
```bash
# Instead of:
[ "$output" = "Expected output" ]

# Use:
[[ "$output" =~ "Expected output" ]]
```

### Cleanup not happening

**Problem:** Test exits before teardown().

**Solution:** Ensure no early exits bypass teardown:
```bash
@test "test with potential early exit" {
    run ./script.sh || true  # Don't exit on failure
    [ "$status" -ne 0 ]
    # teardown() will still run
}
```

## Summary

**bats strengths:**
- Simple syntax matching Bash idioms
- Easy test isolation with setup/teardown
- TAP-compliant output for CI integration
- Minimal learning curve for Bash developers

**Key patterns:**
- Use `run` to capture command output and status
- Use setup/teardown for isolation
- Test exit codes and output patterns
- Clean up temporary resources
- Skip tests when prerequisites missing

**Best practices:**
- One behavior per test
- Clear, descriptive test names
- Always clean up resources
- Test both success and error paths
- Make tests deterministic and repeatable
