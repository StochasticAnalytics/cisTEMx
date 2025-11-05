# Test-Driven Development Fundamentals

Core principles and methodology for effective unit testing and test-driven development.

## What is TDD?

**Test-Driven Development (TDD)** is a software development approach where:
1. You write a failing test BEFORE writing implementation code
2. You write minimal code to make the test pass
3. You refactor both test and code for quality
4. You repeat the cycle

**TDD is not just about testing** - it's a design methodology that leads to cleaner, more maintainable code.

## The TDD Workflow: Red-Green-Refactor

### 1. Red: Write a Failing Test

Write a test for the next small piece of functionality:
```cpp
TEST_CASE("Calculator can add two numbers", "[calculator]") {
    Calculator calc;
    REQUIRE(calc.add(2, 3) == 5);  // This will fail - add() doesn't exist yet
}
```

**Run the test - it should fail.** This confirms:
- The test actually runs
- The test can detect failure
- You're testing something that doesn't exist yet

### 2. Green: Write Minimal Code to Pass

Write just enough code to make the test pass:
```cpp
class Calculator {
public:
    int add(int a, int b) {
        return a + b;  // Simplest implementation
    }
};
```

**Run the test - it should pass.**

Don't worry about perfection yet. The goal is to get to green as quickly as possible.

### 3. Refactor: Improve Code Quality

Now improve the code without changing behavior:
- Remove duplication
- Improve naming
- Simplify logic
- Extract functions

```cpp
// Maybe extract validation
int add(int a, int b) {
    validate_not_overflow(a, b);
    return a + b;
}
```

**Run the test after each change - it should stay green.**

### 4. Repeat

Pick the next small piece of functionality and start again at Red.

## TDD Benefits (Evidence-Based)

Research shows TDD leads to:
- **40-80% fewer bugs in production** (Microsoft Research, 2009)
- **60% better code coverage** on average
- **30% reduction in maintenance costs**
- **More modular architecture** (forced by testability requirements)
- **Better requirement understanding** (forces clarity before coding)

**TDD costs**: 15-35% more development time upfront, but pays off in maintenance.

## The AAA Pattern

**AAA (Arrange-Act-Assert)** is the universal structure for clear, maintainable tests.

### Structure

```
// Arrange - Set up test preconditions
// Act - Execute the behavior being tested
// Assert - Verify the outcome
```

### Example: C++

```cpp
TEST_CASE("Stack pop returns last pushed value", "[stack]") {
    // Arrange
    Stack<int> stack;
    stack.push(5);
    stack.push(10);

    // Act
    int result = stack.pop();

    // Assert
    REQUIRE(result == 10);
    REQUIRE(stack.size() == 1);
}
```

### Example: Python

```python
def test_user_creation_assigns_unique_id():
    # Arrange
    user_service = UserService()

    # Act
    user = user_service.create_user("alice")

    # Assert
    assert user.id is not None
    assert isinstance(user.id, uuid.UUID)
```

### Example: Bash

```bash
@test "script creates output file" {
    # Arrange
    input_file="test_input.txt"
    echo "test data" > "$input_file"

    # Act
    run ./process_data.sh "$input_file"

    # Assert
    [ "$status" -eq 0 ]
    [ -f "output.txt" ]
}
```

### Why AAA Works

**Benefits:**
- Improves readability - structure is immediately clear
- Ensures each test focuses on single behavior
- Makes debugging easier - failures clearly indicate which phase
- Forces good test design - complex arrange/assert suggests design problems

**Red flags:**
- No clear arrange section → Test relies on global state
- Multiple act sections → Test is doing too much
- Enormous arrange section → Code is hard to set up (design smell)

## FIRST Principles

High-quality unit tests are **FIRST**:

### Fast

**Tests should run quickly** - ideally under 200ms each.

Fast tests enable:
- Frequent execution during development
- Quick feedback loops
- Practical CI/CD integration

**How to keep tests fast:**
- Avoid disk I/O when possible
- Mock slow dependencies (databases, networks, external services)
- Use in-memory implementations
- Keep test data small
- Run CPU-only tests by default (tag GPU tests)

```cpp
// Good: Fast test
TEST_CASE("Parser validates input format", "[parser]") {
    REQUIRE_THROWS(parse("invalid"));  // <1ms
}

// Bad: Slow test (should be tagged [slow])
TEST_CASE("Full system integration", "[integration][slow]") {
    Database db;  // Slow
    loadLargeDataset();  // Slow
    processEntireWorkflow();  // Slow
}
```

### Isolated

**Each test runs independently** with no dependencies on other tests or execution order.

**Isolation requirements:**
- No shared mutable state between tests
- Each test creates its own fixtures
- Tests can run in any order
- Tests can run in parallel

```cpp
// Good: Isolated tests
TEST_CASE("Test A", "[core]") {
    int value = 5;  // Local state
    REQUIRE(process(value) == 10);
}

TEST_CASE("Test B", "[core]") {
    int value = 3;  // Independent local state
    REQUIRE(process(value) == 6);
}

// Bad: Shared state
static int counter = 0;  // DON'T DO THIS

TEST_CASE("Test A") {
    counter++;
    REQUIRE(counter == 1);  // Breaks if tests run out of order
}

TEST_CASE("Test B") {
    counter++;
    REQUIRE(counter == 2);  // Depends on Test A running first
}
```

**Isolation strategies:**
- Use test fixtures that set up fresh state
- Use SECTION blocks in Catch2 (each gets fresh TEST_CASE state)
- Create temporary directories for file operations
- Mock external dependencies
- Reset static state in teardown (if absolutely necessary)

### Repeatable

**Tests produce the same result every time**, regardless of:
- Time of day
- Hardware differences
- Test execution order
- Previous test runs

**Threats to repeatability:**
- Actual randomness (use fixed seeds)
- System time dependencies (mock time)
- Network availability (mock or skip)
- Filesystem state (use temp directories, clean up)
- Race conditions (proper synchronization in concurrent tests)

```cpp
// Good: Repeatable
TEST_CASE("Random generator with fixed seed", "[random]") {
    std::mt19937 rng(42);  // Fixed seed
    auto value = generate(rng);
    REQUIRE(value == 123456);  // Always the same
}

// Bad: Non-repeatable
TEST_CASE("Random generator", "[random]") {
    std::random_device rd;
    std::mt19937 rng(rd());  // Different every time
    auto value = generate(rng);
    REQUIRE(value > 0);  // Might pass, might fail
}
```

### Self-Validating

**Tests have boolean output: pass or fail.** No manual verification required.

```cpp
// Good: Self-validating
TEST_CASE("Encryption round-trip", "[crypto]") {
    auto plaintext = "secret message";
    auto encrypted = encrypt(plaintext, key);
    auto decrypted = decrypt(encrypted, key);
    REQUIRE(decrypted == plaintext);  // Clear pass/fail
}

// Bad: Requires manual inspection
TEST_CASE("Print encryption result") {
    auto encrypted = encrypt("secret", key);
    std::cout << "Encrypted: " << encrypted << std::endl;
    // Now what? Developer has to visually check?
}
```

**All assertions should be programmatic:**
- Use `REQUIRE`, `assert`, `[ "$status" -eq 0 ]`
- Don't print output for manual verification
- Don't write results to files for manual checking

### Timely

**Tests are written at the right time** - ideally before or alongside the code.

**Timely in TDD:** Write tests BEFORE code
- Forces clear requirements
- Ensures testable design
- Provides immediate feedback

**Timely in practice:** Write tests AS SOON AS possible
- Right after writing code (if not before)
- Immediately after fixing a bug (regression test)
- Before refactoring (safety net)

**Not timely:** Writing tests months later
- Code may not be testable
- Requirements may be forgotten
- Tests become costly retrofits

## Test Isolation Deep Dive

### What is Test Isolation?

**Test isolation means each test:**
1. Sets up its own preconditions
2. Executes independently
3. Cleans up after itself
4. Does not affect other tests

### Why Isolation Matters

Without isolation:
- Tests become dependent on execution order
- Failures cascade (one failure breaks many tests)
- Debugging is nightmare (which test caused the problem?)
- Parallel execution is impossible
- Refactoring breaks unrelated tests

### Achieving Isolation

#### 1. Avoid Global State

```cpp
// Bad: Global state
int g_counter = 0;

TEST_CASE("Test 1") {
    g_counter = 5;
    REQUIRE(compute(g_counter) == 10);
}

TEST_CASE("Test 2") {
    // Assumes g_counter is 0, but Test 1 changed it!
    REQUIRE(compute(g_counter) == 0);
}

// Good: Local state
TEST_CASE("Test 1") {
    int counter = 5;
    REQUIRE(compute(counter) == 10);
}

TEST_CASE("Test 2") {
    int counter = 0;
    REQUIRE(compute(counter) == 0);
}
```

#### 2. Use Fresh Fixtures

```cpp
// Catch2 SECTION blocks provide automatic isolation
TEST_CASE("Stack operations", "[stack]") {
    Stack<int> stack;  // Created fresh for each SECTION

    SECTION("push adds element") {
        stack.push(5);
        REQUIRE(stack.size() == 1);
        // stack destroyed here
    }

    SECTION("pop removes element") {
        // Fresh stack here, unaffected by previous SECTION
        stack.push(5);
        stack.push(10);
        stack.pop();
        REQUIRE(stack.size() == 1);
    }
}
```

#### 3. Isolate Filesystem Operations

```cpp
#include <filesystem>
#include <cstdlib>

TEST_CASE("File processing", "[io]") {
    // Create unique temporary directory
    auto temp_dir = std::filesystem::temp_directory_path() /
                    ("test_" + std::to_string(std::rand()));
    std::filesystem::create_directory(temp_dir);

    // Test operations in isolated directory
    auto test_file = temp_dir / "test.txt";
    write_file(test_file, "test data");
    REQUIRE(read_file(test_file) == "test data");

    // Clean up
    std::filesystem::remove_all(temp_dir);
}
```

#### 4. Isolate Network Operations

```cpp
TEST_CASE("Socket communication", "[socket]") {
    // Use loopback interface
    // Use ephemeral port (OS assigns)
    Socket server("127.0.0.1", 0);

    if (!server.bind()) {
        SKIP("Cannot bind to loopback");
    }

    // Test with isolated socket
    // No interference with other tests or services
}
```

#### 5. Isolate Time Dependencies

```cpp
// Bad: Depends on actual time
TEST_CASE("Cache expires after timeout") {
    Cache cache;
    cache.set("key", "value", 100ms);
    std::this_thread::sleep_for(150ms);  // Flaky timing
    REQUIRE(!cache.has("key"));
}

// Good: Mock time
TEST_CASE("Cache expires after timeout") {
    MockClock clock;
    Cache cache(clock);
    cache.set("key", "value", 100ms);
    clock.advance(150ms);  // Deterministic
    REQUIRE(!cache.has("key"));
}
```

## Test Doubles

**Test doubles** are objects that stand in for real dependencies during testing.

### Types of Test Doubles

#### 1. Stub

**Returns predetermined values.** Used to provide inputs to the system under test.

```python
# Python example
class StubDatabase:
    def get_user(self, user_id):
        return User(id=user_id, name="Test User")

def test_user_service():
    db = StubDatabase()  # Always returns test user
    service = UserService(db)
    user = service.get_user(123)
    assert user.name == "Test User"
```

#### 2. Mock

**Verifies behavior** - checks that methods were called with expected arguments.

```python
# Python example with mock
from unittest.mock import Mock

def test_email_sent_on_registration():
    email_service = Mock()
    user_service = UserService(email_service)

    user_service.register("alice@example.com")

    # Verify email service was called correctly
    email_service.send.assert_called_once_with(
        to="alice@example.com",
        subject="Welcome"
    )
```

#### 3. Fake

**Simplified working implementation.** More realistic than stub, but simpler than production.

```cpp
// C++ example
class FakeDatabase {
    std::unordered_map<int, User> users_;
public:
    void save(const User& user) {
        users_[user.id] = user;  // In-memory, not real DB
    }

    User get(int id) {
        return users_.at(id);
    }
};

TEST_CASE("User repository", "[repository]") {
    FakeDatabase db;  // No real database needed
    UserRepository repo(db);

    User user{1, "Alice"};
    repo.save(user);
    REQUIRE(repo.get(1).name == "Alice");
}
```

#### 4. Spy

**Records information about calls** for later verification.

```cpp
// C++ example
class SpyLogger {
    std::vector<std::string> messages_;
public:
    void log(const std::string& message) {
        messages_.push_back(message);
    }

    size_t call_count() const { return messages_.size(); }
    const std::string& last_message() const { return messages_.back(); }
};

TEST_CASE("Service logs errors", "[logging]") {
    SpyLogger logger;
    Service service(logger);

    service.process_invalid_input("bad");

    REQUIRE(logger.call_count() == 1);
    REQUIRE(logger.last_message().find("Error") != std::string::npos);
}
```

### When to Use Test Doubles

**Use doubles to isolate external dependencies:**
- Databases
- File systems
- Networks / APIs
- External services
- Time / randomness
- Expensive computations

**Don't overuse doubles:**
- Testing your own simple classes → Use real objects
- Testing integrations → Use real dependencies (integration test, not unit test)
- Over-mocking makes tests brittle and unclear

**Rule of thumb:** Mock dependencies you don't own or control. Use real objects for your own code.

## Common TDD Pitfalls

### 1. Writing Tests After Code

**Problem:** Code may not be testable; tests become retrofits.

**Solution:** Write tests first, or immediately after writing each small piece.

### 2. Testing Implementation, Not Behavior

```cpp
// Bad: Tests implementation details
TEST_CASE("Uses binary search internally") {
    List list{1, 2, 3, 4, 5};
    REQUIRE(list.search_impl_uses_binary_search());  // Overly specific
}

// Good: Tests behavior
TEST_CASE("Find returns correct element") {
    List list{1, 2, 3, 4, 5};
    REQUIRE(list.find(3) == 2);  // Tests what, not how
}
```

### 3. One Giant Test

```cpp
// Bad: Tests everything at once
TEST_CASE("Entire system") {
    // 200 lines testing everything
    // When this fails, where's the problem?
}

// Good: Focused tests
TEST_CASE("Login validation") { /* ... */ }
TEST_CASE("Session creation") { /* ... */ }
TEST_CASE("Authorization check") { /* ... */ }
```

### 4. Ignoring Failing Tests

**Never commit failing tests.** Fix them immediately or remove them.

Failing tests that stay failing train developers to ignore test failures.

### 5. Flaky Tests

**Tests that sometimes pass, sometimes fail** are worse than no tests.

**Common causes:**
- Race conditions
- Time dependencies
- Randomness without fixed seeds
- Shared global state
- Network availability

**Fix them immediately** or remove them. Never tolerate flaky tests.

### 6. Too Much Mocking

**Over-mocking makes tests:**
- Brittle (break when implementation changes)
- Unclear (what's actually being tested?)
- Expensive to maintain

**Balance:** Mock external dependencies, use real objects for your own code.

## TDD for Research Code

### "But Research Code is Different!"

Common objections:
- "Requirements change constantly" → Tests help with refactoring
- "Code is exploratory" → Tests document what works
- "Prototype code" → Tests separate working from broken prototypes

**TDD for research:**
- Verify algorithms produce expected results
- Document numerical properties
- Catch regressions during iteration
- Validate against reference implementations
- Test edge cases in data

### Example: Testing Scientific Code

```cpp
TEST_CASE("FFT round trip preserves data", "[fft][property]") {
    std::vector<float> input = generate_test_signal();

    auto freq_domain = fft_forward(input);
    auto reconstructed = fft_inverse(freq_domain);

    // Property: Forward then inverse should give original
    for (size_t i = 0; i < input.size(); ++i) {
        REQUIRE(reconstructed[i] == Approx(input[i]).epsilon(1e-6));
    }
}

TEST_CASE("FFT matches reference implementation", "[fft][validation]") {
    std::vector<float> input = generate_test_signal();

    auto our_result = our_fft(input);
    auto reference_result = fftw_fft(input);  // FFTW as reference

    for (size_t i = 0; i < input.size(); ++i) {
        REQUIRE(our_result[i] == Approx(reference_result[i]));
    }
}
```

## Summary

**TDD Cycle:** Red → Green → Refactor

**AAA Pattern:** Arrange → Act → Assert

**FIRST Principles:**
- Fast: <200ms per test
- Isolated: No dependencies between tests
- Repeatable: Same result every time
- Self-Validating: Pass/fail, no manual checks
- Timely: Written before or with code

**Test Isolation:** Each test runs independently with no shared state

**Test Doubles:** Stub, Mock, Fake, Spy - use to isolate external dependencies

**Key Insight:** TDD is a design methodology that produces better code through testability requirements, not just a testing technique.
