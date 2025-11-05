# Python Unit Testing with pytest

Comprehensive guide for writing Python unit tests using the pytest framework.

## Framework Overview

**pytest** is the de facto standard for Python testing in 2024-2025:
- Simple, Pythonic syntax
- Powerful fixture system for dependency injection
- Automatic test discovery
- Rich plugin ecosystem
- Detailed failure reports
- Native parametrization support

**Why pytest over unittest?**
- Less boilerplate (no classes required)
- Better assertion introspection
- More powerful fixtures
- Easier parametrization
- Active development and community

## Installation and Setup

### Install pytest

```bash
pip install pytest
# or
pip install pytest pytest-cov  # With coverage plugin
```

### Verify installation

```bash
pytest --version
```

## Test Discovery

pytest automatically discovers tests following these conventions:

### File naming

- `test_*.py` - Files starting with `test_`
- `*_test.py` - Files ending with `_test.py`

### Function/class naming

- `test_*()` - Functions starting with `test_`
- `Test*` - Classes starting with `Test` (no `__init__` method)

### Directory structure

```
project/
├── src/
│   └── mymodule.py
├── tests/
│   ├── conftest.py        # Shared fixtures
│   ├── test_mymodule.py   # Tests for mymodule
│   └── test_utils.py
└── .claude/skills/
    └── skill-name/
        ├── scripts/
        │   └── process.py
        └── tests/
            └── test_process.py  # Tests for skill scripts
```

## Basic Test Structure

### Simple test

```python
def test_addition():
    result = 2 + 2
    assert result == 4
```

### Test with AAA pattern

```python
def test_user_creation():
    # Arrange
    user_service = UserService()
    username = "alice"

    # Act
    user = user_service.create_user(username)

    # Assert
    assert user.username == "alice"
    assert user.id is not None
```

### Multiple assertions

```python
def test_string_operations():
    text = "Hello World"

    assert text.startswith("Hello")
    assert text.endswith("World")
    assert len(text) == 11
    assert "World" in text
```

## Assertion Introspection

pytest provides detailed failure messages automatically:

```python
def test_dict_equality():
    expected = {"a": 1, "b": 2}
    actual = {"a": 1, "b": 3}
    assert actual == expected
    # Failure shows:
    #   AssertionError: assert {'a': 1, 'b': 3} == {'a': 1, 'b': 2}
    #   Differing items:
    #   {'b': 3} != {'b': 2}
```

### Advanced assertions

```python
import pytest

# Exception testing
def test_division_by_zero():
    with pytest.raises(ZeroDivisionError):
        result = 1 / 0

# Exception with message matching
def test_invalid_input():
    with pytest.raises(ValueError, match=r"invalid.*format"):
        parse_input("bad format")

# Approximate comparison (floating point)
def test_float_calculation():
    result = 0.1 + 0.2
    assert result == pytest.approx(0.3)
    assert result == pytest.approx(0.3, rel=1e-9)  # Relative tolerance

# Collection membership
def test_list_contents():
    items = [1, 2, 3, 4, 5]
    assert 3 in items
    assert 10 not in items
```

## Fixtures: The Heart of pytest

**Fixtures** provide reusable setup/teardown logic and dependency injection.

### Basic fixture

```python
import pytest

@pytest.fixture
def sample_data():
    """Provide test data"""
    return [1, 2, 3, 4, 5]

def test_sum(sample_data):
    assert sum(sample_data) == 15

def test_max(sample_data):
    assert max(sample_data) == 5
```

### Fixture with setup and teardown

```python
import tempfile
import shutil
from pathlib import Path

@pytest.fixture
def temp_directory():
    """Create temporary directory for testing"""
    # Setup
    temp_dir = Path(tempfile.mkdtemp())

    # Provide to test
    yield temp_dir

    # Teardown
    shutil.rmtree(temp_dir)

def test_file_creation(temp_directory):
    test_file = temp_directory / "test.txt"
    test_file.write_text("test data")

    assert test_file.exists()
    assert test_file.read_text() == "test data"
    # temp_directory automatically cleaned up
```

### Fixture scopes

Control how often fixtures are created:

```python
@pytest.fixture(scope="function")  # Default: once per test function
def function_fixture():
    return create_resource()

@pytest.fixture(scope="class")  # Once per test class
def class_fixture():
    return create_resource()

@pytest.fixture(scope="module")  # Once per test module
def module_fixture():
    return expensive_resource()

@pytest.fixture(scope="session")  # Once per test session
def session_fixture():
    return very_expensive_resource()
```

**Scope usage:**
- `function` - Default, always safe, isolated
- `class` - Shared across methods in test class
- `module` - Shared across all tests in file (use for expensive setup)
- `session` - Shared across entire test run (databases, external services)

**Warning:** Higher scopes break isolation. Use carefully.

### Parameterized fixtures

```python
@pytest.fixture(params=[1, 2, 3])
def test_value(request):
    return request.param

def test_with_multiple_values(test_value):
    # This test runs 3 times, once for each param
    assert test_value > 0
```

### Autouse fixtures

Fixtures that run automatically without being requested:

```python
@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global state before each test"""
    global_registry.clear()
    yield
    # Cleanup after test

def test_something():
    # reset_global_state automatically runs
    assert len(global_registry) == 0
```

## Fixture Organization

### conftest.py

**`conftest.py`** files define fixtures available to all tests in the directory and subdirectories.

```python
# tests/conftest.py
import pytest

@pytest.fixture
def database_connection():
    """Shared database fixture for all tests"""
    conn = Database.connect("test.db")
    yield conn
    conn.close()
```

**Fixture discovery order:**
1. Test file itself
2. conftest.py in same directory
3. conftest.py in parent directories (up to project root)

### Fixture directory structure

```python
tests/
├── conftest.py              # Shared fixtures for all tests
├── fixtures/                # Organized fixture modules
│   ├── database.py          # Database-related fixtures
│   └── mock_api.py          # API mock fixtures
├── unit/
│   ├── conftest.py          # Unit test specific fixtures
│   └── test_models.py
└── integration/
    ├── conftest.py          # Integration test specific fixtures
    └── test_workflows.py
```

### Using fixture modules

```python
# tests/fixtures/database.py
import pytest

@pytest.fixture(scope="module")
def db_connection():
    return create_database_connection()

# tests/conftest.py
pytest_plugins = ["tests.fixtures.database"]  # Load fixture module

# tests/test_something.py
def test_query(db_connection):
    # db_connection fixture available from fixtures/database.py
    result = db_connection.query("SELECT 1")
    assert result == 1
```

## Fixture Best Practices

### 1. Narrow Scope When Possible

```python
# Good: Function scope (isolated)
@pytest.fixture
def user():
    return User("test@example.com")

# Use wider scope only for expensive operations
@pytest.fixture(scope="module")
def ml_model():
    return load_large_ml_model()  # Expensive, load once
```

### 2. Explicit Dependencies

```python
# Good: Clear dependencies
@pytest.fixture
def authenticated_user(user, auth_service):
    return auth_service.authenticate(user)

# Test clearly shows what it needs
def test_access_control(authenticated_user, protected_resource):
    assert protected_resource.is_accessible_by(authenticated_user)
```

### 3. Minimal Fixtures

```python
# Bad: Kitchen sink fixture
@pytest.fixture
def everything():
    return {
        'db': Database(),
        'cache': Cache(),
        'api': API(),
        # ... 20 more things
    }

# Good: Focused fixtures
@pytest.fixture
def database():
    return Database()

@pytest.fixture
def cache():
    return Cache()

def test_caching(cache):  # Only request what you need
    cache.set("key", "value")
    assert cache.get("key") == "value"
```

### 4. Fixture Factories

For fixtures that need customization:

```python
@pytest.fixture
def make_user():
    """Factory fixture for creating users with custom attributes"""
    def _make_user(username="default", email=None):
        return User(username=username, email=email or f"{username}@example.com")
    return _make_user

def test_user_creation(make_user):
    alice = make_user("alice")
    bob = make_user("bob", "bob@custom.com")

    assert alice.username == "alice"
    assert bob.email == "bob@custom.com"
```

## Parametrization

### Parametrize test functions

```python
import pytest

@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
    (10, 20),
])
def test_doubling(input, expected):
    assert double(input) == expected
```

### Multiple parameters

```python
@pytest.mark.parametrize("base,exponent,expected", [
    (2, 2, 4),
    (2, 3, 8),
    (3, 2, 9),
    (5, 0, 1),
])
def test_power(base, exponent, expected):
    assert base ** exponent == expected
```

### Parametrize with IDs

```python
@pytest.mark.parametrize("input,expected", [
    pytest.param(0, 0, id="zero"),
    pytest.param(1, 1, id="one"),
    pytest.param(-1, 1, id="negative"),
], ids=str)
def test_absolute(input, expected):
    assert abs(input) == expected
    # Test IDs appear in output: test_absolute[zero], test_absolute[one], etc.
```

### Parametrize multiple arguments

```python
@pytest.mark.parametrize("x", [1, 2])
@pytest.mark.parametrize("y", [3, 4])
def test_combinations(x, y):
    # Runs 4 times: (1,3), (1,4), (2,3), (2,4)
    assert x + y > 0
```

### Parametrize fixtures

```python
@pytest.fixture(params=[
    {"name": "alice", "age": 30},
    {"name": "bob", "age": 25},
])
def user_data(request):
    return request.param

def test_user(user_data):
    # Runs twice, once for each user
    user = User(**user_data)
    assert user.name in ["alice", "bob"]
```

## Mocking with unittest.mock

pytest works seamlessly with Python's built-in `unittest.mock`.

### Basic mocking

```python
from unittest.mock import Mock

def test_email_service():
    # Create mock object
    email_client = Mock()

    # Use mock in test
    service = EmailService(email_client)
    service.send_email("user@example.com", "Hello")

    # Verify mock was called correctly
    email_client.send.assert_called_once_with(
        to="user@example.com",
        subject="Hello"
    )
```

### Mock return values

```python
from unittest.mock import Mock

def test_api_client():
    api = Mock()
    api.get_user.return_value = {"id": 1, "name": "Alice"}

    service = UserService(api)
    user = service.get_user(1)

    assert user["name"] == "Alice"
    api.get_user.assert_called_once_with(1)
```

### Mock side effects

```python
from unittest.mock import Mock

def test_retry_logic():
    api = Mock()
    # First two calls fail, third succeeds
    api.request.side_effect = [
        ConnectionError(),
        ConnectionError(),
        {"status": "success"}
    ]

    service = ServiceWithRetry(api)
    result = service.fetch_data()

    assert result == {"status": "success"}
    assert api.request.call_count == 3
```

### Patching

```python
from unittest.mock import patch

# Patch a function
@patch('mymodule.expensive_function')
def test_with_mock(mock_function):
    mock_function.return_value = 42

    result = use_expensive_function()

    assert result == 42
    mock_function.assert_called_once()

# Patch as context manager
def test_with_context():
    with patch('mymodule.external_api') as mock_api:
        mock_api.get.return_value = {"data": "test"}

        result = fetch_from_api()

        assert result == {"data": "test"}
```

### Patch object attributes

```python
from unittest.mock import patch

def test_configuration():
    with patch('mymodule.config.API_KEY', 'test-key'):
        service = APIService()
        assert service.api_key == 'test-key'
```

## Mocking Best Practices

### 1. Group mocks in fixtures

```python
@pytest.fixture
def mock_dependencies():
    """Provide mocked dependencies for service"""
    return {
        'database': Mock(spec=Database),
        'cache': Mock(spec=Cache),
        'logger': Mock(spec=Logger)
    }

def test_service(mock_dependencies):
    service = Service(**mock_dependencies)
    # Test with mocks
```

### 2. Use spec for safety

```python
# Good: spec ensures mock has same interface
mock_db = Mock(spec=Database)
mock_db.query("SELECT 1")  # OK
mock_db.typo_method()  # AttributeError - method doesn't exist

# Bad: no spec, anything goes
mock_db = Mock()
mock_db.typo_method()  # Silently succeeds - bug not caught
```

### 3. Verify only critical interactions

```python
# Bad: Over-specification
def test_process():
    logger = Mock()
    processor = Processor(logger)

    processor.process([1, 2, 3])

    logger.debug.assert_called()  # Brittle
    logger.info.assert_called()   # Implementation detail
    # Test breaks if we change logging

# Good: Verify critical behavior only
def test_process_records_result():
    storage = Mock()
    processor = Processor(storage)

    processor.process([1, 2, 3])

    storage.save.assert_called_once()  # This matters
    # Don't care about internal logging
```

### 4. Mimic real behavior

```python
# Good: Mock behaves like real object
@pytest.fixture
def mock_database():
    db = Mock(spec=Database)
    db.query.return_value = [{"id": 1, "name": "Test"}]  # Realistic return
    return db

# Bad: Mock returns nonsense
mock_db = Mock()
mock_db.query.return_value = 42  # Real DB returns list of dicts
```

## Test Organization Patterns

### AAA Pattern in pytest

```python
def test_user_registration():
    # Arrange - Set up test data and dependencies
    user_service = UserService()
    username = "alice"
    email = "alice@example.com"

    # Act - Execute the behavior being tested
    user = user_service.register(username, email)

    # Assert - Verify expectations
    assert user.username == "alice"
    assert user.email == "alice@example.com"
    assert user.is_active is False  # New users start inactive
```

### Test classes for grouping

```python
class TestUserRegistration:
    """Group related user registration tests"""

    def test_valid_registration(self):
        service = UserService()
        user = service.register("alice", "alice@example.com")
        assert user.username == "alice"

    def test_duplicate_username_rejected(self):
        service = UserService()
        service.register("alice", "alice@example.com")

        with pytest.raises(DuplicateUserError):
            service.register("alice", "different@example.com")

    def test_invalid_email_rejected(self):
        service = UserService()

        with pytest.raises(ValidationError):
            service.register("bob", "not-an-email")
```

### Shared setup with class fixtures

```python
class TestDatabase:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Run before each test method"""
        self.db = Database(":memory:")
        self.db.create_tables()
        yield
        self.db.close()

    def test_insert(self):
        self.db.insert("users", {"name": "Alice"})
        result = self.db.query("SELECT * FROM users")
        assert len(result) == 1

    def test_update(self):
        self.db.insert("users", {"name": "Alice"})
        self.db.update("users", {"name": "Bob"}, where={"name": "Alice"})
        result = self.db.query("SELECT name FROM users")
        assert result[0]["name"] == "Bob"
```

## Running Tests

### Run all tests

```bash
pytest
```

### Run specific file

```bash
pytest tests/test_mymodule.py
```

### Run specific test

```bash
pytest tests/test_mymodule.py::test_function_name
pytest tests/test_mymodule.py::TestClass::test_method
```

### Run by pattern

```bash
pytest -k "test_user"  # Run tests matching pattern
pytest -k "not slow"   # Exclude tests matching pattern
```

### Run with markers

```python
# Mark tests
@pytest.mark.slow
def test_expensive_operation():
    pass

@pytest.mark.integration
def test_api_integration():
    pass
```

```bash
pytest -m slow           # Run only slow tests
pytest -m "not slow"     # Skip slow tests
pytest -m "integration"  # Run integration tests
```

### Verbose output

```bash
pytest -v   # Verbose
pytest -vv  # Very verbose
pytest -s   # Show print statements
```

### Stop on first failure

```bash
pytest -x        # Stop on first failure
pytest --maxfail=3  # Stop after 3 failures
```

### Coverage reporting

```bash
pytest --cov=mymodule
pytest --cov=mymodule --cov-report=html  # HTML report
```

## Testing cisTEMx Python Scripts

### For skill scripts

```python
# .claude/skills/my-skill/scripts/process_data.py
def process_file(input_path, output_path):
    with open(input_path) as f:
        data = f.read()
    result = transform(data)
    with open(output_path, 'w') as f:
        f.write(result)
    return result

# .claude/skills/my-skill/tests/test_process_data.py
import pytest
from pathlib import Path
from ..scripts.process_data import process_file

@pytest.fixture
def temp_files(tmp_path):
    """Provide temporary input/output files"""
    input_file = tmp_path / "input.txt"
    output_file = tmp_path / "output.txt"
    return input_file, output_file

def test_process_file(temp_files):
    # Arrange
    input_file, output_file = temp_files
    input_file.write_text("test data")

    # Act
    result = process_file(str(input_file), str(output_file))

    # Assert
    assert output_file.exists()
    assert result == "transformed test data"
```

### For functional test scripts

```python
# scripts/testing/programs/match_template/test_apoferritin.py
import pytest
import subprocess
from pathlib import Path

@pytest.fixture
def test_data_dir():
    return Path(__file__).parent / "test_data"

def test_match_template_apoferritin(test_data_dir):
    """Test match_template program with apoferritin dataset"""
    input_mrc = test_data_dir / "apoferritin.mrc"
    template = test_data_dir / "template.mrc"
    output = test_data_dir / "output.mrc"

    result = subprocess.run([
        "./match_template",
        "--input", str(input_mrc),
        "--template", str(template),
        "--output", str(output)
    ], capture_output=True, text=True)

    assert result.returncode == 0
    assert output.exists()
    # Verify output properties
```

## Common Patterns

### Testing exceptions

```python
def test_invalid_input_raises_error():
    with pytest.raises(ValueError) as exc_info:
        parse_input("invalid")

    assert "format" in str(exc_info.value).lower()
```

### Testing file operations

```python
def test_file_processing(tmp_path):
    # tmp_path is pytest built-in fixture
    input_file = tmp_path / "input.txt"
    input_file.write_text("test data")

    result = process_file(input_file)

    assert result.success
    assert (tmp_path / "output.txt").exists()
```

### Testing with environment variables

```python
def test_with_env_var(monkeypatch):
    # monkeypatch is pytest built-in fixture
    monkeypatch.setenv("API_KEY", "test-key")

    config = load_config()

    assert config.api_key == "test-key"
```

### Testing warnings

```python
def test_deprecation_warning():
    with pytest.warns(DeprecationWarning, match="deprecated"):
        legacy_function()
```

## Summary

**pytest strengths:**
- Simple, Pythonic syntax with powerful assertions
- Flexible fixture system for dependency injection
- Automatic test discovery
- Easy parametrization
- Excellent failure reporting

**Key patterns:**
- Use fixtures for setup/teardown and dependency injection
- Organize fixtures in conftest.py
- Use parametrization for data-driven tests
- Mock external dependencies with unittest.mock
- Follow AAA pattern for clear test structure
- Use markers to categorize tests

**Best practices:**
- Keep fixtures focused and minimal
- Use narrow scope when possible
- Group related tests in classes
- Verify critical behavior, not implementation
- Make mocks behave like real objects
