# Python Linting

Static analysis, linting, and formatting for Python code in cisTEMx.

## Status

🚧 **Planned - Not Yet Implemented**

This guide is a placeholder for future Python linting integration.

## Planned Tools

### pylint

**Purpose:** Comprehensive Python linting
**Website:** <https://pylint.org/>

**What it detects:**

- Code style issues (PEP 8)
- Programming errors
- Code smells
- Unused imports/variables
- Refactoring suggestions

### flake8

**Purpose:** Fast style and error checking
**Website:** <https://flake8.pycqa.org/>

**What it combines:**

- PyFlakes (logic errors)
- pycodestyle (PEP 8)
- McCabe (complexity)

### black

**Purpose:** Opinionated code formatter
**Website:** <https://black.readthedocs.io/>

**What it does:**

- Automatic code formatting
- Consistent style
- Minimal configuration
- PEP 8 compliant

### mypy

**Purpose:** Static type checking
**Website:** <https://mypy.readthedocs.io/>

**What it does:**

- Type hint validation
- Type inference
- Catches type errors
- Gradual typing support

### isort

**Purpose:** Import statement sorting
**Website:** <https://pycqa.github.io/isort/>

**What it does:**

- Organize imports
- Group by type (stdlib, third-party, local)
- Consistent ordering

## Scope

Python code in cisTEMx (currently limited):

- **Build scripts** - Python-based build tools
- **Testing utilities** - Test harness and validation scripts
- **Data processing** - Image conversion and analysis scripts
- **CI/CD scripts** - GitHub Actions helpers

## Planned Configuration

### `pyproject.toml`

Located at project root (centralizes Python tool config):

```toml
[tool.black]
line-length = 100
target-version = ['py38', 'py39', 'py310']
include = '\.pyi?$'

[tool.isort]
profile = "black"
line_length = 100
multi_line_output = 3
include_trailing_comma = true

[tool.pylint.main]
max-line-length = 100
disable = [
    "C0103",  # Invalid name (too strict for scientific code)
    "R0913",  # Too many arguments (common in scientific functions)
]

[tool.pylint.design]
max-args = 10
max-locals = 20

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false  # Gradual adoption
```

### `.flake8`

Located at project root:

```ini
[flake8]
max-line-length = 100
extend-ignore = E203, W503
exclude =
    .git,
    __pycache__,
    build,
    dist,
    .venv

per-file-ignores =
    __init__.py: F401
```

## Planned Workflow

### Development

**Format code:**

```bash
# Format all Python files
black scripts/

# Check formatting (CI)
black --check scripts/

# Sort imports
isort scripts/
```

**Lint code:**

```bash
# Fast check (pre-commit)
flake8 scripts/

# Comprehensive analysis
pylint scripts/

# Type checking
mypy scripts/
```

### CI Integration

```yaml
# .github/workflows/python_lint.yml
name: Python Linting

on:
  pull_request:
    paths:
      - '**.py'

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install tools
        run: pip install black flake8 pylint mypy isort

      - name: Check formatting
        run: black --check .

      - name: Check imports
        run: isort --check .

      - name: Lint with flake8
        run: flake8 .

      - name: Lint with pylint
        run: pylint scripts/

      - name: Type check
        run: mypy scripts/
```

## Tiered Approach

Similar to C++ linting, use tiers:

**Tier 0 - Blocker:**

- Syntax errors
- Undefined names
- Import errors

**Tier 1 - Critical:**

- PEP 8 violations
- Common bugs (comparison to None with ==)
- Unused variables
- Formatting issues

**Tier 2 - Important:**

- Code complexity
- Documentation strings
- Naming conventions
- Type hints

**Tier 3 - Aspirational:**

- Refactoring suggestions
- Design patterns
- Full type coverage

## VS Code Tasks (Planned)

```json
{
  "label": "Format Python (black)",
  "type": "shell",
  "command": "black ${file}",
  "problemMatcher": []
},
{
  "label": "Lint Python (flake8)",
  "type": "shell",
  "command": "flake8 ${file}",
  "problemMatcher": {
    "owner": "flake8",
    "pattern": {
      "regexp": "^(.*):(\\d+):(\\d+):\\s+([A-Z]\\d+)\\s+(.*)$",
      "file": 1,
      "line": 2,
      "column": 3,
      "code": 4,
      "message": 5
    }
  }
},
{
  "label": "Type Check Python (mypy)",
  "type": "shell",
  "command": "mypy ${file}",
  "problemMatcher": {
    "owner": "mypy",
    "pattern": {
      "regexp": "^(.*):(\\d+):\\s+(error|warning):\\s+(.*)$",
      "file": 1,
      "line": 2,
      "severity": 3,
      "message": 4
    }
  }
}
```

## Common Python Issues to Detect

**Mutable default arguments:**

```python
# BAD
def process_data(data, results=[]):
    results.append(data)
    return results

# GOOD
def process_data(data, results=None):
    if results is None:
        results = []
    results.append(data)
    return results
```

**Comparison to None:**

```python
# BAD
if x == None:

# GOOD
if x is None:
```

**Catching too broad exceptions:**

```python
# BAD
try:
    process()
except:
    pass

# GOOD
try:
    process()
except ValueError as e:
    logging.error(f"Processing failed: {e}")
```

**Missing type hints (gradually add):**

```python
# Current
def calculate_fft(image, size):
    return fft_result

# Future
def calculate_fft(image: np.ndarray, size: int) -> np.ndarray:
    return fft_result
```

## Installation (Future)

Add to `scripts/containers/top_image/Dockerfile`:

```dockerfile
# Python linting and formatting tools
RUN pip3 install --no-cache-dir \
    black \
    flake8 \
    pylint \
    mypy \
    isort \
    flake8-bugbear \
    flake8-comprehensions
```

Or use `requirements-dev.txt`:

```text
black>=23.0.0
flake8>=6.0.0
pylint>=2.17.0
mypy>=1.0.0
isort>=5.12.0
flake8-bugbear>=23.0.0
flake8-comprehensions>=3.10.0
```

## Integration with VS Code

VS Code Python extension supports:

- Auto-formatting on save (black)
- Inline linting (pylint/flake8)
- Type checking (mypy)
- Import organization (isort)

Add to `.vscode/settings.json`:

```json
{
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "python.sortImports.provider": "isort",
  "editor.formatOnSave": true
}
```

## Pre-commit Hooks (Future)

Use `pre-commit` framework:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
```

## Next Steps

1. Audit Python code in cisTEMx (currently minimal)
2. Install linting tools in container
3. Create `pyproject.toml` configuration
4. Run baseline analysis
5. Fix critical issues
6. Add VS Code integration
7. Set up pre-commit hooks
8. Integrate into CI

## References

- [PEP 8 - Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [Black Documentation](https://black.readthedocs.io/)
- [Pylint Documentation](https://pylint.readthedocs.io/)
- [Flake8 Documentation](https://flake8.pycqa.org/)
- [mypy Documentation](https://mypy.readthedocs.io/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

---

**Last Updated:** 2025-10-04
**Implementation Priority:** Low (Python usage in cisTEMx is currently minimal)
