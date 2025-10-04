# Build System Linting

Static analysis and validation for build system files in cisTEM.

## Status

🚧 **Planned - Not Yet Implemented**

This guide is a placeholder for future build system linting and validation.

## Overview

cisTEM uses multiple build systems that could benefit from validation:
- **Autotools** (primary) - configure.ac, Makefile.am files
- **CMake** (alternative) - CMakeLists.txt files
- **Docker** - Dockerfiles for development containers
- **Make** - Custom Makefile fragments

## Planned Tools

### For Autotools

**autoreconf warnings:**
```bash
# Check for common autotools issues
autoreconf -vi -Wall 2>&1 | grep -i warning
```

**automake strict mode:**
```bash
# Enable all warnings
automake --add-missing --copy --warnings=all
```

**Custom validation script:**
```bash
# Check for common issues in Makefile.am
scripts/linting/build_systems/check_automake.sh
```

### For CMake

**cmake-lint**
**Website:** https://github.com/cheshirekow/cmake_format

**What it detects:**
- CMake style issues
- Common mistakes
- Deprecated commands
- Variable naming

**Example:**
```bash
cmake-lint CMakeLists.txt
cmake-format -i CMakeLists.txt  # Format
```

### For Dockerfiles

**hadolint**
**Website:** https://github.com/hadolint/hadolint

**What it detects:**
- Best practice violations
- Common mistakes
- Security issues
- Optimization opportunities

**Example:**
```bash
hadolint scripts/containers/base_image/Dockerfile
hadolint scripts/containers/top_image/Dockerfile
```

## Build System Issues to Detect

### Autotools Issues

**Missing dependencies in Makefile.am:**
```makefile
# BAD - missing header dependency
tensor_test_SOURCES = test_tensor.cpp

# GOOD - includes headers
tensor_test_SOURCES = \
    test_tensor.cpp \
    ../../core/tensor/tensor.h \
    ../../core/tensor/type_traits.h
```

**Incorrect variable usage:**
```makefile
# BAD - using wrong variable
bin_PROGRAMS = my_program
my_program_SOURCES = main.cpp
my_program_CFLAGS = -O2

# GOOD - use CXXFLAGS for C++
bin_PROGRAMS = my_program
my_program_SOURCES = main.cpp
my_program_CXXFLAGS = -O2
```

**Missing SUBDIRS dependencies:**
```makefile
# Check that SUBDIRS are processed in correct order
# for proper dependency handling
```

### CMake Issues

**Deprecated commands:**
```cmake
# BAD
include_directories(${SOME_DIR})
link_directories(${SOME_LIB_DIR})

# GOOD
target_include_directories(my_target PRIVATE ${SOME_DIR})
target_link_directories(my_target PRIVATE ${SOME_LIB_DIR})
```

**Global variables:**
```cmake
# BAD
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# GOOD
target_compile_options(my_target PRIVATE -Wall)
```

### Dockerfile Issues

**Missing version pins:**
```dockerfile
# BAD
RUN apt-get update && apt-get install -y \
    clang \
    cmake

# GOOD
RUN apt-get update && apt-get install -y \
    clang-14 \
    cmake=3.22.* \
    && rm -rf /var/lib/apt/lists/*
```

**Inefficient layers:**
```dockerfile
# BAD
RUN apt-get update
RUN apt-get install -y package1
RUN apt-get install -y package2

# GOOD
RUN apt-get update && apt-get install -y \
    package1 \
    package2 \
    && rm -rf /var/lib/apt/lists/*
```

## Planned Validation Scripts

### `check_automake.sh`

```bash
#!/usr/bin/env bash
# Validate Makefile.am files for common issues

set -euo pipefail

echo "Checking Makefile.am files..."

# Find all Makefile.am files
find . -name "Makefile.am" | while read -r makefile; do
    echo "Checking: $makefile"

    # Check for C++ programs using CFLAGS instead of CXXFLAGS
    if grep -q "CFLAGS" "$makefile" && grep -q "\.cpp\|\.cc\|\.cxx" "$makefile"; then
        echo "  WARNING: Using CFLAGS for C++ code, should use CXXFLAGS"
    fi

    # Check for missing backslash in multi-line lists
    if grep -B1 "^[[:space:]]*[^[:space:]]" "$makefile" | grep -q "\\\\$"; then
        echo "  WARNING: Possible missing backslash in multi-line list"
    fi

    # Check for absolute paths (should be relative)
    if grep -q "/workspaces\|/home\|/usr/local" "$makefile"; then
        echo "  WARNING: Absolute path detected, should use variables"
    fi
done
```

### `check_dockerfiles.sh`

```bash
#!/usr/bin/env bash
# Validate Dockerfiles with hadolint

set -euo pipefail

find scripts/containers -name "Dockerfile" | while read -r dockerfile; do
    echo "Linting: $dockerfile"
    hadolint "$dockerfile" || true
done
```

### `validate_build_system.sh`

```bash
#!/usr/bin/env bash
# Comprehensive build system validation

set -euo pipefail

echo "=== Build System Validation ==="

echo "1. Checking autotools configuration..."
./scripts/linting/build_systems/check_automake.sh

echo "2. Checking CMake files..."
find . -name "CMakeLists.txt" -exec cmake-lint {} \; || true

echo "3. Checking Dockerfiles..."
./scripts/linting/build_systems/check_dockerfiles.sh

echo "4. Validating configure.ac..."
autoconf -Wall 2>&1 | grep -i warning || echo "  No warnings"

echo "=== Validation Complete ==="
```

## CI Integration (Planned)

```yaml
# .github/workflows/build_system_lint.yml
name: Build System Validation

on:
  pull_request:
    paths:
      - 'configure.ac'
      - '**/Makefile.am'
      - '**/CMakeLists.txt'
      - '**/Dockerfile'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install tools
        run: |
          pip install cmake-format
          wget -O /usr/local/bin/hadolint \
            https://github.com/hadolint/hadolint/releases/latest/download/hadolint-Linux-x86_64
          chmod +x /usr/local/bin/hadolint

      - name: Validate Autotools
        run: |
          autoreconf -vi -Wall 2>&1 | tee autotools.log
          if grep -qi warning autotools.log; then
            echo "::warning::Autotools warnings detected"
          fi

      - name: Lint CMake
        run: find . -name "CMakeLists.txt" -exec cmake-lint {} \;

      - name: Lint Dockerfiles
        run: find scripts/containers -name "Dockerfile" -exec hadolint {} \;
```

## VS Code Tasks (Planned)

```json
{
  "label": "Validate Build System",
  "type": "shell",
  "command": "${workspaceFolder}/scripts/linting/build_systems/validate_build_system.sh",
  "problemMatcher": []
},
{
  "label": "Lint Dockerfiles",
  "type": "shell",
  "command": "find scripts/containers -name 'Dockerfile' -exec hadolint {} \\;",
  "problemMatcher": {
    "owner": "hadolint",
    "pattern": {
      "regexp": "^(.*):(\\d+)\\s+(DL\\d+)\\s+(warning|error):\\s+(.*)$",
      "file": 1,
      "line": 2,
      "code": 3,
      "severity": 4,
      "message": 5
    }
  }
}
```

## Specific Checks for cisTEM

### Autotools Consistency

**Check that all source files are listed:**
```bash
# Find .cpp files not in any Makefile.am
find src -name "*.cpp" | while read -r file; do
    if ! grep -r "$(basename "$file")" --include="Makefile.am" .; then
        echo "WARNING: $file not referenced in Makefile.am"
    fi
done
```

**Verify CUDA configuration:**
```bash
# Ensure CUDA files are properly handled
grep -r "\.cu" --include="Makefile.am" . | \
    grep -v "CUDA_SOURCES\|_CUDA_SOURCES"
```

**Check for parallel build safety:**
```bash
# Verify proper dependency ordering
# (complex, would need custom script)
```

### CMake Modern Practices

**Target-based design:**
- All properties set via `target_*` commands
- No global `CMAKE_CXX_FLAGS` modifications
- Proper `PUBLIC`/`PRIVATE`/`INTERFACE` usage

**Generator expressions:**
- Use `$<BUILD_INTERFACE>` and `$<INSTALL_INTERFACE>`
- Conditional compilation flags

### Docker Best Practices

**Multi-stage builds:**
- Separate build and runtime stages
- Minimize final image size

**Layer optimization:**
- Group related operations
- Clean up in same layer

**Security:**
- Run as non-root user
- Scan for vulnerabilities

## Installation (Future)

```dockerfile
# Add to scripts/containers/top_image/Dockerfile

# Build system linting tools
RUN pip3 install --no-cache-dir cmake-format

RUN wget -O /usr/local/bin/hadolint \
    https://github.com/hadolint/hadolint/releases/download/v2.12.0/hadolint-Linux-x86_64 \
    && chmod +x /usr/local/bin/hadolint
```

## Next Steps

1. Create validation scripts for Autotools
2. Install cmake-lint and hadolint
3. Run baseline validation
4. Document common issues found
5. Add VS Code tasks
6. Integrate into CI
7. Create pre-commit hooks

## References

- [Autotools Mythbuster](https://autotools.io/)
- [CMake Best Practices](https://cliutils.gitlab.io/modern-cmake/)
- [Hadolint Rules](https://github.com/hadolint/hadolint#rules)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [GNU Automake Manual](https://www.gnu.org/software/automake/manual/)

---

**Last Updated:** 2025-10-04
**Implementation Priority:** Medium (after core language linting is established)
