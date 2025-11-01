# Static Analysis Reference
## Linting and Code Quality Tools for cisTEMx

### Overview
cisTEMx employs comprehensive static analysis to catch bugs, performance issues, and style inconsistencies early in development.

### Multi-Tier Check System

#### Tier 1: Blocker Checks (Pre-commit)
**Runtime**: < 30 seconds
**Purpose**: Catch critical issues before commit
**Script**: `./scripts/linting/cpp_cuda/analyze_blocker.sh`

```bash
# Run blocker-level checks
./scripts/linting/cpp_cuda/analyze_blocker.sh

# VS Code task
# "clang-tidy: Phase 1 (Fast - Pre-commit)"
```

**Checks include**:
- Null pointer dereferences
- Use-after-free
- Buffer overflows
- Critical security issues

#### Tier 2: Standard Checks (Pre-PR)
**Runtime**: 2-5 minutes
**Purpose**: Comprehensive analysis before pull requests
**Script**: `./scripts/linting/cpp_cuda/analyze_standard.sh`

```bash
# Run standard checks
./scripts/linting/cpp_cuda/analyze_standard.sh

# VS Code task
# "clang-tidy: Phase 2 (Standard - CI)"
```

**Checks include**:
- All blocker checks
- Performance issues
- Readability concerns
- Modern C++ recommendations

#### Tier 3: Deep Analysis (Weekly/Release)
**Runtime**: 15-30 minutes
**Purpose**: Exhaustive analysis for releases
**Script**: `./scripts/linting/cpp_cuda/analyze_deep.sh`

**Checks include**:
- All standard checks
- Complex control flow analysis
- Cross-translation-unit analysis
- CUDA-specific checks

### Primary Tools

#### C++ and CUDA
**Tool**: clang-tidy
**Config**: `.clang-tidy` in project root
**Documentation**: `scripts/linting/cpp_cuda/CLAUDE.md`

Key check categories:
- `bugprone-*`: Common bug patterns
- `performance-*`: Performance anti-patterns
- `readability-*`: Code clarity issues
- `modernize-*`: Modern C++ opportunities
- `cppcoreguidelines-*`: C++ Core Guidelines

#### Shell Scripts (Planned)
**Tool**: shellcheck
**Documentation**: `scripts/linting/shell/CLAUDE.md`

#### Python (Planned)
**Tools**: pylint, flake8, black
**Documentation**: `scripts/linting/python/CLAUDE.md`

#### Build Systems (Planned)
**Focus**: Autotools/CMake validation
**Documentation**: `scripts/linting/build_systems/CLAUDE.md`

### Integration with Development Workflow

#### Pre-commit Hook
```bash
# Add to .git/hooks/pre-commit
#!/bin/bash
./scripts/linting/cpp_cuda/analyze_blocker.sh || exit 1
```

#### VS Code Tasks
Available tasks in Command Palette:
1. "clang-tidy: Phase 1 (Fast - Pre-commit)"
2. "clang-tidy: Phase 2 (Standard - CI)"
3. "clang-tidy: Phase 3 (Deep - Release)"

#### CI Pipeline
Standard checks run automatically on:
- Pull requests
- Commits to protected branches

### Suppressing False Positives

#### Inline Suppression
```cpp
// NOLINTNEXTLINE(bugprone-branch-clone)
if (condition) { /* intentionally similar */ }

// NOLINT(modernize-use-nullptr)
char* ptr = NULL;  // Required for C API compatibility
```

#### File-level Suppression
Add to `.clang-tidy`:
```yaml
Checks: '-readability-identifier-naming'
```

### Common Issues and Solutions

#### "Variable 'x' is not initialized"
- Initialize all variables at declaration
- Use default member initializers for class members

#### "Use nullptr instead of NULL"
- Modern code: Use `nullptr`
- C API boundaries: Suppress with NOLINT

#### "Consider using std::array"
- New code: Use `std::array` for fixed-size arrays
- Legacy code: Gradually migrate when touching the code

### Performance Considerations

#### Parallel Analysis
```bash
# Use all available cores
run-clang-tidy -j$(nproc) -p build/debug
```

#### Incremental Analysis
Only analyze changed files:
```bash
git diff --name-only | grep -E '\.(cpp|h)$' | xargs clang-tidy
```

### Creating Custom Checks

For project-specific patterns, create custom clang-tidy checks:
1. Define check in `scripts/linting/custom/`
2. Register in `.clang-tidy`
3. Add to appropriate tier

### Related Documentation
- Full linting documentation: `scripts/linting/CLAUDE.md`
- C++/CUDA specific: `scripts/linting/cpp_cuda/CLAUDE.md`
- Code style standards: `code_style_standards.md`