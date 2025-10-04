# C++ and CUDA Static Analysis with clang-tidy

Comprehensive guide for static analysis of cisTEM's C++ and CUDA code using clang-tidy and related LLVM tools.

## Overview

**Primary Tool:** clang-tidy-14 (LLVM 14.x)
**Supporting Tools:** Bear (compilation database), scan-build, clang-query
**Target Code:** All C++ and CUDA source files, with initial focus on `src/core/tensor/`

## Multi-Tier Check System

cisTEM uses a **tiered approach** to static analysis, balancing thoroughness with practicality. All checks are defined in the project's `.clang-tidy` configuration, but we selectively run different subsets based on context using the `--checks=` command-line override.

### Tier 0: Blocker Checks ⛔

**Purpose:** Must-fix issues that indicate serious bugs
**When to run:** Pre-commit (< 30 seconds)
**CI policy:** Block merge

```yaml
bugprone-use-after-move
bugprone-dangling-handle
bugprone-undelegated-constructor
performance-move-const-arg
cert-err52-cpp
```

**Rationale:**
- `use-after-move`: Critical for Tensor's move semantics
- `dangling-handle`: Prevents references to temporaries (common in template code)
- `undelegated-constructor`: Memory initialization bugs
- `move-const-arg`: Incorrect move usage that defeats optimization
- `err52-cpp`: Uninitialized variables in numerical code

### Tier 1: Critical Checks 🔴

**Purpose:** Important issues affecting correctness and performance
**When to run:** Before PR, regular development
**CI policy:** Warn, consider blocking

```yaml
# All bugprone checks (with exceptions)
bugprone-*
-bugprone-easily-swappable-parameters
-bugprone-narrowing-conversions  # Too noisy for scientific computing

# Key performance checks
performance-unnecessary-copy-initialization
performance-for-range-copy
performance-noexcept-move-constructor
performance-inefficient-vector-operation

# Essential modernization
modernize-use-nullptr
modernize-use-override
modernize-use-emplace

# Special member functions
cppcoreguidelines-special-member-functions
```

**Key checks explained:**

**Memory Safety:**
- `bugprone-multiple-statement-macro`: Macro safety in parallel regions
- `bugprone-suspicious-memory-comparison`: memcmp on non-trivial types
- `misc-unconventional-assign-operator`: Assignment operator correctness

**Performance:**
- `performance-unnecessary-copy-initialization`: Major impact in template code
- `performance-for-range-copy`: Range loop efficiency
- `performance-noexcept-move-constructor`: Enables optimizations

### Tier 2: Important Checks 🟡

**Purpose:** Quality improvements and best practices
**When to run:** CI standard checks, before significant PRs
**CI policy:** Report as warnings

```yaml
# All performance checks
performance-*

# Most modernization checks
modernize-*
-modernize-use-trailing-return-type  # Style preference
-modernize-use-nodiscard  # Useful but very verbose

# Certificate security
cert-err*
cert-flp30-c  # Floating-point loop counters

# Basic concurrency
concurrency-mt-unsafe

# Memory management
misc-unconventional-assign-operator
misc-new-delete-overloads
cppcoreguidelines-owning-memory
```

**Performance checks detail:**
- `performance-type-promotion-in-math-fn`: sin(float) vs sin(double) - critical for numerical accuracy
- `performance-inefficient-string-concatenation`: wxString building
- `performance-trivially-destructible`: Enables compiler optimizations
- `performance-no-automatic-move`: Catches missed RVO/NRVO opportunities

**Concurrency checks:**
- `concurrency-mt-unsafe`: Non-thread-safe functions (rand, strtok, etc.)
- `bugprone-multiple-statement-macro`: Dangerous in OpenMP regions

### Tier 3: Aspirational Checks 🔵

**Purpose:** Deep analysis, code quality, style consistency
**When to run:** Weekly deep analysis, before major releases
**CI policy:** Informational only

```yaml
# Readability (verbose but valuable)
readability-*
-readability-magic-numbers  # Too noisy for scientific code
-readability-identifier-length  # Conflicts with math notation
-readability-function-cognitive-complexity  # Set threshold high

# Full C++ Core Guidelines
cppcoreguidelines-*
-cppcoreguidelines-avoid-magic-numbers
-cppcoreguidelines-pro-bounds-pointer-arithmetic  # Scientific computing needs this
-cppcoreguidelines-pro-bounds-constant-array-index

# Deep static analysis
clang-analyzer-*

# Additional cert checks
cert-*

# Miscellaneous
misc-*
```

**Readability highlights:**
- `readability-identifier-naming`: Enforce naming conventions (configure for cisTEM style)
- `readability-redundant-member-init`: Clean up constructors
- `readability-static-accessed-through-instance`: Template clarity
- `readability-const-return-type`: Const correctness

**Core Guidelines:**
- `cppcoreguidelines-no-malloc`: Prefer RAII
- `cppcoreguidelines-pro-type-reinterpret-cast`: Type safety (use sparingly in CUDA code)

### Tier 4: Research / Future Custom Checks 🔬

**Purpose:** Document patterns for potential custom checks
**Status:** Not implemented, design phase

**CUDA-specific patterns:**
- Host/device memory transfer validation
- Kernel launch configuration checks
- Shared memory bank conflict detection
- Warp divergence patterns
- Texture memory usage validation

**FFT-specific patterns:**
- FFTW plan lifecycle (create → execute → destroy)
- Thread-safe plan creation with mutexes
- Memory alignment for SIMD (16/32/64 byte)

**Memory pool patterns:**
- Acquire/release pairing
- Pool capacity validation
- Device/host pool selection correctness

**Image processing patterns:**
- Boundary condition consistency
- Index calculation validation (logical to physical)
- Padding/stride arithmetic correctness

See `custom_checks.md` for detailed pattern documentation.

## Command-Line Check Filtering

The `.clang-tidy` config file contains ALL checks, but you can override with `--checks=`:

```bash
# Run only Tier 0 (blocker)
clang-tidy-14 -p build/clang-tidy-debug \
  --checks='bugprone-use-after-move,bugprone-dangling-handle,performance-move-const-arg,cert-err52-cpp' \
  src/core/tensor/tensor.h

# Run Tier 1 (critical) - use wildcards
clang-tidy-14 -p build/clang-tidy-debug \
  --checks='bugprone-*,performance-unnecessary-copy-initialization,modernize-use-nullptr' \
  src/core/tensor/

# Run specific category
clang-tidy-14 -p build/clang-tidy-debug \
  --checks='performance-*' \
  src/core/tensor/

# Run all except certain checks
clang-tidy-14 -p build/clang-tidy-debug \
  --checks='*,-readability-magic-numbers,-cppcoreguidelines-avoid-magic-numbers' \
  src/core/tensor/

# Use full config (Tier 3 - all checks)
clang-tidy-14 -p build/clang-tidy-debug src/core/tensor/
```

## VS Code Tasks

All tasks are defined in `.vscode/tasks.json`. Use `Ctrl+Shift+P` → `Tasks: Run Task`.

### Setup Tasks

**"Configure for Static Analysis (Clang + Bear)"**
- Creates `build/clang-tidy-debug` with Clang compiler
- Required before first analysis run

**"Build with Bear (Generate compile_commands.json)"**
- Builds project while capturing compilation commands
- Generates `build/clang-tidy-debug/compile_commands.json`
- Run after adding new source files or changing build config

**"Link compile_commands.json to Root"**
- Creates symlink in project root for IDE integration
- Enables clangd and other tools

### Analysis Tasks (Tiered)

**"clang-tidy: Critical Checks Only" (Tier 0)**
- Fast pre-commit analysis (< 30 seconds)
- Blocker checks only
- Use before every commit

**"clang-tidy: Phase 1 (Fast - Pre-commit)" (Tier 0 + Tier 1)**
- Standard development analysis (< 1 minute)
- Critical issues that should be fixed promptly
- Use before pushing to remote

**"clang-tidy: Phase 2 (Standard - CI)" (Tier 0-2)**
- Comprehensive pre-PR analysis (< 2 minutes)
- Matches CI check level
- Use before creating pull request

**"clang-tidy: Phase 3 (Deep - Weekly)" (All tiers)**
- Full analysis with all checks (5-10 minutes)
- Uses complete `.clang-tidy` configuration
- Use before major releases or weekly

### Focused Analysis Tasks

**"Analyze Tensor Code (clang-tidy)"**
- Runs standard tier on all `src/core/tensor/` files
- Uses configuration from `.clang-tidy`

**"Analyze Changed Files (clang-tidy-diff)"**
- Only analyzes lines changed vs. base branch
- Very fast, ideal for large PRs
- Prompts for base branch (default: master)

**"Deep Analysis (scan-build)"**
- Uses Clang Static Analyzer (deeper than clang-tidy)
- Generates HTML report in `.claude/cache/scan-results/`
- Slower but catches different bug classes

### Category-Specific Tasks

**"clang-tidy: Memory Safety Focus"**
```json
--checks='bugprone-*,cppcoreguidelines-owning-memory,cppcoreguidelines-no-malloc,misc-unconventional-assign-operator,misc-new-delete-overloads'
```

**"clang-tidy: Performance Focus"**
```json
--checks='performance-*,cert-flp30-c,readability-redundant-*'
```

**"clang-tidy: Concurrency Focus"**
```json
--checks='concurrency-*,bugprone-multiple-statement-macro,misc-static-assert'
```

## Helper Scripts

Located in `scripts/linting/cpp_cuda/`:

### `generate_compile_db.sh`
```bash
# Generate compilation database
./scripts/linting/cpp_cuda/generate_compile_db.sh [build_dir] [clean]

# Examples:
./scripts/linting/cpp_cuda/generate_compile_db.sh  # Uses build/clang-tidy-debug
./scripts/linting/cpp_cuda/generate_compile_db.sh build/clang-tidy-debug true  # Clean build
```

### `analyze_blocker.sh`
```bash
# Fast pre-commit checks (Tier 0)
./scripts/linting/cpp_cuda/analyze_blocker.sh [path]

# Examples:
./scripts/linting/cpp_cuda/analyze_blocker.sh  # Analyze all Tensor code
./scripts/linting/cpp_cuda/analyze_blocker.sh src/core/tensor/tensor.h  # Single file
```

### `analyze_critical.sh`
```bash
# Standard development checks (Tier 0 + 1)
./scripts/linting/cpp_cuda/analyze_critical.sh [path]
```

### `analyze_standard.sh`
```bash
# Pre-PR checks (Tier 0-2, matches CI)
./scripts/linting/cpp_cuda/analyze_standard.sh [path]
```

### `analyze_deep.sh`
```bash
# Full analysis (all tiers)
./scripts/linting/cpp_cuda/analyze_deep.sh [path]
```

### `analyze_diff.sh`
```bash
# Analyze only changed lines
./scripts/linting/cpp_cuda/analyze_diff.sh [tier] [base_branch]

# Examples:
./scripts/linting/cpp_cuda/analyze_diff.sh blocker master
./scripts/linting/cpp_cuda/analyze_diff.sh standard origin/develop
./scripts/linting/cpp_cuda/analyze_diff.sh deep HEAD~5
```

### `analyze_category.sh`
```bash
# Analyze specific category
./scripts/linting/cpp_cuda/analyze_category.sh <category> [path]

# Examples:
./scripts/linting/cpp_cuda/analyze_category.sh performance src/core/tensor/
./scripts/linting/cpp_cuda/analyze_category.sh modernize src/core/
./scripts/linting/cpp_cuda/analyze_category.sh readability src/core/tensor/tensor.h
```

## Developer Workflow

### Daily Development

```bash
# 1. Make changes to code
vim src/core/tensor/tensor.h

# 2. Quick check before commit (Tier 0)
./scripts/linting/cpp_cuda/analyze_blocker.sh src/core/tensor/tensor.h

# 3. Fix any issues
# ... edit code ...

# 4. Commit
git add src/core/tensor/tensor.h
git commit -m "Add new Tensor method"
```

### Before Creating PR

```bash
# 1. Check what files changed
git diff --name-only master

# 2. Analyze only changes (Tier 2)
./scripts/linting/cpp_cuda/analyze_diff.sh standard master

# 3. Or run standard tier on all Tensor code
./scripts/linting/cpp_cuda/analyze_standard.sh src/core/tensor/

# 4. Fix warnings, create PR
```

### Weekly Deep Analysis

```bash
# Run comprehensive analysis on Tensor subsystem
./scripts/linting/cpp_cuda/analyze_deep.sh src/core/tensor/

# Review report, create issues for refactoring
# Many Tier 3 items may be deferred as technical debt
```

### Focused Investigation

```bash
# Check performance issues in specific file
./scripts/linting/cpp_cuda/analyze_category.sh performance src/core/image.cpp

# Memory safety review before release
./scripts/linting/cpp_cuda/analyze_category.sh bugprone src/core/
```

## Suppressing False Positives

Use `// NOLINT` comments **sparingly** and **only with justification**:

```cpp
// Intentional pointer arithmetic for SIMD alignment
float* aligned_ptr = ptr + offset;  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)

// wxWidgets macro requires this pattern
IMPLEMENT_APP(MyApp)  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

// Suppressing specific check
void* raw_data = new char[size];  // NOLINT(cppcoreguidelines-owning-memory)
                                   // Rationale: managed by external C library

// Suppressing all checks on line (use very rarely!)
legacy_function();  // NOLINT
```

**Better alternatives:**
1. Fix the code to comply
2. Refactor to avoid the pattern
3. Move check to lower tier if consistently problematic
4. Document why pattern is necessary in comments

## Configuration Files

### Project `.clang-tidy`

Located at project root: `/workspaces/cisTEM/worktrees/refactor_image_class/.clang-tidy`

```yaml
---
# cisTEM C++/CUDA Static Analysis Configuration
# Comprehensive multi-tier check system
# Use --checks= flag to selectively run tiers

Checks: >
  -*,
  # === TIER 0: Blocker (Always Run) ===
  bugprone-use-after-move,
  bugprone-dangling-handle,
  bugprone-undelegated-constructor,
  performance-move-const-arg,
  cert-err52-cpp,

  # === TIER 1: Critical ===
  bugprone-*,
  -bugprone-easily-swappable-parameters,
  -bugprone-narrowing-conversions,
  performance-unnecessary-copy-initialization,
  performance-for-range-copy,
  performance-noexcept-move-constructor,
  performance-inefficient-vector-operation,
  modernize-use-nullptr,
  modernize-use-override,
  modernize-use-emplace,
  cppcoreguidelines-special-member-functions,

  # === TIER 2: Important ===
  performance-*,
  modernize-*,
  -modernize-use-trailing-return-type,
  -modernize-use-nodiscard,
  cert-err*,
  cert-flp30-c,
  concurrency-mt-unsafe,
  misc-unconventional-assign-operator,
  misc-new-delete-overloads,
  cppcoreguidelines-owning-memory,

  # === TIER 3: Aspirational ===
  readability-*,
  -readability-magic-numbers,
  -readability-identifier-length,
  cppcoreguidelines-*,
  -cppcoreguidelines-avoid-magic-numbers,
  -cppcoreguidelines-pro-bounds-pointer-arithmetic,
  -cppcoreguidelines-pro-bounds-constant-array-index,
  clang-analyzer-*,
  cert-*,
  misc-*

WarningsAsErrors: ''  # Don't block builds, just warn

HeaderFilterRegex: 'src/core/tensor/.*'  # Focus on Tensor initially

CheckOptions:
  # Naming conventions (cisTEM style)
  - key: readability-identifier-naming.ClassCase
    value: CamelCase
  - key: readability-identifier-naming.StructCase
    value: CamelCase
  - key: readability-identifier-naming.FunctionCase
    value: CamelCase
  - key: readability-identifier-naming.MethodCase
    value: CamelCase
  - key: readability-identifier-naming.VariableCase
    value: lower_case
  - key: readability-identifier-naming.ParameterCase
    value: lower_case
  - key: readability-identifier-naming.ConstantCase
    value: lower_case
  - key: readability-identifier-naming.EnumConstantCase
    value: UPPER_CASE

  # Performance tuning
  - key: performance-unnecessary-copy-initialization.AllowedTypes
    value: 'wxString;wxFileName'  # wxWidgets uses copy-on-write

  # Complexity thresholds
  - key: readability-function-cognitive-complexity.Threshold
    value: '50'  # Higher for scientific computing
  - key: readability-function-size.LineThreshold
    value: '200'
```

### Subsystem Overrides

Create `.clang-tidy` in subdirectories to override settings:

**`src/core/tensor/.clang-tidy`:**
```yaml
# Tensor-specific overrides
HeaderFilterRegex: '.*'  # Check all headers in Tensor code

CheckOptions:
  # Stricter for new code
  - key: readability-function-cognitive-complexity.Threshold
    value: '30'
```

**`src/gui/.clang-tidy`:**
```yaml
# GUI code has different patterns
Checks: >
  -*,
  bugprone-*,
  modernize-use-nullptr,
  modernize-use-override

# Allow wxWidgets patterns
CheckOptions:
  - key: performance-unnecessary-copy-initialization.AllowedTypes
    value: 'wxString;wxFileName;wxArrayString'
```

## Compilation Database

clang-tidy requires a **compilation database** (`compile_commands.json`) that records the exact build commands for each source file.

### Generating with Bear

Bear intercepts compiler calls to capture commands:

```bash
# One-time setup
mkdir -p build/clang-tidy-debug
cd build/clang-tidy-debug
CC=clang CXX=clang++ ../../configure --enable-debugmode --with-cuda=/usr/local/cuda

# Generate database (incremental)
bear -- make -j$(nproc)

# Generate database (clean build - more complete)
make clean && bear -- make -j$(nproc)

# Link to project root for IDE tools
cd ../..
ln -sf build/clang-tidy-debug/compile_commands.json .
```

### When to Regenerate

Regenerate when:
- Adding new source files
- Changing build configuration
- Updating compiler flags
- After `./regenerate_project.b`
- Database seems incomplete or stale

### Troubleshooting

**Database not found:**
```bash
Error: Compilation database not found!
```
Solution: Run "Build with Bear" VS Code task

**Stale entries:**
If analysis shows errors for recently modified files, regenerate database.

**Missing entries:**
Files not compiled won't appear in database. Do clean build with Bear.

## Advanced Tools

### scan-build (Clang Static Analyzer)

Deeper analysis than clang-tidy, different bug detection:

```bash
# Run scan-build
cd build/clang-tidy-debug
scan-build-14 -o ../../.claude/cache/scan-results make -j$(nproc)

# View HTML report
firefox .claude/cache/scan-results/*/index.html
```

**What it finds:**
- Null pointer dereferences
- Memory leaks
- Dead code
- Undefined behavior
- Division by zero

**When to use:** Before major releases, investigating crashes

### clang-query (AST Pattern Matching)

Search for code patterns using AST queries:

```bash
# Find all new expressions (memory allocations)
clang-query-14 -p build/clang-tidy-debug src/core/tensor/*.cpp \
  -c "match cxxNewExpr()"

# Find all raw pointer parameters
clang-query-14 -p build/clang-tidy-debug src/core/tensor/tensor.h \
  -c "match parmVarDecl(hasType(pointerType()))"

# Find all CUDA kernel launches
clang-query-14 -p build/clang-tidy-debug src/core/tensor/*.cu \
  -c "match cudaKernelCallExpr()"
```

**When to use:** Researching refactoring opportunities, preparing custom checks

### clang-check (Syntax Validation)

Quick syntax/semantic check without full build:

```bash
clang-check-14 -p build/clang-tidy-debug src/core/tensor/type_traits.h --analyze
```

**When to use:** Validating template instantiation, checking header-only code

## CI Integration

### GitHub Actions Workflow

See `.github/workflows/static_analysis.yml` (planned):

```yaml
name: Static Analysis

on:
  pull_request:
    paths:
      - 'src/core/tensor/**'
      - 'src/core/**/*.h'

jobs:
  clang-tidy:
    runs-on: ubuntu-20.04
    container:
      image: cistem-dev:latest
    steps:
      - uses: actions/checkout@v4

      - name: Generate compilation database
        run: |
          mkdir -p build/clang-analysis
          cd build/clang-analysis
          CC=clang CXX=clang++ ../../configure --enable-debugmode
          bear -- make -j$(nproc)

      - name: Run clang-tidy (Tier 2)
        run: |
          git fetch origin ${{ github.base_ref }}
          git diff -U0 origin/${{ github.base_ref }} | \
            clang-tidy-diff-14.py -p1 -path build/clang-analysis > analysis.txt || true

      - name: Post results
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const output = fs.readFileSync('analysis.txt', 'utf8');
            if (output.trim()) {
              github.rest.issues.createComment({
                issue_number: context.issue.number,
                owner: context.repo.owner,
                repo: context.repo.repo,
                body: '## Static Analysis Results\n```\n' + output + '\n```'
              });
            }
```

### CI Policy Evolution

**Phase 1 (Current):** Informational only
- Post results as PR comments
- Don't block merges
- Gather data on false positive rate

**Phase 2 (After tuning):** Soft enforcement
- Fail CI on Tier 0 violations
- Warn on Tier 1-2
- Can override with "skip-lint" label

**Phase 3 (Mature):** Hard enforcement
- Block merge on Tier 0-1 violations
- Require documented suppression for exceptions
- Trend reporting for technical debt

## Custom Check Development

For cisTEM-specific patterns, we can develop custom clang-tidy checks.

### Process

1. **Document pattern** in `custom_checks.md`
2. **Prototype** using clang-query to match AST nodes
3. **Implement** as clang-tidy check module
4. **Test** on representative code samples
5. **Measure** false positive rate
6. **Deploy** by adding to project configuration

### Example: CUDA Memory Transfer Validation

**Pattern to detect:**
```cpp
// BAD: Copying from device to device with cudaMemcpyHostToDevice
float* d_src;
float* d_dst;
cudaMemcpy(d_dst, d_src, size, cudaMemcpyHostToDevice);  // Wrong direction!
```

**Custom check pseudocode:**
```cpp
// Check name: cistem-cuda-memcpy-direction
// Analyzes cudaMemcpy calls to validate direction matches pointer types
if (call.function == "cudaMemcpy") {
  auto direction = call.arg(3);
  bool src_is_device = isDevicePointer(call.arg(1));
  bool dst_is_device = isDevicePointer(call.arg(0));

  if (direction == cudaMemcpyHostToDevice && src_is_device) {
    diag("using HostToDevice with device source pointer");
  }
  // ... other combinations
}
```

See detailed guide in `custom_checks.md` (to be created).

## Metrics and Success Criteria

### Performance Metrics

**Analysis Speed:**
- Tier 0 (Blocker): < 30 seconds on Tensor subsystem
- Tier 1 (Critical): < 1 minute
- Tier 2 (Standard): < 2 minutes
- Tier 3 (Deep): < 10 minutes

**False Positive Rate:**
- Target: < 5% across all tiers
- Acceptable: < 10% for Tier 3
- Action threshold: > 15% requires check tuning

### Code Quality Metrics

**Issue Trending:**
- Track violations per tier over time
- Measure reduction in new violations
- Monitor technical debt (deferred Tier 3 items)

**Coverage:**
- 100% of new code analyzed (Tier 2) before merge
- 80% of existing code baseline established within 3 months
- Expand beyond Tensor to other subsystems within 6 months

## Troubleshooting

### Common Issues

**Issue:** "compilation database not found"
```bash
# Solution: Generate database
cd build/clang-tidy-debug && bear -- make -j$(nproc)
ln -sf build/clang-tidy-debug/compile_commands.json .
```

**Issue:** "header file not found" during analysis
```bash
# Solution: Verify database has correct include paths
jq '.[0].command' build/clang-tidy-debug/compile_commands.json | grep -- -I
```

**Issue:** Too many false positives
```bash
# Solution 1: Check if using correct tier (maybe using Tier 3 too early)
# Solution 2: Add check to exclusion list in .clang-tidy
# Solution 3: Tune check parameters in CheckOptions
```

**Issue:** Analysis is very slow
```bash
# Solution 1: Use parallel execution
run-clang-tidy-14 -j$(nproc) ...

# Solution 2: Analyze subsystem not entire codebase
./scripts/linting/cpp_cuda/analyze_standard.sh src/core/tensor/

# Solution 3: Use -quiet flag to reduce I/O
run-clang-tidy-14 -quiet ...
```

## References

- [clang-tidy Documentation](https://clang.llvm.org/extra/clang-tidy/)
- [clang-tidy Check List](https://clang.llvm.org/extra/clang-tidy/checks/list.html)
- [Bear Documentation](https://github.com/rizsotto/Bear)
- [JSON Compilation Database Format](https://clang.llvm.org/docs/JSONCompilationDatabase.html)
- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/)
- [Clang Static Analyzer](https://clang.llvm.org/docs/ClangStaticAnalyzer.html)

## Quick Reference

```bash
# Generate compilation database
cd build/clang-tidy-debug && bear -- make -j$(nproc)

# Run specific tier
./scripts/linting/cpp_cuda/analyze_blocker.sh  # Tier 0
./scripts/linting/cpp_cuda/analyze_critical.sh  # Tier 0+1
./scripts/linting/cpp_cuda/analyze_standard.sh  # Tier 0-2
./scripts/linting/cpp_cuda/analyze_deep.sh      # All tiers

# Run on changed files
./scripts/linting/cpp_cuda/analyze_diff.sh standard master

# Run specific category
./scripts/linting/cpp_cuda/analyze_category.sh performance

# List all available checks
clang-tidy-14 -list-checks -checks='*' | less

# Explain configuration
clang-tidy-14 -explain-config
```
