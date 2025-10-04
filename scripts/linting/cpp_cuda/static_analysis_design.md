# Static Analysis Integration Plan for cisTEM

## Executive Summary

This plan outlines the integration of LLVM/Clang static analysis tools into the cisTEM development workflow, with a focus on improving code quality for the heavily templated Tensor system. The integration will use Bear to generate compilation databases and clang-tidy as the primary static analysis tool.

## Objectives

1. **Improve code quality** for templated C++ code in the Tensor system
2. **Catch bugs early** through static analysis during development
3. **Enforce modern C++ idioms** and performance best practices
4. **Provide actionable feedback** to developers without disrupting workflow
5. **Support incremental adoption** - analyze new/changed code first, gradually expand

## Current State Assessment

### Available Tools (LLVM 14)

| Tool | Purpose | Current Usage |
|------|---------|---------------|
| clang-format-14 | Code formatting | ✅ Active (pre-commit hook) |
| clang-tidy-14 | Static analysis & linting | ❌ Not integrated |
| clang-check-14 | Syntax/semantic checking | ❌ Not integrated |
| clang-query-14 | AST querying for patterns | ❌ Not integrated |
| scan-build-14 | Build-time static analysis | ❌ Not integrated |
| run-clang-tidy-14 | Parallel analysis runner | ❌ Not integrated |
| clang-tidy-diff-14.py | Analyze changed lines only | ❌ Not integrated |

### Build Configurations

- **Intel ICC/ICPC** (primary production builds)
- **Clang** (available, should be used for clang-tidy)
- **GCC** (GNU builds for CI)
- Multiple variants: debug, release, static, GPU, etc.

## Phase 1: Compilation Database Generation

### Bear Integration

**Why Bear?**
- Intercepts compiler invocations to capture exact build commands
- Works with any build system (important for our Autotools setup)
- Generates `compile_commands.json` that clang-tidy requires

**Implementation:**

1. **Install Bear** (completed)
   - Script: `scripts/containers/top_image/install_bear.sh`
   - Version: 2.4.3 from Ubuntu repos (simple apt install)
   - Later: Version 3.1.6 available from source (has more features)

2. **VS Code Task: Configure with Bear**
   ```json
   {
     "label": "Configure cisTEM for Static Analysis (Clang + Bear)",
     "type": "shell",
     "command": "cd build/clang-tidy-debug && ../../configure --enable-debugmode CC=clang CXX=clang++",
     "group": "build",
     "problemMatcher": []
   }
   ```

3. **VS Code Task: Build with Bear**
   ```json
   {
     "label": "Build with Bear (Generate compile_commands.json)",
     "type": "shell",
     "command": "cd build/clang-tidy-debug && bear -- make -j$(nproc)",
     "group": "build",
     "dependsOn": ["Configure cisTEM for Static Analysis (Clang + Bear)"],
     "problemMatcher": ["$gcc"]
   }
   ```

4. **VS Code Task: Clean Build with Bear**
   ```json
   {
     "label": "Clean Build with Bear (Full Database)",
     "type": "shell",
     "command": "cd build/clang-tidy-debug && make clean && bear -- make -j$(nproc)",
     "group": "build",
     "problemMatcher": ["$gcc"]
   }
   ```

5. **Link compilation database to project root**
   ```json
   {
     "label": "Link compile_commands.json to Root",
     "type": "shell",
     "command": "ln -sf build/clang-tidy-debug/compile_commands.json .",
     "problemMatcher": []
   }
   ```

**Rationale for Clang build:**
- Clang-tidy works best with Clang-generated compilation commands
- Ensures compiler diagnostics match static analysis
- Avoids ICC-specific extensions that might confuse analysis

## Phase 2: Clang-Tidy Configuration

### Configuration File Strategy

Create `.clang-tidy` in project root with curated checks:

```yaml
---
# cisTEM clang-tidy configuration
# Focus: Modern C++17, performance-critical numerical computing, template best practices

Checks: >
  -*,
  bugprone-*,
  -bugprone-easily-swappable-parameters,
  -bugprone-narrowing-conversions,
  modernize-*,
  -modernize-use-trailing-return-type,
  performance-*,
  readability-*,
  -readability-magic-numbers,
  -readability-identifier-length,
  cppcoreguidelines-*,
  -cppcoreguidelines-avoid-magic-numbers,
  -cppcoreguidelines-pro-bounds-pointer-arithmetic,
  -cppcoreguidelines-pro-bounds-constant-array-index,
  clang-analyzer-*

WarningsAsErrors: ''

HeaderFilterRegex: 'src/core/tensor/.*'

CheckOptions:
  - key: readability-identifier-naming.ClassCase
    value: CamelCase
  - key: readability-identifier-naming.FunctionCase
    value: CamelCase
  - key: readability-identifier-naming.VariableCase
    value: lower_case
  - key: readability-identifier-naming.ConstantCase
    value: lower_case
  - key: performance-unnecessary-copy-initialization.AllowedTypes
    value: 'wxString'
```

### Tensor-Specific Check Priorities

**High Priority (Must Fix):**
- `bugprone-use-after-move` - Critical for move semantics
- `performance-unnecessary-copy-initialization` - Performance critical
- `performance-move-const-arg` - Incorrect move usage
- `performance-type-promotion-in-math-fn` - Numerical accuracy
- `modernize-use-nullptr` - Modern C++ standard
- `cppcoreguidelines-special-member-functions` - Rule of 5/0

**Medium Priority (Should Fix):**
- `modernize-use-auto` - Template type deduction
- `modernize-use-override` - Virtual function safety
- `readability-redundant-member-init` - Code clarity
- `performance-noexcept-move-constructor` - Exception safety

**Low Priority (Consider):**
- `modernize-use-nodiscard` - API safety
- `readability-const-return-type` - Const correctness
- `cppcoreguidelines-pro-type-reinterpret-cast` - Type safety

### Suppression Strategy

Use `NOLINT` comments sparingly for legitimate cases:
```cpp
// Intentional pointer arithmetic for SIMD alignment
float* aligned_ptr = ptr + offset;  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
```

## Phase 3: VS Code Integration

### Additional Tasks for tasks.json

1. **Run clang-tidy on Tensor files**
   ```json
   {
     "label": "Analyze Tensor Code (clang-tidy)",
     "type": "shell",
     "command": "${workspaceFolder}/.claude/scripts/analyze_tensor.sh",
     "group": "test",
     "problemMatcher": {
       "owner": "clang-tidy",
       "fileLocation": ["relative", "${workspaceFolder}"],
       "pattern": {
         "regexp": "^(.*):(\\d+):(\\d+):\\s+(warning|error):\\s+(.*)\\s+\\[(.*)\\]$",
         "file": 1,
         "line": 2,
         "column": 3,
         "severity": 4,
         "message": 5,
         "code": 6
       }
     }
   }
   ```

2. **Run clang-tidy on changed files only**
   ```json
   {
     "label": "Analyze Changed Files (clang-tidy-diff)",
     "type": "shell",
     "command": "${workspaceFolder}/.claude/scripts/analyze_diff.sh",
     "group": "test",
     "problemMatcher": {
       "owner": "clang-tidy",
       "fileLocation": ["relative", "${workspaceFolder}"],
       "pattern": {
         "regexp": "^(.*):(\\d+):(\\d+):\\s+(warning|error):\\s+(.*)\\s+\\[(.*)\\]$",
         "file": 1,
         "line": 2,
         "column": 3,
         "severity": 4,
         "message": 5,
         "code": 6
       }
     }
   }
   ```

3. **Fix issues automatically (careful!)**
   ```json
   {
     "label": "Apply clang-tidy Fixes (Auto-fix)",
     "type": "shell",
     "command": "${workspaceFolder}/.claude/scripts/analyze_and_fix.sh",
     "group": "test",
     "problemMatcher": []
   }
   ```

## Phase 4: Helper Scripts

Create in `.claude/scripts/` directory:

### 1. `analyze_tensor.sh`
```bash
#!/usr/bin/env bash
# Run clang-tidy on all Tensor source files

set -euo pipefail

TENSOR_DIR="src/core/tensor"
BUILD_DIR="build/clang-tidy-debug"

if [[ ! -f "$BUILD_DIR/compile_commands.json" ]]; then
    echo "Error: Compilation database not found!"
    echo "Run: Tasks: Build with Bear (Generate compile_commands.json)"
    exit 1
fi

echo "Running clang-tidy on Tensor code..."
find "$TENSOR_DIR" -name '*.cpp' -o -name '*.cu' | \
    xargs run-clang-tidy-14 -p "$BUILD_DIR" -quiet
```

### 2. `analyze_diff.sh`
```bash
#!/usr/bin/env bash
# Run clang-tidy only on changed lines in current branch

set -euo pipefail

BUILD_DIR="build/clang-tidy-debug"
BASE_BRANCH="${1:-master}"

if [[ ! -f "$BUILD_DIR/compile_commands.json" ]]; then
    echo "Error: Compilation database not found!"
    exit 1
fi

echo "Analyzing changes compared to $BASE_BRANCH..."
git diff -U0 "$BASE_BRANCH" | \
    clang-tidy-diff-14.py -p1 -path "$BUILD_DIR" -quiet
```

### 3. `analyze_and_fix.sh`
```bash
#!/usr/bin/env bash
# Run clang-tidy with auto-fix (USE WITH CAUTION)

set -euo pipefail

TENSOR_DIR="src/core/tensor"
BUILD_DIR="build/clang-tidy-debug"

echo "WARNING: This will modify source files!"
read -p "Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 0
fi

find "$TENSOR_DIR" -name '*.cpp' -o -name '*.cu' | \
    xargs run-clang-tidy-14 -p "$BUILD_DIR" -fix -quiet

echo "Files modified. Please review changes with 'git diff'"
```

### 4. `generate_compile_db.sh`
```bash
#!/usr/bin/env bash
# Generate compilation database with Bear

set -euo pipefail

BUILD_DIR="${1:-build/clang-tidy-debug}"
CLEAN_BUILD="${2:-false}"

if [[ ! -d "$BUILD_DIR" ]]; then
    echo "Creating build directory: $BUILD_DIR"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    ../../configure --enable-debugmode CC=clang CXX=clang++
fi

cd "$BUILD_DIR"

if [[ "$CLEAN_BUILD" == "true" ]]; then
    echo "Clean build requested..."
    make clean
fi

echo "Building with Bear to generate compilation database..."
bear -- make -j$(nproc)

echo "Compilation database generated: $BUILD_DIR/compile_commands.json"

# Create symlink in project root for clangd and other tools
cd ../..
ln -sf "$BUILD_DIR/compile_commands.json" .
echo "Linked to project root for IDE integration"
```

## Phase 5: Development Workflow

### For Developers Working on Tensor Code

1. **One-time setup** (per development session):
   ```
   Tasks: Build with Bear (Generate compile_commands.json)
   ```

2. **Before committing**:
   ```
   Tasks: Analyze Changed Files (clang-tidy-diff)
   ```
   Review and fix warnings before committing

3. **Deep analysis** (periodic):
   ```
   Tasks: Analyze Tensor Code (clang-tidy)
   ```
   Run full analysis on Tensor subsystem

4. **Auto-fix simple issues** (optional):
   ```
   Tasks: Apply clang-tidy Fixes (Auto-fix)
   ```
   Then review with `git diff` before committing

### Integration with Existing Workflow

- Doesn't interfere with normal ICC/GCC builds
- Clang build is separate configuration
- Can be run locally without affecting CI
- Compilation database updates automatically with each Bear build

## Phase 6: CI Integration (Future)

### GitHub Actions Workflow

Create `.github/workflows/static_analysis.yml`:

```yaml
name: Static Analysis

on:
  pull_request:
    paths:
      - 'src/core/tensor/**'
      - 'src/core/**/*.h'
  push:
    branches: [master, develop]

jobs:
  clang-tidy:
    runs-on: ubuntu-20.04
    container:
      image: your-cistem-dev-image
    steps:
      - uses: actions/checkout@v4

      - name: Configure with Clang
        run: |
          mkdir -p build/clang-analysis
          cd build/clang-analysis
          ../../configure --enable-debugmode CC=clang CXX=clang++

      - name: Generate compilation database
        run: |
          cd build/clang-analysis
          bear -- make -j$(nproc)

      - name: Run clang-tidy on changed files
        run: |
          git fetch origin ${{ github.base_ref }}
          git diff -U0 origin/${{ github.base_ref }} | \
            clang-tidy-diff-14.py -p1 -path build/clang-analysis > analysis.txt

      - name: Post results as PR comment
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
                body: '## clang-tidy Analysis\n```\n' + output + '\n```'
              });
            }
```

### CI Integration Strategy

1. **Phase 6a: Warning only** - Post results as PR comments
2. **Phase 6b: Soft failure** - Fail CI but allow merge with override
3. **Phase 6c: Hard failure** - Block merge on clang-tidy errors

## Phase 7: Other Static Analysis Tools

### scan-build (Clang Static Analyzer)

**When to use:** Deep analysis for potential bugs

```bash
# In build directory
scan-build-14 make -j16

# With HTML report
scan-build-14 -o /tmp/scan-results make -j16
```

**VS Code Task:**
```json
{
  "label": "Deep Analysis (scan-build)",
  "type": "shell",
  "command": "cd build/clang-tidy-debug && scan-build-14 -o ${workspaceFolder}/.claude/cache/scan-results make -j$(nproc)",
  "group": "test"
}
```

### clang-query (AST Pattern Matching)

**When to use:** Custom code pattern searches

Example: Find all raw pointer allocations in Tensor code:
```bash
clang-query-14 -p build/clang-tidy-debug src/core/tensor/*.cpp \
  -c "match cxxNewExpr()"
```

### clang-check (Syntax Validation)

**When to use:** Quick syntax/semantic check without full build

```bash
clang-check-14 -p build/clang-tidy-debug src/core/tensor/type_traits.h
```

## Implementation Roadmap

### Week 1: Foundation ✅ COMPLETE (2025-10-04)
- [x] Install Bear 3.1.6 in devcontainer
- [x] Create VS Code tasks for Bear integration
- [x] Generate initial compilation database (12,389 entries)
- [x] Test clang-tidy on small file (complex_types.h)
- [x] Update directory naming to `clang-tidy-debug` for clarity

### Week 2: Configuration ✅ COMPLETE (2025-10-04)
- [x] Create `.clang-tidy` configuration file with multi-tier system
- [x] Create helper scripts in `scripts/linting/cpp_cuda/`
  - [x] analyze_blocker.sh (Tier 0)
  - [x] analyze_critical.sh (Tier 0+1)
  - [x] analyze_standard.sh (Tier 0-2)
  - [x] analyze_deep.sh (Tier 3)
  - [x] generate_compile_db.sh
- [x] Test on Tensor code subsystem
- [x] Document workflow in CLAUDE.md and scripts/linting/CLAUDE.md

### Week 3: Developer Adoption (IN PROGRESS)
- [ ] Run analysis on existing Tensor code
- [ ] Create baseline of current issues
- [ ] Fix high-priority issues
- [ ] Train team on workflow

### Week 4: Refinement
- [ ] Tune checks based on feedback
- [ ] Add suppression comments where needed
- [ ] Create quick reference guide
- [ ] Set up pre-commit hook (optional)

### Future: CI Integration
- [ ] Add GitHub Actions workflow
- [ ] Integrate with PR reviews
- [ ] Generate trend reports
- [ ] Expand to other subsystems

**Note:** Weeks 1-2 completed ahead of schedule. Static analysis infrastructure is fully operational and ready for use.

## Success Metrics

1. **Code Quality:**
   - Reduce template-related bugs by 30%
   - Zero use-after-move bugs in new code
   - Consistent modern C++ idioms

2. **Developer Experience:**
   - Analysis runs in < 2 minutes for changed files
   - < 5% false positive rate
   - Integrated into normal workflow (not disruptive)

3. **Coverage:**
   - 100% of new Tensor code analyzed before commit
   - 80% of existing Tensor code analyzed and fixed
   - Gradual expansion to other subsystems

## Open Questions / Decisions Needed

1. **Should we use Bear 2.4.3 (apt) or 3.1.6 (source)?**
   - 2.4.3: Simpler, faster installation
   - 3.1.6: Better features, actively maintained
   - Recommendation: Start with 2.4.3, upgrade if needed

2. **Enforcement level for CI:**
   - Warning only (informational)
   - Soft failure (can override)
   - Hard failure (blocks merge)
   - Recommendation: Start with warnings, progress to soft failure

3. **Scope of initial rollout:**
   - Tensor code only
   - All core libraries
   - Entire codebase
   - Recommendation: Tensor first, then expand

4. **Auto-fix policy:**
   - Never auto-fix (manual review only)
   - Auto-fix simple issues (modernize-use-nullptr, etc.)
   - Auto-fix all safe transformations
   - Recommendation: Manual review initially, auto-fix simple cases later

## References

- [clang-tidy documentation](https://clang.llvm.org/extra/clang-tidy/)
- [Bear documentation](https://github.com/rizsotto/Bear)
- [Compilation database format](https://clang.llvm.org/docs/JSONCompilationDatabase.html)
- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/)

## Appendix A: Example clang-tidy Output

```
/path/to/tensor.h:123:5: warning: use '= default' to define a trivial default constructor [modernize-use-equals-default]
    Tensor() {}
    ^~~~~~~~~~~
    = default;

/path/to/tensor.cpp:456:10: warning: moved variable 'buffer' is never used [bugprone-use-after-move]
    auto buffer = std::move(temp_buffer);
         ^

/path/to/memory_pool.cpp:789:20: warning: copying parameter 'config' is unnecessary [performance-unnecessary-copy-initialization]
    void Configure(PoolConfig config) {
                   ^
                   const PoolConfig&
```

## Appendix B: Quick Reference Commands

```bash
# Generate compilation database
cd build/clang-tidy-debug && bear -- make -j16

# Run clang-tidy on specific file
clang-tidy-14 -p build/clang-tidy-debug src/core/tensor/tensor.h

# Run on all Tensor files
run-clang-tidy-14 -p build/clang-tidy-debug src/core/tensor/

# Run on changed files only
git diff -U0 master | clang-tidy-diff-14.py -p1 -path build/clang-tidy-debug

# List available checks
clang-tidy-14 --list-checks

# Show check documentation
clang-tidy-14 --explain-config

# Auto-fix issues (CAREFUL!)
clang-tidy-14 -p build/clang-tidy-debug --fix src/core/tensor/tensor.cpp
```
