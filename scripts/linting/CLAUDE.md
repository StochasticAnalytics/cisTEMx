# Static Analysis and Linting for cisTEMx

This directory contains configuration, documentation, and tools for static analysis and linting across the cisTEMx codebase.

## Philosophy

**Proactive quality assurance through automated analysis.** Static analysis catches bugs, performance issues, and style inconsistencies before they reach production, with minimal developer friction.

### Core Principles

1. **Multi-tiered approach** - Different analysis levels for different contexts (pre-commit, CI, deep analysis)
2. **Language-appropriate tools** - Use best-in-class tools for each language
3. **Performance-aware** - Fast checks for frequent use, deep analysis for thorough review
4. **Incremental adoption** - Start with critical checks, expand coverage over time
5. **Low false-positive rate** - Tune configurations to minimize noise
6. **CI integration** - Automated analysis on pull requests and commits

## Language-Specific Guides

### C++ and CUDA

**Location:** `scripts/linting/cpp_cuda/`
**Primary Tool:** clang-tidy (LLVM 14+)
**Status:** ✅ Active

Comprehensive static analysis for:

- Memory safety and ownership
- Performance optimization opportunities
- Modern C++ idioms (C++17)
- Template correctness
- Move semantics
- Thread safety

See [`cpp_cuda/CLAUDE.md`](cpp_cuda/CLAUDE.md) for detailed implementation guide.

### Shell Scripts

**Location:** `scripts/linting/shell/`
**Primary Tools:** shellcheck, shfmt
**Status:** 📋 Planned

Shell script analysis for:

- Common scripting errors
- Portability issues
- Security vulnerabilities
- Style consistency

See [`shell/CLAUDE.md`](shell/CLAUDE.md) for details.

### Python

**Location:** `scripts/linting/python/`
**Primary Tools:** pylint, flake8, black, mypy
**Status:** 📋 Planned

Python code analysis for:

- PEP 8 compliance
- Type checking
- Code formatting
- Common bugs and anti-patterns

See [`python/CLAUDE.md`](python/CLAUDE.md) for details.

### Build Systems

**Location:** `scripts/linting/build_systems/`
**Primary Tools:** (TBD - autotools, cmake linting)
**Status:** 📋 Planned

Build system validation for:

- Makefile.am correctness
- CMakeLists.txt best practices
- Dependency tracking
- Configuration consistency

See [`build_systems/CLAUDE.md`](build_systems/CLAUDE.md) for details.

## General Workflow

### For Developers

**Before committing:**

```bash
# Run fast, critical checks only
./scripts/linting/cpp_cuda/analyze_critical.sh

# Or use VS Code task: "clang-tidy: Phase 1 (Fast - Pre-commit)"
```

**Before creating PR:**

```bash
# Run standard tier checks on changed files
./scripts/linting/cpp_cuda/analyze_diff.sh standard

# Or use VS Code task: "Analyze Changed Files (clang-tidy-diff)"
```

**Deep analysis (periodic):**

```bash
# Run comprehensive analysis on subsystem
./scripts/linting/cpp_cuda/analyze_deep.sh

# Or use VS Code task: "clang-tidy: Phase 3 (Deep - Weekly)"
```

### CI Integration

**Pull Request Checks:**

- Phase 2 (Standard tier) on changed files
- Report findings as PR comments
- Warn on issues, block on critical errors

**Nightly Builds:**

- Phase 3 (Deep tier) on entire codebase
- Generate trend reports
- Identify technical debt

## Tool Installation

All linting tools are installed in the cisTEMx development container:

```dockerfile
# LLVM/Clang tools (already installed)
RUN apt-get install -y \
    clang-14 \
    clang-tidy-14 \
    clang-format-14 \
    bear

# Shell script tools (planned)
RUN apt-get install -y \
    shellcheck \
    shfmt

# Python tools (planned)
RUN pip install pylint flake8 black mypy
```

See `scripts/containers/top_image/Dockerfile` for current installation.

## Configuration Files

### Project Root

- `.clang-tidy` - Comprehensive C++/CUDA check configuration
- `.clang-format` - Code formatting rules (already active)
- `.shellcheckrc` - Shell script linting config (planned)
- `.pylintrc` - Python linting config (planned)
- `pyproject.toml` - Python tool configuration (planned)

### Directory-Specific Overrides

Each subsystem can override settings with local `.clang-tidy` files:

```
src/core/tensor/.clang-tidy  # Tensor-specific rules
src/gui/.clang-tidy           # GUI-specific rules
src/programs/.clang-tidy      # Program-specific rules
```

## VS Code Integration

All linting tasks are integrated into VS Code tasks.json:

**Task naming convention:**

- `Configure for Static Analysis` - Setup build environment
- `Build with Bear` - Generate compilation database
- `clang-tidy: <Level>` - Run analysis at specified level
- `Analyze <Target>` - Run on specific code section

Use `Ctrl+Shift+P` → `Tasks: Run Task` to execute.

## Metrics and Reporting

### Success Criteria

**Developer Experience:**

- Pre-commit checks complete in < 30 seconds
- Standard tier checks complete in < 2 minutes
- False positive rate < 5%

**Code Quality:**

- Zero critical issues in new code
- Trending reduction in warnings
- Consistent application of modern C++ idioms

**Coverage:**

- 100% of new code analyzed before merge
- 80% of existing code analyzed within 6 months
- Expand to all languages within 12 months

## Adding New Checks

### For Existing Tools

1. Propose check addition with rationale
2. Test on representative code sample
3. Measure false positive rate
4. Add to appropriate tier
5. Update documentation
6. Announce to team

### For New Tools or Languages

1. Research tool options and create proposal
2. Document in appropriate `<language>/CLAUDE.md`
3. Create proof-of-concept configuration
4. Add container installation steps
5. Create VS Code tasks
6. Write helper scripts
7. Update this overview document

## Custom Check Development

For cisTEMx-specific patterns not covered by standard tools:

**High-value custom check candidates:**

- CUDA memory transfer patterns (H2D/D2H correctness)
- FFTW plan lifecycle management
- Memory pool acquire/release pairing
- Image boundary condition handling
- Thread-safe MKL usage

**Process:**

1. Document pattern in `cpp_cuda/custom_checks.md`
2. Prototype using clang-query
3. Implement as clang-tidy check module
4. Test on codebase
5. Add to project configuration

See `cpp_cuda/CLAUDE.md` for custom check development guide.

## Troubleshooting

### Compilation Database Issues

```bash
# Regenerate database
cd build/clang-tidy-debug
make clean
bear -- make -j$(nproc)
ln -sf $(pwd)/compile_commands.json ../../
```

### Tool Version Mismatches

All tools should use matching LLVM version:

```bash
clang-tidy-14 --version
clang-format-14 --version
bear --version  # Should be 3.0+
```

### High False Positive Rate

1. Review check documentation: `clang-tidy-14 -checks='*' -list-checks`
2. Add suppressions with `// NOLINT(check-name)` sparingly
3. Consider moving check to lower tier
4. Document rationale in tier comments

### Performance Issues

- Use `-j$(nproc)` with `run-clang-tidy` for parallelization
- Analyze subsystems separately
- Use `-quiet` flag to reduce output overhead
- Consider check filtering for faster feedback

## References

- [cisTEMx Build System](../CLAUDE.md)
- [C++/CUDA Linting Guide](cpp_cuda/CLAUDE.md)
- [clang-tidy Documentation](https://clang.llvm.org/extra/clang-tidy/)
- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/)

## Questions or Issues?

For questions about linting integration:

1. Check the language-specific CLAUDE.md
2. Review tool documentation
3. Ask in team discussions
4. File issue in project tracker
