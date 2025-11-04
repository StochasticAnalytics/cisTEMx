---
name: compile-code
description: Build/compile the cisTEMx project using cmake, make, ninja. Provides build instructions, configuration, and compilation error diagnosis. Use when you need to build the project, compile code changes, or check for build errors.
---

# Compile Code

Systematic workflow for building cisTEMx C++ code with automated error analysis and actionable diagnostics.

## When to Use This Skill

- Building after modifying source files
- Investigating compilation failures
- Diagnosing linker errors (undefined references)
- Verifying code changes compile successfully
- Analyzing verbose compiler output

## Prerequisites

- Git repository configured
- Build directory already exists (run Configure task if needed)
- `.vscode/tasks.json` contains build configuration

## Quick Start

```bash
# Execute build with automatic parallelism and logging
python .claude/skills/compile-code/scripts/run_build.py

# If build fails, analyze the log
python .claude/skills/compile-code/scripts/analyze_build_log.py /path/to/build_log.log
```

## Workflow

### 1. Execute Build
Run the build script which handles configuration extraction, parallelism, and logging.

**Details:** See `resources/build_execution.md`

### 2. Check Build Status
- **Success**: Build completes, log saved to `.claude/cache/`
- **Failure**: Error summary suggests running analysis script

### 3. Analyze Errors (on failure)
Use the log analyzer to extract compilation and linking errors with line numbers.

**Details:** See `resources/error_diagnosis_basics.md`

### 4. Diagnose Root Cause
Navigate to error locations and determine fix:
- **Compilation errors**: Syntax, types, missing declarations
- **Linking errors**: Missing libraries, library order, undefined references

**For linking errors specifically:** See `resources/linker_errors_reference.md`

### 5. Fix and Iterate
Apply fix, rebuild, verify.

## Available Resources

### Core Workflow
- `resources/build_execution.md` - How to run builds, configuration options, parallelism
- `resources/error_diagnosis_basics.md` - Reading compiler output, using analysis scripts, error categories

### Specialized References
- `resources/linker_errors_reference.md` - Common undefined reference patterns, autotools gotchas, debugging techniques

### Automation Scripts
- `scripts/run_build.py` - Execute builds with logging
- `scripts/analyze_build_log.py` - Parse logs for errors

### Maintenance
- `resources/citations.md` - Sources and future maintenance notes

## Common Workflows

### Building After Code Changes
1. Run `run_build.py` with default DEBUG config
2. If success, continue work
3. If failure, run `analyze_build_log.py` on the generated log
4. Navigate to first error location
5. Fix and rebuild

### Investigating Undefined Reference
1. Run build, capture undefined reference error
2. Consult `linker_errors_reference.md`
3. Check decision tree to categorize error
4. Apply appropriate fix (LDADD, library order, missing source)
5. Rebuild and verify

### Building Different Configurations
1. Run `run_build.py --config RELEASE` (or other config name from tasks.json)
2. Same analysis process if failures occur

## Validation

After successful build:
- [ ] Build log shows "✓ BUILD SUCCESS"
- [ ] No compilation or linking errors reported
- [ ] Executable/libraries generated in build directory

## Tips

- **Fix errors top-to-bottom**: First error in each file often causes cascading errors
- **Check library order**: In autotools, libraries must come AFTER objects that use them
- **Save logs**: Timestamped logs in `.claude/cache/` help track patterns over time
- **Consult specific references**: Don't load linker reference unless you have linker errors
