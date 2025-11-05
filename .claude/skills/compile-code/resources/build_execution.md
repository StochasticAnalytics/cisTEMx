# Build Execution

## Overview

Execute cisTEMx builds using the automated build script that handles configuration extraction, parallelism, and logging.

## Quick Start

```bash
# Build with DEBUG configuration (default)
python .claude/skills/compile-code/scripts/run_build.py

# Build with RELEASE configuration
python .claude/skills/compile-code/scripts/run_build.py --config RELEASE

# Override automatic core detection
python .claude/skills/compile-code/scripts/run_build.py --cores 8
```

## How It Works

The build script automates the complete build process:

1. **Finds git project root** - Locates the repository base
2. **Extracts build configuration** - Reads `.vscode/tasks.json` to find build directory for the specified config
3. **Determines parallelism** - Auto-detects CPU cores (capped at 16) or uses your override
4. **Executes make** - Runs `make -j<cores>` in the correct build directory
5. **Logs everything** - Streams output to console AND timestamped log file in `.claude/cache/build_YYYYMMDD_HHMMSS.log`

## Build Configurations

Available configurations are defined in `.vscode/tasks.json`:
- **DEBUG** - Debug symbols, no optimization
- **RELEASE** - Optimized for production
- Additional configs as defined in your tasks.json

## Understanding Output

### Success
```
Build Configuration:
  Build Directory: /path/to/build/debug-build-dir
  Parallel Jobs: 16
  Log File: /path/to/.claude/cache/build_20251103_083045.log

Starting compilation...

[compiler output...]

✓ BUILD SUCCESS
  Log: /path/to/.claude/cache/build_20251103_083045.log
```

### Failure
```
✗ BUILD FAILED
  Log: /path/to/.claude/cache/build_20251103_083045.log

Run analyze_build_log.py to generate error summary:
  python .claude/skills/compile-code/scripts/analyze_build_log.py /path/to/log
```

## Prerequisites

- Git repository must be configured
- `.vscode/tasks.json` must contain build configuration
- Build directory must already exist (run corresponding Configure task if needed)

## Common Issues

**"Build directory does not exist"**
- Run the corresponding Configure task first (e.g., "Configure cisTEMx DEBUG")

**"Could not find build config 'XYZ'"**
- Check `.vscode/tasks.json` for available configurations
- Ensure the config name matches exactly (case-sensitive)

**Build succeeds but with warnings**
- Warnings are not captured in failure analysis
- Review the full log file if needed

## Next Steps

When build fails, use `analyze_build_log.py` to extract actionable errors - see `error_diagnosis_basics.md`.
