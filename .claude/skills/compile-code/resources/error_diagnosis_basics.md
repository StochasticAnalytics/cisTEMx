# Error Diagnosis Basics

## Overview

How to read compiler output, locate errors, and extract actionable information from build logs.

## Using the Analysis Script

```bash
# Analyze a build log
python .claude/skills/compile-code/scripts/analyze_build_log.py /path/to/build_log.log

# Save analysis to file
python .claude/skills/compile-code/scripts/analyze_build_log.py /path/to/build_log.log --output summary.txt
```

## What the Script Finds

The analyzer searches for:
- **Compilation errors**: Lines containing "error" (case-sensitive)
- **Linking errors**: Lines containing "undefined reference" (case-insensitive)

It produces:
- Error counts by category
- First 20 errors of each type with line numbers
- Indication if more errors exist

## Reading Compiler Errors

### Standard Format
```
file.cpp:line:column: error: description
  context line from source
  ^~~~~~ (diagnostic marker)
```

**Key components:**
- `file.cpp` - Source file with the error
- `line` - Line number (for navigation)
- `column` - Character position
- `description` - What went wrong

### Example
```
src/core/Image.cpp:145:12: error: no matching function for call to 'CalculateFFT'
    result = CalculateFFT(input, size);
             ^~~~~~~~~~~~
```

**Action**: Navigate to `src/core/Image.cpp:145` and check the function call.

## Error Categories

### Compilation Errors
Happen during the compilation phase (source → object files):
- Syntax errors
- Type mismatches
- Missing declarations
- Invalid use of language features

### Linking Errors
Happen during the linking phase (object files → executable):
- Undefined references (missing function/variable definitions)
- Multiple definitions
- Missing libraries

See `linker_errors_reference.md` for detailed linker error guidance.

## Filtering Noise

Compilers often produce cascading errors where one root cause triggers many downstream errors.

**Strategy:**
1. Fix the FIRST error in each file
2. Rebuild
3. Many subsequent errors often disappear

**Example cascade:**
```
Image.h:50: error: 'ComplexType' does not name a type
Image.cpp:145: error: 'ComplexType' was not declared in this scope
Image.cpp:167: error: cannot convert 'int*' to 'ComplexType*'
...
```
Fix the first error (likely a missing include or typedef), and the rest may resolve.

## Viewing Log Context

To see surrounding lines from a log file:

```bash
# View lines 140-150 from build log
sed -n '140,150p' /path/to/build_log.log

# Or use Read tool with offset/limit parameters
```

## When Analysis Finds Nothing

If the analyzer reports no errors but the build failed:
- Build system configuration issue (check for "configure" errors)
- Missing dependencies
- Warnings promoted to errors (-Werror flag)
- Review the full log file manually

## Next Steps

- For linking errors specifically, see `linker_errors_reference.md`
- For build system issues, check the full log for "configure" or "autotools" messages
