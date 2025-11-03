# Linker Errors Reference

## Overview

Common linker error patterns in GNU autotools + clang/gcc environments, with diagnostic strategies.

## Understanding Undefined Reference Errors

### Anatomy of an Undefined Reference
```
src/core/Image.o: In function `Image::CalculateFFT()':
Image.cpp:(.text+0x1a3): undefined reference to `fftw_plan_dft_2d'
collect2: error: ld returned 1 exit status
```

**Key components:**
- `Image.o` - Object file that USES the symbol
- `Image::CalculateFFT()` - Function context where it's called
- `fftw_plan_dft_2d` - The MISSING symbol
- `.text+0x1a3` - Location in the object file (less useful)

## Common Root Causes

### 1. Missing Library on Link Line

**Symptom:** Undefined reference to a function from an external library

**Example:**
```
undefined reference to `wxWindow::Create(...)'
```

**Diagnosis:**
- The function exists in a library, but that library isn't being linked
- For wxWidgets: need `-lwx_...` flags
- For FFTW: need `-lfftw3` or `-lfftw3f`

**Solution:**
- Add missing library to `LDADD` in `Makefile.am` (NOT LDFLAGS)
- Check `pkg-config` output for correct flags

### 2. Library Order Issues

**Symptom:** Undefined reference even though library is specified

**Root cause:** GNU ld is a single-pass linker - order matters!

**Example:**
```makefile
# WRONG - library before object file that uses it
LDADD = -lfftw3 mycode.o

# RIGHT - object files first, then libraries they depend on
LDADD = mycode.o -lfftw3
```

**Rule:** Libraries should come AFTER the object files that use them.

### 3. Missing Source File

**Symptom:** Undefined reference to a function you wrote

**Example:**
```
undefined reference to `MyClass::HelperFunction()'
```

**Diagnosis:**
- The `.cpp` file containing `HelperFunction()` wasn't compiled
- The `.o` file doesn't exist

**Solution:**
- Check `_SOURCES` variable in `Makefile.am`
- Ensure the source file is listed
- Rebuild to generate the object file

### 4. Standard Library Mismatch (Clang-Specific)

**Symptom:** Undefined references to standard library symbols, especially with weird namespaces like `std::__1::`

**Root cause:** Mixing libstdc++ (GNU) and libc++ (LLVM/Clang)

**Example:**
```
undefined reference to `std::__1::basic_string<...>'
```

**Diagnosis:**
- You're using `-stdlib=libc++` but not linking with `-lc++`
- Or mixing object files compiled with different standard libraries

**Solution:**
- Use consistent `-stdlib` flag across all compilation units
- Add `-lc++` if using `-stdlib=libc++`
- In autotools: ensure `CXX` and `CXXFLAGS` are consistent

### 5. Inline/Header-Only Functions

**Symptom:** Undefined reference to a function declared in a header

**Root cause:** Function was declared but not defined, or marked inline but address was taken

**Example:**
```cpp
// header.h
class MyClass {
    void DoSomething();  // Declaration only, no definition
};
```

**Solution:**
- Provide definition in a `.cpp` file
- Or make it inline with body in header

## Debugging Techniques

### 1. Verify Symbol Exists
```bash
# Search all object files for a symbol
nm build/debug-build-dir/src/core/*.o | grep SymbolName

# Check if symbol is defined in a library
nm -D /usr/lib/libfftw3.so | grep fftw_plan_dft
```

### 2. Check Link Command
Add verbose linker output to see what's actually being linked:
```bash
# In Makefile or command line
LDFLAGS += -Wl,--verbose
# or
LDFLAGS += -Wl,-v
```

### 3. Verify Library Search Path
```bash
# Check where linker searches for libraries
ld --verbose | grep SEARCH_DIR
```

### 4. Examine Autotools Variables
```bash
# Check what autotools is actually generating
grep LDADD Makefile
grep LIBADD Makefile
```

## Autotools-Specific Gotchas

### LDFLAGS vs LDADD vs LIBADD

**LDFLAGS**: Linker flags like `-L/path/to/libs` (directories and options)
**LDADD**: Libraries and objects to link into programs (`-lfoo`)
**LIBADD**: Libraries to link into libtool libraries (`.la` files)

**Wrong:**
```makefile
myprogram_LDFLAGS = -lfftw3
```

**Right:**
```makefile
myprogram_LDADD = -lfftw3
```

### The --as-needed Problem

Modern linkers use `--as-needed` by default, which drops libraries that appear unused.

**Symptom:** Library is on link line but still get undefined reference

**Workaround:**
```makefile
LDFLAGS += -Wl,--no-as-needed
```

## Quick Decision Tree

```
Undefined reference to symbol X
│
├─ Is X from external library (fftw, wx, cuda)?
│  └─ YES → Add -lLIBRARY to LDADD (check pkg-config)
│
├─ Is X in your own code?
│  ├─ Does the .cpp file exist with definition?
│  │  ├─ YES → Check _SOURCES in Makefile.am
│  │  └─ NO → Write the function definition
│  │
│  └─ Is it in a separate library you're building?
│     └─ YES → Link against that library
│
└─ Strange namespace (std::__1::)?
   └─ Check for stdlib mixing (libstdc++ vs libc++)
```

## When to Escalate

If you've checked:
- ✓ Library is in LDADD
- ✓ Library order is correct
- ✓ Source file is in _SOURCES
- ✓ Symbol exists in the library (verified with nm)

And still getting undefined reference, likely:
- Configure script issue (missing dependency detection)
- Symbol versioning problem
- ABI incompatibility

Time to review the full build log and configure output.
