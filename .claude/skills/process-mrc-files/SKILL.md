---
name: process-mrc-files
description: |
  Process MRC (Medical Research Council) files for electron microscopy and structural biology. Use when: reading/writing MRC files in Python (mrcfile module) or C++ (cisTEM), understanding file structure and header fields, handling 2D images/3D volumes/stacks, implementing partial I/O, debugging interoperability issues between tools, or ensuring proper coordinate conventions and data type handling. Covers MRC2014 specification, Python mrcfile API, cisTEM C++ implementation, and cross-platform compatibility pitfalls.
---

# Process MRC Files

Comprehensive guide for working with MRC (Medical Research Council) files, the standard format for storing 2D/3D image data in electron microscopy and structural biology.

## What is MRC Format?

The MRC format stores image and volume data with a 1024-byte header followed by binary data. The current standard is MRC2014, maintained by CCP-EM (Collaborative Computational Project for Electron cryo-Microscopy).

**Key characteristics:**
- Fixed 1024-byte header with 56 fields + 10 text labels
- Multiple data types: int8, int16, float32, uint16, float16, complex
- Coordinate convention: X=fastest (columns), Y=medium (rows), Z=slowest (sections)
- Supports 2D images, 3D volumes, and stacks
- Optional extended headers for vendor-specific metadata

## When to Use This Skill

Load this skill when you need to:
- **Read or write MRC files** in Python or C++
- **Understand MRC file structure** (header layout, data organization)
- **Handle different data types** (MODE values 0-12, conversions)
- **Work with coordinate conventions** (axis ordering, origin, pixel size)
- **Implement partial I/O** (reading/writing specific slices from stacks)
- **Debug interoperability issues** (format compatibility between tools)
- **Validate MRC files** or convert between formats

## Quick Start

### Python (mrcfile module)

```python
import mrcfile
import numpy as np

# Read a file
with mrcfile.open('file.mrc') as mrc:
    data = mrc.data  # NumPy array
    pixel_size = mrc.voxel_size.x

    # Read specific slices
    stack_subset = mrc.data[10:20, :, :]

# Write a file
data = np.random.random((100, 256, 256)).astype(np.float32)
mrcfile.write('output.mrc', data, voxel_size=1.5)
```

### C++ (cisTEM)

```cpp
#include "core/core_headers.h"

// Read slices from file
MRCFile input_file("input.mrc", false);
int nz = input_file.ReturnZSize();
float* data = new float[input_file.ReturnXSize() * input_file.ReturnYSize()];

// Read slices 0-9
input_file.ReadSlicesFromDisk(0, 9, data);

// Write slices to file
MRCFile output_file("output.mrc", true);
output_file.SetPixelSize(1.5);
output_file.WriteSlicesToDisk(0, 9, data);
```

## Available Resources

### Core Documentation

- **`resources/format_specification.md`** - MRC2014 header structure, field definitions, data types, official specification
- **`resources/python_mrcfile_guide.md`** - Complete Python mrcfile API, reading/writing patterns, examples
- **`resources/cistem_cpp_guide.md`** - cisTEM MRCHeader and MRCFile classes, methods, usage patterns
- **`resources/interoperability_guide.md`** - Cross-platform pitfalls, compatibility issues, best practices
- **`resources/coordinate_conventions.md`** - Axis ordering (X/Y/Z), origin field, pixel size, MAPC/MAPR/MAPS
- **`resources/data_types.md`** - MODE values (0-12), byte sizes, conversion rules, precision considerations

### Code Examples

- **`templates/python_read_example.py`** - Python patterns for reading 2D/3D/stacks, partial I/O
- **`templates/python_write_example.py`** - Python patterns for creating files, modifying headers
- **`templates/cpp_read_example.cpp`** - C++ patterns for reading slices, handling formats
- **`templates/cpp_write_example.cpp`** - C++ patterns for writing data, setting metadata

### Citations and References

- **`resources/citations.md`** - MRC2014 paper, official specifications, library documentation, version tracking

## Progressive Disclosure Pattern

This skill follows progressive disclosure:

1. **Start here (SKILL.md)** - Overview and quick start
2. **Choose your language**:
   - Python → `python_mrcfile_guide.md`
   - C++ → `cistem_cpp_guide.md`
3. **Dive deeper as needed**:
   - Format details → `format_specification.md`
   - Coordinate systems → `coordinate_conventions.md`
   - Data types → `data_types.md`
   - Compatibility → `interoperability_guide.md`
4. **Use templates** - Copy/adapt code examples
5. **Check citations** - Verify you're using current versions

## Common Tasks

### Task: Read a 2D image slice from a stack

**Python:**
```python
with mrcfile.open('stack.mrc') as mrc:
    slice_10 = mrc.data[10, :, :]  # Zero-indexed
```

**C++:**
```cpp
MRCFile file("stack.mrc", false);
float* slice = new float[file.ReturnXSize() * file.ReturnYSize()];
file.ReadSliceFromDisk(10, slice);  # Zero-indexed
```

**Details:** See `python_mrcfile_guide.md` or `cistem_cpp_guide.md`

### Task: Distinguish 2D stack from 3D volume

**Issue:** MRC format doesn't inherently distinguish these.

**RELION convention:**
- 3D volumes: `.mrc` extension, `ISPG ≥ 1`
- 2D stacks: `.mrcs` extension, `ISPG = 0`

**Details:** See `interoperability_guide.md` → "Stack vs Volume Ambiguity"

### Task: Handle endianness

**Detection:** Check machine stamp field (header word 54)
- Little-endian: `[68, 65]` or `[68, 68]`
- Big-endian: `[17, 17]`

**Reality:** Machine stamp often unreliable; use heuristics (validate MODE field range)

**Details:** See `interoperability_guide.md` → "Endianness Detection"

### Task: Convert data types

**Python:** Set dtype when writing
```python
data_float32 = data.astype(np.float32)
mrcfile.write('output.mrc', data_float32)  # MODE 2
```

**C++:** cisTEM converts all formats to float32 in memory

**Details:** See `data_types.md` for precision/range considerations

## Key Concepts

### 1. Coordinate Convention

Standard EM convention:
- **X** (MAPC=1): columns, fastest-changing in memory
- **Y** (MAPR=2): rows, medium-changing
- **Z** (MAPS=3): sections/slices, slowest-changing

Array indexing: `data[z][y][x]` (C-order, row-major)

### 2. Header Fields

56 4-byte words (224 bytes) + 10 × 80-byte labels (800 bytes) = 1024 bytes total

Critical fields:
- **NX, NY, NZ** (words 1-3): dimensions
- **MODE** (word 4): data type (0-12)
- **MAPC, MAPR, MAPS** (words 17-19): axis mapping
- **XORG, YORG, ZORG** (words 50-52): origin in Ångströms

### 3. Data Types (MODE)

| Mode | Type | Bytes | Common Use |
|------|------|-------|------------|
| 0 | int8 | 1 | Masks, labels |
| 1 | int16 | 2 | Raw detector images |
| 2 | float32 | 4 | Processed volumes (most common) |
| 6 | uint16 | 2 | Unsigned detector data |
| 12 | float16 | 2 | Compressed storage |

### 4. Partial I/O

Both Python and C++ support reading/writing specific slices without loading entire file:

**Python:** Use slicing on memory-mapped data
**C++:** `ReadSlicesFromDisk(start, end, buffer)`

Essential for large datasets (e.g., 1000+ image stacks)

## Interoperability Warnings

Common pitfalls when exchanging files between tools:

1. **Extended headers**: No standard format across software (SERI, FEI1, FEI2, AGAR)
2. **Stack vs volume ambiguity**: Use `.mrcs` for stacks, `.mrc` for volumes
3. **Origin sign convention**: IMOD vs Chimera use opposite conventions
4. **Mode 0/6 sign**: Historical inconsistencies in signed/unsigned interpretation
5. **Endianness**: Machine stamp field often unreliable

**Recommendation:** Write MODE 2 (float32) files with standard axis ordering (MAPC=1, MAPR=2, MAPS=3) for maximum compatibility.

See `interoperability_guide.md` for detailed analysis and solutions.

## Validation

**Python:**
```bash
# Command-line tool
mrcfile-validate file.mrc

# Programmatic
import mrcfile
mrcfile.validate('file.mrc')
```

**C++:**
```cpp
// cisTEM validates during read
MRCFile file("file.mrc", false);  # Assertions check header validity
file.PrintInfo();  # Print header details for inspection
```

## References

All citations, paper DOIs, library versions, and documentation URLs are tracked in `resources/citations.md` for maintainability and version checking.

**Key sources:**
- MRC2014 specification (Cheng et al., J Struct Biol 2015)
- CCP-EM official format page
- mrcfile Python library documentation
- cisTEM source code

## Next Steps

1. **Choose your implementation language** → Python or C++ guide
2. **Understand the format** → Format specification and coordinate conventions
3. **Review examples** → Copy and adapt templates for your use case
4. **Check compatibility** → Read interoperability guide before exchanging files with other tools

For questions about format evolution or tool compatibility, consult `citations.md` to verify you're referencing current specifications.
