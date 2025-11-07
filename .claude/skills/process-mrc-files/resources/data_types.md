# MRC Data Types (MODE Values)

Complete reference for MRC file data types, conversions, and precision considerations.

## MODE Field Overview

The MODE field (header word 4, bytes 12-15) specifies the data type for pixel/voxel values.

## Standard Data Types

| MODE | Type | Bytes/Pixel | Range | Common Use |
|------|------|-------------|-------|------------|
| 0 | int8 (signed byte) | 1 | -128 to 127 | Masks, segmentations, labels |
| 1 | int16 (signed short) | 2 | -32,768 to 32,767 | Raw detector images (older) |
| 2 | float32 (float) | 4 | ±3.4×10³⁸ (7 digits) | **Processed volumes (most common)** |
| 4 | complex64 (complex float) | 8 | Real + Imag (float32 each) | Fourier transforms, rare |
| 6 | uint16 (unsigned short) | 2 | 0 to 65,535 | Raw detector images (modern) |
| 12 | float16 (half float) | 2 | ±65,504 (3 digits) | Compressed storage (MRC2014 extension) |

### MODE 0: Signed 8-bit Integer

**Standard Interpretation:** Signed byte (-128 to 127)

**Historical Issue:** IMOD used opposite sign convention until ~2010s
- Old IMOD files: unsigned interpretation (0-255)
- Detection: IMOD stamp (1146047817) + `imodFlags[0] & 1`

**Best Practice:**
- When **writing**: Use MODE 2 (float32) instead to avoid ambiguity
- When **reading**: Check IMOD compatibility fields if present

**cisTEM:** Handles both signed/unsigned via IMOD flag detection

**Python mrcfile:** Interprets as signed int8 by default

### MODE 1: Signed 16-bit Integer

**Range:** -32,768 to 32,767

**Common Use:** Raw detector images from older cameras

**Conversion:** Automatically converted to float32 in cisTEM; NumPy int16 in Python

**Precision:** No fractional values; quantization may be visible in processed images

### MODE 2: 32-bit Float (RECOMMENDED)

**Range:** ±3.4×10³⁸ with ~7 significant digits

**Common Use:**
- Processed density maps
- Reconstructed volumes
- Normalized/filtered images
- **Default output format for most EM software**

**Advantages:**
- Handles negative values (important for CTF correction, phase info)
- Fractional precision (no quantization artifacts)
- Universal compatibility across all EM software

**Disadvantages:**
- 4× larger than MODE 0
- 2× larger than MODE 1/6

**Recommendation:** Use MODE 2 for all processed data unless file size is critical

### MODE 4: Complex 64-bit (Rare)

**Structure:** 8 bytes = 4-byte real + 4-byte imaginary

**Use Case:** Fourier space representations (uncommon in practice)

**Note:** Most software converts to real space before saving MRC files

### MODE 6: Unsigned 16-bit Integer

**Range:** 0 to 65,535

**Common Use:** Modern direct detector raw images (Falcon, K2/K3, etc.)

**Added:** MRC2014 standard (2015)

**Issue:** Not universally supported in older software

**Conversion:**
- cisTEM: Reads as unsigned, converts to float32
- Python mrcfile: NumPy uint16

**Warning:** Some software may misinterpret as signed int16, causing half the range to appear negative

### MODE 12: 16-bit Half-Precision Float

**Range:** ±65,504 with ~3 significant digits

**Precision:** Much lower than float32 (3 vs 7 digits)

**Added:** MRC2014 proposed extension

**Support:**
- cisTEM: Full support via ieee-754-half library
- Python mrcfile: Limited support (proposed extension status)
- Other tools: Variable support

**Use Case:**
- Storage compression (50% size of MODE 2)
- Acceptable for visualization
- **NOT recommended for quantitative analysis**

**File Size Example:**
- 256×256×100 stack:
  - MODE 2 (float32): 25.6 MB
  - MODE 12 (float16): 12.8 MB (50% reduction)

## Data Type Conversions

### Python (NumPy dtypes)

```python
import mrcfile
import numpy as np

# Explicit type conversion
data_int16 = np.random.randint(-1000, 1000, (100, 256, 256), dtype=np.int16)
data_float32 = data_int16.astype(np.float32)

# Write specific type
mrcfile.write('output.mrc', data_float32)  # MODE 2

# mrcfile automatically selects MODE based on dtype:
# np.int8 → MODE 0
# np.int16 → MODE 1
# np.float32 → MODE 2
# np.complex64 → MODE 4
# np.uint16 → MODE 6
# np.float16 → MODE 12
```

### cisTEM (C++)

```cpp
// cisTEM ALWAYS converts to float32 in memory regardless of file MODE
MRCFile input_file("input.mrc", false);  // Could be any MODE
float* data = new float[input_file.ReturnXSize() * input_file.ReturnYSize()];
input_file.ReadSliceFromDisk(0, data);  // data is float32

// Writing: Default is MODE 2 (float32)
MRCFile output_file("output.mrc", true);
output_file.WriteSliceToDisk(0, data);  // Written as MODE 2

// Optional: Write as FP16 (MODE 12)
output_file.SetOutputToFP16();
output_file.WriteSliceToDisk(0, data);  // Written as MODE 12
```

## Precision and Range Considerations

### Dynamic Range

| Type | Dynamic Range | Quantization Levels |
|------|---------------|---------------------|
| int8 | 256 levels | Visible banding |
| int16 | 65,536 levels | Good for raw data |
| uint16 | 65,536 levels | Modern detectors |
| float16 | ~2000 effective | Acceptable for vis |
| float32 | ~16M effective | No visible quantization |

### Precision Loss Pathways

**Scenario 1: float32 → int16**
```python
# Data range: -0.5 to 1.5
data_float = np.array([-0.5, 0.0, 0.5, 1.0, 1.5], dtype=np.float32)

# Convert to int16 (LOSSY)
data_int16 = data_float.astype(np.int16)
# Result: [0, 0, 0, 1, 1] - fractional information lost
```

**Solution:** Scale before conversion
```python
# Scale to use full int16 range
scale_factor = 10000
data_scaled = (data_float * scale_factor).astype(np.int16)
# Result: [-5000, 0, 5000, 10000, 15000]

# Later: recover with inverse scaling
recovered = data_scaled.astype(np.float32) / scale_factor
```

**Scenario 2: float32 → float16**
```python
# Large values lose precision
data_f32 = np.array([1.23456789], dtype=np.float32)
data_f16 = data_f32.astype(np.float16)
print(data_f16)  # Prints: [1.234] (digits 5-9 lost)
```

**Acceptable when:** Relative error < 0.1% is tolerable (visualization, not quantitative analysis)

### Conversion Recommendations

| Source → Target | Safe? | Notes |
|-----------------|-------|-------|
| int8 → float32 | ✓ Yes | No information loss |
| int16 → float32 | ✓ Yes | No information loss |
| uint16 → float32 | ✓ Yes | No information loss |
| float32 → float16 | ⚠️ Caution | 50% size savings, ~0.1% precision loss |
| float32 → int16 | ❌ No | Requires scaling; fractional loss |
| float32 → int8 | ❌ No | Requires heavy scaling; severe quantization |

**Golden Rule:** When in doubt, use MODE 2 (float32)

## Complex Data (MODE 4)

### Structure

```
Byte 0-3: Real part (float32)
Byte 4-7: Imaginary part (float32)
```

### Use Cases

- Fourier transform output
- Phase information preservation
- Rarely used in practice (most tools save real space only)

### Python Example

```python
import numpy as np
import mrcfile

# Complex data
complex_data = np.zeros((10, 256, 256), dtype=np.complex64)
complex_data.real = np.random.rand(10, 256, 256)
complex_data.imag = np.random.rand(10, 256, 256)

# Write as MODE 4
mrcfile.write('complex.mrc', complex_data)  # Automatically MODE 4

# Read complex data
with mrcfile.open('complex.mrc') as mrc:
    assert mrc.header.mode == 4
    real_part = mrc.data.real
    imag_part = mrc.data.imag
    magnitude = np.abs(mrc.data)
    phase = np.angle(mrc.data)
```

### cisTEM Handling

cisTEM **does not directly support MODE 4**. Convert to real space before saving in cisTEM workflows.

## Byte Order and Endianness

Data byte order is determined by the machine stamp (header word 54), not the MODE field.

- Little-endian (modern): bytes stored LSB first
- Big-endian (legacy): bytes stored MSB first

**Important:** MODE value itself is also affected by endianness when reading headers.

See `interoperability_guide.md` for endianness detection strategies.

## File Size Calculations

**Formula:**
```
File size = 1024 (header) + NSYMBT (extended header) + (NX × NY × NZ × bytes_per_pixel)
```

**Example:** 256×256×100 stack
- MODE 0 (int8): 1024 + 6,553,600 = 6.25 MB
- MODE 1 (int16): 1024 + 13,107,200 = 12.5 MB
- MODE 2 (float32): 1024 + 26,214,400 = 25 MB
- MODE 6 (uint16): 1024 + 13,107,200 = 12.5 MB
- MODE 12 (float16): 1024 + 13,107,200 = 12.5 MB

## Compatibility Matrix

| MODE | mrcfile (Python) | cisTEM (C++) | IMOD | RELION | Chimera |
|------|------------------|--------------|------|--------|---------|
| 0 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 1 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 2 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 4 | ✓ | ❌ | ✓ | ⚠️ | ⚠️ |
| 6 | ✓ | ✓ | ✓ (recent) | ✓ | ✓ (recent) |
| 12 | ⚠️ | ✓ | ⚠️ | ⚠️ | ⚠️ |

Legend:
- ✓ Full support
- ⚠️ Partial/recent support
- ❌ Not supported

## Best Practices

### For Maximum Compatibility

1. **Use MODE 2 (float32)** for all processed data
2. **Avoid MODE 12 (float16)** unless file size is critical and precision loss is acceptable
3. **Avoid MODE 4 (complex)** unless essential; convert to real space first
4. **Use MODE 6 (uint16)** only for raw detector data where appropriate
5. **Never use MODE 0 (int8)** unless creating masks or labels

### For cisTEM Workflows

1. cisTEM reads all modes and converts to float32 internally
2. cisTEM writes MODE 2 (default) or MODE 12 (if explicitly requested)
3. For output: Use MODE 2 unless storage is critical (then MODE 12)

### For Python Workflows

1. Use NumPy dtype to control MODE:
   ```python
   data = data.astype(np.float32)  # Ensures MODE 2
   mrcfile.write('output.mrc', data)
   ```

2. Check mode when reading:
   ```python
   with mrcfile.open('file.mrc') as mrc:
       if mrc.header.mode == 0:
           # Handle potential sign convention issue
           pass
   ```

3. For large datasets, consider memory mapping:
   ```python
   with mrcfile.mmap('huge_stack.mrc', mode='r') as mrc:
       subset = mrc.data[100:200, :, :]  # Only loads needed slices
   ```

## Summary

- **MODE 2 (float32)** is the universal standard for processed EM data
- **MODE 6 (uint16)** is standard for modern raw detector images
- **MODE 12 (float16)** offers 50% compression with minimal precision loss for visualization
- **Avoid MODE 0** (ambiguous sign convention)
- **Avoid MODE 4** (complex) unless absolutely necessary
- **Always check MODE** when reading files from other software

For conversion examples and handling edge cases, see templates in `../templates/`.
