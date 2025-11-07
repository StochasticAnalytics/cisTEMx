# MRC Interoperability Guide

**Critical Reference:** When implementing MRC readers/writers, assume NO two tools implement the format identically. Defensive programming and validation are essential.

## Executive Summary

The MRC format has accumulated significant interoperability challenges over 30+ years due to vendor-specific variants, inconsistent implementations, and ambiguities in the original specification. The MRC2014 standard addresses many issues while maintaining backward compatibility, but legacy files and non-compliant software remain common.

**Key principle:** Write conservative (standard-compliant), read liberal (defensive).

## Critical Issues and Solutions

### 1. Extended Header Chaos

#### The Problem

No agreement on extended header type, format, or content across software packages.

**Consequences:**
- Metadata may be lost when reading with different tools
- Different software uses incompatible structures (FEI EPU: 128-byte records, SERI: SerialEM format, AGAR: Agard format)
- Cannot assume data starts at byte 1024

####

 Extended Header Types

| EXTTYP | Name | Software | Structure |
|--------|------|----------|-----------|
| SERI | SerialEM | SerialEM | Tilt angles, montage coordinates |
| FEI1 | FEI Extended | FEI Titan/Polara | Microscope parameters (old format) |
| FEI2 | FEI Extended v2 | FEI Krios | 768-byte structured records |
| AGAR | Agard format | UCSF software | Generic metadata |
| (none) | Unknown/legacy | Various | Unstructured or proprietary |

#### Solution

**When writing:**
```python
# Python: Set EXTTYPE if using extended headers
with mrcfile.new('output.mrc', overwrite=True) as mrc:
    mrc.set_extended_header(extended_data)
    # mrcfile handles NSYMBT automatically
```

**When reading:**
```python
# Python: Check for extended headers
with mrcfile.open('input.mrc') as mrc:
    nsymbt = mrc.header.nsymbt  # Extended header size in bytes
    exttype = mrc.header.exttyp  # Extended header type (4 chars)

    # Data starts at: 1024 + nsymbt
    data_offset = 1024 + nsymbt

    if nsymbt > 0:
        if exttype in (b'SERI', b'FEI1', b'FEI2', b'AGAR'):
            # Known format: can parse
            extended = mrc.extended_header
        else:
            # Unknown format: skip gracefully
            print(f"Warning: Unknown extended header type: {exttype}")
```

```cpp
// C++: cisTEM handles NSYMBT automatically
int data_offset = 1024 + my_header.SymmetryDataBytes();
```

**Best practice:** If not using extended headers, set NSYMBT=0 and don't include EXTTYPE.

### 2. Stack vs. Volume Ambiguity

#### The Problem

MRC format cannot inherently distinguish:
- True 3D volumes (continuous 3D structure)
- Stacks of 2D particle images (separate 2D images)

**Why it matters:**
- Processing pipelines treat them differently
- Particle stacks: each Z-slice is independent image
- 3D volumes: Z-slices are continuous structure

#### ISPG Field (Unreliable)

| ISPG Value | Official Meaning | Reality |
|------------|------------------|---------|
| 0 | Image/stack | Many 3D volumes incorrectly use this |
| 1 | 3D volume | More reliable indicator |
| 401-630 | Volume stack | Rarely used |

**Problem:** Many files have ISPG=0 even for 3D volumes.

#### RELION Convention (More Reliable)

**De facto standard in EM community:**
- **3D volumes/maps:** `.mrc` extension, ISPG ≥ 1
- **Particle stacks:** `.mrcs` extension, ISPG = 0
- **Particle reference:** `1@mystack.mrcs`, `2@mystack.mrcs`, etc.

#### Solution

**When writing:**
```python
# Python: Set ISPG correctly
with mrcfile.new('volume.mrc', overwrite=True) as mrc:
    mrc.set_data(volume_data)
    mrc.set_volume()  # Sets ISPG=1, uses .mrc extension

with mrcfile.new('stack.mrcs', overwrite=True) as mrc:
    mrc.set_data(stack_data)
    mrc.set_image_stack()  # Sets ISPG=0, uses .mrcs extension
```

**When reading:**
```python
# Python: Check both ISPG and extension
import os

with mrcfile.open('file.mrc') as mrc:
    ispg = mrc.header.ispg
    ext = os.path.splitext('file.mrc')[1]

    if ext == '.mrcs' or ispg == 0:
        # Treat as particle stack
        n_particles = mrc.data.shape[0]
    elif ext == '.mrc' or ispg >= 1:
        # Treat as 3D volume
        pass
```

**Best practice:**
- **Always use correct extension**: `.mrcs` for stacks, `.mrc` for volumes
- **Set ISPG correctly**: 0 for stacks, 1 for volumes
- **When reading:** Trust extension > ISPG field

### 3. Endianness and Machine Stamp

#### The Problem

MACHST field (header word 54) is **unreliable** in real-world files.

**Expected values:**
- Little-endian: `[68, 65]` or `[68, 68]` (0x44 0x41 or 0x44 0x44)
- Big-endian: `[17, 17]` (0x11 0x11)

**Reality:**
- Rarely properly set in files from last decade
- Often ignored by software
- Cannot trust for endianness detection

#### Heuristic Detection Strategy

**Method 1: Validate MODE field**
```python
import struct

def detect_endianness(file_path):
    with open(file_path, 'rb') as f:
        # Read MODE field (bytes 12-15)
        f.seek(12)
        mode_bytes = f.read(4)

        # Try little-endian
        mode_le = struct.unpack('<i', mode_bytes)[0]
        # Try big-endian
        mode_be = struct.unpack('>i', mode_bytes)[0]

        # Valid MODE values: 0, 1, 2, 4, 6, 12
        valid_modes = [0, 1, 2, 4, 6, 12]

        if mode_le in valid_modes and mode_be not in valid_modes:
            return 'little'
        elif mode_be in valid_modes and mode_le not in valid_modes:
            return 'big'
        else:
            # Ambiguous: default to little-endian (x86/x64 standard)
            return 'little'
```

**Method 2: Validate multiple fields**
```python
def detect_endianness_robust(file_path):
    with open(file_path, 'rb') as f:
        # Read NX, NY, NZ, MODE
        header_bytes = f.read(16)

        nx_le, ny_le, nz_le, mode_le = struct.unpack('<iiii', header_bytes)
        nx_be, ny_be, nz_be, mode_be = struct.unpack('>iiii', header_bytes)

        # Validate dimensions (should be < 100,000 typically)
        # and MODE (0, 1, 2, 4, 6, or 12)
        le_valid = (0 < nx_le < 100000 and 0 < ny_le < 100000 and
                    0 <= nz_le < 100000 and mode_le in [0, 1, 2, 4, 6, 12])
        be_valid = (0 < nx_be < 100000 and 0 < ny_be < 100000 and
                    0 <= nz_be < 100000 and mode_be in [0, 1, 2, 4, 6, 12])

        if le_valid and not be_valid:
            return 'little'
        elif be_valid and not le_valid:
            return 'big'
        else:
            # Default to little-endian
            return 'little'
```

**cisTEM approach:** Assumes little-endian on x86/x64 systems (standard).

**mrcfile approach:** Checks MACHST, validates header fields, defaults to little-endian.

#### Solution

**Best practice:**
- **When writing:** Always set MACHST correctly for your platform
- **When reading:** Use heuristic validation, don't trust MACHST alone
- **Default assumption:** Little-endian (x86/x64 dominance)

### 4. Data Type Mode Inconsistencies

#### MODE 0 Sign Convention Issue

**Historical problem:** IMOD used opposite signed/unsigned convention from everyone else.

**Timeline:**
- **Pre-2010s IMOD:** MODE 0 = unsigned byte (0-255)
- **MRC2014 + modern IMOD:** MODE 0 = signed byte (-128 to 127)
- **Other software:** Always interpreted as signed

**Detection:** IMOD files include compatibility fields
- `imodStamp` = 1146047817
- `imodFlags[0] & 1` indicates sign convention

**cisTEM handling:**
```cpp
// Source: mrc_header.cpp
// Checks for IMOD stamp and flags to determine signedness
if (imodStamp == 1146047817 && (imodFlags[0] & 1)) {
    // Old IMOD: treat as unsigned
} else {
    // Standard: treat as signed
}
```

**Solution:**
- **Avoid MODE 0 when writing** unless creating masks/labels
- **When reading MODE 0:** Check for IMOD stamp if available
- **Prefer MODE 2 (float32)** for general use

#### MODE 6 Compatibility Issue

**Problem:** MODE 6 (uint16) added in MRC2014 (2015) - not universally supported in older software.

**Risk:** Older tools may misinterpret as MODE 1 (signed int16), causing half the range to appear negative.

**Solution:**
- **When writing for broad compatibility:** Use MODE 2 (float32) instead
- **When writing modern raw data:** MODE 6 is appropriate, document software requirements
- **When reading:** Validate software version supports MODE 6

### 5. Origin Sign Convention

#### The Problem

Different software uses opposite sign conventions for ORIGIN field (words 50-52).

#### Standard EM / Chimera Convention

**Definition:** ORIGIN = position of **outside corner** of first voxel relative to coordinate origin

**Example:**
```
XORG=100.0, YORG=50.0, ZORG=0.0
→ Volume starts at physical position (100, 50, 0) Å
```

#### IMOD Convention

**Definition:** Uses opposite sign for cropped images

**Detection:** Bit 2 of header word 40 indicates convention

**Example:**
```
IMOD file: XORG=-100.0 (bit 2 set)
Equivalent standard: XORG=+100.0
```

#### Solution

**When exchanging between IMOD and other tools:**
```python
# Python: Check IMOD convention flag
with mrcfile.open('imod_file.mrc') as mrc:
    # mrcfile handles this automatically in recent versions
    origin_x = mrc.header.origin.x

    # Manual check if needed:
    # Read header word 40, check bit 2
```

**Best practice:**
- **When writing:** Use standard convention (positive for physical offset)
- **When reading:** Check for IMOD flag, apply sign correction if needed
- **Verify visually:** Open in multiple tools to confirm origin interpretation

### 6. Coordinate Handedness and Y-axis

#### The Problem

NO rigorous definition of handedness or Y-axis direction in MRC2014.

**Variations:**
- **Standard EM:** Origin at lower-left, Y increases upward (right-handed)
- **RELION:** Origin at upper-left, Y increases downward
- **FEI EPU:** Top-left convention (left-handed or flipped)

#### Practical Consequences

- Images may appear upside-down when exchanged
- Particle orientations can be inverted
- Symmetry operations become ambiguous

#### Solution

**Defensive approach:**
```python
# Python: Document your convention and provide flip option
def read_mrc_with_convention(file_path, flip_y=False):
    with mrcfile.open(file_path) as mrc:
        data = mrc.data.copy()

        if flip_y:
            # Flip Y-axis
            data = np.flip(data, axis=1)  # Assuming (NZ, NY, NX)

        return data
```

**Best practice:**
- **Document your convention** explicitly in code/documentation
- **Provide Y-flip option** when reading/writing
- **Visual validation:** Always spot-check with known reference
- **Store metadata:** Note which convention was used

## Data Exchange Checklist

When exchanging MRC files between tools or projects:

**Before writing:**
- [ ] Use MODE 2 (float32) unless specific reason for other modes
- [ ] Set ISPG correctly (0 for stacks, 1 for volumes)
- [ ] Use correct extension (`.mrcs` for stacks, `.mrc` for volumes)
- [ ] Set standard axis ordering (MAPC=1, MAPR=2, MAPS=3)
- [ ] Set voxel size (XLEN, YLEN, ZLEN based on NX, NY, NZ and pixel size)
- [ ] Update header statistics (DMIN, DMAX, DMEAN, RMS)
- [ ] Set MACHST correctly for your platform
- [ ] If using extended headers, set EXTTYPE and NSYMBT
- [ ] Use standard origin convention (positive for offset)

**After reading (validation):**
- [ ] Check MODE is supported (0, 1, 2, 6, or 12)
- [ ] Validate dimensions are reasonable (NX, NY, NZ > 0, < 100,000)
- [ ] Check NSYMBT for extended headers
- [ ] Verify data offset = 1024 + NSYMBT
- [ ] Validate axis ordering (MAPC, MAPR, MAPS)
- [ ] Check ISPG and extension match expectations
- [ ] If MODE=0, check for IMOD sign convention
- [ ] Verify voxel size is reasonable
- [ ] Spot-check data values for sanity

**Tool-specific considerations:**
- [ ] **IMOD → other:** Check origin sign convention flag
- [ ] **FEI/RELION → other:** Verify Y-axis handedness
- [ ] **Old files:** Verify endianness heuristically
- [ ] **MODE 6/12:** Confirm target software supports these modes

## Implementation Best Practices

### Write Conservative, Read Liberal

**When writing MRC files:**
1. **Use MODE 2 (float32)** for maximum compatibility
2. **Set ISPG and extension correctly**
3. **Use standard axis ordering** (MAPC=1, MAPR=2, MAPS=3)
4. **Avoid extended headers** unless necessary
5. **Update all header statistics** (min, max, mean, RMS)
6. **Set MACHST correctly** (even if readers ignore it)

**When reading MRC files:**
1. **Validate all header fields** before trusting them
2. **Use heuristic endianness detection**
3. **Handle all standard MODEs** (0, 1, 2, 4, 6, 12)
4. **Check for IMOD compatibility flags** if MODE=0
5. **Gracefully skip unknown extended headers**
6. **Provide warnings** for non-standard fields

### Error Handling

**Python example:**
```python
def read_mrc_safe(file_path):
    try:
        with mrcfile.open(file_path, permissive=True) as mrc:
            # Check for warnings
            if not mrcfile.validate(file_path):
                print(f"Warning: {file_path} has non-standard header")

            # Validate dimensions
            if mrc.header.nx <= 0 or mrc.header.ny <= 0:
                raise ValueError(f"Invalid dimensions: {mrc.header.nx}×{mrc.header.ny}")

            # Validate MODE
            if mrc.header.mode not in [0, 1, 2, 4, 6, 12]:
                raise ValueError(f"Unsupported MODE: {mrc.header.mode}")

            # Check data integrity
            expected_size = mrc.header.nx * mrc.header.ny * mrc.header.nz
            if mrc.data.size != expected_size:
                raise ValueError(f"Data size mismatch: expected {expected_size}, got {mrc.data.size}")

            return mrc.data.copy()

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
```

**cisTEM approach:**
```cpp
// Use debug assertions for validation
MyDebugAssertTrue(my_file->is_open(), "File not open!");
MyDebugAssertTrue(nx > 0 && ny > 0 && nz >= 0, "Invalid dimensions!");
MyDebugAssertTrue(mode >= 0 && mode <= 12, "Invalid MODE!");
```

### Testing for Interoperability

**Test matrix:**
```
Test files:
- MODE 0 (int8) with and without IMOD stamp
- MODE 1 (int16) signed data
- MODE 2 (float32) standard
- MODE 6 (uint16) modern detectors
- MODE 12 (float16) compressed
- Little-endian and big-endian
- With and without extended headers
- Stack vs volume (ISPG=0 vs ISPG=1)
- Standard and non-standard axis ordering
- Different origin sign conventions
```

**Cross-tool validation:**
1. Write file with your tool
2. Read with IMOD, Chimera, RELION
3. Visual inspection: do images match?
4. Numerical validation: check pixel values
5. Header inspection: verify metadata preserved

## Software-Specific Notes

### RELION
- Uses `.mrcs` extension for particle stacks
- Particle indexing: `N@stack.mrcs` (N = 1-indexed)
- Y-axis flipping when interfacing with IMOD/MotionCor2

### IMOD
- Origin sign convention (check bit 2 of word 40)
- MODE 0 sign convention (old files: unsigned)
- Handles non-standard axis orderings

### Chimera/ChimeraX
- Standard origin convention
- Handles MODE 2, 6 reliably
- May struggle with non-standard axis ordering

### FEI Software
- Extended headers (FEI1/FEI2)
- MODE 6 for unsigned detector data
- Potential Y-axis handedness differences

### cisTEM
- Converts all MODEs to float32 in memory
- Supports MODE 0, 1, 2, 6, 12 for reading
- Writes MODE 2 (default) or MODE 12 (FP16)
- Assumes little-endian on x86/x64
- Handles 2 axis orderings (standard + one alternative)

### Python mrcfile
- MRC2014 compliant reference implementation
- Validates files (use `mrcfile.validate()`)
- Permissive mode for non-compliant files
- Handles endianness automatically
- Supports all standard MODEs

## Quick Troubleshooting

**Symptom: Images appear inverted or mirrored**
- **Cause:** Y-axis handedness difference or axis ordering mismatch
- **Solution:** Check MAPC/MAPR/MAPS, try Y-axis flip

**Symptom: Data values are garbage/nonsense**
- **Cause:** Endianness mismatch
- **Solution:** Check MACHST, validate with heuristics, try byte-swap

**Symptom: File appears truncated or has extra data**
- **Cause:** Extended header size (NSYMBT) incorrect
- **Solution:** Verify data_offset = 1024 + NSYMBT

**Symptom: MODE 0 images have inverted contrast**
- **Cause:** IMOD sign convention mismatch
- **Solution:** Check imodStamp and imodFlags[0]

**Symptom: MODE 6 data appears to have negative values**
- **Cause:** Software treating uint16 as int16
- **Solution:** Convert to MODE 2 (float32) for compatibility

**Symptom: Particle stack misinterpreted as volume (or vice versa)**
- **Cause:** ISPG field or extension incorrect
- **Solution:** Use `.mrcs` for stacks, set ISPG=0; `.mrc` for volumes, set ISPG=1

## Summary

**Golden rules for interoperability:**
1. **Write MODE 2 (float32)** unless specific reason otherwise
2. **Set ISPG and extension correctly** (stack vs volume)
3. **Use standard axis ordering** (MAPC=1, MAPR=2, MAPS=3)
4. **Avoid extended headers** unless necessary
5. **Validate everything when reading** - trust nothing
6. **Test with multiple tools** before releasing files
7. **Document your conventions** explicitly
8. **Provide options** for Y-flip, endianness, etc.

The MRC format's flexibility is both strength and weakness. Defensive programming and comprehensive validation are essential for reliable data exchange.

## References

- MRC2014 specification: See `citations.md`
- CCP-EM format page: https://www.ccpem.ac.uk/mrc-format/mrc2014/
- mrcfile documentation: https://mrcfile.readthedocs.io/
- IMOD MRC format: https://bio3d.colorado.edu/imod/doc/mrc_format.txt
