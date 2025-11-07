# MRC Coordinate Conventions

Complete reference for understanding axis ordering, origin specification, and coordinate systems in MRC files.

## Quick Reference

**Standard EM Convention:**
- **X axis (MAPC=1):** Columns, fastest-changing in memory
- **Y axis (MAPR=2):** Rows, medium-changing in memory
- **Z axis (MAPS=3):** Sections/slices, slowest-changing in memory
- **Array indexing:** `data[z][y][x]` (C-order, row-major)
- **Origin:** Lower-left corner of first section (looking at data)

## Axis Ordering Fundamentals

### Memory Layout

MRC files use **C-style row-major ordering** (also called C-contiguous):

```
First voxel read from disk  → data[0][0][0] → position (x=0, y=0, z=0)
Second voxel               → data[0][0][1] → position (x=1, y=0, z=0)
...
After NX voxels            → data[0][1][0] → position (x=0, y=1, z=0)
...
After NX×NY voxels         → data[1][0][0] → position (x=0, y=0, z=1)
```

**X changes fastest, Y changes medium, Z changes slowest.**

### Header Fields: NX, NY, NZ

| Field | Word # | Description | Memory Speed |
|-------|--------|-------------|--------------|
| NX | 1 | Number of columns | Fastest (X axis) |
| NY | 2 | Number of rows | Medium (Y axis) |
| NZ | 3 | Number of sections | Slowest (Z axis) |

**Example:**
```
NX=256, NY=256, NZ=100
→ 256 columns × 256 rows × 100 sections
→ Array shape in Python: (100, 256, 256) = (NZ, NY, NX)
```

### Axis Mapping: MAPC, MAPR, MAPS

These fields specify which spatial axis corresponds to columns/rows/sections:

| Field | Word # | Standard Value | Meaning |
|-------|--------|----------------|---------|
| MAPC | 17 | 1 | Column axis is X |
| MAPR | 18 | 2 | Row axis is Y |
| MAPS | 19 | 3 | Section axis is Z |

**Standard EM:** MAPC=1, MAPR=2, MAPS=3
- Columns → X (fastest)
- Rows → Y (medium)
- Sections → Z (slowest)

**Alternative (rare):** MAPC=3, MAPR=2, MAPS=1
- Columns → Z
- Rows → Y
- Sections → X
- **Requires coordinate remapping when reading**

### cisTEM Handling

```cpp
// Source: mrc_file.cpp:313-386
// cisTEM checks MAPS/MAPC values
if (my_header.ReturnMapping() == 1) {
    // Standard: X fastest, Y medium, Z slowest
    // No remapping needed
} else if (my_header.ReturnMapping() == 2) {
    // Alternative: Z fastest, X slowest
    // Applies voxel-by-voxel remapping
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                output_array[k*ny*nx + j*nx + i] = temp_array[i*ny*nz + j*nz + k];
            }
        }
    }
}
```

**Important:** cisTEM only handles 2 specific axis orderings. Non-standard orderings trigger assertions.

### Python Handling

```python
import mrcfile
import numpy as np

with mrcfile.open('file.mrc') as mrc:
    # mrcfile handles axis mapping automatically
    # data.shape is always (NZ, NY, NX) in Python

    # Check axis mapping
    mapc = mrc.header.mapc  # Should be 1 (X)
    mapr = mrc.header.mapr  # Should be 2 (Y)
    maps = mrc.header.maps  # Should be 3 (Z)

    # If non-standard, mrcfile may:
    # 1. Remap automatically (transparent to user)
    # 2. Issue warning
    # 3. Require manual handling depending on version
```

## Origin and Physical Coordinates

### Origin Fields: XORG, YORG, ZORG

| Field | Word # | Units | Meaning |
|-------|--------|-------|---------|
| XORG | 50 | Ångströms | X coordinate of origin |
| YORG | 51 | Ångströms | Y coordinate of origin |
| ZORG | 52 | Ångströms | Z coordinate of origin |

**Purpose:** Specify where in physical space (real-world coordinates) the image/volume is located.

**Use cases:**
- Placing multiple volumes in a common reference frame
- Tracking subvolume extraction from larger volume
- Aligning maps for visualization

### Origin Sign Convention (CRITICAL ISSUE)

**Problem:** Different software uses opposite sign conventions.

#### Standard EM / Chimera Convention

**Definition:** ORIGIN = position of **outside corner** of first voxel relative to coordinate origin

**Example:**
```
XORG=100.0, YORG=50.0, ZORG=0.0
→ Volume starts at physical position (100, 50, 0) Å
→ First voxel data[0][0][0] located at (100, 50, 0) Å
```

#### IMOD Convention (Historical)

**Definition:** ORIGIN uses opposite sign for cropped images

**Detection:** IMOD uses bit 2 of header word 40 to indicate convention
- Bit 2 = 0: Standard convention
- Bit 2 = 1: IMOD inverted convention

**Example:**
```
IMOD file: XORG=-100.0 (bit 2 set)
Equivalent standard: XORG=+100.0
```

#### Practical Handling

**Python (mrcfile):**
```python
with mrcfile.open('file.mrc', mode='r') as mrc:
    origin_x = mrc.header.origin.x  # In Ångströms
    origin_y = mrc.header.origin.y
    origin_z = mrc.header.origin.z

    # Check for IMOD convention (header word 40)
    # mrcfile typically handles this transparently
```

**cisTEM:**
```cpp
float origin_x = my_header.ReturnOriginX();  // Ångströms
float origin_y = my_header.ReturnOriginY();
float origin_z = my_header.ReturnOriginZ();

// cisTEM uses standard convention
// If reading IMOD files, may need manual sign check
```

**Recommendation:** When exchanging files between IMOD and other tools, verify origin values visually.

### Start Fields: NXSTART, NYSTART, NZSTART

| Field | Word # | Units | Meaning |
|-------|--------|-------|---------|
| NXSTART | 5 | Pixels | Starting index for X |
| NYSTART | 6 | Pixels | Starting index for Y |
| NZSTART | 7 | Pixels | Starting index for Z |

**Purpose:** Integer pixel indices for subvolume extraction.

**Example:**
```
Original volume: 512×512×512
Extract 256×256×256 starting at (128, 128, 128)

Output file header:
  NX=256, NY=256, NZ=256
  NXSTART=128, NYSTART=128, NZSTART=128
```

**Relationship to ORIGIN:**
```
XORG = NXSTART × pixel_size_x
YORG = NYSTART × pixel_size_y
ZORG = NZSTART × pixel_size_z
```

**Note:** Many tools ignore START fields and use ORIGIN instead.

## Pixel Size and Cell Dimensions

### Cell Dimensions: XLEN, YLEN, ZLEN (CELLA)

| Field | Word # | Units | Meaning |
|-------|--------|-------|---------|
| XLEN (CELLA.x) | 11 | Ångströms | Unit cell size in X |
| YLEN (CELLA.y) | 12 | Ångströms | Unit cell size in Y |
| ZLEN (CELLA.z) | 13 | Ångströms | Unit cell size in Z |

**Purpose:** Specify physical dimensions of the entire grid.

### Grid Intervals: MX, MY, MZ

| Field | Word # | Units | Meaning |
|-------|--------|-------|---------|
| MX | 8 | Pixels | Sampling intervals in X |
| MY | 9 | Pixels | Sampling intervals in Y |
| MZ | 10 | Pixels | Sampling intervals in Z |

**Purpose:** Specify grid sampling.

**Common convention:** MX=NX, MY=NY, MZ=NZ (grid matches image dimensions)

### Calculating Pixel Size

**Formula:**
```
pixel_size_x = XLEN / MX  (Ångströms/pixel)
pixel_size_y = YLEN / MY
pixel_size_z = ZLEN / MZ
```

**Example:**
```
NX=256, NY=256, NZ=100
MX=256, MY=256, MZ=100
XLEN=384.0 Å, YLEN=384.0 Å, ZLEN=150.0 Å

pixel_size_x = 384.0 / 256 = 1.5 Å/pixel
pixel_size_y = 384.0 / 256 = 1.5 Å/pixel
pixel_size_z = 150.0 / 100 = 1.5 Å/pixel
→ Isotropic 1.5 Å voxels
```

### Python (mrcfile)

```python
with mrcfile.open('file.mrc', mode='r+') as mrc:
    # Read voxel size (convenience property)
    voxel_x = mrc.voxel_size.x  # Ångströms/pixel
    voxel_y = mrc.voxel_size.y
    voxel_z = mrc.voxel_size.z

    # Set isotropic voxel size
    mrc.voxel_size = 1.5  # All axes set to 1.5 Å

    # Set anisotropic voxel sizes
    mrc.voxel_size = (1.0, 1.0, 2.0)  # X, Y, Z in Å

    # Low-level: set cell dimensions directly
    mrc.header.cella.x = 256.0 * 1.5  # NX × pixel_size
    mrc.header.cella.y = 256.0 * 1.5  # NY × pixel_size
    mrc.header.cella.z = 100.0 * 1.5  # NZ × pixel_size
```

**Note:** `mrcfile` automatically keeps MX/MY/MZ = NX/NY/NZ when using `voxel_size` property.

### cisTEM

```cpp
// Read pixel size
float pixel_size = my_file.ReturnPixelSize();  # Assumes isotropic

// Set pixel size (updates XLEN, YLEN, ZLEN)
my_file.SetPixelSize(1.5);  # 1.5 Å/pixel

// Set and write header immediately
my_file.SetPixelSizeAndWriteHeader(1.5);
```

**Important:** cisTEM assumes **isotropic voxels** (pixel_size_x = pixel_size_y = pixel_size_z).

## Cell Angles: ALPHA, BETA, GAMMA

| Field | Word # | Units | Meaning |
|-------|--------|-------|---------|
| ALPHA | 14 | Degrees | Angle between Y and Z axes |
| BETA | 15 | Degrees | Angle between X and Z axes |
| GAMMA | 16 | Degrees | Angle between X and Y axes |

**Standard EM:** ALPHA = BETA = GAMMA = 90.0° (orthogonal axes)

**Purpose:** Crystallographic convention; rarely used in EM.

**For EM data:** Almost always 90°, 90°, 90°.

## Space Group: ISPG

| Value | Meaning |
|-------|---------|
| 0 | Image or image stack (2D data, or stack of 2D images) |
| 1 | Volume (3D data, no symmetry) |
| 2-230 | Crystallographic space groups (rarely used in EM) |
| 401-630 | Volume stacks (multiple 3D volumes) |

**Critical distinction:**
- **ISPG=0:** Z dimension represents separate 2D images (particle stack)
- **ISPG=1:** Z dimension is part of 3D volume
- **ISPG=401:** Stack of 3D volumes

**Problem:** Format doesn't enforce this; many files have ISPG=0 even for volumes.

**RELION Convention (more reliable):**
- `.mrcs` extension → particle stack (ISPG should be 0)
- `.mrc` extension → 3D volume (ISPG should be 1)

## Array Indexing Summary

### Python (NumPy)

```python
import mrcfile

with mrcfile.open('file.mrc') as mrc:
    # Shape is ALWAYS (NZ, NY, NX) regardless of MAPS/MAPC
    print(mrc.data.shape)  # (100, 256, 256) for example

    # Access voxel at physical position (x=50, y=100, z=10)
    value = mrc.data[10, 100, 50]  # data[z, y, x]

    # Extract slice at z=10
    slice_z10 = mrc.data[10, :, :]  # Shape: (NY, NX) = (256, 256)

    # Extract column at x=50, y=100
    column = mrc.data[:, 100, 50]  # Shape: (NZ,) = (100,)
```

### C++ (cisTEM)

```cpp
// Data stored as 1D array
float* data = new float[nx * ny * nz];
MRCFile file("file.mrc", false);
file.ReadSlicesFromDisk(0, nz-1, data);

// Access voxel at position (x=50, y=100, z=10)
int index = (10 * ny * nx) + (100 * nx) + 50;  // z*ny*nx + y*nx + x
float value = data[index];

// Extract slice at z=10
int slice_size = nx * ny;
float* slice_z10 = &data[10 * slice_size];
```

### Fortran (For Reference)

**NOT used in cisTEM or mrcfile**, but relevant for understanding CCP4/crystallography tools:

```fortran
! Fortran uses column-major order (opposite of C)
! First index changes fastest
REAL DATA(NX, NY, NZ)  ! X fastest, Y medium, Z slowest

! Access voxel at (x=50, y=100, z=10)
VALUE = DATA(50, 100, 10)  ! (x, y, z) - intuitive order
```

**Implication:** When interfacing with Fortran-based tools, be aware of potential transpose.

## Visual Conventions

### Looking at Image Slices

Standard visualization convention:
```
      X →
   ┌──────────┐
Y  │          │
↓  │  IMAGE   │
   │          │
   └──────────┘
```

- **Origin:** Lower-left corner (0, 0)
- **X increases:** Left to right
- **Y increases:** Top to bottom (in memory) BUT often displayed bottom to top
- **Z increases:** Through the stack

**Display convention inconsistency:**
- Some software displays Y increasing upward (origin at bottom-left visually)
- Others display Y increasing downward (origin at top-left visually)
- **MRC format defines origin at lower-left**, but display varies

### 3D Volume Orientation

```
     Z (sections)
     ↑
     |
     |
     o────→ X (columns)
    /
   /
  ↓
  Y (rows)
```

- Right-handed coordinate system
- X = fastest (columns)
- Y = medium (rows)
- Z = slowest (sections)

## Common Pitfalls

### Pitfall 1: Confusing NZ with Z-axis in Memory

**Wrong assumption:** "NZ controls how fast Z changes in memory"

**Reality:** Z is SLOWEST in memory. X is fastest.

**Array indexing:**
```python
data[z, y, x]  # Z outer loop (slowest), X inner loop (fastest)
```

### Pitfall 2: Assuming ISPG Reliably Indicates Stack vs Volume

**Reality:** Many 3D volumes have ISPG=0.

**Better indicator:** File extension (.mrc vs .mrcs) or context.

### Pitfall 3: Ignoring MAPC/MAPR/MAPS

**Risk:** Non-standard axis mappings will cause data corruption if not handled.

**Check:** Validate MAPC=1, MAPR=2, MAPS=3 when reading critical data.

### Pitfall 4: Origin Sign Convention

**Risk:** IMOD files may have opposite sign convention.

**Solution:** Verify origin visually when exchanging files between IMOD and other tools.

### Pitfall 5: Anisotropic Voxels

**Assumption:** "All voxels are cubic"

**Reality:** Voxel sizes can differ (e.g., 1.0 Å × 1.0 Å × 1.5 Å).

**Check:**
```python
if not (mrc.voxel_size.x == mrc.voxel_size.y == mrc.voxel_size.z):
    print("Warning: Anisotropic voxels")
```

## Best Practices

1. **Always use standard axis ordering** (MAPC=1, MAPR=2, MAPS=3) when writing files
2. **Check axis mapping** when reading files from unknown sources
3. **Set ISPG correctly**: 0 for stacks, 1 for volumes
4. **Use `.mrcs` extension** for particle stacks, `.mrc` for volumes
5. **Verify origin values** when exchanging files between tools
6. **Document voxel size** in metadata or filenames
7. **Test with visualization** to catch orientation errors early

## References

- MRC2014 specification: See `citations.md`
- Axis mapping implementation: `mrc_file.cpp:313-386`
- Python voxel_size property: mrcfile documentation
