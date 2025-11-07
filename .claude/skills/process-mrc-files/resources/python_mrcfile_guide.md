# Python mrcfile Module Research Documentation

## Overview

**mrcfile** is a Python implementation of the MRC2014 file format, used extensively in structural biology and electron microscopy for storing image and volume data.

- **Official Documentation**: https://mrcfile.readthedocs.io/
- **GitHub Repository**: https://github.com/ccpem/mrcfile
- **PyPI Package**: https://pypi.org/project/mrcfile/
- **Current Stable Version**: 1.5.4
- **Development Version**: 1.6.0b0
- **Python Support**: Python 3.x

## Installation

### Via pip (Recommended)
```bash
pip install mrcfile
```

### Via conda
```bash
conda install -c conda-forge mrcfile
```

### From source (development)
```bash
git clone https://github.com/ccpem/mrcfile.git
cd mrcfile
pip install -e .
```

**Dependencies**:
- numpy (primary runtime dependency)
- Minimal external dependencies beyond numpy

## Core Concepts

### File Types and Dimensionality

The mrcfile library distinguishes four types of data based on dimensions and space group (`ispg`):

| Type | Dimensions | Space Group | Use Case |
|------|-----------|-------------|----------|
| Single image | 2D (ny × nx) | 0 | 2D projection or snapshot |
| Image stack | 3D (nz × ny × nx) | 0 | Series of 2D images |
| 3D volume | 3D (nz × ny × nx) | 1–230 | Reconstructed density map |
| Volume stack | 4D (nv × nz × ny × nx) | 401–630 | Multiple 3D volumes |

### Coordinate Conventions

The library follows **Python/C-style axis ordering** (row-major):
- First index is the slowest-changing axis (typically Z)
- Last index is the fastest-changing axis (typically X)
- Data accessed as `data[z][y][x]`
- Arrays are C-contiguous

**Important**: This differs from FORTRAN conventions used in some crystallographic software, which can cause confusion when interfacing with other tools.

### Voxel Size Management

The `voxel_size` attribute provides convenient voxel dimension access:

```python
import mrcfile

with mrcfile.open('file.mrc', mode='r+') as mrc:
    # Read voxel sizes
    voxel_x = mrc.voxel_size.x
    voxel_y = mrc.voxel_size.y
    voxel_z = mrc.voxel_size.z

    # Set isotropic voxel size (scalar)
    mrc.voxel_size = 1.5

    # Set anisotropic voxel sizes (tuple)
    mrc.voxel_size = (1.0, 2.0, 3.0)  # x, y, z

    # Using record array
    mrc.voxel_size = np.array([(1.0, 1.5, 2.0)], dtype=[('x', float), ('y', float), ('z', float)])
```

Voxel sizes derive from header cell and grid dimensions; changing cell dimensions automatically updates voxel sizes and vice versa.

## Data Type Support

Supported numpy dtypes with corresponding MRC modes:

| numpy dtype | MRC Mode | Description |
|-------------|----------|-------------|
| `int8` | 0 | Signed 8-bit integer |
| `int16` | 1 | Signed 16-bit integer |
| `float32` | 2 | 32-bit floating point |
| `complex64` | 4 | 64-bit complex (32-bit real/imag) |
| `uint8` | 6 | Unsigned 8-bit integer |
| `uint16` | 6 | Unsigned 16-bit integer |
| `float16` | 12 | 16-bit floating point (proposed extension) |

**Important Limitations**:
- `int64`, `float64`, and other types are rejected
- Use `float32` for compatibility (float16 is non-standard)
- Be cautious with type conversions—narrowing operations can cause silent overflow

## Reading Operations

### Basic File Reading

#### Quick read (data only)
```python
import mrcfile
import numpy as np

# Read data as numpy array
data = mrcfile.read('file.mrc')
```

#### Open and inspect
```python
with mrcfile.open('file.mrc', mode='r') as mrc:
    # Access header as record array
    print(mrc.header.nx, mrc.header.ny, mrc.header.nz)

    # Access voxel sizes
    print(mrc.voxel_size)

    # Access raw data array
    data = mrc.data
    print(data.dtype, data.shape)

    # Access extended header (if present)
    ext_header = mrc.extended_header
```

### Reading Specific Regions (Slicing)

```python
with mrcfile.open('file.mrc', mode='r') as mrc:
    # Read single 2D slice
    slice_z5 = mrc.data[5, :, :]

    # Read sub-volume
    sub_volume = mrc.data[10:20, 50:100, 30:80]

    # Read with stride
    downsampled = mrc.data[::2, ::2, ::2]
```

### File Modes

Three file opening modes:

- **`'r'`** - Read-only (default for `open()`)
- **`'r+'`** - Read-write access to existing files
- **`'w+'`** - Write new file (overwrites if exists)

```python
# Read-only access
with mrcfile.open('file.mrc', mode='r') as mrc:
    data = mrc.data.copy()  # Must copy to avoid issues

# Read-write access
with mrcfile.open('file.mrc', mode='r+') as mrc:
    mrc.data[0, 0, 0] = 42  # Modify in-place

# Create new file
with mrcfile.new('new_file.mrc', mode='w+', overwrite=True) as mrc:
    mrc.set_data(np.arange(100).reshape(10, 10, 1))
```

### Compressed File Support

Automatic compression detection and handling:

```python
# Gzip-compressed files
data = mrcfile.read('file.mrc.gz')

# Bzip2-compressed files
data = mrcfile.read('file.mrc.bz2')

# Automatic detection from filename extension
with mrcfile.open('file.mrc.gz', mode='r') as mrc:
    data = mrc.data

# Specify compression when creating
with mrcfile.new('new_file.mrc.gz', compression='gzip', overwrite=True) as mrc:
    mrc.set_data(data_array)
```

## Writing Operations

### Creating New Files

#### Simple write (data only)
```python
import numpy as np
import mrcfile

data = np.random.rand(10, 20, 30).astype(np.float32)

# Quick write with default settings
mrcfile.write('new_file.mrc', data)

# With voxel size specification
mrcfile.write('new_file.mrc', data, voxel_size=1.5)
```

#### Detailed file creation
```python
import mrcfile
import numpy as np

data = np.arange(6000, dtype=np.int16).reshape(10, 20, 30)

with mrcfile.new('file.mrc', overwrite=True) as mrc:
    # Set data (dimensions auto-updated in header)
    mrc.set_data(data)

    # Set voxel size
    mrc.voxel_size = 1.2

    # Access and modify header
    mrc.header.nlabrec = 1
    mrc.header.ispg = 1  # Mark as volume

    # Set file type explicitly
    mrc.set_volume()  # or mrc.set_image_stack()
```

### Appending Data

```python
with mrcfile.new('stack.mrc', overwrite=True) as mrc:
    # Initialize with first slice
    mrc.set_data(np.zeros((10, 256, 256), dtype=np.float32))

    # For an image stack, build slices incrementally
    for i in range(10):
        mrc.data[i, :, :] = some_processing_function(i)
```

### Modifying Existing Files

```python
# Read-write mode allows in-place modifications
with mrcfile.open('file.mrc', mode='r+') as mrc:
    # Modify data in-place
    mrc.data[0, :, :] *= 2.0

    # Update header when modifying data
    mrc.update_header_from_data()
    mrc.update_header_stats()  # Recalculate min/max/mean/rms

    # Replace entire dataset
    new_data = np.zeros_like(mrc.data)
    mrc.set_data(new_data)  # Auto-updates header dimensions
```

## Header Manipulation

### Accessing Header Fields

```python
with mrcfile.open('file.mrc', mode='r') as mrc:
    header = mrc.header

    # Dimensions
    nx, ny, nz = header.nx, header.ny, header.nz

    # Data type mode
    mode = header.mode

    # File type identification
    ispg = header.ispg  # Space group (0 for image, 1-230 for volume)

    # Statistics
    dmin, dmax, dmean = header.dmin, header.dmax, header.dmean
    rms = header.rms

    # Crystallographic cell
    cell_x, cell_y, cell_z = header.cella.x, header.cella.y, header.cella.z

    # Axis order (maps, mapy, mapz)
    mapx = header.mapx  # Column axis (1=X, 2=Y, 3=Z)
    mapy = header.mapy  # Row axis
    mapz = header.mapz  # Section axis

    # Map ID string (should be 'MAP ')
    map_id = header.map
```

### Modifying Header Fields

```python
with mrcfile.open('file.mrc', mode='r+') as mrc:
    # Set voxel size via cell dimensions
    mrc.header.cella.x = 100.0
    mrc.header.cella.y = 100.0
    mrc.header.cella.z = 100.0

    # Update grid dimensions
    mrc.header.mx = mrc.header.nx
    mrc.header.my = mrc.header.ny
    mrc.header.mz = mrc.header.nz

    # Set file type
    mrc.header.ispg = 1  # Mark as 3D volume

    # Update header stats (should match data)
    mrc.update_header_stats()
```

### Header Statistics

```python
with mrcfile.open('file.mrc', mode='r+') as mrc:
    # Calculate and update statistics
    mrc.update_header_stats()

    # Statistics are stored in header
    print(f"Min: {mrc.header.dmin}")
    print(f"Max: {mrc.header.dmax}")
    print(f"Mean: {mrc.header.dmean}")
    print(f"RMS: {mrc.header.rms}")
```

**Important**: When data is modified in-place, header statistics become stale until `update_header_stats()` is called.

## Extended Headers

### Indexed Extended Header Access

```python
with mrcfile.open('file.mrc', mode='r') as mrc:
    # Access extended header with known format
    ext_header = mrc.indexed_extended_header

    # Supported formats: 'FEI1', 'FEI2'
    # Returns structured access to sequential metadata blocks

    # Raw extended header as bytes
    raw_ext = mrc.extended_header
```

### Setting Extended Headers

```python
with mrcfile.open('file.mrc', mode='r+') as mrc:
    # Set extended header from bytes
    new_ext_header = b'Custom extended header data here'
    mrc.set_extended_header(new_ext_header)

    # Note: Library does NOT automatically set exttyp field
    # May need to set manually if specific format is required
    mrc.header.exttyp = b'FEI2'
```

## Large File Handling

### Memory-Mapped File Access

For very large files, memory-mapped access loads data on-demand:

```python
# Open file as memory-mapped array
with mrcfile.mmap('large_file.mrc', mode='r') as mrc:
    # Data is accessed lazily from disk
    # Slice operations load only needed chunks

    # Read single slice (fast, minimal memory)
    slice_data = mrc.data[50, :, :]

    # Iterate through slices without loading entire file
    for i in range(mrc.header.nz):
        slice_data = mrc.data[i, :, :]
        process(slice_data)
```

### Creating Memory-Mapped Files

```python
import numpy as np
import mrcfile

# Create empty memory-mapped file
with mrcfile.new_mmap('output.mrc', shape=(100, 512, 512), mrc_mode=2, overwrite=True) as mrc:
    # Fill slice-by-slice
    for i in range(100):
        mrc.data[i, :, :] = generate_slice(i)

    # Set voxel size and other metadata
    mrc.voxel_size = 1.5
    mrc.set_volume()
```

### Asynchronous File Opening

For processing multiple files in parallel:

```python
import mrcfile

# Start background loading
future_file = mrcfile.open_async('large_file.mrc', mode='r')

# Do other work while file loads...
if future_file.done():
    # File is ready
    mrc = future_file.result()
else:
    # Wait for file to load
    mrc = future_file.result()  # Blocks until ready

with mrc:
    data = mrc.data.copy()
```

## Data Shape Identification and Manipulation

### Identifying File Type

```python
with mrcfile.open('file.mrc', mode='r') as mrc:
    # Check file type
    if mrc.is_image_stack():
        print("This is a 2D image stack")
    elif mrc.is_volume():
        print("This is a 3D volume")
    elif mrc.is_single_image():
        print("This is a single 2D image")
    # Check header.ispg for 4D volume stack (401-630)
```

### Setting File Type

```python
with mrcfile.new('output.mrc', overwrite=True) as mrc:
    mrc.set_data(data_array)

    # Explicitly set type
    mrc.set_image_stack()  # Sets ispg=0
    # or
    mrc.set_volume()       # Sets ispg=1
```

## File Validation

### Programmatic Validation

```python
import mrcfile

# Validate single file
is_valid = mrcfile.validate('file.mrc')  # Prints diagnostic messages
# Returns True if valid, False otherwise

# Validate with custom output stream
from io import StringIO
output = StringIO()
is_valid = mrcfile.validate('file.mrc', print_file=output)
validation_report = output.getvalue()
```

### Command-Line Validation

```bash
# Validate one or more MRC files
mrcfile-validate file1.mrc file2.mrc.gz file3.mrc.bz2

# Displays diagnostic information and exits with status code
```

### Permissive Mode

For opening invalid or corrupt files:

```python
# Open file with validation warnings (not errors)
with mrcfile.open('corrupt_file.mrc', mode='r', permissive=True) as mrc:
    # Issues converted to warnings instead of raising exceptions
    # Allows accessing and potentially repairing malformed files
    data = mrc.data
```

Issues detected in permissive mode:
1. Incorrect MAP ID string
2. Invalid machine stamp
3. Unrecognized mode numbers
4. Insufficient data block size

## Command-Line Tools

### mrcfile-validate

```bash
# Basic usage
mrcfile-validate file.mrc

# Multiple files
mrcfile-validate file1.mrc file2.mrc.gz

# Returns exit code indicating validity
```

Output: Diagnostic messages about file validity

### mrcfile-header

```bash
# Display header information
mrcfile-header file.mrc

# Shows all header fields with values
```

## Common Patterns and Best Practices

### Pattern 1: Safe File Reading

```python
import mrcfile
import numpy as np

try:
    with mrcfile.open('file.mrc', mode='r') as mrc:
        # Always copy data when exiting context
        data = mrc.data.copy()
        voxel_size = mrc.voxel_size
finally:
    # File automatically closed
    pass

# Process data outside context manager
processed = data * 2.0
```

### Pattern 2: Batch Processing Large Files

```python
import mrcfile

filenames = ['large1.mrc', 'large2.mrc', 'large3.mrc']

# Process with memory mapping for efficiency
for filename in filenames:
    with mrcfile.mmap(filename, mode='r') as mrc:
        # Iterate through slices without loading entire file
        for z in range(mrc.header.nz):
            slice_data = mrc.data[z, :, :]
            process_slice(slice_data)
```

### Pattern 3: Incremental File Building

```python
import mrcfile
import numpy as np

output_shape = (100, 256, 256)

with mrcfile.new_mmap('output.mrc', shape=output_shape, mrc_mode=2, overwrite=True) as mrc:
    # Fill data incrementally
    for i in range(100):
        mrc.data[i, :, :] = compute_slice(i)

    # Set metadata when complete
    mrc.voxel_size = 1.5
    mrc.set_volume()
```

### Pattern 4: Format Conversion

```python
import mrcfile
import numpy as np

# Read from one format, write to MRC
with mrcfile.open('source.mrc', mode='r') as src:
    data = src.data.copy()
    voxel_size = src.voxel_size

# Write with compression
mrcfile.write('output.mrc.gz', data, voxel_size=voxel_size)
```

### Pattern 5: Header-Preserving Modification

```python
import mrcfile

with mrcfile.open('file.mrc', mode='r+') as mrc:
    # Modify data
    mrc.data[0, :, :] *= scale_factor

    # Update header to reflect changes
    mrc.update_header_from_data()  # Updates nx, ny, nz, mode, ispg
    mrc.update_header_stats()      # Recalculates dmin, dmax, dmean, rms
```

## API Reference Summary

### Core Functions

| Function | Purpose |
|----------|---------|
| `mrcfile.read(filename)` | Read data array from file |
| `mrcfile.write(filename, data, voxel_size=None)` | Write data to new file |
| `mrcfile.open(filename, mode='r', permissive=False)` | Open file with full access |
| `mrcfile.new(filename, overwrite=False)` | Create new file (mode='w+') |
| `mrcfile.mmap(filename, mode='r')` | Open as memory-mapped array |
| `mrcfile.new_mmap(filename, shape, mrc_mode=2, overwrite=False)` | Create new memory-mapped file |
| `mrcfile.open_async(filename, mode='r')` | Open file asynchronously |
| `mrcfile.validate(filename, print_file=None)` | Validate file compliance |

### MrcFile Class Methods

| Method | Purpose |
|--------|---------|
| `set_data(data)` | Replace entire data array |
| `set_extended_header(ext_header)` | Set extended header bytes |
| `set_image_stack()` | Mark file as 2D image stack (ispg=0) |
| `set_volume()` | Mark file as 3D volume (ispg=1) |
| `is_image_stack()` | Check if file is image stack |
| `is_volume()` | Check if file is volume |
| `is_single_image()` | Check if file is single 2D image |
| `update_header_from_data()` | Update dimension and type fields |
| `update_header_stats()` | Recalculate min/max/mean/rms |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `data` | numpy array | Raw data access (read/write when mode='r+') |
| `header` | numpy record array | MRC header record |
| `extended_header` | bytes | Raw extended header |
| `indexed_extended_header` | dict-like | Structured extended header access |
| `voxel_size` | record array | Voxel dimensions (x, y, z) |

## Known Limitations and Gotchas

1. **Header becomes stale**: When data is modified in-place, header statistics become incorrect until `update_header_stats()` is called.

2. **Axis ordering confusion**: Python/C-style indexing (first=Z) differs from FORTRAN conventions—verify when interfacing with other tools.

3. **Type support restrictions**: `int64` and `float64` are not supported; must use `float32` for floating-point data.

4. **Extended header format**: Library does not automatically set `exttyp` field when assigning extended headers; may need manual configuration.

5. **Read-only restrictions**: Cannot reassign header or data attributes directly; must use `set_data()` or in-place modification.

6. **Silent overflow on type narrowing**: Converting from wider to narrower types can cause undetected data loss.

7. **Memory mapping readonly**: Memory-mapped files opened in read-only mode cannot be modified.

## Related Resources

- **GitHub**: https://github.com/ccpem/mrcfile
- **Documentation**: https://mrcfile.readthedocs.io/
- **PyPI**: https://pypi.org/project/mrcfile/
- **MRC2014 Format Specification**: Referenced in official documentation
- **CCP-EM Suite**: mrcfile included in official CCP-EM Python environment

## Version History Notes

- **Current stable**: 1.5.4
- **Development**: 1.6.0b0 (beta)
- **Python support**: Python 3.x
- **No Python 2 support** in recent versions

---

*Research completed: 2025-11-07*
*Documentation version: mrcfile 1.5.4 stable*
