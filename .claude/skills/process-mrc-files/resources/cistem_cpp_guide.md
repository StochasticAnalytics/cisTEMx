# cisTEMx MRC C++ File Implementation Research

**Comprehensive analysis of the MRC file format implementation in cisTEMx**

## File Locations and Overview

| Component | Path | Lines |
|-----------|------|-------|
| MRC Header Class | `/workspaces/cisTEMx/src/core/mrc_header.h` | 1-182 |
| MRC Header Implementation | `/workspaces/cisTEMx/src/core/mrc_header.cpp` | 1-399 |
| MRC File Class | `/workspaces/cisTEMx/src/core/mrc_file.h` | 1-91 |
| MRC File Implementation | `/workspaces/cisTEMx/src/core/mrc_file.cpp` | 1-493 |
| Abstract Image File Base | `/workspaces/cisTEMx/src/core/abstract_image_file.h` | 1-40 |

---

## MRC Format Specification

### Header Structure

The MRC format header is 1024 bytes (256 32-bit words), followed by 10 optional 80-byte text labels:

| Word # | Name | Type | Bytes | Description |
|--------|------|------|-------|-------------|
| 1 | NX | int | 0-3 | Number of columns (fastest changing in memory) |
| 2 | NY | int | 4-7 | Number of rows |
| 3 | NZ | int | 8-11 | Number of sections (slowest changing in memory) |
| 4 | MODE | int | 12-15 | Data type |
| 5-7 | NXSTART, NYSTART, NZSTART | int | 16-27 | Starting indices |
| 8-10 | MX, MY, MZ | int | 28-39 | Number of intervals along each axis |
| 11-13 | CELLA | float | 40-51 | Cell dimensions in Angstroms (X, Y, Z) |
| 14-16 | CELLB | float | 52-63 | Cell angles in degrees (X, Y, Z) |
| 17-19 | MAPC, MAPR, MAPS | int | 64-75 | Axis correspondence (1=X, 2=Y, 3=Z) |
| 20-22 | DMIN, DMAX, DMEAN | float | 76-87 | Density statistics |
| 23 | ISPG | int | 88-91 | Space group number (0 for images, 1 for volumes) |
| 24 | NSYMBT | int | 92-95 | Symmetry data bytes |
| 25-49 | EXTRA | int | 96-151 | Extra space (25 words) |
| 50-52 | IMOD fields | int | 152-163 | IMOD compatibility flags |
| 53-55 | ORIGIN | float | 196-207 | Origin for transforms (X, Y, Z) |
| 56 | MAP | char | 208-211 | "MAP " identifier |
| 57 | MACHST | int | 212-215 | Machine stamp |
| 58 | RMS | float | 216-219 | RMS deviation |
| 59 | NLABL | int | 220-223 | Number of labels used |
| 60-256 | LABEL | char | 224-1023 | 10 × 80-byte text labels |

**Source:** `mrc_header.h:14-57`

---

## Data Types (MODE Values)

**Location:** `mrc_header.h:60-64`, `mrc_header.cpp:224-312`

```cpp
enum MRCDataTypes {
    MRCByte,        // 0: unsigned 8-bit (default) or signed (with IMOD flag)
    MRCInteger,     // 1: signed 16-bit halfwords
    MRCFloat,       // 2: 32-bit float (most common)
    MRCFloat16,     // 3: 16-bit float (IEEE 754 half precision, mode 12)
    MRC4Bit         // 4: 4-bit packed format (Mastronarde hack)
};
```

### Mode Mapping Details

**Supported reading modes:**

| Mode | Name | Bytes/Pixel | Type | Complex | Notes |
|------|------|-------------|------|---------|-------|
| 0 | Byte | 1 | uint8 (usually) | No | Can be signed with IMOD stamp 1146047817 + imodFlags[0] & 1 |
| 1 | Integer | 2 | int16 | No | 16-bit signed integers |
| 2 | Float | 4 | float32 | No | **Default for writing** |
| 5 | Unsigned Short | 2 | uint16 | No | Unsigned variant |
| 6 | Unsigned Integer | 2 | uint16 | No | Duplicate of mode 5 behavior |
| 12 | Half Float | 2 | float16 | No | IEEE 754 half precision (FP16) |
| 101 | 4-bit packed | 0.5 | uint4 | No | Mastronarde 4-bit hack format |
| 3 | Complex Integer | 2 | cmplx int16 | Yes | Complex 16-bit |
| 4 | Complex Float | 4 | cmplx float32 | Yes | Complex 32-bit |

**Conversion strategy in reading:** All file formats are converted to `float` in memory for processing. The `ReadSlicesFromDisk()` method handles format conversion automatically.

**Source:** `mrc_header.cpp:213-312`, `mrc_file.cpp:148-400`

---

## MRCHeader Class Design

### Class Structure

**Location:** `mrc_header.h:66-181`

#### Memory Layout Pattern

The MRCHeader uses a clever pointer-based design that maps directly to the 1024-byte buffer:

```cpp
class MRCHeader {
private:
    char* buffer;  // 1024-byte buffer containing raw header

    // Pointers positioned at specific offsets in buffer
    int*   nx;              // &buffer[0]
    int*   ny;              // &buffer[4]
    int*   nz;              // &buffer[8]
    int*   mode;            // &buffer[12]
    int*   nxstart;         // &buffer[16]
    // ... etc
    float* cell_a_x;        // &buffer[40]
    float* cell_a_y;        // &buffer[44]
    float* cell_a_z;        // &buffer[48]
    // ... all 56 header fields follow at specific offsets

    // Derived state (computed during ReadHeader)
    float bytes_per_pixel;
    bool  pixel_data_are_signed;
    int   pixel_data_are_of_type;
    bool  pixel_data_are_complex;
    bool  this_is_in_mastronarde_4bit_hack_format;
    bool  dimensions_set;
};
```

**Pointer initialization:** `InitPointers()` method sets all pointers to correct offsets in the buffer using pointer arithmetic.

**Source:** `mrc_header.h:66-130`, `mrc_header.cpp:175-211`

#### Key Methods

```cpp
// Constructor & Lifecycle
MRCHeader();                       // Allocates 1024-byte buffer, initializes pointers
~MRCHeader();                      // Deallocates buffer
void InitPointers();               // Sets all header field pointers

// I/O Operations
void ReadHeader(std::fstream* MRCFile);   // Reads first 1024 bytes, analyzes MODE
void WriteHeader(std::fstream* MRCFile);  // Writes first 1024 bytes
void BlankHeader();                       // Initializes header to default values

// Dimension and Mode Setting
void SetDimensionsImage(int x, int y);               // Sets NX, NY, NZ=0
void SetDimensionsVolume(int x, int y, int z);      // Sets NX, NY, NZ
void SetNumberOfImages(int count);                  // Sets NZ (for image stacks)
void SetNumberOfVolumes(int count);                 // Sets NZ (for 3D volumes)
void SetMode(int wanted_mode);                      // Sets MODE (validates 2 or 12)

// Density Statistics
void SetDensityStatistics(float min, float max, float mean, float rms);

// Origin and Pixel Size
void SetOrigin(float x, float y, float z);
void SetPixelSize(float wanted_pixel_size);
float ReturnPixelSize();

// Label Management
void ResetLabels();                 // Sets default label "**GuiX"
void ResetOrigin();                 // Sets origin to (0,0,0)

// Information Display
void PrintInfo();
void SetLocalMachineStamp();        // Detects byte order

// Inline Accessors
inline int ReturnDimensionX() { return nx[0]; }
inline int ReturnDimensionY() { return ny[0]; }
inline int ReturnDimensionZ() { return nz[0]; }
inline int ReturnMapC/R/S() { return map_c/r/s[0]; }
inline bool PixelDataAreSigned() { return pixel_data_are_signed; }
inline float BytesPerPixel() { return bytes_per_pixel; }
inline int Mode() { return mode[0]; }
inline int SymmetryDataBytes() { return symmetry_data_bytes[0]; }
```

**Source:** `mrc_header.h:127-181`, `mrc_header.cpp:1-399`

---

## MRCFile Class Design

### Class Structure

**Location:** `mrc_file.h:17-89`

```cpp
class MRCFile : public AbstractImageFile {
public:
    std::fstream* my_file;     // Underlying file stream
    MRCHeader     my_header;   // Header object
    wxString      filename;    // Current filename

    bool do_nothing;           // Skip all I/O when file is /dev/null
    bool rewrite_header_on_close;
    int  max_number_of_seconds_to_wait_for_file_to_exist;

    // Constructors & Lifecycle
    MRCFile();
    MRCFile(std::string filename, bool overwrite = false);
    MRCFile(std::string filename, bool overwrite, bool wait_for_file_to_exist);
    ~MRCFile();

    // Assignment operators
    MRCFile& operator=(const MRCFile& other_file);
    MRCFile& operator=(const MRCFile* other_file);

    // AbstractImageFile interface methods
    inline int ReturnXSize() { return my_header.ReturnDimensionX(); }
    inline int ReturnYSize() { return my_header.ReturnDimensionY(); }
    inline int ReturnZSize() { return my_header.ReturnDimensionZ(); }
    inline int ReturnNumberOfSlices() { return my_header.ReturnDimensionZ(); }
    inline bool IsOpen() { return my_file->is_open(); }

    // File Operations
    bool OpenFile(std::string filename, bool overwrite,
                  bool wait_for_file_to_exist = false,
                  bool check_only_the_first_image = false,
                  int eer_super_res_factor = 1,
                  int eer_frames_per_image = 0);
    void CloseFile();
    void FlushFile();
    void SetOutputToFP16();  // Switches mode to 12 (FP16)

    // Slice I/O - Partial Reading/Writing
    void ReadSliceFromDisk(int slice_number, float* output_array);
    void ReadSlicesFromDisk(int start_slice, int end_slice, float* output_array);
    void WriteSliceToDisk(int slice_number, float* input_array);
    void WriteSlicesToDisk(int start_slice, int end_slice, float* input_array);

    // Header Management
    void WriteHeader();
    void PrintInfo();

    // Pixel Size and Metadata
    float ReturnPixelSize();
    void SetPixelSize(float wanted_pixel_size);
    void SetPixelSizeAndWriteHeader(float wanted_pixel_size);
    void SetDensityStatistics(float min, float max, float mean, float rms);

    // Validation
    bool HasSameDimensionsAs(MRCFile& other_file);
};
```

**Source:** `mrc_file.h:17-89`

---

## Core Functionality: Reading and Writing

### Reading Data from Disk

**Method:** `MRCFile::ReadSlicesFromDisk(int start_slice, int end_slice, float* output_array)`

**Location:** `mrc_file.cpp:148-400`

#### Algorithm Overview

1. **Seek Calculation:** Computes file offset based on slice index and bytes per pixel
   - Accounts for 1024-byte header
   - Adds symmetry data bytes offset (`SymmetryDataBytes()`)
   - Handles special cases (4-bit Mastronarde hack, mode 101)

2. **Format-Specific Reading:** Reads data according to MODE field
   ```cpp
   switch (my_header.Mode()) {
       case 0:    // 1-byte (with 4-bit unpacking)
       case 1:    // 2-byte signed integer
       case 2:    // 4-byte float (direct read)
       case 6:    // 2-byte unsigned integer
       case 12:   // 2-byte float (FP16, uses half_float library)
       case 101:  // 4-bit packed (unpacking logic)
   }
   ```

3. **Format Conversion:** All types converted to float in memory
   - Byte → float
   - Short → float
   - Float → float (direct)
   - Half → float (using IEEE 754 conversion)

4. **Coordinate System Correction:** Handles non-standard axis mappings
   - Standard: MAPC=1, MAPR=2, MAPS=3 (no remapping needed)
   - Alternative: MAPS=1, MAPC=3 (requires coordinate transformation)
   - Applies temp array + voxel-by-voxel coordinate remapping

#### Key Implementation Details

- **Mastronarde 4-bit hack detection** (`sizeCanBe4BitK2SuperRes()` utility)
- **IMOD compatibility:** Reads `imodStamp` (1146047817) and `imodFlags` to determine signedness
- **FP16 support:** Uses `#include "../../include/ieee-754-half/half.hpp"` for half-precision conversion
- **Partial I/O:** Reads only specified slice range (memory efficient for large stacks)
- **Assertion validation:**
  ```cpp
  MyDebugAssertTrue(my_file->is_open(), "File not open!");
  MyDebugAssertTrue(start_slice <= ReturnNumberOfSlices(), "Start slice > total!");
  MyDebugAssertTrue(start_slice <= end_slice, "Start > end!");
  ```

**Source:** `mrc_file.cpp:148-400`

### Writing Data to Disk

**Method:** `MRCFile::WriteSlicesToDisk(int start_slice, int end_slice, float* input_array)`

**Location:** `mrc_file.cpp:402-471`

#### Algorithm Overview

1. **Seek Calculation:** Same logic as reading
   - Computes file position: `1024 + image_offset + SymmetryDataBytes()`
   - `image_offset = (start_slice - 1) * bytes_per_slice`

2. **Format-Specific Writing:** Only supports limited modes
   ```cpp
   switch (my_header.Mode()) {
       case 0:    // Convert float → char
       case 1:    // Convert float → short
       case 2:    // Direct write (float32)
       case 12:   // Convert float → half (FP16)
       default:   // Error on other modes
   }
   ```

3. **Data Type Conversion:**
   - Input array is always `float*`
   - Casts/converts to target format
   - Writes to disk

4. **Error Handling:**
   - `MyDebugAssertTrue(my_file->is_open())` - file must be open
   - `MyDebugAssertTrue(start_slice <= end_slice)` - logical ordering
   - Mode validation (rejects unsupported types with abort)

#### Supported Write Modes

| Mode | Support | Notes |
|------|---------|-------|
| 0 | Yes | Byte output |
| 1 | Yes | 16-bit integer |
| 2 | Yes | **Preferred** - 32-bit float |
| 12 | Yes | 16-bit float (FP16) |
| 3-6, 101 | No | Reading only |

**Default write mode:** Mode 2 (32-bit float), but can be switched to 12 (FP16) via `SetOutputToFP16()`

**Source:** `mrc_file.cpp:402-471`

---

## File Opening and Closing

### OpenFile Method

**Location:** `mrc_file.cpp:65-132`

```cpp
bool OpenFile(std::string wanted_filename, bool overwrite,
              bool wait_for_file_to_exist = false,
              bool check_only_first_image = false,
              int eer_super_res_factor = 1,
              int eer_frames_per_image = 0);
```

#### Behavior

**Special case handling:**
- If filename starts with "/dev/null", sets `do_nothing = true` (all I/O operations become no-ops)

**Existing file handling:**
- `overwrite = true`: Deletes file, creates new with write mode
- `overwrite = false`: Opens existing file with read/write mode (or read-only if write fails)

**New file handling:**
- Creates file with `std::ios::trunc` flag (truncate to zero)
- Calls `BlankHeader()` to initialize default header

**File waiting:**
- If `wait_for_file_to_exist = true`, waits up to `max_number_of_seconds_to_wait_for_file_to_exist` (default 30 seconds)

**Header state:**
- Existing files: `ReadHeader()` is called to populate header structure
- New files: `BlankHeader()` is called with defaults:
  - Mode = 2 (32-bit float)
  - Space group = 1
  - Map identification: "MAP "
  - Machine stamp set based on endianness

**Returns:** `true` (always, except on fatal open failure which calls `DEBUG_ABORT`)

**Source:** `mrc_file.cpp:65-132`

### CloseFile and FlushFile

**CloseFile:** `mrc_file.cpp:50-57`
- If `rewrite_header_on_close == true`, writes header before closing
- Closes the fstream
- Resets `do_nothing` flag

**FlushFile:** `mrc_file.cpp:59-63`
- Calls `std::fstream::flush()` if file is open
- Used to ensure data is written to disk without closing

**Source:** `mrc_file.cpp:50-63`

---

## Coordinate Conventions and Data Layout

### Memory Layout Convention

**Standard cisTEMx convention (MRC defaults):**
- **X axis** (fastest changing): MAPC = 1
- **Y axis** (medium): MAPR = 2
- **Z axis** (slowest): MAPS = 3

**Memory order:** In the float array, data is laid out as:
```
[slice_0, y_0, x_0], [slice_0, y_0, x_1], ..., [slice_0, y_1, x_0], ...
```

i.e., X varies fastest, then Y, then Z (column-major for 2D slices).

### Non-Standard Axis Mapping Handling

**Location:** `mrc_file.cpp:342-398`

If the MRC file has `MAPS=1` and `MAPC=3` (axes swapped), the reading function:
1. Reads raw data into temporary array
2. Iterates through each voxel with triple loop:
   ```cpp
   for (sec_index = 0; sec_index < NZ; sec_index++)
       for (row_index = 0; row_index < NY; row_index++)
           for (col_index = 0; col_index < NX; col_index++)
               output_array[counter] = temp_array[sec_index + NZ * row_index + NZ * NY * col_index];
   ```
3. Remaps from file coordinate system to cisTEMx convention

**Note:** Only handles one specific alternate mapping; other orderings trigger assertion failure with "strange ordering of data in MRC file not yet supported".

**Source:** `mrc_file.cpp:342-398`

---

## 2D Images vs 3D Volumes vs Stacks

### Image Stacks (Multiple 2D slices)

**Header setup:**
```cpp
my_header.SetDimensionsImage(x_size, y_size);      // Sets NX, NY, NZ=0
my_header.SetNumberOfImages(num_images);           // Sets NZ = num_images
// ISPG = 1 (contravenes MRC2014 but used in cisTEMx)
```

**Typical usage:**
```cpp
MRCFile particle_stack(filename, true);  // Create new file
particle_stack.my_header.SetDimensionsImage(256, 256);
particle_stack.my_header.SetNumberOfImages(1000);
particle_stack.WriteHeader();

// Write particles
for (int i = 0; i < 1000; i++) {
    Image particle;
    particle.ReadSlice(&input_file, i+1);
    particle.WriteSlice(&particle_stack, i+1);
}
```

### 3D Volumes

**Header setup:**
```cpp
my_header.SetDimensionsVolume(x_size, y_size, z_size);  // Sets NX, NY, NZ
// ISPG = 1
```

**Typical usage:**
```cpp
MRCFile reconstruction(filename, true);
reconstruction.my_header.SetDimensionsVolume(256, 256, 256);
reconstruction.WriteHeader();

// Write volume slices
Image reconstruction_slice;
for (int slice = 0; slice < 256; slice++) {
    reconstruction_slice.ReadSlice(&temp_file, slice+1);
    reconstruction_slice.WriteSlice(&reconstruction, slice+1);
}
```

### Reading entire 3D volumes or stacks

**Image class integration:**
```cpp
// In Image class (image.h:460-470)
void ReadSlices(MRCFile* input_file, long start_slice, long end_slice);
void WriteSlices(MRCFile* input_file, long start_slice, long end_slice);

// Usage example from multiply_two_stacks.cpp:63-66
Image my_image;
my_image.ReadSlice(&input_file, slice_number);    // Reads 1 slice
my_image.WriteSlice(&output_file, slice_number);  // Writes 1 slice

// Or for batches:
my_image.ReadSlices(&input_file, 1, num_slices);   // Reads all slices into 3D array
```

**Source:** `mrc_file.h:37-71`, `image.h:436-471`

---

## Pixel Size and Physical Coordinates

### Pixel Size Storage and Retrieval

**Pixel size is derived from CELLA (cell dimensions) and MX/MY/MZ (grid points):**

```cpp
float pixel_size = cell_a_x / mx;  // In Angstroms
```

**Methods:**
```cpp
float ReturnPixelSize();                               // Returns pixel size in Angstroms
void SetPixelSize(float wanted_pixel_size);            // Updates CELLA based on MX
void SetPixelSizeAndWriteHeader(float pixel_size);     // Sets pixel size and writes to disk
```

**Implementation:**
```cpp
// In mrc_header.cpp:80-95
float MRCHeader::ReturnPixelSize() {
    if (cell_a_x[0] == 0.0) return 0.0;
    return cell_a_x[0] / mx[0];
}

void MRCHeader::SetPixelSize(float wanted_pixel_size) {
    cell_a_x[0] = wanted_pixel_size * mx[0];
    cell_a_y[0] = wanted_pixel_size * my[0];
    cell_a_z[0] = wanted_pixel_size * mz[0];
}
```

### MX/MY/MZ Grid Definition

**MX, MY, MZ:** Number of intervals (grid points) along each axis
- Usually equals NX, NY, NZ (set during `SetDimensionsImage/Volume`)
- Used as denominator when converting Angstrom cell dimensions to pixel size
- From `mrc_header.cpp:105-106`: `mx[0] = wanted_x_dim; my[0] = wanted_y_dim;`

**Source:** `mrc_header.cpp:80-95`, `mrc_header.h:160-161`

---

## Integration with Image Class

### Key Interaction Methods

The `Image` class provides convenience methods for working with MRC files:

```cpp
// Single slice I/O
void ReadSlice(MRCFile* input_file, long slice_to_read);
void WriteSlice(MRCFile* output_file, long slice_to_write);

// Multi-slice I/O
void ReadSlices(MRCFile* input_file, long start_slice, long end_slice);
void WriteSlices(MRCFile* output_file, long start_slice, long end_slice);

// Batch writing with header fill
void WriteSlicesAndFillHeader(std::string wanted_filename, float wanted_pixel_size);
```

**Source:** `image.h:436-471`

### Usage Pattern Example

From `/workspaces/cisTEMx/src/programs/multiply_two_stacks/multiply_two_stacks.cpp:40-76`:

```cpp
// Open input files (read mode)
ImageFile my_input_file_one(input_filename_one, false);
ImageFile my_input_file_two(input_filename_two, false);

// Open output file (write mode)
MRCFile my_output_file(output_filename, true);

// Validate dimensions match
if (my_input_file_one.ReturnXSize() != my_input_file_two.ReturnXSize()) {
    MyPrintfRed("Error: Image dimensions are not the same\n");
    exit(-1);
}

// Pixel size preservation
float input_pixel_size = my_input_file_one.ReturnPixelSize();

// Slice-by-slice processing
for (long image_counter = 0; image_counter < my_input_file_one.ReturnNumberOfSlices(); image_counter++) {
    Image my_image_one;
    Image my_image_two;

    // Read individual slices
    my_image_one.ReadSlice(&my_input_file_one, image_counter + 1);
    my_image_two.ReadSlice(&my_input_file_two, image_counter + 1);

    // Process
    my_image_one.MultiplyPixelWise(my_image_two);

    // Write back
    my_image_one.WriteSlice(&my_output_file, image_counter + 1);
}

// Update header with pixel size after all data written
my_output_file.SetPixelSize(input_pixel_size);
my_output_file.WriteHeader();
```

**Source:** `/workspaces/cisTEMx/src/programs/multiply_two_stacks/multiply_two_stacks.cpp:40-76`

---

## Special Features

### FP16 (Half-Precision Float) Support

**Location:** `mrc_file.cpp:14, 268-276, 454-463`

- Uses external library: `#include "../../include/ieee-754-half/half.hpp"`
- Mode 12 (FP16) is fully supported for both reading and writing
- Conversion handled transparently via `half_float::half` type
- Memory efficient for large maps where precision loss is acceptable

```cpp
void SetOutputToFP16() {
    my_header.SetMode(12);
}

// Reading FP16
case 12: {
    std::vector<half> temp_half_array(records_to_read);
    my_file->read((char*)temp_half_array.data(), records_to_read * 2);
    for (long counter = 0; counter < records_to_read; counter++) {
        output_array[counter] = float(temp_half_array[counter]);
    }
}

// Writing FP16
case 12: {
    std::vector<half> temp_half_array(records_to_read);
    for (long counter = 0; counter < records_to_read; counter++) {
        temp_half_array[counter] = half(input_array[counter]);
    }
    my_file->write((char*)temp_half_array.data(), records_to_read * 2);
}
```

**Source:** `mrc_file.h:62`, `mrc_file.cpp:37-40, 268-276, 454-463`

### Mastronarde 4-bit Format

**Location:** `mrc_file.cpp:168-179, 223-244, 290-334`

David Mastronarde's 4-bit hack packs 2 pixels per byte, with special handling for odd-width images:
- Detection: `sizeCanBe4BitK2SuperRes(nx, ny)` utility function
- When detected: `bytes_per_pixel = 0.5`, `nx *= 2`
- Unpacking: Extracts low 4 bits and high 4 bits from each byte
- Special case: If X is odd, padding byte at end of row

**Example unpacking logic:**
```cpp
for (long counter = 0; counter < records_to_read; counter++) {
    low_4bits = temp_char_array[counter] & 0x0F;
    hi_4bits = (temp_char_array[counter] >> 4) & 0x0F;
    output_array[output_counter++] = float(low_4bits);
    output_array[output_counter++] = float(hi_4bits);
}
```

**Source:** `mrc_file.cpp:168-179, 223-244, 290-334`

### /dev/null Optimization

**Location:** `mrc_file.cpp:69-70`

If filename is "/dev/null", all I/O operations become no-ops:
```cpp
do_nothing = StartsWithDevNull(wanted_filename);
if (do_nothing) {
    filename = wanted_filename;
    // ... rest of operations skipped
}
```

Used to suppress output during testing or when output isn't needed.

**Source:** `mrc_file.cpp:69-72`

### Header Rewriting on Close

**Location:** `mrc_file.cpp:52-53, 50-57`

If `rewrite_header_on_close == true`, the header is automatically written when the file closes:

```cpp
void CloseFile() {
    if (my_file->is_open()) {
        if (rewrite_header_on_close == true)
            WriteHeader();
        my_file->close();
    }
}
```

Useful for workflows where density statistics or pixel size are updated during processing.

**Source:** `mrc_file.cpp:50-57`

---

## Known Limitations and Notes

### Reading Only

The following data types can be read but not written:
- Mode 3 (complex 16-bit integers)
- Mode 4 (complex 32-bit reals)
- Mode 5 (unsigned 8-bit)
- Mode 6 (unsigned 16-bit)
- Mode 101 (4-bit packed)

Writing is restricted to modes 0, 1, 2, and 12.

**Source:** `mrc_file.cpp:336-340, 465-469`

### Coordinate System Limitations

Only two axis mappings are handled without assertion:
1. Standard: MAPC=1, MAPR=2, MAPS=3
2. Swapped sections: MAPS=1, MAPC=3

Any other mapping triggers: `"Ooops, strange ordering of data in MRC file not yet supported"`

**Source:** `mrc_file.cpp:359-397`

### Integer Overflow Risk

- File offset calculations use `long` type for seek positions
- Data sizes computed as `(end_slice - start_slice + 1) * bytes_per_slice`
- Large volumes (e.g., 4096³ at 4 bytes/pixel = 256 GB) could theoretically overflow, but practical cisTEM datasets rarely exceed this

**Source:** `mrc_file.cpp:162-182, 414-418`

### IMOD Compatibility

cisTEMx respects IMOD format extensions:
- Checks `imodStamp == 1146047817` to detect IMOD-created files
- Uses `imodFlags[0] & 1` to determine if mode 0 (byte) data is signed
- Sets ISPG=1 (non-standard) for compatibility with IMOD workflows

**Source:** `mrc_header.cpp:227-233`

---

## Error Handling Strategy

### Assertions

Uses `MyDebugAssertTrue()` and `MyDebugAssertFalse()` for runtime validation:
- File must be open before operations
- Slice indices must be valid and ordered
- Data type conversions validated

**Source:** Throughout `mrc_file.cpp:153-156, 407-410`

### Failures

Fatal errors trigger `DEBUG_ABORT`:
- Unsupported data mode
- File opening failure
- Invalid axis mapping

**Source:** `mrc_header.cpp:310, mrc_file.cpp:102-104, 117-118, 395-396`

### Graceful Degradation

- If file open fails with write mode, retries with read-only
- Returns `false` on validation failures where possible
- Supports "do nothing" mode for /dev/null

**Source:** `mrc_file.cpp:93-105`

---

## Typical Usage Patterns

### Pattern 1: Read entire volume

```cpp
MRCFile input_file(filename, false);
Image volume;
volume.ReadSlices(&input_file, 1, input_file.ReturnNumberOfSlices());
```

### Pattern 2: Write new file with header

```cpp
MRCFile output_file(filename, true);
output_file.my_header.SetDimensionsVolume(256, 256, 256);
output_file.my_header.SetPixelSize(1.5);
output_file.WriteHeader();

// Write data slices...
for (int slice = 1; slice <= 256; slice++) {
    Image slice_data;
    // ... populate slice_data
    slice_data.WriteSlice(&output_file, slice);
}
```

### Pattern 3: Stream processing large stacks

```cpp
MRCFile input_file(input_filename, false);
MRCFile output_file(output_filename, true);

// Configure output header
output_file.my_header.SetDimensionsImage(
    input_file.ReturnXSize(),
    input_file.ReturnYSize()
);
output_file.my_header.SetNumberOfImages(input_file.ReturnNumberOfSlices());
output_file.WriteHeader();

// Process slice by slice
for (int i = 1; i <= input_file.ReturnNumberOfSlices(); i++) {
    Image img;
    img.ReadSlice(&input_file, i);
    // ... process img
    img.WriteSlice(&output_file, i);
}

// Update header statistics after processing
output_file.SetPixelSize(input_file.ReturnPixelSize());
output_file.WriteHeader();
```

### Pattern 4: Handle both read and write files polymorphically

```cpp
// AbstractImageFile interface allows polymorphic handling
MRCFile mrc_file(filename, false);
// or
EerFile eer_file(filename, false);
// or
DMFile dm_file(filename, false);

// All support common interface
int x_size = file->ReturnXSize();
int num_slices = file->ReturnNumberOfSlices();
float* buffer = new float[x_size * y_size];
file->ReadSliceFromDisk(1, buffer);
```

---

## Summary: Key Design Characteristics

1. **Pointer-based header:** 1024-byte buffer with field pointers for efficient access
2. **Format agnostic:** Reads multiple formats, converts all to float internally
3. **Partial I/O:** Efficient reading/writing of slice ranges (not entire file)
4. **Integration:** Seamless with Image class for typical EM workflows
5. **Compatibility:** Respects IMOD extensions and non-standard axis mappings (where supported)
6. **Modern features:** FP16 support, assertion-based error handling, graceful degradation
7. **Limitations:** Write support limited to float32, float16, and lower-precision formats

---

## Files Referenced in Implementation

- IEEE 754 half-precision library: `include/ieee-754-half/half.hpp`
- Core headers: `core/core_headers.h` (includes all core functionality)
- Abstract base: `core/abstract_image_file.h`
- Image class: `core/image.h` (slice reading/writing methods)
- Utility functions: `core/functions.h`, `core/non_wx_functions.h`
- Example usage: `programs/multiply_two_stacks/multiply_two_stacks.cpp`
- Example usage: `programs/project3d/project3d.cpp`

---

**Research completed:** November 7, 2025
**Document version:** 1.0
**Coverage:** MRCFile and MRCHeader classes, complete I/O pipeline, coordinate systems, data type handling
