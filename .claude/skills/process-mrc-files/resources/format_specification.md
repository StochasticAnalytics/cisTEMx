# MRC File Format Specification Research

## Overview

The MRC (Medical Research Council) file format is the standard binary format for storing three-dimensional image and volume data in the structural biology and electron microscopy fields. The format has evolved through several versions, with MRC2014 being the current standardized specification maintained by the CCP-EM (CCP-Electron Microscopy) consortium.

**Key Publication:**
- **Authors:** Cheng, A., Henderson, R., Mastronarde, D., Ludtke, S.J., Schoenmakers, R.H.M., Short, J., Marabini, R., Dallakyan, S., Agard, D., and Winn, M.
- **Title:** "MRC2014: Extensions to the MRC format header for electron cryo-microscopy and tomography"
- **Journal:** Journal of Structural Biology
- **Volume:** 192(2)
- **Pages:** 146-150
- **Published:** 2015
- **DOI:** 10.1016/j.jsb.2015.04.002
- **PubMed ID:** 25882513
- **PMC ID:** PMC4642651

**Official Specifications:**
- CCP-EM MRC Format Page: https://www.ccpem.ac.uk/mrc-format/mrc2014/
- CCP-EM General MRC Format: https://www.ccpem.ac.uk/mrc-format/
- IMOD Documentation: https://bio3d.colorado.edu/imod/doc/mrc_format.txt
- mrcfile Python Library Documentation: https://mrcfile.readthedocs.io/

---

## File Structure

### Overall Organization

An MRC file consists of three sequential components:

1. **Fixed Header** (1024 bytes)
   - 224 bytes of structured header fields (56 4-byte words)
   - 800 bytes of text label space (10 labels × 80 bytes each)

2. **Extended Header** (optional)
   - Variable size, indicated by NSYMBT field in fixed header
   - Contains structured metadata (SERI, FEI1, FEI2, AGAR types)
   - Not always present or may be zero-length

3. **Data Array** (uncompressed binary)
   - Raw voxel/pixel data organized by NX, NY, NZ dimensions
   - Data type determined by MODE field
   - Endianness determined by machine stamp field

### Header Layout

**Fixed Header (1024 bytes):**

```
Offset (bytes)  | Offset (words) | Field Name           | Type      | Size
0-3             | 0              | NX                   | int32     | 4
4-7             | 1              | NY                   | int32     | 4
8-11            | 2              | NZ                   | int32     | 4
12-15           | 3              | MODE                 | int32     | 4
16-19           | 4              | NXSTART              | int32     | 4
20-23           | 5              | NYSTART              | int32     | 4
24-27           | 6              | NZSTART              | int32     | 4
28-31           | 7              | MX                   | int32     | 4
32-35           | 8              | MY                   | int32     | 4
36-39           | 9              | MZ                   | int32     | 4
40-43           | 10             | XLEN                 | float32   | 4
44-47           | 11             | YLEN                 | float32   | 4
48-51           | 12             | ZLEN                 | float32   | 4
52-55           | 13             | ALPHA                | float32   | 4
56-59           | 14             | BETA                 | float32   | 4
60-63           | 15             | GAMMA                | float32   | 4
64-67           | 16             | MAPC                 | int32     | 4
68-71           | 17             | MAPR                 | int32     | 4
72-75           | 18             | MAPS                 | int32     | 4
76-79           | 19             | DMIN                 | float32   | 4
80-83           | 20             | DMAX                 | float32   | 4
84-87           | 21             | DMEAN                | float32   | 4
88-91           | 22             | ISPG                 | int32     | 4
92-95           | 23             | NSYMBT               | int32     | 4
96-99           | 24             | EXTRA (unused)       | int32     | 4
100-103         | 25             | EXTRA (unused)       | int32     | 4
104-107         | 26             | EXTTYPE              | char[4]   | 4
108-111         | 27             | NVERSION             | int32     | 4
112-115         | 28             | IMODSTAMP            | int32     | 4
116-119         | 29             | IDTYPE (NINT)        | int32     | 4
120-123         | 30             | BLANK                | int32     | 4
124-127         | 31             | IMODTAG (NREAL)      | int32     | 4
128-131         | 32             | JDTYPE               | int32     | 4
132-135         | 33             | BLANK                | int32     | 4
136-139         | 34             | IMODCH2 (NZFACT)     | int32     | 4
140-143         | 35             | BLANK                | int32     | 4
144-147         | 36             | BLANK                | int32     | 4
148-151         | 37             | BLANK                | int32     | 4
152-155         | 38             | BLANK                | int32     | 4
156-159         | 39             | BLANK                | int32     | 4
160-163         | 40             | BLANK                | int32     | 4
164-167         | 41             | BLANK                | int32     | 4
168-171         | 42             | BLANK                | int32     | 4
172-175         | 43             | BLANK                | int32     | 4
176-179         | 44             | BLANK                | int32     | 4
180-183         | 45             | BLANK                | int32     | 4
184-187         | 46             | BLANK                | int32     | 4
188-191         | 47             | BLANK                | int32     | 4
192-195         | 48             | XORG                 | float32   | 4
196-199         | 49             | YORG                 | float32   | 4
200-203         | 50             | ZORG                 | float32   | 4
204-207         | 51             | CMAP                 | char[4]   | 4
208-211         | 52             | STAMP (Machine)      | int32     | 4
212-215         | 53             | RMS                  | float32   | 4
216-219         | 54             | NLABL                | int32     | 4
220-223         | 55             | NLABEL               | int32     | 4
224-1023        | -              | LABEL[10][80]        | char      | 800
```

---

## Header Field Descriptions

### Dimension Fields

**NX, NY, NZ** (Words 0-2)
- Number of columns, rows, and sections (the 3D grid dimensions)
- NX: number of columns (fastest changing in file storage)
- NY: number of rows
- NZ: number of sections (slowest changing in file storage)
- Data type: int32
- Units: pixel count
- Required: Yes

**NXSTART, NYSTART, NZSTART** (Words 4-6)
- Starting indices for the sub-region of the data
- Indicates where this sub-volume begins in the larger crystal/map
- Data type: int32
- Units: pixels
- Default: 0 (if no sub-region)

### Data Type and Organization

**MODE** (Word 3)
- Specifies the data type of the pixel/voxel values
- Data type: int32
- **Valid modes:**
  - **0:** Signed 8-bit integers (int8, range: -128 to 127)
    - Clarified as signed in MRC2014
  - **1:** Signed 16-bit integers (int16, 2 bytes per voxel)
  - **2:** 32-bit IEEE floating-point (float32)
  - **3:** Complex 16-bit integers (pairs of int16, 4 bytes per voxel)
  - **4:** Complex 32-bit floating-point (pairs of float32, 8 bytes per voxel)
  - **6:** Unsigned 16-bit integers (uint16, 2 bytes per voxel)
    - Added in MRC2014, recommended for new files requiring unsigned integers
  - **12:** 16-bit floating-point (float16, IEEE 754 half-precision)
    - Proposed extension, not yet widely supported
  - **16:** RGB data (3 × uint8 per voxel) - IMOD extension
  - **101:** 4-bit data - IMOD extension

### Axis Mapping and Coordinate System

**MAPC, MAPR, MAPS** (Words 16-18)
- Control the orientation of the data grid relative to the coordinate system
- Each field specifies which axis (X, Y, Z) corresponds to columns (C), rows (R), and sections (S)
- Data type: int32
- Valid values: 1 (X-axis), 2 (Y-axis), 3 (Z-axis)
- **Standard EM/ET convention:** MAPC=1, MAPR=2, MAPS=3
  - Columns map to X (fastest changing)
  - Rows map to Y
  - Sections map to Z (slowest changing)
  - Images/sections are perpendicular to the Z-axis
- **Crystallography:** Other orderings possible depending on space group

**Note on axis conventions:**
- Data is accessed in C/Python style: `data[z][y][x]`
- First index (Z) is slowest-changing axis
- Last index (X) is fastest-changing axis
- Pixel origin is lower-left corner, looking down on the volume
- First NX values in file represent the lowest Y line of the first image

### Space Group and Data Organization

**ISPG** (Word 22)
- Space group number indicating the type and organization of data
- Data type: int32
- **Electron microscopy/tomography conventions:**
  - **0:** 2D images or image stacks (default for non-crystallographic data)
  - **1:** Single 3D volume (EM volume or electron tomography reconstruction)
  - **401-630:** Volume stacks (4D data - stack of 3D volumes)
    - Convention: ISPG = space_group_number + 400
    - Value 401 most common for EM volume stacks
- **Crystallography:** Space groups 1-230 used for crystal symmetry

### Origin and Coordinate Mapping

**XORG, YORG, ZORG** (Words 48-50)
- Location of the coordinate system origin relative to the first pixel
- Units: **pixel spacing units**, NOT pixels (normalized by voxel dimensions)
- Typical interpretation: "where is the coordinate origin relative to the corner of the first voxel?"
- Data type: float32
- Common practice: Set to (0, 0, 0) if origin coincides with the first pixel
- **Important note:** These fields may be swapped (ZORG, XORG, YORG) in older format variants

**NXSTART, NYSTART, NZSTART** (Words 4-6)
- Alternative origin specification in some software
- Used in CCP4 (crystallography) files
- Different from XORG/YORG/ZORG

**Confusing aspect:** Chimera and other software may use different origin fields depending on file type (XORG/YORG/ZORG for cryo-EM or NXSTART/NYSTART/NZSTART for CCP4)

### Cell Parameters

**XLEN, YLEN, ZLEN** (Words 10-12)
- Unit cell dimensions in Ångströms (Å)
- Data type: float32
- Used in crystallography for mapping physical coordinates
- In EM/ET, represents the size of the entire data volume in Ångströms

**ALPHA, BETA, GAMMA** (Words 13-15)
- Unit cell angles in degrees
- Data type: float32
- ALPHA: angle between Y and Z axes
- BETA: angle between X and Z axes
- GAMMA: angle between X and Y axes
- Orthogonal cells have all angles = 90 degrees

**MX, MY, MZ** (Words 7-9)
- Intervals/sampling along each axis
- Data type: int32
- Represents the number of intervals in the unit cell for each dimension
- Often equals NX, NY, NZ for simple cases

### Statistical Information

**DMIN, DMAX, DMEAN** (Words 19-21)
- Density minimum, maximum, and mean values in the data
- Data type: float32
- Derived from actual data (should be updated when data changes)
- Used for display scaling and statistics

**RMS** (Word 53)
- Root mean square deviation of density from mean
- Data type: float32
- Useful for contrast adjustment and normalization

### Version and Format Identification

**NVERSION** (Word 27)
- MRC format version identifier
- Data type: int32
- Calculated as: Year × 10 + version_within_year (base 0)
- **Standard values:**
  - **20140:** Original MRC2014 format (2014, version 0)
  - **20141:** MRC2014 with updates (2014, version 1)
- Critical for determining format compatibility

**CMAP** (Bytes 204-207, Word 51)
- 4-byte character field containing format identifier
- Standard value: "MAP " (with trailing space)
- Used to validate file format

### Extended Header Information

**NSYMBT** (Word 23)
- Number of bytes in extended header
- Data type: int32
- Value of 0 means no extended header
- Size must be divisible by NINT × 4 + NREAL × 4 (for Agard-style headers)

**EXTTYPE** (Bytes 104-107, Word 26)
- 4-character code identifying the type of structured metadata in extended header
- Data type: char[4]
- **Known types:**
  - **"SERI":** SerialEM format (tilt angles, coordinates, microscope parameters)
  - **"FEI1":** FEI microscope extended header (first version)
  - **"FEI2":** FEI microscope extended header (second version)
  - **"AGAR":** Agard-style header (generic format with configurable fields)
- If not set to a recognized code, interpreted as generic/unstructured extended header

**IDTYPE** or **NINT** (Bytes 116-119, Word 29)
- For Agard-style headers: number of 32-bit integers per extended header record
- Helps parse extended header structure

**IMODTAG** or **NREAL** (Bytes 124-127, Word 31)
- For Agard-style headers: number of 32-bit floats per extended header record
- Helps parse extended header structure

**IMODSTAMP** (Bytes 112-115, Word 28)
- Specific to IMOD files
- Value: 0x444F4D49 ("IMOD" in hex)
- Indicates file was created by IMOD software

### Endianness Detection

**Machine Stamp** (Bytes 212-215, Word 54)
- Used to detect byte order (endianness) of the file
- Standard values indicate system byte order:
  - **0x11110000 (little-endian):** Bytes 212-213 contain 0x44 0x44 (68, 68)
    - Also 0x44 0x41 (68, 65) is valid
  - **0x00000011 (big-endian):** Bytes 212-213 contain 0x11 0x11 (17, 17)
- Critical for correct interpretation of all multi-byte values
- Standard convention: First two bytes are 0x44 0x44 (little-endian) or 0x11 0x11 (big-endian)

### Text Labels

**NLABL** (Word 54)
- Number of text labels in header
- Data type: int32
- Maximum: 10 labels

**NLABEL** (Word 55)
- Not commonly used

**LABEL[10][80]** (Bytes 224-1023)
- Array of 10 text labels, each 80 bytes
- ASCII strings
- Used for user annotations and metadata
- Null-terminated or space-padded

---

## Extended Header Types

### Agard-Style Header

**Format:** Generic extensible format with configurable fields
- Each extended header record has fixed size
- Record size = (NINT × 4 bytes) + (NREAL × 4 bytes)
- Number of records = NSYMBT / record_size
- One record per image/slice (for stacks) or single record (for volumes)

**Structure:**
- First NINT × 4 bytes: 32-bit integers
- Next NREAL × 4 bytes: 32-bit floats

### SerialEM Extended Header ("SERI")

**Format:** Series of short integers per image
- Variable field structure specified by bit flags
- Typical fields: tilt angles, X/Y coordinates, microscope parameters
- Field presence indicated by flags in Word 33 (NINT and NREAL values)
- Data size and field flags stored in two 2-byte integers at the beginning of each record

### FEI Extended Header ("FEI1", "FEI2")

**Format:** Structured records, typically 768 bytes per image
- Contains detailed acquisition metadata from FEI microscopes
- Fields include: stage positions, detector information, calibration data, timing data
- Standard size: NSYMBT is often 131072 bytes (large structured metadata blocks)
- Two versions exist (FEI1 and FEI2) with different field layouts

---

## Data Organization and Storage

### Byte Order (Endianness)

- **Little-endian** is the standard modern convention (Intel x86, ARM)
  - Machine stamp bytes: 0x44 0x44 or 0x44 0x41
  - Most modern software defaults to little-endian
- **Big-endian** was used on older systems (Motorola, SPARC)
  - Machine stamp bytes: 0x11 0x11
- **Detection:** Read machine stamp to determine actual byte order on disk

### Voxel Storage Order

**Row-major (C-style) ordering:**
- Slowest-changing: Z (sections)
- Middle: Y (rows)
- Fastest-changing: X (columns)
- First voxel in file: lower-left corner of first section (looking down)
- First NX values: lowest Y line of first section

### Compression

- **Native MRC format:** Uncompressed binary only
- **Modern extension:** Transparent compression via filename extensions
  - `.mrc.gz`: gzip compression
  - `.mrc.bz2`: bzip2 compression
  - Automatically detected and decompressed by modern libraries (mrcfile)

---

## Format Evolution and Variants

### MRC (Original, Pre-2000)

**Early format without standardized endianness detection**
- 1024-byte header, but structure varied
- No machine stamp for endianness detection
- Limited extended header support

### MRC2000

**Major standardization effort (2000)**
- Added machine stamp (bytes 212-215) for endianness detection
- Standardized header field meanings across research groups
- Merger of CCP4 (crystallography) and LMB (electron microscopy) variants
- Established compatibility between different software packages
- Foundation for modern MRC format

### IMOD MRC (Extended from MRC2000)

**Electron microscopy specific variant**
- Uses IMODSTAMP field (0x444F4D49) at bytes 112-115
- Uses IMODTAG field for Agard-style extended header structure
- SerialEM common extended header type ("SERI")
- Additional MODE values (16 for RGB, 101 for 4-bit)

### CCP4/Crystallography Variant

**Crystallography specific variant**
- Uses NXSTART/NYSTART/NZSTART for origin specification
- Uses space groups 1-230 for crystal symmetry
- CELLA/CELLB fields for unit cell parameters

### FEI MRC (FEI Instruments)

**FEI microscope software variant**
- Often assumes MODE 6 (unsigned int16) for future compatibility
- FEI1 and FEI2 extended header types
- Large extended header blocks (typically 131072 bytes)
- Detailed acquisition metadata from FEI microscope systems

### MRC2014 (Current Standard, 2015)

**Comprehensive standardization and extension**
- Unified multiple variants into single standard
- Clarified MODE 0 as signed (not unsigned)
- Added MODE 6 officially for unsigned 16-bit integers
- Introduced NVERSION field for explicit version tracking
- Extended header type identification (EXTTYPE field)
- Accommodates SerialEM, FEI, and other specialized extended headers
- Published in Journal of Structural Biology as peer-reviewed standard
- Maintains backward compatibility with MRC2000

---

## Critical Specifications for Developers

### Data Type Interpretation

**Must implement support for:**
- MODE 0: signed int8
- MODE 1: signed int16
- MODE 2: float32 (IEEE 754)
- MODE 4: complex float32 (pairs)
- MODE 6: unsigned int16 (MRC2014 addition)

**Should consider supporting:**
- MODE 3: complex int16 (rarely used)
- MODE 12: float16 (IEEE 754 half-precision, experimental)
- IMOD extensions (MODE 16, 101)

### Endianness Handling

1. Read machine stamp at bytes 212-215
2. Check first two bytes:
   - 0x44, 0x44 (or 0x44, 0x41): little-endian
   - 0x11, 0x11: big-endian
3. Apply byte-swapping to ALL multi-byte values if system endianness differs from file endianness
4. This includes: NX, NY, NZ, MODE, all float32 and int32 fields

### Axis and Coordinate Conventions

**For EM/ET (MAPC=1, MAPR=2, MAPS=3):**
- Data layout: `data[z][y][x]` in C/Python indexing
- Z is slowest-changing, X is fastest-changing
- Pixel origin is lower-left corner of first image
- XORG, YORG, ZORG: origin in physical units (Ångströms)

**Coordinate transformation:**
```
pixel_index = [z, y, x]  (0-indexed)
world_coord = [x, y, z] * voxel_spacing + [xorg, yorg, zorg]
```

### Version Detection and Compatibility

- Always check NVERSION field to determine format version
- NVERSION = 20140 or 20141 indicates MRC2014 compliance
- For older files, check for presence of CMAP="MAP "
- Use EXTTYPE to identify specialized extended headers

### Extended Header Handling

**If NSYMBT > 0:**
1. Check EXTTYPE field:
   - Known types (SERI, FEI1, FEI2, AGAR): parse according to type specifications
   - Unknown types: treat as unstructured binary data
2. Read extended header immediately after 1024-byte fixed header
3. Size = NSYMBT bytes
4. For image stacks: may contain multiple records (one per image)

---

## Important Implementation Considerations

### Backward Compatibility

- Must support older MRC2000 files (NVERSION not set)
- Many scientific archives contain pre-2014 format files
- Graceful degradation for unsupported extended header types
- Conservative approach: if uncertain, treat extended header as opaque binary

### Data Validation

- Verify CMAP="MAP " (or detect by machine stamp if CMAP unreliable)
- Check MODE is in valid range (0, 1, 2, 3, 4, 6, 12, or software-specific)
- Validate NX, NY, NZ are positive and reasonable
- Calculate expected file size: 1024 + NSYMBT + (NX × NY × NZ × bytes_per_voxel)
- Verify actual file size matches expected size

### File Type Detection

**Detection strategy:**
1. Check for 1024-byte header (fixed size)
2. Read machine stamp (bytes 212-215)
3. Check CMAP field for "MAP "
4. If IMODSTAMP (0x444F4D49) present: IMOD variant
5. If EXTTYPE is recognized: use type-specific parser
6. Default to MRC2000 if standard features present

### Common Pitfalls

1. **Forgetting byte order conversion:** Most modern files are little-endian, but some (especially older SPARC/Motorola systems) are big-endian
2. **Incorrect axis ordering:** Mixing up C-style (Z slowest) with Fortran-style (Z fastest) indexing
3. **Origin confusion:** Different software uses different origin specifications (XORG vs NXSTART)
4. **Extended header assumptions:** Assuming extended header structure without checking EXTTYPE
5. **Unsigned vs signed:** MODE 0 was historically ambiguous; MRC2014 clarifies as signed
6. **Voxel spacing units:** XORG/YORG/ZORG are in physical units (Ångströms), not pixels

---

## References and Additional Resources

### Official Documentation

1. **CCP-EM MRC Format Page:** https://www.ccpem.ac.uk/mrc-format/
   - Links to all format versions (MRC2000, MRC2014)
   - Official specifications and updates

2. **CCP-EM MRC2014 Page:** https://www.ccpem.ac.uk/mrc-format/mrc2014/
   - Current standard documentation
   - Links to peer-reviewed paper

3. **IMOD Documentation:** https://bio3d.colorado.edu/imod/doc/mrc_format.txt
   - Comprehensive technical reference
   - Historical context and variants

### Academic Publication

4. **Cheng et al. (2015):** MRC2014: Extensions to the MRC format header for electron cryo-microscopy and tomography
   - Journal of Structural Biology, 192(2):146-150
   - DOI: 10.1016/j.jsb.2015.04.002
   - PubMed: https://pubmed.ncbi.nlm.nih.gov/25882513/
   - PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC4642651/

### Reference Implementations

5. **mrcfile (Python):** https://mrcfile.readthedocs.io/
   - CCP-EM official Python library
   - Reference implementation for MRC2014
   - Supports compression, validation, metadata handling

6. **IMOD (C++):** https://bio3d.colorado.edu/imod/
   - Electron microscopy image processing suite
   - Comprehensive MRC support with extended headers

### Historical Documentation

7. **CCP-EM MRC2000 Page:** https://www.ccpem.ac.uk/mrc-format/mrc2000.php
   - Previous standard documentation
   - Historical context for format evolution

8. **File Format Archive:** http://fileformats.archiveteam.org/wiki/MRC
   - Community-maintained format documentation
   - Preservation-focused perspective

---

## Research Notes

This research was conducted to understand the MRC file format specification for potential integration with the cisTEMx project. The information compiled represents:

- Official specifications from CCP-EM (leading authority in the field)
- Peer-reviewed academic publication (Cheng et al. 2015)
- Reference implementations from established electron microscopy software
- Practical implementation guidance from mrcfile library documentation

**Key takeaways for cisTEMx development:**

1. MRC is the standard format in the electron microscopy community
2. MRC2014 is the current standard (2015 publication, well-established)
3. Backward compatibility with MRC2000 files is essential
4. Extended headers require careful handling for different microscope systems
5. Endianness detection is critical but straightforward
6. Axis ordering conventions are standardized for EM (MAPC=1, MAPR=2, MAPS=3)
7. Origin specifications may vary between software packages (XORG/YORG/ZORG vs NXSTART/NYSTART/NZSTART)
8. The Python mrcfile library provides reference implementations for all features

**Critical for robust implementation:**
- Proper endianness conversion
- Support for both signed (MODE 0, MRC2000) and clarified signed (MODE 0, MRC2014)
- Extended header detection and graceful handling of unknown types
- Validation of header fields before data access
- Clear documentation of coordinate system conventions used

---

## Document Information

**Created:** 2025-11-07
**Research Scope:** MRC File Format Specification (MRC2014)
**Status:** Research compilation (skills development phase)
**Next Steps:** Use this research to develop MRC format reading/writing skill
