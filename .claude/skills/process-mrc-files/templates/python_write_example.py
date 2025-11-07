#!/usr/bin/env python3
"""
MRC File Writing Examples (Python mrcfile)

Demonstrates common patterns for writing MRC files in Python.
"""

import mrcfile
import numpy as np

# ============================================================================
# Example 1: Simple Write (Quick Method)
# ============================================================================

def write_simple(file_path, data, voxel_size=1.0):
    """
    Simplest way to write MRC file.

    Args:
        file_path: Output path
        data: NumPy array (2D or 3D)
        voxel_size: Pixel size in Ångströms (default: 1.0)
    """
    # Quick write - automatically sets all header fields
    mrcfile.write(file_path, data, voxel_size=voxel_size, overwrite=True)
    print(f"Wrote {file_path}: shape={data.shape}, voxel_size={voxel_size} Å")


# ============================================================================
# Example 2: Write with Full Header Control
# ============================================================================

def write_with_header(file_path, data, voxel_size=1.5):
    """Write MRC file with explicit header field control."""
    with mrcfile.new(file_path, overwrite=True) as mrc:
        # Set data (automatically updates NX, NY, NZ)
        mrc.set_data(data.astype(np.float32))

        # Set voxel size (updates XLEN, YLEN, ZLEN)
        mrc.voxel_size = voxel_size

        # Set as volume (ISPG=1)
        mrc.set_volume()

        # Update statistics (DMIN, DMAX, DMEAN, RMS)
        mrc.update_header_stats()

        # Set origin
        mrc.header.origin.x = 0.0
        mrc.header.origin.y = 0.0
        mrc.header.origin.z = 0.0

        # Set number of labels
        mrc.header.nlabl = 1
        # Set label text (80 characters max)
        mrc.set_label(b'Created by Python mrcfile', 0)

        print(f"Wrote {file_path} with full header control")


# ============================================================================
# Example 3: Write 3D Volume
# ============================================================================

def write_volume(file_path, volume_data, voxel_size=1.0):
    """
    Write 3D volume with proper ISPG setting.

    Args:
        file_path: Should use .mrc extension
        volume_data: 3D NumPy array (NZ, NY, NX)
        voxel_size: Pixel size in Ångströms
    """
    with mrcfile.new(file_path, overwrite=True) as mrc:
        # Ensure float32 for maximum compatibility
        mrc.set_data(volume_data.astype(np.float32))

        # Mark as 3D volume
        mrc.set_volume()  # Sets ISPG=1

        # Set voxel size
        mrc.voxel_size = voxel_size

        # Update statistics
        mrc.update_header_stats()

        print(f"Wrote 3D volume: {file_path}")
        print(f"  Shape: {mrc.data.shape}")
        print(f"  ISPG: {mrc.header.ispg}")


# ============================================================================
# Example 4: Write 2D Image Stack
# ============================================================================

def write_stack(file_path, stack_data, voxel_size=1.0):
    """
    Write 2D image stack (particle stack).

    Args:
        file_path: Should use .mrcs extension
        stack_data: 3D NumPy array (N_images, NY, NX)
        voxel_size: Pixel size in Ångströms
    """
    with mrcfile.new(file_path, overwrite=True) as mrc:
        # Ensure float32
        mrc.set_data(stack_data.astype(np.float32))

        # Mark as image stack
        mrc.set_image_stack()  # Sets ISPG=0

        # Set voxel size
        mrc.voxel_size = voxel_size

        # Update statistics
        mrc.update_header_stats()

        n_images = stack_data.shape[0]
        print(f"Wrote image stack: {file_path}")
        print(f"  Number of images: {n_images}")
        print(f"  Image size: {stack_data.shape[1]} × {stack_data.shape[2]}")
        print(f"  ISPG: {mrc.header.ispg}")


# ============================================================================
# Example 5: Write with Specific Data Type
# ============================================================================

def write_specific_dtype(file_path, data, dtype=np.float32):
    """
    Write MRC file with specific data type (MODE).

    Supported dtypes:
        np.int8 → MODE 0
        np.int16 → MODE 1
        np.float32 → MODE 2 (recommended)
        np.complex64 → MODE 4
        np.uint16 → MODE 6
        np.float16 → MODE 12
    """
    # Convert to desired type
    data_typed = data.astype(dtype)

    with mrcfile.new(file_path, overwrite=True) as mrc:
        mrc.set_data(data_typed)
        mrc.update_header_stats()

        mode_names = {np.int8: 'MODE 0 (int8)',
                      np.int16: 'MODE 1 (int16)',
                      np.float32: 'MODE 2 (float32)',
                      np.complex64: 'MODE 4 (complex64)',
                      np.uint16: 'MODE 6 (uint16)',
                      np.float16: 'MODE 12 (float16)'}

        print(f"Wrote {file_path} as {mode_names.get(dtype, dtype)}")
        print(f"  MODE: {mrc.header.mode}")


# ============================================================================
# Example 6: Write Compressed (FP16 for 50% Size Reduction)
# ============================================================================

def write_compressed(file_path, data, voxel_size=1.0):
    """
    Write as float16 (MODE 12) for 50% file size reduction.
    Acceptable for visualization; NOT for quantitative analysis.
    """
    with mrcfile.new(file_path, overwrite=True) as mrc:
        # Convert to float16
        data_fp16 = data.astype(np.float16)

        mrc.set_data(data_fp16)
        mrc.voxel_size = voxel_size
        mrc.update_header_stats()

        # Calculate file sizes
        size_f32 = data.size * 4 / 1e6  # MB
        size_f16 = data.size * 2 / 1e6  # MB

        print(f"Wrote compressed {file_path}")
        print(f"  MODE 12 (float16)")
        print(f"  Size reduction: {size_f32:.1f} MB → {size_f16:.1f} MB (50%)")
        print(f"  Precision: ~3 significant digits")


# ============================================================================
# Example 7: Incremental Writing (Build Stack Slice by Slice)
# ============================================================================

def write_incremental(file_path, n_slices, ny, nx, voxel_size=1.0):
    """
    Build MRC stack incrementally (memory efficient for large stacks).

    Args:
        file_path: Output path
        n_slices: Number of slices to write
        ny, nx: Image dimensions
        voxel_size: Pixel size
    """
    # Initialize with zeros
    initial_data = np.zeros((n_slices, ny, nx), dtype=np.float32)

    with mrcfile.new(file_path, overwrite=True) as mrc:
        mrc.set_data(initial_data)
        mrc.voxel_size = voxel_size
        mrc.set_image_stack()

        # Now write slices one by one
        for i in range(n_slices):
            # Generate or process slice data
            slice_data = np.random.randn(ny, nx).astype(np.float32)

            # Write to specific Z position
            mrc.data[i, :, :] = slice_data

            if (i + 1) % 100 == 0:
                print(f"  Written {i + 1}/{n_slices} slices")

        # Update header statistics at the end
        mrc.update_header_stats()

    print(f"Wrote {n_slices} slices incrementally to {file_path}")


# ============================================================================
# Example 8: Modifying Existing File
# ============================================================================

def modify_existing(file_path):
    """
    Modify data in existing MRC file (in-place editing).
    """
    with mrcfile.open(file_path, mode='r+') as mrc:
        print(f"Original shape: {mrc.data.shape}")
        print(f"Original mean: {np.mean(mrc.data):.3f}")

        # Modify data in-place
        mrc.data *= 2.0  # Scale by 2

        # IMPORTANT: Update header statistics after modifying data
        mrc.update_header_from_data()
        mrc.update_header_stats()

        print(f"Modified mean: {np.mean(mrc.data):.3f}")
        print("File updated in-place")


# ============================================================================
# Example 9: Write with Extended Header (SerialEM Format)
# ============================================================================

def write_with_extended_header(file_path, stack_data, tilt_angles):
    """
    Write stack with SerialEM-style extended header (tilt angles).

    Args:
        file_path: Output path
        stack_data: 3D array (n_tilts, ny, nx)
        tilt_angles: Array of tilt angles (degrees)
    """
    n_tilts = stack_data.shape[0]

    if len(tilt_angles) != n_tilts:
        raise ValueError(f"Tilt angles ({len(tilt_angles)}) must match slices ({n_tilts})")

    with mrcfile.new(file_path, overwrite=True) as mrc:
        mrc.set_data(stack_data.astype(np.float32))

        # Create SerialEM-style extended header
        # Each record is 1024 bytes for SerialEM
        ext_header_dtype = np.dtype([
            ('tilt_angle', 'f4'),  # Tilt angle in degrees
            ('stage_x', 'f4'),     # Stage X position
            ('stage_y', 'f4'),     # Stage Y position
            ('stage_z', 'f4'),     # Stage Z position
            ('reserved', 'f4', 252)  # Padding to 1024 bytes (252 floats)
        ])

        ext_header = np.zeros(n_tilts, dtype=ext_header_dtype)
        ext_header['tilt_angle'] = tilt_angles

        mrc.set_extended_header(ext_header)

        # Set extended header type
        mrc.header.exttyp = b'SERI'  # SerialEM format

        mrc.update_header_stats()

        print(f"Wrote {file_path} with SerialEM extended header")
        print(f"  Tilt angles: {tilt_angles}")


# ============================================================================
# Example 10: Write with Custom Origin and Axes
# ============================================================================

def write_with_coordinates(file_path, data, origin=(0, 0, 0), voxel_size=1.5):
    """
    Write MRC file with specified origin and coordinate system.

    Args:
        file_path: Output path
        data: NumPy array
        origin: (x, y, z) origin in Ångströms
        voxel_size: Pixel size in Ångströms
    """
    with mrcfile.new(file_path, overwrite=True) as mrc:
        mrc.set_data(data.astype(np.float32))
        mrc.voxel_size = voxel_size

        # Set origin
        mrc.header.origin.x = origin[0]
        mrc.header.origin.y = origin[1]
        mrc.header.origin.z = origin[2]

        # Ensure standard axis ordering
        mrc.header.mapc = 1  # X = columns
        mrc.header.mapr = 2  # Y = rows
        mrc.header.maps = 3  # Z = sections

        mrc.update_header_stats()

        print(f"Wrote {file_path}")
        print(f"  Origin: ({origin[0]}, {origin[1]}, {origin[2]}) Å")
        print(f"  Voxel size: {voxel_size} Å")
        print(f"  Axis mapping: MAPC={mrc.header.mapc}, MAPR={mrc.header.mapr}, MAPS={mrc.header.maps}")


# ============================================================================
# Example 11: Safe Write with Validation
# ============================================================================

def write_safe(file_path, data, voxel_size=1.0):
    """
    Write MRC file with pre-write validation.
    """
    # Validate data
    if data.ndim not in [2, 3]:
        raise ValueError(f"Data must be 2D or 3D, got {data.ndim}D")

    if data.size == 0:
        raise ValueError("Data is empty")

    # Check for NaN/Inf
    if np.any(np.isnan(data)):
        print("Warning: Data contains NaN values - replacing with 0")
        data = np.nan_to_num(data, nan=0.0)

    if np.any(np.isinf(data)):
        print("Warning: Data contains Inf values - clipping")
        data = np.clip(data, -1e38, 1e38)

    # Write
    try:
        with mrcfile.new(file_path, overwrite=True) as mrc:
            mrc.set_data(data.astype(np.float32))
            mrc.voxel_size = voxel_size
            mrc.update_header_stats()

            # Validate written file
            if mrcfile.validate(file_path):
                print(f"✓ Successfully wrote and validated {file_path}")
            else:
                print(f"⚠ File written but has non-standard header: {file_path}")

    except Exception as e:
        print(f"✗ Error writing file: {e}")
        raise


# ============================================================================
# Main: Run Examples
# ============================================================================

if __name__ == '__main__':
    # Generate test data
    test_data_2d = np.random.randn(256, 256).astype(np.float32)
    test_data_3d = np.random.randn(10, 256, 256).astype(np.float32)

    print("=" * 70)
    print("Example 1: Simple Write")
    print("=" * 70)
    # write_simple('output_simple.mrc', test_data_3d, voxel_size=1.5)

    print("\n" + "=" * 70)
    print("Example 3: Write 3D Volume")
    print("=" * 70)
    # write_volume('output_volume.mrc', test_data_3d, voxel_size=1.0)

    print("\n" + "=" * 70)
    print("Example 4: Write 2D Image Stack")
    print("=" * 70)
    # write_stack('output_stack.mrcs', test_data_3d, voxel_size=1.0)

    print("\n" + "=" * 70)
    print("Example 6: Write Compressed (FP16)")
    print("=" * 70)
    # write_compressed('output_compressed.mrc', test_data_3d, voxel_size=1.0)

    print("\nTo use these examples, uncomment the function calls above")
