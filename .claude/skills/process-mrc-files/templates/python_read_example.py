#!/usr/bin/env python3
"""
MRC File Reading Examples (Python mrcfile)

Demonstrates common patterns for reading MRC files in Python.
"""

import mrcfile
import numpy as np

# ============================================================================
# Example 1: Basic File Reading
# ============================================================================

def read_basic(file_path):
    """Read entire MRC file into memory."""
    with mrcfile.open(file_path, mode='r') as mrc:
        # Access data as NumPy array
        data = mrc.data  # Shape: (NZ, NY, NX)

        # Access header fields
        nx, ny, nz = mrc.header.nx, mrc.header.ny, mrc.header.nz
        mode = mrc.header.mode
        voxel_size = mrc.voxel_size.x  # Assumes isotropic

        print(f"Dimensions: {nx} × {ny} × {nz}")
        print(f"Mode: {mode}")
        print(f"Voxel size: {voxel_size} Å")
        print(f"Data shape: {data.shape}")
        print(f"Data type: {data.dtype}")

        return data.copy()  # Return copy to use after file closes


# ============================================================================
# Example 2: Reading Specific Slices (Memory Efficient)
# ============================================================================

def read_slices(file_path, start_slice, end_slice):
    """Read only specific Z-slices from a stack."""
    with mrcfile.open(file_path, mode='r') as mrc:
        # Python slicing: data[start:end, :, :]
        # Note: end is exclusive
        slice_data = mrc.data[start_slice:end_slice+1, :, :]

        print(f"Read slices {start_slice}-{end_slice}")
        print(f"Shape: {slice_data.shape}")

        return slice_data.copy()


# ============================================================================
# Example 3: Memory-Mapped Reading (Large Files)
# ============================================================================

def read_mmap(file_path):
    """
    Memory-mapped reading for files too large to fit in RAM.
    Data is loaded on-demand when accessed.
    """
    with mrcfile.mmap(file_path, mode='r') as mrc:
        # Data is NOT loaded into memory yet
        print(f"File shape: {mrc.data.shape}")
        print(f"File size: {mrc.data.nbytes / 1e9:.2f} GB")

        # Access specific region - only this loads into memory
        region = mrc.data[100:110, 500:600, 500:600]
        print(f"Loaded region shape: {region.shape}")

        # Process region
        mean_val = np.mean(region)
        print(f"Region mean: {mean_val}")

        return region.copy()


# ============================================================================
# Example 4: Reading Header Information Only
# ============================================================================

def read_header_only(file_path):
    """Read header metadata without loading data array."""
    with mrcfile.open(file_path, mode='r', header_only=True) as mrc:
        # Data is NOT loaded, only header

        # Dimensions
        print(f"NX={mrc.header.nx}, NY={mrc.header.ny}, NZ={mrc.header.nz}")

        # Data type
        mode_names = {0: 'int8', 1: 'int16', 2: 'float32',
                      4: 'complex64', 6: 'uint16', 12: 'float16'}
        print(f"MODE: {mrc.header.mode} ({mode_names.get(mrc.header.mode, 'unknown')})")

        # Pixel size
        print(f"Voxel size: {mrc.voxel_size.x:.3f} Å")

        # Statistics
        print(f"Min: {mrc.header.dmin:.3f}, Max: {mrc.header.dmax:.3f}")
        print(f"Mean: {mrc.header.dmean:.3f}, RMS: {mrc.header.rms:.3f}")

        # File type
        ispg = mrc.header.ispg
        if ispg == 0:
            print("Type: 2D image or stack")
        elif ispg == 1:
            print("Type: 3D volume")
        elif ispg >= 401:
            print("Type: 4D volume stack")

        # Axis mapping
        print(f"Axis mapping: MAPC={mrc.header.mapc}, MAPR={mrc.header.mapr}, MAPS={mrc.header.maps}")


# ============================================================================
# Example 5: Reading 2D Image
# ============================================================================

def read_2d_image(file_path):
    """Read a single 2D image (NZ=1)."""
    with mrcfile.open(file_path, mode='r') as mrc:
        if mrc.data.ndim == 3 and mrc.data.shape[0] == 1:
            # Shape is (1, NY, NX) - squeeze to (NY, NX)
            image = np.squeeze(mrc.data)
        elif mrc.data.ndim == 2:
            # Already 2D
            image = mrc.data
        else:
            raise ValueError(f"Expected 2D image, got shape {mrc.data.shape}")

        print(f"2D image shape: {image.shape}")
        return image.copy()


# ============================================================================
# Example 6: Reading Particle Stack
# ============================================================================

def read_particle_stack(file_path, particle_indices=None):
    """
    Read particle stack (.mrcs file).

    Args:
        file_path: Path to .mrcs file
        particle_indices: List of particle indices to read (0-indexed),
                         or None to read all
    """
    with mrcfile.open(file_path, mode='r') as mrc:
        n_particles = mrc.data.shape[0]
        print(f"Stack contains {n_particles} particles")

        if particle_indices is None:
            # Read all particles
            particles = mrc.data.copy()
        else:
            # Read specific particles
            particles = mrc.data[particle_indices, :, :].copy()
            print(f"Read {len(particle_indices)} particles")

        return particles


# ============================================================================
# Example 7: Reading with Validation
# ============================================================================

def read_with_validation(file_path):
    """Read MRC file with format validation."""
    # First validate the file
    if not mrcfile.validate(file_path):
        print(f"Warning: {file_path} has non-standard header")
        # Use permissive mode for non-compliant files
        with mrcfile.open(file_path, mode='r', permissive=True) as mrc:
            data = mrc.data.copy()
    else:
        # Standard compliant file
        with mrcfile.open(file_path, mode='r') as mrc:
            data = mrc.data.copy()

    return data


# ============================================================================
# Example 8: Reading Extended Header
# ============================================================================

def read_extended_header(file_path):
    """Read extended header metadata (e.g., SerialEM, FEI data)."""
    with mrcfile.open(file_path, mode='r') as mrc:
        # Check if extended header exists
        nsymbt = mrc.header.nsymbt
        print(f"Extended header size: {nsymbt} bytes")

        if nsymbt > 0:
            # Check extended header type
            exttype = mrc.header.exttyp.tobytes().decode('ascii', errors='ignore')
            print(f"Extended header type: '{exttype}'")

            # Access extended header
            ext_header = mrc.extended_header

            if exttype == 'SERI':
                # SerialEM format - typically contains tilt angles
                print("SerialEM extended header detected")
                print(f"Extended header shape: {ext_header.shape}")
                # SerialEM has fields like 'tilt_angle', 'stage_x', 'stage_y'
                if 'tilt_angle' in ext_header.dtype.names:
                    tilt_angles = ext_header['tilt_angle']
                    print(f"Tilt angles: {tilt_angles}")

            elif exttype.startswith('FEI'):
                # FEI microscope metadata
                print(f"FEI extended header detected: {exttype}")

            else:
                print("Unknown or generic extended header type")
                print(f"Extended header dtype: {ext_header.dtype}")


# ============================================================================
# Example 9: Reading Coordinate Information
# ============================================================================

def read_coordinate_info(file_path):
    """Read origin and coordinate system information."""
    with mrcfile.open(file_path, mode='r') as mrc:
        # Origin (in Ångströms)
        origin_x = mrc.header.origin.x
        origin_y = mrc.header.origin.y
        origin_z = mrc.header.origin.z
        print(f"Origin: ({origin_x}, {origin_y}, {origin_z}) Å")

        # Voxel size
        voxel_x = mrc.voxel_size.x
        voxel_y = mrc.voxel_size.y
        voxel_z = mrc.voxel_size.z
        print(f"Voxel size: ({voxel_x}, {voxel_y}, {voxel_z}) Å")

        # Physical dimensions
        nx, ny, nz = mrc.header.nx, mrc.header.ny, mrc.header.nz
        phys_x = nx * voxel_x
        phys_y = ny * voxel_y
        phys_z = nz * voxel_z
        print(f"Physical size: ({phys_x:.1f}, {phys_y:.1f}, {phys_z:.1f}) Å")

        # Axis mapping
        mapc = mrc.header.mapc
        mapr = mrc.header.mapr
        maps = mrc.header.maps
        print(f"Axis mapping: MAPC={mapc}, MAPR={mapr}, MAPS={maps}")

        if (mapc, mapr, maps) == (1, 2, 3):
            print("Standard axis ordering: X=columns, Y=rows, Z=sections")
        else:
            print(f"Non-standard axis ordering: columns→{mapc}, rows→{mapr}, sections→{maps}")


# ============================================================================
# Example 10: Type-Safe Reading with Error Handling
# ============================================================================

def read_safe(file_path):
    """
    Read MRC file with comprehensive error handling and validation.
    """
    try:
        with mrcfile.open(file_path, mode='r') as mrc:
            # Validate dimensions
            nx, ny, nz = mrc.header.nx, mrc.header.ny, mrc.header.nz
            if nx <= 0 or ny <= 0 or nz < 0:
                raise ValueError(f"Invalid dimensions: {nx}×{ny}×{nz}")

            # Validate MODE
            mode = mrc.header.mode
            valid_modes = [0, 1, 2, 4, 6, 12]
            if mode not in valid_modes:
                raise ValueError(f"Unsupported MODE: {mode}")

            # Check data size consistency
            expected_size = nx * ny * max(nz, 1)
            if mrc.data.size != expected_size:
                raise ValueError(f"Data size mismatch: expected {expected_size}, got {mrc.data.size}")

            # Check for NaN or Inf
            if mode in [2, 4, 12]:  # Float types
                if np.any(np.isnan(mrc.data)):
                    print("Warning: Data contains NaN values")
                if np.any(np.isinf(mrc.data)):
                    print("Warning: Data contains Inf values")

            print("File validation passed")
            return mrc.data.copy()

    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except IOError as e:
        print(f"Error: Cannot read file: {e}")
        return None
    except ValueError as e:
        print(f"Error: Invalid file format: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


# ============================================================================
# Main: Run Examples
# ============================================================================

if __name__ == '__main__':
    # Replace with your actual file path
    test_file = 'example.mrc'

    print("=" * 70)
    print("Example 1: Basic File Reading")
    print("=" * 70)
    # data = read_basic(test_file)

    print("\n" + "=" * 70)
    print("Example 4: Header Only")
    print("=" * 70)
    # read_header_only(test_file)

    print("\n" + "=" * 70)
    print("Example 9: Coordinate Information")
    print("=" * 70)
    # read_coordinate_info(test_file)

    print("\nTo use these examples, uncomment the function calls above")
    print("and replace 'example.mrc' with your actual file path.")
