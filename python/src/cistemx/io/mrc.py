"""
MRC file I/O utilities for cryo-EM data.

This module provides functions for reading MRC image files commonly used
in cryo-EM, including micrographs and template matching output images.

These utilities were extracted from analysis scripts developed for cisTEMx
template matching workflows.
"""

import mrcfile
import numpy as np


def load_mrc_image(filepath: str, *, require_2d: bool = True) -> tuple[np.ndarray, float]:
    """
    Load a 2D MRC image file and return pixel size.

    Handles both true 2D arrays and 3D arrays with singleton first dimension
    (shape (1, Y, X) which is common for cisTEM output images).

    Args:
        filepath: Path to MRC file
        require_2d: If True, raise ValueError if result isn't 2D after squeeze

    Returns:
        Tuple of (2D NumPy array (Y, X ordering), pixel_size in Angstroms)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If require_2d=True and file cannot be squeezed to 2D
    """
    with mrcfile.open(filepath, permissive=True) as mrc:
        data = np.squeeze(mrc.data)  # Remove singleton dimensions
        if require_2d and data.ndim != 2:
            raise ValueError(f"Expected 2D image after squeeze, got {data.ndim}D: {filepath}")
        pixel_size = float(mrc.voxel_size.x)
        return data.astype(np.float32), pixel_size


def get_mrc_pixel_size(filepath: str) -> float:
    """
    Read pixel size from MRC header without loading image data.

    This is more efficient than load_mrc_image() when only the pixel size
    is needed, as it avoids reading the (potentially large) image data.

    Args:
        filepath: Path to MRC file

    Returns:
        Pixel size in Angstroms

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    with mrcfile.open(filepath, permissive=True, header_only=True) as mrc:
        return float(mrc.voxel_size.x)


# Convenience alias
load_micrograph = load_mrc_image
