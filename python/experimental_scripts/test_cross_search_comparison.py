#!/usr/bin/env python3
"""
Cross-Search Comparison Script

Compares template matching results between two searches to identify:
1. False negatives: peaks found in search1 but missed in search2
2. Detection consistency: whether shared peaks have similar orientations

For each peak in search1, samples the output images from search2 at that
position to extract the score and orientation values, regardless of whether
search2 detected a peak there.

Usage:
    python test_cross_search_comparison.py <db_path> <job_id1> <job_id2> [options]

Example:
    python test_cross_search_comparison.py cp.db 1 8 --position-tolerance 3.0 --angle-tolerance 10.0

    # With histogram output
    python test_cross_search_comparison.py cp.db 1 8 --histogram --histogram-subset found
"""

import sys
import argparse
import os
import numpy as np
import pandas as pd
import mrcfile
from cistemx.io import database as tma

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_mrc_image(filepath: str) -> tuple[np.ndarray, float]:
    """
    Load a 2D MRC image file and return pixel size.

    Handles both true 2D arrays and 3D arrays with singleton first dimension
    (shape (1, Y, X) which is common for cisTEM output images).

    Args:
        filepath: Path to MRC file

    Returns:
        Tuple of (2D NumPy array (Y, X ordering), pixel_size in Angstroms)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file cannot be squeezed to 2D
    """
    with mrcfile.open(filepath, permissive=True) as mrc:
        data = np.squeeze(mrc.data)  # Remove singleton dimensions
        if data.ndim != 2:
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


def sample_at_positions(image: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """
    Sample image values at specified (x, y) positions.

    Uses nearest-neighbor sampling (integer pixel coordinates).
    Positions outside image bounds return NaN.

    Args:
        image: 2D array with shape (ny, nx)
        positions: Nx2 array of (x, y) positions

    Returns:
        1D array of sampled values
    """
    ny, nx = image.shape
    values = np.full(len(positions), np.nan)

    for i, (x, y) in enumerate(positions):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < nx and 0 <= yi < ny:
            values[i] = image[yi, xi]

    return values


def find_max_in_radius(image: np.ndarray, x: float, y: float,
                       radius_pixels: float) -> float:
    """
    Find maximum value within circular radius of a position.

    Used to find "potential" peak values - what the peak height would be
    if we searched within a tolerance radius of a given position.

    Args:
        image: 2D array with shape (ny, nx)
        x, y: Center position in pixels
        radius_pixels: Search radius in pixels

    Returns:
        Maximum value within the radius, or NaN if position is out of bounds
    """
    ny, nx = image.shape
    xi, yi = int(round(x)), int(round(y))
    r = int(np.ceil(radius_pixels))

    # Define bounding box
    x_min, x_max = max(0, xi - r), min(nx, xi + r + 1)
    y_min, y_max = max(0, yi - r), min(ny, yi + r + 1)

    if x_min >= x_max or y_min >= y_max:
        return np.nan

    # Create circular mask within bounding box
    yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
    mask = ((xx - xi)**2 + (yy - yi)**2) <= radius_pixels**2

    region = image[y_min:y_max, x_min:x_max]
    masked_values = region[mask]

    return float(np.max(masked_values)) if len(masked_values) > 0 else np.nan


def find_edge_peaks(scaled_mip: np.ndarray, pixel_size: float,
                    template_size_pixels: int, threshold: float,
                    exclusion_radius_ang: float = 10.0,
                    debug: bool = False) -> pd.DataFrame:
    """
    Find peaks in the edge exclusion zone that wouldn't be saved to the database.

    During template matching, peaks within (template_size / 4 + 1) pixels of
    the image edge are excluded from database insertion. This function finds
    peaks in those edge regions by scanning the scaled_mip.

    Args:
        scaled_mip: 2D peak height image
        pixel_size: Pixel size in Angstroms
        template_size_pixels: Template box size in pixels
        threshold: Minimum peak height to consider
        exclusion_radius_ang: Minimum distance between peaks in Angstroms
        debug: If True, print diagnostic information

    Returns:
        DataFrame with columns matching peaks table structure:
        PEAK_NUMBER, X_POSITION, Y_POSITION, PEAK_HEIGHT, PSI, THETA, PHI,
        DEFOCUS, PIXEL_SIZE (positions in Angstroms)
    """
    ny, nx = scaled_mip.shape

    # Edge margin where peaks are excluded: template_size / 4 + 1
    edge_margin = template_size_pixels // 4 + 1

    # Create mask for edge regions only
    edge_mask = np.zeros((ny, nx), dtype=bool)
    edge_mask[:edge_margin, :] = True  # Top edge
    edge_mask[-edge_margin:, :] = True  # Bottom edge
    edge_mask[:, :edge_margin] = True  # Left edge
    edge_mask[:, -edge_margin:] = True  # Right edge

    if debug:
        edge_values = scaled_mip[edge_mask]
        print(f"    Image size: {nx}x{ny}, edge_margin={edge_margin}px")
        print(f"    Edge region pixels: {np.sum(edge_mask)}, above threshold: {np.sum(edge_values > threshold)}")
        if len(edge_values) > 0:
            print(f"    Edge values: min={edge_values.min():.2f}, max={edge_values.max():.2f}, threshold={threshold:.2f}")

    # Find local maxima above threshold in edge regions
    exclusion_radius_pix = exclusion_radius_ang / pixel_size

    peaks = []
    peak_num = -1  # Negative numbers to distinguish from DB peaks

    # Apply threshold mask combined with edge mask
    candidate_mask = edge_mask & (scaled_mip > threshold)
    candidate_positions = np.argwhere(candidate_mask)  # Returns (y, x) pairs

    if debug:
        print(f"    Candidate positions above threshold in edge: {len(candidate_positions)}")

    # Sort candidates by peak height (descending) to prioritize strongest peaks
    if len(candidate_positions) > 0:
        heights = scaled_mip[candidate_positions[:, 0], candidate_positions[:, 1]]
        sorted_indices = np.argsort(-heights)  # Descending
        candidate_positions = candidate_positions[sorted_indices]

    for pos in candidate_positions:
        y, x = pos[0], pos[1]
        val = scaled_mip[y, x]

        # Check if this is a local maximum within exclusion radius
        r = int(np.ceil(exclusion_radius_pix))
        y_min, y_max = max(0, y - r), min(ny, y + r + 1)
        x_min, x_max = max(0, x - r), min(nx, x + r + 1)
        local_region = scaled_mip[y_min:y_max, x_min:x_max]
        local_max_val = np.max(local_region)

        # Only keep if this pixel IS the local max (within numerical tolerance)
        if val < local_max_val - 0.001:
            continue

        # Check not already found (within exclusion radius of existing peak)
        is_duplicate = False
        for existing in peaks:
            dist = np.sqrt((x - existing['x_pix'])**2 + (y - existing['y_pix'])**2)
            if dist < exclusion_radius_pix:
                is_duplicate = True
                break

        if is_duplicate:
            continue

        # Convert to Angstroms
        x_ang = x * pixel_size
        y_ang = y * pixel_size

        peaks.append({
            'PEAK_NUMBER': peak_num,
            'X_POSITION': x_ang,
            'Y_POSITION': y_ang,
            'PEAK_HEIGHT': float(val),
            'PSI': 0.0,  # Unknown - would need angle images
            'THETA': 0.0,
            'PHI': 0.0,
            'DEFOCUS': 0.0,
            'PIXEL_SIZE': 0.0,
            'x_pix': x,
            'y_pix': y
        })
        peak_num -= 1

        if debug and len(peaks) <= 5:
            print(f"    Found edge peak: ({x}, {y}) = {val:.2f}")

    if not peaks:
        return pd.DataFrame(columns=['PEAK_NUMBER', 'X_POSITION', 'Y_POSITION',
                                     'PEAK_HEIGHT', 'PSI', 'THETA', 'PHI',
                                     'DEFOCUS', 'PIXEL_SIZE'])

    df = pd.DataFrame(peaks)
    # Drop temporary pixel columns
    df = df.drop(columns=['x_pix', 'y_pix'])
    return df


def euler_to_rotation_matrix(phi: float, theta: float, psi: float) -> np.ndarray:
    """
    Convert ZYZ passive intrinsic Euler angles to rotation matrix.

    cisTEM convention: ZYZ passive intrinsic
    - PHI: first rotation about Z axis
    - THETA: rotation about Y axis
    - PSI: final rotation about Z axis

    For passive (alias) rotations, the matrix transforms coordinates
    from the rotated frame back to the original frame.

    Args:
        phi, theta, psi: Euler angles in degrees

    Returns:
        3x3 rotation matrix
    """
    # Convert to radians
    phi_r = np.radians(phi)
    theta_r = np.radians(theta)
    psi_r = np.radians(psi)

    # Precompute trig values
    c1, s1 = np.cos(phi_r), np.sin(phi_r)
    c2, s2 = np.cos(theta_r), np.sin(theta_r)
    c3, s3 = np.cos(psi_r), np.sin(psi_r)

    # ZYZ passive intrinsic: R = Rz(phi) @ Ry(theta) @ Rz(psi)
    # Each rotation matrix for passive convention
    R = np.array([
        [c1*c2*c3 - s1*s3,  -c1*c2*s3 - s1*c3,  c1*s2],
        [s1*c2*c3 + c1*s3,  -s1*c2*s3 + c1*c3,  s1*s2],
        [-s2*c3,             s2*s3,              c2]
    ])

    return R


def orientation_difference(phi1: float, theta1: float, psi1: float,
                           phi2: float, theta2: float, psi2: float) -> float:
    """
    Calculate the true angular difference between two orientations.

    Computes the geodesic distance on SO(3) - the minimum rotation angle
    needed to transform one orientation into the other.

    This is more robust than comparing individual Euler angles because:
    - Handles gimbal lock correctly
    - Accounts for equivalent Euler angle representations
    - Gives a single meaningful scalar (rotation angle)

    Args:
        phi1, theta1, psi1: First orientation (degrees)
        phi2, theta2, psi2: Second orientation (degrees)

    Returns:
        Angular difference in degrees [0, 180]
    """
    # Build rotation matrices
    R1 = euler_to_rotation_matrix(phi1, theta1, psi1)
    R2 = euler_to_rotation_matrix(phi2, theta2, psi2)

    # Relative rotation: R_diff = R1^T @ R2
    # This is the rotation that takes orientation 1 to orientation 2
    R_diff = R1.T @ R2

    # Extract rotation angle from trace
    # For a rotation matrix: trace(R) = 1 + 2*cos(angle)
    # Therefore: angle = arccos((trace(R) - 1) / 2)
    trace = np.trace(R_diff)

    # Clamp to [-1, 1] to handle numerical errors
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)

    return np.degrees(angle_rad)


def compare_searches(analyzer: tma.TemplateMatchAnalyzer,
                     job_id1: int, job_id2: int,
                     position_tolerance: float = 10.0,
                     angle_tolerance: float = 15.0,
                     debug: bool = False,
                     exclude_edge_search: int = None,
                     template_size_pixels: int = None,
                     min_peaks: int = 3) -> tuple[pd.DataFrame, dict]:
    """
    Compare peaks from search1 against search2 output images.

    For each peak in search1:
    - Extracts score/orientation from search2 output images at same position
    - Checks if search2 detected a peak nearby (within tolerance)
    - Calculates orientation difference using rotation matrices (SO(3) geodesic)

    Args:
        analyzer: Initialized TemplateMatchAnalyzer
        job_id1: Reference search (peaks come from here)
        job_id2: Comparison search (images sampled from here)
        position_tolerance: Max distance (Angstroms) to consider "same peak"
        angle_tolerance: Max angular difference (degrees) for "same orientation"
        debug: If True, process only first image and print detailed tables
        exclude_edge_search: If set, find peaks in edge exclusion zone for this search ID
        template_size_pixels: Template box size for edge exclusion calculation

    Returns:
        DataFrame with comparison results, one row per search1 peak
    """
    # Find overlapping images
    overlap = analyzer.get_overlapping_images(job_id1, job_id2)
    print(f"Found {len(overlap)} images analyzed in both searches")

    if not overlap:
        return pd.DataFrame()

    # Get file paths for both searches
    paths1 = analyzer.get_result_file_paths(job_id1, overlap)
    paths2 = analyzer.get_result_file_paths(job_id2, overlap)

    # Load peaks for search1 (with USED_THRESHOLD metadata)
    peaks1 = analyzer.load_all_peaks_for_jobs([job_id1])
    peaks1 = peaks1[peaks1['IMAGE_ASSET_ID'].isin(overlap)]
    peaks1 = analyzer.add_metadata_columns(peaks1, ['USED_THRESHOLD'])

    # Load peaks for search2 (with USED_THRESHOLD metadata)
    peaks2 = analyzer.load_all_peaks_for_jobs([job_id2])
    peaks2 = peaks2[peaks2['IMAGE_ASSET_ID'].isin(overlap)]
    peaks2 = analyzer.add_metadata_columns(peaks2, ['USED_THRESHOLD'])

    # Get representative thresholds (first value since typically same per job)
    threshold1 = float(peaks1['USED_THRESHOLD'].iloc[0]) if len(peaks1) > 0 else 0.0
    threshold2 = float(peaks2['USED_THRESHOLD'].iloc[0]) if len(peaks2) > 0 else 0.0

    images_with_s1_peaks = set(peaks1['IMAGE_ASSET_ID'].unique())
    images_with_s2_peaks = set(peaks2['IMAGE_ASSET_ID'].unique())
    images_s1_only = images_with_s1_peaks - images_with_s2_peaks
    images_s2_only = images_with_s2_peaks - images_with_s1_peaks

    print(f"Search1 (job {job_id1}): {len(peaks1)} peaks across {len(images_with_s1_peaks)} images")
    print(f"Search2 (job {job_id2}): {len(peaks2)} peaks across {len(images_with_s2_peaks)} images")
    print(f"Thresholds: search1={threshold1:.2f}, search2={threshold2:.2f}")

    if images_s1_only:
        print(f"  Note: {len(images_s1_only)} image(s) have search1 peaks but NO search2 peaks: {sorted(images_s1_only)}")
    if images_s2_only:
        print(f"  Note: {len(images_s2_only)} image(s) have search2 peaks but NO search1 peaks: {sorted(images_s2_only)}")

    # Filter to images with minimum peaks in BOTH searches
    if min_peaks > 0:
        s1_counts = peaks1.groupby('IMAGE_ASSET_ID').size()
        s2_counts = peaks2.groupby('IMAGE_ASSET_ID').size()

        # Images meeting minimum in both searches
        valid_images = set(s1_counts[s1_counts >= min_peaks].index) & \
                       set(s2_counts[s2_counts >= min_peaks].index)

        excluded_count = len(overlap) - len(valid_images)
        if excluded_count > 0:
            print(f"  Filtering: excluded {excluded_count} image(s) with <{min_peaks} peaks in either search")

        # Filter peaks to only valid images
        peaks1 = peaks1[peaks1['IMAGE_ASSET_ID'].isin(valid_images)]
        peaks2 = peaks2[peaks2['IMAGE_ASSET_ID'].isin(valid_images)]
        overlap = list(valid_images)

        print(f"  After filtering: {len(peaks1)} search1 peaks, {len(peaks2)} search2 peaks across {len(valid_images)} images")

    # Build lookup for file paths by IMAGE_ASSET_ID
    paths1_lookup = paths1.set_index('IMAGE_ASSET_ID').to_dict('index')
    paths2_lookup = paths2.set_index('IMAGE_ASSET_ID').to_dict('index')

    # Build spatial index for search2 peaks (for detecting matching peaks)
    # Keep as DataFrame for cleaner column access
    peaks2_by_image = {
        img_id: group[['X_POSITION', 'Y_POSITION', 'PEAK_NUMBER', 'PEAK_HEIGHT',
                       'PSI', 'THETA', 'PHI', 'DEFOCUS', 'PIXEL_SIZE']].reset_index(drop=True)
        for img_id, group in peaks2.groupby('IMAGE_ASSET_ID')
    }

    # Results accumulator
    results = []

    # Bidirectional tracking: count search2 peaks not found in search1
    reverse_stats = {
        'total_search2_peaks': 0,
        'search2_only_peaks': 0,  # In search2 but not in search1
        'search2_only_above_threshold1': 0,  # Above search1's threshold
    }

    # Process each image (limit to first in debug mode)
    unique_images = peaks1['IMAGE_ASSET_ID'].unique()
    if debug:
        unique_images = unique_images[:1]
        print(f"\n[DEBUG MODE] Processing only first image (ID: {unique_images[0]})")

    for img_idx, image_id in enumerate(unique_images):
        if not debug:
            print(f"  Processing image {img_idx + 1}/{len(unique_images)} (ID: {image_id})...", end='\r')
        else:
            print(f"  Processing image {img_idx + 1}/{len(unique_images)} (ID: {image_id})...")

        # Get file paths for this image from both searches
        if image_id not in paths1_lookup:
            print(f"  Warning: No search1 paths for image {image_id}")
            continue
        if image_id not in paths2_lookup:
            print(f"  Warning: No search2 paths for image {image_id}")
            continue

        paths1_img = paths1_lookup[image_id]
        paths2_img = paths2_lookup[image_id]

        # Load search2 output images (returns image and pixel_size)
        # Also load search1's scaled_mip for potential peak detection
        try:
            scaled_mip2, pixel_size2 = load_mrc_image(paths2_img['SCALED_MIP_OUTPUT_FILE'])
            psi_img, _ = load_mrc_image(paths2_img['PSI_OUTPUT_FILE'])
            theta_img, _ = load_mrc_image(paths2_img['THETA_OUTPUT_FILE'])
            phi_img, _ = load_mrc_image(paths2_img['PHI_OUTPUT_FILE'])
            defocus_img, _ = load_mrc_image(paths2_img['DEFOCUS_OUTPUT_FILE'])

            # Load search1's scaled_mip and get pixel size from header
            scaled_mip1, pixel_size1 = load_mrc_image(paths1_img['SCALED_MIP_OUTPUT_FILE'])
        except FileNotFoundError as e:
            print(f"\n  Warning: Missing file for image {image_id}: {e}")
            continue
        except Exception as e:
            print(f"\n  Warning: Error loading images for {image_id}: {e}")
            continue

        # Get search1 peaks for this image
        img_peaks1 = peaks1[peaks1['IMAGE_ASSET_ID'] == image_id]

        # Get search2 peaks for this image (for matching)
        # Returns DataFrame or empty DataFrame with correct columns
        empty_peaks2 = pd.DataFrame(columns=['X_POSITION', 'Y_POSITION', 'PEAK_NUMBER',
                                              'PEAK_HEIGHT', 'PSI', 'THETA', 'PHI',
                                              'DEFOCUS', 'PIXEL_SIZE'])
        img_peaks2 = peaks2_by_image.get(image_id, empty_peaks2)

        # Find peaks in edge exclusion zones if requested
        if exclude_edge_search is not None and template_size_pixels is not None:
            edge_margin = template_size_pixels // 4 + 1
            if exclude_edge_search == job_id1:
                # Find edge peaks in search1's MIP using search1's threshold
                if debug:
                    print(f"  Searching for edge peaks in search1 (margin={edge_margin}px, threshold={threshold1:.2f})")
                edge_peaks = find_edge_peaks(scaled_mip1, pixel_size1,
                                             template_size_pixels, threshold1,
                                             exclusion_radius_ang=position_tolerance,
                                             debug=debug)
                if debug:
                    print(f"  Found {len(edge_peaks)} edge peaks in search1")
                if len(edge_peaks) > 0:
                    # Add USED_THRESHOLD column to match img_peaks1
                    edge_peaks['USED_THRESHOLD'] = threshold1
                    edge_peaks['IMAGE_ASSET_ID'] = image_id
                    edge_peaks['TEMPLATE_MATCH_ID'] = -1  # Mark as synthetic
                    img_peaks1 = pd.concat([img_peaks1, edge_peaks], ignore_index=True)

            elif exclude_edge_search == job_id2:
                # Find edge peaks in search2's MIP using search2's threshold
                edge_peaks = find_edge_peaks(scaled_mip2, pixel_size2,
                                             template_size_pixels, threshold2,
                                             exclusion_radius_ang=position_tolerance)
                if len(edge_peaks) > 0:
                    edge_peaks['USED_THRESHOLD'] = threshold2
                    edge_peaks['IMAGE_ASSET_ID'] = image_id
                    edge_peaks['TEMPLATE_MATCH_ID'] = -1
                    img_peaks2 = pd.concat([img_peaks2, edge_peaks], ignore_index=True)
                    if debug:
                        print(f"  Added {len(edge_peaks)} edge peaks from search2 (threshold={threshold2:.2f})")

        # Pixel sizes from MRC headers (actual pixel sizes, not refinement offsets)
        # Note: PIXEL_SIZE column in peaks table stores refinement OFFSETS, not absolute values

        if debug:
            # Show pixel sizes from both searches
            ps_offset1 = float(img_peaks1['PIXEL_SIZE'].iloc[0]) if len(img_peaks1) > 0 else 0.0
            ps_offset2 = float(img_peaks2['PIXEL_SIZE'].iloc[0]) if len(img_peaks2) > 0 else 0.0
            print(f"\n  Pixel sizes (from MRC headers): search1={pixel_size1:.4f} Å, search2={pixel_size2:.4f} Å")
            print(f"  Pixel size offsets (refinement): search1={ps_offset1:.4f}, search2={ps_offset2:.4f}")
            print(f"  Search1 peaks: {len(img_peaks1)}, Search2 peaks: {len(img_peaks2)}")
            print(f"\n  Scaled MIP paths:")
            print(f"    Search1: {paths1_img['SCALED_MIP_OUTPUT_FILE']}")
            print(f"    Search2: {paths2_img['SCALED_MIP_OUTPUT_FILE']}")

            # Print debug peak comparison table (with images for potential peak detection)
            print_debug_peak_table(img_peaks1, img_peaks2, pixel_size1, pixel_size2,
                                   position_tolerance, image_id,
                                   threshold1=threshold1, threshold2=threshold2,
                                   scaled_mip1=scaled_mip1, scaled_mip2=scaled_mip2)

        # Database stores positions in Angstroms; images are indexed by pixels
        # Convert Angstrom positions to pixel coordinates for image sampling
        positions_angstroms = img_peaks1[['X_POSITION', 'Y_POSITION']].values
        positions_pixels = positions_angstroms / pixel_size2

        if debug and len(positions_angstroms) > 0:
            print(f"\n  Coordinate conversion (Å -> pixels for image sampling):")
            print(f"    Using pixel_size2: {pixel_size2:.4f} Å")
            print(f"    First peak: ({positions_angstroms[0,0]:.2f}, {positions_angstroms[0,1]:.2f}) Å -> "
                  f"({positions_pixels[0,0]:.2f}, {positions_pixels[0,1]:.2f}) pix")

        # Sample search2 images at search1 peak positions (in pixels)
        scores2 = sample_at_positions(scaled_mip2, positions_pixels)
        psi2 = sample_at_positions(psi_img, positions_pixels)
        theta2 = sample_at_positions(theta_img, positions_pixels)
        phi2 = sample_at_positions(phi_img, positions_pixels)
        defocus2 = sample_at_positions(defocus_img, positions_pixels)

        # Build 1:1 peak matching with validation
        # First pass: find nearest search2 peak for each search1 peak
        peak1_to_peak2 = {}  # search1_idx -> (search2_idx, distance)

        if len(img_peaks2) > 0:
            peaks2_x = img_peaks2['X_POSITION'].values.astype(np.float64)
            peaks2_y = img_peaks2['Y_POSITION'].values.astype(np.float64)

            for i, (_, peak1) in enumerate(img_peaks1.iterrows()):
                x1, y1 = float(peak1['X_POSITION']), float(peak1['Y_POSITION'])
                distances = np.sqrt((peaks2_x - x1)**2 + (peaks2_y - y1)**2)
                nearest_idx = np.argmin(distances)
                nearest_distance = float(distances[nearest_idx])

                if nearest_distance <= position_tolerance:
                    peak1_to_peak2[i] = (nearest_idx, nearest_distance)

        # Validate 1:1 matching - check for duplicate assignments
        peak2_assignments = {}  # search2_idx -> list of (search1_idx, distance)
        for peak1_idx, (peak2_idx, dist) in peak1_to_peak2.items():
            if peak2_idx not in peak2_assignments:
                peak2_assignments[peak2_idx] = []
            peak2_assignments[peak2_idx].append((peak1_idx, dist))

        # Check for any search2 peak matched by multiple search1 peaks
        duplicates = {k: v for k, v in peak2_assignments.items() if len(v) > 1}
        if duplicates:
            # Build detailed error message
            error_lines = [
                f"\nError: Position tolerance {position_tolerance:.1f} Å is too loose!",
                f"Multiple search1 peaks matched to the same search2 peak in image {image_id}:",
            ]
            for peak2_idx, matches in duplicates.items():
                peak2_row = img_peaks2.iloc[peak2_idx]
                peak2_x = float(peak2_row['X_POSITION'])
                peak2_y = float(peak2_row['Y_POSITION'])
                error_lines.append(f"\n  Search2 peak at ({peak2_x:.1f}, {peak2_y:.1f}) matched by:")
                for peak1_idx, dist in matches:
                    peak1_row = img_peaks1.iloc[peak1_idx]
                    error_lines.append(
                        f"    - Search1 peak {int(peak1_row['PEAK_NUMBER'])} "
                        f"at ({float(peak1_row['X_POSITION']):.1f}, {float(peak1_row['Y_POSITION']):.1f}) "
                        f"distance={dist:.1f} Å"
                    )
            error_lines.append(f"\nSuggestion: Reduce position_tolerance below the minimum inter-peak distance.")
            raise ValueError('\n'.join(error_lines))

        # For each search1 peak, process with validated matching
        for i, (_, peak1) in enumerate(img_peaks1.iterrows()):
            x1, y1 = float(peak1['X_POSITION']), float(peak1['Y_POSITION'])

            # Look up pre-validated match
            if i in peak1_to_peak2:
                found_in_search2 = True
                nearest_peak2_idx, nearest_distance = peak1_to_peak2[i]
            else:
                found_in_search2 = False
                nearest_distance = np.inf
                nearest_peak2_idx = None

            # Calculate orientation difference using rotation matrices
            # This is the geodesic distance on SO(3) - much more meaningful than
            # comparing individual Euler angles
            #
            # If peak was FOUND in search2, compare against the DETECTED peak's
            # database angles (more accurate than image-sampled values)
            # If NOT found, compare against image-sampled values at that position
            if found_in_search2 and nearest_peak2_idx is not None:
                # Use detected peak's angles from database
                peak2_row = img_peaks2.iloc[nearest_peak2_idx]
                peak2_psi = float(peak2_row['PSI'])
                peak2_theta = float(peak2_row['THETA'])
                peak2_phi = float(peak2_row['PHI'])
                orientation_diff = orientation_difference(
                    float(peak1['PHI']), float(peak1['THETA']), float(peak1['PSI']),
                    peak2_phi, peak2_theta, peak2_psi
                )
            elif not (np.isnan(psi2[i]) or np.isnan(theta2[i]) or np.isnan(phi2[i])):
                # Use image-sampled values for missed peaks
                orientation_diff = orientation_difference(
                    float(peak1['PHI']), float(peak1['THETA']), float(peak1['PSI']),
                    phi2[i], theta2[i], psi2[i]
                )
            else:
                orientation_diff = np.nan

            # Check if orientation matches within tolerance
            orientation_matches = (
                not np.isnan(orientation_diff) and
                orientation_diff <= angle_tolerance
            )

            result = {
                # Identifiers
                'IMAGE_ASSET_ID': image_id,
                'SEARCH1_PEAK_NUMBER': peak1['PEAK_NUMBER'],

                # Search1 values
                'SEARCH1_X': x1,
                'SEARCH1_Y': y1,
                'SEARCH1_SCORE': float(peak1['PEAK_HEIGHT']),
                'SEARCH1_PSI': float(peak1['PSI']),
                'SEARCH1_THETA': float(peak1['THETA']),
                'SEARCH1_PHI': float(peak1['PHI']),
                'SEARCH1_DEFOCUS': float(peak1['DEFOCUS']),

                # Search2 values (from images at search1 position)
                'SEARCH2_SCORE_AT_XY': scores2[i],
                'SEARCH2_PSI_AT_XY': psi2[i],
                'SEARCH2_THETA_AT_XY': theta2[i],
                'SEARCH2_PHI_AT_XY': phi2[i],
                'SEARCH2_DEFOCUS_AT_XY': defocus2[i],

                # Detection status
                'FOUND_IN_SEARCH2': found_in_search2,
                'NEAREST_PEAK_DISTANCE': nearest_distance if nearest_distance != np.inf else np.nan,

                # Orientation difference (geodesic distance on SO(3))
                'ORIENTATION_DIFF': orientation_diff,

                # Summary flags
                'ORIENTATION_MATCHES': orientation_matches,
            }

            # If found in search2, add the detected peak's values
            if found_in_search2 and nearest_peak2_idx is not None:
                peak2_data = img_peaks2.iloc[nearest_peak2_idx]
                result['SEARCH2_DETECTED_SCORE'] = float(peak2_data['PEAK_HEIGHT'])
                result['SEARCH2_DETECTED_PSI'] = float(peak2_data['PSI'])
                result['SEARCH2_DETECTED_THETA'] = float(peak2_data['THETA'])
                result['SEARCH2_DETECTED_PHI'] = float(peak2_data['PHI'])
                result['SEARCH2_DETECTED_DEFOCUS'] = float(peak2_data['DEFOCUS'])
            else:
                result['SEARCH2_DETECTED_SCORE'] = np.nan
                result['SEARCH2_DETECTED_PSI'] = np.nan
                result['SEARCH2_DETECTED_THETA'] = np.nan
                result['SEARCH2_DETECTED_PHI'] = np.nan
                result['SEARCH2_DETECTED_DEFOCUS'] = np.nan

            results.append(result)

        # Track search2 peaks not matched to any search1 peak (bidirectional stats)
        matched_search2_indices = set(peak2_idx for peak2_idx, _ in peak1_to_peak2.values())
        reverse_stats['total_search2_peaks'] += len(img_peaks2)

        for j in range(len(img_peaks2)):
            if j not in matched_search2_indices:
                reverse_stats['search2_only_peaks'] += 1
                # Check if this unmatched peak would be above search1's threshold
                peak2_score = float(img_peaks2.iloc[j]['PEAK_HEIGHT'])
                if peak2_score > threshold1:
                    reverse_stats['search2_only_above_threshold1'] += 1

    print()  # Clear the progress line

    return pd.DataFrame(results), reverse_stats


def print_summary(df: pd.DataFrame, job_id1: int, job_id2: int,
                  position_tolerance: float, angle_tolerance: float,
                  reverse_stats: dict = None):
    """Print summary statistics of the comparison."""

    print("\n" + "=" * 70)
    print(f"CROSS-SEARCH COMPARISON SUMMARY")
    print(f"Reference: Job {job_id1} | Comparison: Job {job_id2}")
    print(f"Position tolerance: {position_tolerance} Å | Angle tolerance: {angle_tolerance}°")
    print("=" * 70)

    total_peaks = len(df)
    found = df['FOUND_IN_SEARCH2'].sum()
    missed = total_peaks - found

    # Orientation matches only meaningful for FOUND peaks
    found_df = df[df['FOUND_IN_SEARCH2']]
    orientation_match_found = found_df['ORIENTATION_MATCHES'].sum() if found > 0 else 0

    print(f"\nSearch1 → Search2:")
    print(f"  Total peaks in search1: {total_peaks}")
    print(f"  Found in search2:       {found} ({100*found/total_peaks:.1f}%)")
    print(f"  Missed in search2:      {missed} ({100*missed/total_peaks:.1f}%)")
    if found > 0:
        print(f"  Orientation matches:    {orientation_match_found}/{found} ({100*orientation_match_found/found:.1f}% of found peaks)")

    # Reverse direction: search2 peaks not in search1
    if reverse_stats:
        total_s2 = reverse_stats['total_search2_peaks']
        s2_only = reverse_stats['search2_only_peaks']
        s2_only_above = reverse_stats['search2_only_above_threshold1']
        matched_s2 = total_s2 - s2_only

        print(f"\nSearch2 → Search1:")
        print(f"  Total peaks in search2: {total_s2}")
        print(f"  Found in search1:       {matched_s2} ({100*matched_s2/total_s2:.1f}%)" if total_s2 > 0 else "  Found in search1:       0")
        print(f"  Missed in search1:      {s2_only} ({100*s2_only/total_s2:.1f}%)" if total_s2 > 0 else "  Missed in search1:      0")
        if s2_only > 0:
            print(f"    Above search1 threshold: {s2_only_above} ({100*s2_only_above/s2_only:.1f}% of missed)")

    # Score statistics for missed peaks
    if missed > 0:
        missed_df = df[~df['FOUND_IN_SEARCH2']]
        print(f"\nMISSED PEAKS - Search2 scores at those positions:")
        print(f"  Mean score:   {missed_df['SEARCH2_SCORE_AT_XY'].mean():.3f}")
        print(f"  Std score:    {missed_df['SEARCH2_SCORE_AT_XY'].std():.3f}")
        print(f"  Min score:    {missed_df['SEARCH2_SCORE_AT_XY'].min():.3f}")
        print(f"  Max score:    {missed_df['SEARCH2_SCORE_AT_XY'].max():.3f}")

    # Score statistics for found peaks
    if found > 0:
        found_df = df[df['FOUND_IN_SEARCH2']]
        print(f"\nFOUND PEAKS - Score comparison:")
        print(f"  Search1 mean: {found_df['SEARCH1_SCORE'].mean():.3f}")
        print(f"  Search2 mean: {found_df['SEARCH2_DETECTED_SCORE'].mean():.3f}")

        print(f"\nFOUND PEAKS - Orientation difference (geodesic on SO(3)):")
        print(f"  Mean:   {found_df['ORIENTATION_DIFF'].mean():.2f}°")
        print(f"  Std:    {found_df['ORIENTATION_DIFF'].std():.2f}°")
        print(f"  Median: {found_df['ORIENTATION_DIFF'].median():.2f}°")
        print(f"  Max:    {found_df['ORIENTATION_DIFF'].max():.2f}°")


def print_debug_peak_table(peaks1_df: pd.DataFrame, peaks2_df: pd.DataFrame,
                            pixel_size1: float, pixel_size2: float,
                            position_tolerance: float, image_id: int,
                            threshold1: float = 0.0, threshold2: float = 0.0,
                            scaled_mip1: np.ndarray = None, scaled_mip2: np.ndarray = None):
    """
    Print aligned table of peaks from both searches for visual comparison.

    Matches peaks by position (within tolerance) and displays them side-by-side.
    Unmatched peaks show "potential" scores and positions (max value within
    tolerance radius) in dark blue instead of '---'.

    Args:
        peaks1_df: Search1 peaks for this image
        peaks2_df: Search2 peaks for this image (can be empty DataFrame)
        pixel_size1: Pixel size for search1 (from MRC header)
        pixel_size2: Pixel size for search2 (from MRC header)
        position_tolerance: Max distance in Angstroms for matching
        image_id: Image asset ID for labeling
        threshold1: USED_THRESHOLD for search1
        threshold2: USED_THRESHOLD for search2
        scaled_mip1: Search1's scaled MIP image (optional, for potential peak detection)
        scaled_mip2: Search2's scaled MIP image (optional, for potential peak detection)
    """
    # ANSI color codes for terminal output
    DARK_BLUE = '\033[38;5;24m'  # Dark blue (256-color mode)
    RESET = '\033[0m'
    print(f"\n{'='*168}")
    print(f"DEBUG: Peak Comparison Table for Image {image_id}")
    print(f"{'='*168}")
    print(f"Search1 pixel size: {pixel_size1:.4f} Å | Search2 pixel size: {pixel_size2:.4f} Å")
    print(f"Search1 threshold:  {threshold1:.2f}    | Search2 threshold:  {threshold2:.2f}")
    print(f"Position tolerance: {position_tolerance:.1f} Å")
    print()

    # Database stores positions in ANGSTROMS
    # Also compute pixel positions for reference (using respective pixel sizes)
    peaks1_list = []
    for _, p in peaks1_df.iterrows():
        x_ang = float(p['X_POSITION'])
        y_ang = float(p['Y_POSITION'])
        peaks1_list.append({
            'peak_num': int(p['PEAK_NUMBER']),
            'x_pix': x_ang / pixel_size1,  # Convert Å to pixels
            'y_pix': y_ang / pixel_size1,
            'x_ang': x_ang,
            'y_ang': y_ang,
            'score': float(p['PEAK_HEIGHT']),
            'defocus': float(p['DEFOCUS']),
            'psi': float(p['PSI']),
            'theta': float(p['THETA']),
            'phi': float(p['PHI']),
            'pixel_size': float(p['PIXEL_SIZE']),
            'matched_to': None,
            'match_dist': None
        })

    peaks2_list = []
    if len(peaks2_df) > 0:
        for _, p in peaks2_df.iterrows():
            x_ang = float(p['X_POSITION'])
            y_ang = float(p['Y_POSITION'])
            peaks2_list.append({
                'peak_num': int(p['PEAK_NUMBER']),
                'x_pix': x_ang / pixel_size2,  # Convert Å to pixels using passed-in pixel size
                'y_pix': y_ang / pixel_size2,
                'x_ang': x_ang,
                'y_ang': y_ang,
                'score': float(p['PEAK_HEIGHT']),
                'defocus': float(p['DEFOCUS']),
                'psi': float(p['PSI']),
                'theta': float(p['THETA']),
                'phi': float(p['PHI']),
                'pixel_size': pixel_size2,
                'matched_to': None,
                'match_dist': None
            })

    # Match peaks by Angstrom distance
    for p1 in peaks1_list:
        best_dist = float('inf')
        best_p2_idx = None
        for j, p2 in enumerate(peaks2_list):
            if p2['matched_to'] is not None:
                continue  # Already matched
            dist = np.sqrt((p1['x_ang'] - p2['x_ang'])**2 + (p1['y_ang'] - p2['y_ang'])**2)
            if dist < best_dist and dist <= position_tolerance:
                best_dist = dist
                best_p2_idx = j
        if best_p2_idx is not None:
            p1['matched_to'] = peaks2_list[best_p2_idx]['peak_num']
            p1['match_dist'] = best_dist
            peaks2_list[best_p2_idx]['matched_to'] = p1['peak_num']
            peaks2_list[best_p2_idx]['match_dist'] = best_dist

    # Build display rows: matched pairs first, then unmatched
    matched_rows = []
    unmatched_search1 = []
    unmatched_search2 = []

    matched_p2_nums = set()
    for p1 in peaks1_list:
        if p1['matched_to'] is not None:
            # Find the matched search2 peak
            p2 = next(p for p in peaks2_list if p['peak_num'] == p1['matched_to'])
            matched_rows.append((p1, p2))
            matched_p2_nums.add(p2['peak_num'])
        else:
            unmatched_search1.append(p1)

    for p2 in peaks2_list:
        if p2['peak_num'] not in matched_p2_nums:
            unmatched_search2.append(p2)

    # Sort matched rows by search1 score (descending)
    matched_rows.sort(key=lambda pair: pair[0]['score'], reverse=True)

    # Sort unmatched lists by score (descending) for consistency
    unmatched_search1.sort(key=lambda p: p['score'], reverse=True)
    unmatched_search2.sort(key=lambda p: p['score'], reverse=True)

    # Print header (with score/threshold ratio, defocus, and angles)
    hdr1 = "SEARCH 1"
    hdr2 = "SEARCH 2"
    print(f"{'':3} | {hdr1:^72} | {hdr2:^72} | {'Dist':>6} {'AngΔ':>6}")
    print(f"{'#':>3} | {'Pk#':>4} {'Score':>7} {'S/T':>5} {'X(Å)':>9} {'Y(Å)':>9} {'Def':>7} {'Psi':>7} {'Theta':>7} {'Phi':>7} | "
          f"{'Pk#':>4} {'Score':>7} {'S/T':>5} {'X(Å)':>9} {'Y(Å)':>9} {'Def':>7} {'Psi':>7} {'Theta':>7} {'Phi':>7} | {'(Å)':>6} {'(°)':>6}")
    print("-" * 168)

    row_num = 0
    # Print matched pairs
    for p1, p2 in matched_rows:
        row_num += 1
        # Compute score/threshold ratios
        st1 = p1['score'] / threshold1 if threshold1 > 0 else 0.0
        st2 = p2['score'] / threshold2 if threshold2 > 0 else 0.0
        # Compute angular difference (SO(3) geodesic distance)
        ang_diff = orientation_difference(
            p1['phi'], p1['theta'], p1['psi'],
            p2['phi'], p2['theta'], p2['psi']
        )
        print(f"{row_num:3} | {p1['peak_num']:4} {p1['score']:7.2f} {st1:5.2f} {p1['x_ang']:9.2f} {p1['y_ang']:9.2f} "
              f"{p1['defocus']:7.0f} {p1['psi']:7.1f} {p1['theta']:7.1f} {p1['phi']:7.1f} | "
              f"{p2['peak_num']:4} {p2['score']:7.2f} {st2:5.2f} {p2['x_ang']:9.2f} {p2['y_ang']:9.2f} "
              f"{p2['defocus']:7.0f} {p2['psi']:7.1f} {p2['theta']:7.1f} {p2['phi']:7.1f} | {p1['match_dist']:6.2f} {ang_diff:6.1f}")

    # Print unmatched search1 peaks (not found in search2)
    # Show potential score from search2 at the search1 position (in blue)
    if unmatched_search1:
        print("-" * 168)
        print(f"Unmatched Search1 peaks (search2 potential score in {DARK_BLUE}blue{RESET}):")
        for p1 in unmatched_search1:
            row_num += 1
            st1 = p1['score'] / threshold1 if threshold1 > 0 else 0.0
            # Find potential score in search2's image at this position
            if scaled_mip2 is not None:
                x_pix_s2 = p1['x_ang'] / pixel_size2
                y_pix_s2 = p1['y_ang'] / pixel_size2
                radius_pix = position_tolerance / pixel_size2
                potential_s2 = find_max_in_radius(scaled_mip2, x_pix_s2, y_pix_s2, radius_pix)
                score_str = f"{DARK_BLUE}{potential_s2:7.2f}{RESET}"
                st2_str = f"{DARK_BLUE}{potential_s2/threshold2:5.2f}{RESET}" if threshold2 > 0 else f"{'---':>5}"
            else:
                score_str = f"{'---':>7}"
                st2_str = f"{'---':>5}"
            print(f"{row_num:3} | {p1['peak_num']:4} {p1['score']:7.2f} {st1:5.2f} {p1['x_ang']:9.2f} {p1['y_ang']:9.2f} "
                  f"{p1['defocus']:7.0f} {p1['psi']:7.1f} {p1['theta']:7.1f} {p1['phi']:7.1f} | "
                  f"{'---':>4} {score_str} {st2_str} {'---':>9} {'---':>9} {'---':>7} {'---':>7} {'---':>7} {'---':>7} | {'---':>6} {'---':>6}")

    # Print unmatched search2 peaks (not found in search1)
    # Show potential score from search1 at the search2 position (in blue)
    # Track peaks above threshold for additional diagnostics
    above_threshold_s2 = []  # (p2, potential_s1)

    if unmatched_search2:
        print("-" * 168)
        print(f"Unmatched Search2 peaks (search1 potential score in {DARK_BLUE}blue{RESET}):")
        for p2 in unmatched_search2:
            row_num += 1
            st2 = p2['score'] / threshold2 if threshold2 > 0 else 0.0
            potential_s1 = None
            # Find potential score in search1's image at this position
            if scaled_mip1 is not None:
                x_pix_s1 = p2['x_ang'] / pixel_size1
                y_pix_s1 = p2['y_ang'] / pixel_size1
                radius_pix = position_tolerance / pixel_size1
                potential_s1 = find_max_in_radius(scaled_mip1, x_pix_s1, y_pix_s1, radius_pix)
                score_str = f"{DARK_BLUE}{potential_s1:7.2f}{RESET}"
                st1_str = f"{DARK_BLUE}{potential_s1/threshold1:5.2f}{RESET}" if threshold1 > 0 else f"{'---':>5}"
                # Track if above threshold
                if potential_s1 > threshold1:
                    above_threshold_s2.append((p2, potential_s1))
            else:
                score_str = f"{'---':>7}"
                st1_str = f"{'---':>5}"
            print(f"{row_num:3} | {'---':>4} {score_str} {st1_str} {'---':>9} {'---':>9} {'---':>7} {'---':>7} {'---':>7} {'---':>7} | "
                  f"{p2['peak_num']:4} {p2['score']:7.2f} {st2:5.2f} {p2['x_ang']:9.2f} {p2['y_ang']:9.2f} "
                  f"{p2['defocus']:7.0f} {p2['psi']:7.1f} {p2['theta']:7.1f} {p2['phi']:7.1f} | {'---':>6} {'---':>6}")

    # Diagnostic: For unmatched search2 peaks above threshold1, find nearest search1 peak
    YELLOW = '\033[33m'
    if above_threshold_s2 and peaks1_list:
        print()
        print(f"{YELLOW}━━━ Diagnostic: Search2 peaks above threshold1 ({threshold1:.2f}) not matched ━━━{RESET}")
        print(f"{'S2 Pk#':>7} {'S1 Potential':>12} {'S2 X(Å)':>10} {'S2 Y(Å)':>10} │ "
              f"{'Nearest S1 Pk#':>14} {'S1 Score':>9} {'S1 X(Å)':>10} {'S1 Y(Å)':>10} {'Dist(Å)':>9}")
        print("-" * 105)

        for p2, potential_s1 in above_threshold_s2:
            # Find nearest search1 peak (any distance)
            best_dist = float('inf')
            nearest_p1 = None
            for p1 in peaks1_list:
                dist = np.sqrt((p2['x_ang'] - p1['x_ang'])**2 + (p2['y_ang'] - p1['y_ang'])**2)
                if dist < best_dist:
                    best_dist = dist
                    nearest_p1 = p1

            if nearest_p1:
                # Color distance based on how close it is to tolerance
                if best_dist <= position_tolerance:
                    dist_str = f"{DARK_BLUE}{best_dist:9.2f}{RESET}"  # Within tolerance (shouldn't happen)
                elif best_dist <= 2 * position_tolerance:
                    dist_str = f"{YELLOW}{best_dist:9.2f}{RESET}"  # Close, just outside tolerance
                else:
                    dist_str = f"{best_dist:9.2f}"  # Far away

                print(f"{p2['peak_num']:>7} {potential_s1:>12.2f} {p2['x_ang']:>10.2f} {p2['y_ang']:>10.2f} │ "
                      f"{nearest_p1['peak_num']:>14} {nearest_p1['score']:>9.2f} "
                      f"{nearest_p1['x_ang']:>10.2f} {nearest_p1['y_ang']:>10.2f} {dist_str}")
        print(f"{YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}")

        # # Save diagnostic MRC images showing only unmatched peak regions
        # if scaled_mip1 is not None and scaled_mip2 is not None:
        #     # Radius in Angstroms for the region to preserve (same as exclusion radius)
        #     exclusion_radius_ang = 10.0  # Angstroms
        #
        #     # Create masks for both images (start with all False)
        #     mask1 = np.zeros(scaled_mip1.shape, dtype=bool)
        #     mask2 = np.zeros(scaled_mip2.shape, dtype=bool)
        #
        #     ny1, nx1 = scaled_mip1.shape
        #     ny2, nx2 = scaled_mip2.shape
        #
        #     # For each above-threshold unmatched peak, mark circular region
        #     for p2, potential_s1, _, _ in above_threshold_s2:
        #         # Position in Angstroms
        #         x_ang, y_ang = p2['x_ang'], p2['y_ang']
        #
        #         # Mark region in search1 image
        #         x_pix1 = x_ang / pixel_size1
        #         y_pix1 = y_ang / pixel_size1
        #         r_pix1 = int(np.ceil(exclusion_radius_ang / pixel_size1))
        #         xi1, yi1 = int(round(x_pix1)), int(round(y_pix1))
        #         for dy in range(-r_pix1, r_pix1 + 1):
        #             for dx in range(-r_pix1, r_pix1 + 1):
        #                 if dx*dx + dy*dy <= r_pix1*r_pix1:
        #                     px, py = xi1 + dx, yi1 + dy
        #                     if 0 <= px < nx1 and 0 <= py < ny1:
        #                         mask1[py, px] = True
        #
        #         # Mark region in search2 image
        #         x_pix2 = x_ang / pixel_size2
        #         y_pix2 = y_ang / pixel_size2
        #         r_pix2 = int(np.ceil(exclusion_radius_ang / pixel_size2))
        #         xi2, yi2 = int(round(x_pix2)), int(round(y_pix2))
        #         for dy in range(-r_pix2, r_pix2 + 1):
        #             for dx in range(-r_pix2, r_pix2 + 1):
        #                 if dx*dx + dy*dy <= r_pix2*r_pix2:
        #                     px, py = xi2 + dx, yi2 + dy
        #                     if 0 <= px < nx2 and 0 <= py < ny2:
        #                         mask2[py, px] = True
        #
        #     # Create output images: preserved regions + average elsewhere
        #     if np.any(mask1):
        #         avg1 = float(np.mean(scaled_mip1[mask1]))
        #         diag1 = np.full(scaled_mip1.shape, avg1, dtype=np.float32)
        #         diag1[mask1] = scaled_mip1[mask1]
        #
        #         # Save as MRC (to current working directory)
        #         diag1_path = f"diag_search1_img{image_id}.mrc"
        #         with mrcfile.new(diag1_path, overwrite=True) as mrc:
        #             mrc.set_data(diag1)
        #             mrc.voxel_size = pixel_size1
        #         print(f"  Saved diagnostic image: {diag1_path}")
        #
        #         # Also save unmodified search1 scaled_mip for comparison
        #         full1_path = f"full_search1_img{image_id}.mrc"
        #         with mrcfile.new(full1_path, overwrite=True) as mrc:
        #             mrc.set_data(scaled_mip1.astype(np.float32))
        #             mrc.voxel_size = pixel_size1
        #         print(f"  Saved full image: {full1_path}")
        #
        #     if np.any(mask2):
        #         avg2 = float(np.mean(scaled_mip2[mask2]))
        #         diag2 = np.full(scaled_mip2.shape, avg2, dtype=np.float32)
        #         diag2[mask2] = scaled_mip2[mask2]
        #
        #         # Save as MRC (to current working directory)
        #         diag2_path = f"diag_search2_img{image_id}.mrc"
        #         with mrcfile.new(diag2_path, overwrite=True) as mrc:
        #             mrc.set_data(diag2)
        #             mrc.voxel_size = pixel_size2
        #         print(f"  Saved diagnostic image: {diag2_path}")

    print(f"\nSummary: {len(matched_rows)} matched, "
          f"{len(unmatched_search1)} search1-only, {len(unmatched_search2)} search2-only")
    print(f"{'='*168}\n")


def plot_angular_difference_histogram(df: pd.DataFrame, job_id1: int, job_id2: int,
                                      angle_tolerance: float, output_dir: str = '.',
                                      subset: str = 'found') -> str:
    """
    Plot histogram of angular differences between search1 and search2 orientations.

    The angular difference is the geodesic distance on SO(3) - the minimum rotation
    angle needed to transform one orientation into the other. This is computed from
    the trace of the relative rotation matrix: angle = arccos((trace(R1^T @ R2) - 1) / 2).

    Args:
        df: Comparison results DataFrame with 'ORIENTATION_DIFF' column
        job_id1: Reference search job ID
        job_id2: Comparison search job ID
        angle_tolerance: Angle tolerance used (for vertical line annotation)
        output_dir: Directory to save plot (default: current directory)
        subset: Which peaks to include:
            - 'found': Only peaks detected in both searches (default)
            - 'missed': Only peaks missed in search2
            - 'all': All peaks with valid orientation data

    Returns:
        Path to saved histogram image

    Raises:
        ImportError: If matplotlib is not installed
        ValueError: If no valid orientation data exists
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

    # Filter based on subset
    if subset == 'found':
        plot_df = df[df['FOUND_IN_SEARCH2']].copy()
        subset_label = "Found Peaks"
    elif subset == 'missed':
        plot_df = df[~df['FOUND_IN_SEARCH2']].copy()
        subset_label = "Missed Peaks"
    else:  # 'all'
        plot_df = df.copy()
        subset_label = "All Peaks"

    # Remove NaN values
    valid_diffs = plot_df['ORIENTATION_DIFF'].dropna()

    if len(valid_diffs) == 0:
        raise ValueError(f"No valid orientation differences for subset '{subset}'")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create histogram with sensible bins
    # Angular differences range from 0 to 180 degrees (geodesic distance on SO(3))
    max_angle = min(180.0, valid_diffs.max() * 1.1)  # Slightly beyond max for padding
    bins = np.linspace(0, max_angle, 37)  # 5-degree bins up to max

    counts, bin_edges, patches = ax.hist(valid_diffs, bins=bins, edgecolor='black',
                                         alpha=0.7, color='steelblue')

    # Add vertical line at angle tolerance
    ax.axvline(x=angle_tolerance, color='red', linestyle='--', linewidth=2,
               label=f'Tolerance: {angle_tolerance}°')

    # Calculate statistics
    mean_diff = valid_diffs.mean()
    median_diff = valid_diffs.median()
    std_diff = valid_diffs.std()

    # Add vertical lines for mean and median
    ax.axvline(x=mean_diff, color='green', linestyle='-', linewidth=1.5,
               label=f'Mean: {mean_diff:.1f}°')
    ax.axvline(x=median_diff, color='orange', linestyle='-', linewidth=1.5,
               label=f'Median: {median_diff:.1f}°')

    # Count peaks within/outside tolerance
    within_tolerance = (valid_diffs <= angle_tolerance).sum()
    outside_tolerance = len(valid_diffs) - within_tolerance
    pct_within = 100 * within_tolerance / len(valid_diffs)

    # Labels and title
    ax.set_xlabel('Angular Difference (degrees)', fontsize=12)
    ax.set_ylabel('Number of Peaks', fontsize=12)
    ax.set_title(f'Orientation Difference Distribution ({subset_label})\n'
                 f'Job {job_id1} vs Job {job_id2}', fontsize=14)

    # Add statistics text box
    stats_text = (f'N = {len(valid_diffs)}\n'
                  f'Mean = {mean_diff:.2f}°\n'
                  f'Std = {std_diff:.2f}°\n'
                  f'Median = {median_diff:.2f}°\n'
                  f'Within tol: {within_tolerance} ({pct_within:.1f}%)')

    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Legend
    ax.legend(loc='upper left')

    # Grid
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_angle)

    # Save figure
    output_file = os.path.join(output_dir, f'angular_diff_histogram_job{job_id1}_vs_{job_id2}_{subset}.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Compare template matching results between two searches',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script compares peaks from search1 against search2 to understand:
- How many peaks are detected in both searches
- For missed peaks: what was the score at that position in search2?
- For found peaks: are the orientations consistent?

Example:
  %(prog)s /path/to/project.db 1 8 --position-tolerance 5.0 --angle-tolerance 15.0
        """
    )
    parser.add_argument('db_path', help='Path to cisTEM database file')
    parser.add_argument('job_id1', type=int, help='Reference search job ID (peaks from here)')
    parser.add_argument('job_id2', type=int, help='Comparison search job ID (images from here)')
    parser.add_argument('--position-tolerance', type=float, default=10.0,
                        help='Max distance (Angstroms) to consider same peak (default: 10.0)')
    parser.add_argument('--angle-tolerance', type=float, default=15.0,
                        help='Max angular difference (degrees) for same orientation (default: 15.0)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output CSV file for detailed results')
    parser.add_argument('--histogram', action='store_true',
                        help='Generate angular difference histogram plot')
    parser.add_argument('--histogram-subset', type=str, default='found',
                        choices=['found', 'missed', 'all'],
                        help='Which peaks to include in histogram (default: found)')
    parser.add_argument('--histogram-output-dir', type=str, default='.',
                        help='Directory to save histogram plot (default: current directory)')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode: process only first image and print detailed tables')
    parser.add_argument('--exclude-edge', type=int, default=None, metavar='SEARCH_ID',
                        help='Apply edge exclusion to this search ID (requires --template-size)')
    parser.add_argument('--template-size', type=int, default=None, metavar='PIXELS',
                        help='Template box size in pixels for edge exclusion (requires --exclude-edge)')
    parser.add_argument('--min-peaks', type=int, default=3, metavar='N',
                        help='Minimum peaks per image in BOTH searches to include (default: 3)')

    args = parser.parse_args()

    # Validate mutually required arguments
    if (args.exclude_edge is None) != (args.template_size is None):
        parser.error('--exclude-edge and --template-size must be used together')

    print("=" * 70)
    print("Cross-Search Comparison Analysis")
    print("=" * 70)
    print(f"Database: {args.db_path}")
    print(f"Reference search (job1): {args.job_id1}")
    print(f"Comparison search (job2): {args.job_id2}")
    print(f"Position tolerance: {args.position_tolerance} Angstroms")
    print(f"Angle tolerance: {args.angle_tolerance} degrees")
    if args.exclude_edge is not None:
        edge_margin = args.template_size // 4 + 1
        print(f"Edge exclusion: search {args.exclude_edge}, template={args.template_size}px, margin={edge_margin}px")
    print(f"Minimum peaks per image: {args.min_peaks}")
    print()

    try:
        # Initialize analyzer
        print("Initializing analyzer...")
        analyzer = tma.TemplateMatchAnalyzer(args.db_path)
        print("✓ Database validated")
        print()

        # Run comparison
        print("Running comparison...")
        results_df, reverse_stats = compare_searches(
            analyzer,
            args.job_id1,
            args.job_id2,
            position_tolerance=args.position_tolerance,
            angle_tolerance=args.angle_tolerance,
            debug=args.debug,
            exclude_edge_search=args.exclude_edge,
            template_size_pixels=args.template_size,
            min_peaks=args.min_peaks
        )

        if len(results_df) == 0:
            print("No overlapping data to compare.")
            sys.exit(0)

        # Print summary (including bidirectional stats)
        print_summary(results_df, args.job_id1, args.job_id2,
                      args.position_tolerance, args.angle_tolerance,
                      reverse_stats=reverse_stats)

        # Save detailed results if requested
        if args.output:
            results_df.to_csv(args.output, index=False)
            print(f"\n✓ Detailed results saved to: {args.output}")

        # Generate histogram if requested
        if args.histogram:
            if not HAS_MATPLOTLIB:
                print("\n⚠ Warning: matplotlib not installed, skipping histogram generation")
                print("  Install with: pip install matplotlib")
            else:
                try:
                    hist_path = plot_angular_difference_histogram(
                        results_df, args.job_id1, args.job_id2,
                        args.angle_tolerance, args.histogram_output_dir,
                        args.histogram_subset
                    )
                    print(f"\n✓ Histogram saved to: {hist_path}")
                except ValueError as e:
                    print(f"\n⚠ Warning: Could not generate histogram: {e}")

        print("\n" + "=" * 70)
        print("Analysis complete!")
        print("=" * 70)

    except FileNotFoundError as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
