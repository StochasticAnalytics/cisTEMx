#!/usr/bin/env python3
"""
Multi-Search Peak Consensus Analysis

Finds peaks present in ALL searches within a range and computes statistics
(min/max/mean/std) for each consensus peak across all searches.

This is useful for:
- Assessing template matching consistency across different search parameters
- Identifying reliable peaks that are consistently detected
- Quantifying variability in scores, positions, and orientations

Usage:
    python multi_search_consensus.py <db_path> <search_range> [options]

Examples:
    python multi_search_consensus.py cp.db 12:18 --position-tolerance 5.0
    python multi_search_consensus.py cp.db 12,14,16,18 --min-peaks 3 --output consensus.csv
"""

import sys
import argparse
import numpy as np
import pandas as pd
import mrcfile
from cistemx.io import database as tma


def parse_search_range(range_str: str) -> list[int]:
    """
    Parse search range specification into list of job IDs.

    Supports three formats:
    - Colon range: "12:18" → [12, 13, 14, 15, 16, 17, 18]
    - Comma list: "12,14,16" → [12, 14, 16]
    - Single value: "12" → [12]

    Args:
        range_str: Range specification string

    Returns:
        Sorted list of job IDs

    Raises:
        ValueError: If format is invalid or range is empty
    """
    range_str = range_str.strip()

    if ':' in range_str:
        # Range format: "12:18"
        parts = range_str.split(':')
        if len(parts) != 2:
            raise ValueError(f"Invalid range format '{range_str}'. Use 'start:end' (e.g., '12:18')")
        try:
            start, end = int(parts[0]), int(parts[1])
        except ValueError:
            raise ValueError(f"Invalid integers in range '{range_str}'")
        if start > end:
            raise ValueError(f"Start ({start}) must be <= end ({end})")
        return list(range(start, end + 1))

    elif ',' in range_str:
        # Comma-separated list: "12,14,16"
        try:
            job_ids = [int(x.strip()) for x in range_str.split(',')]
        except ValueError:
            raise ValueError(f"Invalid integers in list '{range_str}'")
        return sorted(job_ids)

    else:
        # Single value: "12"
        try:
            return [int(range_str)]
        except ValueError:
            raise ValueError(f"Invalid job ID '{range_str}'")


def euler_to_rotation_matrix(phi: float, theta: float, psi: float) -> np.ndarray:
    """
    Convert ZYZ passive intrinsic Euler angles to rotation matrix.

    cisTEM convention: ZYZ passive intrinsic
    - PHI: first rotation about Z axis
    - THETA: rotation about Y axis
    - PSI: final rotation about Z axis

    Args:
        phi, theta, psi: Euler angles in degrees

    Returns:
        3x3 rotation matrix
    """
    phi_r = np.radians(phi)
    theta_r = np.radians(theta)
    psi_r = np.radians(psi)

    c1, s1 = np.cos(phi_r), np.sin(phi_r)
    c2, s2 = np.cos(theta_r), np.sin(theta_r)
    c3, s3 = np.cos(psi_r), np.sin(psi_r)

    R = np.array([
        [c1*c2*c3 - s1*s3,  -c1*c2*s3 - s1*c3,  c1*s2],
        [s1*c2*c3 + c1*s3,  -s1*c2*s3 + c1*c3,  s1*s2],
        [-s2*c3,             s2*s3,              c2]
    ])

    return R


def orientation_difference(phi1: float, theta1: float, psi1: float,
                           phi2: float, theta2: float, psi2: float) -> float:
    """
    Calculate the geodesic distance on SO(3) between two orientations.

    Args:
        phi1, theta1, psi1: First orientation (degrees)
        phi2, theta2, psi2: Second orientation (degrees)

    Returns:
        Angular difference in degrees [0, 180]
    """
    R1 = euler_to_rotation_matrix(phi1, theta1, psi1)
    R2 = euler_to_rotation_matrix(phi2, theta2, psi2)

    R_diff = R1.T @ R2
    trace = np.trace(R_diff)
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)

    return np.degrees(angle_rad)


def compute_center_of_mass_offset(image: np.ndarray, x_pixel: float, y_pixel: float,
                                   window_half: int = 2) -> tuple[float, float]:
    """
    Compute center-of-mass offset from the integer pixel center.

    This measures how far the intensity-weighted centroid of a peak is from
    the nearest pixel center. Used to test whether sub-pixel positioning
    correlates with score variability between searches.

    Args:
        image: 2D MIP image array (ny, nx)
        x_pixel, y_pixel: Peak position in pixel coordinates
        window_half: Half-size of window (2 for 5x5)

    Returns:
        (x_offset, y_offset) each in range [0, 0.5] representing distance from pixel center
    """
    # Integer center (nearest pixel)
    xi, yi = int(round(x_pixel)), int(round(y_pixel))

    # Extract window (handle boundaries)
    ny, nx = image.shape
    x0 = max(0, xi - window_half)
    x1 = min(nx, xi + window_half + 1)
    y0 = max(0, yi - window_half)
    y1 = min(ny, yi + window_half + 1)

    window = image[y0:y1, x0:x1]

    # Compute center of mass within window
    # Weight by intensity (subtract min to handle negative values)
    weights = window - window.min()
    total = weights.sum()

    if total == 0:
        return 0.0, 0.0

    # Create coordinate grids in image space (not window-local)
    yy, xx = np.meshgrid(range(y0, y1), range(x0, x1), indexing='ij')
    com_x = (xx * weights).sum() / total
    com_y = (yy * weights).sum() / total

    # Offset from integer center (as absolute value 0 to 0.5)
    x_offset = abs(com_x - xi)
    y_offset = abs(com_y - yi)

    # Clamp to [0, 0.5] (should naturally be in this range for well-centered peaks)
    x_offset = min(0.5, x_offset)
    y_offset = min(0.5, y_offset)

    return x_offset, y_offset


def compute_subpixel_offset_fft(image: np.ndarray, x_pixel: float, y_pixel: float,
                                 window_half: int = 2, upsample_factor: int = 10
                                 ) -> tuple[float, float, float]:
    """
    Compute sub-pixel offset using FFT upsampling (sinc interpolation).

    This method zero-pads in Fourier space to upsample the image, which is
    equivalent to sinc interpolation - the optimal interpolation for
    band-limited signals. The maximum of the upsampled image gives both
    the sub-pixel position and the refined peak height.

    Args:
        image: 2D MIP image array (ny, nx)
        x_pixel, y_pixel: Peak position in pixel coordinates
        window_half: Half-size of window (2 for 5x5)
        upsample_factor: Zero-padding factor (10 = 0.1 pixel precision)

    Returns:
        (x_offset, y_offset, refined_peak_height)
        - Offsets in range [0, 0.5] from pixel center
        - Peak height from upsampled maximum
    """
    # Integer center (nearest pixel)
    xi, yi = int(round(x_pixel)), int(round(y_pixel))

    # Extract window (handle boundaries)
    ny, nx = image.shape
    x0 = max(0, xi - window_half)
    x1 = min(nx, xi + window_half + 1)
    y0 = max(0, yi - window_half)
    y1 = min(ny, yi + window_half + 1)

    window = image[y0:y1, x0:x1]
    win_ny, win_nx = window.shape

    # FFT of the window
    fft_window = np.fft.fft2(window)

    # Zero-pad in Fourier space for upsampling
    # Need to carefully place FFT coefficients to maintain correct frequencies
    pad_ny = win_ny * upsample_factor
    pad_nx = win_nx * upsample_factor
    padded = np.zeros((pad_ny, pad_nx), dtype=complex)

    # For odd-sized windows, place all coefficients at corners
    # For a 5x5 window: place [0:3, 0:3] at top-left, [0:3, 3:5] at top-right,
    # [3:5, 0:3] at bottom-left, [3:5, 3:5] at bottom-right

    # Number of positive and negative frequency components
    ny_pos = (win_ny + 1) // 2  # Includes DC
    ny_neg = win_ny // 2
    nx_pos = (win_nx + 1) // 2
    nx_neg = win_nx // 2

    # Top-left quadrant (low positive frequencies including DC)
    padded[:ny_pos, :nx_pos] = fft_window[:ny_pos, :nx_pos]

    # Top-right quadrant (negative x frequencies)
    if nx_neg > 0:
        padded[:ny_pos, -nx_neg:] = fft_window[:ny_pos, -nx_neg:]

    # Bottom-left quadrant (negative y frequencies)
    if ny_neg > 0:
        padded[-ny_neg:, :nx_pos] = fft_window[-ny_neg:, :nx_pos]

    # Bottom-right quadrant (negative x and y frequencies)
    if ny_neg > 0 and nx_neg > 0:
        padded[-ny_neg:, -nx_neg:] = fft_window[-ny_neg:, -nx_neg:]

    # Inverse FFT to get upsampled real-space image
    # Scale by upsample_factor^2 to preserve intensity
    upsampled = np.fft.ifft2(padded).real * (upsample_factor ** 2)

    # Find maximum in upsampled image
    max_idx = np.unravel_index(np.argmax(upsampled), upsampled.shape)
    max_y_up, max_x_up = max_idx

    # Get the refined peak height
    refined_peak_height = upsampled[max_y_up, max_x_up]

    # Convert upsampled coordinates back to original window coordinates
    # then to full image coordinates
    max_x_win = max_x_up / upsample_factor  # Position in window coords
    max_y_win = max_y_up / upsample_factor

    # Position in full image coords
    max_x_img = x0 + max_x_win
    max_y_img = y0 + max_y_win

    # Offset from the integer center
    x_offset = abs(max_x_img - xi)
    y_offset = abs(max_y_img - yi)

    # Clamp to [0, 0.5]
    x_offset = min(0.5, x_offset)
    y_offset = min(0.5, y_offset)

    return x_offset, y_offset, refined_peak_height


def load_mip_for_job_image(analyzer, job_id: int, image_id: int) -> tuple[np.ndarray, float]:
    """
    Load SCALED_MIP image for a specific job/image combination.

    Args:
        analyzer: TemplateMatchAnalyzer instance
        job_id: Template match job ID
        image_id: IMAGE_ASSET_ID

    Returns:
        (image_data, pixel_size) tuple
    """
    paths_df = analyzer.get_result_file_paths(job_id, {image_id})
    if len(paths_df) == 0:
        raise ValueError(f"No MIP path found for job {job_id}, image {image_id}")

    mip_path = paths_df.iloc[0]['SCALED_MIP_OUTPUT_FILE']

    with mrcfile.open(mip_path, permissive=True) as mrc:
        data = np.squeeze(mrc.data).astype(np.float32)
        pixel_size = float(mrc.voxel_size.x)

    return data, pixel_size


def compute_subpixel_offsets(analyzer, results: list[dict], job_ids: list[int],
                              method: str = 'com', upsample_factor: int = 10,
                              window_size: int = 5, debug: bool = False):
    """
    Compute sub-pixel offsets for all consensus peaks.

    Optimized to load each MIP image exactly once per job.
    Modifies results in-place to add X_OFFSET_JOB{id}, Y_OFFSET_JOB{id},
    and (for FFT method) REFINED_HEIGHT_JOB{id}.

    Args:
        analyzer: TemplateMatchAnalyzer instance
        results: List of consensus peak result dicts (modified in-place)
        job_ids: List of job IDs
        method: 'com' (center-of-mass) or 'fft' (FFT upsampling)
        upsample_factor: For FFT method, upsampling factor (default 10)
        window_size: Window size in pixels (default 5)
        debug: Print progress messages
    """
    if not results:
        return

    # Get unique images
    unique_images = set(r['IMAGE_ASSET_ID'] for r in results)

    window_half = window_size // 2
    method_name = "center-of-mass" if method == 'com' else f"FFT ({upsample_factor}x)"
    if debug:
        print(f"\n  Computing sub-pixel offsets ({method_name}, {window_size}x{window_size} window) for {len(results)} peaks across {len(unique_images)} images...")

    for job_id in job_ids:
        for image_id in unique_images:
            # Load MIP once for this (job_id, image_id)
            try:
                mip_data, pixel_size = load_mip_for_job_image(analyzer, job_id, image_id)
            except (ValueError, FileNotFoundError) as e:
                if debug:
                    print(f"    Warning: {e}")
                # Set NaN offsets for this job/image combination
                for result in results:
                    if result['IMAGE_ASSET_ID'] == image_id:
                        result[f'X_OFFSET_JOB{job_id}'] = np.nan
                        result[f'Y_OFFSET_JOB{job_id}'] = np.nan
                        if method == 'fft':
                            result[f'REFINED_HEIGHT_JOB{job_id}'] = np.nan
                continue

            # Process all peaks at this image for this job
            for result in results:
                if result['IMAGE_ASSET_ID'] != image_id:
                    continue

                # Get peak position in Angstroms for this job
                x_ang = result[f'X_JOB{job_id}']
                y_ang = result[f'Y_JOB{job_id}']

                # Convert to pixels
                x_pixel = x_ang / pixel_size
                y_pixel = y_ang / pixel_size

                # Compute offset using selected method
                if method == 'com':
                    x_off, y_off = compute_center_of_mass_offset(
                        mip_data, x_pixel, y_pixel, window_half=window_half)
                else:  # 'fft'
                    x_off, y_off, refined_height = compute_subpixel_offset_fft(
                        mip_data, x_pixel, y_pixel, window_half=window_half,
                        upsample_factor=upsample_factor)
                    result[f'REFINED_HEIGHT_JOB{job_id}'] = refined_height

                # Store offsets
                result[f'X_OFFSET_JOB{job_id}'] = x_off
                result[f'Y_OFFSET_JOB{job_id}'] = y_off

    if debug:
        print(f"  ✓ Sub-pixel offsets computed ({method_name})")


def find_consensus_peaks(peaks_by_job: dict[int, pd.DataFrame],
                         job_ids: list[int],
                         position_tolerance: float,
                         debug: bool = False) -> list[dict]:
    """
    Find peaks that appear in ALL searches using union-then-filter approach.

    Algorithm:
    1. Pool all peaks from all searches
    2. Greedily cluster by position (strongest peak first)
    3. Keep only clusters with exactly one peak from each search

    Args:
        peaks_by_job: Dict mapping job_id -> DataFrame of peaks for one image
        job_ids: List of job IDs (defines expected searches)
        position_tolerance: Maximum distance (Å) for peaks to be considered same
        debug: Print verbose output

    Returns:
        List of consensus peak dicts, each containing matched peak data from all searches
    """
    n_searches = len(job_ids)

    # Pool all peaks with job_id tag
    all_peaks = []
    for job_id in job_ids:
        if job_id not in peaks_by_job:
            continue
        df = peaks_by_job[job_id]
        for _, row in df.iterrows():
            all_peaks.append({
                'job_id': job_id,
                'x': float(row['X_POSITION']),
                'y': float(row['Y_POSITION']),
                'score': float(row['PEAK_HEIGHT']),
                'phi': float(row['PHI']),
                'theta': float(row['THETA']),
                'psi': float(row['PSI']),
                'defocus': float(row['DEFOCUS']),
                'peak_num': int(row['PEAK_NUMBER']),
                'used': False
            })

    if not all_peaks:
        return []

    # Sort by score descending (strongest peaks first for greedy clustering)
    all_peaks.sort(key=lambda p: p['score'], reverse=True)

    # Greedy clustering
    consensus_peaks = []

    for seed in all_peaks:
        if seed['used']:
            continue

        # Start a new cluster with this seed
        cluster = {seed['job_id']: seed}
        seed['used'] = True

        # Find nearest unassigned peak from each other search
        for job_id in job_ids:
            if job_id in cluster:
                continue

            best_match = None
            best_dist = float('inf')

            for candidate in all_peaks:
                if candidate['used'] or candidate['job_id'] != job_id:
                    continue

                dist = np.sqrt((candidate['x'] - seed['x'])**2 +
                               (candidate['y'] - seed['y'])**2)

                if dist <= position_tolerance and dist < best_dist:
                    best_dist = dist
                    best_match = candidate

            if best_match is not None:
                cluster[job_id] = best_match
                best_match['used'] = True

        # Keep cluster only if it has peaks from ALL searches
        if len(cluster) == n_searches:
            consensus_peaks.append(cluster)

    if debug:
        print(f"    Pooled {len(all_peaks)} peaks, found {len(consensus_peaks)} consensus clusters")

    return consensus_peaks


def compute_consensus_stats(consensus_peaks: list[dict],
                            job_ids: list[int],
                            image_id: int) -> list[dict]:
    """
    Compute statistics for each consensus peak cluster.

    Args:
        consensus_peaks: List of cluster dicts from find_consensus_peaks
        job_ids: List of job IDs
        image_id: IMAGE_ASSET_ID for this image

    Returns:
        List of result dicts with statistics and per-search values
    """
    results = []

    for peak_idx, cluster in enumerate(consensus_peaks):
        # Extract arrays of values across searches
        scores = np.array([cluster[j]['score'] for j in job_ids])
        x_vals = np.array([cluster[j]['x'] for j in job_ids])
        y_vals = np.array([cluster[j]['y'] for j in job_ids])
        defocus_vals = np.array([cluster[j]['defocus'] for j in job_ids])

        # Position statistics
        x_centroid = np.mean(x_vals)
        y_centroid = np.mean(y_vals)
        position_std = np.sqrt(np.std(x_vals)**2 + np.std(y_vals)**2)

        # Score statistics
        score_min = np.min(scores)
        score_max = np.max(scores)
        score_mean = np.mean(scores)
        score_std = np.std(scores)

        # Defocus statistics
        defocus_mean = np.mean(defocus_vals)
        defocus_std = np.std(defocus_vals)

        # Orientation statistics
        # Compute mean orientation using first search as reference
        # Then compute angular std as RMS of pairwise differences from mean
        ref = cluster[job_ids[0]]
        phi_ref, theta_ref, psi_ref = ref['phi'], ref['theta'], ref['psi']

        # Compute angular differences from reference
        angle_diffs = []
        for job_id in job_ids:
            p = cluster[job_id]
            diff = orientation_difference(phi_ref, theta_ref, psi_ref,
                                          p['phi'], p['theta'], p['psi'])
            angle_diffs.append(diff)

        orientation_std = np.std(angle_diffs)

        # Use reference orientation as "mean" (simplified - proper averaging uses quaternions)
        phi_mean = phi_ref
        theta_mean = theta_ref
        psi_mean = psi_ref

        # Build result dict
        result = {
            'IMAGE_ASSET_ID': image_id,
            'CONSENSUS_PEAK_ID': peak_idx + 1,
            'X_CENTROID': x_centroid,
            'Y_CENTROID': y_centroid,
            'POSITION_STD': position_std,
            'SCORE_MIN': score_min,
            'SCORE_MAX': score_max,
            'SCORE_MEAN': score_mean,
            'SCORE_STD': score_std,
            'PHI_MEAN': phi_mean,
            'THETA_MEAN': theta_mean,
            'PSI_MEAN': psi_mean,
            'ORIENTATION_STD': orientation_std,
            'DEFOCUS_MEAN': defocus_mean,
            'DEFOCUS_STD': defocus_std,
        }

        # Add per-search values
        for job_id in job_ids:
            p = cluster[job_id]
            result[f'SCORE_JOB{job_id}'] = p['score']
            result[f'X_JOB{job_id}'] = p['x']
            result[f'Y_JOB{job_id}'] = p['y']
            result[f'PHI_JOB{job_id}'] = p['phi']
            result[f'THETA_JOB{job_id}'] = p['theta']
            result[f'PSI_JOB{job_id}'] = p['psi']

        results.append(result)

    return results


def compute_pairwise_consensus(all_peaks: pd.DataFrame, job_ids: list[int],
                                position_tolerance: float) -> dict:
    """
    Compute pairwise consensus counts between all pairs of searches.

    Args:
        all_peaks: DataFrame with all peaks (must have JOB_ID and IMAGE_ASSET_ID columns)
        job_ids: List of job IDs
        position_tolerance: Max distance (Å) for peaks to match

    Returns:
        Dict mapping (job_i, job_j) -> count of matching peaks
    """
    pairwise = {}

    # Initialize counts
    for i in job_ids:
        for j in job_ids:
            pairwise[(i, j)] = 0

    # Process each image
    for image_id in all_peaks['IMAGE_ASSET_ID'].unique():
        img_peaks = all_peaks[all_peaks['IMAGE_ASSET_ID'] == image_id]

        # For each pair of searches
        for i, job_i in enumerate(job_ids):
            peaks_i = img_peaks[img_peaks['JOB_ID'] == job_i]

            for job_j in job_ids[i:]:  # Only upper triangle + diagonal
                peaks_j = img_peaks[img_peaks['JOB_ID'] == job_j]

                if job_i == job_j:
                    # Diagonal: count of peaks in this search
                    pairwise[(job_i, job_j)] += len(peaks_i)
                else:
                    # Off-diagonal: count matches between searches
                    matched = 0
                    used_j = set()

                    for _, pi in peaks_i.iterrows():
                        for idx_j, pj in peaks_j.iterrows():
                            if idx_j in used_j:
                                continue
                            dist = np.sqrt((pi['X_POSITION'] - pj['X_POSITION'])**2 +
                                           (pi['Y_POSITION'] - pj['Y_POSITION'])**2)
                            if dist <= position_tolerance:
                                matched += 1
                                used_j.add(idx_j)
                                break

                    pairwise[(job_i, job_j)] += matched
                    pairwise[(job_j, job_i)] += matched  # Symmetric

    return pairwise


def print_pairwise_table(pairwise: dict, job_ids: list[int]):
    """Print NxN pairwise consensus matrix."""
    n = len(job_ids)

    print(f"\n{'='*70}")
    print("  Pairwise Consensus Matrix (peaks matched between each pair)")
    print(f"{'='*70}")

    # Header row
    header = "        " + " ".join([f"{j:>7}" for j in job_ids])
    print(header)
    print("        " + "-" * (8 * n - 1))

    for job_i in job_ids:
        row = f"  {job_i:>5} |"
        for job_j in job_ids:
            count = pairwise[(job_i, job_j)]
            if job_i == job_j:
                # Diagonal: total peaks (in bold/different format)
                row += f" [{count:>5}]"
            else:
                row += f"  {count:>5} "
        print(row)

    print(f"{'='*70}")
    print("  [Diagonal] = total peaks in that search")
    print()


def print_debug_consensus_table(results: list[dict], job_ids: list[int],
                                 title: str = None, show_offsets: bool = False,
                                 subpixel_method: str = None):
    """
    Print formatted table of consensus peak statistics for debugging.

    Args:
        results: List of result dicts from compute_consensus_stats
        job_ids: List of job IDs for per-search columns
        title: Optional title for the table
        show_offsets: If True, print X and Y offset rows after each peak
        subpixel_method: 'com', 'fft', or None - used to show refined height for FFT
    """
    if not results:
        print(f"\n  No consensus peaks found")
        return

    # Count unique images
    n_images = len(set(r['IMAGE_ASSET_ID'] for r in results))

    print(f"\n{'='*130}")
    if title:
        print(f"  {title}")
    else:
        print(f"  All Consensus Peaks: {len(results)} peaks across {n_images} images")
    print(f"{'='*130}")

    # Build per-search header with job IDs (6 chars each to match score format)
    job_header = ' '.join([f"{j:>6}" for j in job_ids])

    # Header (with Image ID column)
    print(f"  {'Img':>4} {'#':>3} | {'X(Å)':>9} {'Y(Å)':>9} {'Pos σ':>6} | "
          f"{'Score':>7} {'min':>7} {'max':>7} {'σ':>6} | "
          f"{'Ang σ':>6} | {job_header}")
    print(f"  {'-'*4}-{'-'*3}-+-{'-'*9}-{'-'*9}-{'-'*6}-+-"
          f"{'-'*7}-{'-'*7}-{'-'*7}-{'-'*6}-+-"
          f"{'-'*6}-+-{'-'*len(job_header)}")

    # Blank prefix for offset rows (matches the stats columns width)
    blank_prefix = f"  {'':>4} {'':>3} | {'':>9} {'':>9} {'':>6} | {'':>7} {'':>7} {'':>7} {'':>6} | {'':>6} |"

    # Check if refined heights are available (FFT method)
    has_refined = subpixel_method == 'fft' and results and f'REFINED_HEIGHT_JOB{job_ids[0]}' in results[0]

    for r in results:
        # Build per-search score string
        per_search = ' '.join([f"{r[f'SCORE_JOB{j}']:6.2f}" for j in job_ids])

        print(f"  {r['IMAGE_ASSET_ID']:4} {r['CONSENSUS_PEAK_ID']:3} | "
              f"{r['X_CENTROID']:9.1f} {r['Y_CENTROID']:9.1f} {r['POSITION_STD']:6.2f} | "
              f"{r['SCORE_MEAN']:7.2f} {r['SCORE_MIN']:7.2f} {r['SCORE_MAX']:7.2f} {r['SCORE_STD']:6.3f} | "
              f"{r['ORIENTATION_STD']:6.2f} | {per_search}")

        # Print offset rows if requested
        if show_offsets:
            # Check if offset data exists
            has_offsets = f'X_OFFSET_JOB{job_ids[0]}' in r

            if has_offsets:
                # X offset row
                x_offsets = ' '.join([f"{r.get(f'X_OFFSET_JOB{j}', np.nan):6.2f}" for j in job_ids])
                print(f"{blank_prefix} X:{x_offsets[2:]}")  # Remove first 2 chars to make room for "X:"

                # Y offset row
                y_offsets = ' '.join([f"{r.get(f'Y_OFFSET_JOB{j}', np.nan):6.2f}" for j in job_ids])
                print(f"{blank_prefix} Y:{y_offsets[2:]}")  # Remove first 2 chars to make room for "Y:"

                # Refined height row (FFT only)
                if has_refined:
                    heights = ' '.join([f"{r.get(f'REFINED_HEIGHT_JOB{j}', np.nan):6.2f}" for j in job_ids])
                    print(f"{blank_prefix} H:{heights[2:]}")  # H for height

    # Print averages row
    print(f"  {'-'*4}-{'-'*3}-+-{'-'*9}-{'-'*9}-{'-'*6}-+-"
          f"{'-'*7}-{'-'*7}-{'-'*7}-{'-'*6}-+-"
          f"{'-'*6}-+-{'-'*len(job_header)}")

    avg_pos_std = np.mean([r['POSITION_STD'] for r in results])
    avg_score_mean = np.mean([r['SCORE_MEAN'] for r in results])
    avg_score_min = np.mean([r['SCORE_MIN'] for r in results])
    avg_score_max = np.mean([r['SCORE_MAX'] for r in results])
    avg_score_std = np.mean([r['SCORE_STD'] for r in results])
    avg_orient_std = np.mean([r['ORIENTATION_STD'] for r in results])
    avg_per_search = ' '.join([f"{np.mean([r[f'SCORE_JOB{j}'] for r in results]):6.2f}" for j in job_ids])

    print(f"  {'AVG':>4} {'':>3} | {'':>9} {'':>9} {avg_pos_std:6.2f} | "
          f"{avg_score_mean:7.2f} {avg_score_min:7.2f} {avg_score_max:7.2f} {avg_score_std:6.3f} | "
          f"{avg_orient_std:6.2f} | {avg_per_search}")

    # Print average offsets if available
    if show_offsets and results and f'X_OFFSET_JOB{job_ids[0]}' in results[0]:
        avg_x_offsets = ' '.join([f"{np.nanmean([r.get(f'X_OFFSET_JOB{j}', np.nan) for r in results]):6.2f}" for j in job_ids])
        avg_y_offsets = ' '.join([f"{np.nanmean([r.get(f'Y_OFFSET_JOB{j}', np.nan) for r in results]):6.2f}" for j in job_ids])
        print(f"{blank_prefix} X:{avg_x_offsets[2:]}")
        print(f"{blank_prefix} Y:{avg_y_offsets[2:]}")

        # Average refined heights (FFT only)
        if has_refined:
            avg_heights = ' '.join([f"{np.nanmean([r.get(f'REFINED_HEIGHT_JOB{j}', np.nan) for r in results]):6.2f}" for j in job_ids])
            print(f"{blank_prefix} H:{avg_heights[2:]}")

    print(f"{'='*130}\n")


def analyze_consensus(analyzer: tma.TemplateMatchAnalyzer,
                      job_ids: list[int],
                      position_tolerance: float = 10.0,
                      min_peaks: int = 3,
                      debug_images: int = 0,
                      subpixel_method: str = None,
                      fft_upsample: int = 10,
                      window_size: int = 5) -> pd.DataFrame:
    """
    Main analysis function: find consensus peaks across multiple searches.

    Args:
        analyzer: Initialized TemplateMatchAnalyzer
        job_ids: List of search job IDs to compare
        position_tolerance: Max distance (Å) for peaks to be considered same
        min_peaks: Minimum peaks per image in ALL searches to include
        debug_images: Number of images to process with verbose output (0 = all images, no debug)
        subpixel_method: 'com' (center-of-mass), 'fft' (FFT upsampling), or None (disabled)
        fft_upsample: For FFT method, upsampling factor (default 10)
        window_size: Window size in pixels for sub-pixel estimation (default 5)

    Returns:
        DataFrame with consensus peak statistics
    """
    print(f"Loading peaks from {len(job_ids)} searches: {job_ids}")

    # Load peaks from all searches (already includes TEMPLATE_MATCH_JOB_ID column)
    all_peaks = analyzer.load_all_peaks_for_jobs(job_ids)

    if len(all_peaks) == 0:
        print("No peaks found in any search")
        return pd.DataFrame()

    # Rename for clarity
    all_peaks = all_peaks.rename(columns={'TEMPLATE_MATCH_JOB_ID': 'JOB_ID'})

    # Find images present in ALL jobs
    images_by_job = {}
    for job_id in job_ids:
        job_peaks = all_peaks[all_peaks['JOB_ID'] == job_id]
        images_by_job[job_id] = set(job_peaks['IMAGE_ASSET_ID'].unique())

    # Intersection of all image sets
    common_images = set.intersection(*images_by_job.values()) if images_by_job else set()
    print(f"Images analyzed in ALL searches: {len(common_images)}")

    if not common_images:
        print("No images found that were analyzed in all searches")
        return pd.DataFrame()

    # Filter to common images
    all_peaks = all_peaks[all_peaks['IMAGE_ASSET_ID'].isin(common_images)]

    # Apply min-peaks filter
    if min_peaks > 0:
        # Count peaks per image per job
        valid_images = set(common_images)
        for job_id in job_ids:
            job_peaks = all_peaks[all_peaks['JOB_ID'] == job_id]
            counts = job_peaks.groupby('IMAGE_ASSET_ID').size()
            valid_for_job = set(counts[counts >= min_peaks].index)
            valid_images &= valid_for_job

        excluded = len(common_images) - len(valid_images)
        if excluded > 0:
            print(f"  Filtered out {excluded} image(s) with <{min_peaks} peaks in any search")

        all_peaks = all_peaks[all_peaks['IMAGE_ASSET_ID'].isin(valid_images)]
        common_images = valid_images

    print(f"  Processing {len(common_images)} images")

    # Report per-job peak counts
    print("\nPer-search peak counts:")
    job_peak_counts = {}
    for job_id in job_ids:
        count = len(all_peaks[all_peaks['JOB_ID'] == job_id])
        job_peak_counts[job_id] = count
        print(f"  Job {job_id}: {count} peaks")

    # Process each image
    results = []
    image_list = sorted(common_images)
    debug = debug_images > 0

    if debug:
        image_list = image_list[:debug_images]
        print(f"\n[DEBUG MODE] Processing {len(image_list)} image(s)")

        # Compute and print pairwise consensus matrix
        debug_peaks = all_peaks[all_peaks['IMAGE_ASSET_ID'].isin(image_list)]
        pairwise = compute_pairwise_consensus(debug_peaks, job_ids, position_tolerance)
        print_pairwise_table(pairwise, job_ids)

    for img_idx, image_id in enumerate(image_list):
        if not debug:
            print(f"  Processing image {img_idx + 1}/{len(image_list)}...", end='\r')
        else:
            print(f"  Processing image {img_idx + 1}/{len(image_list)} (ID: {image_id})")

        # Get peaks for this image, grouped by job
        img_peaks = all_peaks[all_peaks['IMAGE_ASSET_ID'] == image_id]
        peaks_by_job = {
            job_id: img_peaks[img_peaks['JOB_ID'] == job_id]
            for job_id in job_ids
        }

        if debug:
            for job_id in job_ids:
                print(f"    Job {job_id}: {len(peaks_by_job[job_id])} peaks")

        # Find consensus peaks
        consensus = find_consensus_peaks(peaks_by_job, job_ids, position_tolerance, debug)

        # Compute statistics
        img_results = compute_consensus_stats(consensus, job_ids, image_id)
        results.extend(img_results)

    print()  # Clear progress line

    # Compute sub-pixel offsets if requested (Phase 2: after all consensus peaks are found)
    if subpixel_method and results:
        compute_subpixel_offsets(analyzer, results, job_ids,
                                  method=subpixel_method, upsample_factor=fft_upsample,
                                  window_size=window_size, debug=debug)

    # Print combined debug table at the end
    if debug and results:
        print_debug_consensus_table(results, job_ids,
                                     show_offsets=(subpixel_method is not None),
                                     subpixel_method=subpixel_method)

    return pd.DataFrame(results)


def print_summary(df: pd.DataFrame, job_ids: list[int], job_peak_counts: dict = None):
    """Print summary statistics of the consensus analysis."""

    print("\n" + "=" * 70)
    print("CONSENSUS ANALYSIS SUMMARY")
    print("=" * 70)

    if len(df) == 0:
        print("No consensus peaks found")
        return

    n_images = df['IMAGE_ASSET_ID'].nunique()
    n_peaks = len(df)

    print(f"\nConsensus peaks: {n_peaks} across {n_images} images")
    print(f"  (peaks present in ALL {len(job_ids)} searches)")

    # Score consistency
    print(f"\nScore consistency (std across searches):")
    print(f"  Mean score std:  {df['SCORE_STD'].mean():.3f}")
    print(f"  Median score std: {df['SCORE_STD'].median():.3f}")
    print(f"  Max score std:   {df['SCORE_STD'].max():.3f}")

    # Position consistency
    print(f"\nPosition consistency:")
    print(f"  Mean position std:  {df['POSITION_STD'].mean():.2f} Å")
    print(f"  Median position std: {df['POSITION_STD'].median():.2f} Å")
    print(f"  Max position std:   {df['POSITION_STD'].max():.2f} Å")

    # Orientation consistency
    print(f"\nOrientation consistency:")
    print(f"  Mean orientation std:  {df['ORIENTATION_STD'].mean():.2f}°")
    print(f"  Median orientation std: {df['ORIENTATION_STD'].median():.2f}°")
    print(f"  Max orientation std:   {df['ORIENTATION_STD'].max():.2f}°")

    # Score ranges
    print(f"\nScore statistics:")
    print(f"  Overall mean score: {df['SCORE_MEAN'].mean():.3f}")
    print(f"  Mean score range (max-min): {(df['SCORE_MAX'] - df['SCORE_MIN']).mean():.3f}")


def main():
    parser = argparse.ArgumentParser(
        description='Find consensus peaks across multiple template matching searches',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script finds peaks that are consistently detected across ALL specified searches
and computes statistics on their variability.

Range formats:
  12:18     - Jobs 12 through 18 (inclusive)
  12,14,16  - Jobs 12, 14, and 16
  12        - Single job 12

Examples:
  %(prog)s /path/to/project.db 12:18 --position-tolerance 5.0
  %(prog)s /path/to/project.db 12,14,16,18 --min-peaks 3 --output consensus.csv
        """
    )
    parser.add_argument('db_path', help='Path to cisTEM database file')
    parser.add_argument('search_range', help='Search job IDs (e.g., "12:18" or "12,14,16")')
    parser.add_argument('--position-tolerance', type=float, default=10.0,
                        help='Max distance (Å) for peaks to be same (default: 10.0)')
    parser.add_argument('--min-peaks', type=int, default=3,
                        help='Min peaks per image in ALL searches (default: 3)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output CSV file for detailed results')
    parser.add_argument('--debug', type=int, nargs='?', const=1, default=0, metavar='N',
                        help='Debug mode: process N images with verbose output (default: 1 if flag given)')
    parser.add_argument('--subpixel-method', type=str, choices=['com', 'fft'], default=None,
                        help='Sub-pixel estimation: "com" (center-of-mass) or "fft" (FFT upsampling)')
    parser.add_argument('--fft-upsample', type=int, default=10,
                        help='FFT upsampling factor (default: 10, gives 0.1 pixel precision)')
    parser.add_argument('--window-size', type=int, default=5,
                        help='Window size in pixels for sub-pixel estimation (default: 5, must be odd)')

    args = parser.parse_args()

    # Parse search range
    try:
        job_ids = parse_search_range(args.search_range)
    except ValueError as e:
        print(f"Error parsing search range: {e}", file=sys.stderr)
        sys.exit(1)

    if len(job_ids) < 2:
        print("Error: Need at least 2 searches to find consensus", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("Multi-Search Consensus Analysis")
    print("=" * 70)
    print(f"Database: {args.db_path}")
    print(f"Searches: {job_ids} ({len(job_ids)} searches)")
    print(f"Position tolerance: {args.position_tolerance} Å")
    print(f"Minimum peaks per image: {args.min_peaks}")
    print()

    try:
        # Initialize analyzer
        print("Initializing analyzer...")
        analyzer = tma.TemplateMatchAnalyzer(args.db_path)
        print("✓ Database validated")
        print()

        # Run analysis
        results_df = analyze_consensus(
            analyzer,
            job_ids,
            position_tolerance=args.position_tolerance,
            min_peaks=args.min_peaks,
            debug_images=args.debug,
            subpixel_method=args.subpixel_method,
            fft_upsample=args.fft_upsample,
            window_size=args.window_size
        )

        if len(results_df) == 0:
            print("No consensus peaks found.")
            sys.exit(0)

        # Print summary
        print_summary(results_df, job_ids)

        # Save results if requested
        if args.output:
            results_df.to_csv(args.output, index=False)
            print(f"\n✓ Results saved to: {args.output}")

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
