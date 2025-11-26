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
"""

import sys
import argparse
import numpy as np
import pandas as pd
import mrcfile
import template_match_analysis as tma


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
        [-s2*c3,             s2*s3,              c2   ]
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
                     angle_tolerance: float = 15.0) -> pd.DataFrame:
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

    # Load peaks for search1
    peaks1 = analyzer.load_all_peaks_for_jobs([job_id1])
    # Filter to overlapping images
    peaks1 = peaks1[peaks1['IMAGE_ASSET_ID'].isin(overlap)]

    # Load peaks for search2 (to check for detected peaks)
    peaks2 = analyzer.load_all_peaks_for_jobs([job_id2])
    peaks2 = peaks2[peaks2['IMAGE_ASSET_ID'].isin(overlap)]

    print(f"Search1 (job {job_id1}): {len(peaks1)} peaks across {peaks1['IMAGE_ASSET_ID'].nunique()} images")
    print(f"Search2 (job {job_id2}): {len(peaks2)} peaks across {peaks2['IMAGE_ASSET_ID'].nunique()} images")

    # Build lookup for search2 file paths by IMAGE_ASSET_ID
    paths2_lookup = paths2.set_index('IMAGE_ASSET_ID').to_dict('index')

    # Build spatial index for search2 peaks (for detecting matching peaks)
    peaks2_by_image = {
        img_id: group[['X_POSITION', 'Y_POSITION', 'PEAK_NUMBER', 'PEAK_HEIGHT',
                       'PSI', 'THETA', 'PHI', 'DEFOCUS']].values
        for img_id, group in peaks2.groupby('IMAGE_ASSET_ID')
    }

    # Results accumulator
    results = []

    # Process each image
    unique_images = peaks1['IMAGE_ASSET_ID'].unique()
    for img_idx, image_id in enumerate(unique_images):
        print(f"  Processing image {img_idx + 1}/{len(unique_images)} (ID: {image_id})...", end='\r')

        # Get search2 file paths for this image
        if image_id not in paths2_lookup:
            print(f"  Warning: No search2 paths for image {image_id}")
            continue

        paths2_img = paths2_lookup[image_id]

        # Load search2 output images (returns image and pixel_size)
        try:
            scaled_mip, pixel_size = load_mrc_image(paths2_img['SCALED_MIP_OUTPUT_FILE'])
            psi_img, _ = load_mrc_image(paths2_img['PSI_OUTPUT_FILE'])
            theta_img, _ = load_mrc_image(paths2_img['THETA_OUTPUT_FILE'])
            phi_img, _ = load_mrc_image(paths2_img['PHI_OUTPUT_FILE'])
            defocus_img, _ = load_mrc_image(paths2_img['DEFOCUS_OUTPUT_FILE'])
        except FileNotFoundError as e:
            print(f"\n  Warning: Missing file for image {image_id}: {e}")
            continue
        except Exception as e:
            print(f"\n  Warning: Error loading images for {image_id}: {e}")
            continue

        # Get search1 peaks for this image
        img_peaks1 = peaks1[peaks1['IMAGE_ASSET_ID'] == image_id]

        # Get search2 peaks for this image (for matching)
        img_peaks2 = peaks2_by_image.get(image_id, np.array([]).reshape(0, 8))

        # Extract positions in Angstroms from database, convert to pixels
        # Database stores positions in Angstroms; images are indexed by pixels
        positions_angstroms = img_peaks1[['X_POSITION', 'Y_POSITION']].values
        positions_pixels = positions_angstroms / pixel_size

        # Sample search2 images at search1 peak positions (in pixels)
        scores2 = sample_at_positions(scaled_mip, positions_pixels)
        psi2 = sample_at_positions(psi_img, positions_pixels)
        theta2 = sample_at_positions(theta_img, positions_pixels)
        phi2 = sample_at_positions(phi_img, positions_pixels)
        defocus2 = sample_at_positions(defocus_img, positions_pixels)

        # Build 1:1 peak matching with validation
        # First pass: find nearest search2 peak for each search1 peak
        peak1_to_peak2 = {}  # search1_idx -> (search2_idx, distance)

        if len(img_peaks2) > 0:
            peaks2_x = np.asarray(img_peaks2[:, 0], dtype=np.float64)
            peaks2_y = np.asarray(img_peaks2[:, 1], dtype=np.float64)

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
                peak2_x = float(img_peaks2[peak2_idx, 0])
                peak2_y = float(img_peaks2[peak2_idx, 1])
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
                peak2_psi = float(img_peaks2[nearest_peak2_idx, 4])
                peak2_theta = float(img_peaks2[nearest_peak2_idx, 5])
                peak2_phi = float(img_peaks2[nearest_peak2_idx, 6])
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
                peak2_data = img_peaks2[nearest_peak2_idx]
                result['SEARCH2_DETECTED_SCORE'] = peak2_data[3]  # PEAK_HEIGHT
                result['SEARCH2_DETECTED_PSI'] = peak2_data[4]
                result['SEARCH2_DETECTED_THETA'] = peak2_data[5]
                result['SEARCH2_DETECTED_PHI'] = peak2_data[6]
                result['SEARCH2_DETECTED_DEFOCUS'] = peak2_data[7]
            else:
                result['SEARCH2_DETECTED_SCORE'] = np.nan
                result['SEARCH2_DETECTED_PSI'] = np.nan
                result['SEARCH2_DETECTED_THETA'] = np.nan
                result['SEARCH2_DETECTED_PHI'] = np.nan
                result['SEARCH2_DETECTED_DEFOCUS'] = np.nan

            results.append(result)

    print()  # Clear the progress line

    return pd.DataFrame(results)


def print_summary(df: pd.DataFrame, job_id1: int, job_id2: int,
                  position_tolerance: float, angle_tolerance: float):
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

    print(f"\nTotal peaks in search1: {total_peaks}")
    print(f"  Found in search2:     {found} ({100*found/total_peaks:.1f}%)")
    print(f"  Missed in search2:    {missed} ({100*missed/total_peaks:.1f}%)")
    if found > 0:
        print(f"  Orientation matches:  {orientation_match_found}/{found} ({100*orientation_match_found/found:.1f}% of found peaks)")

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

    args = parser.parse_args()

    print("=" * 70)
    print("Cross-Search Comparison Analysis")
    print("=" * 70)
    print(f"Database: {args.db_path}")
    print(f"Reference search (job1): {args.job_id1}")
    print(f"Comparison search (job2): {args.job_id2}")
    print(f"Position tolerance: {args.position_tolerance} pixels")
    print(f"Angle tolerance: {args.angle_tolerance} degrees")
    print()

    try:
        # Initialize analyzer
        print("Initializing analyzer...")
        analyzer = tma.TemplateMatchAnalyzer(args.db_path)
        print("✓ Database validated")
        print()

        # Run comparison
        print("Running comparison...")
        results_df = compare_searches(
            analyzer,
            args.job_id1,
            args.job_id2,
            position_tolerance=args.position_tolerance,
            angle_tolerance=args.angle_tolerance
        )

        if len(results_df) == 0:
            print("No overlapping data to compare.")
            sys.exit(0)

        # Print summary
        print_summary(results_df, args.job_id1, args.job_id2,
                      args.position_tolerance, args.angle_tolerance)

        # Save detailed results if requested
        if args.output:
            results_df.to_csv(args.output, index=False)
            print(f"\n✓ Detailed results saved to: {args.output}")

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
