#!/usr/bin/env python3
"""
FFT Peak Recovery with Component Upsampling

This script tests whether upsampling the raw MIP components (MIP, average, stddev)
BEFORE computing the scaled value yields higher/sharper peaks than upsampling
the already-computed scaled MIP.

The scaling formula is: scaled_mip = (mip - avg) / std

Hypothesis: If a peak is spread across multiple pixels due to sub-pixel positioning,
upsampling all three components first and then computing the ratio at sub-pixel
resolution may give a more accurate peak height.

Usage:
    python peak_recovery_fft_components.py <db_path> <job_id> [options]

Examples:
    # Run sanity check only (verify scaling formula)
    python peak_recovery_fft_components.py cp.db 12 --sanity-check-only

    # Compare both methods
    python peak_recovery_fft_components.py cp.db 12 --method compare --debug 3
"""

import sys
import argparse
import sqlite3
import numpy as np
import pandas as pd
import mrcfile
from cistemx import parse_job_range
from cistemx.db import database as tma


def get_job_threshold(conn: sqlite3.Connection, job_id: int) -> float:
    """Get the USED_THRESHOLD for a job from TEMPLATE_MATCH_LIST."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT USED_THRESHOLD FROM TEMPLATE_MATCH_LIST
        WHERE TEMPLATE_MATCH_JOB_ID = ? LIMIT 1
    """, (job_id,))
    result = cursor.fetchone()
    if result is None:
        raise ValueError(f"No results found for job {job_id}")
    return float(result[0])


def get_images_for_job(conn: sqlite3.Connection, job_id: int) -> list[int]:
    """Get list of IMAGE_ASSET_IDs analyzed in a job."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT IMAGE_ASSET_ID FROM TEMPLATE_MATCH_LIST
        WHERE TEMPLATE_MATCH_JOB_ID = ?
        ORDER BY IMAGE_ASSET_ID
    """, (job_id,))
    return [row[0] for row in cursor.fetchall()]


def load_all_mip_components(analyzer, job_id: int, image_id: int
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Load all four MIP component images for a job/image combination.

    Returns:
        (mip, scaled_mip, avg, std, pixel_size) tuple
        All images are 2D float32 arrays.
    """
    paths_df = analyzer.get_result_file_paths(job_id, {image_id})
    if len(paths_df) == 0:
        raise ValueError(f"No paths found for job {job_id}, image {image_id}")

    row = paths_df.iloc[0]

    def load_mrc(path: str) -> tuple[np.ndarray, float]:
        with mrcfile.open(path, permissive=True) as mrc:
            data = np.squeeze(mrc.data).astype(np.float32)
            pixel_size = float(mrc.voxel_size.x)
        return data, pixel_size

    mip, pixel_size = load_mrc(row['MIP_OUTPUT_FILE'])
    scaled_mip, _ = load_mrc(row['SCALED_MIP_OUTPUT_FILE'])
    avg, _ = load_mrc(row['AVG_OUTPUT_FILE'])
    std, _ = load_mrc(row['STD_OUTPUT_FILE'])

    return mip, scaled_mip, avg, std, pixel_size


def verify_scaling_formula(mip: np.ndarray, avg: np.ndarray, std: np.ndarray,
                           scaled_mip: np.ndarray, debug: bool = False) -> dict:
    """
    Verify that scaled_mip = (mip - avg) / std.

    Returns dict with verification results. Raises ValueError if mismatch detected.
    """
    # Compute expected scaled values
    # Handle std=0 by setting those positions to 0 (like the C++ code likely does)
    with np.errstate(divide='ignore', invalid='ignore'):
        computed = (mip - avg) / std
        # Where std is 0 or very small, the result is undefined
        # Check what the stored scaled_mip has at these locations
        zero_std_mask = np.abs(std) < 1e-10

    # Replace inf/nan with 0 for comparison
    computed = np.nan_to_num(computed, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute difference only where std is reasonable
    valid_mask = ~zero_std_mask

    if valid_mask.sum() == 0:
        raise ValueError("No valid pixels (all std values are near zero)")

    diff = computed - scaled_mip
    valid_diff = diff[valid_mask]

    max_abs_diff = np.max(np.abs(valid_diff))
    mean_abs_diff = np.mean(np.abs(valid_diff))
    rms_diff = np.sqrt(np.mean(valid_diff ** 2))

    # Check if scaled_mip values match at zero_std locations
    zero_std_scaled_values = scaled_mip[zero_std_mask] if zero_std_mask.sum() > 0 else np.array([])

    results = {
        'max_abs_diff': max_abs_diff,
        'mean_abs_diff': mean_abs_diff,
        'rms_diff': rms_diff,
        'n_valid_pixels': int(valid_mask.sum()),
        'n_zero_std_pixels': int(zero_std_mask.sum()),
        'zero_std_scaled_range': (float(zero_std_scaled_values.min()),
                                   float(zero_std_scaled_values.max())) if len(zero_std_scaled_values) > 0 else None
    }

    if debug:
        print(f"    Scaling verification:")
        print(f"      Max abs diff: {max_abs_diff:.6e}")
        print(f"      Mean abs diff: {mean_abs_diff:.6e}")
        print(f"      RMS diff: {rms_diff:.6e}")
        print(f"      Valid pixels: {results['n_valid_pixels']} / {mip.size}")
        print(f"      Zero-std pixels: {results['n_zero_std_pixels']}")

    # Fail if difference is significant
    # Using a tolerance based on float16 precision (used for MRC storage)
    # float16 has ~3 decimal digits of precision
    tolerance = 1e-2  # Allow for float16 quantization

    if max_abs_diff > tolerance:
        raise ValueError(
            f"Scaling formula mismatch! Max difference {max_abs_diff:.6e} exceeds tolerance {tolerance}. "
            f"Mean diff: {mean_abs_diff:.6e}, RMS diff: {rms_diff:.6e}"
        )

    return results


def fft_upsample_window(image: np.ndarray, x_pixel: int, y_pixel: int,
                        window_half: int, upsample_factor: int) -> np.ndarray:
    """
    Extract a window around (x_pixel, y_pixel) and upsample via FFT zero-padding.

    Returns the upsampled window.
    """
    ny, nx = image.shape
    x0 = max(0, x_pixel - window_half)
    x1 = min(nx, x_pixel + window_half + 1)
    y0 = max(0, y_pixel - window_half)
    y1 = min(ny, y_pixel + window_half + 1)

    window = image[y0:y1, x0:x1]
    win_ny, win_nx = window.shape

    # FFT
    fft_window = np.fft.fft2(window)

    # Zero-pad in frequency domain
    pad_ny = win_ny * upsample_factor
    pad_nx = win_nx * upsample_factor
    padded = np.zeros((pad_ny, pad_nx), dtype=complex)

    # Copy frequencies to padded array (preserving frequency layout)
    ny_pos = (win_ny + 1) // 2
    ny_neg = win_ny // 2
    nx_pos = (win_nx + 1) // 2
    nx_neg = win_nx // 2

    padded[:ny_pos, :nx_pos] = fft_window[:ny_pos, :nx_pos]
    if nx_neg > 0:
        padded[:ny_pos, -nx_neg:] = fft_window[:ny_pos, -nx_neg:]
    if ny_neg > 0:
        padded[-ny_neg:, :nx_pos] = fft_window[-ny_neg:, :nx_pos]
    if ny_neg > 0 and nx_neg > 0:
        padded[-ny_neg:, -nx_neg:] = fft_window[-ny_neg:, -nx_neg:]

    # Inverse FFT (scale by upsample_factor^2 to preserve amplitude)
    upsampled = np.fft.ifft2(padded).real * (upsample_factor ** 2)

    return upsampled


def compute_subpixel_from_scaled_fft(scaled_mip: np.ndarray, x_pixel: int, y_pixel: int,
                                      window_half: int = 2, upsample_factor: int = 10,
                                      verbose: bool = False
                                      ) -> tuple[float, float, float]:
    """
    Method A: Upsample the already-computed scaled MIP.

    Returns: (x_offset, y_offset, refined_height)
    """
    upsampled = fft_upsample_window(scaled_mip, x_pixel, y_pixel, window_half, upsample_factor)

    max_idx = np.unravel_index(np.argmax(upsampled), upsampled.shape)
    max_y_up, max_x_up = max_idx

    refined_height = upsampled[max_y_up, max_x_up]

    if verbose:
        center = upsampled.shape[0] // 2
        print(f"        [Method A] Upsampled scaled_mip: shape={upsampled.shape}, "
              f"center={upsampled[center,center]:.4f}, peak at ({max_x_up},{max_y_up})={refined_height:.4f}")

    # Convert upsampled position back to original coordinates
    win_size = 2 * window_half + 1
    center_up = win_size * upsample_factor // 2

    x_offset = abs(max_x_up - center_up) / upsample_factor
    y_offset = abs(max_y_up - center_up) / upsample_factor

    x_offset = min(0.5, x_offset)
    y_offset = min(0.5, y_offset)

    return x_offset, y_offset, refined_height


def compute_subpixel_from_components_fft(mip: np.ndarray, avg: np.ndarray, std: np.ndarray,
                                          x_pixel: int, y_pixel: int,
                                          window_half: int = 2, upsample_factor: int = 10,
                                          verbose: bool = False
                                          ) -> tuple[float, float, float, float]:
    """
    Method B: Upsample all three components, then compute scaled value.

    Returns: (x_offset, y_offset, refined_scaled_height, refined_raw_height)
    """
    # Upsample each component
    mip_up = fft_upsample_window(mip, x_pixel, y_pixel, window_half, upsample_factor)
    avg_up = fft_upsample_window(avg, x_pixel, y_pixel, window_half, upsample_factor)
    std_up = fft_upsample_window(std, x_pixel, y_pixel, window_half, upsample_factor)

    if verbose:
        # Show intermediate values at center of upsampled window
        center = mip_up.shape[0] // 2
        print(f"        [verbose] At center: mip_up={mip_up[center,center]:.4f}, "
              f"avg_up={avg_up[center,center]:.4f}, std_up={std_up[center,center]:.4f}")
        print(f"        [verbose] Upsampled shapes: mip={mip_up.shape}, ranges: "
              f"mip=[{mip_up.min():.2f},{mip_up.max():.2f}] "
              f"std=[{std_up.min():.4f},{std_up.max():.4f}]")

    # Compute scaled value at upsampled resolution
    with np.errstate(divide='ignore', invalid='ignore'):
        scaled_up = (mip_up - avg_up) / std_up
        # Handle division by zero
        scaled_up = np.nan_to_num(scaled_up, nan=0.0, posinf=0.0, neginf=0.0)

    # Find peak in scaled upsampled image
    max_idx = np.unravel_index(np.argmax(scaled_up), scaled_up.shape)
    max_y_up, max_x_up = max_idx

    refined_scaled_height = scaled_up[max_y_up, max_x_up]
    refined_raw_height = mip_up[max_y_up, max_x_up]

    if verbose:
        print(f"        [verbose] Peak at ({max_x_up},{max_y_up}): "
              f"mip={mip_up[max_y_up,max_x_up]:.4f}, avg={avg_up[max_y_up,max_x_up]:.4f}, "
              f"std={std_up[max_y_up,max_x_up]:.4f} -> scaled={refined_scaled_height:.4f}")

    # Convert upsampled position back to original coordinates
    win_size = 2 * window_half + 1
    center_up = win_size * upsample_factor // 2

    x_offset = abs(max_x_up - center_up) / upsample_factor
    y_offset = abs(max_y_up - center_up) / upsample_factor

    x_offset = min(0.5, x_offset)
    y_offset = min(0.5, y_offset)

    return x_offset, y_offset, refined_scaled_height, refined_raw_height


def apply_circular_mask(image: np.ndarray, cx: int, cy: int, radius: int):
    """Zero out circular region around (cx, cy). Modifies in-place."""
    ny, nx = image.shape
    y, x = np.ogrid[:ny, :nx]
    mask = (x - cx)**2 + (y - cy)**2 <= radius**2
    image[mask] = image.min()


def find_peaks_compare_methods(mip: np.ndarray, scaled_mip: np.ndarray,
                                avg: np.ndarray, std: np.ndarray,
                                threshold: float, pixel_size: float,
                                search_offset: float = 0.8,
                                mask_radius: int = 5,
                                upsample_factor: int = 10,
                                window_half: int = 2,
                                debug: bool = False) -> list[dict]:
    """
    Find peaks and compare both refinement methods.

    Returns list of peaks with heights from both methods.
    """
    working = scaled_mip.copy()
    peaks = []
    search_threshold = search_offset * threshold
    peak_num = 0

    while True:
        # Find maximum in working copy
        max_idx = np.unravel_index(np.argmax(working), working.shape)
        y, x = max_idx
        raw_height = working[y, x]

        # Stop if below search threshold
        if raw_height < search_threshold:
            if debug:
                print(f"      Stop: raw={raw_height:.2f} < search_thr={search_threshold:.2f}")
            break

        peak_num += 1

        # Show verbose output for first 3 peaks in debug mode
        verbose = debug and peak_num <= 3

        # Method A: Upsample scaled MIP directly
        x_off_a, y_off_a, height_a = compute_subpixel_from_scaled_fft(
            scaled_mip, x, y, window_half, upsample_factor, verbose=verbose)

        # Method B: Upsample components, then compute ratio
        x_off_b, y_off_b, height_b, raw_b = compute_subpixel_from_components_fft(
            mip, avg, std, x, y, window_half, upsample_factor, verbose=verbose)

        # Use Method A's refined height for threshold decision
        kept = height_a >= threshold
        recovered = raw_height < threshold

        diff = height_b - height_a
        diff_pct = 100 * diff / height_a if height_a != 0 else 0

        if debug:
            status = "KEPT" if kept else "skip"
            print(f"      Peak {peak_num:3d}: ({x:4d},{y:4d}) raw={raw_height:.2f} "
                  f"A={height_a:.2f} B={height_b:.2f} diff={diff:+.3f} ({diff_pct:+.2f}%) {status}")

        if kept:
            peaks.append({
                'x_pixel': x,
                'y_pixel': y,
                'x_ang': x * pixel_size,
                'y_ang': y * pixel_size,
                'raw_height': raw_height,
                'height_method_a': height_a,
                'height_method_b': height_b,
                'raw_height_b': raw_b,
                'x_offset_a': x_off_a,
                'y_offset_a': y_off_a,
                'x_offset_b': x_off_b,
                'y_offset_b': y_off_b,
                'diff_b_minus_a': diff,
                'diff_pct': diff_pct,
                'recovered': recovered
            })

        # Mask out this peak in working copy
        apply_circular_mask(working, x, y, mask_radius)

    if debug:
        print(f"      Total: {len(peaks)} peaks kept out of {peak_num} examined")

    return peaks


def analyze_image_sanity(analyzer, job_id: int, image_id: int, debug: bool = False) -> dict:
    """
    Run sanity check on one image.
    """
    mip, scaled_mip, avg, std, pixel_size = load_all_mip_components(analyzer, job_id, image_id)
    results = verify_scaling_formula(mip, avg, std, scaled_mip, debug=debug)
    return results


def analyze_image_compare(analyzer, job_id: int, image_id: int,
                          threshold: float, search_offset: float,
                          mask_radius: int, upsample_factor: int,
                          window_half: int, debug: bool = False) -> list[dict]:
    """
    Analyze one image comparing both methods.
    """
    mip, scaled_mip, avg, std, pixel_size = load_all_mip_components(analyzer, job_id, image_id)

    # First verify scaling
    verify_scaling_formula(mip, avg, std, scaled_mip, debug=debug)

    # Find peaks and compare methods
    peaks = find_peaks_compare_methods(
        mip, scaled_mip, avg, std,
        threshold, pixel_size,
        search_offset=search_offset,
        mask_radius=mask_radius,
        upsample_factor=upsample_factor,
        window_half=window_half,
        debug=debug
    )

    # Add image info to each peak
    for p in peaks:
        p['job_id'] = job_id
        p['image_id'] = image_id

    return peaks


def print_comparison_summary(all_peaks: list[dict]):
    """Print summary statistics comparing the two methods."""
    if not all_peaks:
        print("No peaks to compare.")
        return

    df = pd.DataFrame(all_peaks)

    print("\n" + "=" * 80)
    print("METHOD COMPARISON SUMMARY")
    print("=" * 80)
    print(f"\nTotal peaks analyzed: {len(df)}")

    # Overall statistics
    mean_diff = df['diff_b_minus_a'].mean()
    median_diff = df['diff_b_minus_a'].median()
    std_diff = df['diff_b_minus_a'].std()
    mean_pct = df['diff_pct'].mean()

    print(f"\nHeight difference (Method B - Method A):")
    print(f"  Mean:   {mean_diff:+.4f} ({mean_pct:+.3f}%)")
    print(f"  Median: {median_diff:+.4f}")
    print(f"  Std:    {std_diff:.4f}")
    print(f"  Min:    {df['diff_b_minus_a'].min():+.4f}")
    print(f"  Max:    {df['diff_b_minus_a'].max():+.4f}")

    # How often is B better?
    b_better = (df['diff_b_minus_a'] > 0).sum()
    a_better = (df['diff_b_minus_a'] < 0).sum()
    equal = (df['diff_b_minus_a'] == 0).sum()

    print(f"\nMethod comparison:")
    print(f"  B better: {b_better} ({100*b_better/len(df):.1f}%)")
    print(f"  A better: {a_better} ({100*a_better/len(df):.1f}%)")
    print(f"  Equal:    {equal} ({100*equal/len(df):.1f}%)")

    # By recovery status
    if 'recovered' in df.columns:
        recovered = df[df['recovered'] == True]
        not_recovered = df[df['recovered'] == False]

        if len(recovered) > 0:
            print(f"\nFor recovered peaks (raw < threshold): {len(recovered)}")
            print(f"  Mean diff: {recovered['diff_b_minus_a'].mean():+.4f}")

        if len(not_recovered) > 0:
            print(f"\nFor non-recovered peaks (raw >= threshold): {len(not_recovered)}")
            print(f"  Mean diff: {not_recovered['diff_b_minus_a'].mean():+.4f}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='FFT Peak Recovery with Component Upsampling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script compares two methods for FFT-based peak refinement:
  Method A: Upsample the scaled MIP directly (current approach)
  Method B: Upsample raw components (MIP, avg, std), then compute ratio

Examples:
  %(prog)s cp.db 12 --sanity-check-only
  %(prog)s cp.db 12 --method compare --debug 3
        """
    )
    parser.add_argument('db_path', help='Path to cisTEM database file')
    parser.add_argument('job_range', help='Job ID(s) to analyze')
    parser.add_argument('--sanity-check-only', action='store_true',
                        help='Only verify scaling formula, do not find peaks')
    parser.add_argument('--method', choices=['scaled', 'components', 'compare'],
                        default='compare',
                        help='Refinement method (default: compare)')
    parser.add_argument('--search-offset', type=float, default=0.8,
                        help='Stop searching when raw peak < offset * threshold (default: 0.8)')
    parser.add_argument('--mask-radius', type=int, default=10,
                        help='Mask radius in pixels (default: 10)')
    parser.add_argument('--fft-upsample', type=int, default=10,
                        help='FFT upsampling factor (default: 10)')
    parser.add_argument('--window-size', type=int, default=5,
                        help='Window size for FFT refinement (default: 5)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output CSV file for comparison results')
    parser.add_argument('--debug', type=int, nargs='?', const=3, default=0, metavar='N',
                        help='Debug mode: process N images with verbose output')

    args = parser.parse_args()

    # Parse job range
    try:
        job_ids = parse_job_range(args.job_range)
    except ValueError as e:
        print(f"Error parsing job range: {e}", file=sys.stderr)
        sys.exit(1)

    window_half = args.window_size // 2

    print("=" * 80)
    print("FFT Peak Recovery - Component Upsampling Analysis")
    print("=" * 80)
    print(f"Database: {args.db_path}")
    print(f"Jobs: {job_ids}")
    if args.sanity_check_only:
        print("Mode: Sanity check only")
    else:
        print(f"Mode: {args.method}")
        print(f"Search offset: {args.search_offset}")
        print(f"Mask radius: {args.mask_radius}")
        print(f"FFT upsample: {args.fft_upsample}x")
        print(f"Window size: {args.window_size}x{args.window_size}")

    try:
        # Initialize
        print("\nInitializing...")
        analyzer = tma.TemplateMatchAnalyzer(args.db_path)
        conn = analyzer.conn
        print("✓ Database connected")

        all_peaks = []

        for job_id in job_ids:
            threshold = get_job_threshold(conn, job_id)
            images = get_images_for_job(conn, job_id)

            print(f"\nJob {job_id}: threshold={threshold:.2f}, {len(images)} images")

            image_list = images[:args.debug] if args.debug > 0 else images
            debug = args.debug > 0

            for img_idx, image_id in enumerate(image_list):
                if debug:
                    print(f"  Image {img_idx + 1}/{len(image_list)} (ID: {image_id})...")
                else:
                    print(f"  Processing image {img_idx + 1}/{len(image_list)}...", end='\r')

                try:
                    if args.sanity_check_only:
                        analyze_image_sanity(analyzer, job_id, image_id, debug=debug)
                    else:
                        peaks = analyze_image_compare(
                            analyzer, job_id, image_id,
                            threshold, args.search_offset,
                            args.mask_radius, args.fft_upsample,
                            window_half, debug=debug
                        )
                        all_peaks.extend(peaks)

                        if debug:
                            print(f"    Found {len(peaks)} peaks")

                except Exception as e:
                    print(f"\n  ✗ Error on image {image_id}: {e}")
                    if debug:
                        import traceback
                        traceback.print_exc()
                    sys.exit(1)

            if not debug:
                print()  # Clear progress line

        # Print summary
        if args.sanity_check_only:
            print("\n✓ Sanity check passed for all images!")
        else:
            print_comparison_summary(all_peaks)

            # Export if requested
            if args.output and all_peaks:
                df = pd.DataFrame(all_peaks)
                df.to_csv(args.output, index=False)
                print(f"\n✓ Exported {len(all_peaks)} peaks to: {args.output}")

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
