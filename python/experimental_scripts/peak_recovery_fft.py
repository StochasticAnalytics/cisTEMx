#!/usr/bin/env python3
"""
FFT Peak Recovery Analysis

Uses FFT upsampling to recover peaks that fell below threshold due to sub-pixel
positioning artifacts. Instead of relying on pre-detected peaks, this script
scans the MIP directly and uses FFT-refined heights to determine which peaks
pass threshold.

The hypothesis is that peaks landing between pixels suffer from sampling artifacts
that reduce their apparent height. By using FFT upsampling (sinc interpolation),
we can recover the "true" peak height and potentially detect particles that were
missed by the original threshold.

Usage:
    python peak_recovery_fft.py <db_path> <job_id> [options]

Examples:
    python peak_recovery_fft.py cp.db 12 --search-offset 0.8
    python peak_recovery_fft.py cp.db 12:18 --debug 3 --output recovered.csv
"""

import sys
import argparse
import sqlite3
import numpy as np
import pandas as pd
import mrcfile
from cistemx.db import database as tma


def parse_job_range(range_str: str) -> list[int]:
    """
    Parse job range specification into list of job IDs.
    Supports: "12:18" (range), "12,14,16" (list), or "12" (single).
    """
    range_str = range_str.strip()

    if ':' in range_str:
        parts = range_str.split(':')
        if len(parts) != 2:
            raise ValueError(f"Invalid range format '{range_str}'")
        start, end = int(parts[0]), int(parts[1])
        if start > end:
            raise ValueError(f"Start ({start}) must be <= end ({end})")
        return list(range(start, end + 1))

    elif ',' in range_str:
        return sorted([int(x.strip()) for x in range_str.split(',')])

    else:
        return [int(range_str)]


def get_job_threshold(conn: sqlite3.Connection, job_id: int) -> float:
    """
    Get the USED_THRESHOLD for a job from TEMPLATE_MATCH_LIST.
    Assumes threshold is consistent across all images in the job.
    """
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


def get_template_match_id(conn: sqlite3.Connection, job_id: int, image_id: int) -> int:
    """Get TEMPLATE_MATCH_ID for a specific job/image combination."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT TEMPLATE_MATCH_ID FROM TEMPLATE_MATCH_LIST
        WHERE TEMPLATE_MATCH_JOB_ID = ? AND IMAGE_ASSET_ID = ?
    """, (job_id, image_id))
    result = cursor.fetchone()
    if result is None:
        raise ValueError(f"No template match found for job {job_id}, image {image_id}")
    return result[0]


def load_mip_for_job_image(analyzer, job_id: int, image_id: int) -> tuple[np.ndarray, float]:
    """
    Load SCALED_MIP image for a specific job/image combination.

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


def compute_subpixel_offset_fft(image: np.ndarray, x_pixel: float, y_pixel: float,
                                window_half: int = 2, upsample_factor: int = 10
                                ) -> tuple[float, float, float]:
    """
    Compute sub-pixel offset using FFT upsampling (sinc interpolation).

    Returns:
        (x_offset, y_offset, refined_peak_height)
    """
    xi, yi = int(round(x_pixel)), int(round(y_pixel))

    ny, nx = image.shape
    x0 = max(0, xi - window_half)
    x1 = min(nx, xi + window_half + 1)
    y0 = max(0, yi - window_half)
    y1 = min(ny, yi + window_half + 1)

    window = image[y0:y1, x0:x1]
    win_ny, win_nx = window.shape

    fft_window = np.fft.fft2(window)

    pad_ny = win_ny * upsample_factor
    pad_nx = win_nx * upsample_factor
    padded = np.zeros((pad_ny, pad_nx), dtype=complex)

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

    upsampled = np.fft.ifft2(padded).real * (upsample_factor ** 2)

    max_idx = np.unravel_index(np.argmax(upsampled), upsampled.shape)
    max_y_up, max_x_up = max_idx

    refined_peak_height = upsampled[max_y_up, max_x_up]

    max_x_win = max_x_up / upsample_factor
    max_y_win = max_y_up / upsample_factor

    max_x_img = x0 + max_x_win
    max_y_img = y0 + max_y_win

    x_offset = abs(max_x_img - xi)
    y_offset = abs(max_y_img - yi)

    x_offset = min(0.5, x_offset)
    y_offset = min(0.5, y_offset)

    return x_offset, y_offset, refined_peak_height


def apply_circular_mask(image: np.ndarray, cx: int, cy: int, radius: int):
    """Zero out circular region around (cx, cy). Modifies in-place."""
    ny, nx = image.shape
    y, x = np.ogrid[:ny, :nx]
    mask = (x - cx)**2 + (y - cy)**2 <= radius**2
    image[mask] = image.min()  # Set to min instead of 0 for MIPs that may have negative values


def find_peaks_with_fft_recovery(mip: np.ndarray, threshold: float,
                                 pixel_size: float,
                                 search_offset: float = 0.8,
                                 mask_radius: int = 5,
                                 upsample_factor: int = 10,
                                 window_half: int = 2,
                                 debug: bool = False) -> list[dict]:
    """
    Scan MIP for peaks using FFT-refined heights.

    Args:
        mip: 2D MIP image array
        threshold: Detection threshold (peaks with refined height >= this are kept)
        pixel_size: Pixel size in Angstroms
        search_offset: Stop searching when raw peak < offset * threshold (default: 0.8)
        mask_radius: Radius in pixels for masking detected peaks
        upsample_factor: FFT upsampling factor
        window_half: Half-size of FFT window
        debug: Print per-peak progress

    Returns:
        List of peak dicts with x_pixel, y_pixel, x_ang, y_ang, raw_height,
        refined_height, recovered (True if raw < threshold but refined >= threshold)
    """
    working = mip.copy()
    original = mip
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

        # FFT refine using original (unmasked) MIP
        x_off, y_off, refined_height = compute_subpixel_offset_fft(
            original, x, y, window_half, upsample_factor)

        # Keep peak if refined height passes threshold
        kept = refined_height >= threshold
        recovered = raw_height < threshold

        if debug:
            status = "KEPT" if kept else "skip"
            rec_flag = " [RECOVERED]" if (kept and recovered) else ""
            print(f"      Peak {peak_num:3d}: ({x:4d},{y:4d}) raw={raw_height:.2f} "
                  f"refined={refined_height:.2f} {status}{rec_flag}")

        if kept:
            peaks.append({
                'x_pixel': x,
                'y_pixel': y,
                'x_ang': x * pixel_size,
                'y_ang': y * pixel_size,
                'raw_height': raw_height,
                'refined_height': refined_height,
                'x_offset': x_off,
                'y_offset': y_off,
                'recovered': recovered
            })

        # Mask out this peak in working copy
        apply_circular_mask(working, x, y, mask_radius)

    if debug:
        print(f"      Total: {len(peaks)} peaks kept out of {peak_num} examined")

    return peaks


def load_original_peaks(conn: sqlite3.Connection, job_id: int, image_id: int) -> pd.DataFrame:
    """Load original peaks from database for comparison."""
    tm_id = get_template_match_id(conn, job_id, image_id)
    table_name = f"TEMPLATE_MATCH_PEAK_LIST_{tm_id}"

    try:
        df = pd.read_sql_query(f"""
            SELECT PEAK_NUMBER, X_POSITION, Y_POSITION, PEAK_HEIGHT
            FROM {table_name}
        """, conn)
        return df
    except Exception:
        return pd.DataFrame()


def match_peaks(recovered_peaks: list[dict], original_df: pd.DataFrame,
                match_radius: float = 10.0, image_size_ang: tuple[float, float] = None,
                edge_exclusion_ang: float = 0.0) -> dict:
    """
    Match recovered peaks with original database peaks.

    Args:
        recovered_peaks: Peaks found by FFT recovery
        original_df: Original peaks from database
        match_radius: Maximum distance (Å) to consider a match
        image_size_ang: (width, height) in Angstroms for edge detection
        edge_exclusion_ang: Edge margin in Angstroms (peaks within this are "edge excluded")

    Returns:
        Dict with categorized peaks:
        - matched: Recovered peaks that match originals
        - truly_recovered: New peaks where raw < threshold (FFT rescued)
        - edge_excluded: New peaks where raw >= threshold but in edge region
        - missed_original: Original peaks not found by recovery
    """
    matched = []
    truly_recovered = []
    edge_excluded = []
    matched_original_idx = set()

    for peak in recovered_peaks:
        x, y = peak['x_ang'], peak['y_ang']
        found_match = False

        for idx, orig in original_df.iterrows():
            if idx in matched_original_idx:
                continue
            dist = np.sqrt((x - orig['X_POSITION'])**2 + (y - orig['Y_POSITION'])**2)
            if dist <= match_radius:
                peak['original_height'] = orig['PEAK_HEIGHT']
                peak['match_distance'] = dist
                matched.append(peak)
                matched_original_idx.add(idx)
                found_match = True
                break

        if not found_match:
            # Categorize new peaks
            if peak.get('recovered', False):
                # raw < threshold, refined >= threshold: truly rescued by FFT
                truly_recovered.append(peak)
            elif image_size_ang and edge_exclusion_ang > 0:
                # Check if in edge region
                w, h = image_size_ang
                in_edge = (x < edge_exclusion_ang or x > w - edge_exclusion_ang or
                           y < edge_exclusion_ang or y > h - edge_exclusion_ang)
                if in_edge:
                    edge_excluded.append(peak)
                else:
                    # raw >= threshold, not in edge - unexpected, treat as truly_recovered
                    truly_recovered.append(peak)
            else:
                # No edge info, assume truly recovered
                truly_recovered.append(peak)

    # Find missed originals
    missed_original = []
    for idx, orig in original_df.iterrows():
        if idx not in matched_original_idx:
            missed_original.append({
                'x_ang': orig['X_POSITION'],
                'y_ang': orig['Y_POSITION'],
                'original_height': orig['PEAK_HEIGHT']
            })

    return {
        'matched': matched,
        'truly_recovered': truly_recovered,
        'edge_excluded': edge_excluded,
        'missed_original': missed_original
    }


def print_image_summary(image_id: int, results: dict, debug: bool = False):
    """Print summary for one image."""
    matched = results['matched']
    truly_recovered = results['truly_recovered']
    edge_excluded = results['edge_excluded']
    missed = results['missed_original']

    n_matched = len(matched)
    n_truly_recovered = len(truly_recovered)
    n_edge = len(edge_excluded)
    n_missed = len(missed)
    n_original = n_matched + n_missed

    print(f"  Image {image_id}: Original={n_original}, Matched={n_matched}, "
          f"TrulyRecovered={n_truly_recovered}, EdgeExcluded={n_edge}, Missed={n_missed}")

    if debug:
        if truly_recovered:
            print(f"    Truly recovered (raw<thr, refined>=thr):")
            for p in truly_recovered[:5]:
                print(f"      ({p['x_ang']:.1f}, {p['y_ang']:.1f}) raw={p['raw_height']:.2f} "
                      f"refined={p['refined_height']:.2f}")
            if len(truly_recovered) > 5:
                print(f"      ... and {len(truly_recovered) - 5} more")

        if edge_excluded:
            print(f"    Edge excluded (raw>=thr but in edge margin):")
            for p in edge_excluded[:5]:
                print(f"      ({p['x_ang']:.1f}, {p['y_ang']:.1f}) raw={p['raw_height']:.2f}")
            if len(edge_excluded) > 5:
                print(f"      ... and {len(edge_excluded) - 5} more")

        if missed:
            print(f"    Missed (in DB but not found):")
            for p in missed[:5]:
                print(f"      ({p['x_ang']:.1f}, {p['y_ang']:.1f}) height={p['original_height']:.2f}")
            if len(missed) > 5:
                print(f"      ... and {len(missed) - 5} more")


def analyze_job(analyzer, conn: sqlite3.Connection, job_id: int,
                search_offset: float, mask_radius: float, upsample_factor: int,
                window_half: int, match_radius: float,
                edge_exclusion_pix: int = 0,
                base_resolution: float = 3.0, 
                base_mask_radius: float = 10.0,
                base_job_id: int = 12,
                debug_images: int = 0) -> dict:
    """
    Analyze all images in a job for peak recovery.

    Args:
        mask_radius: If > 0, use this fixed radius. If 0, compute from resolution scaling.
        edge_exclusion_pix: Pixels from edge to consider as "edge excluded" region
        base_resolution: Resolution (Å) for mask radius scaling (job 12 default)
        base_mask_radius: Mask radius at base_resolution

    Returns summary statistics dict.
    """
    threshold = get_job_threshold(conn, job_id)
    images = get_images_for_job(conn, job_id)

    # Get job resolution for mask radius scaling (from job parameters if available)
    # For now, infer from job_id assuming job 12 = 3.0Å, incrementing by 0.5
    job_resolution = base_resolution + (job_id - base_job_id) * 0.5

    # Compute mask radius: scales inversely with resolution (higher res = smaller binning = larger mask)
    if mask_radius > 0:
        effective_mask_radius = int(round(mask_radius))
        mask_mode = "fixed"
    else:
        # Scale: mask_radius = base_mask_radius * base_resolution / job_resolution
        effective_mask_radius = int(round(base_mask_radius * base_resolution / job_resolution))
        mask_mode = f"auto ({base_mask_radius}*{base_resolution}/{job_resolution:.1f})"

    # Edge exclusion only applies to unbinned search (job 12)
    # Binned searches don't have the edge exclusion artifact
    # FIXME:
    effective_edge_exclusion = edge_exclusion_pix if job_id == 12 else 0

    print(f"\nJob {job_id}: threshold={threshold:.2f}, resolution~{job_resolution:.1f}Å, {len(images)} images")
    print(f"  Search offset={search_offset} (searching down to {search_offset * threshold:.2f})")
    print(f"  Mask radius={effective_mask_radius} pixels [{mask_mode}], Edge exclusion={effective_edge_exclusion} pixels")

    total_original = 0
    total_matched = 0
    total_truly_recovered = 0
    total_edge_excluded = 0
    total_missed = 0

    all_peaks = []

    image_list = images[:debug_images] if debug_images > 0 else images
    debug = debug_images > 0

    for img_idx, image_id in enumerate(image_list):
        if debug:
            print(f"  Processing image {img_idx + 1}/{len(image_list)} (ID: {image_id})...")
        else:
            print(f"  Processing image {img_idx + 1}/{len(image_list)}...", end='\r')

        try:
            # Load MIP
            mip, pixel_size = load_mip_for_job_image(analyzer, job_id, image_id)
            ny, nx = mip.shape
            image_size_ang = (nx * pixel_size, ny * pixel_size)
            edge_exclusion_ang = effective_edge_exclusion * pixel_size

            # Find peaks with FFT recovery
            recovered_peaks = find_peaks_with_fft_recovery(
                mip, threshold, pixel_size,
                search_offset=search_offset,
                mask_radius=effective_mask_radius,
                upsample_factor=upsample_factor,
                window_half=window_half,
                debug=debug
            )

            # Load original peaks
            original_df = load_original_peaks(conn, job_id, image_id)

            # Match and categorize peaks
            results = match_peaks(
                recovered_peaks, original_df, match_radius,
                image_size_ang=image_size_ang,
                edge_exclusion_ang=edge_exclusion_ang)

            # Update totals
            total_original += len(original_df)
            total_matched += len(results['matched'])
            total_truly_recovered += len(results['truly_recovered'])
            total_edge_excluded += len(results['edge_excluded'])
            total_missed += len(results['missed_original'])

            # Store peaks for export (matched + truly_recovered + edge_excluded)
            for category in ['matched', 'truly_recovered', 'edge_excluded']:
                for p in results[category]:
                    p['job_id'] = job_id
                    p['image_id'] = image_id
                    p['category'] = category
                    all_peaks.append(p)

            if debug:
                print_image_summary(image_id, results, debug=True)

        except Exception as e:
            print(f"  Warning: Error processing image {image_id}: {e}")
            import traceback
            if debug:
                traceback.print_exc()

    if not debug:
        print()  # Clear progress line

    return {
        'job_id': job_id,
        'threshold': threshold,
        'resolution': job_resolution,
        'mask_radius': effective_mask_radius,
        'n_images': len(image_list),
        'original_peaks': total_original,
        'matched_peaks': total_matched,
        'truly_recovered': total_truly_recovered,
        'edge_excluded': total_edge_excluded,
        'missed_peaks': total_missed,
        'all_peaks': all_peaks
    }


def print_summary(results: list[dict]):
    """Print overall summary across all jobs."""
    print("\n" + "=" * 80)
    print("FFT PEAK RECOVERY SUMMARY")
    print("=" * 80)

    total_original = sum(r['original_peaks'] for r in results)
    total_matched = sum(r['matched_peaks'] for r in results)
    total_truly_recovered = sum(r['truly_recovered'] for r in results)
    total_edge_excluded = sum(r['edge_excluded'] for r in results)
    total_missed = sum(r['missed_peaks'] for r in results)

    print(f"\nAcross {len(results)} job(s):")
    print(f"  Original peaks in database:    {total_original}")
    print(f"  Matched (found & in DB):       {total_matched}")
    print(f"  Truly recovered (raw<thr):     {total_truly_recovered} ({100*total_truly_recovered/max(1,total_original):.1f}% increase)")
    print(f"  Edge excluded (raw>=thr, edge):{total_edge_excluded}")
    print(f"  Missed (in DB, not found):     {total_missed}")

    print(f"\nPer-job breakdown:")
    print(f"{'Job':>5} {'Res':>5} {'Thr':>6} {'Mask':>5} {'Orig':>7} {'Match':>7} {'TrueRec':>8} {'Edge':>6} {'Miss':>6}")
    print("-" * 75)
    for r in results:
        print(f"{r['job_id']:>5} {r['resolution']:>5.1f} {r['threshold']:>6.2f} {r['mask_radius']:>5} "
              f"{r['original_peaks']:>7} {r['matched_peaks']:>7} {r['truly_recovered']:>8} "
              f"{r['edge_excluded']:>6} {r['missed_peaks']:>6}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='FFT Peak Recovery - find peaks missed due to sub-pixel positioning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script scans MIP images directly instead of relying on pre-detected peaks.
It uses FFT upsampling to compute refined peak heights and can recover peaks
that fell just below threshold due to sub-pixel positioning artifacts.

Examples:
  %(prog)s cp.db 12 --search-offset 0.8
  %(prog)s cp.db 12:18 --debug 3 --output recovered.csv
        """
    )
    parser.add_argument('db_path', help='Path to cisTEM database file')
    parser.add_argument('job_range', help='Job ID(s) to analyze (e.g., "12", "12:18", "12,14,16")')
    parser.add_argument('--search-offset', type=float, default=0.8,
                        help='Stop searching when raw peak < offset * threshold (default: 0.8)')
    parser.add_argument('--base-resolution', type=float, default=3.0,
                        help='Base resolution (Å) for mask radius scaling (default: 3.0Å for job 12)')
    parser.add_argument('--mask-radius', type=float, default=0,
                        help='Fixed mask radius in pixels (default: 0 = auto-scale by resolution)')
    parser.add_argument('--base-mask-radius', type=float, default=10.0,
                        help='Base mask radius at 3.0Å resolution for auto-scaling (default: 10)')
    parser.add_argument('--edge-exclusion', type=int, default=97,
                        help='Edge exclusion in pixels, matches cisTEM template_size/4+1 (default: 97)')
    parser.add_argument('--fft-upsample', type=int, default=10,
                        help='FFT upsampling factor (default: 10)')
    parser.add_argument('--window-size', type=int, default=5,
                        help='Window size for FFT refinement (default: 5)')
    parser.add_argument('--match-radius', type=float, default=10.0,
                        help='Max distance (Å) to match recovered peaks with originals (default: 10.0)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output CSV file for all recovered peaks')
    parser.add_argument('--debug', type=int, nargs='?', const=3, default=0, metavar='N',
                        help='Debug mode: process N images with verbose output (default: 3 if flag given)')

    args = parser.parse_args()

    # Parse job range
    try:
        job_ids = parse_job_range(args.job_range)
    except ValueError as e:
        print(f"Error parsing job range: {e}", file=sys.stderr)
        sys.exit(1)

    window_half = args.window_size // 2

    print("=" * 80)
    print("FFT Peak Recovery Analysis")
    print("=" * 80)
    print(f"Database: {args.db_path}")
    print(f"Jobs: {job_ids}")
    print(f"Search offset: {args.search_offset}")
    if args.mask_radius > 0:
        print(f"Mask radius: {args.mask_radius} pixels (fixed)")
    else:
        print(f"Mask radius: auto-scaled (base={args.base_mask_radius} at 3.0Å)")
    print(f"Edge exclusion: {args.edge_exclusion} pixels")
    print(f"FFT upsample: {args.fft_upsample}x")
    print(f"Window size: {args.window_size}x{args.window_size}")
    print(f"Base resolution: {args.base_resolution} Å")

    try:
        # Initialize
        print("\nInitializing...")
        analyzer = tma.TemplateMatchAnalyzer(args.db_path)
        conn = analyzer.conn
        print("✓ Database connected")

        # Analyze each job
        # For now, we assume if there is more than one job, the resolution is decreased by 0.5 Å per job increment
        results = []
        base_job_id = job_ids[0]
        for job_id in job_ids:
            result = analyze_job(
                analyzer, conn, job_id,
                search_offset=args.search_offset,
                mask_radius=args.mask_radius,
                upsample_factor=args.fft_upsample,
                window_half=window_half,
                match_radius=args.match_radius,
                edge_exclusion_pix=args.edge_exclusion,
                base_mask_radius=args.base_mask_radius,
                base_resolution=args.base_resolution,
                base_job_id=base_job_id,
                debug_images=args.debug
            )
            results.append(result)

        # Print summary
        print_summary(results)

        # Export if requested
        if args.output:
            all_peaks = []
            for r in results:
                all_peaks.extend(r['all_peaks'])

            if all_peaks:
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
