#!/usr/bin/env python3
"""
Extract Particles with Sub-pixel Offset Testing

Extracts 2D particle images from micrographs based on template match peaks,
computes sub-pixel offsets via FFT upsampling, and creates cisTEM STAR files
with three offset variants for testing reconstruction quality.

Usage:
    python extract_particles_subpixel_test.py <db_path> <job_id> [options]

Examples:
    python extract_particles_subpixel_test.py cp.db 12 --max-images 5 --box-size 384
"""

import sys
import os
import argparse
import sqlite3
import numpy as np
import pandas as pd
import mrcfile
import starfile
from cistemx.db import database as tma


def apply_circular_mask(image: np.ndarray, cx: int, cy: int, radius: int):
    """Zero out a circular region around (cx, cy) with given radius."""
    y, x = np.ogrid[:image.shape[0], :image.shape[1]]
    mask = (x - cx)**2 + (y - cy)**2 <= radius**2
    image[mask] = image.min()


def find_peaks_with_recovery(mip: np.ndarray, threshold: float,
                              search_offset: float, pixel_size: float,
                              mask_radius: int, upsample_factor: int,
                              window_half: int, debug: bool = False) -> list[dict]:
    """
    Scan MIP for peaks, searching down to threshold * search_offset.
    Only keep peaks where FFT-refined height >= threshold.

    Returns list of peak dicts with position, angles placeholder, and heights.
    """
    working = mip.copy()
    original = mip
    peaks = []
    search_threshold = search_offset * threshold
    peak_num = 0

    if debug:
        print(f"    Peak recovery: threshold={threshold:.2f}, "
              f"search_offset={search_offset}, search_thr={search_threshold:.2f}")

    while True:
        max_idx = np.unravel_index(np.argmax(working), working.shape)
        y, x = max_idx
        raw_height = working[y, x]

        if raw_height < search_threshold:
            if debug:
                print(f"      Stop: raw={raw_height:.2f} < search_thr={search_threshold:.2f}")
            break

        peak_num += 1

        # FFT refine
        x_off, y_off, refined_height = compute_subpixel_offset(
            original, x, y, window_half, upsample_factor)

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

        apply_circular_mask(working, x, y, mask_radius)

    if debug:
        n_recovered = sum(1 for p in peaks if p['recovered'])
        print(f"      Total: {len(peaks)} peaks ({n_recovered} recovered) out of {peak_num} examined")

    return peaks


def get_job_threshold(conn: sqlite3.Connection, job_id: int) -> float:
    """Get the USED_THRESHOLD for a job."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT USED_THRESHOLD FROM TEMPLATE_MATCH_LIST
        WHERE TEMPLATE_MATCH_JOB_ID = ? LIMIT 1
    """, (job_id,))
    result = cursor.fetchone()
    if result is None:
        raise ValueError(f"No results found for job {job_id}")
    return float(result[0])


def get_job_ctf_params(conn: sqlite3.Connection, job_id: int) -> dict:
    """Get CTF parameters for a job (from first image, excluding defocus)."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT USED_PIXEL_SIZE, USED_VOLTAGE, USED_SPHERICAL_ABERRATION,
               USED_AMPLITUDE_CONTRAST
        FROM TEMPLATE_MATCH_LIST
        WHERE TEMPLATE_MATCH_JOB_ID = ? LIMIT 1
    """, (job_id,))
    result = cursor.fetchone()
    if result is None:
        raise ValueError(f"No results found for job {job_id}")
    return {
        'pixel_size': float(result[0]),
        'voltage': float(result[1]),
        'cs': float(result[2]),
        'amp_contrast': float(result[3])
    }


def get_image_ctf_params(conn: sqlite3.Connection, job_id: int, image_id: int) -> dict:
    """Get per-image CTF parameters (defocus values vary per image)."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT USED_DEFOCUS1, USED_DEFOCUS2, USED_DEFOCUS_ANGLE
        FROM TEMPLATE_MATCH_LIST
        WHERE TEMPLATE_MATCH_JOB_ID = ? AND IMAGE_ASSET_ID = ?
    """, (job_id, image_id))
    result = cursor.fetchone()
    if result is None:
        raise ValueError(f"No CTF params for job {job_id}, image {image_id}")
    return {
        'defocus1': float(result[0]),
        'defocus2': float(result[1]),
        'defocus_angle': float(result[2])
    }


def get_images_for_job(conn: sqlite3.Connection, job_id: int) -> list[int]:
    """Get list of IMAGE_ASSET_IDs analyzed in a job."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT IMAGE_ASSET_ID FROM TEMPLATE_MATCH_LIST
        WHERE TEMPLATE_MATCH_JOB_ID = ?
        ORDER BY IMAGE_ASSET_ID
    """, (job_id,))
    return [row[0] for row in cursor.fetchall()]


def get_image_filename(conn: sqlite3.Connection, image_id: int) -> str:
    """Get the filename of an image asset."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT FILENAME FROM IMAGE_ASSETS WHERE IMAGE_ASSET_ID = ?
    """, (image_id,))
    result = cursor.fetchone()
    if result is None:
        raise ValueError(f"No image found with ID {image_id}")
    return result[0]


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


def load_peaks_for_image(conn: sqlite3.Connection, job_id: int, image_id: int) -> pd.DataFrame:
    """Load peaks from database for one image."""
    tm_id = get_template_match_id(conn, job_id, image_id)
    table_name = f"TEMPLATE_MATCH_PEAK_LIST_{tm_id}"

    try:
        df = pd.read_sql_query(f"""
            SELECT PEAK_NUMBER, X_POSITION, Y_POSITION, PSI, THETA, PHI,
                   DEFOCUS, PIXEL_SIZE, PEAK_HEIGHT
            FROM {table_name}
        """, conn)
        return df
    except Exception:
        return pd.DataFrame()


def load_mip_images(analyzer, job_id: int, image_id: int) -> dict:
    """
    Load MIP result images (scaled_mip, psi, theta, phi, defocus).
    Returns dict with image arrays and pixel_size.
    """
    paths_df = analyzer.get_result_file_paths(job_id, {image_id})
    if len(paths_df) == 0:
        raise ValueError(f"No paths found for job {job_id}, image {image_id}")

    row = paths_df.iloc[0]

    def load_mrc(path: str) -> np.ndarray:
        with mrcfile.open(path, permissive=True) as mrc:
            return np.squeeze(mrc.data).astype(np.float32)

    # Get pixel size from scaled MIP
    with mrcfile.open(row['SCALED_MIP_OUTPUT_FILE'], permissive=True) as mrc:
        pixel_size = float(mrc.voxel_size.x)

    return {
        'scaled_mip': load_mrc(row['SCALED_MIP_OUTPUT_FILE']),
        'psi': load_mrc(row['PSI_OUTPUT_FILE']),
        'theta': load_mrc(row['THETA_OUTPUT_FILE']),
        'phi': load_mrc(row['PHI_OUTPUT_FILE']),
        'defocus': load_mrc(row['DEFOCUS_OUTPUT_FILE']),
        'pixel_size': pixel_size
    }


def load_micrograph(filename: str) -> tuple[np.ndarray, float]:
    """Load micrograph image and return (data, pixel_size)."""
    with mrcfile.open(filename, permissive=True) as mrc:
        data = np.squeeze(mrc.data).astype(np.float32)
        pixel_size = float(mrc.voxel_size.x)
    return data, pixel_size


def fft_upsample_window(image: np.ndarray, x_pixel: int, y_pixel: int,
                        window_half: int, upsample_factor: int) -> np.ndarray:
    """
    Extract a window around (x_pixel, y_pixel) and upsample via FFT zero-padding.
    """
    ny, nx = image.shape
    x0 = max(0, x_pixel - window_half)
    x1 = min(nx, x_pixel + window_half + 1)
    y0 = max(0, y_pixel - window_half)
    y1 = min(ny, y_pixel + window_half + 1)

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
    return upsampled


def compute_subpixel_offset(scaled_mip: np.ndarray, x_pixel: int, y_pixel: int,
                            window_half: int = 2, upsample_factor: int = 10
                            ) -> tuple[float, float, float]:
    """
    Compute sub-pixel offset using FFT upsampling.
    Returns: (x_offset, y_offset, refined_height) where offsets are in pixels.
    """
    upsampled = fft_upsample_window(scaled_mip, x_pixel, y_pixel, window_half, upsample_factor)

    max_idx = np.unravel_index(np.argmax(upsampled), upsampled.shape)
    max_y_up, max_x_up = max_idx
    refined_height = upsampled[max_y_up, max_x_up]

    # Center of upsampled window corresponds to (x_pixel, y_pixel)
    win_size = 2 * window_half + 1
    center_up_x = win_size * upsample_factor // 2
    center_up_y = win_size * upsample_factor // 2

    # Sub-pixel offset from integer position (signed)
    x_offset = (max_x_up - center_up_x) / upsample_factor
    y_offset = (max_y_up - center_up_y) / upsample_factor

    return x_offset, y_offset, refined_height


def extract_particle(micrograph: np.ndarray, x_center: int, y_center: int,
                     box_size: int) -> np.ndarray:
    """
    Extract a particle box from micrograph, centered at (x_center, y_center).
    Handles edge cases by padding with mean value.
    """
    ny, nx = micrograph.shape
    half_box = box_size // 2
    mean_val = np.mean(micrograph)

    # Create output array filled with mean
    particle = np.full((box_size, box_size), mean_val, dtype=np.float32)

    # Source coordinates in micrograph
    src_x0 = x_center - half_box
    src_y0 = y_center - half_box
    src_x1 = src_x0 + box_size
    src_y1 = src_y0 + box_size

    # Destination coordinates in particle
    dst_x0, dst_y0 = 0, 0
    dst_x1, dst_y1 = box_size, box_size

    # Clip to valid regions
    if src_x0 < 0:
        dst_x0 = -src_x0
        src_x0 = 0
    if src_y0 < 0:
        dst_y0 = -src_y0
        src_y0 = 0
    if src_x1 > nx:
        dst_x1 = box_size - (src_x1 - nx)
        src_x1 = nx
    if src_y1 > ny:
        dst_y1 = box_size - (src_y1 - ny)
        src_y1 = ny

    # Copy valid region
    if src_x1 > src_x0 and src_y1 > src_y0:
        particle[dst_y0:dst_y1, dst_x0:dst_x1] = micrograph[src_y0:src_y1, src_x0:src_x1]

    return particle


def normalize_particle(particle: np.ndarray) -> np.ndarray:
    """
    Normalize particle: subtract edge mean, divide by sqrt(variance).
    Matches cisTEM's normalization in prepare_stack_matchtemplate.
    """
    # Get edge mean (pixels on border)
    edge_pixels = np.concatenate([
        particle[0, :],          # top row
        particle[-1, :],         # bottom row
        particle[1:-1, 0],       # left column (excluding corners)
        particle[1:-1, -1]       # right column (excluding corners)
    ])
    edge_mean = np.mean(edge_pixels)

    # Variance of full image
    variance = np.var(particle)
    if variance == 0:
        variance = 1.0

    # Normalize
    normalized = (particle - edge_mean) / np.sqrt(variance)
    return normalized.astype(np.float32)


def write_cistem_star_file(particles: list[dict], output_path: str,
                           stack_filename: str, x_shift_mode: str = 'zero'):
    """
    Write cisTEM format STAR file.

    Args:
        particles: List of particle dicts with angles, defocus, offsets, etc.
        output_path: Output STAR file path
        stack_filename: Name of the particle stack file
        x_shift_mode: 'zero', 'positive', or 'negative' for offset handling
    """
    n = len(particles)

    # Build DataFrame with cisTEM columns
    # NOTE: starfile library adds '_' prefix automatically, so use names without leading underscore
    data = {
        'cisTEMPositionInStack': list(range(1, n + 1)),  # 1-indexed
        'cisTEMAnglePsi': [p['psi'] for p in particles],
        'cisTEMAngleTheta': [p['theta'] for p in particles],
        'cisTEMAnglePhi': [p['phi'] for p in particles],
        'cisTEMDefocus1': [p['defocus1'] for p in particles],
        'cisTEMDefocus2': [p['defocus2'] for p in particles],
        'cisTEMDefocusAngle': [p['defocus_angle'] for p in particles],
        'cisTEMPixelSize': [p['pixel_size'] for p in particles],
        'cisTEMMicroscopeVoltagekV': [p['voltage'] for p in particles],
        'cisTEMMicroscopeCsMM': [p['cs'] for p in particles],
        'cisTEMAmplitudeContrast': [p['amp_contrast'] for p in particles],
        'cisTEMOccupancy': [1.0] * n,
        'cisTEMSigma': [10.0] * n,
        'cisTEMLogP': [5000.0] * n,
        'cisTEMScore': [p['peak_height'] for p in particles],
        'cisTEMImageIsActive': [1] * n,
    }

    # Handle X/Y shifts based on mode (shifts are in Angstroms)
    # Testing half-pixel convention: positive = offset + 0.5px, negative = offset - 0.5px
    if x_shift_mode == 'zero':
        data['cisTEMXShift'] = [0.0] * n
        data['cisTEMYShift'] = [0.0] * n
    elif x_shift_mode == 'positive':
        # Offset + 0.5 pixels (testing half-pixel convention)
        data['cisTEMXShift'] = [(p['x_offset'] + 0.5) * p['pixel_size'] for p in particles]
        data['cisTEMYShift'] = [(p['y_offset'] + 0.5) * p['pixel_size'] for p in particles]
    elif x_shift_mode == 'negative':
        # Offset - 0.5 pixels (testing half-pixel convention)
        data['cisTEMXShift'] = [(p['x_offset'] - 0.5) * p['pixel_size'] for p in particles]
        data['cisTEMYShift'] = [(p['y_offset'] - 0.5) * p['pixel_size'] for p in particles]

    df = pd.DataFrame(data)
    starfile.write(df, output_path, float_format='%.6f')


def main():
    parser = argparse.ArgumentParser(
        description='Extract particles with sub-pixel offset testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Creates particle stack and three cisTEM STAR files with different offset conventions
for testing whether sub-pixel shifts improve reconstruction quality.

Examples:
  %(prog)s cp.db 12 --max-images 5 --box-size 384
  %(prog)s cp.db 12 --output-dir ./particles
        """
    )
    parser.add_argument('db_path', help='Path to cisTEM database file')
    parser.add_argument('job_id', type=int, help='Job ID (12 for unbinned)')
    parser.add_argument('--box-size', type=int, default=384,
                        help='Particle box size in pixels (default: 384)')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Process only first N images (default: all)')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory (default: current)')
    parser.add_argument('--fft-upsample', type=int, default=10,
                        help='FFT upsampling factor (default: 10)')
    parser.add_argument('--window-size', type=int, default=5,
                        help='Window size for FFT refinement (default: 5)')
    parser.add_argument('--threshold-offset', type=float, default=1.0,
                        help='Search multiplier: find peaks >= threshold*offset, '
                             'extract if refined >= threshold (default: 1.0, range: >0 to <=1)')
    parser.add_argument('--mask-radius', type=int, default=10,
                        help='Mask radius in pixels for peak finding (default: 10)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output showing peak recovery details')

    args = parser.parse_args()
    window_half = args.window_size // 2

    # Validate threshold-offset
    if args.threshold_offset <= 0 or args.threshold_offset > 1:
        parser.error(f"--threshold-offset must be > 0 and <= 1, got {args.threshold_offset}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("Particle Extraction with Sub-pixel Offset Testing")
    print("=" * 70)
    print(f"Database: {args.db_path}")
    print(f"Job ID: {args.job_id}")
    print(f"Box size: {args.box_size}")
    print(f"FFT upsample: {args.fft_upsample}x")
    print(f"Threshold offset: {args.threshold_offset}")
    print(f"Output dir: {args.output_dir}")
    if args.debug:
        print(f"Debug mode: ON")

    try:
        # Initialize
        analyzer = tma.TemplateMatchAnalyzer(args.db_path)
        conn = analyzer.conn

        threshold = get_job_threshold(conn, args.job_id)
        ctf_params = get_job_ctf_params(conn, args.job_id)
        images = get_images_for_job(conn, args.job_id)

        print(f"\nJob threshold: {threshold:.2f}")
        print(f"Pixel size: {ctf_params['pixel_size']:.3f} Å")
        print(f"Images in job: {len(images)}")

        if args.max_images:
            images = images[:args.max_images]
            print(f"Processing first {len(images)} images")

        # Collect all particles
        all_particles = []
        stack_data = []

        for img_idx, image_id in enumerate(images):
            print(f"\nProcessing image {img_idx + 1}/{len(images)} (ID: {image_id})...")

            # Load micrograph
            micrograph_path = get_image_filename(conn, image_id)
            micrograph, _ = load_micrograph(micrograph_path)
            micrograph_name = os.path.basename(micrograph_path)

            # Load MIP results
            mip_images = load_mip_images(analyzer, args.job_id, image_id)
            pixel_size = mip_images['pixel_size']

            # Get per-image CTF parameters (defocus varies per image)
            image_ctf = get_image_ctf_params(conn, args.job_id, image_id)

            # Get peaks: either from database (offset=1.0) or via MIP search (offset<1.0)
            if args.threshold_offset == 1.0:
                # Use database peaks (already passed threshold)
                peaks_df = load_peaks_for_image(conn, args.job_id, image_id)
                if len(peaks_df) == 0:
                    print(f"  No peaks found in database, skipping")
                    continue
                print(f"  Found {len(peaks_df)} peaks in database (threshold_offset=1.0, no recovery)")
                if args.debug:
                    print(f"    Using DB peaks directly, no sub-threshold search")

                # Convert to list of peak dicts
                peak_list = []
                for _, row in peaks_df.iterrows():
                    x_pix = row['X_POSITION'] / pixel_size
                    y_pix = row['Y_POSITION'] / pixel_size
                    x_int = int(round(x_pix))
                    y_int = int(round(y_pix))

                    x_offset, y_offset, refined_height = compute_subpixel_offset(
                        mip_images['scaled_mip'], x_int, y_int,
                        window_half, args.fft_upsample)

                    peak_list.append({
                        'x_pixel': x_int,
                        'y_pixel': y_int,
                        'raw_height': row['PEAK_HEIGHT'],
                        'refined_height': refined_height,
                        'x_offset': x_offset,
                        'y_offset': y_offset,
                        'recovered': False,
                        # DB values for fallback
                        'db_psi': row['PSI'],
                        'db_theta': row['THETA'],
                        'db_phi': row['PHI'],
                        'db_defocus': row['DEFOCUS']
                    })
            else:
                # Search MIP for peaks with FFT recovery
                print(f"  Searching MIP with threshold_offset={args.threshold_offset}")
                peak_list = find_peaks_with_recovery(
                    mip_images['scaled_mip'], threshold, args.threshold_offset,
                    pixel_size, args.mask_radius, args.fft_upsample,
                    window_half, debug=args.debug)

                if len(peak_list) == 0:
                    print(f"  No peaks found, skipping")
                    continue

                n_recovered = sum(1 for p in peak_list if p['recovered'])
                print(f"  Found {len(peak_list)} peaks ({n_recovered} recovered via FFT)")

            # Process each peak
            for peak in peak_list:
                x_int = peak['x_pixel']
                y_int = peak['y_pixel']
                x_offset = peak['x_offset']
                y_offset = peak['y_offset']
                refined_height = peak['refined_height']

                # Get angles from MIP images at this position
                if (0 <= y_int < mip_images['psi'].shape[0] and
                    0 <= x_int < mip_images['psi'].shape[1]):
                    psi = float(mip_images['psi'][y_int, x_int])
                    theta = float(mip_images['theta'][y_int, x_int])
                    phi = float(mip_images['phi'][y_int, x_int])
                    defocus_delta = float(mip_images['defocus'][y_int, x_int])
                else:
                    # Use DB values if available, otherwise defaults
                    psi = peak.get('db_psi', 0.0)
                    theta = peak.get('db_theta', 0.0)
                    phi = peak.get('db_phi', 0.0)
                    defocus_delta = peak.get('db_defocus', 0.0)

                # Extract and normalize particle
                particle = extract_particle(micrograph, x_int, y_int, args.box_size)
                particle = normalize_particle(particle)
                stack_data.append(particle)

                # Store particle info
                # Defocus = per-image base defocus + per-pixel defocus delta from MIP
                all_particles.append({
                    'psi': psi,
                    'theta': theta,
                    'phi': phi,
                    'x_offset': x_offset,
                    'y_offset': y_offset,
                    'defocus1': image_ctf['defocus1'] + defocus_delta,
                    'defocus2': image_ctf['defocus2'] + defocus_delta,
                    'defocus_angle': image_ctf['defocus_angle'],
                    'pixel_size': pixel_size,
                    'voltage': ctf_params['voltage'],
                    'cs': ctf_params['cs'],
                    'amp_contrast': ctf_params['amp_contrast'],
                    'peak_height': peak['raw_height'],
                    'refined_height': refined_height
                })

            print(f"  Extracted {len(peak_list)} particles")

        print(f"\nTotal particles: {len(all_particles)}")

        if len(all_particles) == 0:
            print("No particles to export!")
            return

        # Write particle stack
        stack_filename = 'particles.mrc'
        stack_path = os.path.join(args.output_dir, stack_filename)
        print(f"\nWriting particle stack: {stack_path}")

        stack_array = np.array(stack_data, dtype=np.float32)
        with mrcfile.new(stack_path, overwrite=True) as mrc:
            mrc.set_data(stack_array)
            mrc.voxel_size = ctf_params['pixel_size']

        # Write three STAR file variants
        for mode in ['zero', 'positive', 'negative']:
            star_filename = f'particles_offset_{mode}.star'
            star_path = os.path.join(args.output_dir, star_filename)
            print(f"Writing STAR file: {star_path}")
            write_cistem_star_file(all_particles, star_path, stack_filename, x_shift_mode=mode)

        # Print offset statistics
        x_offsets_pix = [p['x_offset'] for p in all_particles]
        y_offsets_pix = [p['y_offset'] for p in all_particles]
        radial_offsets_pix = [np.sqrt(x**2 + y**2) for x, y in zip(x_offsets_pix, y_offsets_pix)]

        # Convert to Angstroms for display
        pix_size = ctf_params['pixel_size']
        x_offsets_ang = [x * pix_size for x in x_offsets_pix]
        y_offsets_ang = [y * pix_size for y in y_offsets_pix]
        radial_offsets_ang = [r * pix_size for r in radial_offsets_pix]

        print(f"\nSub-pixel offset statistics:")
        print(f"  X offset: mean={np.mean(x_offsets_pix):.4f} px ({np.mean(x_offsets_ang):.3f} Å)")
        print(f"  Y offset: mean={np.mean(y_offsets_pix):.4f} px ({np.mean(y_offsets_ang):.3f} Å)")
        print(f"  Radial:   mean={np.mean(radial_offsets_pix):.4f} px ({np.mean(radial_offsets_ang):.3f} Å), "
              f"max={np.max(radial_offsets_pix):.4f} px ({np.max(radial_offsets_ang):.3f} Å)")

        print("\n✓ Done!")
        print(f"  Stack: {stack_path} ({len(all_particles)} particles)")
        print(f"  STAR files: particles_offset_{{zero,positive,negative}}.star")

    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
