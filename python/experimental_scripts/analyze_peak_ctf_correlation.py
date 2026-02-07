#!/usr/bin/env python3
"""
Analyze correlation between template matching peak scores and CTF parameters.

Examines how peak height (SNR) correlates with defocus and astigmatism
from CTF estimation. This helps identify whether CTF quality affects
template matching performance.

Usage:
    python analyze_peak_ctf_correlation.py <database_path> --tm-job-id <id> [--output plot.png]

Example:
    python analyze_peak_ctf_correlation.py /path/to/project.db --tm-job-id 1 --output ctf_correlation.png
"""

import sys
import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def load_peaks_with_ctf(db_path: str, tm_job_id: int,
                        limit_images: int = None) -> pd.DataFrame:
    """
    Load all peaks from a template matching job with CTF parameters.

    Joins peak data through TEMPLATE_MATCH_LIST -> IMAGE_ASSETS -> ESTIMATED_CTF_PARAMETERS
    to get the independently estimated CTF values (not the values used during TM).

    Args:
        db_path: Path to cisTEM database
        tm_job_id: Template matching job ID
        limit_images: If specified, only load peaks from this many images (for debugging)

    Returns:
        DataFrame with columns:
        - PEAK_NUMBER, X_POSITION, Y_POSITION, PEAK_HEIGHT
        - IMAGE_ASSET_ID, TEMPLATE_MATCH_ID
        - DEFOCUS1, DEFOCUS2, DEFOCUS_ANGLE (from CTF estimation)
        - MEAN_DEFOCUS, ASTIGMATISM (calculated)
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    try:
        cursor = conn.cursor()

        # First, get all TEMPLATE_MATCH_IDs for this job
        query = """
            SELECT TEMPLATE_MATCH_ID, IMAGE_ASSET_ID
            FROM TEMPLATE_MATCH_LIST
            WHERE TEMPLATE_MATCH_JOB_ID = ?
        """
        if limit_images:
            query += f" LIMIT {int(limit_images)}"
        cursor.execute(query, (tm_job_id,))
        match_info = cursor.fetchall()

        if not match_info:
            print(f"No results found for TM job {tm_job_id}")
            return pd.DataFrame()

        # Load peaks from each dynamic table and combine
        all_peaks = []
        n_images = len(match_info)
        for idx, (template_match_id, image_asset_id) in enumerate(match_info, 1):
            print(f"Processing image {idx}/{n_images} (IMAGE_ASSET_ID={image_asset_id}, TM_ID={template_match_id})")
            table_name = f"TEMPLATE_MATCH_PEAK_LIST_{template_match_id}"

            # Check if table exists (some may have zero peaks)
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name=?
            """, (table_name,))

            if cursor.fetchone() is None:
                continue

            # Load peaks from this table
            query = f"""
                SELECT
                    PEAK_NUMBER,
                    X_POSITION,
                    Y_POSITION,
                    PEAK_HEIGHT,
                    DEFOCUS as PEAK_DEFOCUS,
                    PSI, THETA, PHI
                FROM {table_name}
            """
            peaks_df = pd.read_sql_query(query, conn)

            if len(peaks_df) > 0:
                peaks_df['TEMPLATE_MATCH_ID'] = template_match_id
                peaks_df['IMAGE_ASSET_ID'] = image_asset_id
                all_peaks.append(peaks_df)

        if not all_peaks:
            print(f"No peaks found for TM job {tm_job_id}")
            return pd.DataFrame()

        # Combine all peaks
        peaks = pd.concat(all_peaks, ignore_index=True)

        # Now join with CTF parameters through IMAGE_ASSETS
        # Get unique image IDs
        image_ids = peaks['IMAGE_ASSET_ID'].unique().tolist()
        placeholders = ','.join('?' * len(image_ids))

        ctf_query = f"""
            SELECT
                img.IMAGE_ASSET_ID,
                ctf.DEFOCUS1,
                ctf.DEFOCUS2,
                ctf.DEFOCUS_ANGLE,
                ctf.SCORE as CTF_SCORE
            FROM IMAGE_ASSETS img
            JOIN ESTIMATED_CTF_PARAMETERS ctf
                ON img.CTF_ESTIMATION_ID = ctf.CTF_ESTIMATION_ID
            WHERE img.IMAGE_ASSET_ID IN ({placeholders})
        """
        ctf_df = pd.read_sql_query(ctf_query, conn, params=image_ids)

        # Merge CTF parameters with peaks
        peaks = peaks.merge(ctf_df, on='IMAGE_ASSET_ID', how='left')

        # Calculate derived values
        peaks['MEAN_DEFOCUS'] = (peaks['DEFOCUS1'] + peaks['DEFOCUS2']) / 2.0
        peaks['ASTIGMATISM'] = np.abs(peaks['DEFOCUS1'] - peaks['DEFOCUS2'])

        return peaks

    finally:
        conn.close()


def compute_correlation_stats(data: pd.DataFrame, x_col: str, y_col: str) -> dict:
    """
    Compute correlation statistics between two columns.

    Returns:
        Dictionary with pearson_r, pearson_p, spearman_r, spearman_p, n_points
    """
    mask = ~(data[x_col].isna() | data[y_col].isna())
    x = data.loc[mask, x_col].values
    y = data.loc[mask, y_col].values

    if len(x) < 3:
        return {'pearson_r': np.nan, 'pearson_p': np.nan,
                'spearman_r': np.nan, 'spearman_p': np.nan, 'n_points': len(x)}

    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)

    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'n_points': len(x)
    }


def plot_correlation_analysis(data: pd.DataFrame, tm_job_id: int,
                               output_path: str = None):
    """
    Create 1x2 correlation analysis plots.

    Panels:
    1. Peak height vs mean defocus (binned box plot)
    2. 2D heatmap: mean defocus vs astigmatism (colored by peak height)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Peak Score vs CTF Parameters (TM Job {tm_job_id})',
                 fontsize=14, fontweight='bold')

    # Panel 1: Peak height vs Mean Defocus (binned)
    ax1 = axes[0]

    # Get sorted unique bins for proper ordering
    sorted_bins = data.groupby('DEFOCUS_BIN_LABEL', observed=True)['MEAN_DEFOCUS'].mean().sort_values().index.tolist()

    # Create box plot data in sorted order
    box_data = [data.loc[data['DEFOCUS_BIN_LABEL'] == bin_label, 'PEAK_HEIGHT'].dropna().values
                for bin_label in sorted_bins]

    bp = ax1.boxplot(box_data, labels=sorted_bins, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.7)

    ax1.set_xlabel('Mean Defocus Bin (Å)', fontsize=11)
    ax1.set_ylabel('Peak Height (SNR)', fontsize=11)
    ax1.set_title('Peak Height vs Mean Defocus (Binned)')
    ax1.tick_params(axis='x', rotation=45)

    # Panel 2: 2D Heatmap - Mean defocus vs astigmatism
    ax2 = axes[1]

    hb = ax2.hexbin(data['MEAN_DEFOCUS'], data['ASTIGMATISM'],
                    C=data['PEAK_HEIGHT'], reduce_C_function=np.mean,
                    gridsize=30, cmap='RdYlBu_r', mincnt=1)
    cb = plt.colorbar(hb, ax=ax2)
    cb.set_label('Mean Peak Height (SNR)', fontsize=10)

    ax2.set_xlabel('Mean Defocus (Å)', fontsize=11)
    ax2.set_ylabel('Astigmatism |ΔDF| (Å)', fontsize=11)
    ax2.set_title('Mean Peak Height by Defocus & Astigmatism')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()


def print_statistics(data: pd.DataFrame, tm_job_id: int):
    """Print summary statistics for the correlation analysis."""

    print("\n" + "=" * 70)
    print(f"PEAK SCORE vs CTF CORRELATION ANALYSIS (TM Job {tm_job_id})")
    print("=" * 70)

    print(f"\nDataset Summary:")
    print(f"  Total peaks: {len(data):,}")
    print(f"  Unique images: {data['IMAGE_ASSET_ID'].nunique()}")

    print(f"\nPeak Height Statistics:")
    print(f"  Mean:   {data['PEAK_HEIGHT'].mean():.3f}")
    print(f"  Median: {data['PEAK_HEIGHT'].median():.3f}")
    print(f"  Std:    {data['PEAK_HEIGHT'].std():.3f}")
    print(f"  Range:  [{data['PEAK_HEIGHT'].min():.3f}, {data['PEAK_HEIGHT'].max():.3f}]")

    print(f"\nCTF Parameter Ranges:")
    print(f"  Mean Defocus: [{data['MEAN_DEFOCUS'].min():.0f}, {data['MEAN_DEFOCUS'].max():.0f}] Å")
    print(f"  Astigmatism:  [{data['ASTIGMATISM'].min():.0f}, {data['ASTIGMATISM'].max():.0f}] Å")

    # Correlation statistics
    print(f"\nCorrelation: Peak Height vs Mean Defocus")
    stats_defocus = compute_correlation_stats(data, 'MEAN_DEFOCUS', 'PEAK_HEIGHT')
    print(f"  Pearson r:  {stats_defocus['pearson_r']:+.4f} (p={stats_defocus['pearson_p']:.2e})")
    print(f"  Spearman ρ: {stats_defocus['spearman_r']:+.4f} (p={stats_defocus['spearman_p']:.2e})")

    print(f"\nCorrelation: Peak Height vs Astigmatism")
    stats_astig = compute_correlation_stats(data, 'ASTIGMATISM', 'PEAK_HEIGHT')
    print(f"  Pearson r:  {stats_astig['pearson_r']:+.4f} (p={stats_astig['pearson_p']:.2e})")
    print(f"  Spearman ρ: {stats_astig['spearman_r']:+.4f} (p={stats_astig['spearman_p']:.2e})")

    # Interpretation
    print(f"\nInterpretation:")
    if abs(stats_defocus['pearson_r']) > 0.3:
        direction = "higher" if stats_defocus['pearson_r'] > 0 else "lower"
        print(f"  • Moderate correlation with defocus: {direction} defocus → {direction} peak scores")
    else:
        print(f"  • Weak/no correlation between defocus and peak scores")

    if abs(stats_astig['pearson_r']) > 0.3:
        direction = "higher" if stats_astig['pearson_r'] > 0 else "lower"
        print(f"  • Moderate correlation with astigmatism: {direction} astigmatism → {direction} peak scores")
    else:
        print(f"  • Weak/no correlation between astigmatism and peak scores")

    print("=" * 70)


def add_defocus_bins(data: pd.DataFrame, bin_width: float) -> pd.DataFrame:
    """
    Add a DEFOCUS_BIN column that groups defocus values into bins.

    Args:
        data: DataFrame with MEAN_DEFOCUS column
        bin_width: Width of each bin in Angstroms

    Returns:
        DataFrame with added DEFOCUS_BIN and DEFOCUS_BIN_LABEL columns
    """
    # Calculate bin edges based on data range
    min_defocus = data['MEAN_DEFOCUS'].min()
    max_defocus = data['MEAN_DEFOCUS'].max()

    # Round min down and max up to bin boundaries
    bin_start = np.floor(min_defocus / bin_width) * bin_width
    bin_end = np.ceil(max_defocus / bin_width) * bin_width

    # Create bins
    bins = np.arange(bin_start, bin_end + bin_width, bin_width)

    # Assign each peak to a bin
    data['DEFOCUS_BIN'] = pd.cut(data['MEAN_DEFOCUS'], bins=bins, include_lowest=True)

    # Create readable labels like "5000-5100"
    data['DEFOCUS_BIN_LABEL'] = data['DEFOCUS_BIN'].apply(
        lambda x: f"{int(x.left)}-{int(x.right)}" if pd.notna(x) else "N/A"
    )

    return data


def print_binned_statistics(data: pd.DataFrame, bin_width: float):
    """Print per-bin statistics for defocus groups."""

    print(f"\n{'─' * 70}")
    print(f"PER-DEFOCUS-BIN STATISTICS (bin width: {bin_width:.0f} Å)")
    print(f"{'─' * 70}")

    # Group by bin and compute statistics
    bin_stats = data.groupby('DEFOCUS_BIN_LABEL', observed=True).agg({
        'PEAK_HEIGHT': ['count', 'mean', 'std', 'median'],
        'IMAGE_ASSET_ID': 'nunique',
        'MEAN_DEFOCUS': 'mean'
    }).reset_index()

    bin_stats.columns = ['Bin', 'N_Peaks', 'Mean_SNR', 'Std_SNR', 'Median_SNR',
                         'N_Images', 'Bin_Center']

    # Sort by bin center defocus
    bin_stats = bin_stats.sort_values('Bin_Center')

    # Print header
    print(f"\n{'Defocus Bin':<15} {'Images':>8} {'Peaks':>10} {'Mean SNR':>10} {'Std':>8} {'Median':>10}")
    print(f"{'-'*15} {'-'*8} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")

    # Print each bin
    for _, row in bin_stats.iterrows():
        print(f"{row['Bin']:<15} {row['N_Images']:>8} {row['N_Peaks']:>10,} "
              f"{row['Mean_SNR']:>10.3f} {row['Std_SNR']:>8.3f} {row['Median_SNR']:>10.3f}")

    print(f"{'─' * 70}")

    return bin_stats


def main():
    parser = argparse.ArgumentParser(
        description='Analyze correlation between TM peak scores and CTF parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script examines how template matching peak scores (SNR) correlate
with CTF estimation parameters (defocus, astigmatism).

The CTF parameters come from ESTIMATED_CTF_PARAMETERS table, representing
independently estimated values (not the defocus used during template matching).

Output plots:
  1. Peak height vs mean defocus (with linear regression)
  2. Peak height vs astigmatism (with linear regression)
  3. 2D heatmap of mean peak height binned by both parameters
  4. Per-image aggregated statistics

Example:
  python analyze_peak_ctf_correlation.py /path/to/project.db --tm-job-id 1
        """
    )
    parser.add_argument('db_path', help='Path to cisTEM database file')
    parser.add_argument('--tm-job-id', type=int, required=True,
                        help='Template matching job ID to analyze')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output plot file (shows interactively if not specified)')
    parser.add_argument('--limit', '-l', type=int, default=None,
                        help='Limit to N images for faster iteration (debug mode)')
    parser.add_argument('--defocus-bin-width', type=float, default=100.0,
                        help='Width of defocus bins in Angstroms (default: 100)')
    parser.add_argument('--max-astigmatism', type=float, default=1000.0,
                        help='Exclude images with astigmatism above this value in Angstroms (default: 1000)')

    args = parser.parse_args()

    # Validate database exists
    if not Path(args.db_path).exists():
        print(f"Error: Database file not found: {args.db_path}")
        sys.exit(1)

    # Load data
    print(f"Loading peaks from: {args.db_path}")
    print(f"TM Job ID: {args.tm_job_id}")
    if args.limit:
        print(f"Debug mode: limiting to {args.limit} images")

    data = load_peaks_with_ctf(args.db_path, args.tm_job_id, limit_images=args.limit)

    if len(data) == 0:
        print("Error: No data loaded. Check that the TM job ID exists.")
        sys.exit(1)

    print(f"Loaded {len(data):,} peaks from {data['IMAGE_ASSET_ID'].nunique()} images")

    # Filter by astigmatism cutoff
    n_before = len(data)
    images_before = data['IMAGE_ASSET_ID'].nunique()
    data = data[data['ASTIGMATISM'] <= args.max_astigmatism]
    n_after = len(data)
    images_after = data['IMAGE_ASSET_ID'].nunique()
    print(f"After astigmatism filter (≤{args.max_astigmatism:.0f} Å): {n_after:,} peaks from {images_after} images "
          f"(excluded {n_before - n_after:,} peaks from {images_before - images_after} images)")

    if len(data) == 0:
        print("Error: No data remaining after astigmatism filter.")
        sys.exit(1)

    # Add defocus bins
    data = add_defocus_bins(data, args.defocus_bin_width)

    # Print statistics
    print_statistics(data, args.tm_job_id)

    # Print per-bin statistics
    print_binned_statistics(data, args.defocus_bin_width)

    # Create plots
    plot_correlation_analysis(data, args.tm_job_id, args.output)


if __name__ == '__main__':
    main()
