#!/usr/bin/env python3
"""
Analyze correlation between sub-pixel offset and refined peak height increase.

This script examines whether peaks that are further from pixel centers
show larger increases in height after FFT upsampling refinement.

Hypothesis: Peaks landing between pixels suffer from interpolation artifacts
that reduce their apparent height. FFT upsampling (sinc interpolation)
recovers the "true" peak height, and this recovery should be larger for
peaks with greater sub-pixel offset.

Usage:
    python analyze_subpixel_height_correlation.py <consensus_csv> [--output plot.png]
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def extract_job_ids(df: pd.DataFrame) -> list[int]:
    """Extract job IDs from column names like SCORE_JOB12, SCORE_JOB13, etc."""
    job_ids = []
    for col in df.columns:
        if col.startswith('SCORE_JOB'):
            try:
                job_id = int(col.replace('SCORE_JOB', ''))
                job_ids.append(job_id)
            except ValueError:
                continue
    return sorted(job_ids)


def compute_correlations(df: pd.DataFrame, job_ids: list[int]) -> pd.DataFrame:
    """
    Extract per-peak, per-job data for correlation analysis.

    Returns DataFrame with columns:
    - job_id: Which search
    - original_height: SCORE_JOB{id}
    - refined_height: REFINED_HEIGHT_JOB{id}
    - height_increase: refined - original
    - height_increase_pct: (refined - original) / original * 100
    - x_offset, y_offset: Sub-pixel offsets
    - radial_offset: sqrt(x^2 + y^2)
    """
    records = []

    for _, row in df.iterrows():
        for job_id in job_ids:
            orig_col = f'SCORE_JOB{job_id}'
            ref_col = f'REFINED_HEIGHT_JOB{job_id}'
            x_off_col = f'X_OFFSET_JOB{job_id}'
            y_off_col = f'Y_OFFSET_JOB{job_id}'

            # Check if refined height exists
            if ref_col not in df.columns:
                continue

            orig = row[orig_col]
            refined = row[ref_col]
            x_off = row.get(x_off_col, np.nan)
            y_off = row.get(y_off_col, np.nan)

            # Skip if any values are NaN
            if any(pd.isna([orig, refined, x_off, y_off])):
                continue

            height_increase = refined - orig
            height_increase_pct = (height_increase / orig) * 100 if orig != 0 else 0
            radial_offset = np.sqrt(x_off**2 + y_off**2)

            records.append({
                'job_id': job_id,
                'image_id': row['IMAGE_ASSET_ID'],
                'peak_id': row['CONSENSUS_PEAK_ID'],
                'original_height': orig,
                'refined_height': refined,
                'height_increase': height_increase,
                'height_increase_pct': height_increase_pct,
                'x_offset': x_off,
                'y_offset': y_off,
                'radial_offset': radial_offset
            })

    return pd.DataFrame(records)


def plot_correlation(data: pd.DataFrame, output_path: str = None):
    """Create correlation plots."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Sub-Pixel Offset vs Refined Peak Height Increase', fontsize=14)

    # 1. Radial offset vs height increase (absolute)
    ax1 = axes[0, 0]
    ax1.scatter(data['radial_offset'], data['height_increase'], alpha=0.5, s=20)

    # Fit line through origin: y = slope * x
    mask = ~(np.isnan(data['radial_offset']) | np.isnan(data['height_increase']))
    if mask.sum() > 1:
        x = data.loc[mask, 'radial_offset'].values
        y = data.loc[mask, 'height_increase'].values
        slope = np.sum(x * y) / np.sum(x ** 2)  # Least squares through origin
        ss_res = np.sum((y - slope * x) ** 2)
        ss_tot = np.sum(y ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        x_fit = np.linspace(0, data['radial_offset'].max(), 100)
        ax1.plot(x_fit, slope * x_fit, 'r-',
                 label=f'slope={slope:.3f}, R²={r_squared:.3f}')
        ax1.legend()

    ax1.set_xlabel('Radial Sub-Pixel Offset (pixels)')
    ax1.set_ylabel('Height Increase (refined - original)')
    ax1.set_title('Absolute Height Increase (fit through origin)')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # 2. Radial offset vs height increase (percentage)
    ax2 = axes[0, 1]
    ax2.scatter(data['radial_offset'], data['height_increase_pct'], alpha=0.5, s=20)

    if mask.sum() > 1:
        x = data.loc[mask, 'radial_offset'].values
        y = data.loc[mask, 'height_increase_pct'].values
        slope = np.sum(x * y) / np.sum(x ** 2)
        ss_res = np.sum((y - slope * x) ** 2)
        ss_tot = np.sum(y ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        x_fit = np.linspace(0, data['radial_offset'].max(), 100)
        ax2.plot(x_fit, slope * x_fit, 'r-',
                 label=f'slope={slope:.2f}%/pix, R²={r_squared:.3f}')
        ax2.legend()

    ax2.set_xlabel('Radial Sub-Pixel Offset (pixels)')
    ax2.set_ylabel('Height Increase (%)')
    ax2.set_title('Percentage Height Increase')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # 3. X and Y offsets separately (2D heatmap-style)
    ax3 = axes[1, 0]
    scatter = ax3.scatter(data['x_offset'], data['y_offset'],
                          c=data['height_increase_pct'], cmap='RdYlBu_r',
                          alpha=0.7, s=30)
    plt.colorbar(scatter, ax=ax3, label='Height Increase (%)')
    ax3.set_xlabel('X Offset (pixels)')
    ax3.set_ylabel('Y Offset (pixels)')
    ax3.set_title('Offset Position colored by Height Increase')
    ax3.set_xlim(-0.05, 0.55)
    ax3.set_ylim(-0.05, 0.55)

    # 4. Distribution of height increases
    ax4 = axes[1, 1]
    ax4.hist(data['height_increase_pct'], bins=50, edgecolor='black', alpha=0.7)
    ax4.axvline(x=0, color='red', linestyle='--', label='No change')
    ax4.axvline(x=data['height_increase_pct'].mean(), color='green', linestyle='-',
                label=f'Mean: {data["height_increase_pct"].mean():.2f}%')
    ax4.axvline(x=data['height_increase_pct'].median(), color='orange', linestyle='-',
                label=f'Median: {data["height_increase_pct"].median():.2f}%')
    ax4.set_xlabel('Height Increase (%)')
    ax4.set_ylabel('Count')
    ax4.set_title('Distribution of Height Changes')
    ax4.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()


def print_statistics(data: pd.DataFrame):
    """Print summary statistics."""

    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS: Sub-Pixel Offset vs Height Increase")
    print("=" * 70)

    print(f"\nData points: {len(data)}")
    print(f"Unique peaks: {data.groupby(['image_id', 'peak_id']).ngroups}")
    print(f"Job IDs: {sorted(data['job_id'].unique())}")

    # Height increase statistics
    print(f"\nHeight Increase Statistics:")
    print(f"  Mean:   {data['height_increase'].mean():+.4f} ({data['height_increase_pct'].mean():+.2f}%)")
    print(f"  Median: {data['height_increase'].median():+.4f} ({data['height_increase_pct'].median():+.2f}%)")
    print(f"  Std:    {data['height_increase'].std():.4f} ({data['height_increase_pct'].std():.2f}%)")
    print(f"  Min:    {data['height_increase'].min():+.4f} ({data['height_increase_pct'].min():+.2f}%)")
    print(f"  Max:    {data['height_increase'].max():+.4f} ({data['height_increase_pct'].max():+.2f}%)")

    # Offset statistics
    print(f"\nSub-Pixel Offset Statistics:")
    print(f"  Mean radial offset: {data['radial_offset'].mean():.4f} pixels")
    print(f"  Max radial offset:  {data['radial_offset'].max():.4f} pixels")

    # Linear fit through origin: y = slope * x
    print(f"\nLinear Fit (through origin: height_increase = slope * offset):")

    x = data['radial_offset'].values
    y_abs = data['height_increase'].values
    y_pct = data['height_increase_pct'].values

    # Slope through origin
    slope_abs = np.sum(x * y_abs) / np.sum(x ** 2)
    slope_pct = np.sum(x * y_pct) / np.sum(x ** 2)

    # R² for line through origin
    ss_res_abs = np.sum((y_abs - slope_abs * x) ** 2)
    ss_tot_abs = np.sum(y_abs ** 2)
    r2_abs = 1 - ss_res_abs / ss_tot_abs if ss_tot_abs > 0 else 0

    ss_res_pct = np.sum((y_pct - slope_pct * x) ** 2)
    ss_tot_pct = np.sum(y_pct ** 2)
    r2_pct = 1 - ss_res_pct / ss_tot_pct if ss_tot_pct > 0 else 0

    print(f"  Absolute:   slope = {slope_abs:+.4f} per pixel, R² = {r2_abs:.4f}")
    print(f"  Percentage: slope = {slope_pct:+.2f}% per pixel, R² = {r2_pct:.4f}")

    # Interpretation based on slope
    print(f"\nInterpretation:")
    if slope_pct > 0.5:
        print(f"  Positive slope ({slope_pct:.2f}%/pixel): peaks further from pixel centers")
        print(f"  show larger height increases after FFT refinement.")
        print(f"  This supports the hypothesis that sub-pixel positioning affects peak height.")
    elif slope_pct < -0.5:
        print(f"  Negative slope ({slope_pct:.2f}%/pixel): peaks closer to pixel centers")
        print(f"  show larger height increases (unexpected).")
    else:
        print(f"  Slope near zero ({slope_pct:.2f}%/pixel): height increase appears")
        print(f"  largely independent of sub-pixel offset.")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze correlation between sub-pixel offset and refined peak height',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script analyzes the output from multi_search_consensus.py when run with
--subpixel-method fft to test whether peaks further from pixel centers show
larger height increases after FFT upsampling refinement.

Example:
  python analyze_subpixel_height_correlation.py consensus.csv --output correlation.png
        """
    )
    parser.add_argument('csv_file', help='CSV output from multi_search_consensus.py')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output plot file (shows interactively if not specified)')

    args = parser.parse_args()

    # Load data
    print(f"Loading: {args.csv_file}")
    df = pd.read_csv(args.csv_file)
    print(f"  Loaded {len(df)} consensus peaks")

    # Extract job IDs
    job_ids = extract_job_ids(df)
    print(f"  Found job IDs: {job_ids}")

    # Check for refined height columns
    ref_cols = [c for c in df.columns if c.startswith('REFINED_HEIGHT_JOB')]
    if not ref_cols:
        print("\nError: No REFINED_HEIGHT_JOB columns found.")
        print("Make sure you ran multi_search_consensus.py with --subpixel-method fft")
        sys.exit(1)

    # Compute correlations
    data = compute_correlations(df, job_ids)

    if len(data) == 0:
        print("\nError: No valid data points found for correlation analysis.")
        sys.exit(1)

    # Print statistics
    print_statistics(data)

    # Plot
    plot_correlation(data, args.output)


if __name__ == '__main__':
    main()
