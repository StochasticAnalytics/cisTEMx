#!/usr/bin/env python3
"""
Analyze correlation between high-resolution limit and peak height increase.

This script examines whether the FFT refinement height increase varies
systematically with the high-resolution cutoff used in template matching.

Hypothesis: Higher resolution searches (lower Å values) produce sharper
correlation peaks that are more sensitive to sub-pixel positioning, leading
to larger height increases after FFT refinement.

Usage:
    python analyze_resolution_height_correlation.py <consensus_csv> [options]

Resolution mapping (default):
    Job 12: 3.0 Å, Job 13: 3.5 Å, Job 14: 4.0 Å, etc.
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def extract_job_ids(df: pd.DataFrame) -> list[int]:
    """Extract job IDs from column names."""
    job_ids = []
    for col in df.columns:
        if col.startswith('SCORE_JOB'):
            try:
                job_id = int(col.replace('SCORE_JOB', ''))
                job_ids.append(job_id)
            except ValueError:
                continue
    return sorted(job_ids)


def job_id_to_resolution(job_id: int, base_job: int, base_res: float, step: float) -> float:
    """Convert job ID to high-resolution limit in Angstroms."""
    return base_res + (job_id - base_job) * step


def compute_per_job_stats(df: pd.DataFrame, job_ids: list[int],
                           base_job: int, base_res: float, step: float) -> pd.DataFrame:
    """
    Compute average height increase statistics per job/resolution.

    Returns DataFrame with one row per job containing:
    - job_id, resolution
    - mean/median/std of height increase (absolute and %)
    - sample count
    """
    records = []

    for job_id in job_ids:
        orig_col = f'SCORE_JOB{job_id}'
        ref_col = f'REFINED_HEIGHT_JOB{job_id}'

        if ref_col not in df.columns:
            continue

        # Get valid pairs
        mask = df[orig_col].notna() & df[ref_col].notna()
        orig = df.loc[mask, orig_col]
        refined = df.loc[mask, ref_col]

        if len(orig) == 0:
            continue

        height_increase = refined - orig
        height_increase_pct = (height_increase / orig) * 100

        resolution = job_id_to_resolution(job_id, base_job, base_res, step)

        records.append({
            'job_id': job_id,
            'resolution': resolution,
            'n_peaks': len(orig),
            'mean_increase': height_increase.mean(),
            'median_increase': height_increase.median(),
            'std_increase': height_increase.std(),
            'mean_increase_pct': height_increase_pct.mean(),
            'median_increase_pct': height_increase_pct.median(),
            'std_increase_pct': height_increase_pct.std(),
            'mean_original': orig.mean(),
            'mean_refined': refined.mean(),
        })

    return pd.DataFrame(records)


def plot_resolution_correlation(job_stats: pd.DataFrame, output_path: str = None):
    """Create plots showing resolution vs height increase."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('High-Resolution Limit vs FFT Refinement Height Increase', fontsize=14)

    # 1. Resolution vs mean height increase (absolute)
    ax1 = axes[0, 0]
    ax1.errorbar(job_stats['resolution'], job_stats['mean_increase'],
                 yerr=job_stats['std_increase'] / np.sqrt(job_stats['n_peaks']),
                 fmt='o-', capsize=5, markersize=8)

    # Label each point with job ID
    for _, row in job_stats.iterrows():
        ax1.annotate(f"J{int(row['job_id'])}",
                     (row['resolution'], row['mean_increase']),
                     textcoords="offset points", xytext=(5, 5), fontsize=9)

    # Fit line through origin: y = slope * x (no intercept)
    if len(job_stats) > 1:
        x = job_stats['resolution'].values
        y = job_stats['mean_increase'].values
        slope = np.sum(x * y) / np.sum(x ** 2)  # Least squares through origin
        # R² for line through origin
        ss_res = np.sum((y - slope * x) ** 2)
        ss_tot = np.sum(y ** 2)  # Note: for origin-constrained, use y² not (y-mean)²
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        x_fit = np.linspace(0, job_stats['resolution'].max(), 100)
        ax1.plot(x_fit, slope * x_fit, 'r--', alpha=0.7,
                 label=f'slope={slope:.4f}, R²={r_squared:.3f}')
        ax1.legend()

    ax1.set_xlabel('High-Resolution Limit (Å)')
    ax1.set_ylabel('Mean Height Increase')
    ax1.set_title('Absolute Height Increase (fit through origin)')
    ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax1.invert_xaxis()  # Lower Å = higher resolution on left

    # 2. Resolution vs mean height increase (percentage)
    ax2 = axes[0, 1]
    ax2.errorbar(job_stats['resolution'], job_stats['mean_increase_pct'],
                 yerr=job_stats['std_increase_pct'] / np.sqrt(job_stats['n_peaks']),
                 fmt='s-', capsize=5, markersize=8, color='green')

    for _, row in job_stats.iterrows():
        ax2.annotate(f"J{int(row['job_id'])}",
                     (row['resolution'], row['mean_increase_pct']),
                     textcoords="offset points", xytext=(5, 5), fontsize=9)

    if len(job_stats) > 1:
        x = job_stats['resolution'].values
        y = job_stats['mean_increase_pct'].values
        slope = np.sum(x * y) / np.sum(x ** 2)
        ss_res = np.sum((y - slope * x) ** 2)
        ss_tot = np.sum(y ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        x_fit = np.linspace(0, job_stats['resolution'].max(), 100)
        ax2.plot(x_fit, slope * x_fit, 'r--', alpha=0.7,
                 label=f'slope={slope:.4f}, R²={r_squared:.3f}')
        ax2.legend()

    ax2.set_xlabel('High-Resolution Limit (Å)')
    ax2.set_ylabel('Mean Height Increase (%)')
    ax2.set_title('Percentage Height Increase')
    ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax2.invert_xaxis()

    # 3. Bar chart of height increase by job
    ax3 = axes[1, 0]
    x_pos = range(len(job_stats))
    bars = ax3.bar(x_pos, job_stats['mean_increase_pct'],
                   yerr=job_stats['std_increase_pct'] / np.sqrt(job_stats['n_peaks']),
                   capsize=3, alpha=0.7, color='steelblue')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f"J{int(j)}\n{r:.1f}Å"
                         for j, r in zip(job_stats['job_id'], job_stats['resolution'])])
    ax3.set_xlabel('Job ID / Resolution')
    ax3.set_ylabel('Mean Height Increase (%)')
    ax3.set_title('Height Increase by Search')
    ax3.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

    # 4. Original vs refined height comparison
    ax4 = axes[1, 1]
    width = 0.35
    x_pos = np.arange(len(job_stats))
    ax4.bar(x_pos - width/2, job_stats['mean_original'], width, label='Original', alpha=0.7)
    ax4.bar(x_pos + width/2, job_stats['mean_refined'], width, label='Refined', alpha=0.7)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f"J{int(j)}\n{r:.1f}Å"
                         for j, r in zip(job_stats['job_id'], job_stats['resolution'])])
    ax4.set_xlabel('Job ID / Resolution')
    ax4.set_ylabel('Mean Peak Height')
    ax4.set_title('Original vs Refined Peak Heights')
    ax4.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()


def print_statistics(job_stats: pd.DataFrame):
    """Print summary statistics."""

    print("\n" + "=" * 80)
    print("RESOLUTION vs HEIGHT INCREASE ANALYSIS")
    print("=" * 80)

    # Per-job table
    print(f"\nPer-Search Statistics:")
    print("-" * 80)
    print(f"{'Job':>5} {'Res(Å)':>7} {'N':>6} {'Mean Δ':>10} {'Mean Δ%':>10} {'Std Δ%':>10}")
    print("-" * 80)

    for _, row in job_stats.iterrows():
        print(f"{int(row['job_id']):>5} {row['resolution']:>7.1f} {int(row['n_peaks']):>6} "
              f"{row['mean_increase']:>+10.4f} {row['mean_increase_pct']:>+10.2f}% "
              f"{row['std_increase_pct']:>10.2f}%")

    print("-" * 80)

    # Correlation analysis
    print(f"\nLinear Fit (through origin: y = slope * resolution):")

    if len(job_stats) > 1:
        x = job_stats['resolution'].values
        y_abs = job_stats['mean_increase'].values
        y_pct = job_stats['mean_increase_pct'].values

        # Slope through origin: slope = Σ(xy) / Σ(x²)
        slope_abs = np.sum(x * y_abs) / np.sum(x ** 2)
        slope_pct = np.sum(x * y_pct) / np.sum(x ** 2)

        # R² for line through origin
        ss_res_abs = np.sum((y_abs - slope_abs * x) ** 2)
        ss_tot_abs = np.sum(y_abs ** 2)
        r2_abs = 1 - ss_res_abs / ss_tot_abs if ss_tot_abs > 0 else 0

        ss_res_pct = np.sum((y_pct - slope_pct * x) ** 2)
        ss_tot_pct = np.sum(y_pct ** 2)
        r2_pct = 1 - ss_res_pct / ss_tot_pct if ss_tot_pct > 0 else 0

        print(f"  Absolute:   slope = {slope_abs:+.6f}, R² = {r2_abs:.4f}")
        print(f"  Percentage: slope = {slope_pct:+.4f}%/Å, R² = {r2_pct:.4f}")

        # Interpretation
        print(f"\nInterpretation:")
        if slope_pct > 0:
            print(f"  Positive slope: Lower resolution (higher Å) shows LARGER")
            print(f"  height increases. This is unexpected - smoother peaks should be")
            print(f"  less affected by sub-pixel positioning.")
        elif slope_pct < 0:
            print(f"  Negative slope: Higher resolution (lower Å) shows LARGER")
            print(f"  height increases. This supports the hypothesis that sharper peaks")
            print(f"  are more sensitive to sub-pixel positioning artifacts.")
        else:
            print(f"  Slope near zero: Height increase appears independent of resolution.")
    else:
        print("  Not enough data points for analysis (need >1 job)")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze correlation between resolution limit and height increase',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Resolution mapping: The script assumes job IDs map to resolution limits.
Default: Job 12 = 3.0 Å, incrementing by 0.5 Å per job.

Example:
  python analyze_resolution_height_correlation.py consensus.csv -o resolution_plot.png
  python analyze_resolution_height_correlation.py consensus.csv --base-job 12 --base-res 3.0
        """
    )
    parser.add_argument('csv_file', help='CSV output from multi_search_consensus.py')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output plot file (shows interactively if not specified)')
    parser.add_argument('--base-job', type=int, default=12,
                        help='Job ID corresponding to base resolution (default: 12)')
    parser.add_argument('--base-res', type=float, default=3.0,
                        help='Resolution in Å for base job (default: 3.0)')
    parser.add_argument('--step', type=float, default=0.5,
                        help='Resolution increment per job ID (default: 0.5)')

    args = parser.parse_args()

    # Load data
    print(f"Loading: {args.csv_file}")
    df = pd.read_csv(args.csv_file)
    print(f"  Loaded {len(df)} consensus peaks")

    # Extract job IDs
    job_ids = extract_job_ids(df)
    print(f"  Found job IDs: {job_ids}")

    # Show resolution mapping
    print(f"\nResolution mapping (base job {args.base_job} = {args.base_res} Å, step = {args.step} Å):")
    for job_id in job_ids:
        res = job_id_to_resolution(job_id, args.base_job, args.base_res, args.step)
        print(f"  Job {job_id}: {res:.1f} Å")

    # Check for refined height columns
    ref_cols = [c for c in df.columns if c.startswith('REFINED_HEIGHT_JOB')]
    if not ref_cols:
        print("\nError: No REFINED_HEIGHT_JOB columns found.")
        print("Make sure you ran multi_search_consensus.py with --subpixel-method fft")
        sys.exit(1)

    # Compute per-job statistics
    job_stats = compute_per_job_stats(df, job_ids, args.base_job, args.base_res, args.step)

    if len(job_stats) == 0:
        print("\nError: No valid data found for analysis.")
        sys.exit(1)

    # Print statistics
    print_statistics(job_stats)

    # Plot
    plot_resolution_correlation(job_stats, args.output)


if __name__ == '__main__':
    main()
