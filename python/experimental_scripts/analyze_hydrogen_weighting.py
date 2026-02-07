#!/usr/bin/env python3
"""
Analyze template matching peak heights as a function of hydrogen weighting.

Compares experimental TM results against positive and negative controls
to assess how hydrogen atom inclusion in templates affects detection.

Usage:
    python analyze_hydrogen_weighting.py <db_path> \\
        --positive-control <job_id> \\
        --negative-control <job_id> \\
        --jobs <range> \\
        [--n-stddev 1.0] \\
        [--output plot.png] \\
        [--csv results.csv]

Example:
    python analyze_hydrogen_weighting.py /path/to/project.db \\
        --positive-control 5 \\
        --negative-control 6 \\
        --jobs 7:12 \\
        --output h_weighting.png
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from cistemx import parse_job_range
from cistemx.db import TemplateMatchAnalyzer


def extract_hydrogen_weighting(template_name: str) -> float:
    """
    Extract hydrogen weighting factor from template name.

    Expects naming convention: *weight_<value>[.mrc]

    Args:
        template_name: Template asset NAME from database

    Returns:
        Hydrogen weighting as float, or NaN if not extractable

    Examples:
        "6TTE-assembly1-noHOH-weight_-1.50" -> -1.50
        "ribosome-weight_0.5.mrc" -> 0.5
        "template_weight_-0.25.mrc" -> -0.25
    """
    try:
        # Split on "weight_" and take the part after it
        after_weight = template_name.split('weight_')[1]
        # Remove .mrc extension if present
        value_str = after_weight.replace('.mrc', '')
        return float(value_str)
    except (IndexError, ValueError) as e:
        print(f"Warning: Could not extract hydrogen weighting from '{template_name}': {e}")
        return float('nan')


def load_control_stats(analyzer: TemplateMatchAnalyzer,
                       job_id: int,
                       label: str) -> tuple:
    """
    Load peaks and compute stats for a control job.

    Args:
        analyzer: TemplateMatchAnalyzer instance
        job_id: TM job ID for the control
        label: Label for logging (e.g., "Positive control")

    Returns:
        Tuple of (mean_peak_height, n_peaks, job_id)

    Raises:
        ValueError: If control job has no peaks
    """
    peaks = analyzer.load_all_peaks_for_jobs([job_id])
    if len(peaks) == 0:
        raise ValueError(f"{label} (job {job_id}) has no peaks")

    mean = peaks['PEAK_HEIGHT'].mean()
    n_peaks = len(peaks)

    print(f"{label} (job {job_id}): mean={mean:.3f}, n={n_peaks}")
    return (mean, n_peaks, job_id)


def load_experimental_stats(analyzer: TemplateMatchAnalyzer,
                            job_ids: list) -> list:
    """
    Load peaks and compute stats for experimental jobs.

    Args:
        analyzer: TemplateMatchAnalyzer instance
        job_ids: List of TM job IDs to analyze

    Returns:
        List of tuples: (h_weight, mean, std, n_peaks, job_id)
        Sorted by hydrogen weighting
    """
    results = []

    for job_id in job_ids:
        try:
            peaks = analyzer.load_all_peaks_for_jobs([job_id])
            if len(peaks) == 0:
                print(f"Warning: Job {job_id} has no peaks, skipping")
                continue

            template_info = analyzer.get_template_info(job_id)
            h_weight = extract_hydrogen_weighting(template_info['NAME'])

            mean = peaks['PEAK_HEIGHT'].mean()
            std = peaks['PEAK_HEIGHT'].std()
            n_peaks = len(peaks)

            print(f"Job {job_id}: template='{template_info['NAME']}', "
                  f"h_weight={h_weight}, mean={mean:.3f}, "
                  f"std={std:.3f}, n={n_peaks}")

            results.append((h_weight, mean, std, n_peaks, job_id))

        except Exception as e:
            print(f"Warning: Error processing job {job_id}: {e}")
            continue

    # Sort by hydrogen weighting (NaN values go to end)
    results.sort(key=lambda x: x[0] if not np.isnan(x[0]) else float('inf'))
    return results


def plot_hydrogen_analysis(
    pos_control: tuple,
    neg_control: tuple,
    results: list,
    n_stddev: float = 1.0,
    title: str = None,
    output_path: str = None,
    show_job_ids: bool = False
):
    """
    Generate hydrogen weighting analysis plot.

    All values are normalized to the positive control (which equals 1.0).

    Args:
        pos_control: (mean, n_peaks, job_id) for positive control
        neg_control: (mean, n_peaks, job_id) for negative control
        results: List of (h_weight, mean, std, n_peaks, job_id) tuples
        n_stddev: Number of standard deviations for error whiskers
        title: Optional custom plot title
        output_path: If provided, save plot to this path; otherwise show interactively
        show_job_ids: If True, label each point with its job ID
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Normalization factor (positive control mean)
    norm_factor = pos_control[0]

    # Extract and normalize data from results
    h_weights = [r[0] for r in results]
    means_norm = [r[1] / norm_factor for r in results]
    stds_norm = [r[2] / norm_factor for r in results]

    # Normalized control values
    pos_norm = 1.0  # By definition
    neg_norm = neg_control[0] / norm_factor

    # Determine x-axis range for control lines
    if h_weights:
        x_min = min(h_weights) - 0.05
        x_max = max(h_weights) + 0.05
    else:
        x_min, x_max = 0, 1

    # Plot control lines spanning the full x-range (with raw values in legend)
    ax.axhline(y=pos_norm, color='blue', linestyle='-', linewidth=2,
               label=f'Positive control (raw: {pos_control[0]:.2f}, n={pos_control[1]})')
    ax.axhline(y=neg_norm, color='red', linestyle='-', linewidth=2,
               label=f'Negative control (raw: {neg_control[0]:.2f}, n={neg_control[1]})')

    # Plot experimental points (with error bars if n_stddev > 0)
    if results:
        if n_stddev > 0:
            yerr_norm = [s * n_stddev for s in stds_norm]
            ax.errorbar(h_weights, means_norm, yerr=yerr_norm, fmt='o', capsize=5,
                        capthick=1.5, elinewidth=1.5, markersize=8,
                        markerfacecolor='none', markeredgecolor='black', color='black',
                        label=f'Experimental ({n_stddev:.1f}σ whiskers)')
        else:
            ax.plot(h_weights, means_norm, 'o', markersize=8,
                    markerfacecolor='none', markeredgecolor='black',
                    label='Experimental')

        # Annotate points with job IDs if requested
        if show_job_ids:
            for i, r in enumerate(results):
                ax.annotate(f'{r[4]}', (h_weights[i], means_norm[i]),
                            textcoords="offset points", xytext=(3, 5),
                            fontsize=7, rotation=45, alpha=0.8)

    # Configure axes and labels
    ax.set_xlabel('Hydrogen Weighting Factor', fontsize=12)
    ax.set_ylabel('Normalized Mean Peak Height', fontsize=12)
    ax.set_xlim(x_min, x_max)

    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title('Template Matching Peak Height vs Hydrogen Weighting', fontsize=14)

    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()

    return fig, ax


def export_csv(pos_control, neg_control, results, output_path):
    """
    Export results to CSV file.

    Args:
        pos_control: (mean, n_peaks, job_id) for positive control
        neg_control: (mean, n_peaks, job_id) for negative control
        results: List of (h_weight, mean, std, n_peaks, job_id) tuples
        output_path: Path to save CSV file
    """
    rows = []

    # Add controls
    rows.append({
        'job_id': pos_control[2],
        'type': 'positive_control',
        'h_weight': np.nan,
        'mean_peak_height': pos_control[0],
        'std_peak_height': np.nan,
        'n_peaks': pos_control[1]
    })
    rows.append({
        'job_id': neg_control[2],
        'type': 'negative_control',
        'h_weight': np.nan,
        'mean_peak_height': neg_control[0],
        'std_peak_height': np.nan,
        'n_peaks': neg_control[1]
    })

    # Add experimental results
    for r in results:
        rows.append({
            'job_id': r[4],
            'type': 'experimental',
            'h_weight': r[0],
            'mean_peak_height': r[1],
            'std_peak_height': r[2],
            'n_peaks': r[3]
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Results exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze TM peak heights vs hydrogen weighting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('db_path', help='Path to cisTEM database')
    parser.add_argument('--positive-control', type=int, required=True,
                        help='TM job ID for positive control')
    parser.add_argument('--negative-control', type=int, required=True,
                        help='TM job ID for negative control')
    parser.add_argument('--jobs', type=str, required=True,
                        help='Experimental job IDs (e.g., "7:12" or "7,9,11")')
    parser.add_argument('--n-stddev', type=float, default=1.0,
                        help='Number of std devs for error whiskers (default: 1.0)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Save plot to file (interactive if not specified)')
    parser.add_argument('--csv', type=str, default=None,
                        help='Export results to CSV file')
    parser.add_argument('--title', type=str, default=None,
                        help='Custom plot title')
    parser.add_argument('--show-job-ids', action='store_true',
                        help='Show job ID labels on data points')

    args = parser.parse_args()

    # Validate database exists
    if not Path(args.db_path).exists():
        print(f"Error: Database not found: {args.db_path}")
        sys.exit(1)

    # Parse job range
    try:
        job_ids = parse_job_range(args.jobs)
    except ValueError as e:
        print(f"Error parsing job range: {e}")
        sys.exit(1)

    print("=" * 60)
    print("Hydrogen Weighting Analysis")
    print("=" * 60)
    print(f"Database: {args.db_path}")
    print(f"Positive control: job {args.positive_control}")
    print(f"Negative control: job {args.negative_control}")
    print(f"Experimental jobs: {job_ids}")
    print()

    try:
        analyzer = TemplateMatchAnalyzer(args.db_path)

        # Load controls
        pos_control = load_control_stats(analyzer, args.positive_control,
                                         "Positive control")
        neg_control = load_control_stats(analyzer, args.negative_control,
                                         "Negative control")

        # Load experimental data
        print("\nLoading experimental data...")
        results = load_experimental_stats(analyzer, job_ids)

        if not results:
            print("Error: No experimental data loaded")
            sys.exit(1)

        # Generate plot
        print("\nGenerating plot...")
        plot_hydrogen_analysis(pos_control, neg_control, results,
                               n_stddev=args.n_stddev,
                               title=args.title,
                               output_path=args.output,
                               show_job_ids=args.show_job_ids)

        # Export CSV if requested
        if args.csv:
            export_csv(pos_control, neg_control, results, args.csv)

        print("\nAnalysis complete!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
