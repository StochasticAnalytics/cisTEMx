#!/usr/bin/env python3
"""
Plot resolution analysis comparing Part_FSC vs Resolution for different offset conditions.

This script parses Refine3D statistics files and generates a comparative plot
showing Particle FSC as a function of resolution for negative, positive, and zero
offset conditions.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def parse_stats_file(filepath):
    """
    Parse Refine3D statistics file to extract resolution and Part_FSC data.

    Args:
        filepath: Path to statistics file

    Returns:
        tuple: (resolution_array, part_fsc_array)
    """
    resolution = []
    part_fsc = []

    with open(filepath, 'r') as f:
        # Skip header lines (first 35 lines)
        # Use a safer approach that handles files with fewer lines
        for i in range(35):
            try:
                next(f)
            except StopIteration:
                print(f"Warning: {filepath} has fewer than 35 header lines (found {i} lines)")
                return np.array([]), np.array([])

        # Parse data lines
        for line in f:
            parts = line.split()
            if len(parts) >= 5:
                try:
                    res = float(parts[1])  # Column 2: RESOLUTION
                    fsc = float(parts[4])  # Column 5: Part_FSC
                    resolution.append(res)
                    part_fsc.append(fsc)
                except ValueError:
                    continue

    return np.array(resolution), np.array(part_fsc)

def plot_resolution_comparison(stat_files, max_resolution=None):
    """
    Create comparative plot of Part_FSC vs Spatial Frequency for all offset conditions.

    Uses spatial frequency (1/resolution) for x-axis to properly represent uniform
    shell spacing in Fourier space.

    Args:
        stat_files: List of paths to statistics files
        max_resolution: Maximum resolution to plot in Angstroms (optional, plots all if None)
    """
    # Default colors cycle
    default_colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', '#9467bd', '#8c564b']

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each dataset
    for idx, filepath in enumerate(stat_files):
        filepath = Path(filepath)
        if filepath.exists():
            resolution, part_fsc = parse_stats_file(filepath)

            # Skip files that couldn't be parsed
            if len(resolution) == 0:
                continue

            # Apply resolution limit if specified
            # max_resolution cuts off the high-resolution end (small Å values)
            # e.g., max_resolution=10 keeps resolution >= 10Å (spatial freq <= 0.1)
            if max_resolution is not None:
                mask = resolution >= max_resolution
                resolution = resolution[mask]
                part_fsc = part_fsc[mask]

            # Calculate spatial frequency (1/resolution) for uniform spacing
            spatial_freq = 1.0 / resolution

            # Use filename (without extension) as label
            label = filepath.stem.replace('_stats', '').replace('particles_', '').replace('_', ' ').title()
            color = default_colors[idx % len(default_colors)]

            ax.plot(spatial_freq, part_fsc,
                   label=label,
                   color=color,
                   linewidth=2,
                   alpha=0.8)
        else:
            print(f"Warning: {filepath} not found")

    # Add FSC threshold lines
    ax.axhline(y=0.143, color='gray', linestyle='--', linewidth=1,
               alpha=0.7, label='FSC=0.143 (Gold Standard)')
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1,
               alpha=0.7, label='FSC=0.5 (Traditional)')

    # Configure axes
    ax.set_xlabel('Spatial Frequency (1/Å)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Particle FSC', fontsize=12, fontweight='bold')
    ax.set_title('Resolution Analysis: Particle FSC vs Spatial Frequency\nComparison Across Offset Conditions',
                 fontsize=14, fontweight='bold', pad=20)

    # Add secondary x-axis with resolution labels for reference
    # The top axis should show resolution (Å) = 1 / spatial_frequency
    ax2 = ax.twiny()

    # Get the spatial frequency limits from the bottom axis
    sf_min, sf_max = ax.get_xlim()

    # The twin axis shares the same x-transform, so we need to:
    # 1. Choose nice resolution values
    # 2. Convert them to spatial frequency positions (1/resolution)
    # 3. Place ticks at those spatial frequency positions
    # 4. Label them with the resolution values

    # Choose nice round numbers for resolution labels
    res_values = np.array([20, 10, 7, 5, 4, 3.5, 3, 2.5, 2.2, 2.0])

    # Convert to spatial frequency positions
    sf_tick_positions = 1.0 / res_values

    # Filter to only those within the current spatial frequency range
    mask = (sf_tick_positions >= sf_min) & (sf_tick_positions <= sf_max)
    sf_tick_positions = sf_tick_positions[mask]
    res_labels = res_values[mask]

    # Set the ticks at spatial frequency positions, but label with resolution values
    ax2.set_xlim(ax.get_xlim())  # Match the bottom axis exactly
    ax2.set_xticks(sf_tick_positions)
    ax2.set_xticklabels([f'{r:.1f}' for r in res_labels])
    ax2.set_xlabel('Resolution (Å)', fontsize=11, style='italic', color='gray')

    # Set y-axis limits
    ax.set_ylim(-0.1, 1.05)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Legend
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

    # Tight layout
    plt.tight_layout()

    # Save figure to current working directory
    output_path = Path('resolution_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    # Also save as PDF for publication quality
    output_path_pdf = Path('resolution_comparison.pdf')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"PDF saved to: {output_path_pdf}")

    return fig, ax

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Plot FSC resolution analysis from cisTEM Refine3D statistics files',
        epilog='Example: %(prog)s file1_stats.txt file2_stats.txt file3_stats.txt --max-resolution 10'
    )
    parser.add_argument(
        'files',
        nargs='+',
        help='Statistics files to plot (one or more)'
    )
    parser.add_argument(
        '--max-resolution',
        type=float,
        default=None,
        metavar='ANGSTROMS',
        help='Maximum resolution to plot in Angstroms (e.g., --max-resolution 10 plots up to 10Å)'
    )
    args = parser.parse_args()

    # Generate plot
    plot_resolution_comparison(args.files, max_resolution=args.max_resolution)

    print("\nPlot generation complete!")
