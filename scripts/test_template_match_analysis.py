#!/usr/bin/env python3
"""
Test script for template_match_analysis module.

Tests group-based filtering and calculates peak statistics.

Usage:
    python test_template_match_analysis.py <db_path> <job_id> <group_name>

Example:
    python test_template_match_analysis.py /scratch/salina/proc_EMPIAR_11063/New_Project/full_run/cp.db 8 "Good Images"
"""

import sys
import argparse
import template_match_analysis as tma


def main():
    parser = argparse.ArgumentParser(
        description='Test template matching analysis module',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s project.db 8 "Good Images"
  %(prog)s /scratch/salina/proc_EMPIAR_11063/New_Project/full_run/cp.db 8 "All Images"
        """
    )
    parser.add_argument('db_path', help='Path to cisTEM database file')
    parser.add_argument('job_id', type=int, help='Template match job ID to test')
    parser.add_argument('group_name', help='Image group name to filter by')

    args = parser.parse_args()

    print("=" * 70)
    print("Template Match Analysis - Group Filtering Test")
    print("=" * 70)
    print(f"Database path: {args.db_path}")
    print(f"Job ID: {args.job_id}")
    print(f"Image group: {args.group_name}")
    print()

    try:
        # Initialize analyzer
        print("Initializing analyzer...")
        analyzer = tma.TemplateMatchAnalyzer(args.db_path)
        print("✓ Database validated successfully")
        print()

        # List available groups
        print("Available image groups:")
        groups = analyzer.get_all_image_groups()
        for group_id, group_name in groups:
            print(f"  - {group_name} (ID: {group_id})")
        print()

        # Test get_peaks_by_group
        print(f"Loading peaks for job {args.job_id} filtered by group '{args.group_name}'...")
        peaks_df = analyzer.get_peaks_by_group(args.job_id, args.group_name)
        print("✓ Peaks loaded")
        print()

        # Display results
        print("=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"Total peaks returned: {len(peaks_df)}")
        print(f"Number of images: {peaks_df['IMAGE_ASSET_ID'].nunique() if len(peaks_df) > 0 else 0}")
        print()

        if len(peaks_df) > 0:
            # Show DataFrame info
            print("DataFrame shape:", peaks_df.shape)
            print("Columns:", list(peaks_df.columns))
            print()

            # Show first few peaks
            print("First 5 peaks:")
            display_cols = ['TEMPLATE_MATCH_JOB_ID', 'IMAGE_ASSET_ID', 'PEAK_NUMBER', 'PEAK_HEIGHT']
            print(peaks_df[display_cols].head(5).to_string())
            print()

            # Calculate statistics on PEAK_HEIGHT
            print("=" * 70)
            print("PEAK HEIGHT STATISTICS")
            print("=" * 70)
            mean_height = peaks_df['PEAK_HEIGHT'].mean()
            std_height = peaks_df['PEAK_HEIGHT'].std()
            min_height = peaks_df['PEAK_HEIGHT'].min()
            max_height = peaks_df['PEAK_HEIGHT'].max()
            median_height = peaks_df['PEAK_HEIGHT'].median()

            print(f"Mean peak height (SNR): {mean_height:.3f}")
            print(f"Standard deviation:     {std_height:.3f}")
            print(f"Minimum:                {min_height:.3f}")
            print(f"Maximum:                {max_height:.3f}")
            print(f"Median:                 {median_height:.3f}")
            print()

            # Per-image statistics
            print("=" * 70)
            print("PER-IMAGE STATISTICS")
            print("=" * 70)
            per_image_stats = peaks_df.groupby('IMAGE_ASSET_ID')['PEAK_HEIGHT'].agg(['count', 'mean', 'std'])
            per_image_stats.columns = ['Peak Count', 'Mean Height', 'Std Height']
            print(per_image_stats.head(10).to_string())
            print()
            print(f"Total images: {len(per_image_stats)}")

        else:
            print("No peaks found for this group.")

        print()
        print("=" * 70)
        print("Test complete!")
        print("=" * 70)

    except FileNotFoundError as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
