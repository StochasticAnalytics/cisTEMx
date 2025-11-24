#!/usr/bin/env python3
"""
Test script for template_match_analysis module.

Tests database validation and initializes analyzer object.

Usage:
    python test_template_match_analysis.py <db_path>

Example:
    python test_template_match_analysis.py /scratch/salina/proc_EMPIAR_11063/New_Project/full_run/cp.db
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
  %(prog)s project.db
  %(prog)s /scratch/salina/proc_EMPIAR_11063/New_Project/full_run/cp.db
        """
    )
    parser.add_argument('db_path', help='Path to cisTEM database file')
    parser.add_argument('job_id', type=int, help='Template match job ID to test')
    parser.add_argument('--operator', default='>=', help='Comparison operator (default: >=)')
    parser.add_argument('--threshold', type=int, default=4, help='Peak count threshold (default: 4)')

    args = parser.parse_args()

    print("=" * 70)
    print("Template Match Analysis - Peak Count Filter Test")
    print("=" * 70)
    print(f"Database path: {args.db_path}")
    print(f"Job ID: {args.job_id}")
    print(f"Filter: images with {args.operator} {args.threshold} peaks")
    print()

    try:
        # Initialize analyzer
        print("Initializing analyzer...")
        analyzer = tma.TemplateMatchAnalyzer(args.db_path)
        print("✓ Database validated successfully")
        print()

        # Test get_peaks_by_count
        print(f"Loading peaks for job {args.job_id} with filter...")
        peaks_df = analyzer.get_peaks_by_count(args.job_id, args.operator, args.threshold)
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
            # Show peak counts per image
            peak_counts = peaks_df.groupby('IMAGE_ASSET_ID').size().sort_values(ascending=False)
            print("Peak counts per image:")

            # Show DataFrame info
            print("DataFrame shape:", peaks_df.shape)
            print("Columns:", list(peaks_df.columns))
            print()

            # Show first few peaks
            print("First 5 peaks:")
            print(peaks_df.head(5).to_string())
        else:
            print("No images matched the condition.")

        print()
        print("=" * 70)
        print("Test complete!")
        print("=" * 70)

    except FileNotFoundError as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
