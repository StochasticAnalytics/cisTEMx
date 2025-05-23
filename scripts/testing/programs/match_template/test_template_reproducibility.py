#!/usr/bin/env python3
"""
Template Matching Reproducibility Test

This script runs the cisTEM template matching GPU binary multiple times using the same input data
and parameters, then analyzes the reproducibility by comparing the resulting MIP images.

It uses the Apoferritin dataset for testing, and saves the MIP (Maximum Intensity Projection)
files to a temporary directory with unique filenames for each replicate.

The script loads the MIP images using the mrcfile package and compares them using numpy to
measure pixel similarity between replicates with various metrics.
"""

import annoying_hack

from os.path import join, exists
from os import makedirs
import cistem_test_utils.args as tmArgs
import cistem_test_utils.make_tmp_runfile as mktmp
import cistem_test_utils.run_job as runner
from cistem_test_utils.temp_dir_manager import TempDirManager
from cistem_test_utils.threshold_utils import extract_threshold_value
import mrcfile
import numpy as np
import tempfile
import os
import shutil
import sys
import re

# By default the "_gpu" suffix will be added unless the --old-cistem flag is used
# or the --cpu flag is used
wanted_binary_name = 'match_template'

# Define the number of replicates to run
NUM_REPLICATES = 4


def main():
    try:
        # Create a temp_dir_manager instance
        temp_manager = TempDirManager()
        
        # Parse command-line arguments
        args = tmArgs.parse_TM_args(wanted_binary_name)

        # Handle temp directory management options
        if args.list_temp_dirs:
            temp_manager.print_temp_dirs()
            return 0

        if args.rm_temp_dir is not None:
            success, message = temp_manager.remove_temp_dir(args.rm_temp_dir)
            print(message)
            return 0 if success else 1

        if args.rm_all_temp_dirs:
            success_count, failure_count = temp_manager.remove_all_temp_dirs()
            print(f"Successfully removed {success_count} temporary directories.")
            if failure_count > 0:
                print(f"Failed to remove {failure_count} temporary directories.")
            return 0 if failure_count == 0 else 1

        # Create a temporary directory to store our replicate MIPs and track it
        temp_dir = temp_manager.create_temp_dir(prefix="template_match_reproducibility_")
        print(f"Temporary directory created at: {temp_dir}")

        # We'll run template matching for the defined number of replicates
        elapsed_time = [0] * NUM_REPLICATES
        mip_filenames = []
        hist_filenames = []
        threshold_values = []

        # Run the template matching for each replicate
        for replicate in range(NUM_REPLICATES):
            try:
                # Use Apoferritin dataset with image 0
                config = tmArgs.get_config(args, 'Apoferritin', 0, 0)

                # Create a unique output file prefix for each replicate
                original_prefix = config['output_file_prefix']
                config['output_file_prefix'] = join(temp_dir, f"replicate_{replicate+1}")
                makedirs(config['output_file_prefix'], exist_ok=True)

                # Run the template matching
                tmp_filename_match_template, tmp_filename_make_template_results = mktmp.make_tmp_runfile(config)
                elapsed_time[replicate] = runner.run_job(tmp_filename_match_template)
                runner.run_job(tmp_filename_make_template_results)

                # The MIP file is already in our temp directory with the unique replicate prefix
                mip_file = join(config['output_file_prefix'], 'mip.mrc')

                # Check if MIP file exists
                if not exists(mip_file):
                    raise FileNotFoundError(f"MIP file not found at {mip_file}")

                # Get the histogram file path and extract the threshold value
                hist_file = join(config['output_file_prefix'], 'hist.txt')
                threshold_value = extract_threshold_value(hist_file)

                mip_filenames.append(mip_file)
                hist_filenames.append(hist_file)
                threshold_values.append(threshold_value)

                # Print threshold value only for the first replicate
                if replicate == 0:
                    print(f"Threshold value: {threshold_value:.6e}")

                print(f"Completed replicate {replicate+1}, time: {elapsed_time[replicate]:.2f}s")

            except Exception as e:
                print(f"Error during replicate {replicate+1}: {str(e)}")
                continue

        # Check if we have at least 2 replicates
        if len(mip_filenames) < 2:
            print(f"Error: Only {len(mip_filenames)} replicates were successfully processed")
            print("At least 2 replicates are required for comparison. Exiting.")
            return 1

        # Now load the MIP files and compare them
        print("\nLoading MIP files for analysis...")
        mip_data = []
        for filename in mip_filenames:
            try:
                with mrcfile.open(filename) as mrc:
                    mip_data.append(mrc.data)
                    print(f"Loaded {filename}, shape: {mrc.data.shape}, dtype: {mrc.data.dtype}")
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")

        # Check if we have at least 2 MIP files loaded
        if len(mip_data) < 2:
            print("Failed to load at least 2 MIP files. Cannot perform comparison.")
            return 1

        # Verify all threshold values are the same
        if threshold_values:
            if not all(abs(v - threshold_values[0]) < 1e-10 for v in threshold_values):
                print("Warning: Threshold values differ between replicates:")
                for i, val in enumerate(threshold_values):
                    print(f"  Replicate {i+1}: {val:.6e}")

            # Use the first threshold value for calculations
            threshold_value = threshold_values[0]
        else:
            print("Error: No threshold values were extracted.")
            return 1

        # Compute similarity metrics between replicates
        print("\nReproducibility Analysis:")
        print("========================")
        print(f"Number of replicates analyzed: {len(mip_data)}")

        # Pairwise comparisons of available replicates
        pairs = [(i, j) for i in range(len(mip_data)) for j in range(i+1, len(mip_data))]

        all_correlations = []
        all_mean_abs_diffs = []

        for i, j in pairs:
            try:
                # Calculate correlation coefficient
                correlation = np.corrcoef(mip_data[i].flatten(), mip_data[j].flatten())[0,1]
                all_correlations.append(correlation)

                # Calculate mean absolute difference
                mean_abs_diff = np.mean(np.abs(mip_data[i] - mip_data[j]))
                all_mean_abs_diffs.append(mean_abs_diff)

                # Calculate relative error if threshold value is available
                if threshold_value and threshold_value > 0:
                    relative_error_ppm = (mean_abs_diff / threshold_value) * 1e6  # Parts per million
                else:
                    relative_error_ppm = None

                # Print results for this pair
                print(f"Comparing replicate {i+1} vs {j+1}:")
                print(f"  Correlation coefficient: {correlation:.6f}")
                print(f"  Mean absolute difference: {mean_abs_diff:.6f}")

                # Print relative error if available
                if relative_error_ppm is not None:
                    print(f"  Relative error: {relative_error_ppm:.2f} ppm (relative to threshold value: {threshold_value:.6e})")

            except Exception as e:
                print(f"Error comparing replicates {i+1} and {j+1}: {str(e)}")

        # Overall similarity across all replicates
        if all_correlations:
            print("\nOverall reproducibility:")
            print(f"  Mean correlation: {np.mean(all_correlations):.6f}")
            print(f"  Min correlation: {np.min(all_correlations):.6f}")
            print(f"  Max correlation: {np.max(all_correlations):.6f}")
            print(f"  Mean absolute diff (avg): {np.mean(all_mean_abs_diffs):.6f}")

            # Calculate average relative error if threshold is available
            if threshold_value and threshold_value > 0:
                mean_rel_error_ppm = (np.mean(all_mean_abs_diffs) / threshold_value) * 1e6
                print(f"  Relative error (avg): {mean_rel_error_ppm:.2f} ppm (relative to threshold value: {threshold_value:.6e})")

        # Print the directory where files are saved
        print(f"\nMIP files saved in: {temp_dir}")
        print("To list temp directories: --list-temp-dirs")
        print("To remove this directory: --rm-temp-dir INDEX")
        print("To remove all temp directories: --rm-all-temp-dirs")

        return 0

    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        return 130
    except Exception as e:
        print(f"Error in template matching reproducibility test: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


# Check if main function and run
if __name__ == '__main__':
    sys.exit(main())