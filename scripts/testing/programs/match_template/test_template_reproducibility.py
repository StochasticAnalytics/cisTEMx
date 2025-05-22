#!/usr/bin/env python3
"""
Template Matching Reproducibility Test

This script runs the cisTEM template matching GPU binary three times using the same input data
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
import mrcfile
import numpy as np
import tempfile
import os
import shutil
import sys

# By default the "_gpu" suffix will be added unless the --old-cistem flag is used
# or the --cpu flag is used
wanted_binary_name = 'match_template'


def main():
    try:
        # Parse command-line arguments
        args = tmArgs.parse_TM_args(wanted_binary_name)

        # Create a temporary directory to store our replicate MIPs
        temp_dir = tempfile.mkdtemp(dir="/tmp", prefix="template_match_reproducibility_")
        print(f"Temporary directory created at: {temp_dir}")
        
        # We'll run template matching 3 times to generate replicates
        elapsed_time = [0, 0, 0]
        mip_filenames = []
        
        # Run the template matching 3 times
        for replicate in range(0, 3):
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
                
                mip_filenames.append(mip_file)
                print(f"Completed replicate {replicate+1}, time: {elapsed_time[replicate]:.2f}s")
                
            except Exception as e:
                print(f"Error during replicate {replicate+1}: {str(e)}")
                continue
        
        # Check if we have all three replicates
        if len(mip_filenames) != 3:
            print(f"Warning: Only {len(mip_filenames)} replicates were successfully processed")
            if len(mip_filenames) < 2:
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
        
        # Compute similarity metrics between replicates
        print("\nReproducibility Analysis:")
        print("========================")
        
        # Pairwise comparisons of available replicates
        pairs = [(i, j) for i in range(len(mip_data)) for j in range(i+1, len(mip_data))]
        
        all_correlations = []
        all_mean_abs_diffs = []
        all_psnrs = []
        
        for i, j in pairs:
            try:
                # Calculate correlation coefficient
                correlation = np.corrcoef(mip_data[i].flatten(), mip_data[j].flatten())[0,1]
                all_correlations.append(correlation)
                
                # Calculate mean absolute difference
                mean_abs_diff = np.mean(np.abs(mip_data[i] - mip_data[j]))
                all_mean_abs_diffs.append(mean_abs_diff)
                
                # Calculate peak signal to noise ratio (PSNR)
                mse = np.mean((mip_data[i] - mip_data[j]) ** 2)
                max_pixel = max(np.max(mip_data[i]), np.max(mip_data[j]))
                psnr = 20 * np.log10(max_pixel) - 10 * np.log10(mse) if mse > 0 else float('inf')
                all_psnrs.append(psnr)
                
                # Print results for this pair
                print(f"Comparing replicate {i+1} vs {j+1}:")
                print(f"  Correlation coefficient: {correlation:.6f}")
                print(f"  Mean absolute difference: {mean_abs_diff:.6f}")
                print(f"  Peak signal-to-noise ratio (PSNR): {psnr:.2f} dB")
                
            except Exception as e:
                print(f"Error comparing replicates {i+1} and {j+1}: {str(e)}")
        
        # Overall similarity across all replicates
        if all_correlations:
            print("\nOverall reproducibility:")
            print(f"  Mean correlation: {np.mean(all_correlations):.6f}")
            print(f"  Min correlation: {np.min(all_correlations):.6f}")
            print(f"  Max correlation: {np.max(all_correlations):.6f}")
            print(f"  Mean absolute diff (avg): {np.mean(all_mean_abs_diffs):.6f}")
            print(f"  PSNR (avg): {np.mean(all_psnrs):.2f} dB")
        
        # Print the directory where files are saved
        print(f"\nMIP files saved in: {temp_dir}")
        
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