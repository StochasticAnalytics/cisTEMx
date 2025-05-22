#!/usr/bin/env python3
import annoying_hack

from os.path import join
from os import makedirs
import cistem_test_utils.args as tmArgs
import cistem_test_utils.make_tmp_runfile as mktmp
import cistem_test_utils.run_job as runner
import mrcfile
import numpy as np
import tempfile
import os

# By default the "_gpu" suffix will be added unless the --old-cistem flag is used
# or the --cpu flag is used
wanted_binary_name = 'match_template'


def main():
    args = tmArgs.parse_TM_args(wanted_binary_name)

    # Create a temporary directory to store our replicate MIPs
    temp_dir = tempfile.mkdtemp(dir="/tmp", prefix="template_match_reproducibility_")
    print(f"Temporary directory created at: {temp_dir}")
    
    # We'll run template matching 3 times to generate replicates
    elapsed_time = [0, 0, 0]
    mip_filenames = []
    
    # Run the template matching 3 times
    for replicate in range(0, 3):
        # Use Apoferritin dataset with image 0
        config = tmArgs.get_config(args, 'Apoferritin', 0, 0)
        
        # Run the template matching
        tmp_filename_match_template, tmp_filename_make_template_results = mktmp.make_tmp_runfile(config)
        elapsed_time[replicate] = runner.run_job(tmp_filename_match_template)
        runner.run_job(tmp_filename_make_template_results)
        
        # Copy the MIP file to our temp directory with a unique name
        mip_src = join(config['output_file_prefix'], 'mip.mrc')
        mip_dst = join(temp_dir, f"mip_replicate_{replicate+1}.mrc")
        
        # Use os.system to copy the file - using cp preserves the MRC file format
        os.system(f"cp {mip_src} {mip_dst}")
        mip_filenames.append(mip_dst)
        
        print(f"Completed replicate {replicate+1}, time: {elapsed_time[replicate]:.2f}s")
    
    # Now load the MIP files and compare them
    mip_data = []
    for filename in mip_filenames:
        with mrcfile.open(filename) as mrc:
            mip_data.append(mrc.data)
    
    # Compute similarity metrics between replicates
    print("\nReproducibility Analysis:")
    print("========================")
    
    # Pairwise comparisons
    pairs = [(0,1), (0,2), (1,2)]
    for i, j in pairs:
        # Calculate correlation coefficient
        correlation = np.corrcoef(mip_data[i].flatten(), mip_data[j].flatten())[0,1]
        
        # Calculate mean absolute difference
        mean_abs_diff = np.mean(np.abs(mip_data[i] - mip_data[j]))
        
        # Calculate peak signal to noise ratio (PSNR)
        mse = np.mean((mip_data[i] - mip_data[j]) ** 2)
        max_pixel = max(np.max(mip_data[i]), np.max(mip_data[j]))
        psnr = 20 * np.log10(max_pixel) - 10 * np.log10(mse) if mse > 0 else float('inf')
        
        print(f"Comparing replicate {i+1} vs {j+1}:")
        print(f"  Correlation coefficient: {correlation:.6f}")
        print(f"  Mean absolute difference: {mean_abs_diff:.6f}")
        print(f"  Peak signal-to-noise ratio (PSNR): {psnr:.2f} dB")
    
    # Overall similarity across all three replicates
    all_correlations = []
    for i, j in pairs:
        corr = np.corrcoef(mip_data[i].flatten(), mip_data[j].flatten())[0,1]
        all_correlations.append(corr)
    
    print("\nOverall reproducibility:")
    print(f"  Mean correlation: {np.mean(all_correlations):.6f}")
    print(f"  Min correlation: {np.min(all_correlations):.6f}")
    print(f"  Max correlation: {np.max(all_correlations):.6f}")
    
    # Print the directory where files are saved
    print(f"\nMIP files saved in: {temp_dir}")


# Check if main function and run
if __name__ == '__main__':
    main()