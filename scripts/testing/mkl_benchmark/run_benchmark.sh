#!/bin/bash

# This benchmark is to ensure that the crippling of AMD hardware by Intel is subverted.
# The runtime is expected to be ~1.5x-2x slower on AMD after unsetting LD_PRELOAD

# Logical failings:
# There is no check that the hardware is actually AMD, so if the run times are nearly equivalent, it could just be due to the chip
# In the future, Intel could also stop being so shitty, and then the times would also appear nearly equivalent.
# In the future, Intel could also be MORE shitty and change the method name to check hardware at runtimer, and then the times would also appear nearly equivalent.

echo "Running the MKL benchmark"

echo "The value of LD_PRELOAD is ($LD_PRELOAD)"
echo "Not all image sizes show a large difference."

image_size=$1
n_images=$2
n_threads=$3

echo -e "${image_size}\n${n_images}\n${n_threads}\n" |   /cisTEMx/cistem_reference_images/mkl_benchmark/mkl_ld_preload_test


echo "Trying to unset LD_PRELOAD"
unset LD_PRELOAD
echo "The value of LD_PRELOAD is ($LD_PRELOAD) and should be empty parentheses"

echo -e "${image_size}\n${n_images}\n${n_threads}\n" |   /cisTEMx/cistem_reference_images/mkl_benchmark/mkl_ld_preload_test
