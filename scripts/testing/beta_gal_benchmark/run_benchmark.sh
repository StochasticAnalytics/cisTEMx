#!/bin/bash


expected_global_resolution=3.15
expected_local_resolution=2.68

expected_global_runtime=140
expected_local_runtime=63
expected_recon_runtime=30
max_run_time=$(($expected_global_runtime + $expected_local_runtime + $expected_recon_runtime))

data_dir=/cisTEMx/cistem_reference_images/betagal_benchmark

# Run as
# apptainer run --nv current_runtime_version.sif bgal

# Check for the required files, expected to be in the working directory
if [ ! -f ${data_dir}/bgal_stack.mrc ]; then
    echo "${data_dir}/bgal_stack.mrc not found"
    exit 1
fi

if [ ! -f ${data_dir}/bgal_ref.mrc ]; then
    echo "${data_dir}/bgal_ref.mrc not found"
    exit 1
fi

if [ ! -f ${data_dir}/bgal.star ]; then
    echo "${data_dir}/bgal.star not found"
    exit 1
fi

echo "Running beta-gal benchmark"
echo -e "---------------------------------------------------\n"
echo "Expected run time is ~ $((($max_run_time/60))) min $((($max_run_time%60))) sec"
echo -e "---------------------------------------------------\n"
sleep 5

# So we can run locally or in the container
run_dir=$(dirname $0)

# Run the global search
${run_dir}/run_global.sh ${data_dir} $expected_global_runtime

# Check the resolution
${run_dir}/run_global_recon.sh ${data_dir} $expected_global_resolution

# Run the local search
${run_dir}/run_local_refinement.sh ${data_dir} $expected_local_runtime

# Check the resolution
${run_dir}/run_local_recon.sh ${data_dir} $expected_local_resolution


echo -e "---------------------------------------------------\n\n"
grep Result: /tmp/bgal_benchmark.log
echo -e "\n---------------------------------------------------\n"