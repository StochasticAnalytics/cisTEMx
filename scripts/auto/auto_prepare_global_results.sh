#!/bin/bash

source params.sh

dir_name=$1
gpu_id=$2

defocus_1=$(tail -n -1 ${dir_name}/ctf_diagnostic.txt | awk '{print $2}')
defocus_2=$(tail -n -1 ${dir_name}/ctf_diagnostic.txt | awk '{print $3}')
defocus_ang=$(tail -n -1 ${dir_name}/ctf_diagnostic.txt | awk '{print $4}')

mkdir -p ${output_dir}/particle_stacks
mkdir -p ${output_dir}/particle_stacks/$(basename ${dir_name})

wanted_threshold=$(awk '/Expected threshold/{print $5}' ${output_dir}/global_search/$(basename ${dir_name})/aligned_img_histogram.txt)
echo "Wanted threshold is  - $wanted_threshold - "
echo "amp $microscope_amplitude_contrast"
result_number=1

# ${output_dir}/global_search/$(basename ${dir_name})/aligned_img_avg.mrc
#${output_dir}/global_search/$(basename ${dir_name})/aligned_img_std.mrc
if [[ $run_global_results == "yes" ]] ; then
for i_result in $(seq 1 $global_n_peaks) ; do
get_start
# We are just making a star file to work with micrographs at this piont
APPTAINERENV_CUDA_VISIBLE_DEVICES=${gpu_id} ${bin_cmd}/prepare_stack_global_search << EOF 
no
${output_dir}/global_search/$(basename ${dir_name})/aligned_img_mip.mrc
${output_dir}/global_search/$(basename ${dir_name})/aligned_img_psi.mrc
${output_dir}/global_search/$(basename ${dir_name})/aligned_img_theta.mrc
${output_dir}/global_search/$(basename ${dir_name})/aligned_img_phi.mrc
/dev/null
/dev/null
${wanted_threshold}
${global_min_peak_radius}
$i_result
${dir_name}/aligned_img.mrc
${output_dir}/particle_stacks/$(basename ${dir_name})/micrograph_${i_result}.star
${output_dir}/particle_stacks/$(basename ${dir_name})/micrograph_${i_result}.mrc
${sim_output_size}
$output_pixel_size
$defocus_1
$defocus_2
$defocus_ang
$microscope_voltage
$microscope_spherical_aberration
$microscope_amplitude_contrast
EOF
check_exit_status "prepare_stack_global_search" $output_dir
get_stop
add_time_to_file $global_prepare_stack_timing_file
done
fi

#FIXME
# Need a check on zero peaks (or even < N peaks found)
./auto_grid_refinement.sh ${dir_name} $gpu_id