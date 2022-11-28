#!/bin/bash

source params.sh

dir_name=$1
gpu_id=$2

defocus_1=$(tail -n -1 ${dir_name}/ctf_diagnostic.txt | awk '{print $2}')
defocus_2=$(tail -n -1 ${dir_name}/ctf_diagnostic.txt | awk '{print $3}')
defocus_ang=$(tail -n -1 ${dir_name}/ctf_diagnostic.txt | awk '{print $4}')

if [[ $run_global == "yes" ]] ; then
get_start
mkdir -p ${output_dir}/global_search
mkdir -p ${output_dir}/global_search/$(basename ${dir_name})


# {
# echo "${dir_name}/aligned_img.mrc"
# echo "${output_dir}/volumes/${pdb_file}.mrc"
# echo "${output_dir}/global_search/$(basename ${dir_name})"
# echo "$output_pixel_size"
# echo "$microscope_voltage"
# echo "$microscope_spherical_aberration"
# echo "$microscope_amplitude_contrast"
# echo "$defocus_1"
# echo "$defocus_2"
# echo "$defocus_ang"
# echo "$global_phase_shift"
# echo "$global_high_resolution_limit"
# echo "$global_out_of_plane_angle"
# echo "$global_in_plane_angle"
# echo "$global_padding_value"
# echo "$global_mask_radius"
# echo "$global_symmetry"
# echo "$global_min_peak_radius"
# echo "$global_max_threads"
# } > .global.dff
echo "APPTAINERENV_CUDA_VISIBLE_DEVICES=${gpu_id} ${bin_cmd}/global_search_gpu"
APPTAINERENV_CUDA_VISIBLE_DEVICES=${gpu_id} ${bin_cmd}/global_search_gpu << EOF 
${dir_name}/aligned_img.mrc
${output_dir}/volumes/${pdb_file}.mrc
${output_dir}/global_search/$(basename ${dir_name})
$output_pixel_size
$microscope_voltage
$microscope_spherical_aberration
$microscope_amplitude_contrast
$defocus_1
$defocus_2
$defocus_ang
$global_phase_shift
$global_high_resolution_limit
$global_out_of_plane_angle
$global_in_plane_angle
$global_padding_value
$global_mask_radius
$global_symmetry
$global_min_peak_radius
$global_max_threads
EOF
check_exit_status "global_search_gpu" $output_dir
get_stop
add_time_to_file $global_search_timing_file


fi

./auto_prepare_global_results.sh ${dir_name} ${gpu_id}

