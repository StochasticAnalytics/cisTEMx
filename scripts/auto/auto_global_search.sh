#!/bin/bash

source params.sh

dir_name=$1

defocus_1=$(tail -n -1 ${dir_name}/ctf_diagnostic.txt | awk '{print $2}')
defocus_2=$(tail -n -1 ${dir_name}/ctf_diagnostic.txt | awk '{print $3}')
defocus_ang=$(tail -n -1 ${dir_name}/ctf_diagnostic.txt | awk '{print $4}')


mkdir -p ${output_dir}/global_search
mkdir -p ${output_dir}/global_search/$(basename ${dir_name})

APPTAINERENV_CUDA_VISIBLE_DEVICES=${gpu_for_global} ${bin_cmd}/global_search_gpu << EOF 
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


mkdir -p ${output_dir}/particle_stacks
mkdir -p ${output_dir}/particle_stacks/$(basename ${dir_name})

wanted_threshold=$(awk '/Expected threshold/{print $5}' ${output_dir}/global_search/$(basename ${dir_name})/aligned_img_histogram.txt)
echo "Wanted threshold is  - $wanted_threshold - "
echo "amp $microscope_amplitude_contrast"
result_number=1

APPTAINERENV_CUDA_VISIBLE_DEVICES=${gpu_for_global} ${bin_cmd}/prepare_stack_global_search << EOF 
no
${output_dir}/global_search/$(basename ${dir_name})/aligned_img_scaled_mip.mrc
${output_dir}/global_search/$(basename ${dir_name})/aligned_img_psi.mrc
${output_dir}/global_search/$(basename ${dir_name})/aligned_img_theta.mrc
${output_dir}/global_search/$(basename ${dir_name})/aligned_img_phi.mrc
/dev/null
/dev/null
${wanted_threshold}
${global_min_peak_radius}
$result_number
${dir_name}/aligned_img.mrc
${output_dir}/particle_stacks/$(basename ${dir_name})/particle_stack.star
${output_dir}/particle_stacks/$(basename ${dir_name})/particle_stack.mrc
${sim_output_size}
$output_pixel_size
$defocus_1
$defocus_2
$defocus_ang
$microscope_voltage
$microscope_spherical_aberration
$microscope_amplitude_contrast
EOF

#FIXME
# Need a check on zero peaks (or even < N peaks found)
./auto_local_and_ctf.sh ${dir_name}