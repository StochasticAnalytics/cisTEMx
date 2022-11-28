#!/bin/bash

dir_name=$1
gpu_id=$2

source params.sh

if [[ $run_ctf_fit == "yes" ]] ; then
get_start
${bin_cmd}/ctffind << EOF 
${dir_name}/aligned_img.mrc
${dir_name}/ctf_diagnostic.mrc
$output_pixel_size
$microscope_voltage
$microscope_spherical_aberration
$microscope_amplitude_contrast
$ctf_box_size
$ctf_min_res
$ctf_max_res
$ctf_min_defocus
$ctf_max_defocus
$ctf_defocus_step
no
$ctf_exhaustive_search
yes
$ctf_tolerated_astigmatism
$ctf_find_extra_phase_shift
$ctf_find_tilt
yes
yes
no
$ctf_max_threads
EOF
check_exit_status "CTF fitting" $output_dir
get_stop
add_time_to_file $ctf_fit_timing_file



# bin the output diagnostic image
if [[ $ctf_box_size -ne $ctf_diagnostic_box_size ]] ; then

${bin_cmd}/resample << EOF &> /dev/null
${dir_name}/ctf_diagnostic.mrc
${dir_name}/ctf_diagnostic.mrc_rs
no
$ctf_diagnostic_box_size
$ctf_diagnostic_box_size
EOF

mv ${dir_name}/ctf_diagnostic.mrc_rs ${dir_name}/ctf_diagnostic.mrc

fi

fi
./auto_global_search.sh $dir_name $gpu_id
