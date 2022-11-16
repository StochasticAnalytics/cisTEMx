#!/bin/bash


source params.sh

dir_name=$1

for iteration in ${!local_resolution[@]} ; do

if [[ $iteration -eq 0 ]] ; then
    input_starfile=${output_dir}/particle_stacks/$(basename ${dir_name})/particle_stack.star
else
    input_starfile=${output_dir}/particle_stacks/$(basename ${dir_name})/refined_parameters_$((${iteration}-1)).star
fi


wanted_particle_mass=$sim_particle_mass

# TODO: Play around with the masking
inner_mask_radius_ang=0
outer_mask_radius_ang=65 # TODO set this based on particle size
low_res_limit_ang=300 # TODO set this based on particle size
search_range_x=$(echo "print(0.5*$outer_mask_radius_ang)" | python3)
search_range_y=$search_range_x

high_res_limit_ang=${local_resolution[$iteration]}
out_of_plane_angle_step=${local_angle_step[$iteration]}
in_plane_angle_step=$(echo "print(${local_angle_step[$iteration]}/2)" | python3)

if [[ $run_grid == "yes" ]]; then
    # for focused refinement
APPTAINERENV_CUDA_VISIBLE_DEVICES=${gpu_for_global} ${bin_cmd}/global_search_refinement_gpu  << EOF 
${dir_name}/aligned_img.mrc
${output_dir}/volumes/${pdb_file}.mrc
$input_starfile
${output_dir}/particle_stacks/$(basename ${dir_name})/refined_parameters_${iteration}.star
${output_dir}/particle_stacks/
$low_res_limit_ang
$high_res_limit_ang
$out_of_plane_angle_step
$in_plane_angle_step
no
200
50
2
$outer_mask_radius_ang
8
EOF

fi

done

./auto_local_and_ctf.sh ${dir_name} ${output_dir}/particle_stacks/$(basename ${dir_name})/refined_parameters_${iteration}.star