#!/bin/bash


source params.sh

dir_name=$1


for iteration in ${!local_resolution[@]} ; do

get_start
refine_defocus=$(echo "" | awk -v R="${local_resolution[$iteration]}" '{if(R < 0.001) print "yes"; else print "no"}')
if [[ $refine_defocus == "yes" ]] ; then
    high_res_limit_ang=2.8
else
    high_res_limit_ang=${local_resolution[$iteration]}
fi


wanted_particle_mass=$sim_particle_mass

# TODO: Play around with the masking
inner_mask_radius_ang=0
outer_mask_radius_ang=65 # TODO set this based on particle size
low_res_limit_ang=300 # TODO set this based on particle size
search_range_x=$(echo "print(0.5*$outer_mask_radius_ang)" | python3)
search_range_y=$search_range_x

out_of_plane_angle_step=${local_angle_step[$iteration]}
out_of_plane_angle_step=${out_of_plane_angle_step:-0.5}
in_plane_angle_step=$(echo "print(${local_angle_step[$iteration]}/2)" | python3)

for i_result in $(seq 1 $global_n_peaks) ; do

if [[ $iteration -eq 0 ]] ; then
    input_starfile=${output_dir}/particle_stacks/$(basename ${dir_name})/micrograph_${i_result}.star
else
    input_starfile=${output_dir}/particle_stacks/$(basename ${dir_name})/refined_parameters_$((${iteration}-1))_${i_result}.star
fi

if [[ $run_grid == "yes" ]]; then
    # for focused refinement
APPTAINERENV_CUDA_VISIBLE_DEVICES=${gpu_for_global} ${bin_cmd}/global_search_refinement_gpu  << EOF 
${dir_name}/aligned_img.mrc
${output_dir}/volumes/${pdb_file}.mrc
$input_starfile
${output_dir}/particle_stacks/$(basename ${dir_name})/refined_parameters_${iteration}_${i_result}.star
${output_dir}/particle_stacks/
$low_res_limit_ang
$high_res_limit_ang
$out_of_plane_angle_step
$in_plane_angle_step
$refine_defocus
$local_defocus_range_angstroms
$local_defocus_step_angstroms
2
$outer_mask_radius_ang
$local_max_threads
EOF
check_exit_status "global_grid_refinement_gpu" $output_dir
get_stop
add_time_to_file $grid_timing_file


fi

if [[ $? -ne 0 ]] ; then
    echo "global_search_refinement_gpu failed"
    exit 1
fi

done # loop on results
done # loop on iterations

until [[ $(pgrep prepare_stack_global_search | wc | awk '{print $1}' ) -le 4 ]] ; do
    sleep 5
done

get_start
# At this point, we need to switch from micrograph to particle stack
APPTAINERENV_CUDA_VISIBLE_DEVICES=${gpu_for_global} ${bin_cmd}/prepare_stack_global_search << EOF 
yes
${output_dir}/particle_stacks/$(basename ${dir_name})/refined_parameters_${iteration}.star
${dir_name}/aligned_img.mrc
${output_dir}/particle_stacks/$(basename ${dir_name})/refined_parameters_${iteration}_stack.star
${output_dir}/particle_stacks/$(basename ${dir_name})/refined_parameters_${iteration}_stack.mrc
${sim_output_size}
EOF
check_exit_status "prepare_stack_grid_search" $output_dir
get_stop
add_time_to_file $global_prepare_stack_timing_file

exit
./auto_local_and_ctf.sh ${dir_name} ${output_dir}/particle_stacks/$(basename ${dir_name})/refined_parameters_${iteration}.star