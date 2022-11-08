#!/bin/bash


source params.sh

dir_name=$1

# These are intended for diagnostics
save_matching_projections="no"
matching_projections_name="dummy"
# Will not exist until we do a reconstruction 
output_statistics="my_statistics.txt"
use_output_statistics="no"


# Generally we'll do all the particles
first_particle=0
last_particle=0

# TODO: this should come from some output after simulateion
wanted_particle_mass=$sim_particle_mass

# TODO: Play around with the masking
inner_mask_radius_ang=0
outer_mask_radius_ang=65 # TODO set this based on particle size
low_res_limit_ang=300 # TODO set this based on particle size
search_range_x=$(echo "print(0.5*$outer_mask_radius_ang)" | python3)
search_range_y=$search_range_x

# for focused refinement
# TODO: take from params.sh
_2Dfocused_mask_x=0
_2Dfocused_mask_y=0
_2Dfocused_mask_z=0
_2Dfocused_mask_radius=$outer_mask_radius_ang
use_2Dmask="no"


# TODO: we want this to be automatically selected for now, but I think the in-plane search is excessive
# At least add a report back of what is searched for deferred threshold calculations
angular_search_step=0.0

# TODO: see if this matters for just local - may be needed if we end up saving multiple global searches
n_top_hits_to_refine=10

# TODO: see if this makes any measurable difference
padding_factor=1.0

# set high res based on iteration
# set defocus based on iteration

normalize_particles="yes"
exclude_blank_edges="no"
normalize_input_reconstruction="yes"
threshold_input_reconstruction="no"

for iteration in 1 2 3 ; do

if [[ $iteration -eq 2 ]] ; then
    refine_defocus="yes"

    refine_psi="no"
    refine_theta="no"
    refine_phi="no"
    refine_x="yes"
    refine_y="yes"
else    
    refine_defocus="no"

    refine_psi="yes"
    refine_theta="yes"
    refine_phi="yes"
    refine_x="yes"
    refine_y="yes"
fi

case $iteration in
    1)
        high_res_limit_ang=$local_resolution_1
        input_starfile=${output_dir}/particle_stacks/$(basename ${dir_name})/particle_stack.star

        ;;
    2)
        high_res_limit_ang=$local_resolution_2
        input_starfile=${output_dir}/particle_stacks/$(basename ${dir_name})/refined_parameters_$((${iteration}-1)).star
        ;;
    3)
        high_res_limit_ang=$local_resolution_3
        input_starfile=${output_dir}/particle_stacks/$(basename ${dir_name})/refined_parameters_$((${iteration}-1)).star
        ;;
    *)
        echo "ERROR: iteration $iteration not supported"
        exit 1
        ;;
esac

high_res_limit_signed_cc=$high_res_limit_ang



APPTAINERENV_CUDA_VISIBLE_DEVICES=${gpu_for_local} ${bin_cmd}/refine3d_gpu << EOF 
${output_dir}/particle_stacks/$(basename ${dir_name})/particle_stack.mrc
$input_starfile
${output_dir}/volumes/${pdb_file}.mrc
$output_statistics
$use_output_statistics
$save_matching_projections
${output_dir}/particle_stacks/$(basename ${dir_name})/refined_parameters_${iteration}.star
${output_dir}/particle_stacks/$(basename ${dir_name})/parameter_changes_${iteration}.star
$global_symmetry
$first_particle
$last_particle
1.0
$output_pixel_size
$wanted_particle_mass
$inner_mask_radius_ang
$outer_mask_radius_ang
$low_res_limit_ang
$high_res_limit_ang
$high_res_limit_signed_cc
$high_res_limit_ang
$outer_mask_radius_ang
$high_res_limit_ang
$angular_search_step
$n_top_hits_to_refine
$search_range_x
$search_range_y
${_2Dfocused_mask_x}
${_2Dfocused_mask_y}
${_2Dfocused_mask_z}
${_2Dfocused_mask_radius}
$local_defocus_range_angstroms
$local_defocus_step_angstroms
$padding_factor
no
yes
$refine_psi
$refine_theta
$refine_phi
$refine_x
$refine_y
$save_matching_projections
$use_2Dmask
$refine_defocus
$normalize_particles
no
$exclude_blank_edges
$normalize_input_reconstruction
$threshold_input_reconstruction
$local_max_threads
EOF
done


