#!/bin/bash

source params.sh

[[ $# -ne 2 && $# -ne 4 ]] && { echo "Usage: <input_images> <input_starfile> reconstruction_id score_cutoff" ; exit 1; }

if [[ $# -eq 4 ]]; then
    # This seems to depend on the resolution so is not good to use yet.
    score_cutoff=$4
    # TODO: get a reconstruction id an use here also including number of particles
    reconstruction_id=$3
else
    # This seems to depend on the resolution so is not good to use yet.
    score_cutoff=1
    reconstruction_id=1
    # TODO: get a reconstruction id an use here also including number of particles
fi


first_particle=1
last_particle=0

# no GPU yet for reconstruct3d
input_images=$1
input_starfile=$2

# TODO: Play around with the masking
inner_mask_radius_ang=0
outer_mask_radius_ang=65 # TODO set this based on particle size
low_res_limit_ang=300 # TODO set this based on particle size

#TODO do these make any real difference, default 5
particle_weight_factor=5.0
smoothing_factor=1
padding_factor=2.0
normalize_particles='yes'
adjust_scores_for_defocus='yes' # TODO: this makes sense if properly whitented, but I bet it is a bullshit heuristic.
exclude_blank_edges='yes'


# TODO: we will want this to be configurable so that it can be no when accumulating volumes for bfactor calculations.
calculate_fsc='yes'

center_of_mass='no'
likelihood_blurring='no'
threshold_input_reconstruction='no'
input_reconstruction="dummy"

#
${bin_cmd}/reconstruct3d << EOF
${input_images}
${input_starfile}
$input_reconstruction
${output_dir}/volumes/reconstruction_${reconstruction_id}_1.mrc
${output_dir}/volumes/reconstruction_${reconstruction_id}_2.mrc
${output_dir}/volumes/reconstruction_${reconstruction_id}_filtered.mrc
${output_dir}/volumes/reconstruction_${reconstruction_id}_stats.txt
$global_symmetry
$first_particle
$last_particle
$output_pixel_size 
$sim_particle_mass 
$inner_mask_radius_ang
$outer_mask_radius_ang
0 
0
$particle_weight_factor
$score_cutoff
$smoothing_factor
$padding_factor
$normalize_particles
$adjust_scores_for_defocus
no
$exclude_blank_edges
no
$calculate_fsc
$center_of_mass
$likelihood_blurring
$threshold_input_reconstruction
no
dump1.dat
dump2.dat
$reconstruct3d_max_threads
EOF