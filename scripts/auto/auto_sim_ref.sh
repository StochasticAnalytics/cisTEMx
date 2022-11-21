#!/bin/bash

source params.sh


# no GPU yet for simualtor
${bin_cmd}/simulate << EOF 
${output_dir}/volumes/${pdb_file}.mrc
yes
$sim_output_size
$sim_max_threads
$pdb_file
no
$output_pixel_size
$sim_linear_scaling_of_pdb_bfactors
$sim_base_bfactor
1.0
$sim_total_exposure
no
EOF
