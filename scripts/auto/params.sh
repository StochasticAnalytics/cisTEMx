#!/bin/bash

output_dir=no_bp_12.5

bin_cmd="apptainer exec --nv -B /scratch/,/sa_shared/software /sa_shared/software/cisTEMx_production_1.0.5.sif /sa_shared/software/local_cistem/intel-gpu-debug-profile/src"
pdb_file=7a4m_assembly_no_C_T.pdb


# movie_dir=/scratch/salina/proc_EMPIAR-10568-apoferritin-g4-EER/setup_Movie2Map/movies
movie_dir=/scratch/etna/test_auto/apo_tests/movies
use_movie_gain_ref="yes"
movie_gain_ref=/scratch/salina/proc_EMPIAR-10568-apoferritin-g4-EER/setup_Movie2Map/gain_1xUps_8k.mrc

total_exposure=52.7
physical_pixel_size=0.3948
output_pixel_size=0.8
n_eer_frames=720
n_eer_frames_to_average=12
eer_super_res_factor=1
exposure_per_frame=$(echo "print($total_exposure/($n_eer_frames/$n_eer_frames_to_average))" | python3)
pre_exposure=0.0
apply_exposure_filter=yes

# Microscope parameters
microscope_voltage=300
microscope_spherical_aberration=2.7
microscope_amplitude_contrast=0.1


########### TODO add below to validation checks

# Simulation parameters
sim_output_size=256
sim_max_threads=4
sim_linear_scaling_of_pdb_bfactors=1.0
sim_base_bfactor=15.0
sim_total_exposure=$total_exposure
# should be returned from the call to simulate or otherwise set based on PDB info
sim_particle_mass=460

# CTF fitting parameters
ctf_box_size=768
ctf_diagnostic_box_size=$(( $ctf_box_size / 2 ))
ctf_min_res=30
ctf_max_res=2.8
ctf_min_defocus=500
ctf_max_defocus=25000
ctf_defocus_step=25
ctf_tolerated_astigmatism=1000
ctf_restrain_astigmatism=yes
ctf_exhaustive_search=no
ctf_find_extra_phase_shift=no
ctf_find_tilt=no
ctf_max_threads=1 # maybe set this based on find_tilt?

# Movie align parameters
movie_min_shift=2.0
movie_max_shift=80.0
movie_bfactor=1200
movie_fourier_cross_mask_pixels=1
movie_max_iters=10
movie_first_frame_to_average=1
movie_last_frame_to_average=0
movie_running_average=1
movie_save_aligned_frames=no
movie_correct_mag_distortion=no
movie_max_threads=1

global_phase_shift=0.0
global_out_of_plane_angle=12.5
global_in_plane_angle=6.25
global_low_resolution_limit=60.0
global_high_resolution_limit=0.0
global_padding_value=1.0
global_mask_radius=0.0
global_symmetry="O"
global_max_threads=1
# TODO, this should be tied to particle size
global_min_peak_radius=65.0

global_n_peaks=6

# Run with the input model# zero for abberation refinement (max res)
global_resolution=7.5
local_resolution=(8 6.5 5 3.5 0 2.8)
local_angle_step=(4.2 3.25 2.5 1.66 1 0.5)
n_local_iterations=${#local_resolution[@]}
local_defocus_range_angstroms=250
local_defocus_step_angstroms=25

# Run with the reconstruction from the first search
final_resolution=2.0

# Recosntruction parameters
reconstruct3d_max_threads=8

# To use for load balancing
gpu_for_movies=0
max_movies_per_gpu=1

gpu_for_global=0
max_images_per_gpu=3

gpu_for_local=0
max_stacks_per_gpu=3
local_max_threads=8

# lock files for coordination 


# jsut for testing
run_simulate=no
run_movie_align=no
run_ctf_fit=no
run_global=no
run_global_results=no
run_grid=yes
run_local=no
run_reconstruct3d=no

# timing files to append to, will be zeroed at the beginning of a movie run
simulate_timing_file=$output_dir/log/simulate_timing.txt
movie_align_timing_file=$output_dir/log/movie_align_timing.txt
ctf_fit_timing_file=$output_dir/log/ctf_fit_timing.txt
global_search_timing_file=$output_dir/log/global_timing.txt
grid_timing_file=$output_dir/log/grid_timing.txt
local_timing_file=$output_dir/log/local_timing.txt
global_prepare_stack_timing_file=$output_dir/log/global_prepare_stack_timing.txt
total_timing_file=$output_dir/log/total_timing.txt


# some helper functions
source helper_functions.sh
