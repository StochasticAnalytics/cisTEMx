#!/bin/bash

output_dir=test_1

bin_cmd="apptainer exec --nv -B /scratch/ /sa_shared/software/cisTEMx_production_1.0.5.sif ${HOME}/git/cisTEM/build/intel-gpu/src"
pdb_file=7a4m_assembly_no_C_T.pdb

movie_dir=/scratch/salina/proc_EMPIAR-10568-apoferritin-g4-EER/setup_Movie2Map/movies
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

# Simulation parameters
sim_output_size=480
sim_max_threads=4
sim_linear_scaling_of_pdb_bfactors=1.0
sim_base_bfactor=15.0
sim_total_exposure=$total_exposure


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

# Run with the input model
global_resolution=7.5
local_resolution_1=5.0
ctf_resolution=3.0

# Run with the reconstruction from the first search
final_resolution=3.5

# To use for load balancing
gpu_for_movies=0
max_movies_per_gpu=3

gpu_for_global=1
max_images_per_gpu=6

gpu_for_local=2
max_stacks_per_gpu=6




