#!/bin/bash

source params.sh

movie_file=$1

# TODO: when converting to python, use os.path.basename have a list of possible file extensions to source.
# For testing we'll assume eer, otherwise the movie inputs need to be changed as well.
img_base=$(basename $movie_file .eer)
mkdir -p $output_dir/images/$img_base

# Work out the image binning needed
image_binning=$(echo "print($output_pixel_size/$physical_pixel_size)" | python3)

# Set the termination to half an output pixel (expected in angstroms)
termination_criteria=$(echo "print($output_pixel_size/2)" | python3)

# APPTAINERENV_CUDA_VISIBLE_DEVICES=${gpu_for_movies} ${bin_cmd}/unblur_gpu << EOF &> /dev/null
# $movie_file
# $output_dir/images/$img_base/aligned_img.mrc
# $output_dir/images/$img_base/aligned_img_shifts.txt
# $physical_pixel_size
# $image_binning
# $apply_exposure_filter
# $microscope_voltage
# $exposure_per_frame
# $pre_exposure
# yes
# $movie_min_shift
# $movie_max_shift
# $movie_bfactor
# $movie_fourier_cross_mask_pixels
# $movie_fourier_cross_mask_pixels
# $termination_criteria
# $movie_max_iters
# yes
# yes
# no
# $movie_gain_ref
# $movie_first_frame_to_average
# $movie_last_frame_to_average
# $movie_running_average
# $movie_save_aligned_frames
# $n_eer_frames_to_average
# $eer_super_res_factor
# $movie_correct_mag_distortion
# $movie_max_threads
# EOF

${bin_cmd}/ctffind << EOF &> /dev/null
$output_dir/images/$img_base/aligned_img.mrc
$output_dir/images/$img_base/ctf_diagnostic.mrc
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

# bin the output diagnostic image
if [[ $ctf_box_size -ne $ctf_diagnostic_box_size ]] ; then

${bin_cmd}/resample << EOF &> /dev/null
$output_dir/images/$img_base/ctf_diagnostic.mrc
$output_dir/images/$img_base/ctf_diagnostic.mrc_rs
no
$ctf_diagnostic_box_size
$ctf_diagnostic_box_size
EOF

mv $output_dir/images/$img_base/ctf_diagnostic.mrc_rs $output_dir/images/$img_base/ctf_diagnostic.mrc

fi