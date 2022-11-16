#!/bin/bash

echo "Validating inputs: "
# Some helper functions to validate input parameters


function to_lower() {
    echo "${1,,}"
}

function in_range() {
    [ $# -ne 3 ] && echo "ERROR: in_range() requires 3 arguments" && exit 1
    echo "" | awk -v min=$1 -v max=$2 -v val=$3  '{if(val >= min && val <= max) print "true"; else print "false"}' 
}
function check_yes_or_no {
    local_var=$1;
    # convert to lower case and check for yes or no
    if [[ `to_lower $1` != "yes" &&  `to_lower $1` != "no" ]] ; then
        echo "ERROR: $1 must be yes or no"
        exit 1
    fi
}

# All the scripts depend on the params file, check that it exists
[ -f params.sh ] || { echo "params.sh not found, exiting" ; exit 1 ; }

# Okay it exists, source it
source params.sh

mkdir -p $output_dir
if [[ $? -ne 0 ]] ; then
    echo "ERROR: Could not create output directory $output_dir"
    exit 1
fi

# Check that the data directory exists
[ -d $movie_dir ] || { echo "Data directory $movie_dir not found, exiting" ; exit 1 ; }

# Now check the data directory is readable
[ -r $movie_dir ] || { echo "movie_dir not readable, exiting" ; exit 1 ; }

check_yes_or_no "$use_movie_gain_ref"
if [ `to_lower $movie_gain_ref` == "yes" ] ; then
    # Let's confirm the gain reference exists
    [ -f $movie_gain_ref ] || { echo "Gain reference $movie_gain_ref not found, exiting" ; exit 1 ; }
fi


# Check that the atomic coordinates file exists
[ -f $pdb_file ] || { echo "Atomic coordinates file $pdb_file not found, exiting" ; exit 1 ; }


[[ `in_range 0 1000 $total_exposure` == "true" ]] || { echo "total exposure is out of the acceptable range, 0 1000" ; exit 1 ; }
[[ `in_range 0 10 $physical_pixel_size` == "true" ]] || { echo "physical pixels size is out of the acceptable range, 0 10" ; exit 1;}
[[ `in_range $physical_pixel_size 10 $output_pixel_size` == "true" ]] || { echo "output pixel size  is out of the acceptable range, physical pixels size 10" ; exit 1;}
[[ `in_range 0 100000 $n_eer_frames` == "true" ]] || { echo "n_eer_frames is out of the acceptable range, 0 100000" ; exit 1;}
[[ $(($n_eer_frames % $n_eer_frames_to_average)) -eq 0 ]] || { echo "n_eer_frames_to_average ($n_eer_frames_to_average) is not a multiple of n_eer_frames ($n_eer_frames)" ; exit 1;}
[[ $eer_super_res_factor -eq 1 || $eer_super_res_factor -eq 2 || $eer_super_res_factor -eq 4 ]] || { echo "eer_super_res_factor is out of the acceptable range, 1, 2, 4" ; exit 1;}
[[ `in_range 0 10 $exposure_per_frame` == "true" ]] || { echo "exposure_per_frame is out of the acceptable range, 0 10" ; exit 1;}
[[ `in_range 0 100 $pre_exposure` == "true" ]] || { echo "pre_exposure is out of the acceptable range, 0 100" ; exit 1;}   

check_yes_or_no apply_exposure_filter

# Microscope parameters

[[ `in_range 0.5 4.0 $microscope_spherical_aberration` ]] || { echo "microscope_spherical_aberration is out of the acceptable range, 0.5 4.0" ; exit 1;}
[[ `in_range 0.0 1.0 $microscope_amplitude_contrast` ]] || { echo "microscope_amplitude_contrast is out of the acceptable range, 0.0 1.0" ; exit 1;}
[[ `in_range 100 500 $microscope_voltage` ]] || { echo "microscope_voltage is out of the acceptable range, 100 500" ; exit 1;}





