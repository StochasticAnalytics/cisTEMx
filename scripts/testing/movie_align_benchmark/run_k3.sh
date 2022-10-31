#!/bin/bash

data_dir=$1
n_to_align=$2
starting_idx=$3
number_of_frames=$4

file_name_list=($(ls ${data_dir}/HM70*.tif))

# TODO movie these to a param file
pixel_size=1.065
voltage=300
spherical_aberration=2.7
amplitude_contrast=0.07 
output_binning=1
alignment_threshold=$(echo "" | awk -v pixel_size=$pixel_size '{print 0.33*pixel_size}')
exposure_per_frame=0.92

for i in $( seq 0 $(($n_to_align-1)) ); do
    current_idx=$(( ($starting_idx+$i) % $n_to_align))

apptainer exec -B /scratch --nv ~/software/cisTEMx_production_1.0.0.sif ~/git/cisTEM/build/intel-gpu/src/unblur_gpu << EOF 2>&1 > /dev/null
${file_name_list[$current_idx]}
/dev/null
/dev/null
$pixel_size
$output_binning
yes
$voltage
$exposure_per_frame
0.0
yes
2.0
80.0
1200
1
1
$alignment_threshold
10
yes
yes
yes
1
$number_of_frames
1
no
no
1
EOF

# Drop everything for this file from cache
# echo "Dropping cache for ${file_name_list[$current_idx]}"
# free -m
dd of=${file_name_list[$current_idx]} oflag=nocache,sync conv=notrunc,fsync count=0
# free -m

done