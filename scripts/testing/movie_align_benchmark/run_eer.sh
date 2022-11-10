#!/bin/bash

data_dir=$1
n_threads=$2
n_to_align=$3
starting_idx=$4
n_eer_frames_to_avg=$5

file_name_list=($(ls ${data_dir}/*.eer))

# TODO movie these to a param file
# pixel_size=0.197
pixel_size=0.3948
eer_super_res_factor=1

voltage=300
spherical_aberration=2.7
amplitude_contrast=0.07 
output_binning=1
alignment_threshold=$(echo "" | awk -v pixel_size=$pixel_size '{print 0.33*pixel_size}')

exposure_per_frame=$(echo "" | awk -v n_eer_frames_to_avg=$n_eer_frames_to_avg '{print 0.075*n_eer_frames_to_avg}')

for i in $( seq 0 $(($n_to_align-1)) ); do
    current_idx=$(( ($starting_idx+$i) % $n_to_align))

echo "Process $starting_idx aligning ${file_name_list[$current_idx]}"
/cisTEMx/bin/unblur_gpu << EOF 
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
0
1
no
$n_eer_frames_to_avg
$eer_super_res_factor
no
$n_threads
EOF

wait

# # Drop everything for this file from cache
# echo "Dropping cache for ${file_name_list[$current_idx]}"
# free -m
dd of=${file_name_list[$current_idx]} oflag=nocache,sync conv=notrunc,fsync,nocreat count=0 status=none
# free -m
done