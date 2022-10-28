#!/bin/bash

data_dir=$1
n_to_align=$2
starting_idx=$3

file_name_list=($(ls ${data_dir}/*.tif))

# TODO movie these to a param file
pixel_size=0.3948
voltage=300
spherical_aberration=2.7
amplitude_contrast=0.07 
output_binning=1

n_eer_frames_to_avg=15
eer_super_res_factor=1
exposure_per_frame=$(echo "" | awk -v n_eer_frames_to_avg=$n_eer_frames_to_avg '{print 0.075*n_eer_frames_to_avg}')

for i in $( seq 0 $(($n_to_align-1)) ); do
    current_idx=$(( ($starting_idx+$i) % $n_to_align))

echo "Process $starting_idx aligning ${file_name_list[$current_idx]}"
apptainer exec -B /scratch --nv ~/software/cisTEMx_production_1.0.0.sif ~/git/cisTEM/build/intel-gpu/src/unblur_gpu << EOF
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
0.5
10
yes
yes
yes
1
0
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