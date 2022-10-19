#!/bin/bash

data_dir=$1
expected_resolution=$2

start_time=$(date  | awk -F ":" '{print $2*60+$3}')
# Assume we are running from within a production container.
{
echo -e "---------------------------------------------------\n"
echo -e "Running reconstruction from global refinement\n"
} >> /tmp/bgal_benchmark.log
/cisTEMx/bin/reconstruct3d << EOF
${data_dir}/bgal_stack.mrc
/tmp/global_search.star
my_input_reconstruction.mrc
/tmp/my_reconstruction_1.mrc
/tmp/my_reconstruction_2.mrc
/tmp/my_filtered_reconstruction.mrc
/tmp/global_stats.txt
D2
1
0
1.0
465
0.0
126
0.0
0.0
5.0
1.0
1.0
1.0
yes
yes
no
no
no
yes
no
no
no
no
/dev/null/d1.dat
/dev/null/d2.dat
12
EOF
stop_time=$(date  | awk -F ":" '{print $2*60+$3}')

# Find the index where we cross 0.143
idx1=$(awk 'BEGIN{found=0;idx=0}{if($1 != "C" && found==0 && $5 < 0.143) {found=1; idx=FNR-1}}END{print idx}' /tmp/global_stats.txt)
# Interpolate to find 0.143
resolution=$(awk -v I=$idx1 'BEGIN{x=0;y=0;x1=0;y1=0;x2=0;y2=0}{if($1 != "C" && FNR==I) {x1=$5;y1=$2}; if($1 != "C" && FNR==I+1) {x2=$5;y2=$2}}END{y=y1+(0.143-x1)*(y2-y1)/(x2-x1); print y}' /tmp/global_stats.txt)
{
echo "global recon stop time $stop_time" 
echo "global recon start time $start_time" 
echo -e "\nFound a resolution of $resolution after global search\n"
echo -e "Expected a resolution of $expected_resolution\n"
result=$(echo "" | awk -v R=$resolution -v E=$expected_resolution 'BEGIN{if(R < 1.1*E) {print "PASS"} else {print "FAIL"}}')
echo -e "Result: $result, global resolution $resolution, expected $expected_resolution\n"
echo -e "---------------------------------------------------\n"
} >> /tmp/bgal_benchmark.log

rm -f /tmp/my_reconstruction_1.mrc
rm -f /tmp/my_reconstruction_2.mrc
rm -f /tmp/my_filtered_reconstruction.mrc

