#!/bin/bash

do_global=no
refine_defocus=yes


data_dir=$1
expected_time_in_seconds=$2

{
echo -e "Starting the beta_gal benchmarks at: $(date)\n"
echo -e "---------------------------------------------------\n" 
echo -e "Running a local refinement\n"
} >> /tmp/bgal_benchmark.log

start_time=$(date  | awk -F ":" '{print $2*60+$3}')
/cisTEMx/bin/refine3d_gpu << EOF 
${data_dir}/bgal_stack.mrc
/tmp/global_search.star
${data_dir}/bgal_ref.mrc
dummy_stats.txt
no
my_projection_stack.mrc
/tmp/local_search.star
/tmp/local_search_changes.star
D2
1
0
1
1.0
465
0.0
126.75
292.5
4.0
0
4.0
156
4.0
0
20
29.25
29.25
100
100
100
100
300
20
1
$do_global
yes
yes
yes
yes
yes
yes
no
no
$refine_defocus
yes
no
yes
yes
no
16
EOF
stop_time=$(date  | awk -F ":" '{print $2*60+$3}')

{
echo "local search stop time $stop_time" 
echo "local search start time $start_time" 
echo "" | awk -v B=$start_time -v E=$stop_time -v EXT=$expected_time_in_seconds '{if(E-B > 1.15*EXT) print "Result: FAIL, local search time "E-B"s, expected "EXT"s"; else print "Result: PASS, local search time "E-B"s, expected "EXT"s"}'  
echo -e "---------------------------------------------------\n"
} >> /tmp/bgal_benchmark.log