#!/bin/bash

# Assume we are running from within a production container.
# TODO: save some meta data about branch etc of compilation included in the container to reference (or on cisTEM build)

do_global=yes
refine_defocus=no


data_dir=$1
expected_time_in_seconds=$2

{
echo -e "Starting the beta_gal benchmarks at: $(date)\n"
echo -e "---------------------------------------------------\n" 
echo -e "Running a global refinement\n"
} > /tmp/bgal_benchmark.log

start_time=$(date  | awk -F ":" '{print $2*60+$3}')
/cisTEMx/bin/refine3d_gpu << EOF 
${data_dir}/bgal_stack.mrc
${data_dir}/bgal.star
${data_dir}/bgal_ref.mrc
dummy_stats.txt
no
my_projection_stack.mrc
/tmp/global_search.star
/dev/null/global_search_changes.star
D2
1
0
1
1.0
465
0.0
126.75
292.5
10
0
10
156
10
0
20
29.25
29.25
100
100
100
100
0
50
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
12
EOF
stop_time=$(date  | awk -F ":" '{print $2*60+$3}')

{
echo "global search stop time $stop_time" 
echo "global search start time $start_time" 
echo "" | awk -v B=$start_time -v E=$stop_time -v EXT=$expected_time_in_seconds '{if(E-B > 1.15*EXT) print "Result: FAIL, global search time "E-B"s, expected "EXT"s"; else print "Result: PASS, global search time "E-B"s, expected "EXT"s"}'  
echo -e "---------------------------------------------------\n"
} >> /tmp/bgal_benchmark.log


