#!/bin/bash

# In the future, movie names will be fed in here as they appear on disk
# For now, we'll just use what is in the directory.

source params.sh

get_start
mkdir -p $output_dir
mkdir -p $output_dir/images
mkdir -p $output_dir/volumes
mkdir -p $output_dir/global_search
mkdir -p $output_dir/log

for timing_file in $total_timing_file $simulate_timing_file $movie_align_timing_file $ctf_fit_timing_file $global_search_timing_file $local_search_timing_file $grid_timing_file $global_prepare_stack_timing_file; do
                   echo "Starting Movie2Map at $(date)" > $timing_file
done

get_start
if [[ $run_simulate == "yes" ]] ; then
    get_start
    ./auto_sim_ref.sh
    check_exit_status "Simulation"
    get_stop
    add_time_to_file $simulate_timing_file
fi

# When each movie is completed, it will record that it is ready to be processed.
do_parallel=1
if [[ $do_parallel -eq 1 ]] ; then
ls $movie_dir/* > movie_full.txt
split -n 4 movie_full.txt 
./auto_parallel.sh 0 xaa &
./auto_parallel.sh 1 xab &
./auto_parallel.sh 2 xac &
./auto_parallel.sh 3 xad &
wait
# parallel --bar --progress -j${max_movies_per_gpu} ./auto_movie_align.sh {}
else

ls $movie_dir/* | 
    while read a; do  
        ./auto_movie_align.sh $a 
        exit 0 
    done

fi

get_stop
add_time_to_file $total_timing_file







