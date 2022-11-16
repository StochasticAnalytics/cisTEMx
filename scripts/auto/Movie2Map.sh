#!/bin/bash

# In the future, movie names will be fed in here as they appear on disk
# For now, we'll just use what is in the directory.

source params.sh

mkdir -p $output_dir
mkdir -p $output_dir/images
mkdir -p $output_dir/volumes
mkdir -p $output_dir/global_search

[[ $run_simulate == "yes" ]] && ./auto_sim_ref.sh


# When each movie is completed, it will record that it is ready to be processed.
ls $movie_dir/* | parallel --bar --progress -j${max_movies_per_gpu} ./auto_movie_align.sh {}

# ls $movie_dir/* | while read a; do  ./auto_movie_align.sh $a ; exit 0 ; done







