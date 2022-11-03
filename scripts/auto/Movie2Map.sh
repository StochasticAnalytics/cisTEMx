#!/bin/bash

# In the future, movie names will be fed in here as they appear on disk
# For now, we'll just use what is in the directory.

source params.sh

mkdir -p $output_dir
mkdir -p $output_dir/images
mkdir -p $output_dir/volumes
mkdir -p $output_dir/global_search

# ls $movie_dir/* | parallel --bar --progress -j${max_movies_per_gpu} ./auto_movie_align.sh {}

# ./auto_sim_ref.sh





