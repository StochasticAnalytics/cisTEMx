#!/bin/bash

gpu_id=$1
input_file=$2

source params.sh


cat $input_file | parallel -j${max_movies_per_gpu} ./auto_movie_align.sh {} $gpu_id