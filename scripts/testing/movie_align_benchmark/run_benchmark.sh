#!/bin/bash

data_dir=/scratch/salina/proc_EMPIAR-10568-apoferritin-g4-EER/movies_1/for_auto

#dd of=your_file oflag=nocache conv=notrunc,fdatasync count=0
n_movies=4
n_procs=4

if [[ $n_movies -gt 8 || $n_movies -lt 1 ]] ; then
    echo "1-8 Movies.  Exiting."
    exit 1
fi

##############################################
######################## Run the eer benchmark for current conditions (48 frames)

temp_run_file=$(mktemp)

for iProc in $( seq 0 $(($n_procs-1)) ); do
    echo "~/git/cisTEM/scripts/testing/movie_align_benchmark/run_eer.sh $data_dir ${n_movies} ${iProc} 15" >> $temp_run_file
done

start_time=$(date  | awk -F ":" '{print $2*60+$3}')

cat $temp_run_file | parallel -j${n_procs} {} {} {}

rm $temp_run_file

stop_time=$(date  | awk -F ":" '{print $2*60+$3}')

echo "Total time: $(($stop_time-$start_time)) seconds"
echo "48-frame eer Movies per hour = $(echo "" | awk -v Start=${start_time} -v Stop=${stop_time} -v N=${n_movies} -v P=${n_procs} 'BEGIN{print (N*P)/((Stop-Start)/3600)}')"


##############################################
######################## Run the eer benchmark for finner temporal sampling conditions (72 frames)

temp_run_file=$(mktemp)

for iProc in $( seq 0 $(($n_procs-1)) ); do
    echo "~/git/cisTEM/scripts/testing/movie_align_benchmark/run_eer.sh $data_dir ${n_movies} ${iProc} 10" >> $temp_run_file
done

start_time=$(date  | awk -F ":" '{print $2*60+$3}')

cat $temp_run_file | parallel -j${n_procs} {} {} {}

rm $temp_run_file

stop_time=$(date  | awk -F ":" '{print $2*60+$3}')

echo "Total time: $(($stop_time-$start_time)) seconds"
echo "72-frame eer Movies per hour = $(echo "" | awk -v Start=${start_time} -v Stop=${stop_time} -v N=${n_movies} -v P=${n_procs} 'BEGIN{print (N*P)/((Stop-Start)/3600)}')"

##############################################
######################## Run the tif benchmark

temp_run_file=$(mktemp)

for iProc in $( seq 0 $(($n_procs-1)) ); do
    echo "~/git/cisTEM/scripts/testing/movie_align_benchmark/run_tif.sh $data_dir ${n_movies} ${iProc}" >> $temp_run_file
done

start_time=$(date  | awk -F ":" '{print $2*60+$3}')

cat $temp_run_file | parallel -j${n_procs} {} {} {}

rm $temp_run_file

stop_time=$(date  | awk -F ":" '{print $2*60+$3}')

echo "Total time: $(($stop_time-$start_time)) seconds"
echo "48-frame tif Movies per hour = $(echo "" | awk -v Start=${start_time} -v Stop=${stop_time} -v N=${n_movies} -v P=${n_procs} 'BEGIN{print (N*P)/((Stop-Start)/3600)}')"