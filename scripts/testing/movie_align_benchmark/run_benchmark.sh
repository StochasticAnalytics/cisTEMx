#!/bin/bash

trap 'rm -rf /tmp/cistem_reference_images; exit' INT TERM EXIT ERR

stored_data_dir=/cisTEMx/cistem_reference_images/movie_align_benchmark
data_dir=/tmp/cistem_reference_images/movie_align_benchmark
mkdir -p /tmp/cistem_reference_images /tmp/cistem_reference_images/movie_align_benchmark
cp -r $stored_data_dir/* $data_dir
#dd of=your_file oflag=nocache conv=notrunc,fdatasync count=0
n_movies=4
if [[ $1 ]] ; then
    n_procs=$1
else
    n_procs=3 # Number of movies to align at once, these will be reduced for the k3 tests
fi

if [[ $2 ]] ; then
    n_threads=$2
else
    n_threads=1
fi

if [[ $n_movies -gt 4 || $n_movies -lt 1 ]] ; then
    echo "1-4 Movies.  Exiting."
    exit 1
fi

run_eer=1
run_tif=0
run_k3=0

results_file=$(mktemp)


##############################################
######################## Run the eer benchmark for current conditions (48 frames)

if [[ $run_eer -eq 1 ]] ; then 

echo "Running the EER benchmark for 48 frames using ${n_procs} processes w/ ${n_threads} threads"

temp_run_file=$(mktemp)
echo "#!/bin/bash" > $temp_run_file
chmod a+x $temp_run_file

for iProc in $( seq 0 $(($n_procs-1)) ); do
    echo "/cisTEMx/scripts/movie_align_benchmark/run_eer.sh $data_dir ${n_threads} ${n_movies} ${iProc} 15" >> $temp_run_file
done

start_time=$(date  | awk -F ":" '{print $2*60+$3}')

cat $temp_run_file | parallel -j${n_procs} {} {} {}

rm $temp_run_file

stop_time=$(date  | awk -F ":" '{print $2*60+$3}')

echo "Total time: $(($stop_time-$start_time)) seconds" > $results_file
echo "48-frame eer Movies per hour = $(echo "" | awk -v Start=${start_time} -v Stop=${stop_time} -v N=${n_movies} -v P=${n_procs} 'BEGIN{print (N*P)/((Stop-Start)/3600)}')" >> $results_file


##############################################
######################## Run the eer benchmark for finner temporal sampling conditions (72 frames)

echo "Running the EER benchmark for 72 frames using ${n_procs} processes w/ ${n_threads} processes"

temp_run_file=$(mktemp)
echo "#!/bin/bash" > $temp_run_file
chmod a+x $temp_run_file

for iProc in $( seq 0 $(($n_procs-1)) ); do
    echo "/cisTEMx/scripts/movie_align_benchmark/run_eer.sh $data_dir ${n_threads} ${n_movies} ${iProc} 10" >> $temp_run_file
done

start_time=$(date  | awk -F ":" '{print $2*60+$3}')

cat $temp_run_file | parallel -j${n_procs} {} {} {}

rm $temp_run_file

stop_time=$(date  | awk -F ":" '{print $2*60+$3}')

echo "Total time: $(($stop_time-$start_time)) seconds" >> $results_file
echo "72-frame eer Movies per hour = $(echo "" | awk -v Start=${start_time} -v Stop=${stop_time} -v N=${n_movies} -v P=${n_procs} 'BEGIN{print (N*P)/((Stop-Start)/3600)}')"  >> $results_file

fi

##############################################
######################## Run the tif benchmark
if [[ $run_tif -eq 1 ]] ; then

echo "Running the TIF benchmark using ${n_procs} processes w/ ${n_threads} processes"

temp_run_file=$(mktemp)
echo "#!/bin/bash" > $temp_run_file
chmod a+x $temp_run_file

for iProc in $( seq 0 $(($n_procs-1)) ); do
    echo "/cisTEMx/scripts/movie_align_benchmark/run_tif.sh $data_dir ${n_threads} ${n_movies} ${iProc}" >> $temp_run_file
done

start_time=$(date  | awk -F ":" '{print $2*60+$3}')

cat $temp_run_file | parallel -j${n_procs} {} {} {}

rm $temp_run_file

stop_time=$(date  | awk -F ":" '{print $2*60+$3}')

echo "Total time: $(($stop_time-$start_time)) seconds" >> $results_file
echo "48-frame tif Movies per hour = $(echo "" | awk -v Start=${start_time} -v Stop=${stop_time} -v N=${n_movies} -v P=${n_procs} 'BEGIN{print (N*P)/((Stop-Start)/3600)}')" >> $results_file

fi


##############################################
######################## Run the k3 benchmark
if [[ $run_k3 -eq 1 ]] ; then

    for n_frames in 48 70; do
        iProc=$(($iProc-1))
        echo "Running the K3 benchmark for ${n_frames} frames using ${n_procs} processes w/ ${n_threads} processes"
        temp_run_file=$(mktemp)
        echo "#!/bin/bash" > $temp_run_file
        chmod a+x $temp_run_file

        for iProc in $( seq 0 $(($n_procs-1)) ); do
            echo "/cisTEMx/scripts/movie_align_benchmark/run_k3.sh $data_dir ${n_threads} ${n_movies} ${iProc} $n_frames" >> $temp_run_file
        done

        start_time=$(date  | awk -F ":" '{print $2*60+$3}')

        cat $temp_run_file | parallel -j${n_procs} {} {} {}

        rm $temp_run_file

        stop_time=$(date  | awk -F ":" '{print $2*60+$3}')

        echo "Total time: $(($stop_time-$start_time)) seconds" >> $results_file
        echo "$n_frames-frame k3 Movies per hour = $(echo "" | awk -v Start=${start_time} -v Stop=${stop_time} -v N=${n_movies} -v P=${n_procs} 'BEGIN{print (N*P)/((Stop-Start)/3600)}')" >> $results_file

    done
fi

cat $results_file
rm $results_file
rm -r $data_dir