#!/bin/bash

#dd of=${file_name_list[$current_idx]} oflag=nocache,sync conv=notrunc,fsync,nocreat count=0 status=none

trap 'rm -rf $test_directory/cistem_disk_io_test' INT TERM EXIT ERR
test_directory=$1

start_time=0
stop_time=0

function get_ms {
    # Get the number of seconds since the epoch
    date +"%H %M %S %N" | awk '{print ($1*3600+$2*60+$3)*1000+$4/1000000}'
}

function get_difference_in_seconds {
  echo $(echo "print( ( $2-$1 )/1000 )"|python3)
}
function get_start {
    start_time=$(get_ms)
}
function get_stop {
    stop_time=$(get_ms)
}

function get_elapsed {
    echo $(get_difference_in_seconds $start_time $stop_time)
}





block_size=4096
n_blocks_small=1000
n_blocks_large=700000
n_small_files=512
n_large_files=16
n_threads=8

small_size_MB=$(echo "print( $block_size*$n_blocks_small/1024/1024 )" | python3)
large_size_GB=$(echo "print( $block_size*$n_blocks_large/1024/1024/1024 )" | python3)
echo "Small file size: $small_size_MB MB"
echo "Large file size: $large_size_GB GB"


# Make sure the directory exists and is writable
if [[ ! -d $test_directory ]] ; then
    echo "Test directory $test_directory does not exist.  Exiting."
    exit 1
fi
if [[ ! -w $test_directory ]] ; then
    echo "Test directory $test_directory is not writable.  Exiting."
    exit 1
fi

# Add our test directory, first checking if it exists and removing if needed
rm -rf $test_directory/cistem_disk_io_test
[ $? -ne 0 ] && echo "Failed to remove $test_directory/cistem_disk_io_test.  Exiting." && exit 1

mkdir $test_directory/cistem_disk_io_test
[ $? -ne 0 ] && echo "Failed to create $test_directory/cistem_disk_io_test.  Exiting." && exit 1

# first we'll write out many small files serially
get_start
for iFile in $(seq 1 $n_small_files); do
    dd if=/dev/zero of=$test_directory/cistem_disk_io_test/small_file_$iFile bs=$block_size count=$n_blocks_small status=none conv=fdatasync,fsync
    [ $? -ne 0 ] && echo "Failed to write $test_directory/cistem_disk_io_test/small_file_$iFile.  Exiting." && exit 1
    echo $test_directory/cistem_disk_io_test/small_file_$iFile >> $test_directory/cistem_disk_io_test/small_file_list.txt
done
get_stop
echo "Total time for $n_small_files ($small_size_MB MB) files write in serial (): $(get_elapsed) seconds"

cat $test_directory/cistem_disk_io_test/small_file_list.txt | 
    parallel -j${n_threads} dd of={} oflag=nocache,sync conv=notrunc,fsync,nocreat count=0 status=none

# now we'll read  in the small files in serial
get_start
for iFile in $(seq 1 $n_small_files); do
    dd if=$test_directory/cistem_disk_io_test/small_file_$iFile of=/dev/null bs=$block_size count=$n_blocks_small status=none 
    [ $? -ne 0 ] && echo "Failed to read $test_directory/cistem_disk_io_test/small_file_$iFile.  Exiting." && exit 1
done
get_stop


echo "Total time for $n_small_files ($small_size_MB MB) files read  in serial (): $(get_elapsed) seconds"

cat $test_directory/cistem_disk_io_test/small_file_list.txt | parallel -j${n_threads} rm {}

# now we'll write out many small files in parallel
get_start
cat $test_directory/cistem_disk_io_test/small_file_list.txt | 
    parallel -j${n_threads} dd if=/dev/zero of={} bs=$block_size count=$n_blocks_small status=none conv=fdatasync,fsync
get_stop


echo "Total time for $n_small_files ($small_size_MB MB) files write in parallel (): $(get_elapsed) seconds"

cat $test_directory/cistem_disk_io_test/small_file_list.txt | 
    parallel -j${n_threads} dd of={} oflag=nocache,sync conv=notrunc,fsync,nocreat count=0 status=none

# now we'll read  in the small files in parallel

get_start
cat $test_directory/cistem_disk_io_test/small_file_list.txt | 
    parallel -j${n_threads}  dd if={} of=/dev/null bs=$block_size count=$n_blocks_small status=none 
get_stop


echo "Total time for $n_small_files ($small_size_MB MB) files read  in parallel (): $(get_elapsed) seconds"

cat $test_directory/cistem_disk_io_test/small_file_list.txt | parallel -j${n_threads} rm {}

# now we'll write out many large files in serial
get_start
for iFile in $(seq 1 $n_large_files); do
    dd if=/dev/zero of=$test_directory/cistem_disk_io_test/large_file_$iFile bs=$block_size count=$n_blocks_large status=none conv=fdatasync,fsync
    [ $? -ne 0 ] && echo "Failed to write $test_directory/cistem_disk_io_test/large_file_$iFile.  Exiting." && exit 1
    echo $test_directory/cistem_disk_io_test/large_file_$iFile >> $test_directory/cistem_disk_io_test/large_file_list.txt    
done
get_stop

echo "Total time for $n_large_files ($large_size_GB GB) files write in serial (): $(get_elapsed) seconds"

# drop caches again
cat $test_directory/cistem_disk_io_test/large_file_list.txt | 
    parallel -j${n_threads} dd of={} oflag=nocache,sync conv=notrunc,fsync,nocreat count=0 status=none


# now we'll read  in the large files in serial
get_start
for iFile in $(seq 1 $n_large_files); do
    dd if=$test_directory/cistem_disk_io_test/large_file_$iFile of=/dev/null bs=$block_size count=$n_blocks_large status=none 
    [ $? -ne 0 ] && echo "Failed to read $test_directory/cistem_disk_io_test/large_file_$iFile.  Exiting." && exit 1
done
get_stop


echo "Total time for $n_large_files ($large_size_GB GB) files read  in serial (): $(get_elapsed) seconds"

cat $test_directory/cistem_disk_io_test/large_file_list.txt | parallel -j${n_threads} rm {}

# now we'll write out many large files in parallel

get_start
cat $test_directory/cistem_disk_io_test/large_file_list.txt | 
    parallel -j${n_threads}  dd if=/dev/zero of={} bs=$block_size count=$n_blocks_large status=none conv=fdatasync,fsync
get_stop


echo "Total time for $n_large_files ($large_size_GB GB) files write in parallel (): $(get_elapsed) seconds"

cat $test_directory/cistem_disk_io_test/large_file_list.txt | 
    parallel -j${n_threads} dd of={} oflag=nocache,sync conv=notrunc,fsync,nocreat count=0 status=none

# now we'll read  in the large files in parallel

get_start
cat $test_directory/cistem_disk_io_test/large_file_list.txt | 
    parallel -j${n_threads}  dd if={} of=/dev/null bs=$block_size count=$n_blocks_large status=none 
get_stop
echo "Total time for $n_large_files ($large_size_GB GB) files read  in parallel (): $(get_elapsed) seconds"


