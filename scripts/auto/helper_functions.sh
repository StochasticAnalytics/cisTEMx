#!/bin/bash

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

function check_exit_status() {
    if [ $? -ne 0 ]; then
        echo "Error: $1 failed!"
        exit 0
    fi
}

function add_time_to_file() {
    flock -x $1 -c "echo $(get_elapsed) >> $1"
}