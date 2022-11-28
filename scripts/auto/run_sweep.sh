#!/bin/bash

ang_val=(2.5 5 7.5 10 12.5)


for i in 4; do 
    R=${ang_val[$i]}

    case $i in
        0) 
            echo "R is 2.5"
            res_val='(0 2.8)'
            loc_ang='(1 0.5)'
            ;;
        1)
            echo "R is 5"
            res_val='(3.5 0 2.8)'
            loc_ang='(1.66 1 0.5)'
            ;;
        2)
            echo "R is 7.5"
            res_val='(5 3.5 0 2.8)'
            loc_ang='(2.5 1.66 1 0.5)'
            ;;
        3)
            echo "R is 10"
            res_val='(6.5 5 3.5 0 2.8)'
            loc_ang='(3.25 2.5 1.66 1 0.5)'
            ;;
        4)
            echo "R is 12.5"
            res_val='(8 6.5 5 3.5 0 2.8)'
            loc_ang='(4.2 3.25 2.5 1.66 1 0.5)'
            ;;
        *)
            echo "R is not in the list"
            ;;
    esac


    awk -v I="$R" -v A="$loc_ang" -v R="$res_val"  '{if(/^output_dir/) print "output_dir=no_bp_"I ; else print $0}'  apo_params.sh > tmp 
    awk -v I="$R" -v A="$loc_ang" -v R="$res_val"  '{if(/^global_out_of_plane_angle/) print "global_out_of_plane_angle="I; else print $0}' tmp  > tmp2
    awk -v I="$R" -v A="$loc_ang" -v R="$res_val"  '{if(/^global_in_plane_angle/) print "global_in_plane_angle="I/2; else print $0}' tmp2  > tmp
    awk -v I="$R" -v A="$loc_ang" -v R="$res_val"  '{if(/^local_resolution/) print "local_resolution="R; else print $0}' tmp  > tmp2
    awk -v I="$R" -v A="$loc_ang" -v R="$res_val"  '{if(/^local_angle_step/) print "local_angle_step="A; else print $0}' tmp2  > params_${R}.sh
    ln -sf params_${R}.sh params.sh

    ./Movie2Map.sh
done

