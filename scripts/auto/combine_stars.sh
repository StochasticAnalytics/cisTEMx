#!/bin/bash

i=1; 
first_file=1;

#source params.sh
source t.sh
# TODO: move these to params

# wanted number of images
n_img_wanted=2500
n_stacks_recorded=0
header_length=39

# 39 from global_grid
# 89 from local refine



#output_star_name="wlg_${scaling}.star"
#output_stack_name="wlg_${scaling}.mrc"

output_star_name="${output_dir}/combined_1.star"
output_stack_name="${output_dir}/combined_1.mrc"

rm -f $output_star_name
rm -f $output_stack_name
rm -f tmp2.restack
rm -f tmp.restack

ls -d ${output_dir}/particle_stacks/* | 
  while read stack_dir ; do
    echo $stack_dir

    stack_file_name="${stack_dir}/refined_parameters_5_stack.mrc"
    # star_file_name="${stack_dir}/refined_parameters_$((${n_local_iterations}-1)).star"
    # star_file_name="${stack_dir}/final_refinement.star"
    #star_file_name="${stack_dir}/micrograph.star"
    star_file_name="${stack_dir}/refined_parameters_5_stack.star"

    if [[ ! -f $stack_file_name ]] ; then echo "Warning: stack $stack_file_name not found, continuing" ; continue ; fi
    if [[ ! -f $star_file_name ]] ; then echo "Warning: star $star_file_name not found, continuing" ; continue ; fi
    n_expected=$(tail -n -1 $star_file_name | awk '{print $1}')
    
        
        if [[ $first_file -eq 1 ]]; then 
            awk -v HL=$header_length '{if(FNR <= HL ) print $0}' $star_file_name > $output_star_name 
        fi
        first_file=0


        awk -v I=$i -v H=$header_length '{if(FNR>H) {$1=(I+$1-1); print $0}}' $star_file_name >> $output_star_name 
        

        n_stacks_recorded=$(($n_stacks_recorded+1))
        
        n_this_stack=$(tail -n -1 $star_file_name | awk '{print $1 -1 }')
        i=$(($i+$n_this_stack+1))
        
        echo -e "${stack_file_name}\n0-${n_this_stack}" >> tmp.restack
        echo $n_stacks_recorded > tmp2.restack
        

        
        
done

cat tmp.restack >> tmp2.restack
# rm -f tmp.restack
newstack -fileinlist tmp2.restack $output_stack_name 
# rm -f tmp2.restack









