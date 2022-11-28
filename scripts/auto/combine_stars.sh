#!/bin/bash

i=1; 
first_file=1;

#source params.sh
source params.sh
# TODO: move these to params

# wanted number of images
n_img_wanted=2500
n_stacks_recorded=0
header_length=39

# 39 from global_grid
# 89 from local refine



#output_star_name="wlg_${scaling}.star"
#output_stack_name="wlg_${scaling}.mrc"

output_base="combined_1"
output_star_name="${output_dir}/${output_base}.star"
output_star_adj_name="${output_dir}/${output_base}_adj.star"
output_stack_name="${output_dir}/${output_base}.mrc"

rm -f $output_star_name
rm -f $output_stack_name
rm -f tmp2.restack
rm -f tmp.restack

ls -d ${output_dir}/particle_stacks/* | 
  while read stack_dir ; do
    echo $stack_dir

    stack_file_name="${stack_dir}/refined_parameters_1_stack.mrc"
    # star_file_name="${stack_dir}/refined_parameters_$((${n_local_iterations}-1)).star"
    # star_file_name="${stack_dir}/final_refinement.star"
    #star_file_name="${stack_dir}/micrograph.star"
    star_file_name="${stack_dir}/refined_parameters_1_stack.star"

    if [[ ! -f $stack_file_name ]] ; then echo "Warning: stack $stack_file_name not found, continuing" ; continue ; fi
    if [[ ! -f $star_file_name ]] ; then echo "Warning: star $star_file_name not found, continuing" ; continue ; fi
    n_expected=$(tail -n -1 $star_file_name | awk '{print $1}')
    
        
        if [[ $first_file -eq 1 ]]; then 
            awk -v HL=$header_length '{if(FNR <= HL ) print $0}' $star_file_name > $output_star_name 
            awk -v HL=$header_length '{if(FNR <= HL ) print $0}' $star_file_name > $output_star_adj_name
        fi
        first_file=0

        # For now, just set sigma to 100 until I can check into how it is used 
        awk -v I=$i -v H=$header_length '{if(FNR>H) {$1=(I+$1-1); $14=100; print $0}}' $star_file_name >> $output_star_name 
        # Replace the score with the value in score change, which is for now, the re-weighted score based on the estimate of the CCC noise in refinement
        awk -v I=$i -v H=$header_length '{if(FNR>H) {$1=(I+$1-1); $14=100; $15=$16; print $0}}' $star_file_name >> $output_star_adj_name
        

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









