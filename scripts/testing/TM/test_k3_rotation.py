#!/usr/bin/env python3

import time
import subprocess
import os
import util.args as tmArgs
import util.make_tmp_runfile as mktmp


# By default the "_gpu" suffix will be added unless the --old-cistem flag is used
# or the --cpu flag is used
wanted_binary_name = 'match_template'


def run_job(temp_run_filename):
    start_time = time.time()
    subprocess.run(temp_run_filename)
    elapsed_time = time.time() - start_time
    os.remove(temp_run_filename)
    return elapsed_time


def main():

    args = tmArgs.parse_TM_args(wanted_binary_name)

    # Override some default arguments to make this a fast search
    # We want it to take long enough to armortize any overhead, but not too long, ideally a minute or so
    args.out_of_plane_angle = 5.5
    args.in_plane_angle = 3.5
    args.template_symmetry = 'T'

    config = tmArgs.get_config(args, 'Yeast', 2, 0)

    tmp_filename_match_template, tmp_filename_make_template_results = mktmp.make_tmp_runfile(
        args, config)
    elapsed_time = run_job(tmp_filename_match_template)
    # print wanted_stdin list with newlines between each element
    # create a temporary file for feeding to stdin

    config = tmArgs.get_config(args, 'Yeast', 2, 1)

    tmp_filename_match_template, tmp_filename_make_template_results = mktmp.make_tmp_runfile(
        args, config)
    elapsed_time_rotated = run_job(tmp_filename_match_template)

    print('Times are : ' + str(elapsed_time) +
          ' and ' + str(elapsed_time_rotated))
    print('Time difference: ' + str(elapsed_time - elapsed_time_rotated) +
          ' ' + str((elapsed_time - elapsed_time_rotated)/elapsed_time))


# Check if main function and run
if __name__ == '__main__':
    main()
