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

    # We want to do the full search (mostly) with defocus included, so override those defaults
    args.defocus_range = 900
    args.defocus_step = 300

    config = tmArgs.get_config(args, 'Yeast', 2, 0)

    temp_run_filename = mktmp.make_tmp_runfile(args, config)
    elapsed_time = run_job(temp_run_filename)

    print('Time is : ' + str(elapsed_time))


# Check if main function and run
if __name__ == '__main__':
    main()
