#!/usr/bin/env python3
from os.path import join
from os import makedirs
import util.args as tmArgs
import util.make_tmp_runfile as mktmp
import util.run_job as runner

# By default the "_gpu" suffix will be added unless the --old-cistem flag is used
# or the --cpu flag is used
wanted_binary_name = 'match_template'


def main():

    args = tmArgs.parse_TM_args(wanted_binary_name)

    # We want to do a single-plane search which is a good enough quick check for most cases.
    elapsed_time = [0, 0, 0]

    wanted_output_file_prefix = args.output_file_prefix
    for image_number in range(0, 3):
        config = tmArgs.get_config(args, 'Apoferritin', 0, image_number)
        # Set image specific output prefix
        args.output_file_prefix = join(wanted_output_file_prefix, config.get('data')[
                                       image_number]['img_name'])
        makedirs(args.output_file_prefix, exist_ok=True)

        tmp_filename_match_template, tmp_filename_make_template_results = mktmp.make_tmp_runfile(
            args, config)
        elapsed_time[image_number] = runner.run_job(
            tmp_filename_match_template)
        runner.run_job(tmp_filename_make_template_results)

    # Print the individual times
    for image_number in range(0, 3):
        print('Time is : ' + str(elapsed_time[image_number]))


# Check if main function and run
if __name__ == '__main__':
    main()
