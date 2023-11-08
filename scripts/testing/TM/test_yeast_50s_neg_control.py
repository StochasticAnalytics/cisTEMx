#!/usr/bin/env python3


from os import remove as remove
from os import makedirs
from os.path import join as join
from os import symlink
import util.args as tmArgs
import util.make_tmp_runfile as mktmp
import util.run_job as runner

# By default the "_gpu" suffix will be added unless the --old-cistem flag is used
# or the --cpu flag is used
wanted_binary_name = 'match_template'


def main():

    args = tmArgs.parse_TM_args(wanted_binary_name)

    # Try first with a single plane
    # We want to do the full search (mostly) with defocus included, so override those defaults
    # args.defocus_range = 900
    # args.defocus_step = 300

    # 2 is the 6Q8Y_mature_60S.mrc template
    # 3 is neg_control_50S_7bv8.mrc
    config = tmArgs.get_config(args, 'Yeast', 3, 0)

    # tmp_filename_match_template, tmp_filename_make_template_results = mktmp.make_tmp_runfile(
    #     args, config)
    # elapsed_time = runner.run_job(tmp_filename_match_template)
    # runner.run_job(tmp_filename_make_template_results)

    # Now run the results again, overriding the default scaled mip and use the mip
    args.results_mip_to_use = 'mip.mrc'
    original_prefix = args.output_file_prefix
    args.output_file_prefix = join(original_prefix, 'not_scaled')
    makedirs(args.output_file_prefix, exist_ok=True)
    # The outputs from the search won't exist in this dir, so link them
    try:
        symlink(join(original_prefix, 'mip.mrc'), join(args.output_file_prefix, 'mip.mrc'))
    except FileExistsError:
        pass
    try:
        symlink(join(original_prefix, 'mip_scaled.mrc'), join(args.output_file_prefix, 'mip_scaled.mrc'))
    except FileExistsError:
        pass
    try:
        symlink(join(original_prefix, 'psi.mrc'), join(args.output_file_prefix, 'psi.mrc'))
    except FileExistsError:
        pass
    try:
        symlink(join(original_prefix, 'theta.mrc'), join(args.output_file_prefix, 'theta.mrc'))
    except FileExistsError:
        pass
    try:
        symlink(join(original_prefix, 'phi.mrc'), join(args.output_file_prefix, 'phi.mrc'))
    except FileExistsError:
        pass
    try:
        symlink(join(original_prefix, 'defocus.mrc'), join(args.output_file_prefix, 'defocus.mrc'))
    except FileExistsError:
        pass
    try:
        symlink(join(original_prefix, 'pixel_size.mrc'), join(args.output_file_prefix, 'pixel_size.mrc'))
    except FileExistsError:
        pass

    try:
        symlink(join(original_prefix, 'hist.txt'), join(args.output_file_prefix, 'hist.txt'))
    except FileExistsError:
        pass

    tmp_filename_match_template, tmp_filename_make_template_results = mktmp.make_tmp_runfile(
        args, config)
    # WE won't re-run the search, so clean that up
    remove(tmp_filename_match_template)
    runner.run_job(tmp_filename_make_template_results)
    print('Time is : ' + str(elapsed_time))


# Check if main function and run
if __name__ == '__main__':
    main()
