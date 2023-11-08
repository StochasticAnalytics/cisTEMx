from sys import exit
from tempfile import NamedTemporaryFile
from os import chmod, path


def match_template(args, config):

    if (args.cpu):
        use_gpu = "no"
    else:
        use_gpu = "yes"
    high_res_limit = 2*config.get('microscope').get('pixel_size')
    # make a string that will then be used to create a temporary file for feading to stdin
    # TODO: optional output dir
    input_cmd = [
        config.get('full_path_to_img'),
        config.get('full_path_to_ref'),
        path.join(args.output_file_prefix, 'mip.mrc'),
        path.join(args.output_file_prefix, 'mip_scaled.mrc'),
        path.join(args.output_file_prefix, 'psi.mrc'),
        path.join(args.output_file_prefix, 'theta.mrc'),
        path.join(args.output_file_prefix, 'phi.mrc'),
        path.join(args.output_file_prefix, 'defocus.mrc'),
        path.join(args.output_file_prefix, 'pixel_size.mrc'),
        path.join(args.output_file_prefix, 'avg.mrc'),
        path.join(args.output_file_prefix, 'std.mrc'),
        path.join(args.output_file_prefix, 'hist.txt'),
        str(config.get('microscope').get('pixel_size')),
        str(config.get('microscope').get('kv')),
        str(config.get('microscope').get('cs')),
        str(config.get('ctf').get('amplitude_contrast')),
        str(config.get('ctf').get('defocus_1')),
        str(config.get('ctf').get('defocus_2')),
        str(config.get('ctf').get('defocus_angle')),
        str(config.get('ctf').get('extra_phase_shift')),
        str(high_res_limit),
        str(args.out_of_plane_angle),
        str(args.in_plane_angle),
        str(args.defocus_range),
        str(args.defocus_step),
        str(args.pixel_size_range),
        str(args.pixel_size_step),
        str(args.padding_factor),
        str(args.mask_radius),
        str(args.template_symmetry),
        use_gpu,
        str(args.max_threads)]

    pre_process_cmd = " "
    return pre_process_cmd, input_cmd


def make_template_results(args, config):

    read_coordinates = "no"
    pre_process_cmd = str(
        "wanted_threshold=$(awk '/^# Expected/{print $5}' " + path.join(args.output_file_prefix, 'hist.txt)'))
    input_cmd = [
        read_coordinates,
        path.join(args.output_file_prefix, 'mip_scaled.mrc'),
        path.join(args.output_file_prefix, 'psi.mrc'),
        path.join(args.output_file_prefix, 'theta.mrc'),
        path.join(args.output_file_prefix, 'phi.mrc'),
        path.join(args.output_file_prefix, 'defocus.mrc'),
        path.join(args.output_file_prefix, 'pixel_size.mrc'),
        path.join(args.output_file_prefix, 'coordinates.txt'),
        str("$wanted_threshold"),
        str(args.result_min_peak_radius),
        str(args.result_number_to_process),
        config.get('full_path_to_ref'),
        path.join(args.output_file_prefix, 'result.mrc'),
        path.join(args.output_file_prefix, 'slab.mrc'),
        str(args.sample_thickness),
        str(config.get('microscope').get('pixel_size')),
        str(args.result_binning_factor),
        str(args.result_ignore_n_pixels_from_edge)]

    return pre_process_cmd, input_cmd


def actually_make_it(args, pre_process_cmd, wanted_stdin, wanted_binary_name):

    # We want to defer execution of the temp script, so set delete=False,
    # and return the filename. This meanse the caller must clean up the file.
    with NamedTemporaryFile(mode='w', delete=False) as stdin_file:
        stdin_file.write('#!/bin/bash\n\n')
        stdin_file.write(pre_process_cmd)
        stdin_file.write('\n')
        stdin_file.write(wanted_binary_name + ' <<EOF\n')
        stdin_file.write('\n'.join(wanted_stdin))
        stdin_file.write('\n')
        stdin_file.write('EOF\n')
        stdin_file.close()
        # make it executable
        chmod(stdin_file.name, 0o755)

    return stdin_file.name


def make_tmp_runfile(args, config: dict):
    '''
    This function creates a temporary file that can be used to run the match_template binary

    Args:
        args (_type_): _description_
        config (_type_): _description_

    Returns:
        _type_: _description_
    '''

    # Check if we are using the cpu version or old version and if so modify the binary name with _gpu
    if args.binary_name == 'match_template':
        pre_process_cmd, wanted_stdin = match_template(args, config)
        results_preprocess_cmd, results_wanted_stdin = make_template_results(
            args, config)

    elif args.binary_name == 'match_template_gpu':
        # throw an error if the user tries to use the gpu version with the --cpu flag

        print('THIS BLOCK IS BROKEN')
        exit(1)
    else:
        print('Unknown program name')
        exit(1)

    tmp_filename_match_template = actually_make_it(
        args, pre_process_cmd, wanted_stdin, path.join(args.binary_path,
                                                       args.binary_name))
    tmp_filename_make_template_results = actually_make_it(
        args, results_preprocess_cmd, results_wanted_stdin, path.join(args.binary_path,
                                                                      args.results_binary_name))

    return tmp_filename_match_template, tmp_filename_make_template_results
