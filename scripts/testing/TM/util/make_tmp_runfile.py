from sys import exit
from tempfile import NamedTemporaryFile
from os import chmod, path


def match_template(config):

    if (config.get('cpu')):
        use_gpu = "no"
    else:
        use_gpu = "yes"

    high_res_limit = 2*config.get('data')[config.get('img_number')].get('pixel_size')

    # make a string that will then be used to create a temporary file for feading to stdin
    # TODO: optional output dir
    input_cmd = [
        config.get('full_path_to_img'),
        config.get('full_path_to_ref'),
        path.join(config.get('output_file_prefix'), 'mip.mrc'),
        path.join(config.get('output_file_prefix'), 'mip_scaled.mrc'),
        path.join(config.get('output_file_prefix'), 'psi.mrc'),
        path.join(config.get('output_file_prefix'), 'theta.mrc'),
        path.join(config.get('output_file_prefix'), 'phi.mrc'),
        path.join(config.get('output_file_prefix'), 'defocus.mrc'),
        path.join(config.get('output_file_prefix'), 'pixel_size.mrc'),
        path.join(config.get('output_file_prefix'), 'avg.mrc'),
        path.join(config.get('output_file_prefix'), 'std.mrc'),
        path.join(config.get('output_file_prefix'), 'hist.txt'),
        str(config.get('data')[config.get('img_number')].get('pixel_size')),
        str(config.get('microscope').get('kv')),
        str(config.get('microscope').get('cs')),
        str(config.get('data')[config.get('img_number')].get('ctf').get('amplitude_contrast')),
        str(config.get('data')[config.get('img_number')].get('ctf').get('defocus_1')),
        str(config.get('data')[config.get('img_number')].get('ctf').get('defocus_2')),
        str(config.get('data')[config.get('img_number')].get('ctf').get('defocus_angle')),
        str(config.get('data')[config.get('img_number')].get('ctf').get('extra_phase_shift')),
        str(high_res_limit),
        str(config.get('out_of_plane_angle')),
        str(config.get('in_plane_angle')),
        str(config.get('defocus_range')),
        str(config.get('defocus_step')),
        str(config.get('pixel_size_range')),
        str(config.get('pixel_size_step')),
        str(config.get('padding_factor')),
        str(config.get('mask_radius')),
        str(config.get('model')[config.get('ref_number')].get('symmetry')),
        use_gpu,
        str(config.get('max_threads'))]

    pre_process_cmd = " "
    return pre_process_cmd, input_cmd


def make_template_results(config):

    read_coordinates = "no"
    pre_process_cmd = str(
        "wanted_threshold=$(awk '/^# Expected/{print $5}' " + path.join(config.get('output_file_prefix'), 'hist.txt)'))
    input_cmd = [
        read_coordinates,
        path.join(config.get('output_file_prefix'), config.get('results_mip_to_use')),
        path.join(config.get('output_file_prefix'), 'psi.mrc'),
        path.join(config.get('output_file_prefix'), 'theta.mrc'),
        path.join(config.get('output_file_prefix'), 'phi.mrc'),
        path.join(config.get('output_file_prefix'), 'defocus.mrc'),
        path.join(config.get('output_file_prefix'), 'pixel_size.mrc'),
        path.join(config.get('output_file_prefix'), 'coordinates.txt'),
        str("$wanted_threshold"),
        str(config.get('result_min_peak_radius')),
        str(config.get('result_number_to_process')),
        config.get('full_path_to_ref'),
        path.join(config.get('output_file_prefix'), 'result.mrc'),
        path.join(config.get('output_file_prefix'), 'slab.mrc'),
        str(config.get('sample_thickness')),
        str(config.get('data')[config.get('img_number')].get('pixel_size')),
        str(config.get('result_binning_factor')),
        str(config.get('result_ignore_n_pixels_from_edge'))]

    return pre_process_cmd, input_cmd


def actually_make_it(pre_process_cmd, wanted_stdin, wanted_binary_name):

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


def make_tmp_runfile(config: dict):
    '''
    This function creates a temporary file that can be used to run the match_template binary
    Returns:
        _type_: _description_
    '''

    # Check if we are using the cpu version or old version and if so modify the binary name with _gpu
    # FIXME: currently without old_cistem, the binary is set to match_template_gpu, but we need some other specifier if the number/order of
    # input args to match_template changes. (As it will soon with FastFFT as an option.)
    if (config.get('old_cistem') == 'yes' and config.get('binary_name') == 'match_template') or config.get('binary_name') == 'match_template_gpu':
        pre_process_cmd, wanted_stdin = match_template(config)
        results_preprocess_cmd, results_wanted_stdin = make_template_results(config)
    else:
        print('Unknown program name')
        exit(1)

    tmp_filename_match_template = actually_make_it(
        pre_process_cmd, wanted_stdin, path.join(config.get('binary_path'),
                                                 config.get('binary_name')))

    tmp_filename_make_template_results = actually_make_it(
        results_preprocess_cmd, results_wanted_stdin, path.join(config.get('binary_path'),
                                                                config.get('results_binary_name')))

    return tmp_filename_match_template, tmp_filename_make_template_results
