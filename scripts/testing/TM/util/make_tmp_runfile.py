from sys import exit
from tempfile import NamedTemporaryFile
from os import chmod, path


def match_template(args, config, output_file_prefix):

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
        path.join(output_file_prefix, 'mip.mrc'),
        path.join(output_file_prefix, 'mip_scaled.mrc'),
        path.join(output_file_prefix, 'psi.mrc'),
        path.join(output_file_prefix, 'theta.mrc'),
        path.join(output_file_prefix, 'phi.mrc'),
        path.join(output_file_prefix, 'defocus.mrc'),
        path.join(output_file_prefix, 'pixel_size.mrc'),
        path.join(output_file_prefix, 'avg.mrc'),
        path.join(output_file_prefix, 'std.mrc'),
        path.join(output_file_prefix, 'hist.mrc'),
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

    return input_cmd


def actually_make_it(args, wanted_stdin):

    # We want to defer execution of the temp script, so set delete=False,
    # and return the filename. This meanse the caller must clean up the file.
    with NamedTemporaryFile(mode='w', delete=False) as stdin_file:
        stdin_file.write('#!/bin/bash\n')
        stdin_file.write(path.join(args.binary_path,
                         args.binary_name) + ' <<EOF\n')
        stdin_file.write('\n'.join(wanted_stdin))
        stdin_file.write('\n')
        stdin_file.write('EOF\n')
        stdin_file.close()
        # make it executable
        chmod(stdin_file.name, 0o755)

    return stdin_file.name


def make_tmp_runfile(args, config: dict, output_file_prefix: str = '/tmp'):
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
        wanted_stdin = match_template(args, config, output_file_prefix)
    elif args.binary_name == 'match_template_gpu':
        # throw an error if the user tries to use the gpu version with the --cpu flag

        print('THIS BLOCK IS BROKEN')
        exit(1)
    else:
        print('Unknown program name')
        exit(1)

    tmp_filename = actually_make_it(args, wanted_stdin)

    return tmp_filename
