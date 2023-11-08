import os
import sys
import argparse
import toml
# If we are in the container

default_data_dir = '/cisTEMdev/cistem_reference_images'


def get_config(args, data_dir: str, ref_number: int, img_number: int):

    if data_dir not in ['Yeast', 'Crown', 'Lamella_from_je', 'SPA']:
        print('The data directory [' +
              data_dir + '] does not exist')
        print('Please provide a valid path to the test data directory as a second argument')
        sys.exit(1)

    # TODO: this won't work with different tomls for different data sets
    # either make the toml match the name of the image.
    # parse the config file
    config = toml.load(args.test_data_path + '/Yeast/MetaData/yeast.toml')

    # FIXME: the image and ref number are annoying and should be more descriptive
    config['full_path_to_img'] = os.path.join(
        args.test_data_path, 'Yeast/Images', config.get('data').get('img_names')[img_number])
    config['full_path_to_ref'] = os.path.join(
        args.test_data_path, 'Yeast/Templates', config.get('data').get('ref_names')[ref_number])

    return config


def parse_TM_args(wanted_binary_name):
    # Parse arguments requiring a path to the directory with the binary to be tested and optionally a path to the test data directory
    parser = argparse.ArgumentParser(description='Test the k3 rotation binary')
    # Required argument
    parser.add_argument(
        '--binary_path', help='Path to the directory with the binary to be tested (Required)', required=True)
    parser.add_argument('--test_data_path',
                        help='Path to the test data directory (Optional - defaults to /cisTEMdev/cistem_reference_images, then pwd)')
    # Argument for the output file path and prefix default to /tmp
    parser.add_argument('--output_file_prefix',
                        help='Path and prefix for the output files (Optional - defaults to /tmp)', default='/tmp')

    # add another optional flag to specify that we are using an older version of cisTEM
    # TODO: for now, just trying to catch the case where we use match_template not match_template_gpu, however,
    # there could be other cases where we need to be more specific if the input options change more over time.
    parser.add_argument('--old_cistem', action='store_true',
                        help='Use this flag if you are using an older version of cisTEM')

    # add an optional cpu flag
    parser.add_argument('--cpu', action='store_true',
                        help='Use this flag if you are using the cpu version of cisTEM')

    args = parser.parse_args()

    args.binary_name = wanted_binary_name
    # currently no plan to have a gpu version
    args.results_binary_name = 'make_template_result'
    # Check if we are using the cpu version or old version and if so modify the binary name with _gpu
    if not (args.old_cistem or args.cpu):
        args.binary_name += '_gpu'

    # Check if the binary exists
    if not os.path.isfile(os.path.join(args.binary_path, args.binary_name)):
        print('The binary ' + s.path.join(args.binary_path,
              args.binary_name) + ' does not exist')
        sys.exit(1)

    # Check if make_template_result binary exists
    if not os.path.isfile(os.path.join(args.binary_path, args.results_binary_name)):
        print('The binary ' + os.path.join(args.binary_path,
              args.results_binary_name) + ' does not exist')
        sys.exit(1)

    # if the optional data path is not given, use the default
    if args.test_data_path is None:
        args.test_data_path = default_data_dir

    # Check if the test data directory exists
    if not os.path.isdir(args.test_data_path):
        print('The test data directory [' +
              args.test_data_path + '] does not exist')
        print('Please provide a valid path to the test data directory as a second argument')
        sys.exit(1)

    # Check that the wanted output path exists and if not try to make it, if not error
    if not os.path.isdir(args.output_file_prefix):
        try:
            os.makedirs(args.output_file_prefix)
        except OSError:
            print('The output file directory [' +
                  args.output_file_prefix + '] does not exist')
            print(
                'Please provide a valid path to the output file directory as a second argument')
            sys.exit(1)

    # Set some default search args that may be overwritten in a given test match_template
    args.out_of_plane_angle = 2.5
    args.in_plane_angle = 1.5
    args.defocus_range = 0
    args.defocus_step = 0
    args.pixel_size_range = 0
    args.pixel_size_step = 0
    args.padding_factor = 1.0
    args.mask_radius = 0
    args.template_symmetry = 'C1'
    args.max_threads = 4

    # Set some default search args that may be overwritten in a given test make_template_results
    args.result_min_peak_radius = 10.0
    args.result_number_to_process = 1
    args.sample_thickness = 2000.0  # Angstrom
    args.result_binning_factor = 4
    args.result_ignore_n_pixels_from_edge = -1

    return args
