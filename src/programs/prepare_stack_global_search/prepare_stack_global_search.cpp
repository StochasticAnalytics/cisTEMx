#include "../../core/core_headers.h"

class
        MakeParticleStack : public MyApp {
  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(MakeParticleStack)

// override the DoInteractiveUserInput

void MakeParticleStack::DoInteractiveUserInput( ) {

    wxString input_mip_filename;
    wxString input_image_filename;
    wxString input_best_psi_filename;
    wxString input_best_theta_filename;
    wxString input_best_phi_filename;
    wxString input_best_defocus_filename;
    wxString output_star_filename;
    wxString output_particle_stack_filename;
    wxString input_starfilename;

    float wanted_threshold;
    float min_peak_radius;
    float pixel_size                   = 1; // We are getting this from the images now.
    float voltage_kV                   = 300.0;
    float spherical_aberration_mm      = 2.7;
    float amplitude_contrast           = 0.07;
    float average_defocus_1            = 5000.0;
    float average_defocus_2            = 5000.0;
    float average_defocus_angle        = 0.0;
    int   box_size                     = 256;
    int   number_of_results_to_process = 1;

    bool read_coordinates;

    UserInput* my_input = new UserInput("MakeParticleStack", 1.00);

    read_coordinates = my_input->GetYesNoFromUser("Read coordinates from starfile?", "Should the target coordinates be read from a starfile instead of search results?", "No");
    if ( read_coordinates ) {
        input_starfilename = my_input->GetFilenameFromUser("Input starfile", "The file containing information on the targets", "particle_stack.star", true);
        wanted_threshold   = my_input->GetFloatFromUser("Peak threshold", "Peaks over this size will be taken", "7.5", 0.0);
        // Currently not used for read_coordinates, but should be used to remove duplicate particles.
        // For an initial implementation, use "best 2d class to store the image number"
        min_peak_radius              = my_input->GetFloatFromUser("Min Peak Radius (angstroms.)", "Essentially the minimum closeness for peaks", "10.0", 1.0);
        number_of_results_to_process = my_input->GetIntFromUser("Number of results to process (0=all)", "If input files contain results from several searches, how many to include?", "0", 0);
    }
    else {
        input_mip_filename           = my_input->GetFilenameFromUser("Input MIP file", "The file for saving the maximum intensity projection image", "mip.mrc", true);
        input_best_psi_filename      = my_input->GetFilenameFromUser("Input psi file", "The file containing the best psi image", "psi.mrc", true);
        input_best_theta_filename    = my_input->GetFilenameFromUser("Input theta file", "The file containing the best psi image", "theta.mrc", true);
        input_best_phi_filename      = my_input->GetFilenameFromUser("Input phi file", "The file containing the best phi image", "phi.mrc", true);
        input_best_defocus_filename  = my_input->GetFilenameFromUser("Input defocus file", "The file with the best defocus image", "defocus.mrc", false);
        input_starfilename           = my_input->GetFilenameFromUser("Output x,y,z coordinate file", "The file for saving the x,y,z coordinates of the found targets", "coordinates.txt", false);
        wanted_threshold             = my_input->GetFloatFromUser("Peak threshold", "Peaks over this size will be taken", "7.5", 0.0);
        min_peak_radius              = my_input->GetFloatFromUser("Min Peak Radius (angstroms.)", "Essentially the minimum closeness for peaks", "10.0", 1.0);
        number_of_results_to_process = my_input->GetIntFromUser("Number of results to process (0=all)", "If input files contain results from several searches, how many to include?", "0", 0);
    }
    input_image_filename           = my_input->GetFilenameFromUser("Input image file", "The image that was searched", "image.mrc", true);
    output_star_filename           = my_input->GetFilenameFromUser("Output star file", "The star file containing the particle alignment parameters", "particle_stack.star", false);
    output_particle_stack_filename = my_input->GetFilenameFromUser("Output particle stack", "The output image stack, containing the picked particles", "particle_stack.mrc", false);
    box_size                       = my_input->GetIntFromUser("Box size for particles (px.)", "The pixel dimensions of the box used to cut out the particles", "256", 10);
    if ( ! read_coordinates ) {
        // We'll grab these from the starfile
        // pixel_size              = my_input->GetFloatFromUser("Pixel size of image (A)", "Pixel size of input image in Angstroms", "1.0", 0.0);
        average_defocus_1       = my_input->GetFloatFromUser("Average defocus 1 (A)", "The average defocus estimated for the image in direction 1", "5000.0");
        average_defocus_2       = my_input->GetFloatFromUser("Average defocus 2 (A)", "The average defocus estimated for the image in direction 2", "5000.0");
        average_defocus_angle   = my_input->GetFloatFromUser("Average defocus angle (deg)", "The average defocus angle estimated for the image", "0.0");
        voltage_kV              = my_input->GetFloatFromUser("Beam energy (keV)", "The energy of the electron beam used to image the sample in kilo electron volts", "300.0", 0.0);
        spherical_aberration_mm = my_input->GetFloatFromUser("Spherical aberration (mm)", "Spherical aberration of the objective lens in millimeters", "2.7", 0.0);
        amplitude_contrast      = my_input->GetFloatFromUser("Amplitude contrast", "Assumed amplitude contrast", "0.07", 0.0, 1.0);
    }

    delete my_input;

    //	my_current_job.Reset(14);
    my_current_job.ManualSetArguments("tttttttttifffffffffbiii",
                                      input_mip_filename.ToUTF8( ).data( ), // 0
                                      input_best_psi_filename.ToUTF8( ).data( ), // 1
                                      input_best_theta_filename.ToUTF8( ).data( ), // 2
                                      input_best_phi_filename.ToUTF8( ).data( ), // 3
                                      input_best_defocus_filename.ToUTF8( ).data( ), // 4
                                      input_starfilename.ToUTF8( ).data( ), // 5
                                      input_image_filename.ToUTF8( ).data( ), // 6
                                      output_star_filename.ToUTF8( ).data( ), // 7
                                      output_particle_stack_filename.ToUTF8( ).data( ), // 8
                                      box_size, // 9
                                      pixel_size, // 10
                                      average_defocus_1, // 11
                                      average_defocus_2, // 12
                                      average_defocus_angle, // 13
                                      voltage_kV, // 14
                                      spherical_aberration_mm, // 15
                                      amplitude_contrast, // 16
                                      wanted_threshold, // 17
                                      min_peak_radius, // 18
                                      read_coordinates, // 19
                                      number_of_results_to_process); // 20
}

// override the do calculation method which will be what is actually run..

bool MakeParticleStack::DoCalculation( ) {

    wxDateTime start_time = wxDateTime::Now( );

    wxString input_mip_filename             = my_current_job.arguments[0].ReturnStringArgument( );
    wxString input_best_psi_filename        = my_current_job.arguments[1].ReturnStringArgument( );
    wxString input_best_theta_filename      = my_current_job.arguments[2].ReturnStringArgument( );
    wxString input_best_phi_filename        = my_current_job.arguments[3].ReturnStringArgument( );
    wxString input_best_defocus_filename    = my_current_job.arguments[4].ReturnStringArgument( );
    wxString input_starfilename             = my_current_job.arguments[5].ReturnStringArgument( );
    wxString input_image_filename           = my_current_job.arguments[6].ReturnStringArgument( );
    wxString output_star_filename           = my_current_job.arguments[7].ReturnStringArgument( );
    wxString output_particle_stack_filename = my_current_job.arguments[8].ReturnStringArgument( );
    int      box_size                       = my_current_job.arguments[9].ReturnIntegerArgument( );
    float    pixel_size                     = my_current_job.arguments[10].ReturnFloatArgument( );
    float    average_defocus_1              = my_current_job.arguments[11].ReturnFloatArgument( );
    float    average_defocus_2              = my_current_job.arguments[12].ReturnFloatArgument( );
    float    average_defocus_angle          = my_current_job.arguments[13].ReturnFloatArgument( );
    float    voltage_kV                     = my_current_job.arguments[14].ReturnFloatArgument( );
    float    spherical_aberration_mm        = my_current_job.arguments[15].ReturnFloatArgument( );
    float    amplitude_contrast             = my_current_job.arguments[16].ReturnFloatArgument( );
    float    wanted_threshold               = my_current_job.arguments[17].ReturnFloatArgument( );
    float    min_peak_radius                = my_current_job.arguments[18].ReturnFloatArgument( );
    bool     read_coordinates               = my_current_job.arguments[19].ReturnBoolArgument( );
    int      number_of_results_to_process   = my_current_job.arguments[20].ReturnIntegerArgument( );

    float binned_pixel_size;
    Image mip_image;
    Image psi_image;
    Image theta_image;
    Image phi_image;
    Image current_particle;
    Image micrograph;

    Peak current_peak;

    float current_phi;
    float current_theta;
    float current_psi;
    float current_defocus    = 0.f;
    float current_pixel_size = 1.0f;

    int   number_of_peaks_found         = 0;
    int   particle_group                = 0;
    int   n_processed_in_particle_group = 0;
    int   n_processed_in_total          = 0;
    int   n_retained                    = 0;
    float sq_dist_x, sq_dist_y;
    float micrograph_mean;
    float variance;
    long  address;
    long  text_file_access_type;
    int   i, j;

    cisTEMParameters input_star_file;
    cisTEMParameters output_star_file_clean;
    cisTEMParameters output_star_file;

    // Start of by getting the binned and unbinned pixel sizes from the input file.
    // In reality, we only need the binned if we are working on making results from a global_search (read_coordinates == False)
    // While we switch to the unbinned images for global_search_refinement (read_coordinates == True)
    {
        wxFileName binned_input_filename(input_image_filename);
        binned_input_filename.ClearExt( ); // Remove the extension AND the dot
        std::string binned_input_name = binned_input_filename.GetFullPath( ).ToStdString( ) + wxString::Format("_bin%d.mrc", 2).ToStdString( );
        MRCFile     get_binned_pixel_size(binned_input_name, false);
        MRCFile     get_pixel_size(input_image_filename.ToStdString( ), false);
        // get_binned_pixel_size.OpenFile( );
        // get_pixel_size.OpenFile( );

        binned_pixel_size = get_binned_pixel_size.my_header.ReturnPixelSize( );
        pixel_size        = get_pixel_size.my_header.ReturnPixelSize( );

        get_binned_pixel_size.CloseFile( );
        get_pixel_size.CloseFile( );
    }

    wxPrintf("binned_pixel_size = %f pixel_size = %f\n", binned_pixel_size, pixel_size);

    // Min peak radius is excluded around each detected peak, only used when ! readcoordinas (binned pixels)
    min_peak_radius /= binned_pixel_size;
    float min_peak_radius_squared = min_peak_radius * min_peak_radius;
    std::cerr << "min_peak_radius_squared = " << min_peak_radius_squared << std::endl;
    if ( ! read_coordinates ) {

        MRCFile input_mip_file(input_mip_filename.ToStdString( ), false);
        MRCFile input_psi_file(input_best_psi_filename.ToStdString( ), false);
        MRCFile input_theta_file(input_best_theta_filename.ToStdString( ), false);
        MRCFile input_phi_file(input_best_phi_filename.ToStdString( ), false);
        // Make sure the number of results to process is not greater than the number of particles in the input file. Zero to process all.
        if ( number_of_results_to_process == 0 )
            number_of_results_to_process = input_mip_file.ReturnNumberOfSlices( );
        else if ( number_of_results_to_process > input_mip_file.ReturnNumberOfSlices( ) ) {
            SendInfo("Warning: Number of results to process is greater than the number of particles in the input file. Setting number of results to process to the number of particles in the input file.");
            number_of_results_to_process = input_mip_file.ReturnNumberOfSlices( );
        }

        // Now let's make sure all the results files are the same size
        MyDebugAssertTrue(input_mip_file.HasSameDimensionsAs(input_psi_file), "Input MIP file and input psi file do not have the same dimensions");
        MyDebugAssertTrue(input_mip_file.HasSameDimensionsAs(input_theta_file), "Input MIP file and input theta file do not have the same dimensions");
        MyDebugAssertTrue(input_mip_file.HasSameDimensionsAs(input_phi_file), "Input MIP file and input phi file do not have the same dimensions");

        mip_image.ReadSlices(&input_mip_file, 1, input_mip_file.ReturnNumberOfSlices( ));
        psi_image.ReadSlices(&input_psi_file, 1, input_psi_file.ReturnNumberOfSlices( ));
        theta_image.ReadSlices(&input_theta_file, 1, input_theta_file.ReturnNumberOfSlices( ));
        phi_image.ReadSlices(&input_phi_file, 1, input_phi_file.ReturnNumberOfSlices( ));
    }
    // We'll reset the peaks values in each loop, so no need to initialize here.
    std::vector<cisTEMParameterLine> output_parameters(number_of_results_to_process);
    std::vector<double>              avg_occupancy(number_of_results_to_process, 0.f);

    if ( read_coordinates ) {
        input_star_file.ReadFromcisTEMStarFile(input_starfilename.ToStdString( ), false);
        std::cerr << "Input star file size is " << input_star_file.ReturnNumberofLines( ) << std::endl;

        // FIXME: For the time being, we'll only take the top scoring paricle from each group
        output_star_file = input_star_file.ReturnTopNFromParticleGroups(1);
        output_star_file_clean.PreallocateMemoryAndBlank(output_star_file.ReturnNumberofLines( ));
        std::cerr << "Output star file size is " << output_star_file.ReturnNumberofLines( ) << std::endl;
    }
    else {
        std::cerr << "Processing : " << number_of_results_to_process << " results " << std::endl;
        output_star_file.PreallocateMemoryAndBlank(cistem::maximum_number_of_detections * number_of_results_to_process);
    }

    micrograph.QuickAndDirtyReadSlice(input_image_filename.ToStdString( ), 1);
    // This is used for Image::CliptInto to catch cases where the particle is too close to the edge of the image.
    micrograph_mean = micrograph.ReturnAverageOfRealValues( );

    // This will be redundant if there are not multiple results, but copying the first mip is the
    // easiest way to get things right when the mip is a stack of results.
    Image mip_to_search;

    // assume square
    // loop until the found peak is below the threshold
    MRCFile output_file;
    if ( read_coordinates ) {
        current_particle.Allocate(box_size, box_size, true);

        output_file.OpenFile(output_particle_stack_filename.ToStdString( ), true, false);
        output_file.SetPixelSize(pixel_size);
        output_file.SetOutputToFP16( );
    }
    else {
        mip_to_search.Allocate(mip_image.logical_x_dimension, mip_image.logical_y_dimension, 1, true);
        for ( int iPixel = 0; iPixel < mip_to_search.real_memory_allocated; iPixel++ ) {
            // TODO: could use a memcpy here.
            mip_to_search.real_values[iPixel] = mip_image.real_values[iPixel];
        }
    }

    wxPrintf("\n");
    for ( int possible_peak = 0; possible_peak < cistem::maximum_number_of_detections; possible_peak++ ) {

        if ( ! read_coordinates ) {

            // Get the next highest value.
            // TODO: make this configurable as in make_template_results
            current_peak = mip_to_search.FindPeakWithIntegerCoordinates(0.0, FLT_MAX, box_size / cistem::fraction_of_box_size_to_exclude_for_border + 1);

            // current_peak = mip_image.FindPeakWithIntegerCoordinates(0.0, FLT_MAX, box_size / cistem::fraction_of_box_size_to_exclude_for_border + 1);
            if ( current_peak.value < wanted_threshold )
                break;

            // Set the origin to the lower-left rather than the center.
            current_peak.x = current_peak.x + mip_to_search.physical_address_of_box_center_x;
            current_peak.y = current_peak.y + mip_to_search.physical_address_of_box_center_y;
            address        = myroundint(current_peak.y) * (mip_to_search.logical_x_dimension + mip_to_search.padding_jump_value) + myroundint(current_peak.x);

            for ( int iResult = 0; iResult < number_of_results_to_process; iResult++ ) {
                // Implicitly assuming the wanted threshold is always > 0.
                output_parameters[iResult].SetAllToZero( );
            }

            output_parameters[0].psi     = psi_image.real_values[address];
            output_parameters[0].theta   = theta_image.real_values[address];
            output_parameters[0].phi     = phi_image.real_values[address];
            output_parameters[0].x_shift = current_peak.x * binned_pixel_size; // FIXME: if saving a particle stack this is no longer valid, need option for both
            output_parameters[0].y_shift = current_peak.y * binned_pixel_size;
            output_parameters[0].score   = current_peak.value;

            address = 0;

            float sq_dist_x, sq_dist_y;
            bool  peak_is_a_duplicate;
            for ( int j = std::max(myroundint(current_peak.y - min_peak_radius), 0); j < std::min(myroundint(current_peak.y + min_peak_radius), mip_to_search.logical_y_dimension); j++ ) {
                sq_dist_y = float(j) - current_peak.y;
                sq_dist_y *= sq_dist_y;

                for ( int i = std::max(myroundint(current_peak.x - min_peak_radius), 0); i < std::min(myroundint(current_peak.x + min_peak_radius), mip_to_search.logical_x_dimension); i++ ) {
                    sq_dist_x = float(i) - current_peak.x;
                    sq_dist_x *= sq_dist_x;
                    address = mip_to_search.ReturnReal1DAddressFromPhysicalCoord(i, j, 0);

                    if ( sq_dist_x + sq_dist_y <= min_peak_radius_squared ) {
                        // We are within the exclusion radius, so first we'll zero out values in the mip_to_search.
                        mip_to_search.real_values[address] = 0.f;
                        // We are assuming that any alternate peaks are within the exclusion radius of the first peak in the additional mips.
                        for ( int iResult = 1; iResult < number_of_results_to_process; iResult++ ) {
                            address += mip_to_search.real_memory_allocated;
                            if ( mip_image.real_values[address] > output_parameters[iResult].score ) {
                                // Make sure this isn't a duplicate on the angles
                                for ( int iDuplicate = iResult - 1; iDuplicate > 0; iDuplicate-- ) {
                                    if ( FloatsAreAlmostTheSame(output_parameters[iDuplicate].psi, psi_image.real_values[address]) &&
                                         FloatsAreAlmostTheSame(output_parameters[iDuplicate].theta, theta_image.real_values[address]) &&
                                         FloatsAreAlmostTheSame(output_parameters[iDuplicate].phi, phi_image.real_values[address]) ) {
                                        std::cerr << "Duplicate peak found at " << i << ", " << j << ", " << iResult << std::endl;
                                        break;
                                    }
                                    output_parameters[iResult].psi     = psi_image.real_values[address];
                                    output_parameters[iResult].theta   = theta_image.real_values[address];
                                    output_parameters[iResult].phi     = phi_image.real_values[address];
                                    output_parameters[iResult].x_shift = i * binned_pixel_size; // FIXME: if saving a particle stack this is no longer valid, need option for both
                                    output_parameters[iResult].y_shift = j * binned_pixel_size;
                                    output_parameters[iResult].score   = mip_image.real_values[address];
                                }
                            }
                            mip_image.real_values[address] = 0.f;
                        }
                    }
                }
            }

            for ( int iResult = 0; iResult < number_of_results_to_process; iResult++ ) {
                // We don't want to include extra peaks that are below the threshold
                // In reconstruct3d the weight is determined by occupancy / average occupancy, so to make the occupancy
                // simply fractional, ensure the sum of the occupancy for each particle = N particles, st. the average = 1

                output_parameters[iResult].position_in_stack = number_of_peaks_found + 1;

                output_parameters[iResult].defocus_1                          = average_defocus_1 + current_defocus;
                output_parameters[iResult].defocus_2                          = average_defocus_2 + current_defocus;
                output_parameters[iResult].defocus_angle                      = average_defocus_angle;
                output_parameters[iResult].pixel_size                         = pixel_size; // We want the pixel size in the un binned image
                output_parameters[iResult].microscope_voltage_kv              = voltage_kV;
                output_parameters[iResult].microscope_spherical_aberration_mm = spherical_aberration_mm;
                output_parameters[iResult].amplitude_contrast                 = amplitude_contrast;
                output_parameters[iResult].occupancy                          = (output_parameters[iResult].score > wanted_threshold) ? 1.0f : 0.f; // for now, include all peaks, even if below threshold and use occupancy to turn off
                output_parameters[iResult].sigma                              = 10.0f;
                output_parameters[iResult].logp                               = 5000.0f;

                output_parameters[iResult].image_is_active = (output_parameters[iResult].score > wanted_threshold) ? 1 : -1;
                output_parameters[iResult].particle_group  = particle_group + 1;
                output_parameters[iResult].assigned_subset = IsOdd(particle_group + 1) ? 1 : 2; // make sure that different possibilites for a given particle are in the same FSC subset

                output_star_file.all_parameters.Item(number_of_peaks_found) = output_parameters[iResult];
                number_of_peaks_found++;
            }
            // All the peaks in a given exclusion radius will have the same particle group number.
            particle_group++;
        }
        else {
            output_star_file.all_parameters.Item(n_processed_in_total).score = std::max(output_star_file.all_parameters.Item(n_processed_in_total).score_change,
                                                                                        output_star_file.all_parameters.Item(n_processed_in_total).score);
            // We don't want to re-activate something that is otherwise shutdown
            if ( output_star_file.all_parameters.Item(n_processed_in_total).image_is_active > 0 )
                output_star_file.all_parameters.Item(n_processed_in_total).image_is_active = (output_star_file.all_parameters.Item(n_processed_in_total).score < wanted_threshold) ? -1 : 1;

            // Get the x/y position and convert from ang to pixels
            current_peak.x = output_star_file.all_parameters.Item(n_processed_in_total).x_shift / output_star_file.all_parameters.Item(n_processed_in_total).pixel_size;
            current_peak.y = output_star_file.all_parameters.Item(n_processed_in_total).y_shift / output_star_file.all_parameters.Item(n_processed_in_total).pixel_size;

            // First adjust to an offset from the center of the image
            output_star_file.all_parameters.Item(n_processed_in_total).x_shift = current_peak.x - float(micrograph.physical_address_of_box_center_x);
            output_star_file.all_parameters.Item(n_processed_in_total).y_shift = current_peak.y - float(micrograph.physical_address_of_box_center_y);

            // Now grab the non-integer part of the shift
            output_star_file.all_parameters.Item(n_processed_in_total).x_shift = output_star_file.all_parameters.Item(n_processed_in_total).x_shift - myroundint(output_star_file.all_parameters.Item(n_processed_in_total).x_shift);
            output_star_file.all_parameters.Item(n_processed_in_total).y_shift = output_star_file.all_parameters.Item(n_processed_in_total).y_shift - myroundint(output_star_file.all_parameters.Item(n_processed_in_total).y_shift);

            // Finally convert back to angstroms
            output_star_file.all_parameters.Item(n_processed_in_total).x_shift = output_star_file.all_parameters.Item(n_processed_in_total).x_shift * output_star_file.all_parameters.Item(n_processed_in_total).pixel_size;
            output_star_file.all_parameters.Item(n_processed_in_total).y_shift = output_star_file.all_parameters.Item(n_processed_in_total).y_shift * output_star_file.all_parameters.Item(n_processed_in_total).pixel_size;

            if ( output_star_file.all_parameters.Item(n_processed_in_total).image_is_active > 0 ) {
                output_star_file_clean.all_parameters.Item(n_retained)                   = output_star_file.all_parameters.Item(n_processed_in_total);
                output_star_file_clean.all_parameters.Item(n_retained).position_in_stack = n_retained + 1;
                n_retained++;
                n_processed_in_total++;
                micrograph.ClipInto(&current_particle, micrograph_mean, false, 1.0,
                                    myroundint(current_peak.x - micrograph.physical_address_of_box_center_x),
                                    myroundint(current_peak.y - micrograph.physical_address_of_box_center_y), 0);
                //		micrograph.ClipInto(&current_particle, micrograph_mean, false, 1.0, int(current_peak.x * pixel_size), int(current_peak.y * pixel_size), 0);
                //		micrograph.ClipInto(&current_particle, micrograph_mean, false, 1.0, int(- current_peak.x * pixel_size + current_particle.physical_address_of_box_center_x), int(- current_peak.y * pixel_size + current_particle.physical_address_of_box_center_y), 0);
                variance = current_particle.ReturnVarianceOfRealValues( );
                if ( variance == 0.0f )
                    variance = 1.0f;
                current_particle.AddMultiplyConstant(-current_particle.ReturnAverageOfRealValuesOnEdges( ), 1.0f / sqrtf(variance));

                current_particle.WriteSlice(&output_file, n_retained);
            }
            else {
                output_star_file.all_parameters.Item(n_processed_in_total).position_in_stack = n_processed_in_total + 1;
                // For now, we'll also still include inactive images in the output star file and hence the output stack
                // lastly post increment the counter on peaks found

                n_processed_in_total++;
            }
        }

        if ( read_coordinates && output_star_file.ReturnNumberofLines( ) == n_processed_in_total )
            break;
    }

    // mip_to_search.QuickAndDirtyWriteSlice("mip_to_search.mrc", 1);
    // mip_image.QuickAndDirtyWriteSlices("mip_image.mrc", 1, mip_image.logical_z_dimension);
    // exit(1);

    if ( read_coordinates ) {
        output_file.CloseFile( );
        wxPrintf("Overwriting starfile with new coordinates\n\n");
        output_star_file = output_star_file_clean;
    }

    // By using SetAllToZero, we inadvertently activate some parameters that aren't necessary.
    // The filenames in particular make the star file harder to read and for auto_functionality are not necessary.
    output_star_file.parameters_to_write.stack_filename          = false;
    output_star_file.parameters_to_write.original_image_filename = false;
    output_star_file.parameters_to_write.reference_3d_filename   = false;
    output_star_file.parameters_to_write.best_2d_class           = false;

    // FIXME: unify
    int n_to_write = (read_coordinates) ? n_retained : number_of_peaks_found;
    output_star_file.WriteTocisTEMStarFile(output_star_filename, -1, -1, 1, n_to_write);

    if ( is_running_locally == true ) {
        wxPrintf("\nMake Particle Stack: Normal termination\n");
        wxDateTime finish_time = wxDateTime::Now( );
        wxPrintf("Total Run Time : %s\n\n", finish_time.Subtract(start_time).Format("%Hh:%Mm:%Ss"));
    }

    return true;
}
