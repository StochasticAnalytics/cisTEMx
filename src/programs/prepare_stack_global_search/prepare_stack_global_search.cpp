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
    float pixel_size                   = 1;
    float voltage_kV                   = 300.0;
    float spherical_aberration_mm      = 2.7;
    float amplitude_contrast           = 0.07;
    float average_defocus_1            = 5000.0;
    float average_defocus_2            = 5000.0;
    float average_defocus_angle        = 0.0;
    int   box_size                     = 256;
    int   number_of_results_to_process = 1;
    int   mip_x_dimension              = 0;
    int   mip_y_dimension              = 0;
    bool  read_coordinates;

    UserInput* my_input = new UserInput("MakeParticleStack", 1.00);

    read_coordinates = my_input->GetYesNoFromUser("Read coordinates from starfile?", "Should the target coordinates be read from a starfile instead of search results?", "No");
    if ( read_coordinates ) {
        input_starfilename = my_input->GetFilenameFromUser("Input starfile", "The file containing information on the targets", "particle_stack.star", true);
        wanted_threshold   = my_input->GetFloatFromUser("Peak threshold", "Peaks over this size will be taken", "7.5", 0.0);
        // Currently not used for read_coordinates, but should be used to remove duplicate particles.
        // For an initial implementation, use "best 2d class to store the image number"
        min_peak_radius              = my_input->GetFloatFromUser("Min Peak Radius (px.)", "Essentially the minimum closeness for peaks", "10.0", 1.0);
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
        min_peak_radius              = my_input->GetFloatFromUser("Min Peak Radius (px.)", "Essentially the minimum closeness for peaks", "10.0", 1.0);
        number_of_results_to_process = my_input->GetIntFromUser("Number of results to process (0=all)", "If input files contain results from several searches, how many to include?", "0", 0);
    }
    input_image_filename           = my_input->GetFilenameFromUser("Input image file", "The image that was searched", "image.mrc", true);
    output_star_filename           = my_input->GetFilenameFromUser("Output star file", "The star file containing the particle alignment parameters", "particle_stack.star", false);
    output_particle_stack_filename = my_input->GetFilenameFromUser("Output particle stack", "The output image stack, containing the picked particles", "particle_stack.mrc", false);
    box_size                       = my_input->GetIntFromUser("Box size for particles (px.)", "The pixel dimensions of the box used to cut out the particles", "256", 10);
    if ( ! read_coordinates ) {
        // We'll grab these from the starfile
        pixel_size              = my_input->GetFloatFromUser("Pixel size of image (A)", "Pixel size of input image in Angstroms", "1.0", 0.0);
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
                                      mip_x_dimension, // 20
                                      mip_y_dimension, // 21
                                      number_of_results_to_process); // 22
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
    int      mip_x_dimension                = my_current_job.arguments[20].ReturnIntegerArgument( );
    int      mip_y_dimension                = my_current_job.arguments[21].ReturnIntegerArgument( );
    int      number_of_results_to_process   = my_current_job.arguments[22].ReturnIntegerArgument( );

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
    float sq_dist_x, sq_dist_y;
    float micrograph_mean;
    float variance;
    long  address;
    long  text_file_access_type;
    int   i, j;

    cisTEMParameters    input_star_file;
    cisTEMParameterLine output_parameters;
    cisTEMParameters    output_star_file;

    float min_peak_radius_squared = min_peak_radius * min_peak_radius;
    std::cerr << "min_peak_radius_squared = " << min_peak_radius_squared << std::endl;
    if ( ! read_coordinates ) {

        MRCFile input_mip_file(input_mip_filename.ToStdString( ), false);
        MRCFile input_psi_file(input_best_psi_filename.ToStdString( ), false);
        MRCFile input_theta_file(input_best_theta_filename.ToStdString( ), false);
        MRCFile input_phi_file(input_best_phi_filename.ToStdString( ), false);
        // Make sure the number of results to process is not greater than the number of particles in the input file
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
        mip_x_dimension = mip_image.logical_x_dimension;
        mip_y_dimension = mip_image.logical_y_dimension;
    }
    std::vector<Peak>   peaks_found(number_of_results_to_process);
    std::vector<double> avg_occupancy(number_of_results_to_process, 0.f);
    for ( i = 0; i < number_of_results_to_process; i++ ) {
        peaks_found[i].value = -std::numeric_limits<float>::max( );
    }

    if ( read_coordinates ) {
        input_star_file.ReadFromcisTEMStarFile(input_starfilename.ToStdString( ), false);

        // For the time being, we'll only take the top scoring paricle from each group
        output_star_file = input_star_file.ReturnTopNFromParticleGroups(1);
    }
    else {
        std::cerr << "Processing : " << number_of_results_to_process << " results " << std::endl;
        output_star_file.PreallocateMemoryAndBlank(cistem::maximum_number_of_detections * number_of_results_to_process);
        MyDebugAssertTrue(mip_image.HasSameDimensionsAs(&psi_image), "MIP and psi images have different dimensions");
        MyDebugAssertTrue(mip_image.HasSameDimensionsAs(&theta_image), "MIP and theta images have different dimensions");
        MyDebugAssertTrue(mip_image.HasSameDimensionsAs(&phi_image), "MIP and phi images have different dimensions");
    }

    micrograph.QuickAndDirtyReadSlice(input_image_filename.ToStdString( ), 1);
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
        mip_to_search.Allocate(mip_image.logical_x_dimension, mip_image.logical_y_dimension, true);
        for ( int iPixel = 0; iPixel < mip_to_search.real_memory_allocated; iPixel++ ) {
            // TODO: could use a memcpy here.
            mip_to_search.real_values[iPixel] = mip_image.real_values[iPixel];
        }
    }

    wxPrintf("\n");
    while ( 1 == 1 ) {
        if ( ! read_coordinates ) {
            // look for a peak..

            current_peak = mip_to_search.FindPeakWithIntegerCoordinates(0.0, FLT_MAX, box_size / cistem::fraction_of_box_size_to_exclude_for_border + 1);

            // current_peak = mip_image.FindPeakWithIntegerCoordinates(0.0, FLT_MAX, box_size / cistem::fraction_of_box_size_to_exclude_for_border + 1);
            if ( current_peak.value < wanted_threshold )
                break;

            // ok we have peak..

            // get angles and mask out the local area so it won't be picked again..

            current_peak.x = current_peak.x + mip_to_search.physical_address_of_box_center_x;
            current_peak.y = current_peak.y + mip_to_search.physical_address_of_box_center_y;
            address        = myroundint(current_peak.y) * (mip_to_search.logical_x_dimension + mip_to_search.padding_jump_value) + myroundint(current_peak.x);

            peaks_found[0].x                             = current_peak.x;
            peaks_found[0].y                             = current_peak.y;
            peaks_found[0].value                         = current_peak.value;
            peaks_found[0].physical_address_within_image = address;

            address = 0;
            for ( int iResult = 1; iResult < number_of_results_to_process; iResult++ ) {
                peaks_found[iResult].value = -std::numeric_limits<float>::max( );
            }

            float sq_dist_x, sq_dist_y;
            for ( int j = std::max(myroundint(current_peak.y - min_peak_radius), 0); j < std::min(myroundint(current_peak.y + min_peak_radius), mip_to_search.logical_y_dimension); j++ ) {
                sq_dist_y = float(j) - current_peak.y;
                sq_dist_y *= sq_dist_y;

                for ( int i = std::max(myroundint(current_peak.x - min_peak_radius), 0); i < std::min(myroundint(current_peak.x + min_peak_radius), mip_to_search.logical_x_dimension); i++ ) {
                    sq_dist_x = float(i) - current_peak.x;
                    sq_dist_x *= sq_dist_x;
                    address = mip_to_search.ReturnReal1DAddressFromPhysicalCoord(i, j, 0);

                    if ( sq_dist_x + sq_dist_y <= min_peak_radius_squared ) {
                        mip_to_search.real_values[address] = -std::numeric_limits<float>::max( );
                        mip_image.real_values[address]     = -std::numeric_limits<float>::max( );
                        // Check the other results if the exist
                        for ( int iResult = 1; iResult < number_of_results_to_process; iResult++ ) {
                            address += mip_to_search.real_memory_allocated;
                            if ( mip_image.real_values[address] > peaks_found[iResult].value ) {
                                peaks_found[iResult].x                             = i;
                                peaks_found[iResult].y                             = j;
                                peaks_found[iResult].value                         = mip_image.real_values[address];
                                peaks_found[iResult].physical_address_within_image = address;
                            }
                            mip_image.real_values[address] = -std::numeric_limits<float>::max( );
                        }
                    }
                }
            }

            for ( int iResult = 0; iResult < number_of_results_to_process; iResult++ ) {
                // We don't want to include extra peaks that are below the threshold
                // In reconstruct3d the weight is determined by occupancy / average occupancy, so to make the occupancy
                // simply fractional, ensure the sum of the occupancy for each particle = N particles, st. the average = 1

                output_parameters.SetAllToZero( );
                output_parameters.position_in_stack                  = number_of_peaks_found + 1;
                output_parameters.psi                                = psi_image.real_values[peaks_found[iResult].physical_address_within_image];
                output_parameters.theta                              = theta_image.real_values[peaks_found[iResult].physical_address_within_image];
                output_parameters.phi                                = phi_image.real_values[peaks_found[iResult].physical_address_within_image];
                output_parameters.x_shift                            = peaks_found[iResult].x * pixel_size; // FIXME: if saving a particle stack this is no longer valid, need option for both
                output_parameters.y_shift                            = peaks_found[iResult].y * pixel_size;
                output_parameters.defocus_1                          = average_defocus_1 + current_defocus;
                output_parameters.defocus_2                          = average_defocus_2 + current_defocus;
                output_parameters.defocus_angle                      = average_defocus_angle;
                output_parameters.pixel_size                         = pixel_size;
                output_parameters.microscope_voltage_kv              = voltage_kV;
                output_parameters.microscope_spherical_aberration_mm = spherical_aberration_mm;
                output_parameters.amplitude_contrast                 = amplitude_contrast;
                output_parameters.occupancy                          = (peaks_found[iResult].value > wanted_threshold) ? 1.0f : 0.f; // for now, include all peaks, even if below threshold and use occupancy to turn off
                output_parameters.sigma                              = 10.0f;
                output_parameters.logp                               = 5000.0f;
                output_parameters.score                              = peaks_found[iResult].value;
                output_parameters.image_is_active                    = 1;
                output_parameters.particle_group                     = particle_group + 1;
                output_parameters.assigned_subset                    = IsOdd(particle_group) ? 1 : 2; // make sure that different possibilites for a given particle are in the same FSC subset

                output_star_file.all_parameters.Item(number_of_peaks_found) = output_parameters;
                number_of_peaks_found++;
            }
            particle_group++;
        }
        else {

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

            output_star_file.all_parameters.Item(n_processed_in_total).position_in_stack = n_processed_in_total + 1;
            // For now, we'll also still include inactive images in the output star file and hence the output stack
            // lastly post increment the counter on peaks found

            output_star_file.all_parameters.Item(n_processed_in_total) = output_parameters;

            number_of_peaks_found++;
        }

        // wxPrintf("Peak %4i at x, y, psi, theta, phi, defocus, pixel size = %12.6f, %12.6f, %12.6f, %12.6f, %12.6f, %12.6f, %12.6f : %10.6f\n", number_of_peaks_found, current_peak.x * pixel_size, current_peak.y * pixel_size, current_psi, current_theta, current_phi, current_defocus, current_pixel_size, current_peak.value);

        if ( read_coordinates ) {
            micrograph.ClipInto(&current_particle, micrograph_mean, false, 1.0,
                                myroundint(current_peak.x - micrograph.physical_address_of_box_center_x),
                                myroundint(current_peak.y - micrograph.physical_address_of_box_center_y), 0);
            //		micrograph.ClipInto(&current_particle, micrograph_mean, false, 1.0, int(current_peak.x * pixel_size), int(current_peak.y * pixel_size), 0);
            //		micrograph.ClipInto(&current_particle, micrograph_mean, false, 1.0, int(- current_peak.x * pixel_size + current_particle.physical_address_of_box_center_x), int(- current_peak.y * pixel_size + current_particle.physical_address_of_box_center_y), 0);
            variance = current_particle.ReturnVarianceOfRealValues( );
            if ( variance == 0.0f )
                variance = 1.0f;
            current_particle.AddMultiplyConstant(-current_particle.ReturnAverageOfRealValuesOnEdges( ), 1.0f / sqrtf(variance));

            current_particle.WriteSlice(&output_file, number_of_peaks_found);
        }

        if ( read_coordinates && output_star_file.ReturnNumberofLines( ) == number_of_peaks_found )
            break;
    }

    // mip_to_search.QuickAndDirtyWriteSlice("mip_to_search.mrc", 1);
    // mip_image.QuickAndDirtyWriteSlices("mip_image.mrc", 1, mip_image.logical_z_dimension);
    // exit(1);

    if ( read_coordinates ) {
        output_file.CloseFile( );
    }

    // By using SetAllToZero, we inadvertently activate some parameters that aren't necessary.
    // The filenames in particular make the star file harder to read and for auto_functionality are not necessary.
    output_star_file.parameters_to_write.stack_filename          = false;
    output_star_file.parameters_to_write.original_image_filename = false;
    output_star_file.parameters_to_write.reference_3d_filename   = false;
    output_star_file.parameters_to_write.best_2d_class           = false;

    output_star_file.WriteTocisTEMStarFile(output_star_filename, -1, -1, 1, number_of_peaks_found);

    if ( is_running_locally == true ) {
        wxPrintf("\nFound %i peaks.\n\n", number_of_peaks_found);
        wxPrintf("\nMake Particle Stack: Normal termination\n");
        wxDateTime finish_time = wxDateTime::Now( );
        wxPrintf("Total Run Time : %s\n\n", finish_time.Subtract(start_time).Format("%Hh:%Mm:%Ss"));
    }

    return true;
}
