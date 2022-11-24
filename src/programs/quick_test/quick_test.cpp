#include "../../core/core_headers.h"
#include "../../core/scattering_potential.h"

class
        QuickTestApp : public MyApp {

  public:
    bool     DoCalculation( );
    void     DoInteractiveUserInput( );
    wxString symmetry_symbol;
    bool     my_test_1    = false;
    bool     my_test_2    = true;
    float    angular_step = 1.0f;
    float    psi_step     = 1.0f;

    std::array<wxString, 2> input_starfile_filename;

  private:
};

IMPLEMENT_APP(QuickTestApp)

// override the DoInteractiveUserInput

void QuickTestApp::DoInteractiveUserInput( ) {

    UserInput* my_input = new UserInput("Unblur", 2.0);

    input_starfile_filename.at(0) = my_input->GetFilenameFromUser("Input starfile filename 1", "", "", false);
    input_starfile_filename.at(1) = my_input->GetFilenameFromUser("Input starfile filename 2", "", "", false);
    symmetry_symbol               = my_input->GetSymmetryFromUser("Particle symmetry", "The assumed symmetry of the particle to be reconstructed", "C1");
    angular_step                  = my_input->GetFloatFromUser("Angular step size", "The angular step size for the grid search", "1.0", 0.0);
    psi_step                      = my_input->GetFloatFromUser("Psi step size", "The psi step size for the grid search", "1.0", 0.0);

    delete my_input;
}

// override the do calculation method which will be what is actually run..

bool QuickTestApp::DoCalculation( ) {

    MRCFile input_file_1(input_starfile_filename.at(0).ToStdString( ), false);
    Image   vol;
    vol.ReadSlices(&input_file_1, 1, input_file_1.ReturnNumberOfSlices( ));
    ParameterMap parameter_map; // needed for euler search init
    //for (int i = 0; i < 5; i++) {parameter_map[i] = true;}
    parameter_map.SetAllTrue( );
    EulerSearch     global_euler_search;
    AnglesAndShifts angles;

    float wanted_phi                   = 0.0;
    float wanted_theta                 = 0.0;
    float psi_max                      = 360.f;
    float psi_start                    = 0.0f;
    float pixel_size                   = 1.2f;
    float high_resolution_limit_search = pixel_size * 2.0f;
    float mask_radius_search           = vol.logical_x_dimension / 2.0f * pixel_size;
    int   best_parameters_to_keep      = 20;
    parameter_map.phi                  = true;
    parameter_map.theta                = true;

    global_euler_search.InitGrid(symmetry_symbol, angular_step, wanted_phi, wanted_theta, psi_max, psi_step, psi_start,
                                 pixel_size / high_resolution_limit_search, parameter_map, best_parameters_to_keep);

    if ( symmetry_symbol.StartsWith("C") ) // TODO 2x check me - w/o this O symm at least is broken
    {
        if ( global_euler_search.test_mirror == true ) // otherwise the theta max is set to 90.0 and test_mirror is set to true.  However, I don't want to have to test the mirrors.
        {
            global_euler_search.theta_max = 180.0f;
        }
    }

    global_euler_search.CalculateGridSearchPositions(false);

    int first_search_position            = 0;
    int last_search_position             = global_euler_search.number_of_search_positions;
    int total_number_of_search_positions = 0;

    for ( int current_search_position = first_search_position; current_search_position < last_search_position; current_search_position++ ) {
        for ( int current_psi = psi_start; current_psi < psi_max; current_psi += psi_step ) {
            total_number_of_search_positions++;
        }
    }
    std::vector<std::array<float, 3>> angle_list(total_number_of_search_positions);
    total_number_of_search_positions = 0;
    for ( int current_search_position = first_search_position; current_search_position < last_search_position; current_search_position++ ) {
        for ( int current_psi = psi_start; current_psi < psi_max; current_psi += psi_step ) {
            angle_list.at(total_number_of_search_positions).at(0) = global_euler_search.list_of_search_parameters[current_search_position][0];
            angle_list.at(total_number_of_search_positions).at(1) = global_euler_search.list_of_search_parameters[current_search_position][1];
            angle_list.at(total_number_of_search_positions).at(2) = current_psi;
            total_number_of_search_positions++;
        }
    }

    std::cerr << "There are " << total_number_of_search_positions << " search positions" << std::endl;

    // We store a sparse array, encoding the angular parameters in the filename, then instead of an N x N ccc matrix, store
    // just the lower left triangular excluding the diagonal, which is trivial so we have a
    // N-1 + N-2 + N-3 + ... + 1 = N(N-1)/2 elements
    // For convenience, we'll store these in a sqrt(N(N-1)/2) x sqrt(N(N-1)/2) matrix, and then we can use the
    // the image class to save as half precision
    int output_size = 0;
    for ( int outer_counter = 0; outer_counter < angle_list.size( ); outer_counter++ ) {
        for ( int current_search_position = first_search_position + outer_counter; current_search_position < angle_list.size( ); current_search_position++ ) {
            output_size++;
        }
    }
    int total_number_to_record = output_size;
    output_size                = myroundint(0.5f + sqrt(output_size));
    std::cerr << "Output size is " << output_size << std::endl;

    Image ccc_matrix;
    ccc_matrix.Allocate(output_size, output_size, 1);

    // std::vector<std::array<float, 4>> search_results(total_number_of_search_positions);

    Image ref_prj;
    ref_prj.Allocate(2 * vol.logical_x_dimension, 2 * vol.logical_y_dimension, 1, false);
    Image test_prj;
    test_prj = ref_prj;
    vol.ForwardFFT( );
    vol.SwapRealSpaceQuadrants( );

    Image small_prj;
    small_prj.Allocate(vol.logical_x_dimension, vol.logical_y_dimension, 1, false);

    float  variance;
    Image* current_projection;
    Peak   found_peak;
    int    max_pix_x      = 20;
    int    max_pix_y      = 20;
    int    search_counter = 0;
    bool   first_time     = true;

    long number_angles_searched            = 0;
    int  number_search_positions_completed = 0;
    int  n_x                               = 0;
    long address                           = 0;
    int  printed_complete                  = 0;
    // Saving only the lower triangular so we need one less value each time, use the outer loop to decrement the starting point of the inner loop
    for ( int outer_counter = 0; outer_counter < angle_list.size( ); outer_counter++ ) {
        first_time = true;
        for ( int current_search_position = first_search_position + outer_counter; current_search_position < angle_list.size( ); current_search_position++ ) {
            angles.Init(angle_list.at(current_search_position).at(0),
                        angle_list.at(current_search_position).at(1),
                        angle_list.at(current_search_position).at(2),
                        0.0f, 0.0f);

            if ( first_time ) {
                current_projection = &ref_prj;
            }
            else {
                current_projection = &test_prj;
            }
            small_prj.is_in_real_space = false;

            vol.ExtractSlice(small_prj, angles, 1.0f, false);
            small_prj.CosineMaskAndNormalizeInPassBand(400, 4, 1, small_prj.number_of_real_space_pixels * 3);

            small_prj.SwapRealSpaceQuadrants( );
            small_prj.BackwardFFT( );

            float average_on_edge  = small_prj.ReturnAverageOfRealValuesOnEdges( );
            float average_of_reals = small_prj.ReturnAverageOfRealValues( ) - average_on_edge;

            small_prj.AddConstant(-average_on_edge);

            // The average in the full padded image will be different;
            average_of_reals *= small_prj.number_of_real_space_pixels / current_projection->number_of_real_space_pixels;

            small_prj.DivideByConstant(sqrtf(small_prj.ReturnSumOfSquares( ) * small_prj.number_of_real_space_pixels / current_projection->number_of_real_space_pixels - average_of_reals * average_of_reals));

            small_prj.ClipInto(current_projection);
            current_projection->ForwardFFT( );
            // Zeroing the central pixel is probably not doing anything useful...
            current_projection->ZeroCentralPixel( );

            //                    padded_reference.DivideByConstant(sqrtf(variance));

            if ( first_time ) {
                test_prj.CopyFrom(current_projection);
                current_projection = &test_prj;
            }

            // Use the MKL
            vmcMulByConj(ref_prj.real_memory_allocated / 2, reinterpret_cast<MKL_Complex8*>(current_projection->complex_values), reinterpret_cast<MKL_Complex8*>(ref_prj.complex_values), reinterpret_cast<MKL_Complex8*>(current_projection->complex_values), VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);

            current_projection->BackwardFFT( );
            current_projection->is_in_real_space         = true;
            current_projection->object_is_centred_in_box = false;
            found_peak                                   = current_projection->FindPeakAtOriginFast2D(max_pix_x, max_pix_y);
            // std::cerr << "For phi theta psi " << angles.ReturnPhiAngle( ) << " " << angles.ReturnThetaAngle( ) << " " << angles.ReturnPsiAngle( ) << " found peak at " << found_peak.x << " " << found_peak.y << " " << found_peak.value << std::endl;
            // search_results.at(search_counter)[0] = found_peak.value;
            // search_results.at(search_counter)[1] = angles.ReturnPhiAngle( );
            // search_results.at(search_counter)[2] = angles.ReturnThetaAngle( );
            // search_results.at(search_counter)[3] = angles.ReturnPsiAngle( );
            if ( found_peak.value > 2.0 ) {
                current_projection->QuickAndDirtyWriteSlice("test.mrc", 1);
                exit(1);
            }
            search_counter++;
            if ( first_time ) {

                first_time = false;
            }
            else {
                // We don't care about the autocorrlation value (the diagn)
                ccc_matrix.real_values[address] = found_peak.value;
                n_x++;
                if ( n_x == ccc_matrix.logical_x_dimension ) {
                    n_x = 0;
                    address += ccc_matrix.padding_jump_value;
                }
                else
                    address++;
                number_angles_searched++;

                float percent_complete = 100.0 * float(number_angles_searched) / float(total_number_to_record);
                if ( int(percent_complete) % 2 == 0 && int(percent_complete) > printed_complete ) {
                    std::cerr << "percent complete: " << percent_complete << std::endl;
                    printed_complete = int(percent_complete);
                }
            }
        } // loop comparing i to j
    } // loop over h

    MRCFile output_file;
    output_file.OpenFile(input_starfile_filename.at(1).ToStdString( ), false);
    output_file.SetOutputToFP16( );
    ccc_matrix.WriteSlice(&output_file, 1);
    output_file.CloseFile( );
    // NumericTextFile output_file("search_results.txt", OPEN_TO_WRITE, 4);
    // for ( int i = 0; i < search_results.size( ); i++ ) {
    //     output_file.WriteLine(search_results.at(i).data( ));
    // }
    // output_file.Close( );
    exit(0);

    std::cerr << "my_test 1 = " << my_test_1 << std::endl;
    std::cerr << "my_test 2 = " << my_test_2 << std::endl;
    exit(0);
    wxPrintf("%20.100f\n", std::numeric_limits<float>::epsilon( ));
    wxPrintf("%20.100f\n", sqrt(std::numeric_limits<float>::epsilon( )));
    wxPrintf("%20.100f\n", sqrt(sqrt(std::numeric_limits<float>::epsilon( ))));

    wxPrintf("%20.100f\n", std::numeric_limits<double>::epsilon( ));
    wxPrintf("%20.100f\n", sqrt(std::numeric_limits<double>::epsilon( )));
    wxPrintf("%20.100f\n", sqrt(sqrt(std::numeric_limits<double>::epsilon( ))));
    exit(0);

    float bfactor;
    vol.QuickAndDirtyReadSlices("no_bfactor.mrc", 1, 192);
    bfactor = vol.CalculateBFactor(1.0f);
    std::cerr << "bfactor = " << bfactor << std::endl;

    vol.QuickAndDirtyReadSlices("no_bfactor.mrc", 1, 192);
    vol.ForwardFFT( );
    vol.ApplyBFactor(100.f);
    bfactor = vol.CalculateBFactor(1.0f);
    std::cerr << "bfactor = " << bfactor << std::endl;

    vol.QuickAndDirtyReadSlices("no_bfactor.mrc", 1, 192);
    vol.ForwardFFT( );
    vol.ApplyBFactor(500.f);
    bfactor = vol.CalculateBFactor(1.0f);
    std::cerr << "bfactor = " << bfactor << std::endl;

    vol.QuickAndDirtyReadSlices("bfactor_100.mrc", 1, 192);
    bfactor = vol.CalculateBFactor(1.0f);
    std::cerr << "bfactor 100 = " << bfactor << std::endl;

    vol.QuickAndDirtyReadSlices("bfactor_500.mrc", 1, 192);
    bfactor = vol.CalculateBFactor(1.0f);
    std::cerr << "bfactor 500 = " << bfactor << std::endl;

    exit(1);
    MRCFile my_file;
    my_file.OpenFile(input_starfile_filename.at(0).ToStdString( ), false);

    exit(0);
    RotationMatrix my_rotation_matrix;
    RotationMatrix other_rotation_matrix;
    SymmetryMatrix my_symmetry_matrix;
    float          angular_error;

    float l_psi   = 100.f;
    float l_phi   = -500.f;
    float l_theta = 20.f;
    my_rotation_matrix.SetToEulerRotation(-l_psi, -l_theta, -l_phi);
    my_symmetry_matrix.Init("C1");

    ParameterMap my_dummy_map;
    EulerSearch  my_angle_restrictions;
    my_angle_restrictions.InitGrid(symmetry_symbol, 0.5, 0., 0., 360, 1.5, 0., 1.0f * 2, my_dummy_map, 5);
    NumericTextFile my_results_file("results.txt", OPEN_TO_WRITE, 3);
    float           r[3];
    for ( int i = 0; i < 10000; i++ ) {
        my_angle_restrictions.GetRandomEulerAngles(l_phi, l_theta, l_psi);
        my_rotation_matrix.SetToEulerRotation(l_phi, l_theta, 0);

        my_rotation_matrix.RotateCoords(0.f, 0.f, 1.f, r[0], r[1], r[2]);
        my_results_file.WriteLine(r);
    }
    exit(1);
    for ( int theta = -4; theta < 5; theta++ ) {
        float my_theta = float(theta * 5.f);
        other_rotation_matrix.SetToEulerRotation(-l_psi, -l_theta + my_theta, -l_phi);
        angular_error = my_symmetry_matrix.GetMinimumAngularDistance(my_rotation_matrix, other_rotation_matrix);

        std::cerr << "dTheta: " << my_theta << " Angular error: " << angular_error << std::endl;
    }

    std::cerr << "Done" << std::endl;

    for ( int theta = -4; theta < 5; theta++ ) {
        float my_theta = float(theta * 5.f);
        other_rotation_matrix.SetToEulerRotation(-l_psi + my_theta, -l_theta, -l_phi);
        angular_error = my_symmetry_matrix.GetMinimumAngularDistance(my_rotation_matrix, other_rotation_matrix);

        std::cerr << "dPhi: " << my_theta << " Angular error: " << angular_error << std::endl;
    }

    std::cerr << "Done" << std::endl;

    for ( int theta = -4; theta < 5; theta++ ) {
        float my_theta = float(theta * 5.f);
        other_rotation_matrix.SetToEulerRotation(-l_psi, -l_theta, -l_phi + my_theta);
        angular_error = my_symmetry_matrix.GetMinimumAngularDistance(my_rotation_matrix, other_rotation_matrix);

        std::cerr << "dPsi: " << my_theta << " Angular error: " << angular_error << std::endl;
    }

    exit(0);

    std::array<cisTEMParameters, 2> star_file;

    for ( int i = 0; i < star_file.size( ); i++ ) {
        // Check to see if we have a text or binary star file.
        wxFileName star_filename(input_starfile_filename.at(i));
        if ( star_filename.GetExt( ) == "cistem" )
            star_file.at(i).ReadFromcisTEMBinaryFile(input_starfile_filename.at(i));
        else
            star_file.at(i).ReadFromcisTEMStarFile(input_starfile_filename.at(i));
    }

    cisTEMParameterLine averages  = star_file.at(0).ReturnParameterAverages(star_file.at(1), symmetry_symbol);
    cisTEMParameterLine variances = star_file.at(0).ReturnParameterVariances(star_file.at(1), symmetry_symbol);

    wxPrintf("\n\n");
    wxPrintf("Angular distance axis: %f %f\n", averages.psi, sqrtf(variances.psi));
    wxPrintf("Angular distance inplane: %f %f\n", averages.phi, sqrtf(variances.phi));
    wxPrintf("X shift distance: %f %f\n", averages.x_shift, sqrtf(variances.x_shift));
    wxPrintf("Y shift distance: %f %f\n", averages.y_shift, sqrtf(variances.y_shift));

    return true;
}
