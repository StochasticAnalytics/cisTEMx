#include "../../core/core_headers.h"
#include "../../core/scattering_potential.h"

class
        QuickTestApp : public MyApp {

  public:
    bool     DoCalculation( );
    void     DoInteractiveUserInput( );
    wxString symmetry_symbol;

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

    delete my_input;
}

// override the do calculation method which will be what is actually run..

bool QuickTestApp::DoCalculation( ) {

    MRCFile input_file_1("brafMex_1.0.mrc", false);
    Image   vol;
    vol.ReadSlices(&input_file_1, 1, input_file_1.ReturnNumberOfSlices( ));
    ParameterMap parameter_map; // needed for euler search init
    //for (int i = 0; i < 5; i++) {parameter_map[i] = true;}
    parameter_map.SetAllTrue( );
    EulerSearch     global_euler_search;
    AnglesAndShifts angles;
    float           angular_step                 = 1.5;
    float           wanted_phi                   = 0.0;
    float           wanted_theta                 = 0.0;
    float           psi_max                      = 360.f;
    float           psi_start                    = 0.0f;
    float           pixel_size                   = 1.2f;
    float           high_resolution_limit_search = pixel_size * 2.0f;
    float           mask_radius_search           = vol.logical_x_dimension / 2.0f * pixel_size;
    int             best_parameters_to_keep      = 20;
    parameter_map.phi                            = false;
    parameter_map.theta                          = false;

    float psi_step = rad_2_deg(pixel_size / mask_radius_search);
    psi_step       = 360.0 / int(360.0 / psi_step + 0.5);
    global_euler_search.InitGrid(symmetry_symbol, angular_step, wanted_phi, wanted_theta, psi_max, psi_step, psi_start,
                                 pixel_size / high_resolution_limit_search, parameter_map, best_parameters_to_keep);
    int first_search_position            = 0;
    int last_search_position             = global_euler_search.number_of_search_positions;
    int total_number_of_search_positions = 0;
    for ( int current_search_position = first_search_position; current_search_position < last_search_position; current_search_position++ ) {
        for ( int current_psi = psi_start; current_psi <= psi_max; current_psi += angular_step ) {
            total_number_of_search_positions++;
        }
    }
    std::cerr << "There are " << total_number_of_search_positions << " search positions" << std::endl;
    std::vector<float> search_results(total_number_of_search_positions);

    if ( symmetry_symbol.StartsWith("C") ) // TODO 2x check me - w/o this O symm at least is broken
    {
        if ( global_euler_search.test_mirror == true ) // otherwise the theta max is set to 90.0 and test_mirror is set to true.  However, I don't want to have to test the mirrors.
        {
            global_euler_search.theta_max = 180.0f;
        }
    }

    global_euler_search.CalculateGridSearchPositions(false);

    Image prj;
    prj.Allocate(2 * vol.logical_x_dimension, 2 * vol.logical_y_dimension, 1, false);
    Image other_prj;
    other_prj = prj;
    vol.ForwardFFT( );

    Image small_prj;
    small_prj.Allocate(vol.logical_x_dimension, vol.logical_y_dimension, 1, false);

    float  variance;
    Image* current_projection;
    Peak   found_peak;
    int    max_pix_x      = 20;
    int    max_pix_y      = 20;
    int    search_counter = 0;
    bool   first_time     = true;
    for ( int current_search_position = first_search_position; current_search_position < last_search_position; current_search_position++ ) {
        std::cerr << "working on search position " << current_search_position << "out of" << last_search_position << std::endl;
        for ( int current_psi = psi_start; current_psi <= psi_max; current_psi += angular_step ) {
            angles.Init(global_euler_search.list_of_search_parameters[current_search_position][0], global_euler_search.list_of_search_parameters[current_search_position][1], current_psi, 0.0, 0.0);

            if ( first_time ) {
                current_projection = &prj;
                first_time         = false;
            }
            else {
                current_projection = &other_prj;
            }
            small_prj.is_in_real_space = false;

            vol.ExtractSlice(small_prj, angles, 1.0f, false);

            small_prj.BackwardFFT( );
            small_prj.AddConstant(-small_prj.ReturnAverageOfRealValuesOnEdges( ));

            variance = small_prj.ReturnSumOfSquares( ) * current_projection->number_of_real_space_pixels / current_projection->number_of_real_space_pixels -
                       powf(small_prj.ReturnAverageOfRealValues( ) *
                                    current_projection->number_of_real_space_pixels / current_projection->number_of_real_space_pixels,
                            2);
            small_prj.DivideByConstant(sqrtf(variance));

            small_prj.ClipIntoLargerRealSpace2D(current_projection);
            current_projection->ForwardFFT( );
            // Zeroing the central pixel is probably not doing anything useful...
            current_projection->ZeroCentralPixel( );
            //                    padded_reference.DivideByConstant(sqrtf(variance));

            if ( first_time ) {
                other_prj.CopyFrom(current_projection);
                current_projection = &other_prj;
            }

            // Use the MKL
            vmcMulByConj(prj.real_memory_allocated / 2, reinterpret_cast<MKL_Complex8*>(current_projection->complex_values), reinterpret_cast<MKL_Complex8*>(prj.complex_values), reinterpret_cast<MKL_Complex8*>(current_projection->complex_values), VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);

            current_projection->BackwardFFT( );
            current_projection->is_in_real_space = true;

            found_peak = current_projection->FindPeakAtOriginFast2D(max_pix_x, max_pix_y);
            // std::cerr << "For phi theta psi " << angles.ReturnPhiAngle( ) << " " << angles.ReturnThetaAngle( ) << " " << angles.ReturnPsiAngle( ) << " found peak at " << found_peak.x << " " << found_peak.y << " " << found_peak.value << std::endl;
            search_results.at(search_counter) = found_peak.value;
            if ( found_peak.value > 2.0 ) {
                current_projection->QuickAndDirtyWriteSlice("test.mrc", 1);
                exit(1);
            }
            search_counter++;
        }
    }
    NumericTextFile output_file("search_results.txt", OPEN_TO_WRITE, 1);
    for ( int i = 0; i < search_results.size( ); i++ ) {
        output_file.WriteLine(search_results.data( ) + i);
    }
    output_file.Close( );
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
