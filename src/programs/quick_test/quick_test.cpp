#include "../../core/core_headers.h"
#include "../../core/scattering_potential.h"

class
        QuickTestApp : public MyApp {

  public:
    bool     DoCalculation( );
    void     DoInteractiveUserInput( );
    wxString symmetry_symbol;
    bool     my_test_1 = false;
    bool     my_test_2 = true;

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

    Image   test_1;
    Image   test_2;
    Image   vol;
    MRCFile mrcf;
    mrcf.OpenFile("/cisTEMx/cistem_reference_images/ribo_ref.mrc", false);
    vol.ReadSlices(&mrcf, 1, mrcf.ReturnNumberOfSlices( ));
    Curve whitening_filter;
    Curve number_of_terms;
    whitening_filter.SetupXAxis(0.0, 0.5 * sqrtf(3.0), int((vol.logical_x_dimension / 2.0 + 1.0) * sqrtf(3.0) + 1.0));
    number_of_terms.SetupXAxis(0.0, 0.5 * sqrtf(3.0), int((vol.logical_x_dimension / 2.0 + 1.0) * sqrtf(3.0) + 1.0));

    // remove outliers
    // This won't work for movie frames (13.0 is used in unblur) TODO use poisson stats
    vol.ReplaceOutliersWithMean(5.0f);
    vol.ForwardFFT( );
    vol.SwapRealSpaceQuadrants( );

    vol.ZeroCentralPixel( );
    vol.Compute1DPowerSpectrumCurve(&whitening_filter, &number_of_terms);

    std::vector<int> S = {64,
                          128,
                          256,
                          512};
    CTF              ctf;
    ctf.Init(300, 2.7, 0.1, 12000, 12000, 0, 1.0, 0);
    double sum_neg = 0;
    double sum_pos = 0;
    int    n_pts   = 1000;
    float  val;
    float  freq;
    for ( int i = 0; i < whitening_filter.number_of_points; i++ ) {
        val = sqrtf(whitening_filter.data_y[i]) * ctf.Evaluate(pow(whitening_filter.data_x[i], 2), 0);
        if ( val < 0 ) {
            sum_neg += val;
        }
        else {
            sum_pos += val;
        }
    }
    std::cerr << "sum_neg = " << sum_neg << std::endl;
    std::cerr << "sum_pos = " << sum_pos << std::endl;
    std::cerr << "Ratio   = " << sum_neg / sum_pos << std::endl;

    // const long                        locs = pow(S[S.size( ) - 1], 3);
    // std::vector<std::array<float, 2>> full_results(4);
    // for ( int i = 0; i < full_results.size( ); i++ ) {
    //     full_results.at(i).at(0) = 0.0;
    //     full_results.at(i).at(1) = 0.0;
    // }
    // for ( int i = 0; i < full_results.size( ); i++ ) {
    //     for ( auto& s : S ) {
    //         test_1.Allocate(s, s, 1, true);
    //         test_2.Allocate(s, s, 1, true);
    //         std::vector<float> results(locs / (s * s));
    //         for ( long i = 0; i < results.size( ); i++ ) {
    //             test_1.FillWithNoiseFromNormalDistribution(0.0, 1.0);
    //             test_2.FillWithNoiseFromNormalDistribution(0.0, 1.0);
    //             test_1.MultiplyPixelWise(test_2);
    //             test_1.DivideByConstant(s);
    //             results[i] = test_1.ReturnSumOfRealValues( );
    //         }

    //         double sum  = 0.0;
    //         double sum2 = 0.0;
    //         for ( auto& r : results ) {
    //             sum += r;
    //             sum2 += r * r;
    //         }
    //         double mean = sum / results.size( );
    //         double var  = sum2 / results.size( ) - mean * mean;
    //         std::cerr << "s = " << s << " mean = " << mean << " std = " << sqrt(var) << std::endl;
    //         full_results.at(i).at(0) = mean;
    //         full_results.at(i).at(1) = sqrt(var);
    //     }
    //     double sum  = 0.0;
    //     double sum2 = 0.0;
    //     for ( auto& r : full_results ) {
    //         sum += r;
    //         sum2 += r * r;
    //     }
    //     double mean = sum / full_results.size( );
    //     double var  = sum2 / full_results.size( ) - mean * mean;
    //     std::cerr << "FULL = " << s << " mean = " << mean << " std = " << sqrt(var) << std::endl;
    // }

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
