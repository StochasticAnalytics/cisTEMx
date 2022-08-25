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

    RotationMatrix my_rotation_matrix;
    RotationMatrix other_rotation_matrix;
    SymmetryMatrix my_symmetry_matrix;
    float          angular_error;

    float l_psi   = 100.f;
    float l_phi   = 0.f;
    float l_theta = 20.f;
    my_rotation_matrix.SetToEulerRotation(l_phi, l_theta, l_psi);
    my_symmetry_matrix.Init("C1");

    for ( int theta = -4; theta < 5; theta++ ) {
        float my_theta = float(theta * 5.f);
        other_rotation_matrix.SetToEulerRotation(l_phi, l_theta + my_theta, l_psi);
        angular_error = my_symmetry_matrix.GetMinimumAngularDistance(my_rotation_matrix, other_rotation_matrix);

        std::cerr << "dTheta: " << my_theta << " Angular error: " << angular_error << std::endl;
    }

    std::cerr << "Done" << std::endl;

    for ( int theta = -4; theta < 5; theta++ ) {
        float my_theta = float(theta * 5.f);
        other_rotation_matrix.SetToEulerRotation(l_phi + my_theta, l_theta, l_psi);
        angular_error = my_symmetry_matrix.GetMinimumAngularDistance(my_rotation_matrix, other_rotation_matrix);

        std::cerr << "dPhi: " << my_theta << " Angular error: " << angular_error << std::endl;
    }

    std::cerr << "Done" << std::endl;

    for ( int theta = -4; theta < 5; theta++ ) {
        float my_theta = float(theta * 5.f);
        other_rotation_matrix.SetToEulerRotation(l_phi, l_theta, l_psi + my_theta);
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
