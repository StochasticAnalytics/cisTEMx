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

    input_starfile_filename.at(0) = my_input->GetFilenameFromUser("Input starfile filename 1", "", "", true);
    input_starfile_filename.at(1) = my_input->GetFilenameFromUser("Input starfile filename 2", "", "", true);
    symmetry_symbol               = my_input->GetSymmetryFromUser("Particle symmetry", "The assumed symmetry of the particle to be reconstructed", "C1");

    delete my_input;
}

// override the do calculation method which will be what is actually run..

bool QuickTestApp::DoCalculation( ) {

    wxString pdb_name = "6tte_full.pdb";
    Image    test_sim;
    test_sim.Allocate(256, 256, 256, true, true);

    ScatteringPotential sp = ScatteringPotential(pdb_name, test_sim.logical_x_dimension);
    sp.SetImagingParameters(1.0, 300.0);
    sp.InitPdbObject(false);
    RotationMatrix rotate_waters;
    rotate_waters.SetToIdentity( );
    sp.calc_scattering_potential(test_sim, rotate_waters, 1);
    test_sim.QuickAndDirtyWriteSlices("test_sim.mrc", 1, 256);

    rotate_waters.SetToEulerRotation(0.0, 0.0, 90.0);
    sp.calc_scattering_potential(test_sim, rotate_waters, 1);
    test_sim.QuickAndDirtyWriteSlices("test_sim90.mrc", 1, 256);
    exit(0);

    std::array<cisTEMParameters, 2>
            star_file;

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