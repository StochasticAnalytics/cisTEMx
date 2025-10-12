#include <cistem_config.h>

#include "../../core/core_headers.h"
#include "../../constants/constants.h"

#ifdef ENABLEGPU
#include "../../gpu/gpu_core_headers.h"
#include "../../gpu/DeviceManager.h"
#include "../../gpu/TemplateMatchingCore.h"
#include "quick_test_gpu.h"
#else
#include "../../core/core_headers.h"
#endif

#include "../../constants/constants.h"

#include "../../core/scattering_potential.h"

class
        QuickTestApp : public MyApp {

  public:
    bool     DoCalculation( );
    void     DoInteractiveUserInput( );
    void     AddCommandLineOptions( );
    wxString symmetry_symbol;
    bool     my_test_1 = false;
    bool     my_test_2 = true;
    int      idx;

    std::array<wxString, 2> input_starfile_filename;

  private:
};

IMPLEMENT_APP(QuickTestApp)

// Optional command-line stuff
void QuickTestApp::AddCommandLineOptions( ) {
    command_line_parser.AddLongSwitch("disable-user-input", "Disable interactive user input prompts. Default false");
}

// override the DoInteractiveUserInput

void QuickTestApp::DoInteractiveUserInput( ) {
    // This flag allows skipping interactive prompts, useful for automated testing with Copilot.
    if ( command_line_parser.FoundSwitch("disable-user-input") ) {
        std::cout << "Skipping interactive user input as per command line flag." << std::endl;
        return;
    }
    UserInput* my_input = new UserInput("QuickTest", 2.0);

    idx                           = my_input->GetIntFromUser("Index", "", "", 0, 1000);
    input_starfile_filename.at(0) = my_input->GetFilenameFromUser("Input starfile filename 1", "", "", false);
    input_starfile_filename.at(1) = my_input->GetFilenameFromUser("Input starfile filename 2", "", "", false);
    symmetry_symbol               = my_input->GetSymmetryFromUser("Particle symmetry", "The assumed symmetry of the particle to be reconstructed", "C1");

    delete my_input;
}

bool QuickTestApp::DoCalculation( ) {

#ifdef ENABLEGPU
    // DeviceManager gpuDev;
    // gpuDev.ListDevices( );

    // QuickTestGPU quick_test_gpu;
    // quick_test_gpu.callHelloFromGPU(idx);
#endif

    // revert - FastFFT crash test for apoferritin template matching
    wxPrintf("\n=== Testing match_template with FastFFT ===\n");

    // Create temp directory for output
    wxString temp_dir  = "/tmp/cistem_fastfft_test";
    wxString mkdir_cmd = wxString::Format("mkdir -p %s", temp_dir);
    system(mkdir_cmd.c_str( ));

    // Apoferritin test data paths
    wxString template_file = input_starfile_filename.at(0) + "/cistem_reference_images/TM_tests/SPA/Apoferritin/Templates/apo_ref_no_C_T_480.mrc";
    wxString search_image  = input_starfile_filename.at(0) + "/cistem_reference_images/TM_tests/SPA/Apoferritin/Images/apoferritin_1600.mrc";

    // Output file paths
    wxString mip_output        = wxString::Format("%s/mip.mrc", temp_dir);
    wxString scaled_mip_output = wxString::Format("%s/mip_scaled.mrc", temp_dir);
    wxString psi_output        = wxString::Format("%s/psi.mrc", temp_dir);
    wxString theta_output      = wxString::Format("%s/theta.mrc", temp_dir);
    wxString phi_output        = wxString::Format("%s/phi.mrc", temp_dir);
    wxString defocus_output    = wxString::Format("%s/defocus.mrc", temp_dir);
    wxString pixel_size_output = wxString::Format("%s/pixel_size.mrc", temp_dir);
    wxString corr_avg_output   = wxString::Format("%s/corr_average.mrc", temp_dir);
    wxString corr_std_output   = wxString::Format("%s/corr_stddev.mrc", temp_dir);
    wxString histogram_output  = wxString::Format("%s/histogram.txt", temp_dir);
    wxString results_dir       = wxString::Format("%s/results", temp_dir);
    wxString result_file       = wxString::Format("%s/result.txt", temp_dir);

    // Create heredoc input for match_template_gpu with FastFFT enabled
    // Parameter order from match_template.cpp:236-376 DoInteractiveUserInput
    float    wanted_pixel_size    = 0.7896f;
    wxString match_template_input = wxString::Format(
            "%s\n" // 1. input_search_images
            "%s\n" // 2. input_reconstruction (template)
            "%s\n" // 3. mip_output_file
            "%s\n" // 4. scaled_mip_output_mrcfile
            "%s\n" // 5. best_psi_output_file
            "%s\n" // 6. best_theta_output_file
            "%s\n" // 7. best_phi_output_file
            "%s\n" // 8. best_defocus_output_file
            "%s\n" // 9. best_pixel_size_output_file
            "%s\n" // 10. correlation_avg_output_file
            "%s\n" // 11. correlation_std_output_file
            "%s\n" // 12. output_histogram_file
            "%f\n" // 13. input_pixel_size (Angstroms)
            "300.0\n" // 14. voltage_kV
            "2.7\n" // 15. spherical_aberration_mm
            "0.1\n" // 16. amplitude_contrast
            "1752.0\n" // 17. defocus1 (Angstroms)
            "1529.0\n" // 18. defocus2 (Angstroms)
            "-74.26\n" // 19. defocus_angle (degrees)
            "0.0\n" // 20. phase_shift (degrees)
            "%2.4f\n" // 21. high_resolution_limit
            "2.5\n" // 22. angular_step (0.0 = auto)
            "1.5\n" // 23. in_plane_angular_step (0.0 = auto)
            "0.0\n" // 24. defocus_search_range
            "50.0\n" // 25. defocus_step (0.0 = no search)
            "0.0\n" // 26. pixel_size_search_range
            "0.02\n" // 27. pixel_size_step (0.0 = no search)
            "1.0\n" // 28. padding
            "0.0\n" // 29. particle_radius_angstroms (0.0 = max)
            "O\n" // 30. my_symmetry (octahedral)
            "Yes\n" // 31. use_gpu_input
            "Yes\n" // 32. use_fast_fft
            "2\n", // 33. max_threads
            search_image.c_str( ),
            template_file.c_str( ),
            mip_output.c_str( ),
            scaled_mip_output.c_str( ),
            psi_output.c_str( ),
            theta_output.c_str( ),
            phi_output.c_str( ),
            defocus_output.c_str( ),
            pixel_size_output.c_str( ),
            corr_avg_output.c_str( ),
            corr_std_output.c_str( ),
            histogram_output.c_str( ),
            4.0f * wanted_pixel_size);

    // Write input to file
    wxString input_file = wxString::Format("%s/input.txt", temp_dir);
    wxFFile  file(input_file, "w");
    if ( file.IsOpened( ) ) {
        file.Write(match_template_input);
        file.Close( );
        wxPrintf("Input written to: %s\n", input_file.c_str( ));
    }
    else {
        wxPrintf("ERROR: Failed to write input file\n");
        return false;
    }

    // Run match_template_gpu with input file
    wxString binary_path = input_starfile_filename.at(1) + "/match_template_gpu";
    wxString command     = wxString::Format("%s < %s 2>&1", binary_path.c_str( ), input_file.c_str( ));

    wxPrintf("\nRunning command:\n%s\n\n", command.c_str( ));

    int result = system(command.c_str( ));

    if ( result == 0 ) {
        wxPrintf("\n=== match_template_gpu completed successfully ===\n");
    }
    else {
        wxPrintf("\n=== match_template_gpu FAILED with exit code: %d ===\n", result);
    }

    wxPrintf("\nOutput files in: %s\n", temp_dir.c_str( ));
    // revert - end FastFFT crash test

    return true;
}
