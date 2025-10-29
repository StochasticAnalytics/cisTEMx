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

    // Counted Values: 4171655 out of 4175850 fractions: 0.998995
    // Over n_cccs 89411 the Global mean and variance are 2.05256e+26 and -4.213e+52

    double global_ccc_mean         = 0.0;
    double global_ccc_std_variance = 0.0;
    double n_angles_in_search      = 4171655.0;

    Image mip;
    Image sum_of_sqs;
    Image sum;

    mip.QuickAndDirtyReadSlice("/scratch/salina/EMPIAR_11063/DebugMip.mrc", 1);
    sum.QuickAndDirtyReadSlice("/scratch/salina/EMPIAR_11063/DebugSum.mrc", 1);
    sum_of_sqs.QuickAndDirtyReadSlice("/scratch/salina/EMPIAR_11063/DebugSumSqs.mrc", 1);

    double global_sum            = 0.0;
    double global_sum_of_squares = 0.0;

    long      counted_values = 0;
    long      address        = 0;
    const int N              = mip.real_memory_allocated;
    // for ( int y = 0; y < mip.logical_y_dimension; y++ ) {
    //     for ( int x = 0; x < mip.logical_x_dimension; x++ ) {
    //         if ( sum_of_sqs.real_values[address] > cistem::float_epsilon ) {
    //             global_sum += double(sum.real_values[address]);
    //             global_sum_of_squares += double(sum_of_sqs.real_values[address]);
    //             counted_values++;
    //         }
    //         address++;
    //     }
    //     address += mip.padding_jump_value;
    // }
    for ( address = 0; address < mip.real_memory_allocated; address++ ) {
        if ( sum_of_sqs.real_values[address] > cistem::float_epsilon ) {
            global_sum += double(sum.real_values[address]);
            global_sum_of_squares += double(sum_of_sqs.real_values[address]);
            counted_values++;
        }
    }

    const double total_number_of_ccs = double(n_angles_in_search) * double(counted_values);
    std::cerr << "Counted Values: " << counted_values << " out of " << N << " fractions: " << float(counted_values) / float(N) << std::endl;

    MyDebugAssertTrue(counted_values > 0, "No valid pixels counted - all correlation_pixel_sum_of_squares below epsilon");

    std::cerr << " Global sum and sumsq " << global_sum / total_number_of_ccs << " " << global_sum_of_squares / total_number_of_ccs << "\n";
    std::cerr << " Global mean sq " << pow(global_sum / total_number_of_ccs, 2) << "\n";

    global_ccc_mean = global_sum / total_number_of_ccs;

    global_ccc_std_variance = global_sum_of_squares / total_number_of_ccs - double(global_ccc_mean * global_ccc_mean);

    std::cerr << "global mean " << global_ccc_mean << " and variance " << global_ccc_std_variance << "\n";
    return true;
}
