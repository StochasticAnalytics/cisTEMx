

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

    // TEST half shift recovery
    // Image vol;
    // vol.QuickAndDirtyReadSlices("/scratch/salina/other_data/6swa_less_f15-65_1.5_apix.mrc", 1, 256);
    // vol.ZeroFloatAndNormalize( );

    // CTF my_ctf(300.0f,
    //            2.7f,
    //            0.07f,
    //            7000.f,
    //            6000.f,
    //            45.f,
    //            1.5f,
    //            0.f);

    // RotationMatrix rot_mat;
    // rot_mat.SetToIdentity( );

    // vol.ForwardFFT( );
    // vol.SwapRealSpaceQuadrants( );

    // std::vector<float> ratio;
    // std::vector<float> shifted_ratio;

    // for ( int i = 0; i < 100; i++ ) {
    //     Image ref, slice, shifted_slice;
    //     slice.Allocate(vol.logical_x_dimension, vol.logical_y_dimension, 1, false);

    //     vol.ExtractSliceByRotMatrix(slice, rot_mat);

    //     slice.SwapRealSpaceQuadrants( );

    //     slice.ApplyCTF(my_ctf);
    //     ref = slice;
    //     ref.QuickAndDirtyWriteSlice("ref.mrc", 1, true);
    //     ref.SwapRealSpaceQuadrants( );

    //     Image noise_img(256.f, 256.f, 1.f, true);
    //     noise_img.FillWithNoise(GAUSSIAN, 0.0f, .2f);
    //     noise_img.ForwardFFT( );

    //     shifted_slice = slice;
    //     shifted_slice.PhaseShift(0.5f, 0.5f, 0.0f);

    //     slice.AddImage(&noise_img);
    //     shifted_slice.AddImage(&noise_img);

    //     slice.QuickAndDirtyWriteSlice("slice.mrc", 1, true);
    //     shifted_slice.QuickAndDirtyWriteSlice("shifted_slice.mrc", 1, true);

    //     slice.ConjugateMultiplyPixelWise(ref);
    //     shifted_slice.ConjugateMultiplyPixelWise(ref);
    //     slice.BackwardFFT( );
    //     shifted_slice.BackwardFFT( );

    //     slice.MultiplyByConstant(256.f);
    //     shifted_slice.MultiplyByConstant(256.f);

    //     slice.QuickAndDirtyWriteSlice("slice_ctf_inv.mrc", 1, true);
    //     shifted_slice.QuickAndDirtyWriteSlice("shifted_slice_inv.mrc", 1, true);

    //     Image clip, shifted_clip;
    //     clip.Allocate(11, 11, 1, true);
    //     shifted_clip = clip;

    //     slice.ClipInto(&clip);
    //     shifted_slice.ClipInto(&shifted_clip);

    //     EmpiricalDistribution<double> dist;

    //     dist            = slice.ReturnDistributionOfRealValues( );
    //     float max_slice = dist.GetMaximum( );
    //     dist.Reset( );

    //     dist                    = shifted_slice.ReturnDistributionOfRealValues( );
    //     float max_shifted_slice = dist.GetMaximum( );
    //     dist.Reset( );

    //     clip.ForwardFFT( );
    //     clip.Resize(120, 120, 1);
    //     clip.BackwardFFT( );

    //     shifted_clip.ForwardFFT( );
    //     shifted_clip.Resize(60, 60, 1);
    //     shifted_clip.BackwardFFT( );

    //     dist           = clip.ReturnDistributionOfRealValues( );
    //     float max_clip = dist.GetMaximum( );
    //     dist.Reset( );

    //     dist                   = shifted_clip.ReturnDistributionOfRealValues( );
    //     float max_shifted_clip = dist.GetMaximum( );
    //     dist.Reset( );

    //     wxPrintf("Slice: %3.3f, %3.3f\n", max_slice, max_clip);
    //     wxPrintf("Shift: %3.3f, %3.3f\n", max_shifted_slice, max_shifted_clip);

    //     float r  = max_clip / max_slice;
    //     float sr = max_shifted_clip / max_shifted_slice;
    //     wxPrintf("Ratio: %3.3f, %3.3f\n", r, sr);

    //     ratio.push_back(r);
    //     shifted_ratio.push_back(sr);
    // }

    // double sum;

    // sum = 0;
    // for ( auto& val : ratio ) {
    //     sum += val;
    // }
    // std::cerr << "Mean ratio: " << 100.0 * sum / double(ratio.size( )) - 100.0 << "\n";

    // sum = 0;
    // for ( auto& val : shifted_ratio ) {
    //     sum += val;
    // }
    // std::cerr << "Mean shifted_ratio: " << 100.0 * sum / double(shifted_ratio.size( )) - 100.0 << "\n";

    Image vol;

    vol.QuickAndDirtyReadSlices("/scratch/siracusa/get_scaling_images/7A4M-assembly1-apoF_big.mrc", 1, 768);
    vol.ZeroFloatAndNormalize( );

    Image mask = vol;
    mask.Binarise(2.f);
    // smooth the mask
    mask.ForwardFFT( );
    mask.GaussianLowPassFilter(.5f);
    mask.BackwardFFT( );
    float min, max;
    mask.GetMinMax(min, max);
    mask.MultiplyAddConstant(1. / max, -1.);

    Image noise = mask;
    noise.FillWithNoise(GAUSSIAN, 0.f, 1.5f);
    noise.MultiplyPixelWise(mask);
    vol.AddImage(&noise);
    RotationMatrix rot_mat;
    rot_mat.SetToIdentity( );

    vol.ForwardFFT( );
    vol.SwapRealSpaceQuadrants( );

    std::vector<std::array<float, 3>> energy;

    Image ref, slice;
    slice.Allocate(vol.logical_x_dimension, vol.logical_y_dimension, 1, false);
    vol.ExtractSliceByRotMatrix(slice, rot_mat);
    slice.SwapRealSpaceQuadrants( );
    slice.BackwardFFT( );
    slice.ZeroFloatAndNormalize( );
    slice.ForwardFFT( );

    slice.QuickAndDirtyWriteSlice("slice.mrc", 1);
    exit(1);
    const float          min_threshold = 0.05f;
    std::array<float, 3> tmp;
    // loop in nm
    for ( int iCTF = 70; iCTF < 1300; iCTF += 20 ) {

        ref = slice;

        CTF my_ctf(300.0f,
                   2.7f,
                   0.07f,
                   10 * float(iCTF),
                   10 * float(iCTF),
                   45.f,
                   1.5f,
                   0.f);

        ref.ApplyCTF(my_ctf);
        ref.BackwardFFT( );
        // long   counted_values = 0;
        // double sum            = 0.;
        // double sum_sq         = 0.;
        // long   address        = 0;
        // // We want the variance of only the non-z
        // for ( int j = 0; j < ref.logical_y_dimension; j++ ) {
        //     for ( int i = 0; i < ref.logical_x_dimension; i++ ) {
        //         float val = 30.0f * ref.real_values[address];
        //         if ( val > min_threshold || val < -min_threshold ) {
        //             sum += val;
        //             sum_sq += val * val;
        //             counted_values++;
        //         }
        //         address++;
        //     }
        //     address += ref.padding_jump_value;
        // }
        // double variance = sum_sq / counted_values - powf(sum / counted_values, 2);
        // ref.Abs( );
        // ref.Binarise(min_threshold);
        // float n_above = sqrtf(30.f) * ref.ReturnSumOfRealValues( );

        // tmp = {float(iCTF), variance, n_above};
        // energy.emplace_back(tmp);
        // std::cerr << "var, sum: " << variance << " " << n_above << "\n ";
    }

    // NumericTextFile my_file("energy_spread.txt", OPEN_TO_WRITE, 3);
    // for ( auto& record : energy ) {
    //     std::cerr << "Values: " << record[0] << " " << record[1] << " " << record[2] << "\n";
    //     my_file.WriteLine(record.data( ));
    // }
    // my_file.Close( );

    return true;
}
