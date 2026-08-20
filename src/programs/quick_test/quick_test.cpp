

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

class
        QuickTestApp : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );
    void AddCommandLineOptions( );

    std::string input_volume_filename;
    std::string output_directory;
    int         padded_size{512};

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

    input_volume_filename = my_input->GetFilenameFromUser("Input volume", "Cubic 3d reference volume (mrc)", "input.mrc", true);
    output_directory      = my_input->GetStringFromUser("Output directory", "Existing directory the swapped/spectrum volumes are written into", ".");
    padded_size           = my_input->GetIntFromUser("Padded box size", "Output box edge for the implicit zero-padding condition", "512", 2, 4096);

    delete my_input;
}

bool QuickTestApp::DoCalculation( ) {

#if defined(ENABLEGPU) && defined(cisTEM_USING_FastFFT)
    // Correctness probe for the FastFFT centered-texture template prep against the CPU
    // reference prep used by match_template's legacy path. Both must deliver the same
    // centered momentum-space volume -- DC at tile (1, N/2, N/2), kx fastest, the
    // conjugate-mirror kx = -1 plane at slot 0 -- which is the interpolable layout
    // GpuImage::ExtractSliceShiftAndCtf samples.

    MRCFile input_file(input_volume_filename, false);
    Image   input_volume;
    input_volume.ReadSlices(&input_file, 1, input_file.ReturnNumberOfSlices( ));
    MyAssertTrue(input_volume.IsCubic( ), "Input volume must be cubic");

    // Zero-float + normalize per the FastFFT input contract: implicit zero-padding is only
    // continuous against zero-mean data, and the library never does this for you. Applied
    // once, up front, so BOTH paths transform the identical input.
    input_volume.ZeroFloatAndNormalize( );

    const int L = input_volume.logical_x_dimension;

    for ( int condition = 0; condition < 2; condition++ ) {
        const int N = (condition == 0) ? L : padded_size;
        if ( condition == 1 && N == L )
            continue; // padded condition degenerates to the first; nothing new to measure
        wxPrintf("\n==== condition: %i -> %i ====\n", L, N);

        // ---------- CPU reference (the legacy match_template prep) ----------
        Image cpu_vol;
        cpu_vol.CopyFrom(&input_volume);
        if ( N != L ) {
            // Materialized equivalent of the implicit pad: center-embed in the N box with a
            // zero border BEFORE any swap, while the object is still compact. Padding after
            // the swap would separate the wrap-split halves with zeros.
            cpu_vol.Resize(N, N, N, 0.0f);
        }
        // Object center -> origin (wrap-split): the projection-convention placement.
        cpu_vol.SwapRealSpaceQuadrants( );
        cpu_vol.QuickAndDirtyWriteSlices(output_directory + "/cpu_swapped_real_" + std::to_string(N) + ".mrc", 1, N, true);

        // Forward transform + centered spectrum layout (does its own FFT; needs real-space input).
        cpu_vol.SwapFourierSpaceQuadrants(false);
        // Raw complex-tile dump: override so the writer does not attempt a backward fft.
        cpu_vol.is_in_real_space = true;
        cpu_vol.QuickAndDirtyWriteSlices(output_directory + "/cpu_swapped_fourier_" + std::to_string(N) + ".mrc", 1, N, true);

        // ---------- FastFFT (the new prep: UNSWAPPED input, placement handled internally) ----------
        GpuImage d_input;
        d_input.Init(input_volume);
        d_input.CopyHostToDevice(input_volume);

        FastFFT::PlanDescriptor plan{ };
        plan.input_size = {static_cast<std::size_t>(L), static_cast<std::size_t>(L), static_cast<std::size_t>(L)};
        if ( N != L )
            plan.fourier_size = {static_cast<std::size_t>(N), static_cast<std::size_t>(N), static_cast<std::size_t>(N)};
        plan.centered_fwd_output = true;
        plan.fwd_output_delivery = FastFFT::FwdOutputDelivery::surface3d;
        // fp32 texels: this probe measures layout/phase agreement, not fp16 storage loss.
        plan.fwd_output_texel = FastFFT::FwdOutputTexel::fp32;

        FastFFT::FourierTransformer<float, float, float2, 3> FT(plan);
        FT.FwdFFTToTexture(d_input.real_values);

        // Read the centered tile back into an Image's complex buffer: the tight tile pitch
        // (N/2+1 float2 texels per row) equals the FFTW complex row of an N-box Image, so
        // every address aligns 1:1 with the CPU reference above.
        Image ff_vol;
        ff_vol.Allocate(N, N, N, false);
        const size_t host_bytes = static_cast<size_t>(N / 2 + 1) * static_cast<size_t>(N) * static_cast<size_t>(N) * sizeof(float2);
        FT.CopyCenteredOutputToHostAndSynchronize(ff_vol.complex_values, host_bytes);

        // Consumer's half of the split normalization: the store multiplied by 1/sqrt(NxNyNz)
        // and the documented residual 1/sqrt(NxNyNz) belongs to the consumer; applying it
        // lands on the 1/(NxNyNz) Image::ForwardFFT convention the CPU reference used.
        const float residual = 1.0f / sqrtf(float(N) * float(N) * float(N));
        for ( long address = 0; address < ff_vol.real_memory_allocated / 2; address++ )
            ff_vol.complex_values[address] *= residual;

        ff_vol.is_in_real_space = true;
        ff_vol.QuickAndDirtyWriteSlices(output_directory + "/fastfft_fourier_" + std::to_string(N) + ".mrc", 1, N, true);

        // ---------- compare ----------
        const long       n_complex    = cpu_vol.real_memory_allocated / 2;
        constexpr double tolerance    = 1e-4;
        double           max_abs_diff = 0.0, sum_sq_diff = 0.0, max_mag_cpu = 0.0;
        long             n_above_tolerance = 0;
        for ( long address = 0; address < n_complex; address++ ) {
            const double diff_real = double(real(cpu_vol.complex_values[address])) - double(real(ff_vol.complex_values[address]));
            const double diff_imag = double(imag(cpu_vol.complex_values[address])) - double(imag(ff_vol.complex_values[address]));
            const double diff_max  = std::max(std::abs(diff_real), std::abs(diff_imag));

            max_abs_diff = std::max(max_abs_diff, diff_max);
            sum_sq_diff += diff_real * diff_real + diff_imag * diff_imag;
            max_mag_cpu = std::max(max_mag_cpu, double(std::abs(cpu_vol.complex_values[address])));
            if ( diff_max > tolerance )
                n_above_tolerance++;
        }
        wxPrintf("cpu vs fastfft (%i^3): max component diff %.3e, rms diff %.3e, cpu max magnitude %.3e, %li / %li components over %.0e\n",
                 N, max_abs_diff, sqrt(sum_sq_diff / double(2 * n_complex)), max_mag_cpu, n_above_tolerance, n_complex, tolerance);
    }
#else
    wxPrintf("This test requires a GPU + FastFFT build (ENABLEGPU and cisTEM_USING_FastFFT).\n");
#endif

    return true;
}
