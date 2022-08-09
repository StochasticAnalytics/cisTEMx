#include <cistem_config.h>

#ifdef ENABLEGPU
#include "../../../gpu/gpu_core_headers.h"
#else
#error "GPU is not enabled"
#include "../../../core/core_headers.h"
#endif

#include "../../../gpu/GpuImage.h"

#include "../common/common.h"
#include "simple_cufft.h"

bool GpuFftOps(const wxString& hiv_image_80x80x1_filename, wxString& temp_directory) {

    bool passed;
    bool all_passed = true;

    SamplesPrintTestStartMessage("Starting GPU FFT tests:", false);

    all_passed = all_passed && DoInPlaceR2CandC2R(hiv_image_80x80x1_filename, temp_directory);

    SamplesBeginTest("GPU FFT tests overall", passed);
    SamplesPrintResult(all_passed, __LINE__);
    wxPrintf("\n\n");

    return all_passed;
}

bool DoInPlaceR2CandC2R(const wxString& hiv_image_80x80x1_filename, wxString& temp_directory) {

    bool passed     = true;
    bool all_passed = true;

    SamplesBeginTest("Square 2D R2C/C2R inplace ffts", passed);

    // Compare the default ops to the cpu (FFTW/MKL version)
    // These are single precision, inplace ffts, with FFTW layout.

    Image test_image;

    std::vector<size_t> fft_sizes   = {64, 256, 384, 512, 648, 1024, 3456, 4096};
    constexpr size_t    max_3d_size = 648;

    constexpr bool should_allocate_in_real_space = true;
    constexpr bool should_make_fftw_plan         = true;
    // First we will test square images.
    // TODO: add fuzzing over noise distributions, maybe that fits better somewhere else? Or better yet, this method
    // could be called for various noise distributions.
    for ( auto iSize : fft_sizes ) {

        test_image.Allocate(iSize, iSize, 1, should_allocate_in_real_space, should_make_fftw_plan);
        test_image.FillWithNoiseFromNormalDistribution(0.f, 1.f);
        GpuImage gpu_test_image;
        gpu_test_image.Init(test_image);
        gpu_test_image.CopyHostToDeviceAndSynchronize( );

        test_image.ForwardFFT( );
        gpu_test_image.ForwardFFT( );

        // This call also frees the GPU memory and unpins the hsot memory
        Image gpu_cpu_buffer = gpu_test_image.CopyDeviceToNewHost(true, true, true);

        CompareComplexValues(test_image, gpu_cpu_buffer);
    }

    all_passed = all_passed && passed;
    SamplesTestResult(passed);

    SamplesBeginTest("Cubic  3D R2C/C2R inplace ffts", passed);
    for ( auto iSize : fft_sizes ) {
        if ( iSize > max_3d_size ) {
            continue;
        }
        test_image.Allocate(iSize, iSize, 1, should_allocate_in_real_space, should_make_fftw_plan);
        test_image.FillWithNoiseFromNormalDistribution(0.f, 1.f);
        GpuImage gpu_test_image;
        gpu_test_image.Init(test_image);
        gpu_test_image.CopyHostToDeviceAndSynchronize( );

        test_image.ForwardFFT( );
        gpu_test_image.ForwardFFT( );

        // This call also frees the GPU memory and unpins the hsot memory
        Image gpu_cpu_buffer = gpu_test_image.CopyDeviceToNewHost(true, true, true);

        CompareComplexValues(test_image, gpu_cpu_buffer);
    }

    all_passed = all_passed && passed;
    SamplesTestResult(passed);

    SamplesBeginTest("Non-Sq 2D R2C/C2R inplace ffts", passed);
    for ( int i = 1; i < fft_sizes.size( ); i++ ) {

        test_image.Allocate(fft_sizes[i], fft_sizes[i - 1], 1, should_allocate_in_real_space, should_make_fftw_plan);
        test_image.FillWithNoiseFromNormalDistribution(0.f, 1.f);
        GpuImage gpu_test_image;
        gpu_test_image.Init(test_image);
        gpu_test_image.CopyHostToDeviceAndSynchronize( );

        test_image.ForwardFFT( );
        gpu_test_image.ForwardFFT( );

        // This call also frees the GPU memory and unpins the hsot memory
        Image gpu_cpu_buffer = gpu_test_image.CopyDeviceToNewHost(true, true, true);

        CompareComplexValues(test_image, gpu_cpu_buffer);
    }

    all_passed = all_passed && passed;
    SamplesTestResult(passed);

    return all_passed;
}