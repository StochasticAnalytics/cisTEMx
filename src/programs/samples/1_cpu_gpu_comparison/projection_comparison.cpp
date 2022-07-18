#include <cistem_config.h>

#ifdef ENABLEGPU
#include "../../../gpu/gpu_core_headers.h"
#else
#error "GPU is not enabled"
#include "../../../core/core_headers.h"
#endif

#include "../../../gpu/GpuImage.h"

#include "../common/common.h"
#include "projection_comparison.h"

// Convenience function to return the abs(complex) in a real valued image that can be saved for inspection
Image GetAbsAsReal(Image& input_image) {

    Image tmp_img;
    int   pixel_pitch = (input_image.logical_x_dimension + input_image.padding_jump_value) / 2;
    tmp_img.Allocate(pixel_pitch, input_image.logical_y_dimension, input_image.logical_z_dimension, true, true);

    int address_complex = 0;
    int address_real    = 0;
    for ( int k = 0; k < input_image.logical_z_dimension; k++ ) {
        for ( int j = 0; j < input_image.logical_y_dimension; j++ ) {
            for ( int i = 0; i < pixel_pitch; i++ ) {
                tmp_img.real_values[address_real] = abs(input_image.complex_values[address_complex]);
                address_complex++;
                address_real++;
            }
            address_real += tmp_img.padding_jump_value;
        }
    }
    return tmp_img;
}

bool CPUvsGPUProjectionTest(const wxString& temp_directory) {

    bool passed;
    bool all_passed = true;

    SamplesPrintTestStartMessage("Starting CPU vs GPU projection tests:", false);

    wxString cistem_ref_var = "CISTEM_REF_IMAGES";
    wxString cistem_ref_dir;
    // If we are in the dev container the CISTEM_REF_IMAGES variable should be defined, pointing to images we need.
    passed = wxGetEnv(cistem_ref_var, &cistem_ref_dir);

    if ( passed ) {

        all_passed = all_passed && DoCPUvsGPUProjectionTest(cistem_ref_dir, temp_directory);

        // SamplesPrintResult(all_passed, __LINE__);
        // wxPrintf("\n\n");
    }
    else {
        // If we are not in the dev container, we can't do the tests.
        all_passed = false;
        wxPrintf("Failed to resolve the (%s) environment variable.\n", cistem_ref_var);
        wxPrintf("We can't run the test without images!\n\n");
    }

    return all_passed;
}

bool DoCPUvsGPUProjectionTest(const wxString& cistem_ref_dir, const wxString& temp_directory) {

    MyAssertFalse(cistem_ref_dir == temp_directory, "The temp directory should not be the same as the CISTEM_REF_IMAGES directory.");

    bool passed     = true;
    bool all_passed = true;

    SamplesBeginTest("Extract slice CPU vs ground truth", passed);

    std::string volume_filename          = cistem_ref_dir.ToStdString( ) + "/ribo_ref.mrc";
    std::string prj_input_filename_base  = cistem_ref_dir.ToStdString( ) + "/ribo_ref_prj_";
    std::string prj_output_filename_base = temp_directory.ToStdString( ) + "/ribo_ref_prj_";

    bool      over_write_input = false;
    Image     cpu_volume;
    Image     cpu_prj;
    Image     ref_prj;
    ImageFile cpu_volume_file;

    cpu_volume_file.OpenFile(volume_filename, over_write_input);

    // TODO: create samples and testing that tries to download the volume from somewhere.
    cpu_volume.ReadSlices(&cpu_volume_file, 1, cpu_volume_file.ReturnNumberOfSlices( ));
    cpu_volume.ZeroFloatAndNormalize( );

    cpu_volume.ForwardFFT( );
    cpu_volume.SwapRealSpaceQuadrants( );

    // Allocate in fourier space and do fft planning.
    cpu_prj.Allocate(cpu_volume.logical_x_dimension, cpu_volume.logical_y_dimension, 1, false, true);

    // Mask radius limiting the correlation calc, the regions with mainly zeros will inflate the scores in the negative control.
    float mask_radius = float(cpu_prj.logical_x_dimension) / 2.0f;
    float score       = 0.f;

    int   n_projections   = 6;
    float my_angles[6][3] = {
            {130.0, 30.0, 280.0},
            {130.0, 30.0, -280.0},
            {50.0, 100.0, 30.0},
            {-50.0, 100.0, 30.0},
            {30.0, -100.0, 50.0},
            {30.0, 100.0, 50.0}};

    AnglesAndShifts my_angles_and_shifts;
    AnglesAndShifts zero_angles(0.f, 0.f, 0.f, 0.f, 0.f);
    float           pixel_size = 1.0f;

    // Make a default projection to see the unrotated.
    cpu_volume.ExtractSlice(cpu_prj, zero_angles, 0.f, false);

    // Recenter and save. Projection is in Fourier space, but saving will invert it.
    cpu_prj.SwapRealSpaceQuadrants( );

    Image cimg;
    cimg.CopyFrom(&cpu_prj);
    cimg.BackwardFFT( );

    for ( int iPrj = 0; iPrj < n_projections; iPrj++ ) {
        // Load the reference image.
        ref_prj.QuickAndDirtyReadSlice(prj_input_filename_base + std::to_string(iPrj) + "_.mrc", 1);
        ref_prj.ZeroFloatAndNormalize(1.f, mask_radius);
        // ref_prj.QuickAndDirtyWriteSlice(prj_output_filename_base + std::to_string(iPrj) + "ref.mrc", 1, true);

        // First we'll reproduce the projection with the CPU.

        // Make a projection the angles and shifts are set to.
        my_angles_and_shifts.Init(my_angles[iPrj][0], my_angles[iPrj][1], my_angles[iPrj][2], 0.f, 0.f);
        cpu_volume.ExtractSlice(cpu_prj, my_angles_and_shifts, 0.f, false);

        // Prepare for real-space correlation score.
        cpu_prj.SwapRealSpaceQuadrants( );
        cpu_prj.BackwardFFT( );
        cpu_prj.ZeroFloatAndNormalize(1.f, mask_radius);

        // Cacluate the normalized correlation.
        score  = ref_prj.ReturnCorrelationCoefficientUnnormalized(cpu_prj, mask_radius);
        passed = passed && (score > 0.999f);

        // wxPrintf("Projection match %d: %f\n", iPrj, score);

        // Now for a negative control, mess up the angles.

        // Make a projection the angles and shifts are *NOT* set to.
        my_angles_and_shifts.Init(my_angles[iPrj][0] / 2, my_angles[iPrj][1] / 2, my_angles[iPrj][2] / 2, 0.f, 0.f);
        cpu_volume.ExtractSlice(cpu_prj, my_angles_and_shifts, 0.f, false);

        // Prepare for real-space correlation score.
        cpu_prj.SwapRealSpaceQuadrants( );
        cpu_prj.BackwardFFT( );
        cpu_prj.ZeroFloatAndNormalize(1.f, mask_radius);

        // Cacluate the normalized correlation.
        score  = ref_prj.ReturnCorrelationCoefficientUnnormalized(cpu_prj, mask_radius);
        passed = passed && (score < 0.9f);
    }

    all_passed = all_passed && passed;
    SamplesTestResult(passed);

    SamplesBeginTest("Extract slice GPU vs ground truth", passed);

    GpuImage gpu_volume;
    GpuImage gpu_prj;

    cpu_volume.BackwardFFT( );
    cpu_volume.SwapFourierSpaceQuadrants(false);

    // Associate the gpu volume with the cpu volume, getting meta data and pinning the host pointer.
    gpu_volume.CopyFromCpuImage(cpu_volume);

    // The volume is already in Fourier space, so we can copy it to texture cache for interpolation.
    gpu_volume.CopyHostToDeviceTextureComplex3d( );

    // Image centered;
    // centered.CopyFrom(&cimg);
    gpu_prj.Init(cimg);
    gpu_prj.CopyHostToDevice( ); // FIXME: just allocate in fourier space
    gpu_prj.ForwardFFT( );

    gpu_prj.SetToConstant(0.f);
    gpu_prj.RecordAndWait( );

    float3 xtrashifts = make_float3(0.0f, 0.0f, 0.0f);

    for ( int iPrj = 0; iPrj < n_projections; iPrj++ ) {
        // Load the reference image.
        ref_prj.QuickAndDirtyReadSlice(prj_input_filename_base + std::to_string(iPrj) + "_.mrc", 1);
        ref_prj.ZeroFloatAndNormalize(1.f, mask_radius);

        // First we'll reproduce the projection with the CPU.

        // Make a projection the angles and shifts are set to.
        my_angles_and_shifts.Init(my_angles[iPrj][0], my_angles[iPrj][1], my_angles[iPrj][2], 0.f, 0.f);
        gpu_prj.ExtractSlice(&gpu_volume, my_angles_and_shifts, pixel_size, 0.f, false);

        // Prepare for real-space correlation score.
        gpu_prj.SwapRealSpaceQuadrants( );
        gpu_prj.BackwardFFT( );
        gpu_prj.CopyDeviceToHost(false, false);
        gpu_prj.RecordAndWait( );

        cimg.ZeroFloatAndNormalize(1.f, mask_radius);

        // Cacluate the normalized correlation.
        score  = ref_prj.ReturnCorrelationCoefficientUnnormalized(cimg, mask_radius);
        passed = passed && (score > 0.999f);

        // cimg.QuickAndDirtyWriteSlice(prj_output_filename_base + std::to_string(iPrj) + ".mrc", 1, true);
    }

    all_passed = all_passed && passed;
    SamplesTestResult(passed);

    int         n_loops   = 1000;
    std::string test_name = "Extract slice CPU vs GPU fuzzing(" + std::to_string(n_loops) + ") loops";
    SamplesBeginTest(test_name.c_str( ), passed);

    cistem_timer::StopWatch timer;
    RandomNumberGenerator   my_rand(pi_v<float>);

    // Clean copy
    Image new_cpu_volume;
    new_cpu_volume.ReadSlices(&cpu_volume_file, 1, cpu_volume_file.ReturnNumberOfSlices( ));
    new_cpu_volume.ZeroFloatAndNormalize( );

    new_cpu_volume.ForwardFFT( );
    new_cpu_volume.SwapRealSpaceQuadrants( );

    for ( int iLoop = 0; iLoop < n_loops; iLoop++ ) {
        my_angles_and_shifts.Init(my_rand.GetUniformRandomSTD(-180.f, 180), my_rand.GetUniformRandomSTD(0.f, 180), my_rand.GetUniformRandomSTD(0.f, 360), 0.f, 0.f);
        new_cpu_volume.ExtractSlice(cpu_prj, my_angles_and_shifts, 0.f, false);
        gpu_prj.ExtractSlice(&gpu_volume, my_angles_and_shifts, pixel_size, 0.f, false);

        // Prepare for real-space correlation score.
        gpu_prj.SwapRealSpaceQuadrants( );
        gpu_prj.BackwardFFT( );
        gpu_prj.CopyDeviceToHost(false, false);
        gpu_prj.RecordAndWait( );

        cpu_prj.SwapRealSpaceQuadrants( );
        cpu_prj.BackwardFFT( );

        cpu_prj.ZeroFloatAndNormalize(1.f, mask_radius);
        cimg.ZeroFloatAndNormalize(1.f, mask_radius);

        score  = cpu_prj.ReturnCorrelationCoefficientUnnormalized(cimg, mask_radius);
        passed = passed && (score > 0.999f);
    }

    all_passed = all_passed && passed;
    SamplesTestResult(passed);

    return all_passed;
}
