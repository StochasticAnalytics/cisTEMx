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

void CPUvsGPUProjectionRunner(const wxString& temp_directory) {

    SamplesPrintTestStartMessage("Starting CPU vs GPU projection tests:", false);

    wxString cistem_ref_dir = CheckForReferenceImages( );
    // If we are in the dev container the CISTEM_REF_IMAGES variable should be defined, pointing to images we need.
    TEST(DoCPUvsGPUProjectionTest(cistem_ref_dir, temp_directory));

    SamplesPrintEndMessage( );

    return;
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

    // FIXME: Testing amplitude contrast
    cpu_prj.SwapRealSpaceQuadrants( );
    cpu_prj.QuickAndDirtyWriteSlice("tmp_0.mrc", 1, true);
    CTF   my_ctf;
    CTF   my_ctf_R;
    CTF   my_ctf_I;
    Image my_ctf_img, my_ctf_img_R, my_ctf_img_I;
    my_ctf_img.CopyFrom(&cpu_prj);
    my_ctf_img.SetToConstant(1.f);
    my_ctf_img_R.CopyFrom(&my_ctf_img);
    my_ctf_img_I.CopyFrom(&my_ctf_img);
    Image aperture_grating = my_ctf_img;
    for ( int i = 0; i < aperture_grating.real_memory_allocated; i += 2 ) {
        aperture_grating.real_values[i]     = 1.f;
        aperture_grating.real_values[i + 1] = 0.f;
    }
    aperture_grating.is_in_real_space = false;

    fftwf_plan plan_fwd_w; // !< FFTW plan for the image (fwd)
    fftwf_plan plan_inv_w; // !< FFTW plan for the image (fwd)
    fftwf_plan plan_fwd_p;
    fftwf_plan plan_inv_p;

    // Make the complex arrays;
    std::vector<std::complex<float>> complex_aperture(cpu_prj.number_of_real_space_pixels, std::complex<float>(0.f, 0.f));
    std::vector<std::complex<float>> input_prj(cpu_prj.number_of_real_space_pixels, std::complex<float>(0.f, 0.f));
    plan_fwd_w = fftwf_plan_dft_2d(cpu_prj.logical_y_dimension, cpu_prj.logical_x_dimension, reinterpret_cast<fftwf_complex*>(complex_aperture.data( )), reinterpret_cast<fftwf_complex*>(complex_aperture.data( )), FFTW_FORWARD, FFTW_ESTIMATE);
    plan_inv_w = fftwf_plan_dft_2d(cpu_prj.logical_y_dimension, cpu_prj.logical_x_dimension, reinterpret_cast<fftwf_complex*>(complex_aperture.data( )), reinterpret_cast<fftwf_complex*>(complex_aperture.data( )), FFTW_BACKWARD, FFTW_ESTIMATE);
    plan_fwd_p = fftwf_plan_dft_2d(cpu_prj.logical_y_dimension, cpu_prj.logical_x_dimension, reinterpret_cast<fftwf_complex*>(input_prj.data( )), reinterpret_cast<fftwf_complex*>(input_prj.data( )), FFTW_FORWARD, FFTW_ESTIMATE);
    plan_inv_p = fftwf_plan_dft_2d(cpu_prj.logical_y_dimension, cpu_prj.logical_x_dimension, reinterpret_cast<fftwf_complex*>(input_prj.data( )), reinterpret_cast<fftwf_complex*>(input_prj.data( )), FFTW_BACKWARD, FFTW_ESTIMATE);

    cpu_prj.BackwardFFT( );
    for ( int i = 0; i < cpu_prj.number_of_real_space_pixels; i++ ) {
        complex_aperture[i] = std::complex<float>(1., 0.);
        input_prj[i]        = std::complex<float>(0., 0.); //std::exp(std::complex<float>(0.f, cpu_prj.real_values[i]));
    }
    cpu_prj.ForwardFFT( );

    fftwf_execute_dft(plan_fwd_p, reinterpret_cast<fftwf_complex*>(input_prj.data( )), reinterpret_cast<fftwf_complex*>(input_prj.data( )));
    {
        float       acceleration_voltage_in_kV = 300.0f; // keV
        float       spherical_aberration_in_mm = 2.7f; // mm
        float       amplitude_contrast;
        float       defocus_1_in_angstroms            = 8000.f; // A
        float       defocus_2_in_angstroms            = 7000.f; //A
        float       astigmatism_azimuth_in_degrees    = 30.0f; // degrees
        float       pixel_size_in_angstroms           = 1.0f; // A
        float       additional_phase_shift_in_radians = 0.f; // rad
        float       sample_thickness_in_nms           = 0.0f; //nm
        const float objective_lens_focal_length       = 3.5; //mm
        const float objective_aperture_diameter       = 100.f; // um
        float       wave_length                       = 1226.39 / sqrtf(acceleration_voltage_in_kV * 1000 + 0.97845e-6 * powf(acceleration_voltage_in_kV * 1000, 2)) * 1e-2;
        float       objective_aperture_resolution     = (wave_length * objective_lens_focal_length * 1e7) / (objective_aperture_diameter / 2.0f * 1e4);
        float       wanted_falloff                    = 14.f;

        std::cerr << "obj " << objective_aperture_diameter << " res " << objective_aperture_resolution << std::endl;
        aperture_grating.ReturnCosineMaskBandpassResolution(pixel_size_in_angstroms, objective_aperture_resolution, wanted_falloff);
        std::cerr << " res, and fallof " << objective_aperture_resolution << " " << wanted_falloff << std::endl;
        aperture_grating.CosineRingMask(-1.0f, objective_aperture_resolution, wanted_falloff);
        amplitude_contrast = 0.07f;
        my_ctf.Init(acceleration_voltage_in_kV, spherical_aberration_in_mm, amplitude_contrast,
                    defocus_1_in_angstroms, defocus_2_in_angstroms, astigmatism_azimuth_in_degrees,
                    pixel_size_in_angstroms, additional_phase_shift_in_radians);
        amplitude_contrast = 1.f;
        my_ctf_R.Init(acceleration_voltage_in_kV, spherical_aberration_in_mm, amplitude_contrast,
                      defocus_1_in_angstroms, defocus_2_in_angstroms, astigmatism_azimuth_in_degrees,
                      pixel_size_in_angstroms, additional_phase_shift_in_radians);
        amplitude_contrast = 0.f;
        my_ctf_I.Init(acceleration_voltage_in_kV, spherical_aberration_in_mm, amplitude_contrast,
                      defocus_1_in_angstroms, defocus_2_in_angstroms, astigmatism_azimuth_in_degrees,
                      pixel_size_in_angstroms, additional_phase_shift_in_radians);
    }

    Image complex_aperture_real = aperture_grating;
    complex_aperture_real.BackwardFFT( );
    int counter         = 0;
    int counter_complex = 0;
    for ( int i = 0; i < complex_aperture_real.logical_y_dimension; i++ ) {
        for ( int j = 0; j < complex_aperture_real.logical_x_dimension; j++ ) {
            complex_aperture[counter_complex] = complex_aperture_real.real_values[counter];
            complex_aperture[counter_complex] = 0.f;
            counter++;
            counter_complex++;
        }
        counter += complex_aperture_real.padding_jump_value;
    }
    fftwf_execute_dft(plan_fwd_w, reinterpret_cast<fftwf_complex*>(complex_aperture.data( )), reinterpret_cast<fftwf_complex*>(complex_aperture.data( )));
    for ( int i = 0; i < cpu_prj.number_of_real_space_pixels; i++ ) {
        // my_ptr[i] *= complex_aperture[i];
        // my_ptr[i] = std::norm(my_ptr[i]);
    }
    fftwf_execute_dft(plan_inv_p, reinterpret_cast<fftwf_complex*>(input_prj.data( )), reinterpret_cast<fftwf_complex*>(input_prj.data( )));

    aperture_grating.is_in_real_space = true;
    aperture_grating.QuickAndDirtyWriteSlice("tmp_0_aperture.mrc", 1, true);
    my_ctf_img.ApplyCTF(my_ctf);
    my_ctf_img_R.ApplyCTF(my_ctf_R);
    my_ctf_img_I.ApplyCTF(my_ctf_I);
    Image tmp;
    tmp.CopyFrom(&cpu_prj);
    tmp.object_is_centred_in_box = cpu_prj.object_is_centred_in_box;
    tmp.ApplyCTF(my_ctf);
    tmp.BackwardFFT( );
    tmp.QuickAndDirtyWriteSlice("tmp_1.mrc", 1, true);

    tmp.CopyFrom(&cpu_prj);
    tmp.object_is_centred_in_box = cpu_prj.object_is_centred_in_box;
    tmp.ApplyCTF(my_ctf_I);
    tmp.BackwardFFT( );
    tmp.QuickAndDirtyWriteSlice("tmp_1_phase.mrc", 1, true);

    tmp.CopyFrom(&cpu_prj);
    tmp.object_is_centred_in_box = cpu_prj.object_is_centred_in_box;
    tmp.SetToConstant(0.f);
    tmp.is_in_real_space         = true;
    tmp.object_is_centred_in_box = true;
    for ( int i = 0; i < tmp.real_memory_allocated; i++ ) {
        tmp.real_values[i] = std::norm(input_prj[i]);
    }
    tmp.QuickAndDirtyWriteSlice("tmp_1_complex.mrc", 1, true);
    // destroy the plans
    fftwf_destroy_plan(plan_fwd_w);
    fftwf_destroy_plan(plan_inv_w);
    fftwf_destroy_plan(plan_fwd_p);
    fftwf_destroy_plan(plan_inv_p);

    cpu_prj.BackwardFFT( );
    Image v_R                    = cpu_prj;
    v_R.object_is_centred_in_box = cpu_prj.object_is_centred_in_box;
    Image v_I                    = cpu_prj;
    v_I.object_is_centred_in_box = cpu_prj.object_is_centred_in_box;

    for ( int i = 0; i < v_R.real_memory_allocated; i++ ) {
        v_R.real_values[i] = std::cos(cpu_prj.real_values[i]) - std::sin(cpu_prj.real_values[i]);
        v_I.real_values[i] = std::cos(cpu_prj.real_values[i]) + std::sin(cpu_prj.real_values[i]);
    }
    v_R.ForwardFFT( );
    v_I.ForwardFFT( );
    Image w_Ra                    = v_R;
    Image w_Ia                    = v_I;
    Image w_Rb                    = v_I;
    Image w_Ib                    = v_R;
    w_Ra.object_is_centred_in_box = v_R.object_is_centred_in_box;
    w_Ia.object_is_centred_in_box = v_I.object_is_centred_in_box;
    w_Rb.object_is_centred_in_box = v_I.object_is_centred_in_box;
    w_Ib.object_is_centred_in_box = v_R.object_is_centred_in_box;

    w_Ra.ApplyCTF(my_ctf_R);
    w_Rb.ApplyCTF(my_ctf_I);
    w_Ia.ApplyCTF(my_ctf_R);
    w_Ib.ApplyCTF(my_ctf_I);

    w_Ra.SubtractImage(&w_Rb);
    w_Ia.AddImage(&w_Ib);

    aperture_grating.is_in_real_space = false;
    w_Ra.MultiplyPixelWise(aperture_grating);
    w_Rb.MultiplyPixelWise(aperture_grating);

    w_Ra.BackwardFFT( );
    w_Ia.BackwardFFT( );

    for ( int i = 0; i < v_R.real_memory_allocated; i++ ) {
        v_R.real_values[i] = std::real(w_Ra.real_values[i] * w_Ra.real_values[i] + w_Ia.real_values[i] * w_Ia.real_values[i]);
    }

    v_R.is_in_real_space         = true;
    v_R.object_is_centred_in_box = true;
    v_R.QuickAndDirtyWriteSlice("tmp_70.mrc", 1, true);
    exit(0);
    // FIXME: end

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

    all_passed = passed ? all_passed : false;
    SamplesTestResult(passed);

    SamplesBeginTest("Extract slice GPU vs ground truth", passed);

    GpuImage gpu_volume;
    GpuImage gpu_prj;

    cpu_volume.BackwardFFT( );
    cpu_volume.ZeroFloatAndNormalize( );

    cpu_volume.SwapFourierSpaceQuadrants(false);

    // Associate the gpu volume with the cpu volume, getting meta data and pinning the host pointer.
    gpu_volume.Init(cpu_volume, false, true);

    // The volume is already in Fourier space, so we can copy it to texture cache for interpolation.
    gpu_volume.CopyHostToDeviceTextureComplex3d(cpu_volume);

    // Image centered;
    // centered.CopyFrom(&cimg);
    gpu_prj.Init(cimg);
    gpu_prj.CopyHostToDevice(cimg); // FIXME: just allocate in fourier space
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
        gpu_prj.CopyDeviceToHostAndSynchronize(cimg, false);
        gpu_prj.RecordAndWait( );

        cimg.ZeroFloatAndNormalize(1.f, mask_radius);

        // Cacluate the normalized correlation.
        score  = ref_prj.ReturnCorrelationCoefficientUnnormalized(cimg, mask_radius);
        passed = passed && (score > 0.999f);

        // cimg.QuickAndDirtyWriteSlice(prj_output_filename_base + std::to_string(iPrj) + ".mrc", 1, true);
    }

    all_passed = passed ? all_passed : false;
    SamplesTestResult(passed);

    int         n_loops   = 100;
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
        gpu_prj.CopyDeviceToHostAndSynchronize(cimg, false);
        gpu_prj.RecordAndWait( );

        cpu_prj.SwapRealSpaceQuadrants( );
        cpu_prj.BackwardFFT( );

        cpu_prj.ZeroFloatAndNormalize(1.f, mask_radius);
        cimg.ZeroFloatAndNormalize(1.f, mask_radius);

        score  = cpu_prj.ReturnCorrelationCoefficientUnnormalized(cimg, mask_radius);
        passed = passed && (score > 0.999f);
    }

    all_passed = passed ? all_passed : false;
    SamplesTestResult(passed);

    std::vector<std::string> condition_name = {"Extract and whiten w/Fuzz(" + std::to_string(n_loops) + ")",
                                               "Extract and phase shift w/Fuzz(" + std::to_string(n_loops) + ")",
                                               "Extract and swap quadrants w/Fuzz(" + std::to_string(n_loops) + ")"};

    // For particle alignment, the res-limit defaults to 0.5 (nyquist).
    // This means this testing is not strictly valid for the corners in Fourier space, which are used in TM i think.
    float             res_limit      = 0.5f;
    std::vector<bool> limit_res      = {true, true, true};
    std::vector<bool> swap_quadrants = {false, false, true};
    std::vector<bool> apply_shifts   = {false, true, true};
    std::vector<bool> apply_ctf      = {false, false, false};
    std::vector<bool> absolute_ctf   = {false, false, false};
    std::vector<bool> whiten         = {true, true, true};

    // Dummy ctf imag; TODO: add random CTFs w/ w/o absolute CTF. needs to be updated with
    GpuImage ctf_img;
    ctf_img.CopyFrom(&gpu_prj);
    ctf_img.SetToConstant(1.f);

    for ( int iCondition = 0; iCondition < condition_name.size( ); iCondition++ ) {
        SamplesBeginTest(condition_name[iCondition].c_str( ), passed);
        for ( int iLoop = 0; iLoop < n_loops; iLoop++ ) {
            // Compared to the previous, we now pass a bool to pug Extract slice and add and extra method call for the GPU to get whitening of the PS.
            my_angles_and_shifts.Init(my_rand.GetUniformRandomSTD(-180.f, 180), my_rand.GetUniformRandomSTD(0.f, 180), my_rand.GetUniformRandomSTD(0.f, 360), 0.f, 0.f);
            new_cpu_volume.ExtractSlice(cpu_prj, my_angles_and_shifts, res_limit, limit_res[iCondition]);
            gpu_prj.ExtractSliceShiftAndCtf(&gpu_volume, &ctf_img, my_angles_and_shifts, pixel_size, res_limit, limit_res[iCondition],
                                            swap_quadrants[iCondition], apply_shifts[iCondition], apply_ctf[iCondition], absolute_ctf[iCondition]);
            if ( whiten[iCondition] ) {
                gpu_prj.Whiten( );
            }

            if ( ! swap_quadrants[iCondition] ) {
                // If true, then the swapping is done by ExtractSliceShiftAndCtf, otherwise do it here
                gpu_prj.SwapRealSpaceQuadrants( );
            }
            gpu_prj.BackwardFFT( );
            gpu_prj.CopyDeviceToHostAndSynchronize(cimg, false);

            cpu_prj.Whiten( );
            cpu_prj.SwapRealSpaceQuadrants( );
            cpu_prj.BackwardFFT( );

            cpu_prj.ZeroFloatAndNormalize(1.f, mask_radius);
            cimg.ZeroFloatAndNormalize(1.f, mask_radius);

            passed = CompareRealValues(cpu_prj, cimg, 0.999f, mask_radius);
        }

        all_passed = passed ? all_passed : false;
        SamplesTestResultCanFail(passed);
    }

    all_passed = passed ? all_passed : false;

    return all_passed;
}
