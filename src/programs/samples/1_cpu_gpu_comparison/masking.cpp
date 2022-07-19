#include <cistem_config.h>

#ifdef ENABLEGPU
#include "../../../gpu/gpu_core_headers.h"
#else
#error "GPU is not enabled"
#include "../../../core/core_headers.h"
#endif

#include "../../../gpu/GpuImage.h"

#include "../common/common.h"
#include "masking.h"

bool CPUvsGPUMaskingTest(const wxString& hiv_image_80x80x1_filename, wxString& temp_directory) {

    bool passed;
    bool all_passed = true;

    SamplesPrintTestStartMessage("Starting CPU vs GPU masking tests:", false);

    all_passed = all_passed && DoCosineMaskingTest(hiv_image_80x80x1_filename, temp_directory);

    SamplesBeginTest("CPU vs GPU overall", passed);
    SamplesPrintResult(all_passed, __LINE__);
    wxPrintf("\n\n");

    return all_passed;
}

bool DoCosineMaskingTest(const wxString& hiv_image_80x80x1_filename, wxString& temp_directory) {

    bool passed     = true;
    bool all_passed = true;

    SamplesBeginTest("Cosine mask real space", passed);

    wxString tmp_img_filename = temp_directory + "/tmp1.mrc";

    MRCFile input_file(hiv_image_80x80x1_filename.ToStdString( ), false);
    MRCFile output_file(tmp_img_filename.ToStdString( ), false);

    Image    cpu_image;
    Image    gpu_host_image;
    GpuImage gpu_image;

    cpu_image.ReadSlice(&input_file, 1);
    gpu_host_image.ReadSlice(&input_file, 1);

    gpu_image.Init(gpu_host_image);
    gpu_image.CopyHostToDevice( );

    float wanted_mask_radius;
    float wanted_mask_edge;
    bool  invert;
    bool  force_mask_value;
    float wanted_mask_value;

    RandomNumberGenerator my_rand(pi_v<float>);

    int n_loops = 1;
    for ( int i = 0; i < n_loops; i++ ) {

        // Make some random parameters.
        wanted_mask_radius = 0.f; // GetUniformRandomSTD(0.0f, cpu_image.logical_x_dimension / 2.0f);
        wanted_mask_edge   = 20.f; //GetUniformRandomSTD(0.0f, 20.0f);
        wanted_mask_value  = 0.f; //GetUniformRandomSTD(0.0f, 1.0f);
        if ( my_rand.GetUniformRandomSTD(0.0f, 1.0f > 0.5f) ) {
            invert = true;
        }
        else {
            invert = false;
        }
        if ( my_rand.GetUniformRandomSTD(0.0f, 1.0f > 0.5f) ) {
            force_mask_value = true;
        }
        else {
            force_mask_value = false;
        }

        // FIXME: for intial run, fix the values.
        invert           = false;
        force_mask_value = false;

        cpu_image.CosineMask(wanted_mask_radius, wanted_mask_edge, invert, force_mask_value, wanted_mask_value);
        // gpu_image.CosineMask(wanted_mask_radius, wanted_mask_edge, invert, force_mask_value, wanted_mask_value);
    }

    all_passed = all_passed && passed;
    SamplesTestResult(passed);

    return all_passed;
}