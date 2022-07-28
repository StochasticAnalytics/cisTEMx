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

bool DoCosineMaskingTest( ) {

    bool passed     = true;
    bool all_passed = true;

    SamplesBeginTest("Extract slice CPU vs ground truth", passed);

    all_passed = all_passed && passed;
    SamplesTestResult(passed);

    return all_passed
}