#include <torch/torch.h>
#include <iostream>

int main( ) {
    torch::Tensor tensor = torch::eye(3);
    std::cout << tensor << std::endl;
}

// #include <cistem_config.h>

// #ifdef ENABLEGPU
// #include "../../../gpu/gpu_core_headers.h"
// #else
// #include "../../../core/core_headers.h"
// #endif

// #include "../../../gpu/GpuImage.h"

// #include "../common/common.h"
// #include "torch_vs_pytorch.h"

// void TorchVsPytorchRunner(const wxString& hiv_image_80x80x1_filename, wxString& temp_directory) {

//     SamplesPrintTestStartMessage("Starting the torch vs pytorch test", false);

//     TEST(TorchVsPytorch(hiv_image_80x80x1_filename, temp_directory));

//     SamplesPrintEndMessage( );

//     return;
// }

// bool TorchVsPytorch(const wxString& hiv_image_80x80x1_filename, wxString& temp_directory) {

//     bool passed     = true;
//     bool all_passed = true;

//     SamplesBeginTest("Basic pytorch vs torch", passed);

//     // Do some things, set the value for passed

//     all_passed = passed ? all_passed : false;

//     SamplesTestResult(passed);

//     return all_passed;
// }