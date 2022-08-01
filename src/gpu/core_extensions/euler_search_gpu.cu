
#include <cistem_config.h>

#include "../../gpu/gpu_core_headers.h"
#include "../../gpu/GpuImage.h"

template <>
void EulerSearch::RunGPU<GpuImage>(GpuImage* testCompile, Particle& particle, Image& input_3d, Image* projections) {
    wxPrintf("Hello from the GPU code~\n");
    return;
}
