
#include <cistem_config.h>

#ifdef ENABLEGPU
#include "../../gpu/gpu_core_headers.h"
#include "../../gpu/GpuImage.h"
#else
#include "../../core/core_headers.h"
#endif

#include "ProjectionComparisonObjects.h"

ProjectionComparisonObjects::ProjectionComparisonObjects( ) {
}

ProjectionComparisonObjects::~ProjectionComparisonObjects( ) {
}

void ProjectionComparisonObjects::DummyMethod( ) {
    wxPrintf("ProjectionComparisonObjects::DummyMethod\n");
}

#ifndef ENABLEGPU

void ProjectionComparisonObjects::PrepareGpuVolumeProjection(ReconstructedVolume& input_density, GpuImage* external_gpu_volume) {
    return;
}

void ProjectionComparisonObjects::PrepareGpuImagesProjection(GpuImage* external_gpu_projection) {
    return;
}

float ProjectionComparisonObjects::DoGpuProjection( ) {
    return 0.0f;
}

#else

void ProjectionComparisonObjects::PrepareGpuVolumeProjection(ReconstructedVolume& input_density, GpuImage* external_gpu_volume) {

    gpu_density_map = external_gpu_volume;
    // Make a copy since we cannot take a back fft after FourierSpaceQuadrant Swap
    Image temp_image;
    temp_image = input_density.density_map;
    temp_image.BackwardFFT( );

    // Realspace quadrants should already be swapped. TODO: just add a check inside the method and don't bother with the argument passing
    temp_image.SwapFourierSpaceQuadrants(false);
    // This is a shared resource, and we don't copy the host real_values anyway, so DONOT pin the memory
    // gpu_density_map->Init(temp_image, false, false);
    gpu_density_map->CopyHostToDeviceTextureComplex3d( );

    return;
};

void ProjectionComparisonObjects::PrepareGpuImagesProjection(GpuImage* external_gpu_projection) {

    // We only want a pointer in the ImageComparisionObject so

    gpu_projection = external_gpu_projection;
    if ( ! gpu_projection->is_meta_data_initialized ) {
        // Note this is not copying any image data, just the meta data
        gpu_projection->Init(*projection_image);
        gpu_projection->CopyHostToDevice( );
    }

    return;
};

float ProjectionComparisonObjects::DoGpuProjection( ) {
    MyDebugAssertTrue(gpu_density_map->is_allocated_texture_cache, "gpu_density_map is not allocated");
    MyDebugAssertTrue(gpu_projection->is_in_memory_gpu, "gpu_projection is not allocated");
    MyDebugAssertTrue(gpu_particle_image->is_in_memory_gpu, "gpu_particle_image is not allocated");
    // wxPrintf("Is host memory pinned (%d\n)", gpu_projection->is_host_memory_pinned);

    // gpu_projection->ExtractSliceShiftAndCtf(gpu_density_map, gpu_ctf_image, particle->alignment_parameters, reference_volume->pixel_size, particle->pixel_size / particle->filter_radius_high, true,
    //                                         swap_quadrants, apply_shifts, apply_ctf, absolute_ctf);
    if ( whiten ) {
        // gpu_projection->Whiten(particle->pixel_size / particle->filter_radius_high);
    }
    if ( mask_radius > 0.f ) {
        // CosineMask is not implemented yet, but we can at least do the backFFT
        gpu_projection->BackwardFFT( );
    }
    // cudaErr(cudaStreamSynchronize(cudaStreamPerThread));

    // global_timer.start("copy device to host");
    // gpu_projection->CopyDeviceToHostAndSynchronize(false, false);
    // global_timer.lap("copy device to host");
    float filter_radius_high = fminf(powf(particle->pixel_size / particle->filter_radius_high, 2), 0.25);
    float filter_radius_low  = 0.0f;
    if ( particle->filter_radius_low != 0.0 )
        filter_radius_low = powf(particle->pixel_size / particle->filter_radius_low, 2);

    // if ( copy_gpu_particle_image ) {
    //     if ( gpu_particle_image == nullptr ) {
    //         gpu_particle_image = new GpuImage;
    //     }
    //     gpu_particle_image->Init(*particle->particle_image);
    //     gpu_particle_image->CopyHostToDevice( );
    //     copy_gpu_particle_image = false;
    // }

    // gpu_particle_image->Init(*particle->particle_image, false);
    gpu_particle_image->CopyHostToDevice( );
    float tmp_corr = 0.f; //gpu_particle_image->GetWeightedCorrelationWithImage(*gpu_projection, score_buffer, &score_buffer_size, filter_radius_low, filter_radius_high, particle->pixel_size / particle->signed_CC_limit);

    return tmp_corr;
};

#endif
