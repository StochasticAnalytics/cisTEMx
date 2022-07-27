
#include <cistem_config.h>

#ifdef ENABLEGPU
#include "../../gpu/gpu_core_headers.h"
#include "../../gpu/GpuImage.h"
#else
#include "../../core/core_headers.h"
#endif

#include "ProjectionComparisonObjects.h"

ProjectionComparisonObjects::ProjectionComparisonObjects( ) {

    x_shift_limit      = std::numeric_limits<float>::max( );
    y_shift_limit      = std::numeric_limits<float>::max( );
    angle_change_limit = std::numeric_limits<float>::max( );

    mask_radius  = 0.0f;
    mask_falloff = 0.0f;

    is_allocated_gpu_density_map    = false;
    is_allocated_gpu_projection     = false;
    is_allocated_gpu_ctf_image      = false;
    is_allocated_gpu_particle_image = false;

    is_allocated_gpu_search_density_map    = false;
    is_allocated_gpu_search_projection     = false;
    is_allocated_gpu_search_ctf_image      = false;
    is_allocated_gpu_search_particle_image = false;

    current_cpu_pointers_are_for_global_search = false;

    is_allocated_weighted_correlation_buffers = false;

#ifdef DEBUG
    nprj = 0;
    // Get some extra info to make sure all the allocation/deallocation is working. I.e. even if we succeed (no segfaults and correct results)
    // we still want to be sure we aren't alloc/dealloc or copying data around unecessarily.
    n_particle_image_allocations   = 0;
    n_projection_image_allocations = 0;
    n_ctf_image_allocations        = 0;

    n_search_particle_image_allocations   = 0;
    n_search_projection_image_allocations = 0;
    n_search_ctf_image_allocations        = 0;

    n_particle_image_HtoD_copies   = 0;
    n_projection_image_HtoD_copies = 0;
    n_ctf_image_HtoD_copies        = 0;

    n_search_particle_image_HtoD_copies   = 0;
    n_search_projection_image_HtoD_copies = 0;
    n_search_ctf_image_HtoD_copies        = 0;

    n_calls_to_prep_images            = 0;
    n_calls_to_prep_search_images     = 0;
    n_calls_to_prep_ctf_images        = 0;
    n_calls_to_prep_search_ctf_images = 0;

#endif
}

ProjectionComparisonObjects::~ProjectionComparisonObjects( ) {
    Deallocate( );

#ifdef DEBUG
    wxPrintf("\n----------------------------------------------------\n");
    wxPrintf("Image type : Calls : Allocs : HtoD copies\n");
    wxPrintf("Particley image : %i : %i : %i\n", n_calls_to_prep_images, n_particle_image_allocations, n_particle_image_HtoD_copies);
    wxPrintf("Projection image : %i : %i : %i\n", n_calls_to_prep_images, n_projection_image_allocations, n_projection_image_HtoD_copies);
    wxPrintf("CTF image : %i : %i : %i\n", n_calls_to_prep_images, n_ctf_image_allocations, n_ctf_image_HtoD_copies);
    wxPrintf("Particle search image : %i : %i : %i\n", n_calls_to_prep_search_images, n_search_particle_image_allocations, n_search_particle_image_HtoD_copies);
    wxPrintf("Projection search image : %i : %i : %i\n", n_calls_to_prep_search_images, n_search_projection_image_allocations, n_search_projection_image_HtoD_copies);
    wxPrintf("CTF search image : %i : %i : %i\n", n_calls_to_prep_search_images, n_search_ctf_image_allocations, n_search_ctf_image_HtoD_copies);
#endif
}

void ProjectionComparisonObjects::Deallocate( ) {
}

// These are here to prevent copying of pointers.
ProjectionComparisonObjects::ProjectionComparisonObjects(const ProjectionComparisonObjects& other_pcos) {

    MyDebugAssertTrue(false, "A ProjectionComparisonObject should not be assigned or copied!");

    // *this = other_pcos;
}

ProjectionComparisonObjects& ProjectionComparisonObjects::operator=(const ProjectionComparisonObjects& other_pcos) {

    MyDebugAssertTrue(false, "A ProjectionComparisonObject should not be assigned or copied!");
    *this = &other_pcos;
    return *this;
}

ProjectionComparisonObjects& ProjectionComparisonObjects::operator=(const ProjectionComparisonObjects* other_pcos) {

    MyDebugAssertTrue(false, "A ProjectionComparisonObject should not be assigned or copied!");

    // We would actually do a bunch of other stuff here, but we don't need to.
    *this = other_pcos;
    return *this;
}

#ifndef ENABLEGPU

void ProjectionComparisonObjects::PrepareGpuVolumeProjection(ReconstructedVolume& input_3d_local, const bool is_for_global_search) {
    return;
}

void ProjectionComparisonObjects::PrepareGpuImages(Particle& host_particle, Image& host_projection_image, const bool is_for_global_search, c_img_t image_type) {
    return;
}

void ProjectionComparisonObjects::PrepareGpuCTFImages(Particle& host_particle, const bool is_for_global_search) {
    return;
}

float ProjectionComparisonObjects::DoGpuProjection( ) {
    return 0.0f;
}

void ProjectionComparisonObjects::GetCleanCopyOfParticleImage(const bool is_for_global_search) {
    return;
}

void ProjectionComparisonObjects::ResetCleanCopyOfParticleImage(const bool is_for_global_search) {
    return;
}

void ProjectionComparisonObjects::DeallocateCleanCopyOfParticleImage( ) {
    return;
}

#else

void ProjectionComparisonObjects::GetCleanCopyOfParticleImage(const bool is_for_global_search) {

    GpuImage* tmp_ptr = is_for_global_search ? &gpu_search_particle_image : &gpu_particle_image;

    MyDebugAssertTrue(tmp_ptr->is_in_memory_gpu, "Image is not in GPU memory");

    clean_copy = *tmp_ptr;
}

void ProjectionComparisonObjects::DeallocateCleanCopyOfParticleImage( ) {
    { clean_copy.Deallocate( ); };
    return;
}

void ProjectionComparisonObjects::ResetCleanCopyOfParticleImage(const bool is_for_global_search) {

    GpuImage* tmp_ptr = is_for_global_search ? &gpu_search_particle_image : &gpu_particle_image;

    MyDebugAssertTrue(clean_copy.is_in_memory_gpu, "Clean copy of particle image is not in memory!");
    MyDebugAssertTrue(tmp_ptr->is_in_memory_gpu, "Particle image is not in memory!");

    // Leave this as a blocking call
    cudaErr(cudaMemcpy(tmp_ptr->real_values_gpu, clean_copy.real_values_gpu, clean_copy.real_memory_allocated * sizeof(float), cudaMemcpyDeviceToDevice));
}

/** 
 * @brief Prepare complex input image and then FFT to obtain a Fourier Quadrant swapped half-FFT that includes the X=-1 plane.
 * This is currently irreversible, as the Image class only supports R2C/C2R FFTs. This means we have a copy operation on the host, and a subsequent
 * allocation and copy to GPU texture cache. Meaning this is an expensive call, and should be done *outside* of the parallel region.
 * 
 * @param external_gpu_projection The GPU image to use for the projection.
 * @param external_gpu_volume The GPU image to use for the volume.
*/

void ProjectionComparisonObjects::PrepareGpuVolumeProjection(ReconstructedVolume& input_3d_local, const bool is_for_global_search) {

    // Make a copy since we cannot take a back fft after FourierSpaceQuadrant Swap
    Image temp_image;
    temp_image.CopyFrom(input_3d_local.density_map);

    MyDebugAssertTrue(temp_image.is_in_memory, "Density map is not in memory");
    MyDebugAssertFalse(temp_image.is_in_real_space, "Density map is in real space");
    temp_image.BackwardFFT( );

    // Realspace quadrants should already be swapped. TODO: just add a check inside the method and don't bother with the argument passing
    temp_image.SwapFourierSpaceQuadrants(false);
    // This is a shared resource, and we don't copy the host real_values anyway, so DONOT pin the memory
    // gpu_density_map->Init(temp_image, false, false);
    if ( is_for_global_search ) {
        MyDebugAssertFalse(is_allocated_gpu_search_density_map, "Gpu search density map is already allocated");
        gpu_search_density_map.Init(temp_image, false, false);
        gpu_search_density_map.CopyHostToDeviceTextureComplex3d( );
        is_allocated_gpu_search_density_map = true;
    }
    else {
        MyDebugAssertFalse(is_allocated_gpu_density_map, "Gpu density map is already allocated");
        gpu_density_map.Init(temp_image, false, false);
        gpu_density_map.CopyHostToDeviceTextureComplex3d( );
        is_allocated_gpu_density_map = true;
    }

    return;
};

void ProjectionComparisonObjects::PrepareGpuImages(Particle& host_particle, Image& host_projection_image, const bool is_for_global_search, c_img_t image_type) {

    MyDebugAssertTrue(host_particle.particle_image->is_in_memory, "Particle is not in memory");
    MyDebugAssertTrue(host_projection_image.is_in_memory, "Projection is not in memory");

    constexpr bool pin_host_memory               = true;
    constexpr bool allocate_gpu_memory_if_needed = true;

    // Let's see which image we are dealing with, check to see if it has been modified since we last (or never) interactied with it and also
    // reset the associated Particle flag to checkpoint our new association with the GpuImage.
    bool host_particle_data_has_changed;
    bool gpu_memory_was_changed;
    switch ( image_type ) {
        case c_img_t::particle_image_t: {

            // To avoid a bunch of redundant checks, we'll assign some temporary pointers
            GpuImage* tmp_gpu_projection     = is_for_global_search ? &gpu_search_projection : &gpu_projection;
            GpuImage* tmp_gpu_particle_image = is_for_global_search ? &gpu_search_particle_image : &gpu_particle_image;

            // Init checks for equivalent size and whether or not to allocate.
            gpu_memory_was_changed         = tmp_gpu_particle_image->Init(*host_particle.particle_image, pin_host_memory, allocate_gpu_memory_if_needed);
            host_particle_data_has_changed = host_particle.HasParticleImageDataChanged( );

            // If we altered the gpu memory, or if the host particle has recorded a change to its underlying data, we need to copy host - > device.
            if ( gpu_memory_was_changed || host_particle_data_has_changed ) {
                tmp_gpu_particle_image->CopyHostToDevice( ); // TODO: does this need to be synchronize?
            }

            // Now the same for the projection image, except we only care about it's size and pointer association, not the host data so no need for a copy.
            host_particle.RecordGpuParticleImageAssociation( );

            // for the projection, we don't care about the host data, just the size and pointer association.
            bool was_gpu_projection_memory_changed = tmp_gpu_projection->Init(host_projection_image, pin_host_memory, allocate_gpu_memory_if_needed);

#ifdef DEBUG
            int& particle_alloc                    = is_for_global_search ? n_search_particle_image_allocations : n_particle_image_allocations;
            int& projection_alloc                  = is_for_global_search ? n_search_projection_image_allocations : n_projection_image_allocations;
            int& calls_to                          = is_for_global_search ? n_calls_to_prep_search_images : n_calls_to_prep_images;
            int& particle_copy_calls               = is_for_global_search ? n_search_particle_image_HtoD_copies : n_particle_image_HtoD_copies;

            calls_to++;
            if ( gpu_memory_was_changed || host_particle_data_has_changed )
                particle_copy_calls++;
            if ( gpu_memory_was_changed )
                particle_alloc++;
            if ( was_gpu_projection_memory_changed )
                projection_alloc++;
#endif
            break;
        }

        case c_img_t::ctf_image_t: {
            // To avoid a bunch of redundant checks, we'll assign some temporary pointers
            GpuImage* tmp_gpu_ctf = is_for_global_search ? &gpu_search_ctf_image : &gpu_ctf_image;

            // Note: we don't want to allocate the real values here since we'll use the fp16 buffer on the GpuImage
            gpu_memory_was_changed         = tmp_gpu_ctf->Init(*host_particle.ctf_image, pin_host_memory, false);
            host_particle_data_has_changed = host_particle.HasCTFImageDataChanged( );

            // If we altered the gpu memory, or if the host particle has recorded a change to its underlying data, we need to copy host - > device.
            if ( gpu_memory_was_changed || host_particle_data_has_changed ) {
                tmp_gpu_ctf->CopyHostToDevice16f( );
            }
            host_particle.RecordGpuCTFImageAssociation( );

#ifdef DEBUG
            int& ctf_alloc      = is_for_global_search ? n_search_ctf_image_allocations : n_ctf_image_allocations;
            int& calls_to       = is_for_global_search ? n_calls_to_prep_search_ctf_images : n_calls_to_prep_ctf_images;
            int& ctf_copy_calls = is_for_global_search ? n_search_ctf_image_HtoD_copies : n_ctf_image_HtoD_copies;

            calls_to++;
            if ( gpu_memory_was_changed || host_particle_data_has_changed )
                ctf_copy_calls++;
            if ( gpu_memory_was_changed )
                ctf_alloc++;
#endif

            break;
        }

        case c_img_t::beamtilt_image_t: {
            MyDebugAssertTrue(false, "Beamtilt image is not supported yet");
            // host_particle_data_has_changed = host_particle.HasBeamTiltImageDataChanged( );
            // host_particle.RecordGpuBeamTiltImageAssociation( );

            break;
        }

        default: {
            // TODO: send error or something
            MyDebugAssertTrue(false, "Unknown image type");
            break;
        }
    }
}

void ProjectionComparisonObjects::PrepareGpuCTFImages(Particle& host_particle, const bool is_for_global_search) {
    MyDebugAssertTrue(host_particle.ctf_image->is_in_memory, "CTF is not in memory");
    MyDebugAssertTrue(host_particle.ctf_image_calculated, "CTF is not calculated");

    // This is often going to be in a brute force loop where the defocus is being iterated over, so we have a separate function to avoid all the checks on the projection/particle image.
    PrepareGpuImages(host_particle, *host_particle.ctf_image, is_for_global_search, c_img_t::ctf_image_t);

    // TODO: not yet implemented, also add DebugAsserts
    // PrepareGpuImages(host_particle, host_particle.host_beamtilt_image, is_for_global_search, c_img_t::beamtilt_image_t);
}

float ProjectionComparisonObjects::DoGpuProjection( ) {
    MyDebugAssertTrue(gpu_density_map.is_allocated_texture_cache, "gpu_density_map is not allocated");
    MyDebugAssertTrue(gpu_projection.is_in_memory_gpu, "gpu_projection is not allocated");
    MyDebugAssertTrue(gpu_particle_image.is_in_memory_gpu, "gpu_particle_image is not allocated");
    // wxPrintf("Is host memory pinned (%d\n)", gpu_projection->is_host_memory_pinned);

    gpu_projection.ExtractSliceShiftAndCtf(&gpu_density_map, &gpu_ctf_image, particle->alignment_parameters, reference_volume->pixel_size, particle->pixel_size / particle->filter_radius_high, true,
                                           swap_quadrants, apply_shifts, apply_ctf, absolute_ctf);
    if ( whiten ) {
        gpu_projection.Whiten(particle->pixel_size / particle->filter_radius_high);
    }
    if ( mask_radius > 0.f ) {
        // CosineMask is not implemented yet, but we can at least do the backFFT
        gpu_projection.BackwardFFT( );
    }

    // #ifdef CISTEM_PROFILING
    //     // When profiling, we want to distinguish between projection time and copy time
    //     // We still have synchronization in the weighted correlation methods.
    //     cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
    // #endif

    // #if defined(COMPARE_GPU_CPU_SCORE) || defined(CALCULATE_SCORE_ON_CPU_pcos)
    //     gpu_projection.CopyDeviceToHostAndSynchronize(false, false);
    // #endif

    float filter_radius_high = fminf(powf(particle->pixel_size / particle->filter_radius_high, 2), 0.25);
    float filter_radius_low  = 0.0f;
    if ( particle->filter_radius_low != 0.0 )
        filter_radius_low = powf(particle->pixel_size / particle->filter_radius_low, 2);

    // In the cpu method the bins < 1 and > n_bins - 1 are ignored so incorporate this logic into the filter radius
    // for even sized images, this should just be dims.y
    int   number_of_bins  = gpu_projection.dims.y / 2 + 1;
    float number_of_bins2 = 2 * (number_of_bins - 1);
    filter_radius_low     = std::max(filter_radius_low, powf(1.f / number_of_bins2, 2));
    filter_radius_high    = std::min(filter_radius_high, powf(float(number_of_bins - 1) / number_of_bins2, 2));
    // int bin             = int(sqrtf(frequency_sq) * number_of_bins2);

    if ( ! is_allocated_weighted_correlation_buffers ) {
        buffer_cross_terms.Allocate(gpu_projection.dims.w / 2, gpu_projection.dims.y, 1, true);
        buffer_image_ps.CopyFrom(&buffer_cross_terms);
        buffer_projection_ps.CopyFrom(&buffer_cross_terms);
        is_allocated_weighted_correlation_buffers = true;
    }
    float tmp_corr = gpu_particle_image.GetWeightedCorrelationWithImage(gpu_projection, buffer_cross_terms, buffer_image_ps, buffer_projection_ps, filter_radius_low, filter_radius_high, particle->pixel_size / particle->signed_CC_limit);

    return tmp_corr;
};

#endif
