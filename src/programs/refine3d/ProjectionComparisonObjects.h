/*

Written by: Ben Himes at some time before now.

The goal for this class is to organize and update a series of pointers during the various cycles and enumerations of particle refinement in refine3d.

The reason this is needed, is partly for convenience but also for efficiency and thread safety. This object stores pointers to several types of image objects, 
optionally gpu image objects as well and is passed by reference (via a pointer) to the conjugate gradient minimizer and the FrealignObjective function.

We also use this to drop in some gpu methods that are ifdef'd here so that the (already hard to read) code in refine3d doesn't get harder to read.

The other primary function of this class is to ensure parity between GpuImage and Image objects. The gpu integration is implemented in a fashion,
to only do what is strictly necessary, while cpu Image objects still handle much of the work. This means both the meta-data defining an image object, as well as 
the actual data itself may fall out of sync with an associated GpuImage object.

*/

#ifndef _SRC_PROGRAMS_REFINE3D_PROJECTION_COMPARISON_OBJECTS_H_
#define _SRC_PROGRAMS_REFINE3D_PROJECTION_COMPARISON_OBJECTS_H_

#ifdef ENABLEGPU
#warning "Experimental GPU code from ProjectionComparisonObjects.h will be used in refine3d_gpu"
#else
#warning "CPU code is not yet implemented"
class GpuImage;
#endif

class ProjectionComparisonObjects {

  public:
    ProjectionComparisonObjects( );
    ~ProjectionComparisonObjects( );
    void DummyMethod( );

    Particle*            particle;
    ReconstructedVolume* reference_volume;
    Image*               projection_image;

#ifdef ENABLEGPU

    GpuImage* gpu_density_map;
    GpuImage* gpu_projection;
    GpuImage* gpu_ctf_image;
    GpuImage* gpu_particle_image;

#endif

    // These are used in the projection step. ifndef ENABLEGPU, they are just no-ops.
    float DoGpuProjection( );
    void  PrepareGpuImagesProjection(GpuImage* external_gpu_projection);
    void  PrepareGpuVolumeProjection(ReconstructedVolume& input_density, GpuImage* external_gpu_volume);

  private:
    float  mask_radius;
    float  mask_falloff;
    bool   swap_quadrants, apply_shifts, whiten, apply_ctf, absolute_ctf;
    float* score_buffer;
    int    score_buffer_size;
    bool   is_set_gpu_ctf_image;
    bool   copy_gpu_particle_image;
    bool   copy_gpu_search_particle_image;

    int   nprj;
    float x_shift_limit;
    float y_shift_limit;
    float angle_change_limit;

    float initial_x_shift;
    float initial_y_shift;
    float initial_psi_angle;
    float initial_phi_angle;
    float initial_theta_angle;
};

#endif // _SRC_PROGRAMS_REFINE3D_PROJECTION_COMPARISON_OBJECTS_H_
