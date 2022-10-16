#include <cistem_config.h>

#include "../../gpu/gpu_core_headers.h"
#include "../../gpu/GpuImage.h"

// #define ELECTRON_DOSE_DEBUG_PRINT

class StealStdoutBackFromWX {

    //   private:

  public:
    StealStdoutBackFromWX(const char* file_name) : _my_file(fopen(file_name, "w")) {
        _store_original_stdout_ptr = stdout;
        stdout                     = _my_file;
    };

    ~StealStdoutBackFromWX( ) { RestoreStdoutToWX( ); };

    void FlushCapturedStdout( ) { fflush(stdout); };

    inline void RestoreStdoutToWX( ) { stdout = _store_original_stdout_ptr; };

    FILE* _store_original_stdout_ptr;
    FILE* _my_file;
};

__device__ __inline__ float
ReturnCriticalDose(float spatial_frequency, float voltage_scaling_factor) {

    return (cistem::electron_dose::critical_dose_a * powf(spatial_frequency, cistem::electron_dose::reduced_critical_dose_b) + cistem::electron_dose::critical_dose_c) * voltage_scaling_factor;
};

__device__ __inline__ float
ReturnDoseFilter(float dose_at_end_of_frame, float critical_dose) {
    return expf((-0.5 * dose_at_end_of_frame) / critical_dose);
};

__device__ __inline__ float
ReturnCummulativeDoseFilter(float dose_at_start_of_exposure, float dose_at_end_of_exosure, float critical_dose) {
    // The integrated exposure. Included in particular for the matched filter.
    // Calculated on Wolfram Alpha = integrate exp[ -0.5 * (x/a) ] from x=0 to x=t
    return 2.0f * critical_dose * (exp((-0.5 * dose_at_start_of_exposure) / critical_dose) - exp((-0.5 * dose_at_end_of_exosure) / critical_dose)) / dose_at_end_of_exosure;
};

__global__ void
ApplyDoseFilterKernel(float2**    image_data,
                      float       pre_exposure,
                      const float dose_per_frame,
                      float2*     output_data,
                      const float voltage_scaling_factor,
                      const float fourier_voxel_size_x,
                      const float fourier_voxel_size_y,
                      const int   pixel_pitch,
                      const int   NY,
                      const int   NZ,
                      const int   physical_index_of_first_negative_frequency_y) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if ( x >= pixel_pitch )
        return;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ( y >= NY )
        return;

    const int address = x + y * pixel_pitch;
    // logical fourier index
    y        = (y >= physical_index_of_first_negative_frequency_y) ? (y - NY) : y;
    float ky = float(y) * fourier_voxel_size_y;
    ky *= ky;

    float kx = float(x) * fourier_voxel_size_x;
    kx       = kx * kx + ky;

    float real_sum = 0.f;
    float imag_sum = 0.f;
    float filter_coeff;

    for ( int z = 0; z < NZ; z++ ) {
#ifdef ELECTRON_DOSE_DEBUG_PRINT
        printf("Index is %i, %i, %i, %i \n", x, y, z, address); //, image_data[z].complex_values_gpu[address].x);
#endif

        filter_coeff = ReturnDoseFilter(pre_exposure, ReturnCriticalDose(kx, voltage_scaling_factor));
        real_sum += image_data[z][address].x * filter_coeff;
        imag_sum += image_data[z][address].y * filter_coeff;
        pre_exposure += dose_per_frame;
    }

    // #ifdef ELECTRON_DOSE_DEBUG_PRINT
    //     printf("Index is %i, %i, %i, %i \n", x, y, 0, address); //, image_data[z].complex_values_gpu[address].x);
    // #endif

    if ( address == 0 ) {
        output_data[0].x = 1.0f;
        output_data[0].y = 0.0f;
    }
    else {
        output_data[address].x = real_sum;
        output_data[address].y = imag_sum;
    }
};

__global__ void ApplyDoseFilterAndRestorePowerKernel(float2**    image_data,
                                                     float       pre_exposure,
                                                     const float dose_per_frame,
                                                     float2*     output_data,
                                                     const float voltage_scaling_factor,
                                                     const float fourier_voxel_size_x,
                                                     const float fourier_voxel_size_y,
                                                     const int   pixel_pitch,
                                                     const int   NY,
                                                     const int   NZ,
                                                     const int   physical_index_of_first_negative_frequency_y) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if ( x >= pixel_pitch )
        return;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ( y >= NY )
        return;

    const int address = x + y * pixel_pitch;
    // logical fourier index
    y        = (y >= physical_index_of_first_negative_frequency_y) ? (y - NY) : y;
    float ky = float(y) * fourier_voxel_size_y;
    ky *= ky;

    float kx = float(x) * fourier_voxel_size_x;
    kx       = kx * kx + ky;

    float real_sum = 0.f;
    float imag_sum = 0.f;
    float filter_coeff;
    float sum_of_squares = 0.f;
    for ( int z = 0; z < NZ; z++ ) {
        filter_coeff = ReturnDoseFilter(pre_exposure, ReturnCriticalDose(kx, voltage_scaling_factor));
        sum_of_squares += (filter_coeff * filter_coeff);
        real_sum += image_data[z][address].x * filter_coeff;
        imag_sum += image_data[z][address].y * filter_coeff;
        pre_exposure += dose_per_frame;
    }

    // This should never be zero
    sum_of_squares = sqrtf(sum_of_squares);

    if ( address == 0 ) {
        output_data[0].x = 1.0f;
        output_data[0].y = 0.0f;
    }
    else {
        output_data[address].x = real_sum / sum_of_squares;
        output_data[address].y = imag_sum / sum_of_squares;
    }
};

template <>
void ElectronDose::CalculateDoseFilterAs1DArray(std::vector<GpuImage>& ref_image, float2* output_data, float pre_exposure, float exposure_per_frame, bool restore_power) {

    // Different than the CPU implementation, dose_start is assumed to be pre_exposure and dose per frame = dose_finish
    ref_image[0].ReturnLaunchParameters(ref_image[0].dims, false);

#ifdef ELECTRON_DOSE_DEBUG_PRINT
    StealStdoutBackFromWX my_stdout("/tmp/gpulog_class.txt");
#endif
    ref_image[0].ptr_array_32fc.resize(ref_image.size( ));
    for ( int iPtr = 0; iPtr < ref_image.size( ); iPtr++ ) {
        ref_image[0].ptr_array_32fc.SetPointer((float2*)ref_image[iPtr].complex_values_gpu, iPtr);
    }
    if ( restore_power ) {
        precheck
                ApplyDoseFilterAndRestorePowerKernel<<<ref_image[0].gridDims, ref_image[0].threadsPerBlock, 0, cudaStreamPerThread>>>(ref_image[0].ptr_array_32fc.ptr_array,
                                                                                                                                      pre_exposure,
                                                                                                                                      exposure_per_frame,
                                                                                                                                      output_data,
                                                                                                                                      voltage_scaling_factor,
                                                                                                                                      ref_image[0].fourier_voxel_size.x / pixel_size,
                                                                                                                                      ref_image[0].fourier_voxel_size.y / pixel_size,
                                                                                                                                      ref_image[0].dims.w / 2,
                                                                                                                                      ref_image[0].dims.y,
                                                                                                                                      ref_image.size( ),
                                                                                                                                      ref_image[0].physical_index_of_first_negative_frequency.y);
        postcheck
    }
    else {
        precheck
                ApplyDoseFilterKernel<<<ref_image[0].gridDims, ref_image[0].threadsPerBlock, 0, cudaStreamPerThread>>>(ref_image[0].ptr_array_32fc.ptr_array,
                                                                                                                       pre_exposure,
                                                                                                                       exposure_per_frame,
                                                                                                                       output_data,
                                                                                                                       voltage_scaling_factor,
                                                                                                                       ref_image[0].fourier_voxel_size.x / pixel_size,
                                                                                                                       ref_image[0].fourier_voxel_size.y / pixel_size,
                                                                                                                       ref_image[0].dims.w / 2,
                                                                                                                       ref_image[0].dims.y,
                                                                                                                       ref_image.size( ),
                                                                                                                       ref_image[0].physical_index_of_first_negative_frequency.y);
#ifdef ELECTRON_DOSE_DEBUG_PRINT
        cudaDeviceSynchronize( );
        cudaDeviceReset( );
        my_stdout.FlushCapturedStdout( );
        exit(0);
#endif

        postcheck
    }

    //	MyDebugAssertTrue(ref_image->logical_z_dimension == 1, "Reference Image is a 3D!");
}