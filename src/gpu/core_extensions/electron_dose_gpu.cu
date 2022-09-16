#include <cistem_config.h>

#include "../../gpu/gpu_core_headers.h"
#include "../../gpu/GpuImage.h"

__device__ __inline__ ReturnCriticalDose(float spatial_frequency, float voltage_scaling_factor) {
    return (cistem::electron_dose::critical_dose_a * powf(spatial_frequency, cistem::electron_dose::reduced_critical_dose_b) + cistem::electron_dose::critical_dose_c) * voltage_scaling_factor;
}

__device__ __inline__ ReturnDoseFilter(float dose_at_end_of_frame, float critical_dose) {
    return expf((-0.5 * dose_at_end_of_frame) / critical_dose);
}

__device__ __inline__ ReturnCummulativeDoseFilter(float dose_at_start_of_exposure, float dose_at_end_of_exosure, float critical_dose) {
    // The integrated exposure. Included in particular for the matched filter.
    // Calculated on Wolfram Alpha = integrate exp[ -0.5 * (x/a) ] from x=0 to x=t
    return 2.0f * critical_dose * (exp((-0.5 * dose_at_start_of_exposure) / critical_dose) - exp((-0.5 * dose_at_end_of_exosure) / critical_dose)) / dose_at_end_of_exosure;
}

__global__ void ApplyDoseFilterKernel(const float* __restrict___ image_data,
                                      float                      pre_exposure,
                                      const float                dose_per_frame,
                                      float2*                    output_data,
                                      const float                pixel_size,
                                      const float                voltage_scaling_factor,
                                      const float                fourier_voxel_size_x,
                                      const float                fourier_voxel_size_y,
                                      const float                pixel_size_sq,
                                      const int                  pixel_pitch,
                                      const int                  NY,
                                      const int                  NZ,
                                      const int                  physical_index_of_first_negative_frequency_y) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if ( x > pixel_pitch )
        return;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ( y > NY )
        return;

    int address = x + y * pixel_pitch;
    // logical fourier index
    y = (y >= physical_index_of_first_negative_frequency_y) ? (y - NY) : y;
    y *= fourier_voxel_size_y;
    y *= y;

    x *= fourier_voxel_size_x;
    x = x * x + y;

    // reuse y for output address
    y = address;

    float real_sum = 0.f;
    float imag_sum = 0.f;
    float filter_coeff;
    for ( int z = 0; z < NZ; z++ ) {
        filter_coeff = ReturnDoseFilter(pre_exposure, ReturnCriticalDose(x, voltage_scaling_factor));
        real_sum += image_data[address] * filter_coeff;
        imag_sum += image_data[address + 1] * filter_coeff;
        address += pixel_pitch * NY;
        pre_exposure += dose_per_frame;
    }

    if ( x == 0 && y == 0 )
        output_array[0].x = 1.0;
    output_array[0].y = 0.0;
    else {
        output_data[y].x = real_sum;
        output_data[y].y = imag_sum;
    }
};

__global__ void ApplyDoseFilterAndRestorePowerKernel(const float* __restrict___ image_data,
                                                     float                      pre_exposure,
                                                     const float                dose_per_frame,
                                                     float2*                    output_data,
                                                     const float                pixel_size,
                                                     const float                voltage_scaling_factor,
                                                     const float                fourier_voxel_size_x,
                                                     const float                fourier_voxel_size_y,
                                                     const float                pixel_size_sq,
                                                     const int                  pixel_pitch,
                                                     const int                  NY,
                                                     const int                  NZ,
                                                     const int                  physical_index_of_first_negative_frequency_y) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if ( x > pixel_pitch )
        return;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ( y > NY )
        return;

    int address = x + y * pixel_pitch;
    // logical fourier index
    y = (y >= physical_index_of_first_negative_frequency_y) ? (y - NY) : y;
    y *= fourier_voxel_size_y;
    y *= y;

    x *= fourier_voxel_size_x;
    x = x * x + y;

    // reuse y for output address
    y = address;

    float real_sum = 0.f;
    float imag_sum = 0.f;
    float filter_coeff;
    float sum_of_squares = 0.f;
    for ( int z = 0; z < NZ; z++ ) {
        filter_coeff   = ReturnDoseFilter(pre_exposure, ReturnCriticalDose(x, voltage_scaling_factor));
        sum_of_squares = filter_coeff * filter_coeff;
        real_sum += image_data[address] * filter_coeff;
        imag_sum += image_data[address + 1] * filter_coeff;
        address += pixel_pitch * NY;
        pre_exposure += dose_per_frame;
    }

    // This should never be zero
    sum_of_squares = sqrtf(sum_of_squares);

    if ( x == 0 && y == 0 ) {
        output_array[0].x = 1.0;
        output_array[0].y = 0.0;
    }
    else {
        output_data[y].x = real_sum / sum_of_squares;
        output_data[y].y = imag_sum / sum_of_squares;
    }
};

template <>
void ElectronDose::CalculateDoseFilterAs1DArray<GpuImage>(GpuImage* ref_image, float* filter_array, float dose_start, float dose_finish) {

    const float reduced_fourier_voxel_size = ref_image->fourier_voxel_size.x / pixel_size;
    const float pixel_size_sq              = pixel_size * pixel_size;

    float2* output_data = reinterpret_cast<float2*>(filter_array);

    // Different than the CPU implementation, dose_start is assumed to be pre_exposure and dose per frame = dose_finish -dose_start
    ref_image->ReturnLaunchParameters(ref_image->dims, false);
    precheck
            ApplyDoseFilterAndRestorePowerKernel<<<ref_image->gridDims, ref_image->threadsPerBlock, 0, cudaStreamPerThread>>>(ref_image->real_values_gpu,
                                                                                                                              dose_start,
                                                                                                                              dose_finish - dose_start,
                                                                                                                              float2 * output_data,
                                                                                                                              pixel_size,
                                                                                                                              voltage_scaling_factor,
                                                                                                                              ref_image->fourier_voxel_size.x,
                                                                                                                              ref_image->fourier_voxel_size.y,
                                                                                                                              pixel_size_sq,
                                                                                                                              ref_image->dims.w / 2,
                                                                                                                              ref_image->dims.y,
                                                                                                                              ref_image->dims.z,
                                                                                                                              ref_image->physical_index_of_first_negative_frequency.y);
    postcheck
    //	MyDebugAssertTrue(ref_image->logical_z_dimension == 1, "Reference Image is a 3D!");
}