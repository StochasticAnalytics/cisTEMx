
#include "gpu_core_headers.h"
#include "gpu_indexing_functions.h"

#include "GpuImage.h"
#include "template_matching_empricial_distribution.h"
#include "../constants/constants.h"

using namespace cistem::match_template as TM;

/**
 * @brief Construct a new TM_EmpiricalDistribution
 * Note: both histogram_min and histogram step must be > 0 or no histogram will be created
 * Note: the number of histogram bins is fixed by TM::histogram_number_of_points
 * 
 * @param reference_image - used to determine the size of the input images and set gpu launch configurations
 * @param histogram_min - the minimum value of the histogram
 * @param histogram_step - the step size of the histogram
 * @param n_images_to_accumulate_concurrently - the number of images to accumulate concurrently
 * 
 */
template <InputType input_type, bool per_image>
TM_EmpiricalDistribution::TM_EmpiricalDistribution(GpuImage&           reference_image,
                                                   histogram_storage_t histogram_min,
                                                   histogram_storage_t histogram_step,
                                                   int                 n_border_pixels_to_ignore_for_histogram,
                                                   int                 n_images_to_accumulate_concurrently,
                                                   cudaStream_t*       calc_stream) : n_images_to_accumulate_concurrently_{n_images_to_accumulate_concurrently},
                                                                                n_border_pixels_to_ignore_for_histogram_{n_border_pixels_to_ignore_for_histogram},
                                                                                reference_image_{reference_image},
                                                                                calc_stream_{calc_stream} {

    static_assert(per_image == false, "This class does not support per image accumulation yet");

    // I suspect we'll move to bfloat16 for the input data, as it was not available at the time the
    // original code was implemented. The extended dynamic range, and ease of conversion to/from histogram_storage_t
    // are likely a benefit, while the further reduced precision is unlikely to be a problem in the raw data values.
    // If anything, given that the output of the matched filter is ~ Gaussian, all the numbers closer to zero are less
    // likely to be flushed to zero when denormal, so in that respect, bflaot16 may actually maintain higher precision.
    if constexpr ( std::is_same_v<InputType, __half> ) {
        histogram_min_  = __float2half_rn(histogram_min);
        histogram_step_ = __float2half_rn(histogram_step);
    }
    else if constexpr ( std::is_same_v<InputType, __nv_bfloat16> ) {
        histogram_min_  = __float2bfloat16_rn(histogram_min);
        histogram_step_ = __float2bfloat16_rn(histogram_step);
    }
    else if constexpr ( std::is_same_v<InputType, histogram_storage_t> ) {
        histogram_min_  = histogram_min;
        histogram_step_ = histogram_step;
    }
    else {
        MyDebugAssertTrue(false, "input_type must be either __half __nv_bfloat16, or histogram_storage_t");
    }

    if ( histogram_min_ > 0.0f && histogram_step_ > 0.0f ) {
        MyDebugAssertTrue(TM::histogram_number_of_points <= 1024, "The histogram kernel assumes <= 1024 threads per block");
        MyDebugAssertTrue(TM::histogram_number_of_points % cistem::gpu::warp_size == 0, "The histogram kernel assumes a multiple of 32 threads per block");
        histogram_n_bins_ = TM::histogram_number_of_points;
    }
    else {
        // will be used as check on which kernels to call
        histogram_n_bins_ = 0;
    }

    image_dims_.x = reference_image.dims.x;
    image_dims_.y = reference_image.dims.y;
    image_dims_.z = reference_image.dims.z;
    image_dims_.w = reference_image.dims.w;

    MyDebugAssertTrue(image_dims_.x > 0 && image_dims_.y > 0 && image_dims_.z > 0 && image_dims_.w > 0, "Image dimensions must be > 0");

    // Set-up the launch configuration - assumed to be a real space image.
    // WARNING: this is up to the developer to ensure, as we'll use pointers for the input arrays
    // Note: we prefer the "1d" grid as a NxN patch is more likely to have similar values than a N^2x1 line, and so more atomic collisions in the histogram kernel.
    reference_image_.ReturnLaunchParameters<TM::histogram_number_of_points, 1>(image_dims_, true);
    gridDims_  = reference_image_.gridDims;
    blockDims_ = reference_image_.blockDims;

    // Every block will have a shared memory array of the size of the number of bins and aggregate those into their own
    // temp arrays. Only at the end of the search will these be added together'

    // Array of temporary storage to accumulate the shared mem to
    cudaErr(cudaMallocAsync(&histogram_,
                            gridDims_img.x * gridDims_img.y * TM::histogram_number_of_points * sizeof(histogram_storage_t), *calc_stream_)));
};

template <InputType input_type, bool per_image>
TM_EmpiricalDistribution::~TM_EmpiricalDistribution( ) {
    cudaErr(cudaFreeAsync(histogram_, *calc_stream_));
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Kernels and inline helper functions called from EmpiricalDistribution::AccumulateDistribution
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline __device__ int convert_input(T* input_ptr, int x, int y, int NW, T bin_min, T bin_inc) {
    if constexpr ( std::is_same_v<T, __half> )
        return __half2int_rd((input_ptr[y * NW + x] - bin_min) / bin_inc);
    if constexpr ( std::is_same_v<T, __nv_bfloat16> )
        return __bfloat162int_rd((input_ptr[y * NW + x] - bin_min) / bin_inc);
    if constexpr ( std::is_same_v<T, histogram_storage_t> )
        return __float2int_rd((input_ptr[y * NW + x] - bin_min) / bin_inc);
}

template <InputType input_type>
__global__ void
histogram_smem_atomics(const InputType* __restrict__ input_ptr,
                       histogram_storage_t* output_ptr,
                       int4                 dims,
                       const InputType      bin_min,
                       const InputType      bin_inc,
                       const int            max_padding,
                       const int            n_slices_to_process) {

    // initialize temporary accumulation array input_ptr shared memory, this is equal to the number of bins input_ptr the histogram,
    // which may  be more or less than the number of threads input_ptr a block
    __shared__ int smem[TM::histogram_number_of_points];

    // Each block has it's own copy of the histogram stored input_ptr global memory, found at the linear block index
    histogram_storage_t* stored_array = &output_ptr[LinearBlockIdx_2dGrid( ) * TM::histogram_number_of_points];

    // Since the number of x-threads is enforced to be = to the number of bins, we can just copy the bins to shared memory
    // Otherwise, we would need a loop to copy the bins to shared memory e.g. -> for ( int i = threadIdx.x; i < TM::histogram_number_of_points; i += BlockDimension_2d( ) )
    smem[i] = __float2int_rn(stored_array[i]);
    __syncthreads( );

    int pixel_idx;
    int previous_pixel_idx;
    int n_counts = 0;
    // updates our block's partial histogram input_ptr shared memory

    for ( int j = max_padding + physical_Y( ); j < dims.y - max_padding; j += blockDim.y * gridDim.y ) {
        for ( int i = max_padding + physical_X( ); i < dims.x - max_padding; i += blockDim.x * gridDim.x ) {
            for ( int k = 0; k < n_slices_to_process; k++ ) {
                pixel_idx = convert_input(input_ptr, i, j, dims.w, bin_min, bin_inc);
                // we have to check n_counts first otherwise the results are undefined on the first pass.
                if ( n_counts > 0 && pixel_idx != previous_pixel_idx ) {
                    atomicAdd(&smem[previous_pixel_idx], n_counts);
                    n_counts = 0;
                }
                else {
                    n_counts++;
                }
                previous_pixel_idx = pixel_idx;
            }
        }
    }
    // We have to do a final cleanup in case we've been accumulating the same value:
    if ( n_counts > 0 ) {
        atomicAdd(&smem[previous_pixel_idx], n_counts);
    }
    __syncthreads( );

    // write partial histogram into the global memory
    // Converting to long was super slow. Given that I don't care about representing the number exactly, but do care about overflow, just switch the bins to histogram_storage_t
    // As in the read case, we would need a loop if the number of threads != number of bins e.g. -> for ( int i = threadIdx.x; i < TM::histogram_number_of_points; i += BlockDimension_2d( ) )
    stored_array[i] = __int2float_rn(smem[i]);
}

template <InputType input_type>
__global__ void
histogram_final_accum(histogram_storage_t* input_ptr, int n_bins, int n_blocks) {

    int lIDX = physical_X_1d_grid( );

    if ( lIDX < n_bins ) {
        histogram_storage_t total{0.0};
        for ( int j = 0; j < n_blocks; j++ ) {
            total += input_ptr[lIDX + n_bins * j];
        }
        // We accumulate all histograms into the first block
        input_ptr[lIDX] = total;
    }
}

/**
 * @brief Accumulate new values into the pixel wise distribution.
 * If set to record a histogram, a fused kernal will be called to accumulate the histogram and the pixel wise distribution
 * If set to track 3rd and 4th moments of the distribution, a fused kernel will be called to accumulate the moments and the pixel wise distribution
 * 
 * @param input_data - pointer to the input data to accumulate, a stack of images.
 * @param n_images_this_batch - number of slices to accumulate, must be <= n_images_to_accumulate_concurrently
 */
template <InputType input_type, bool per_image>
void TM_EmpiricalDistribution::AccumulateDistribution(InputType* input_data, int n_images_this_batch) {
    MyDebugAssertTrue(input_data, "The data to acmmulate is not input_ptr memory.");
    MyDebugAssertTrue(n_images_this_batch <= n_images_to_accumulate_concurrently_, "The number of images to accumulate is greater than the number of images to accumulate concurrently");

    if ( histogram_n_bins_ == 0 ) {
        precheck;
        histogram_smem_atomics<<<gridDims_, blockDims_, 0, *calc_stream_>>>(
                input_data,
                histogram_,
                image_dims_,
                histogram_min_,
                histogram_step_,
                n_border_pixels_to_ignore_for_histogram_);
        postcheck;
    }
    else if ( higher_order_moments_ ) {
        MyDebugAssertTrue(false, "Skew and kurtosis not implemented yet");
        // call the pixel wise kernel
    }
    else {
        MyDebugAssertFalse(true, "This should never happen");
        precheck;
        histogram_and_stats_smem_atomics<<<gridDims_, blockDims_, 0, *calc_stream_>>>(
                input_data,
                image_dims_,
                histogram_,
                histogram_min_,
                histogram_step_,
                n_border_pixels_to_ignore_for_histogram_);
        postcheck;
    }
};

template <InputType input_type, bool per_image>
void TM_EmpiricalDistribution::FinalAccumulate( ) {
    precheck;
    histogram_final_accum<<<gridDims_, blockDims_, 0, *calc_stream_>>>(histogram_, TM::histogram_number_of_points, gridDims_.x * gridDims_.y);
    postcheck;
}

template <InputType input_type, bool per_image>
void TM_EmpiricalDistribution::CopyToHostAndAdd(long* array_to_add_to) {

    // Make a temporary copy of the cummulative histogram on the host and then add on the host. TODO errorchecking
    histogram_storage_t* tmp_array;
    cudaErr(cudaMallocHost(&tmp_array, TM::histogram_number_of_points * sizeof(histogram_storage_t)));
    cudaErr(cudaMemcpy(tmp_array, histogram_, TM::histogram_number_of_points * sizeof(histogram_storage_t), cudaMemcpyDeviceToHost));

    for ( int iBin = 0; iBin < TM::histogram_number_of_points; iBin++ ) {
        array_to_add_to[iBin] += long(tmp_array[iBin]);
    }

    cudaErr(cudaFreeHost(tmp_array));
}