#ifndef _SRC_GPU_TEMPLATE_MATCHING_EMPIRICAL_DISTRIBUTION_H_
#define _SRC_GPU_TEMPLATE_MATCHING_EMPIRICAL_DISTRIBUTION_H_

/**
 * @brief Construct a new tm empiricaldistribution object
 * 
 * @tparam input_type - this is the the data storage type holding the values to be tracked.
 * Internally, this class uses cascading summation in single precision (fp32)
 *  
 * @tparam per_image - this is a boolean flag indicating whether the class should track the statistics per image 
 * like the cpu version of EmpiricalDistribution or per pixel across many images.
 */

using histogram_storage_t = float;

template <InputType input_type, bool per_image = false>
class TM_EmpiricalDistribution( ) {

  private:
    int       current_image_index_;
    InputType histogram_min_;
    InputType histogram_step_;
    int       histogram_n_bins_;
    int       n_border_pixels_to_ignore_for_histogram_;

    dim3 threadsPerBlock_;
    dim3 gridDims_;

    int4 image_dims_;

    histogram_storage_t* histogram_;
    cudaStream_t*        calc_stream_; // Managed by some external resource

  public:
    TM_EmpiricalDistribution(GpuImage & reference_image,
                             histogram_storage_t histogram_min,
                             histogram_storage_t histogram_step,
                             int                 n_images_to_accumulate_concurrently = 1,
                             cudaStream_t        calc_stream                         = cudaStreamPerThread);

    ~TM_EmpiricalDistribution( );

    void AccumulateDistribution(InputType * input_data, int n_images_this_batch);
    void FinalAccumulate( );
}

#endif