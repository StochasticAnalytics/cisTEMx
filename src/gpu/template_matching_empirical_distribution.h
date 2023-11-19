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

namespace TM_AccumulationType {
enum Enum : int { HistogramOnly,
                  HistogramAndMip,
                  HistogramAndMipAndHigherOrderMoments };

} // namespace TM_AccumulationType

template <typename ccfType, typename mipType, bool per_image = false>
class TM_EmpiricalDistribution {

  private:
    bool      higher_order_moments_;
    ccfType   histogram_min_;
    ccfType   histogram_step_;
    int       histogram_n_bins_;
    int       padding_x_;
    int       padding_y_;
    const int n_images_to_accumulate_concurrently_;
    int       current_mip_to_process_;
    long      total_mips_processed_;

    histogram_storage_t* sum_array_;
    histogram_storage_t* sum_sq_array_;
    mipType*             mip_psi_;
    mipType*             theta_phi_;
    ccfType*             psi_theta_phi_;
    ccfType*             d_psi_theta_phi_;
    ccfType*             ccf_array_;

    dim3 threadsPerBlock_;
    dim3 gridDims_;

    int4 image_dims_;
    int  image_n_elements_allocated_;

    histogram_storage_t* histogram_;
    cudaStream_t         calc_stream_; // Managed by some external resource
    cudaEvent_t          mip_is_done_Event_;

  public:
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
    TM_EmpiricalDistribution(GpuImage&            reference_image,
                             histogram_storage_t* sum_array,
                             histogram_storage_t* sum_sq_array,
                             int                  padding_x,
                             int                  padding_y,
                             const int            n_images_to_accumulate_before_final_accumulation,
                             cudaStream_t         calc_stream = cudaStreamPerThread);

    ~TM_EmpiricalDistribution( );

    void AccumulateDistribution(TM_AccumulationType::Enum accumulation_type);
    void FinalAccumulate( );
    void CopyToHostAndAdd(long* array_to_add_to);

    void SetCalcStream(cudaStream_t calc_stream) {
        MyDebugAssertFalse(cudaStreamQuery(calc_stream_) == cudaErrorInvalidResourceHandle, "The cuda stream is invalid");
        calc_stream_ = calc_stream;
    }

    void AllocateStatisticsArrays( );
    void DeallocateStatisticsArrays( );
    void CopyStatisticsToArrays(float* output_mip, float* output_psi, float* output_theta, float* output_phi);
    void AddValues(ccfType psi, ccfType theta, ccfType phi);
    void CopyPsiThetaPhiHostToDevice( );

    // The reason to have one array is to reduce calls to memcopy, for which the API overhead is measurable.
    inline ccfType* GetHostPsiPtr( ) { return &psi_theta_phi_[0]; };

    inline ccfType* GetHostThetaPtr( ) { return &psi_theta_phi_[n_images_to_accumulate_concurrently_]; };

    inline ccfType* GetHostPhiPtr( ) { return &psi_theta_phi_[2 * n_images_to_accumulate_concurrently_]; };

    inline ccfType* GetDevicePsiPtr( ) { return &d_psi_theta_phi_[0]; };

    inline ccfType* GetDeviceThetaPtr( ) { return &d_psi_theta_phi_[n_images_to_accumulate_concurrently_]; };

    inline ccfType* GetDevicePhiPtr( ) { return &d_psi_theta_phi_[2 * n_images_to_accumulate_concurrently_]; };

    inline ccfType* GetDeviceCCFPtr( ) { return &ccf_array_[current_mip_to_process_]; };
};

#endif