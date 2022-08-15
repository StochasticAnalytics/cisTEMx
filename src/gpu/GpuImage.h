/*
 * GpuImage.h
 *
 *  Created on: Jul 31, 2019
 *      Author: himesb
 */

#ifndef GPUIMAGE_H_
#define GPUIMAGE_H_

#include "../core/cistem_constants.h"
#include "TensorManager.h"

class BatchedSearch;

class GpuImage {

  public:
    GpuImage( );
    GpuImage(const GpuImage& other_gpu_image); // copy constructor
    GpuImage(Image& cpu_image);
    ~GpuImage( );

    GpuImage& operator=(const GpuImage& t);
    GpuImage& operator=(const GpuImage* t);

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // START MEMBER VARIABLES FROM THE cpu IMAGE CLASS

    // TODO: These should mostly be made private since they are properties of the data and should not be modified unless a method modifies the data.
    int4 dims;
    // FIXME: Temporary for compatibility with the image class.
    int  logical_x_dimension, logical_y_dimension, logical_z_dimension;
    bool is_in_real_space; // !< Whether the image is in real or Fourier space
    bool object_is_centred_in_box; //!<  Whether the object or region of interest is near the center of the box (as opposed to near the corners and wrapped around). This refers to real space and is meaningless in Fourier space.
    bool is_fft_centered_in_box;
    int3 physical_upper_bound_complex;
    int3 physical_address_of_box_center;
    int3 physical_index_of_first_negative_frequency;
    int3 logical_upper_bound_complex;
    int3 logical_lower_bound_complex;
    int3 logical_upper_bound_real;
    int3 logical_lower_bound_real;

    int device_idx;
    int number_of_streaming_multiprocessors;
    int limit_SMs_by_threads;

    float3 fourier_voxel_size;

    int real_memory_allocated; // !<  Number of floats allocated in real space;
    int padding_jump_value; // !<  The FFTW padding value, if odd this is 2, if even it is 1.  It is used in loops etc over real space.
    int insert_into_which_reconstruction; // !<  Determines which reconstruction the image will be inserted into (for FSC calculation).

    int   number_of_real_space_pixels; // !<	Total number of pixels in real space
    float ft_normalization_factor; // !<	Normalization factor for the Fourier transform (1/sqrt(N), where N is the number of pixels in real space)
    // Arrays to hold voxel values

    float*               real_values; // !<  Real array to hold values for REAL images.
    std::complex<float>* complex_values; // !<  Complex array to hold values for COMP images.
    bool                 is_in_memory; // !<  Whether image values are in-memory, in other words whether the image has memory space allocated to its data array.
    bool                 image_memory_should_not_be_deallocated; // !< Don't deallocate the memory, generally should only be used when doing something funky with the pointers
    int                  gpu_plan_id;

    // end  MEMBER VARIABLES FROM THE cpu IMAGE CLASS
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    float*  real_values_gpu; // !<  Real array to hold values for REAL images.
    float2* complex_values_gpu; // !<  Complex array to hold values for COMP images.

    // To make it easier to switch between different types of FFT plans we have void pointers for them here
    void* position_space_ptr;
    void* momentum_space_ptr;

    // The half precision buffers may be used as fp16 or bfloat16 and it is up to the user to track what is what.
    // This of course assumes they have the same size, which they should.
    static constexpr size_t size_of_half = sizeof(__half);
    static_assert(size_of_half == sizeof(nv_bfloat16), "it is assumed sizeof(fp16) == sizeof(bfloat16)");
    static_assert(size_of_half * 2 == sizeof(nv_bfloat162), "it is assumed sizeof(fp16) == sizeof(bfloat16)");
    static_assert(size_of_half == sizeof(half_float::half), "GPU and CPU half precision types must be the same size");

    void* real_values_16f;
    void* complex_values_16f;
    void* ctf_buffer_16f;
    void* ctf_complex_buffer_16f;

    __half*  real_values_fp16        = reinterpret_cast<__half*>(real_values_16f);
    __half2* complex_values_fp16     = reinterpret_cast<__half2*>(complex_values_16f);
    __half*  ctf_buffer_fp16         = reinterpret_cast<__half*>(ctf_buffer_16f);
    __half2* ctf_complex_buffer_fp16 = reinterpret_cast<__half2*>(ctf_complex_buffer_16f);

    nv_bfloat16*  real_values_bf16        = reinterpret_cast<nv_bfloat16*>(real_values_16f);
    nv_bfloat162* complex_values_bf16     = reinterpret_cast<nv_bfloat162*>(complex_values_16f);
    nv_bfloat16*  ctf_buffer_bf16         = reinterpret_cast<nv_bfloat16*>(ctf_buffer_16f);
    nv_bfloat162* ctf_complex_buffer_bf16 = reinterpret_cast<nv_bfloat162*>(ctf_complex_buffer_16f);

    // We want to be able to re-use the texture object, so only set it up once.
    cudaTextureObject_t tex_real;

    cudaTextureObject_t tex_imag;
    cudaArray*          cuArray_real = 0;
    cudaArray*          cuArray_imag = 0;

    bool is_allocated_texture_cache;

    enum ImageType : size_t { real16f    = sizeof(__half),
                              complex16f = sizeof(__half2),
                              real32f    = sizeof(float),
                              complex32f = sizeof(float2),
                              real64f    = sizeof(double),
                              complex64f = sizeof(double2) };

    ImageType img_type;

    bool   is_in_memory_gpu; // !<  Whether image values are in-memory, in other words whether the image has memory space allocated to its data array. Default = .FALSE.
    bool   is_host_memory_pinned; // !<  Is the host memory already page locked (2x bandwith and required for asynchronous xfer);
    float* pinnedPtr;
    Image* host_image_ptr; // Primarily for access to host image methods for preprocessing.

    cudaMemcpy3DParms h_3dparams = {0};
    cudaExtent        h_extent;
    cudaPos           h_pos;
    cudaPitchedPtr    h_pitchedPtr;

    cudaMemcpy3DParms d_3dparams = {0};
    cudaExtent        d_extent;
    cudaPos           d_pos;
    cudaPitchedPtr    d_pitchedPtr;

    size_t pitch;

    dim3 threadsPerBlock;
    dim3 gridDims;

    bool    is_meta_data_initialized;
    float*  tmpVal;
    double* tmpValComplex;
    bool    is_in_memory_managed_tmp_vals;

    ////////////////////////////////////////////////////////

    cudaEvent_t npp_calc_event;
    bool        is_npp_calc_event_initialized;
    //	cublasHandle_t cublasHandle;

    cufftHandle cuda_plan_forward;
    cufftHandle cuda_plan_inverse;
    size_t      cuda_plan_worksize_forward;
    size_t      cuda_plan_worksize_inverse;

    int cufft_batch_size;

    //Stream for asynchronous command execution
    cudaStream_t     calcStream;
    cudaStream_t     copyStream;
    NppStreamContext nppStream;

    void PrintNppStreamContext( );

    //	bool is_cublas_loaded;
    bool is_npp_loaded;
    //	cublasStatus_t cublas_stat;
    NppStatus npp_stat;

    // For the full image set width/height, otherwise set on function call.
    NppiSize npp_ROI;
    NppiSize npp_ROI_real_space;
    NppiSize npp_ROI_fourier_space;
    NppiSize npp_ROI_fourier_with_real_functor;

    ////////////////////////////////////////////////////////////////////////
    ///// Methods that should behave as their counterpart in the Image class
    ///// have /**CPU_eq**/
    ////////////////////////////////////////////////////////////////////////

    void QuickAndDirtyWriteSlices(std::string filename, int first_slice, int last_slice); /**CPU_eq**/
    void PhaseShift(float wanted_x_shift, float wanted_y_shift, float wanted_z_shift); /**CPU_eq**/
    void MultiplyByConstant(float scale_factor); /**CPU_eq**/
    void SetToConstant(float val);
    void SetToConstant(Npp32fc val);
    void Conj( ); // FIXME
    void MultiplyPixelWise(GpuImage& other_image); /**CPU_eq**/
    void MultiplyPixelWise(GpuImage& other_image, GpuImage& output_image); /**CPU_eq**/

    void MultiplyPixelWiseComplexConjugate(GpuImage& other_image, GpuImage& result_image);

    void SwapFourierSpaceQuadrants( );
    void SwapRealSpaceQuadrants( ); /**CPU_eq**/
    void ClipInto(GpuImage* other_image, float wanted_padding_value, /**CPU_eq**/
                  bool fill_with_noise, float wanted_noise_sigma,
                  int wanted_coordinate_of_box_center_x,
                  int wanted_coordinate_of_box_center_y,
                  int wanted_coordinate_of_box_center_z);

    void ClipIntoFourierSpace(GpuImage* destination_image, float wanted_padding_value);

    void ClipIntoReturnMask(GpuImage* other_image);

    // Used with explicit specializtion
    template <class InputType, class OutputType>
    void _ForwardFFT( );

    template <class InputType, class OutputType>
    void _BackwardFFT( );

    void ForwardFFT(bool should_scale = true); /**CPU_eq**/
    void ForwardFFTBatched(bool should_scale = true);

    void BackwardFFT( ); /**CPU_eq**/
    void BackwardFFTBatched(int wanted_batch_size = 0); // if zero, defaults to dims.z

    void ForwardFFTAndClipInto(GpuImage& image_to_insert, bool should_scale);
    template <typename T>
    void BackwardFFTAfterComplexConjMul(T* image_to_multiply, bool load_half_precision);

    void Resize(int wanted_x_dimension, int wanted_y_dimension, int wanted_z_dimension, float wanted_padding_value);
    void Consume(GpuImage* other_image);
    void CopyCpuImageMetaData(Image& cpu_image);
    void CopyGpuImageMetaData(const GpuImage* other_image);
    void CopyLoopingAndAddressingFrom(GpuImage* other_image);

    float ReturnSumOfSquares( );
    float ReturnAverageOfRealValuesOnEdges( );
    void  Deallocate( );
    void  ConvertToHalfPrecision(bool deallocate_single_precision = true);
    void  AllocateTmpVarsAndEvents( );
    bool  Allocate(int wanted_x_size, int wanted_y_size, int wanted_z_size, bool should_be_in_real_space);

    bool Allocate(int wanted_x_size, int wanted_y_size, bool should_be_in_real_space) { return Allocate(wanted_x_size, wanted_y_size, 1, should_be_in_real_space); };

    // Combines this and UpdatePhysicalAddressOfBoxCenter and SetLogicalDimensions
    void UpdateLoopingAndAddressing(int wanted_x_size, int wanted_y_size, int wanted_z_size);

    ////////////////////////////////////////////////////////////////////////
    ///// Methods that do not have a counterpart in the image class
    ////////////////////////////////////////////////////////////////////////

    void CopyHostToDevice(bool should_block_until_complete = false);

    void CopyHostToDeviceAndSynchronize( ) { CopyHostToDevice(true); };

    void CopyHostToDeviceTextureComplex3d( );
    void CopyHostToDevice16f(bool should_block_until_finished = false); // CTF images in the ImageClass are stored as complex, even if they only have a real part. This is a waste of memory bandwidth on the GPU
    void CopyDeviceToHostAndSynchronize(bool free_gpu_memory = true, bool unpin_host_memory = true);
    void CopyDeviceToHost(bool free_gpu_memory = true, bool unpin_host_memory = true);
    void CopyDeviceToHost(Image& cpu_image, bool should_block_until_complete = false, bool free_gpu_memory = true, bool unpin_host_memory = true);

    void  CopyDeviceToNewHost(Image& cpu_image, bool should_block_until_complete, bool free_gpu_memory, bool unpin_host_memory = true);
    Image CopyDeviceToNewHost(bool should_block_until_complete, bool free_gpu_memory, bool unpin_host_memory = true);
    // The volume copies with memory coalescing favoring padding are not directly
    // compatible with the memory layout in Image().
    void CopyVolumeHostToDevice( );
    void CopyVolumeDeviceToHost(bool free_gpu_memory = true, bool unpin_host_memory = true);
    // Synchronize the full stream.
    void Record( );
    void Wait( );
    void RecordAndWait( );
    // Maximum intensity projection
    void MipPixelWise(GpuImage& other_image);
    void MipPixelWise(GpuImage& other_image, GpuImage& psi, GpuImage& phi, GpuImage& theta,
                      float c_psi, float c_phi, float c_theta);
    void MipPixelWise(GpuImage& other_image, GpuImage& psi, GpuImage& phi, GpuImage& theta, GpuImage& defocus, GpuImage& pixel,
                      float c_psi, float c_phi, float c_theta, float c_defocus, float c_pixel);

    // FIXME: These are added for the unblur refinement but are untested.
    void ApplyBFactor(float bfactor);
    void ApplyBFactor(float bfactor, float vertical_mask_size, float horizontal_mask_size); // Specialization for unblur refinement, merges MaskCentralCross()
    void Whiten(float resolution_limit = 1.f);

    float GetWeightedCorrelationWithImage(GpuImage& projection_image, GpuImage& cross_terms, GpuImage& image_PS, GpuImage& projection_PS, float filter_radius_low_sq, float filter_radius_high_sq, float signed_CC_limit);

    inline void MaskCentralCross(float vertical_mask_size, float horizontal_mask_size) { return; }; // noop

    void CalculateCrossCorrelationImageWith(GpuImage* other_image);
    Peak FindPeakWithParabolaFit(float inner_radius_for_peak_search, float outer_radius_for_peak_search);
    Peak FindPeakAtOriginFast2D(int wanted_max_pix_x, int wanted_max_pix_y, Peak* pinned_host_buffer, Peak* device_buffer, int wanted_batch_size, bool load_half_precision = false);
    Peak FindPeakAtOriginFast2D(BatchedSearch* batch, bool load_half_precision = false);
    bool Init(Image& cpu_image, bool pin_host_memory = true, bool allocate_real_values = true);
    void SetupInitialValues( );
    void UpdateBoolsToDefault( );
    void SetCufftPlan(cistem::fft_type::Enum plan_type, void* input_buffer, void* output_buffer);

    cistem::fft_type::Enum set_plan_type;
    bool                   is_batched_transform;

    template <int ntds_x = 32, int ntds_y = 32>
    __inline__ void ReturnLaunchParameters(int4 input_dims, bool real_space) {
        static_assert(ntds_x % cistem::gpu::warp_size == 0);
        static_assert(ntds_x * ntds_y <= cistem::gpu::max_threads_per_block);
        int div = 1;
        if ( ! real_space )
            div++;

        threadsPerBlock = dim3(ntds_x, ntds_y, 1);
        gridDims        = dim3((input_dims.w / div + threadsPerBlock.x - 1) / threadsPerBlock.x,
                               (input_dims.y + threadsPerBlock.y - 1) / threadsPerBlock.y,
                               input_dims.z);
    };

    __inline__ void ReturnLaunchParameters1d_X(const int4 input_dims, const bool real_space) {
        int div = 1;
        if ( ! real_space )
            div++;

        using namespace cistem::gpu;
        // Note: that second set of parens changes the division!
        threadsPerBlock = dim3(std::max(min_threads_per_block, std::min(max_threads_per_block, warp_size * ((input_dims.w / div + warp_size - 1) / warp_size))), 1, 1);
        gridDims        = dim3((input_dims.w / div + threadsPerBlock.x - 1) / threadsPerBlock.x,
                               (input_dims.y + threadsPerBlock.y - 1) / threadsPerBlock.y,
                               input_dims.z);
    };

    __inline__ void ReturnLaunchParameters1d_X_strided_Y(const int4 input_dims, const bool real_space, const int stride_y) {
        int div = 1;
        if ( ! real_space )
            div++;

        using namespace cistem::gpu;
        // Note: that second set of parens changes the division!
        threadsPerBlock = dim3(std::max(min_threads_per_block, std::min(max_threads_per_block / stride_y, warp_size * ((input_dims.w / div + warp_size - 1) / warp_size))), stride_y, 1);
        gridDims        = dim3((input_dims.w / div + threadsPerBlock.x - 1) / threadsPerBlock.x,
                               (input_dims.y + threadsPerBlock.y - 1) / threadsPerBlock.y,
                               input_dims.z);
    };

    __inline__ void ReturnLaunchParametersLimitSMs(float N, int M) {
        // This should only be called for kernels with grid stride loops setup. The idea
        // is to limit the number of SMs available for some kernels so that other threads on the device can run in parallel.
        // limit_SMs_by_threads is default 1, so this must be set prior to this call.
        threadsPerBlock = dim3(M, 1, 1);
        gridDims        = dim3(myroundint(N * number_of_streaming_multiprocessors));
    };

    void CopyFrom(GpuImage* other_image);
    bool InitializeBasedOnCpuImage(Image& cpu_image, bool pin_host_memory, bool allocate_real_values);
    void UpdateCpuFlags( );
    void printVal(std::string msg, int idx);
    bool HasSameDimensionsAs(GpuImage* other_image);
    void Zeros( );

    void ExtractSlice(GpuImage* volume_to_extract_from, AnglesAndShifts& angles_and_shifts, float pixel_size, float resolution_limit = 1.f, bool apply_resolution_limit = true, bool whiten_spectrum = false);

    void ExtractSliceShiftAndCtf(GpuImage* volume_to_extract_from, GpuImage* ctf_image, AnglesAndShifts& angles_and_shifts, float pixel_size, float resolution_limit, bool apply_resolution_limit,
                                 bool swap_quadrants, bool apply_shifts, bool apply_ctf, bool absolute_ctf);

    void Abs( );
    void AbsDiff(GpuImage& other_image); // inplace
    void AbsDiff(GpuImage& other_image, GpuImage& output_image);
    void SquareRealValues( );
    void SquareRootRealValues( );
    void LogarithmRealValues( );
    void ExponentiateRealValues( );
    void AddConstant(const float add_val);
    void AddConstant(const Npp32fc add_val);

    void AddImage(GpuImage& other_image);

    void AddImage(GpuImage* other_image) { AddImage(*other_image); }; // for compatibility with Image class

    void SubtractImage(GpuImage& other_image);

    void SubtractImage(GpuImage* other_image) { SubtractImage(*other_image); }; // for compatibility with Image class

    void AddSquaredImage(GpuImage& other_image);

    // Statitical Methods
    float ReturnSumOfRealValues( );
    // float3    ReturnSumOfRealValues3Channel( );
    NppiPoint min_idx;
    NppiPoint max_idx;
    float     min_value;
    float     max_value;
    float     img_mean;
    float     img_stdDev;
    Npp64f    npp_mean;
    Npp64f    npp_stdDev;
    int       number_of_pixels_in_range;
    void      Min( );
    void      MinAndCoords( );
    void      Max( );
    void      MaxAndCoords( );
    void      MinMax( );
    void      MinMaxAndCoords( );
    void      Mean( );
    void      MeanStdDev( );
    void      AverageError(const GpuImage& other_image); // TODO add me
    void      AverageRelativeError(const GpuImage& other_image); // TODO addme
    void      CountInRange(float lower_bound, float upper_bound);
    void      HistogramEvenBins( ); // TODO add me
    void      HistogramDefinedBins( ); // TODO add me

    // TODO
    /*

  Mean, Mean_StdDev 
  */

    ////////////////////////////////////////////////////////////////////////
    ///// Methods for creating or storing masks used for otherwise slow looping operations
    ////////////////////////////////////////////////////////////////////////

    enum BufferType : int { b_image,
                            b_sum,
                            b_min,
                            b_minIDX,
                            b_max,
                            b_maxIDX,
                            b_minmax,
                            b_minmaxIDX,
                            b_mean,
                            b_meanstddev,
                            b_countinrange,
                            b_histogram,
                            b_16f,
                            b_ctf_16f,
                            b_l2norm,
                            b_dotproduct,
                            b_clip_into_mask,
                            b_weighted_correlation };

    //  void CublasInit();
    void NppInit( );
    void BufferInit(BufferType bt, int n_elements = 0);
    void BufferDestroy( );

    // Real buffer = size real_values
    GpuImage* image_buffer;
    bool      is_allocated_image_buffer;

    // Npp specific buffers;
    Npp8u* sum_buffer;
    bool   is_allocated_sum_buffer;
    Npp8u* min_buffer;
    bool   is_allocated_min_buffer;
    Npp8u* minIDX_buffer;
    bool   is_allocated_minIDX_buffer;
    Npp8u* max_buffer;
    bool   is_allocated_max_buffer;
    Npp8u* maxIDX_buffer;
    bool   is_allocated_maxIDX_buffer;
    Npp8u* minmax_buffer;
    bool   is_allocated_minmax_buffer;
    Npp8u* minmaxIDX_buffer;
    bool   is_allocated_minmaxIDX_buffer;
    Npp8u* mean_buffer;
    bool   is_allocated_mean_buffer;
    Npp8u* meanstddev_buffer;
    bool   is_allocated_meanstddev_buffer;
    Npp8u* countinrange_buffer;
    bool   is_allocated_countinrange_buffer;
    Npp8u* l2norm_buffer;
    bool   is_allocated_l2norm_buffer;
    Npp8u* dotproduct_buffer;
    bool   is_allocated_dotproduct_buffer;
    bool   is_allocated_16f_buffer;
    bool   is_allocated_ctf_16f_buffer;
    int*   clip_into_mask;
    bool   is_allocated_clip_into_mask;
    bool   is_set_realLoadAndClipInto;
    float* weighted_correlation_buffer;
    bool   is_allocated_weighted_correlation_buffer;
    int    weighted_correlation_buffer_size;

    GpuImage* mask_CSOS;
    bool      is_allocated_mask_CSOS;
    float     ReturnSumSquareModulusComplexValues( );

    // Callback related parameters
    bool is_set_convertInputf16Tof32;
    bool is_set_scaleFFTAndStore;
    bool is_set_complexConjMulLoad;

    /*template void d_MultiplyByScalar<T>(T* d_input, T* d_multiplicators, T* d_output, size_t elements, int batch);*/

  private:
};

#endif /* GPUIMAGE_H_ */
