#ifndef __SRC_PROGRAMS_MATCH_TEMPLATE_TEMPLATE_MATCHING_DATA_SIZER_H_
#define __SRC_PROGRAMS_MATCH_TEMPLATE_TEMPLATE_MATCHING_DATA_SIZER_H_

#include <memory>

#include "../../constants/constants.h"

// #define USE_NEAREST_NEIGHBOR_INTERPOLATION
// #define USE_REPLICATIVE_PADDING

constexpr bool  MUST_BE_POWER_OF_TWO                   = false; // Required for half-precision xforms
constexpr int   MUST_BE_FACTOR_OF                      = 0; // May be faster
constexpr float max_reduction_by_fraction_of_reference = 0.000001f; // FIXME the cpu version is crashing when the image is reduced, but not the GPU
constexpr int   MAX_3D_PADDING                         = 196;

constexpr int n_outputs = 8;

enum Enum : int {

    // Images for the statistical data
    max_intensity_projection,
    best_psi,
    best_theta,
    best_phi,
    best_defocus,
    best_pixel_size,
    correlation_pixel_sum_image,
    correlation_pixel_sum_of_squares_image
};

/**
 * @brief This class is used to optionally resample, pad or cut into chunks the search image.
 * 
 * It relies on knowing the template size, particularly in the case of chunking.
 * This is a draft object and will be moved to a more appropriate location. TODO
 * 
 */
class TemplateMatchingDataSizer {

    // This is a non-data owning class, but we want references to the underlying image/template data
    int4 image_size;
    int4 image_pre_scaling_size;
    int4 image_cropped_size;
    int4 image_search_size;

    // n elements that are not valid search results,  pre_padding.x then is also the first valid physical x-index in zero based coordinates
    std::array<int2, 4> pre_padding;
    // logical x/y size of the region of interest.
    std::array<int2, 4> roi;

    Image valid_area_mask;

    long number_of_valid_search_pixels;

    int4 template_size;
    int4 template_pre_scaling_size;
    int4 template_cropped_size;
    int4 template_search_size;

    float pixel_size{0.f};
    float search_pixel_size{0.f};
    float template_padding{ };
    float high_resolution_limit{-1.f};
    bool  resampling_is_wanted{false};
    bool  use_fast_fft{false};
    bool  sizing_is_set{false};
    bool  padding_is_set{false};
    bool  input_image_has_been_preprocessed{false};
    bool  valid_bounds_are_set{false};
    bool  roi_is_set{false};
    bool  image_is_resampled{false};
    bool  image_is_rotated_by_90{false};
    int   n_chunks_in_x{1};
    int   n_chunks_in_y{1};

    std::vector<int> primes;

    float max_increase_by_fraction_of_image;

    void SetHighResolutionLimit(const float wanted_high_resolution_limit);
    void GetFFTSize( );
    void CheckSizingForGreaterThan4k( );
    void CheckSizingFinalSearchSize( );

    void GetInputImageToEvenAndSquareOrPrimeFactoredSizePadding(int& pre_padding_x, int& pre_padding_y, int& post_padding_x, int& post_padding_y);
    void SetValidSearchImageIndiciesFromPadding(const int pre_padding_x, const int pre_padding_y, const int post_padding_x, const int post_padding_y);

    void   FillInNearestNeighbors(Image& output_image, Image& nn_upsampled_image, Image& valid_area_mask, const float no_value);
    void   GetChunkOffsets(int3& wanted_origin, const int i_chunk, bool pre_shift);
    MyApp* parent_match_template_app_ptr;

  public:
    // Keep a copy of the original image following pre-processing but no resizing. We'll use this
    // to deterimine what the peak would be if the image were not resized.
    // We are currently supporting at most 4 chunks (to make a k3 without super res into 2 4k images)
    std::vector<Image> pre_processed_image;

    TemplateMatchingDataSizer(MyApp*    wanted_parent_ptr,
                              const int input_image_logical_x_dimension,
                              const int input_image_logical_y_dimension,
                              const int template_logical_x_dimension,
                              const int template_logical_y_dimension,
                              const int template_logical_z_dimension,
                              float     wanted_pixel_size,
                              float     wanted_template_padding);
    ~TemplateMatchingDataSizer( );

    std::unique_ptr<Curve> whitening_filter_ptr;

    // Don't allow copy or move. FIXME: if we don't add any dynamically allocated data, we can remove this.
    // TemplateMatchingDataSizer(const TemplateMatchingDataSizer&)            = delete;
    // TemplateMatchingDataSizer& operator=(const TemplateMatchingDataSizer&) = delete;
    // TemplateMatchingDataSizer(TemplateMatchingDataSizer&&)                 = delete;
    // TemplateMatchingDataSizer& operator=(TemplateMatchingDataSizer&&)      = delete;

    void SetImageAndTemplateSizing(const float wanted_high_resolution_limit, const bool use_fast_fft);
    void PreProcessInputImage(Image& input_image);

    void PreProcessResizedInputImage(Image& input_image);

    void ResizeTemplate_preSearch(Image& template_image, const bool use_lerp_not_fourier_resampling = false);
    void ResizeTemplate_postSearch(Image& template_image);

    // All statistical images (mip, psi etc.) are originally allocated based on the pre-processed input_image size,
    // and so only the input image needs attention at the outset. Following the search, all statistical images
    // will also need to be resized.
    void  ResizeImage_preSearch(const int central_cross_half_width);
    float ResizeImage_postSearch(std::array<Image*, n_outputs>& statistical_images, long* total_number_of_stats_samples);

    inline void PrintImageSizes( ) {
        if ( ReturnThreadNumberOfCurrentThread( ) == 0 ) {
            wxPrintf("old x, y = %i %i\n  new x, y = %i %i\n", image_size.x, image_size.y, image_search_size.x, image_search_size.y);
        }
    }

    inline int GetNumberOfChunks( ) const {
        MyDebugAssertTrue(sizing_is_set, "Sizing has not been set");
        return n_chunks_in_x * n_chunks_in_y;
    }

    inline bool IsResamplingNeeded( ) const {
        return resampling_is_wanted;
    }

    inline bool IsRotatedBy90( ) const {
        return image_is_rotated_by_90;
    }

    inline int GetImageSizeX( ) const {
        return image_search_size.x;
    }

    inline int GetImageSizeY( ) const {
        return image_search_size.y;
    }

    inline int GetImageSearchSizeX( ) const {
        return image_search_size.x;
    }

    inline int GetImageSearchSizeY( ) const {
        return image_search_size.y;
    }

    inline int GetTemplateSearchSizeX( ) const {
        return template_search_size.x;
    }

    inline int GetTemplateSizeX( ) const {
        return template_size.x;
    }

    inline long GetNumberOfValidSearchPixels( ) const {
        MyDebugAssertTrue(valid_bounds_are_set, "Valid bounds not set");
        return number_of_valid_search_pixels;
    }

    inline long GetNumberOfPixelsForNormalization( ) const {
        MyDebugAssertTrue(valid_bounds_are_set, "Valid bounds not set");
        return long(image_search_size.x * image_search_size.y);
    }

    inline long GetSearchImageRealMemoryAllocated( ) const {
        return pre_processed_image[0].real_memory_allocated;
    }

    inline float GetPixelSize( ) const {
        MyDebugAssertFalse(pixel_size == 0.0f, "Pixel size not set");
        return pixel_size;
    }

    inline float GetHighResolutionLimit( ) const {
        MyDebugAssertFalse(high_resolution_limit == -1.0f, "High resolution limit not set");
        return high_resolution_limit;
    }

    inline float GetSearchPixelSize( ) const {
        MyDebugAssertFalse(search_pixel_size == 0.0f, "Search pixel size not set");
        return search_pixel_size;
    }

    inline float GetFullBinningFactor( ) const {
        return GetSearchPixelSize( ) / GetPixelSize( );
    }

    inline int GetBinnedSize(float input_size, float wanted_binning_factor) {
        return int(input_size / wanted_binning_factor + 0.5f);
    }

    // Note that the input_size will be cast from int -> float on the function call.
    inline float GetRealizedBinningFactor(float wanted_binning_factor, float input_size) {
        int wanted_binned_size = GetBinnedSize(input_size, wanted_binning_factor);
        if ( IsOdd(wanted_binned_size) )
            wanted_binned_size++;
        return input_size / float(wanted_binned_size);
    }

    inline int2 GetPrePadding(const int i_chunk) const {
        return pre_padding[i_chunk];
    }

    inline int2 GetRoi(const int i_chunk) const {
        return roi[i_chunk];
    }

    inline int GetPrePaddingX(const int i_chunk) const {
        return pre_padding[i_chunk].x;
    }

    inline int GetPrePaddingY(const int i_chunk) const {
        return pre_padding[i_chunk].y;
    }

    inline int GetRoiX(const int i_chunk) const {
        return roi[i_chunk].x;
    }

    inline int GetRoiY(const int i_chunk) const {
        return roi[i_chunk].y;
    }

    inline float* GetValidAreaMask(const int i_chunk) {
        return valid_area_mask.real_values;
    }
};

#endif