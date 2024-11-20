#include <cistem_config.h>

#ifdef ENABLEGPU
#include "../../gpu/gpu_core_headers.h"
#include "../../gpu/DeviceManager.h"
#include "../../gpu/TemplateMatchingCore.h"
#else
#include "../../core/core_headers.h"
#endif

#include "../../constants/constants.h"

#if defined(cisTEM_USING_FastFFT) && defined(ENABLEGPU)
#include "../../../include/FastFFT/include/FastFFT.h"
#endif

#include "template_matching_data_sizer.h"

// #define DEBUG_IMG_PREPROCESS_OUTPUT "/scratch/salina/TM_test_scaling/yeast/Assets/TemplateMatching"
// #define DEBUG_IMG_POSTPROCESS_OUTPUT "/scratch/salina/TM_test_scaling/yeast/Assets/TemplateMatching"
#define DEBUG_TM_SIZER_PRINT

TemplateMatchingDataSizer::TemplateMatchingDataSizer(MyApp*    wanted_parent_ptr,
                                                     const int input_image_logical_x_dimension,
                                                     const int input_image_logical_y_dimension,
                                                     const int template_logical_x_dimension,
                                                     const int template_logical_y_dimension,
                                                     const int template_logical_z_dimension,
                                                     float     wanted_pixel_size,
                                                     float     wanted_template_padding)
    : parent_match_template_app_ptr{wanted_parent_ptr},
      image_size{int4{input_image_logical_x_dimension, input_image_logical_y_dimension, 1,
                      IsEven(input_image_logical_x_dimension) ? (input_image_logical_x_dimension + 2) / 2 : (input_image_logical_x_dimension + 1) / 2}},
      template_size{int4{template_logical_x_dimension, template_logical_y_dimension, template_logical_z_dimension,
                         IsEven(template_logical_x_dimension) ? (template_logical_x_dimension + 2) / 2 : (template_logical_x_dimension + 1) / 2}},

      pixel_size{wanted_pixel_size},
      template_padding{wanted_template_padding} {

    MyDebugAssertTrue(pixel_size > 0.0f, "Pixel size must be greater than zero");
    // TODO: remove this constraint
    MyAssertTrue(template_padding == 1.0f, "Padding must be  equal to 1.0");
    // TODO:
    // Is the case of an odd sized image handled properly?
};

TemplateMatchingDataSizer::~TemplateMatchingDataSizer( ){
        // Nothing to do here

};

/**
 * @brief Peform checks on the wanted high resolution limit, set range of prime factors that are acceptable based on whether we are using FastFFT or not.
 * 
 * @param wanted_high_resolution_limit 
 * @param use_fast_fft 
 */
void TemplateMatchingDataSizer::SetImageAndTemplateSizing(const float wanted_high_resolution_limit, const bool use_fast_fft) {
    MyDebugAssertFalse(sizing_is_set, "Sizing has already been set");
    // Make sure we aren't trying to limit beyond Nyquist, and if < Nyquist set resampling needed to true.
    SetHighResolutionLimit(wanted_high_resolution_limit);
    this->use_fast_fft = use_fast_fft;

    // Setup some limits. These could probably just go directly into their specific methods in this class
    if ( use_fast_fft ) {
        primes.assign({2});
        max_increase_by_fraction_of_image = 2.f;
    }
    else {
        primes.assign({2, 3, 5, 7, 9, 13});
        max_increase_by_fraction_of_image = 0.1f;
    }

    GetFFTSize( );

    // For now, let's simplify a little bit and disallow resampling at the same time as chunking.
    // Ultimately there is no good reason to do this other than it being more logic to track.
    MyDebugAssertFalse((n_chunks_in_y > 1 || n_chunks_in_x > 1) && resampling_is_wanted, "Resampling and chunking are not compatible");
};

/**
 * @brief Always remove outliers, center and whiten prior to any transormations, resampling or chunking of the input image.
 * 
 * We ALWAYS want the starting image statistics to be the same, regardless of the final size.
 * 
 * @param input_image 
 */
void TemplateMatchingDataSizer::PreProcessInputImage(Image& input_image) {

    // We could also check and FFT if necessary similar to Resize() but we are assuming the input image is in real space.
    MyDebugAssertTrue(input_image.is_in_real_space, "Input image must be in real space");
    MyDebugAssertTrue(whitening_filter_ptr == nullptr, "Whitening filter is already set");
    MyDebugAssertTrue(sizing_is_set, "Sizing has already been set");

    // We whiten the image prior to any padding etc in particular to remove any low-frequency gradients that would add to boundary dislocations.
    // We may also whiten following any further resampling and resizing or other ops that are done to the image. We need to keep track of the total filtering applied.
    Curve number_of_terms;
    Curve local_whitening_filter;
    local_whitening_filter.SetupXAxisForFourierSpace(input_image.logical_x_dimension, 2.0f);
    number_of_terms.SetupXAxisForFourierSpace(input_image.logical_x_dimension, 2.0f);

    // FIXME: confirm this is being cleaned up
    for ( int i_chunk = 0; i_chunk < GetNumberOfChunks( ); i_chunk++ ) {
        pre_processed_image.emplace_back(image_search_size.x, image_search_size.y, 1, true);
    }
    whitening_filter_ptr = std::make_unique<Curve>(local_whitening_filter);
    whitening_filter_ptr->SetYToConstant(1.0f);
    // We'll accumulate the local whitening filter at the end of the method

    // This won't work for movie frames (13.0 is used in unblur) TODO use poisson stats
    input_image.ReplaceOutliersWithMean(5.0f);

#ifdef DEBUG_IMG_PREPROCESS_OUTPUT
    if ( ReturnThreadNumberOfCurrentThread( ) == 0 )
        input_image.QuickAndDirtyWriteSlice(DEBUG_IMG_PREPROCESS_OUTPUT "/input_image.mrc", 1);
#endif

    input_image.ForwardFFT( );
    input_image.ZeroCentralPixel( );

    input_image.Compute1DPowerSpectrumCurve(&local_whitening_filter, &number_of_terms);
    local_whitening_filter.SquareRoot( );
    local_whitening_filter.Reciprocal( );
    local_whitening_filter.MultiplyByConstant(1.0f / local_whitening_filter.ReturnMaximumValue( ));

    input_image.ApplyCurveFilter(&local_whitening_filter);

    // Record this filtering for later use
    whitening_filter_ptr->MultiplyBy(local_whitening_filter);

    input_image.ZeroCentralPixel( );
    // Presumably for Pre-processing (where we need the realspace variance = 1 so noise in padding reginos matchs)
    // TODO: rename so the real space vs FFT is clear
    input_image.DivideByConstant(sqrtf(input_image.ReturnSumOfSquares( )));
    input_image.BackwardFFT( );

#ifdef DEBUG_IMG_PREPROCESS_OUTPUT
    if ( ReturnThreadNumberOfCurrentThread( ) == 0 )
        input_image.QuickAndDirtyWriteSlice(DEBUG_IMG_PREPROCESS_OUTPUT "/input_image_whitened.mrc", 1);
#endif

    input_image_has_been_preprocessed = true;
};

void TemplateMatchingDataSizer::PreProcessResizedInputImage(Image& input_image) {
    MyDebugAssertFalse(whitening_filter_ptr == nullptr, "Whitening filter not set");

    for ( int i_chunk = 0; i_chunk < GetNumberOfChunks( ); i_chunk++ ) {
        if ( GetNumberOfChunks( ) > 1 ) {
            pre_processed_image.at(i_chunk).Allocate(image_search_size.x, image_search_size.y, 1, true);

            // Now if we are splitting into chunks do it here.

#ifdef USE_REPLICATIVE_PADDING
            bool skip_padding_in_clipinto = false;
#else
            bool skip_padding_in_clipinto = true;
            pre_processed_image.at(i_chunk).FillWithNoiseFromNormalDistribution(0.f, 1.0f);
#endif
            int3 wanted_origin;
            // Chunks are taken from lower left, then lower right, then upper left, then upper right
            GetChunkOffsets(wanted_origin, i_chunk, true);

#ifdef USE_REPLICATIVE_PADDING
            input_image.ClipIntoWithReplicativePadding(&pre_processed_image.at(i_chunk),
                                                       wanted_origin.x,
                                                       wanted_origin.y,
                                                       0,
                                                       skip_padding_in_clipinto);
#else
            input_image.ClipInto(&pre_processed_image.at(i_chunk),
                                 0.0f,
                                 false,
                                 1.0f,
                                 wanted_origin.x,
                                 wanted_origin.y,
                                 0,
                                 skip_padding_in_clipinto);
#endif
        }
        pre_processed_image.at(i_chunk).ForwardFFT( );
        pre_processed_image.at(i_chunk).SwapRealSpaceQuadrants( );
        // FIXME: chunks move to new method
        // When used in cross-correlation, we need the extra division.
        pre_processed_image.at(i_chunk).DivideByConstant(sqrtf(pre_processed_image.at(i_chunk).ReturnSumOfSquares( ) / float(GetNumberOfPixelsForNormalization( ))));
    }

#ifdef DEBUG_IMG_PREPROCESS_OUTPUT
    if ( ReturnThreadNumberOfCurrentThread( ) == 0 ) {
        for ( int i_chunk = 0; i_chunk < GetNumberOfChunks( ); i_chunk++ ) {
            std::string chunk_name = DEBUG_IMG_PREPROCESS_OUTPUT "/input_image_" + std::to_string(i_chunk) + ".mrc";
            pre_processed_image.at(i_chunk).QuickAndDirtyWriteSlice(chunk_name, 1);
        }
    }
    DEBUG_ABORT;
#endif
}

void TemplateMatchingDataSizer::GetChunkOffsets(int3& wanted_origin, const int i_chunk, bool pre_shift) {

    // Chunks are taken from lower left, then lower right, then upper left, then upper right

    int nx_source{ }, ny_source{ }, ox_source{ }, oy_source{ };
    int nx_dest{ }, ny_dest{ }, ox_dest{ }, oy_dest{ };

    // Note: we are assuming the origin is always N/2 for both even and odd
    if ( pre_shift ) {
        nx_source = image_size.x;
        ny_source = image_size.y;
        ox_source = nx_source / 2;
        oy_source = ny_source / 2;
        nx_dest   = image_search_size.x;
        ny_dest   = image_search_size.y;
        ox_dest   = nx_dest / 2;
        oy_dest   = ny_dest / 2;
    }
    else {
        nx_source = image_search_size.x;
        ny_source = image_search_size.y;
        ox_source = nx_source / 2;
        oy_source = ny_source / 2;
        nx_dest   = image_size.x;
        ny_dest   = image_size.y;
        ox_dest   = nx_dest / 2;
        oy_dest   = ny_dest / 2;
    }

    switch ( i_chunk ) {
        case 0:
            wanted_origin.x = ox_dest - ox_source;
            wanted_origin.y = oy_dest - oy_source;
            break;
        case 1:
            wanted_origin.x = (nx_source - ox_dest) - ox_source;
            wanted_origin.y = oy_dest - oy_source;
            break;
        case 2:
            wanted_origin.x = ox_dest - ox_source;
            wanted_origin.y = (ny_source - oy_dest) - oy_source;
            break;
        case 3:
            wanted_origin.x = (nx_source - ox_dest) - ox_source;
            wanted_origin.y = (ny_source - oy_dest) - oy_source;
            break;
    }
    wanted_origin.z = i_chunk;
}

void TemplateMatchingDataSizer::CheckSizingForGreaterThan4k( ) {

    // If we are not using FastFFT then we'll just use whatever cuFFT does, however,
    // There may be cases where splitting the image into chunks may be faster even for cuFFT, for example,
    // If someone were to search a huge image (8k for example.)
    if ( use_fast_fft ) {
        image_pre_scaling_size.x = image_size.x;
        image_pre_scaling_size.y = image_size.y;

        image_cropped_size.x = image_size.x;
        image_cropped_size.y = image_size.y;
        // We can only use FastFFT for power of 2 size images and template
        // for now, we'll only deal with the case where we can split the image into either 2 or 4 chunks.
        if ( image_size.x > 4096 ) {
            if ( image_size.x <= 8192 ) {
                n_chunks_in_x        = 2;
                image_cropped_size.x = 4096;
            }
            else
                parent_match_template_app_ptr->SendError("Image size is too large in X dimension for FastFFT\n");
        }
        if ( image_size.y > 4096 ) {
            if ( image_size.y <= 8192 ) {
                n_chunks_in_y        = 2;
                image_cropped_size.y = 4096;
            }
            else
                parent_match_template_app_ptr->SendError("Image size is too large in Y dimension for FastFFT\n");
        }
    }
    else {
        image_pre_scaling_size.x = image_size.x;
        image_pre_scaling_size.y = image_size.y;

        image_cropped_size.x = image_size.x;
        image_cropped_size.y = image_size.y;
    }

    image_pre_scaling_size.z = 1;
    image_cropped_size.z     = 1;
}

/**
 * @brief Private: Internal check that the search size is valid for the given conditions
 * 
 */
void TemplateMatchingDataSizer::CheckSizingFinalSearchSize( ) {
    // TODO: remove thiss
    if ( use_fast_fft ) {
        // We currently can only use FastFFT for power of 2 size images and template
        // TODO: the limit on templates should be trivial to remove since they are padded.
        // TOOD: Should probably consider this in the code above working out next good size, rather than only allowing power of 2,
        //       which will take a k3 to 8k x 8k it would be better to pad either
        //       a) crop to 4k x 4k (lossy but immediately supported)
        //       b) split into two images and pad each to either 4k x 4k (possibly a bit slower and not yet supported)
        // FIXME:
        if ( image_search_size.x != image_search_size.y ) {
            parent_match_template_app_ptr->SendInfo("FastFFT currently only supports square images, padding smaller up\n");
        }
        if ( image_search_size.x > image_search_size.y ) {
            image_search_size.y = image_search_size.x;
        }
        else {
            image_search_size.x = image_search_size.y;
        }
        if ( image_search_size.x > 4096 || image_search_size.y > 4096 ) {
            parent_match_template_app_ptr->SendError("FastFFT only supports images up to 4096x4096\n");
        }
    }
    else {
        // TODO: what else should be be verifying - rotation if warranted and power of two / nice siziing?
    }
};

/**
 * @brief Private: determin the padding needed to make the image square and then to resample and finally to get to a nice FFT size.
 * 
 */
void TemplateMatchingDataSizer::GetFFTSize( ) {

    // We can downsample the template at arbitrary pixel sizes using LERP, so we only need to consider the image size.
    int   bin_offset_2d             = 0;
    float closest_2d_binning_factor = 1.f;

    const float target_binning_factor = high_resolution_limit / pixel_size / 2.0f;

    float current_binning_factor;
    int   current_binned_size;

    if ( resampling_is_wanted ) {

        // We want the binning to be isotropic, and the easiest way to ensure that is to first pad any non-square input_image to a square size in realspace.
        const int max_square_size = std::max(image_size.x, image_size.y);
        // In the unusual case that the cropped image is larger than the input, we need to start again, and n_tries will be set to -1
        int n_tries  = 0;
        int bin_scan = 1;
        while ( n_tries < 2 ) {
            // Presumably we'll be using a power of 2 square size anyway for FastFFT (though rectangular images should be supported at some point.)
            // The other requirement is to ensure the resulting pixel size is the same for the reference and the search images.
            // Ideally, we would just calculate a scattering potential at the correct size, but even when that capability is added, we still should
            // allow the user to supply a template that is generated from a map that may not have a good model (e.g. something at 5-6 A resolution may still be usefule for TM in situ,
            // but may be to low res to build a decent atomic model into.)

            // The most challenging part is matching the pixel size of the input 3d and the input images. Presumably, the smaller 3d will be the limiting factor b/c of the larger
            // Fourier voxel step.

            // Start by taking the (possibly) largest deviation from the wanted pixel size, but the (possibly) smallest final image size to determine subsequent penalties for padding.
            current_binning_factor = GetRealizedBinningFactor(target_binning_factor, max_square_size);
            current_binned_size    = GetBinnedSize(max_square_size, current_binning_factor);

            // FIXME: We should enforce the 4k restriction somwhere else.
            constexpr int   max_2d_power_of_2_size      = 4096;
            constexpr float acceptable_pixel_size_error = 0.00005f;

            int closest_2d_binned_size = current_binned_size;
            closest_2d_binning_factor  = current_binning_factor;

            float smallest_error = fabsf(target_binning_factor - closest_2d_binning_factor);

            int bin_offset_2d = 0;
            //  && IsEven(image_size.x - (max_square_size + bin_offset_2d))
            while ( current_binned_size <= max_2d_power_of_2_size ) {
                if ( smallest_error * pixel_size < acceptable_pixel_size_error ) {
                    break;
                }
                bin_offset_2d += bin_scan;
                current_binning_factor = GetRealizedBinningFactor(target_binning_factor, max_square_size + bin_offset_2d);
                current_binned_size    = GetBinnedSize(max_square_size + bin_offset_2d, current_binning_factor);
                float current_error    = fabsf(current_binning_factor - target_binning_factor);
                if ( current_error < smallest_error ) {
                    closest_2d_binned_size    = current_binned_size;
                    closest_2d_binning_factor = current_binning_factor;
                    smallest_error            = current_error;
                }
            }

            image_pre_scaling_size.x = max_square_size + bin_offset_2d;
            image_pre_scaling_size.y = max_square_size + bin_offset_2d;
            image_pre_scaling_size.z = 1;

            image_cropped_size.x = closest_2d_binned_size;
            image_cropped_size.y = closest_2d_binned_size;
            image_cropped_size.z = 1;

            n_tries++;

            if ( image_cropped_size.x <= image_size.x && image_cropped_size.y <= image_size.y ) {
                n_tries++;
            }
            else
                bin_scan = -1;
        }
    }
    else {
        CheckSizingForGreaterThan4k( );
    }

    // When FastFFT is enabled, primes are limited to 2 by the calling method.
    int factor_result_pos;
    for ( auto& prime_value : primes ) {
        factor_result_pos = ReturnClosestFactorizedUpper(image_cropped_size.x, prime_value, true, MUST_BE_FACTOR_OF);
        if ( (float)(-image_cropped_size.x + factor_result_pos) < float(image_cropped_size.x) * max_increase_by_fraction_of_image ) {
            image_search_size.x = factor_result_pos;
            break;
        }
    }

    for ( auto& prime_value : primes ) {
        factor_result_pos = ReturnClosestFactorizedUpper(image_cropped_size.y, prime_value, true, MUST_BE_FACTOR_OF);
        if ( (float)(-image_cropped_size.y + factor_result_pos) < float(image_cropped_size.y) * max_increase_by_fraction_of_image ) {
            image_search_size.y = factor_result_pos;
            break;
        }
    }

    // In the case where this is note == 1, image_cropped_size = image_pre_scaling_size
    image_search_size.z = 1;

    // Assuming the template is cubic and we handle sampling during projection, there is no need for pre-scaling, only resizing to ensure power of 2 (if FastFFT)
    // TODO: This should work without power of two if we project into a power of 2 size, esp important for templates > 512
    template_pre_scaling_size.x = template_size.x;
    template_pre_scaling_size.y = template_size.y;
    template_pre_scaling_size.z = template_size.z;

    template_cropped_size.x = template_size.x;
    template_cropped_size.y = template_size.y;
    template_cropped_size.z = template_size.z;

    // In the general case, there are no restrictions on the template being a power of two, but we should want a decent size
    int prime_factor_3d    = use_fast_fft ? 2 : 5;
    template_search_size.x = ReturnClosestFactorizedUpper(template_cropped_size.x, prime_factor_3d, true, MUST_BE_POWER_OF_TWO);
    template_search_size.y = template_search_size.x;
    template_search_size.z = template_search_size.x;
    // We know this is an even dimension so adding 2
    template_search_size.w = (template_search_size.x + 2) / 2;

    search_pixel_size = pixel_size * closest_2d_binning_factor;

#ifdef DEBUG_TM_SIZER_PRINT
    wxPrintf("The input image will be padded by %d,%d, cropped to %d,%d and then padded again to %d,%d\n",
             image_pre_scaling_size.x - image_size.x, image_pre_scaling_size.y - image_size.y,
             image_cropped_size.x, image_cropped_size.y,
             image_search_size.x, image_search_size.y);
    wxPrintf("template_size = %i\n", template_size.x);
    wxPrintf("closest 2d binning factor = %f\n", closest_2d_binning_factor);
    wxPrintf("closest 2d binning factor * pixel_size = %f\n", closest_2d_binning_factor * pixel_size);
    wxPrintf("original image size = %i\n", int(image_size.x));
    wxPrintf("wanted_binned_size = %i,%i\n", image_cropped_size.x, image_cropped_size.y);
    wxPrintf("input  pixel size: %3.6f\n", pixel_size);
    wxPrintf("target pixel size: %3.6f\n", target_binning_factor * pixel_size);
    wxPrintf("search pixel size: %3.6f\n", search_pixel_size);
#endif
    // Now try to increase the padding of the input image to match the 3d
    CheckSizingFinalSearchSize( );
    sizing_is_set = true;

    int pre_binning_padding_x;
    int post_binning_padding_x;
    int pre_binning_padding_y;
    int post_binning_padding_y;

    // Things are simplified because the padding is always resulting in an even dimensions
    // NOTE: assuming integer division.
    // This is the first padding, if resampling it will be to an even and square size, otherwise it will be to a nice fourier size and the final step.
    GetInputImageToEvenAndSquareOrPrimeFactoredSizePadding(pre_binning_padding_x, pre_binning_padding_y, post_binning_padding_x, post_binning_padding_y);
#ifdef DEBUG_TM_SIZER_PRINT
    wxPrintf("pre_binning_padding_x = %i\n", pre_binning_padding_x);
    wxPrintf("pre_binning_padding_y = %i\n", pre_binning_padding_y);
    wxPrintf("post_binning_padding_x = %i\n", post_binning_padding_x);
    wxPrintf("post_binning_padding_y = %i\n", post_binning_padding_y);
#endif

    if ( resampling_is_wanted ) {
        // Here we need to scale the padding to account for resampling.
        // I think the easiest way to handle fractional reduction, which could result in an odd number of invalid rows/columns is to round up
        pre_binning_padding_x  = myroundint(ceilf(float(pre_binning_padding_x) / GetFullBinningFactor( )));
        pre_binning_padding_y  = myroundint(ceilf(float(pre_binning_padding_y) / GetFullBinningFactor( )));
        post_binning_padding_x = myroundint(ceilf(float(post_binning_padding_x) / GetFullBinningFactor( )));
        post_binning_padding_y = myroundint(ceilf(float(post_binning_padding_y) / GetFullBinningFactor( )));

#ifdef DEBUG_TM_SIZER_PRINT
        wxPrintf("binning factor = %f\n", GetFullBinningFactor( ));
        wxPrintf("pre_binning_padding_x = %i\n", pre_binning_padding_x);
        wxPrintf("pre_binning_padding_y = %i\n", pre_binning_padding_y);
        wxPrintf("post_binning_padding_x = %i\n", post_binning_padding_x);
        wxPrintf("post_binning_padding_y = %i\n", post_binning_padding_y);
#endif
        // Now add on any padding needed to make the image a power of two
        // These are both even dimensions, so we can just use the symmetric padding.
        pre_binning_padding_x += (image_search_size.x - image_cropped_size.x) / 2;
        pre_binning_padding_y += (image_search_size.y - image_cropped_size.y) / 2;
        post_binning_padding_x += (image_search_size.x - image_cropped_size.x) / 2;
        post_binning_padding_y += (image_search_size.y - image_cropped_size.y) / 2;
#ifdef DEBUG_TM_SIZER_PRINT

        wxPrintf("+= pre_binning_padding_x = %i\n", pre_binning_padding_x);
        wxPrintf("+= pre_binning_padding_y = %i\n", pre_binning_padding_y);
        wxPrintf("+= post_binning_padding_x = %i\n", post_binning_padding_x);
        wxPrintf("+= post_binning_padding_y = %i\n", post_binning_padding_y);
#endif
    }
    SetValidSearchImageIndiciesFromPadding(pre_binning_padding_x, pre_binning_padding_y, post_binning_padding_x, post_binning_padding_y);
};

/**
 * @brief Define the regions where there is non-padding values, which impacts the normalization and is in turn used to define the  ROI
 * which is needed to both correctly histogram the data and also to reduce wasted computation.
 * 
 * @param pre_padding_x 
 * @param pre_padding_y 
 * @param post_padding_x 
 * @param post_padding_y 
 */
void TemplateMatchingDataSizer::SetValidSearchImageIndiciesFromPadding(const int pre_padding_x, const int pre_padding_y, const int post_padding_x, const int post_padding_y) {
    MyDebugAssertTrue(sizing_is_set, "Sizing has not been set");
    MyDebugAssertFalse(valid_bounds_are_set, "Valid bounds have already been set");

    number_of_valid_search_pixels = 0;

    for ( int i_chunk = 0; i_chunk < GetNumberOfChunks( ); i_chunk++ ) {
        pre_padding[i_chunk].x = pre_padding_x;
        pre_padding[i_chunk].y = pre_padding_y;
        // Each chunk has a large area of redundant information (i suppose it could be called implicit padding.)
        // Each chunk is unique in the corner it is pushed to and will overlap with neighboring chunks.
        int overlap_x{ }, overlap_y{ };
        if ( n_chunks_in_x > 1 ) {
            overlap_x = (2 * image_search_size.x - image_size.x) / 2;
        }
        else {
            overlap_x = 0;
        }
        if ( n_chunks_in_y > 1 ) {
            overlap_y = (2 * image_search_size.y - image_size.y) / 2;
        }
        else {
            overlap_y = 0;
        }

        std::cerr << "image_search_size.x = " << image_search_size.x << std::endl;
        std::cerr << "post_padding_y = " << post_padding_y << std::endl;
        std::cerr << "pre_padding[i_chunk].x = " << pre_padding[i_chunk].x << std::endl;
        std::cerr << "overlap_x = " << overlap_x << std::endl;
        std::cerr << "image_search_size.y = " << image_search_size.y << std::endl;
        std::cerr << "post_padding_y = " << post_padding_y << std::endl;
        std::cerr << "pre_padding[i_chunk].y = " << pre_padding[i_chunk].y << std::endl;
        std::cerr << "overlap_y = " << overlap_y << std::endl;

        // FIXME: confirm there isn't an off by one error here
        roi[i_chunk].x = image_search_size.x - post_padding_y - pre_padding[i_chunk].x - overlap_x;
        roi[i_chunk].y = image_search_size.y - post_padding_y - pre_padding[i_chunk].y - overlap_y;
        if ( GetNumberOfChunks( ) > 1 ) {
            roi[i_chunk].x++;
            roi[i_chunk].y++;
        }

        switch ( i_chunk + 1 ) {
            case 1: {
                // The lower left corner, no need to modify as the search images are indexed from pre_padding -> pre_padding + roi
                break;
            }
            case 2: {
                // The lower right corner, pre padding needs to be modified
                pre_padding[i_chunk].x = pre_padding_x + overlap_x - 1;
                break;
            }
            case 3: {
                // The upper left corner, pre padding needs to be modified
                pre_padding[i_chunk].y = pre_padding_y + overlap_y - 1;
                break;
            }
            case 4: {
                // The upper right corner, pre padding needs to be modified
                pre_padding[i_chunk].x = pre_padding_x + overlap_x - 1;
                pre_padding[i_chunk].y = pre_padding_y + overlap_y - 1;
                break;
            }
        }

#ifdef DEBUG_TM_SIZER_PRINT
        wxPrintf("pre_padding[%i] = %i %i\n", i_chunk, pre_padding[i_chunk].x, pre_padding[i_chunk].y);
        wxPrintf("roi[%i] = %i %i\n", i_chunk, roi[i_chunk].x, roi[i_chunk].y);
#endif

        MyDebugAssertTrue(roi[i_chunk].x > 0 && roi[i_chunk].y > 0, "The ROI is less than 1 pixel in size");
        // Note this is used for calculating the full size of the search space and should represent all pixels that will be represented
        // in the final result mip. If we are downsampling, each pixel in number_of_valid_search_pixels may correspond to multiple pixels in
        // the final result, which are co-dependent. I.e. the threshold for a binned search should be lower than a full search.
        number_of_valid_search_pixels += long(roi[i_chunk].x) * long(roi[i_chunk].y);
    }

    MyDebugAssertTrue(number_of_valid_search_pixels > 0, "The number of valid search pixels is less than 1");

    valid_bounds_are_set = true;
};

void TemplateMatchingDataSizer::GetInputImageToEvenAndSquareOrPrimeFactoredSizePadding(int& pre_padding_x, int& pre_padding_y, int& post_padding_x, int& post_padding_y) {
    MyDebugAssertTrue(sizing_is_set, "Sizing has not been set");
    MyDebugAssertFalse(padding_is_set, "Padding has already been set");
    // There are no restrictions on the input image for this function, it may be sq or rect, even or odd,
    // but presumably there is only one layer of padding and it is >= 0
    // TODO: we need to consider the template
    int padding_x_TOTAL;
    int padding_y_TOTAL;

    if ( resampling_is_wanted ) {
        padding_x_TOTAL = image_pre_scaling_size.x - image_size.x;
        padding_y_TOTAL = image_pre_scaling_size.y - image_size.y;
    }
    else if ( GetNumberOfChunks( ) > 1 ) {
        // When breaking into chunks, we currently only allow images (4096, 8192] in size. We only need to pad for any dimensions < 4096
        padding_x_TOTAL = std::max(0, 4096 - image_size.x);
        padding_y_TOTAL = std::max(0, 4096 - image_size.y);
    }
    else {
        // When not useing fast FFT, there is at most one padding step from input size to a nice fourier size.
        padding_x_TOTAL = image_search_size.x - image_cropped_size.x;
        padding_y_TOTAL = image_search_size.y - image_cropped_size.y;
    }

#ifdef DEBUG_TM_SIZER_PRINT
    wxPrintf("in get input image to even and square or prime factored size padding\n");
    wxPrintf("image_size = %i %i %i\n", image_size.x, image_size.y, image_size.z);
    wxPrintf("image_pre_scaling_size = %i %i %i\n", image_pre_scaling_size.x, image_pre_scaling_size.y, image_pre_scaling_size.z);
    wxPrintf("image_cropped_size = %i %i %i\n", image_cropped_size.x, image_cropped_size.y, image_cropped_size.z);
    wxPrintf("image_search_size = %i %i %i\n", image_search_size.x, image_search_size.y, image_search_size.z);
#endif

    // An odd sized image has an equal number of pixels left/right of the origin
    //  So if the image is add padding symmetrically and any "extra" padding to the left.
    // An even sized image has 1 more pixel to the left.
    // So if the image is even, add padding symmetrically and any "extra" padding to the right.
    post_padding_x = padding_x_TOTAL / 2;
    pre_padding_x  = padding_x_TOTAL / 2;
    post_padding_y = padding_y_TOTAL / 2;
    pre_padding_y  = padding_y_TOTAL / 2;
    if ( IsEven(image_size.x) ) {
        post_padding_x += padding_x_TOTAL % 2;
    }
    else {
        pre_padding_x += padding_x_TOTAL % 2;
    }
    if ( IsEven(image_size.y) ) {
        post_padding_y += padding_y_TOTAL % 2;
    }
    else {
        pre_padding_y += padding_y_TOTAL % 2;
    }

    padding_is_set = true;
    return;
}

/**
 * @brief Check to see if the resolution limit is within the Nyquist limit. Also set a flag that indicates whether resampling is needed.
 * 
 * @param wanted_high_resolution_limit 
 */
void TemplateMatchingDataSizer::SetHighResolutionLimit(const float wanted_high_resolution_limit) {
    if ( wanted_high_resolution_limit < 2.0f * pixel_size )
        high_resolution_limit = 2.0f * pixel_size;
    else
        high_resolution_limit = wanted_high_resolution_limit;

    if ( FloatsAreAlmostTheSame(high_resolution_limit, 2.0f * pixel_size) )
        resampling_is_wanted = false;
    else
        resampling_is_wanted = true;

    // TODO: this isn't perfect
    float approx_binning_factor = high_resolution_limit / (pixel_size * 2.0f);
    if ( image_size.x / approx_binning_factor > 4096.f || image_size.y / approx_binning_factor > 4096.f ) {
        parent_match_template_app_ptr->SendInfo("The image size is too large for the requested resolution limit, chunking instead\n");
        resampling_is_wanted  = false;
        high_resolution_limit = 2.0f * pixel_size;
    }
};

void TemplateMatchingDataSizer::ResizeTemplate_preSearch(Image& template_image, const bool use_lerp_not_fourier_resampling) {

    if ( use_lerp_not_fourier_resampling ) {
        // We only need to set the 3d to be the padded power of two size and have the resampling be handled during the projection step.
        // search size always >= cropped size
        template_image.Resize(std::max(template_size.x, template_search_size.x),
                              std::max(template_size.y, template_search_size.y),
                              std::max(template_size.z, template_search_size.z),
                              template_image.ReturnAverageOfRealValuesOnEdges( ));
    }
    else {
#ifdef DEBUG_IMG_PREPROCESS_OUTPUT
        if ( ReturnThreadNumberOfCurrentThread( ) == 0 ) {
            // Print out the size of each step
            wxPrintf("template_size = %i %i %i\n", template_size.x, template_size.y, template_size.z);
            wxPrintf("template_pre_scaling_size = %i %i %i\n", template_pre_scaling_size.x, template_pre_scaling_size.y, template_pre_scaling_size.z);
            wxPrintf("template_cropped_size = %i %i %i\n", template_cropped_size.x, template_cropped_size.y, template_cropped_size.z);
            wxPrintf("template_search_size = %i %i %i\n", template_search_size.x, template_search_size.y, template_search_size.z);
            template_image.QuickAndDirtyWriteSlices(DEBUG_IMG_PREPROCESS_OUTPUT "/template_image.mrc", 1, template_size.z / 2);
        }
#endif
        template_image.Resize(template_pre_scaling_size.x, template_pre_scaling_size.y, template_pre_scaling_size.z, template_image.ReturnAverageOfRealValuesOnEdges( ));
#ifdef DEBUG_IMG_PREPROCESS_OUTPUT
        template_image.QuickAndDirtyWriteSlices(DEBUG_IMG_PREPROCESS_OUTPUT "/template_image_resized_pre_scale.mrc", 1, template_pre_scaling_size.z / 2);
#endif
        template_image.ForwardFFT( );
        template_image.Resize(template_cropped_size.x, template_cropped_size.y, template_cropped_size.z);
        template_image.BackwardFFT( );
#ifdef DEBUG_IMG_PREPROCESS_OUTPUT
        template_image.QuickAndDirtyWriteSlices(DEBUG_IMG_PREPROCESS_OUTPUT "/template_image_resized_cropped.mrc", 1, template_cropped_size.z / 2);
#endif
        template_image.Resize(template_search_size.x, template_search_size.y, template_search_size.z, template_image.ReturnAverageOfRealValuesOnEdges( ));
#ifdef DEBUG_IMG_PREPROCESS_OUTPUT
        if ( ReturnThreadNumberOfCurrentThread( ) == 0 ) {
            template_image.QuickAndDirtyWriteSlices(DEBUG_IMG_PREPROCESS_OUTPUT "/template_image_resized.mrc", 1, template_search_size.z / 2);
        }
#endif
    }
};

void TemplateMatchingDataSizer::ResizeTemplate_postSearch(Image& template_image) {
    MyAssertTrue(false, "Not yet implemented");
};

void TemplateMatchingDataSizer::ResizeImage_preSearch(const int central_cross_half_width) {
    MyDebugAssertTrue(sizing_is_set, "Sizing has not been set");
    MyDebugAssertTrue(input_image_has_been_preprocessed, "The input image has not been preprocessed");
    MyDebugAssertFalse(resampling_is_wanted && (n_chunks_in_x > 1 || n_chunks_in_y > 1), "Resampling is wanted, but the image is being split into chunks");
    MyDebugAssertFalse((! is_power_of_two(image_search_size.x) && is_power_of_two(image_search_size.y)) && (n_chunks_in_x > 1 || n_chunks_in_y > 1), "The image is not a power of two, but the image is being split into chunks");

    // NOTE: this is currently mutually exclusive with breaking the image into chunks
    if ( resampling_is_wanted ) {
        wxPrintf("Resampling the input image\n");
        Image tmp_sq;

        tmp_sq.Allocate(image_pre_scaling_size.x, image_pre_scaling_size.y, image_pre_scaling_size.z, true);
#ifdef USE_REPLICATIVE_PADDING
        bool skip_padding_in_clipinto = false;
#else
        bool skip_padding_in_clipinto = true;
        tmp_sq.FillWithNoiseFromNormalDistribution(0.f, 1.0f);
#endif

#ifdef USE_REPLICATIVE_PADDING
        pre_processed_image.at(0).ClipIntoWithReplicativePadding(&tmp_sq);
#else
        pre_processed_image.at(0).ClipInto(&tmp_sq, 0.0f, false, 1.0f, 0, 0, 0, skip_padding_in_clipinto);
#endif

#ifdef DEBUG_IMG_PREPROCESS_OUTPUT
        if ( ReturnThreadNumberOfCurrentThread( ) == 0 )
            tmp_sq.QuickAndDirtyWriteSlice(DEBUG_IMG_PREPROCESS_OUTPUT "/tmp_sq.mrc", 1);
#endif
        tmp_sq.ForwardFFT( );
        tmp_sq.Resize(image_cropped_size.x, image_cropped_size.y, image_cropped_size.z);
        tmp_sq.ZeroCentralPixel( );
        tmp_sq.DivideByConstant(sqrtf(tmp_sq.ReturnSumOfSquares( )));
        tmp_sq.BackwardFFT( );
#ifdef DEBUG_IMG_PREPROCESS_OUTPUT
        if ( ReturnThreadNumberOfCurrentThread( ) == 0 )
            tmp_sq.QuickAndDirtyWriteSlice(DEBUG_IMG_PREPROCESS_OUTPUT "/tmp_sq_resized.mrc", 1);
#endif

        pre_processed_image.at(0).Allocate(image_search_size.x, image_search_size.y, 1, true);

#ifdef USE_REPLICATIVE_PADDING
        tmp_sq.ClipIntoWithReplicativePadding(&pre_processed_image.at(0));
#else
        pre_processed_image.at(0).FillWithNoiseFromNormalDistribution(0.f, 1.0f);
        tmp_sq.ClipInto(&pre_processed_image.at(0), 0.0f, false, 1.0f, 0, 0, 0, skip_padding_in_clipinto);
#endif // USE_REPLICATIVE_PADDING

#ifdef DEBUG_IMG_PREPROCESS_OUTPUT
        if ( ReturnThreadNumberOfCurrentThread( ) == 0 )
            pre_processed_image.at(0).QuickAndDirtyWriteSlice(DEBUG_IMG_PREPROCESS_OUTPUT "/input_image_resized.mrc", 1);
#endif

        if ( central_cross_half_width > 0 ) {
            pre_processed_image.at(0).ForwardFFT( );
            pre_processed_image.at(0).MaskCentralCross(central_cross_half_width, central_cross_half_width);
            pre_processed_image.at(0).BackwardFFT( );
        }

#ifdef DEBUG_IMG_PREPROCESS_OUTPUT
        if ( ReturnThreadNumberOfCurrentThread( ) == 0 )
            pre_processed_image.at(0).QuickAndDirtyWriteSlice(DEBUG_IMG_PREPROCESS_OUTPUT "/input_image_resized_masked.mrc", 1);
        DEBUG_ABORT;
#endif
    }
    else {
        wxPrintf("not resampling\n");

        if ( central_cross_half_width > 0 ) {
            pre_processed_image.at(0).ForwardFFT( );
            pre_processed_image.at(0).MaskCentralCross(central_cross_half_width, central_cross_half_width);
            pre_processed_image.at(0).BackwardFFT( );
        }
    }

// NOTE: rotation must always be the FINAL step in pre-processing / resizing and it is always the first to be inverted at the end.
// NOTE2: This is mutually exclusive with the chunking operation for the time being
#ifdef ROTATEFORSPEED
    if ( ! is_power_of_two(image_search_size.x) && is_power_of_two(image_search_size.y) ) {
        // The speedup in the FFT for better factorization is also dependent on the dimension. The full transform (in cufft anyway) is faster if the best dimension is on X.
        // TODO figure out how to check the case where there is no factor of two, but one dimension is still faster. Probably getting around to writing an explicit planning tool would be useful.
        if ( ReturnThreadNumberOfCurrentThread( ) == 0 ) {
            wxPrintf("Rotating the search image for speed\n");
        }
        pre_processed_image.at(0).RotateInPlaceAboutZBy90Degrees(true);
        // bool preserve_origin = true;
        // input_reconstruction.RotateInPlaceAboutZBy90Degrees(true, preserve_origin);
        // The amplitude spectrum is also rotated
        image_is_rotated_by_90 = true;
#ifdef DEBUG_IMG_PREPROCESS_OUTPUT
        if ( ReturnThreadNumberOfCurrentThread( ) == 0 )
            pre_processed_image.at(0).QuickAndDirtyWriteSlice(DEBUG_IMG_PREPROCESS_OUTPUT "/input_image_rotated.mrc", 1);
#endif
    }
    else {
        if ( ReturnThreadNumberOfCurrentThread( ) == 0 ) {
            wxPrintf("Not rotating the search image for speed even though it is enabled\n");
        }
        image_is_rotated_by_90 = false;
    }
#endif
};

float TemplateMatchingDataSizer::ResizeImage_postSearch(std::array<Image*, n_outputs>& statistical_images, long* total_number_of_stats_samples) {

    MyDebugAssertTrue(sizing_is_set, "Sizing has not been set");
    MyDebugAssertFalse(use_fast_fft ? image_is_rotated_by_90 : false, "Rotating the search image when using fastfft does  not make sense given the current square size restriction of FastFFT");
    if ( resampling_is_wanted ) {
        MyDebugAssertTrue(statistical_images.at(max_intensity_projection)[0].logical_x_dimension <= (image_is_rotated_by_90 ? image_size.y : image_size.x), "The max intensity projection is larger than the original image size");
        MyDebugAssertTrue(statistical_images.at(max_intensity_projection)[0].logical_y_dimension <= (image_is_rotated_by_90 ? image_size.x : image_size.y), "The max intensity projection is larger than the original image size");
    }
    for ( int i_chunk = 0; i_chunk < GetNumberOfChunks( ); i_chunk++ ) {
        MyDebugAssertTrue(pre_processed_image.at(i_chunk).is_in_memory, "The pre-processed image is not in memory");
    }

    // The stats and histogram sampling is stochastic if < 1, so we want to rescale the avg/std img to the relative avg of the sampling
    float avg_sampling{ };
    for ( int i_chunk = 0; i_chunk < GetNumberOfChunks( ); i_chunk++ ) {
        avg_sampling += float(total_number_of_stats_samples[i_chunk]);
    }
    avg_sampling /= float(GetNumberOfChunks( ));
    for ( int i_chunk = 0; i_chunk < GetNumberOfChunks( ); i_chunk++ ) {
        statistical_images.at(correlation_pixel_sum_image)[i_chunk].MultiplyByConstant(avg_sampling / total_number_of_stats_samples[i_chunk]);
        statistical_images.at(correlation_pixel_sum_of_squares_image)[i_chunk].MultiplyByConstant(avg_sampling / total_number_of_stats_samples[i_chunk]);
    }

    // If we have split the image into chunks, our task is much easier because we have NOT rotated by 90 (all chunks are square)
    // and we have NOT done any resampling (in the current implementation.)
    if ( GetNumberOfChunks( ) > 1 ) {
#ifdef DEBUG_IMG_POSTPROCESS_OUTPUT
        std::cerr << " Resizing the image post search\n";

#endif
        // Now if we are splitting into chunks do it here.
        Image tmp_output;
        int3  wanted_origin;
        // we don't want to deal with explicitly figuring out the overlap, so just write in where we have a valid search, some
        // points will be overwritten but that shouldn't matter.
        bool skip_padding_in_clipinto = true;
        for ( int i_img = 0; i_img < statistical_images.size( ); i_img++ ) {
            tmp_output.Allocate(image_size.x, image_size.y, image_size.z, true);
            double avg       = 0.0;
            long   n_samples = 0;
            if ( i_img != correlation_pixel_sum_image && i_img != correlation_pixel_sum_of_squares_image ) {
                for ( int i_chunk = 0; i_chunk < GetNumberOfChunks( ); i_chunk++ ) {
                    for ( int y = pre_padding[i_chunk].y; y < pre_padding[i_chunk].y + roi[i_chunk].y; y++ ) {
                        for ( int x = pre_padding[i_chunk].x; x < pre_padding[i_chunk].x + roi[i_chunk].x; x++ ) {
                            avg += statistical_images[i_img][i_chunk].ReturnRealPixelFromPhysicalCoord(x, y, 0);
                            n_samples++;
                        }
                    }
                }
            }
            if ( n_samples > 0 )
                avg /= double(n_samples);
            tmp_output.SetToConstant(0.f);

            for ( int i_chunk = 0; i_chunk < GetNumberOfChunks( ); i_chunk++ ) {
                // This assumes the overlaping region is zeroed out by proper consideration of ROI during mipping and that
                // the mip and other stats arrays were initialized to zero.

                // Chunks are taken from lower left, then lower right, then upper left, then upper right
                GetChunkOffsets(wanted_origin, i_chunk, false);
#ifdef DEBUG_IMG_POSTPROCESS_OUTPUT
                std::string output_name = DEBUG_IMG_POSTPROCESS_OUTPUT "/output_" + std::to_string(i_img) + "_" + std::to_string(i_chunk) + ".mrc";
                statistical_images[i_img][i_chunk].QuickAndDirtyWriteSlice(output_name, 1);
#endif

                // The offsets are calculated for Clip into  (this int -> other)
                // InsertOther adds (other -> this.)
                // FIXME as part of unifying clip into
                tmp_output.InsertOtherImageAtSpecifiedPosition(&statistical_images[i_img][i_chunk], -wanted_origin.x, -wanted_origin.y, 0);
            }

            if ( i_img != correlation_pixel_sum_image && i_img != correlation_pixel_sum_of_squares_image ) {
                for ( long i_pixel = 0; i_pixel < tmp_output.real_memory_allocated; i_pixel++ ) {
                    if ( tmp_output.real_values[i_pixel] == 0.f ) {
                        tmp_output.real_values[i_pixel] = avg;
                    }
                }
            }

            // We start with an std::array of Image pointers, that point to arrays of Images. Delete the array of images and move the resources
            // to the output which is std::array of Image pointers that point to a single resized image.
            delete[] statistical_images[i_img];
            statistical_images[i_img] = new Image;
            statistical_images[i_img]->Consume(&tmp_output);

        } // end of loop over images
    }
    else {

        // We want the unsampled regions to have the same mean AND variance as the sampled regions,

        // now loop over and update the mean values int rand_y_idx;
        int  rand_x_idx, rand_y_idx;
        long rand_address;

        // To avoid dislocations around the edge, the random padding is not enough for these images with their much larger dynamic range.
        // We might devise a mirrored or replicative padding, but a simpler solution is just to handle dividing by N here rather than latter in the processing.

        int x_lower_bound = pre_padding[0].x;
        int y_lower_bound = pre_padding[0].y;
        int x_upper_bound = pre_padding[0].x + roi[0].x;
        int y_upper_bound = pre_padding[0].y + roi[0].y;
        int i_x, i_y;

        float x_radius = float(statistical_images.at(max_intensity_projection)[0].physical_address_of_box_center_x - x_lower_bound);
        float y_radius = float(statistical_images.at(max_intensity_projection)[0].physical_address_of_box_center_y - y_lower_bound);

        Image tmp_trim;
        tmp_trim.Allocate(roi[0].x, roi[0].y, 1, true);

        for ( auto& working_image : statistical_images ) {
            working_image[0].ClipInto(&tmp_trim);
            tmp_trim.ClipIntoWithReplicativePadding(&working_image[0]);
        }

        // Work through the transformations backward to get to the original image size
        if ( image_is_rotated_by_90 ) {
            // swap the bounds
            float tmp_x = x_radius;
            x_radius    = y_radius;
            y_radius    = tmp_x;

            for ( auto& working_image : statistical_images ) {
                working_image[0].RotateInPlaceAboutZBy90Degrees(false);
            }

            statistical_images.at(best_psi)[0].AddConstant(90.0f);
            for ( int idx = 0; idx < statistical_images.at(best_psi)[0].real_memory_allocated; idx++ ) {
                statistical_images.at(best_psi)[0].real_values[idx] = clamp_angular_range_0_to_2pi(statistical_images.at(best_psi)[0].real_values[idx], true);
            }
        }

        // We need to use nearest neighbor interpolation to cast all existing values back to the original size.
        Image           tmp_mip, tmp_psi, tmp_phi, tmp_theta, tmp_defocus, tmp_pixel_size, tmp_sum, tmp_sum_sq;
        constexpr float NN_no_value = -std::numeric_limits<float>::max( );
        constexpr float no_value    = 0.f;

        if ( resampling_is_wanted ) {
            // original size -> pad to square -> crop to binned -> pad to fourier
            // The new images at the square binned size (remove the padding to power of two)

            // We'll fill all the images with -FLT_MAX to indicate to downstream code that the values are not valid measurements from an experiment.
            tmp_phi.Allocate(image_size.x, image_size.y, image_size.z, true);
            tmp_theta.Allocate(image_size.x, image_size.y, image_size.z, true);
            tmp_psi.Allocate(image_size.x, image_size.y, image_size.z, true);
            tmp_defocus.Allocate(image_size.x, image_size.y, image_size.z, true);
            tmp_pixel_size.Allocate(image_size.x, image_size.y, image_size.z, true);

            tmp_phi.SetToConstant(NN_no_value);
            tmp_theta.SetToConstant(NN_no_value);
            tmp_psi.SetToConstant(NN_no_value);
            tmp_defocus.SetToConstant(NN_no_value);
            tmp_pixel_size.SetToConstant(NN_no_value);

            long        searched_image_address = 0;
            long        out_of_bounds_value    = 0;
            long        address                = 0;
            const float actual_image_binning   = GetFullBinningFactor( );

            // Loop over the (possibly) binned image coordinates
            for ( int j = pre_padding[0].y; j < pre_padding[0].y + roi[0].y; j++ ) {
                int y_offset_from_origin = j - statistical_images.at(max_intensity_projection)[0].physical_address_of_box_center_y;
                for ( int i = pre_padding[0].x; i < pre_padding[0].y + roi[0].y; i++ ) {
                    // Get this pixels offset from the center of the box
                    int x_offset_from_origin = i - statistical_images.at(max_intensity_projection)[0].physical_address_of_box_center_x;

                    // Scale by the binning
                    // Using std::trunc (round to zero) as some pathological cases produce OOB +/- 1 otherwise
                    int x_non_binned = tmp_phi.physical_address_of_box_center_x + std::truncf(float(x_offset_from_origin) * actual_image_binning);
                    int y_non_binned = tmp_phi.physical_address_of_box_center_y + std::truncf(float(y_offset_from_origin) * actual_image_binning);

                    if ( x_non_binned >= 0 && x_non_binned < tmp_phi.logical_x_dimension && y_non_binned >= 0 && y_non_binned < tmp_phi.logical_y_dimension ) {
                        address                = tmp_phi.ReturnReal1DAddressFromPhysicalCoord(x_non_binned, y_non_binned, 0);
                        searched_image_address = statistical_images.at(max_intensity_projection)[0].ReturnReal1DAddressFromPhysicalCoord(i, j, 0);
                    }
                    else {
                        // FIXME: This print block needs to be removed after initial debugging.
                        wxPrintf("x_non_binned = %d, y_non_binned = %d\n", x_non_binned, y_non_binned);
                        wxPrintf("%f actual_image_binning = %f\n", search_pixel_size, actual_image_binning);
                        wxPrintf("tmp mip size = %d %d\n", tmp_phi.logical_x_dimension, tmp_phi.logical_y_dimension);
                        wxPrintf("x offset = %d, y offset = %d\n", x_offset_from_origin, y_offset_from_origin);
                        wxPrintf("box center = %d %d\n", tmp_phi.physical_address_of_box_center_x, tmp_phi.physical_address_of_box_center_y);
                        wxPrintf("tmp size = %d %d\n", tmp_phi.logical_x_dimension, tmp_phi.logical_y_dimension);
                        wxPrintf("max_intensity_projection  size = %d %d\n", statistical_images.at(max_intensity_projection)[0].logical_x_dimension, statistical_images.at(max_intensity_projection)[0].logical_y_dimension);
                        address = -1;
                        exit(1);
                    }

                    // There really shouldn't be any peaks out of bounds
                    // I think we should only every update an address once, so let's check it here for now.
                    if ( address < 0 || address > tmp_phi.real_memory_allocated ) {
                        out_of_bounds_value++;
                    }
                    else {
                        MyDebugAssertFalse(tmp_phi.real_values[address] == no_value, "Address already updated");
                        tmp_phi.real_values[address]        = statistical_images.at(best_phi)[0].real_values[searched_image_address];
                        tmp_theta.real_values[address]      = statistical_images.at(best_theta)[0].real_values[searched_image_address];
                        tmp_psi.real_values[address]        = statistical_images.at(best_psi)[0].real_values[searched_image_address];
                        tmp_defocus.real_values[address]    = statistical_images.at(best_defocus)[0].real_values[searched_image_address];
                        tmp_pixel_size.real_values[address] = statistical_images.at(best_pixel_size)[0].real_values[searched_image_address];
                    }
                }
            }

            MyDebugAssertTrue(out_of_bounds_value == 0, "There are out of bounds values in calculating the NN interpolation of the max intensity projection");

            tmp_mip.Allocate(image_pre_scaling_size.x, image_pre_scaling_size.y, image_pre_scaling_size.z, false);
            tmp_sum.Allocate(image_pre_scaling_size.x, image_pre_scaling_size.y, image_pre_scaling_size.z, false);
            tmp_sum_sq.Allocate(image_pre_scaling_size.x, image_pre_scaling_size.y, image_pre_scaling_size.z, false);

            // Resize from any fourier padding to the cropped size
            statistical_images.at(max_intensity_projection)[0].Resize(image_cropped_size.x, image_cropped_size.y, image_cropped_size.z);
            statistical_images.at(correlation_pixel_sum_image)[0].Resize(image_cropped_size.x, image_cropped_size.y, image_cropped_size.z);
            statistical_images.at(correlation_pixel_sum_of_squares_image)[0].Resize(image_cropped_size.x, image_cropped_size.y, image_cropped_size.z);

            // Now undo the fourier binning
            statistical_images.at(max_intensity_projection)[0].ForwardFFT( );
            statistical_images.at(max_intensity_projection)[0].ClipInto(&tmp_mip, 0.0f, false, 1.0f, 0, 0, 0, true);
            tmp_mip.BackwardFFT( );
            statistical_images.at(max_intensity_projection)[0].Allocate(image_size.x, image_size.y, image_size.z, true);
            tmp_mip.ClipInto(&statistical_images.at(max_intensity_projection)[0], 0.0f, false, 1.0f, 0, 0, 0, true);

            // Dilate the radius of the valid area mask
            x_radius *= GetFullBinningFactor( );
            y_radius *= GetFullBinningFactor( );

            statistical_images.at(correlation_pixel_sum_image)[0].ForwardFFT( );
            statistical_images.at(correlation_pixel_sum_image)[0].ClipInto(&tmp_sum, 0.0f, false, 1.0f, 0, 0, 0, true);
            tmp_sum.BackwardFFT( );
            statistical_images.at(correlation_pixel_sum_image)[0].Allocate(image_size.x, image_size.y, image_size.z, true);
            tmp_sum.ClipInto(&statistical_images.at(correlation_pixel_sum_image)[0], 0.0f, false, 1.0f, 0, 0, 0, true);

            statistical_images.at(correlation_pixel_sum_of_squares_image)[0].ForwardFFT( );
            statistical_images.at(correlation_pixel_sum_of_squares_image)[0].ClipInto(&tmp_sum_sq, 0.0f, false, 1.0f, 0, 0, 0, true);
            tmp_sum_sq.BackwardFFT( );
            statistical_images.at(correlation_pixel_sum_of_squares_image)[0].Allocate(image_size.x, image_size.y, image_size.z, true);
            tmp_sum_sq.ClipInto(&statistical_images.at(correlation_pixel_sum_of_squares_image)[0], 0.0f, false, 1.0f, 0, 0, 0, true);
        } // end resampling_is_wanted

        // Create a mask that will be filled based on the possibly rotated and resized search image, and then rescaled in the same manner, so that we can use this for adjusting the
        // stats images/ histogram elsewhere post resizing.
        valid_area_mask.Allocate(statistical_images.at(max_intensity_projection)[0].logical_x_dimension, statistical_images.at(max_intensity_projection)[0].logical_y_dimension, 1, true);
        valid_area_mask.SetToConstant(1.0f);
        constexpr float mask_radius = 7.f;

        valid_area_mask.CosineRectangularMask(x_radius, y_radius, 0, mask_radius, false, true, 0.f);

        valid_area_mask.Binarise(0.9f);
        valid_area_mask.ZeroFFTWPadding( );

        if ( resampling_is_wanted ) {
            FillInNearestNeighbors(statistical_images.at(best_psi)[0], tmp_psi, valid_area_mask, NN_no_value);
            FillInNearestNeighbors(statistical_images.at(best_phi)[0], tmp_phi, valid_area_mask, NN_no_value);
            FillInNearestNeighbors(statistical_images.at(best_theta)[0], tmp_theta, valid_area_mask, NN_no_value);
            FillInNearestNeighbors(statistical_images.at(best_defocus)[0], tmp_defocus, valid_area_mask, NN_no_value);
            FillInNearestNeighbors(statistical_images.at(best_pixel_size)[0], tmp_pixel_size, valid_area_mask, NN_no_value);
        }

        // For the other images, calculate the mean under the mask and change the padding to this so the display contrast is okay
        double mip_mean        = 0.0;
        double phi_mean        = 0.0;
        double theta_mean      = 0.0;
        double psi_mean        = 0.0;
        double defocus_mean    = 0.0;
        double pixel_size_mean = 0.0;
        double n_counted       = 0.0;

        for ( long address = 0; address < statistical_images.at(max_intensity_projection)[0].real_memory_allocated; address++ ) {
            n_counted += valid_area_mask.real_values[address];
            if ( valid_area_mask.real_values[address] > 0.0f ) {
                mip_mean += statistical_images.at(max_intensity_projection)[0].real_values[address] * valid_area_mask.real_values[address];
                phi_mean += statistical_images.at(best_phi)[0].real_values[address] * valid_area_mask.real_values[address];
                theta_mean += statistical_images.at(best_theta)[0].real_values[address] * valid_area_mask.real_values[address];
                psi_mean += statistical_images.at(best_psi)[0].real_values[address] * valid_area_mask.real_values[address];
                defocus_mean += statistical_images.at(best_defocus)[0].real_values[address] * valid_area_mask.real_values[address];
                pixel_size_mean += statistical_images.at(best_pixel_size)[0].real_values[address] * valid_area_mask.real_values[address];
            }
        }

        for ( long address = 0; address < statistical_images.at(max_intensity_projection)[0].real_memory_allocated; address++ ) {
            if ( valid_area_mask.real_values[address] == 0.0f ) {
                statistical_images.at(max_intensity_projection)[0].real_values[address]               = mip_mean / n_counted;
                statistical_images.at(best_phi)[0].real_values[address]                               = phi_mean / n_counted;
                statistical_images.at(best_theta)[0].real_values[address]                             = theta_mean / n_counted;
                statistical_images.at(best_psi)[0].real_values[address]                               = psi_mean / n_counted;
                statistical_images.at(best_defocus)[0].real_values[address]                           = defocus_mean / n_counted;
                statistical_images.at(best_pixel_size)[0].real_values[address]                        = pixel_size_mean / n_counted;
                statistical_images.at(correlation_pixel_sum_of_squares_image)[0].real_values[address] = 0.0f;
                statistical_images.at(correlation_pixel_sum_image)[0].real_values[address]            = 0.0f;
            }
        }
    }

    return avg_sampling;
}

void TemplateMatchingDataSizer::FillInNearestNeighbors(Image& output_image, Image& nn_upsampled_image, Image& valid_area_mask, const float no_value) {

    // Set the non-valid area to zero (not no_value) so that we can use the no_value to check if the pixel has been filled in.
    nn_upsampled_image.MultiplyPixelWise(valid_area_mask);
    output_image.CopyFrom(&nn_upsampled_image);
    int size_neighborhood = 3;
    while ( float(size_neighborhood) < cistem::match_template::MAX_BINNING_FACTOR ) {
        if ( GetFullBinningFactor( ) <= float(size_neighborhood) ) {
            break;
        }
        else
            size_neighborhood += 2;
    }
    // We could try to dilate out each neighborhood, but this will be slower given the bad memory access. Better to do a little extra.
    int offset_max = size_neighborhood / 2;

    // Loop over the image
    for ( int j = 0; j < nn_upsampled_image.logical_y_dimension; j++ ) {
        for ( int i = 0; i < nn_upsampled_image.logical_x_dimension; i++ ) {
            float current_value = nn_upsampled_image.ReturnRealPixelFromPhysicalCoord(i, j, 0);
            if ( current_value == no_value ) {
                int   min_distance_squared = std::numeric_limits<int>::max( );
                float closest_value        = no_value;

                // First check the line in memory that includes the current pixel, setting boundaries in the for loop
                for ( int x = std::max(i - offset_max, 0); x <= std::min(i + offset_max, nn_upsampled_image.logical_x_dimension - 1); x++ ) {
                    // We don't need to check the current pixel
                    if ( x != i ) {
                        // No need to load the value if the distance is already too large
                        if ( x * x < min_distance_squared ) {
                            current_value = nn_upsampled_image.ReturnRealPixelFromPhysicalCoord(x, j, 0);
                            if ( current_value != no_value ) {
                                min_distance_squared = x * x;
                                closest_value        = current_value;
                            }
                        }
                    }
                }
                // If we still haven't found it, we'll check each row left and right,
                int y_offset = 1;

                // We can't get any closer than 1, so if we've already found a value, we can stop
                if ( min_distance_squared == 1 ) {
                    goto endOfElse;
                }

                while ( y_offset <= offset_max ) {
                    for ( int y = j - y_offset; y <= j + y_offset; y += 2 * y_offset ) {
                        // We can't set the limits in the for loop initializer and just bracket the end, so use a conditional here
                        if ( y < 0 || y >= nn_upsampled_image.logical_y_dimension ) {
                            continue;
                        }
                        for ( int x = std::max(i - offset_max, 0); x <= std::min(i + offset_max, nn_upsampled_image.logical_x_dimension - 1); x++ ) {
                            // This time we hit all pixels in the line

                            // No need to load the value if the distance is already too large
                            if ( y * y + x * x < min_distance_squared ) {
                                current_value = nn_upsampled_image.ReturnRealPixelFromPhysicalCoord(x, y, 0);
                                if ( current_value != no_value ) {
                                    min_distance_squared = y * y + x * x;
                                    closest_value        = current_value;
                                }
                            }
                            // The smallest distance we can get now is == to the x offset
                            if ( min_distance_squared == y_offset * y_offset ) {
                                goto endOfElse;
                            }
                        }
                    }
                    // If we get here, we've checked the full neighborhood size y_offset * 2, which meanse the smallest distance > y_offset is the corner, = sqrt(y_offset^2 + y_offset^2)
                    if ( min_distance_squared == 2 * y_offset * y_offset ) {
                        goto endOfElse;
                    }
                    y_offset++;
                }

            endOfElse:
                MyDebugAssertFalse(closest_value == no_value, "No value found");
                output_image.real_values[output_image.ReturnReal1DAddressFromPhysicalCoord(i, j, 0)] = closest_value;
            }
        }
    }
}
