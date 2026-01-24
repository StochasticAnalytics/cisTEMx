#include "template_matching_peak_extractor.h"

TemplateMatchingPeakExtractor::TemplateMatchingPeakExtractor(
        Image&           mip_image,
        Image&           phi_image,
        Image&           theta_image,
        Image&           psi_image,
        Image&           defocus_image,
        Image&           pixel_size_image,
        Image&           result_image,
        Image&           input_reconstruction,
        Image&           current_projection,
        Image*           padded_projection,
        Image*           slab,
        Image*           binned_reconstruction,
        NumericTextFile* coordinate_file,
        float            threshold,
        float            min_peak_radius_squared,
        float            input_pixel_size,
        float            search_pixel_size,
        float            binned_3d_pixel_size,
        bool             enable_peak_correction,
        float            peak_search_threshold_scale)
    : mip_image_(mip_image),
      phi_image_(phi_image),
      theta_image_(theta_image),
      psi_image_(psi_image),
      defocus_image_(defocus_image),
      pixel_size_image_(pixel_size_image),
      result_image_(result_image),
      input_reconstruction_(input_reconstruction),
      current_projection_(current_projection),
      padded_projection_(padded_projection),
      slab_(slab),
      binned_reconstruction_(binned_reconstruction),
      coordinate_file_(coordinate_file),
      threshold_(threshold),
      min_peak_radius_squared_(min_peak_radius_squared),
      input_pixel_size_(input_pixel_size),
      search_pixel_size_(search_pixel_size),
      binned_3d_pixel_size_(binned_3d_pixel_size),
      enable_peak_correction_(enable_peak_correction),
      peak_search_threshold_scale_(peak_search_threshold_scale) {

    // Verify all parameter images have the same dimensions as the MIP
    MyDebugAssertTrue(phi_image_.HasSameDimensionsAs(&mip_image_), "Phi image must have same dimensions as MIP");
    MyDebugAssertTrue(theta_image_.HasSameDimensionsAs(&mip_image_), "Theta image must have same dimensions as MIP");
    MyDebugAssertTrue(psi_image_.HasSameDimensionsAs(&mip_image_), "Psi image must have same dimensions as MIP");
    MyDebugAssertTrue(defocus_image_.HasSameDimensionsAs(&mip_image_), "Defocus image must have same dimensions as MIP");
    MyDebugAssertTrue(pixel_size_image_.HasSameDimensionsAs(&mip_image_), "Pixel size image must have same dimensions as MIP");

    // Calculate binning factor and resize reconstruction if needed
    float binning_factor = search_pixel_size_ / input_pixel_size_;

    if ( binning_factor > 1.0f ) {
        // Resize reconstruction to match search pixel size
        int new_size = int(input_reconstruction_.logical_x_dimension / binning_factor + 0.5f);
        if ( IsOdd(new_size) )
            new_size++;
        input_reconstruction_.ForwardFFT( );
        input_reconstruction_.Resize(new_size, new_size, new_size);
        input_reconstruction_.BackwardFFT( );
    }

    // Normalize reconstruction
    float max_density = input_reconstruction_.ReturnAverageOfMaxN( );
    input_reconstruction_.DivideByConstant(max_density);

    // Prepare reconstruction for projection extraction
    input_reconstruction_.ForwardFFT( );
    input_reconstruction_.MultiplyByConstant(sqrtf(input_reconstruction_.logical_x_dimension * input_reconstruction_.logical_y_dimension * sqrtf(input_reconstruction_.logical_z_dimension)));
    input_reconstruction_.ZeroCentralPixel( );
    input_reconstruction_.SwapRealSpaceQuadrants( );

    masked_mip_.CopyFrom(&mip_image_);

    const int base_peak_size      = 7;
    const int resampled_peak_size = 10 * base_peak_size;
    base_peak_size_               = base_peak_size;
    resampled_peak_size_          = resampled_peak_size;
    int neighborhood              = base_peak_size / 2;

    min_peak_radius_squared_ = std::max(min_peak_radius_squared_, float(pow(neighborhood, 2)));

    int mip_stride                  = mip_image_.logical_x_dimension + mip_image_.padding_jump_value;
    base_peak_first_element_offset_ = neighborhood * mip_stride + neighborhood;

    if ( enable_peak_correction_ ) {
        base_peak_.Allocate(base_peak_size, base_peak_size_, 1, true);
        resampled_peak_.Allocate(resampled_peak_size, resampled_peak_size, 1, false);
    }
}

std::pair<bool, TemplateMatchFoundPeakInfo> TemplateMatchingPeakExtractor::ProcessNextPeak(AnglesAndShifts& angles, int& number_of_peaks_found) {

    TemplateMatchFoundPeakInfo peak_info;
    Peak                       current_peak;
    float                      current_phi;
    float                      current_theta;
    float                      current_psi;
    float                      current_defocus;
    float                      current_pixel_size;

    // Step 1: Get peak information (either by searching or reading from file)
    if ( coordinate_file_ != nullptr ) {
        // Read coordinates from file
        float coordinates[8];
        coordinate_file_->ReadLine(coordinates);
        number_of_peaks_found++;

        current_psi        = coordinates[0];
        current_theta      = coordinates[1];
        current_phi        = coordinates[2];
        current_peak.x     = coordinates[3] / search_pixel_size_;
        current_peak.y     = coordinates[4] / search_pixel_size_;
        current_defocus    = coordinates[5];
        current_pixel_size = coordinates[6];
        current_peak.value = coordinates[7];
    }
    else {
        // Search for peak in MIP - loop until we find a valid peak or run out
        // When peak correction is enabled, use scaled threshold for initial search
        float search_threshold = enable_peak_correction_ ? (threshold_ * peak_search_threshold_scale_) : threshold_;
        int   min_peak_radius  = int(sqrtf(min_peak_radius_squared_));
        bool  peak_accepted    = false;

        while ( ! peak_accepted ) {
            peak_timer.start("Find Peak");
            current_peak = masked_mip_.FindPeakWithIntegerCoordinates(0.0, std::numeric_limits<float>::max( ));
            peak_timer.lap("Find Peak");

            if ( current_peak.value < search_threshold )
                return {false, peak_info};

            // Adjust peak coordinates
            current_peak.x = current_peak.x + mip_image_.physical_address_of_box_center_x;
            current_peak.y = current_peak.y + mip_image_.physical_address_of_box_center_y;

            // Extract angles and metadata using efficient loop from match_template
            float sq_dist_x, sq_dist_y;
            long  address;
            bool  peak_corrected_and_gt_thr = false;
            bool  peak_out_of_bounds        = false;

            for ( int j = std::max(myroundint(current_peak.y) - min_peak_radius, 0); j < std::min(myroundint(current_peak.y) + min_peak_radius, mip_image_.logical_y_dimension); j++ ) {
                sq_dist_y = float(j) - current_peak.y;
                sq_dist_y *= sq_dist_y;

                for ( int i = std::max(myroundint(current_peak.x) - min_peak_radius, 0); i < std::min(myroundint(current_peak.x) + min_peak_radius, mip_image_.logical_x_dimension); i++ ) {
                    sq_dist_x = float(i) - current_peak.x;
                    sq_dist_x *= sq_dist_x;
                    address = phi_image_.ReturnReal1DAddressFromPhysicalCoord(i, j, 0);

                    // Extract metadata at peak center
                    if ( sq_dist_x == 0 && sq_dist_y == 0 ) {
                        peak_timer.start("Read stats");
                        current_phi        = phi_image_.real_values[address];
                        current_theta      = theta_image_.real_values[address];
                        current_psi        = psi_image_.real_values[address];
                        current_defocus    = defocus_image_.real_values[address];
                        current_pixel_size = pixel_size_image_.real_values[address];
                        peak_timer.lap("Read stats");
                        if ( enable_peak_correction_ ) {
                            // Extract base peak region
                            long peak_address_mip = address - base_peak_first_element_offset_;
                            int  peak_address     = 0;
                            int  mip_stride       = mip_image_.logical_x_dimension + mip_image_.padding_jump_value;
                            peak_timer.start("Resample peak stats");
                            if ( peak_address_mip > 0 && peak_address_mip + base_peak_size_ * mip_stride + base_peak_size_ < mip_image_.real_memory_allocated ) {
                                for ( int peak_j = 0; peak_j < base_peak_size_; peak_j++ ) {
                                    for ( int peak_i = 0; peak_i < base_peak_size_; peak_i++ ) {
                                        base_peak_.real_values[peak_address] = mip_image_.real_values[peak_address_mip];
                                        peak_address++;
                                        peak_address_mip++;
                                    }
                                    peak_address += base_peak_.padding_jump_value;
                                    peak_address_mip += mip_stride - base_peak_size_;
                                }

                                // base_peak_.QuickAndDirtyWriteSlice(stack_fn, number_of_peaks_found + 1);
                                // base_peak_.GaussianLowPassFilter(5.f / search_pixel_size_);
                                // Resample peak to higher resolution
                                resampled_peak_.is_in_real_space = false;
                                resampled_peak_.SetToConstant(0.f);
                                base_peak_.ForwardFFT( );

                                base_peak_.ClipInto(&resampled_peak_);
                                resampled_peak_.BackwardFFT( );
                                // resampled_peak_.MultiplyByConstant(4.f);

                                Peak resampled_peak_val = resampled_peak_.FindPeakWithIntegerCoordinates(0.0, std::numeric_limits<float>::max( ));

                                // Only accept the corrected peak if it exceeds the original threshold
                                if ( resampled_peak_val.value >= threshold_ ) {
                                    current_peak.value        = resampled_peak_val.value;
                                    peak_corrected_and_gt_thr = true;
                                }

                                // Clean up
                                base_peak_.is_in_real_space = true;
                                base_peak_.SetToConstant(0.f);
                            }
                            peak_timer.start("Resample peak stats");
                            // No need for an else clause. If we cannot extract the peak because it is out of bounds,
                            // then peak_corrected_and_gt_thr remains false. We do need to catch the case that the orignal peak was
                            // already > the threshold below when we check acceptance
                        }
                    }

                    peak_timer.start("Zero out radius");
                    // Mask out the region around this peak
                    if ( sq_dist_x + sq_dist_y <= min_peak_radius_squared_ ) {
                        masked_mip_.real_values[address] = -std::numeric_limits<float>::max( );
                    }
                    peak_timer.lap("Zero out radius");
                }
            }

            // Accept peak if: no correction enabled then we already checked the third condition (peak > thr), otherwise check the bool to
            // see if we have a corrected peak > threshold
            if ( ! enable_peak_correction_ || peak_corrected_and_gt_thr || current_peak.value > threshold_ ) {
                peak_accepted = true;
                number_of_peaks_found++;
            }
            // Otherwise loop continues to search for next peak
        }
    }

    // Step 2: Populate peak_info structure
    peak_info.x_pos       = current_peak.x * search_pixel_size_;
    peak_info.y_pos       = current_peak.y * search_pixel_size_;
    peak_info.phi         = current_phi;
    peak_info.theta       = current_theta;
    peak_info.psi         = current_psi;
    peak_info.defocus     = current_defocus;
    peak_info.pixel_size  = current_pixel_size;
    peak_info.peak_height = current_peak.value;

    // Step 3: Extract projection from reconstruction
    angles.Init(current_phi, current_theta, current_psi, 0.0, 0.0);

    peak_timer.start("extract result slice");
    if ( padded_projection_ != nullptr ) {
        // Handle padding workflow (make_template_result)
        input_reconstruction_.ExtractSlice(*padded_projection_, angles, 1.0f, false);
        padded_projection_->SwapRealSpaceQuadrants( );
        padded_projection_->BackwardFFT( );
        padded_projection_->ClipInto(&current_projection_);
        current_projection_.ForwardFFT( );
    }
    else {
        // Standard workflow (match_template)
        input_reconstruction_.ExtractSlice(current_projection_, angles, 1.0f, false);
        current_projection_.SwapRealSpaceQuadrants( );
    }
    peak_timer.lap("extract result slice");

    peak_timer.start("Normalize");
    current_projection_.MultiplyByConstant(sqrtf(current_projection_.logical_x_dimension * current_projection_.logical_y_dimension));
    current_projection_.BackwardFFT( );
    current_projection_.AddConstant(-current_projection_.ReturnAverageOfRealValuesOnEdges( ));
    peak_timer.lap("Normalize");

    peak_timer.start("Insert result slice");
    // Step 4: Insert projection into result image
    result_image_.InsertOtherImageAtSpecifiedPosition(&current_projection_,
                                                      current_peak.x - result_image_.physical_address_of_box_center_x,
                                                      current_peak.y - result_image_.physical_address_of_box_center_y,
                                                      0, 0.0f);
    peak_timer.lap("Normalize");

    peak_timer.start("Slab insertion");
    // Step 5: Handle slab insertion (make_template_result only)
    if ( slab_ != nullptr && binned_reconstruction_ != nullptr ) {
        Image rotated_reconstruction;
        angles.Init(-current_psi, -current_theta, -current_phi, 0.0, 0.0);
        rotated_reconstruction.CopyFrom(binned_reconstruction_);
        rotated_reconstruction.Rotate3DByRotationMatrixAndOrApplySymmetry(angles.euler_matrix);

        slab_->InsertOtherImageAtSpecifiedPosition(&rotated_reconstruction,
                                                   myroundint((current_peak.x - result_image_.physical_address_of_box_center_x) / (search_pixel_size_ / binned_3d_pixel_size_)),
                                                   myroundint((current_peak.y - result_image_.physical_address_of_box_center_y) / (search_pixel_size_ / binned_3d_pixel_size_)),
                                                   -myroundint(current_defocus / binned_3d_pixel_size_),
                                                   0.0f);
    }
    peak_timer.lap("Slab insertion");

    return {true, peak_info};
}

// Comparator: return <0, 0, >0 like strcmp
int wxCMPFUNC_CONV ComparePeakInfoByPeakHeight(TemplateMatchFoundPeakInfo** a, TemplateMatchFoundPeakInfo** b) {
    if ( (*a)->peak_height < (*b)->peak_height )
        return 1;
    if ( (*a)->peak_height > (*b)->peak_height )
        return -1;
    return 0;
}

void TemplateMatchingPeakExtractor::SortPeakInfoByPeakHeight(ArrayOfTemplateMatchFoundPeakInfos& arr) {
    arr.Sort(ComparePeakInfoByPeakHeight);
}