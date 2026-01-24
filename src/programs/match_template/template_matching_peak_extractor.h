#ifndef __SRC_PROGRAMS_MATCH_TEMPLATE_TEMPLATE_MATCHING_PEAK_EXTRACTOR_H_
#define __SRC_PROGRAMS_MATCH_TEMPLATE_TEMPLATE_MATCHING_PEAK_EXTRACTOR_H_

#include "../../core/core_headers.h"

/**
 * @brief Handles peak extraction, masking, and projection insertion for template matching results.
 *
 * This class consolidates the shared peak processing logic between match_template and
 * make_template_result programs, eliminating code duplication and ensuring both use
 * the efficient peak masking algorithm.
 *
 * Features:
 * - Dual mode: search for peaks in MIP or read from coordinate file
 * - Efficient peak masking (bounded loop, not full image scan)
 * - Optional peak correction via FFT resampling
 * - Projection extraction and insertion into result image
 * - Optional slab insertion (make_template_result only)
 * - Optional padding support (make_template_result only)
 */
class TemplateMatchingPeakExtractor {
  public:
    /**
     * @brief Constructor for peak extractor
     *
     * @param mip_image Maximum intensity projection image (will be modified by masking peaks)
     * @param phi_image, theta_image, psi_image Euler angle images
     * @param defocus_image, pixel_size_image Metadata images
     * @param result_image Output montage image where projections are inserted
     * @param input_reconstruction 3D reconstruction for extracting projections
     * @param current_projection Workspace image for projection extraction
     * @param padded_projection Optional workspace for padded projections (nullptr if not used)
     * @param slab Optional 3D slab image for insertion (nullptr if not used)
     * @param binned_reconstruction Optional binned reconstruction for slab (nullptr if not used)
     * @param coordinate_file Optional file to read peaks from (nullptr for search mode)
     * @param threshold Peak height threshold for search mode
     * @param min_peak_radius_squared Minimum peak separation (squared)
     * @param input_pixel_size Original/unbinned pixel size in Angstroms
     * @param search_pixel_size Pixel size used during search (may be binned) in Angstroms
     * @param binned_pixel_size Binned pixel size (only needed if slab is used)
     * @param enable_peak_correction Enable FFT-based peak correction
     * @param peak_search_threshold_scale Scale factor for initial peak search when resampling (default 0.95)
     */
    TemplateMatchingPeakExtractor(
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
            float            binned_pixel_size,
            bool             enable_peak_correction,
            float            peak_search_threshold_scale = 0.95f);

    /**
     * @brief Process the next peak: find/read, extract metadata, create projection, insert into result
     *
     * @param angles Workspace for angle calculations
     * @param number_of_peaks_found Counter for peaks processed (will be incremented)
     * @return std::pair<bool, TemplateMatchFoundPeakInfo> - bool indicates success, peak_info contains the data
     */
    std::pair<bool, TemplateMatchFoundPeakInfo> ProcessNextPeak(AnglesAndShifts& angles, int& number_of_peaks_found);

    void                    SortPeakInfoByPeakHeight(ArrayOfTemplateMatchFoundPeakInfos& arr);
    cistem_timer::StopWatch peak_timer;

  private:
    // Image references
    Image& mip_image_;
    Image& phi_image_;
    Image& theta_image_;
    Image& psi_image_;
    Image& defocus_image_;
    Image& pixel_size_image_;
    Image& result_image_;
    Image& input_reconstruction_;
    Image& current_projection_;

    // In case we are fixing peaks, we need to have a seperate image for erasing the peak radius.
    Image masked_mip_;

    // Optional features (nullptr if not used)
    Image*           padded_projection_;
    Image*           slab_;
    Image*           binned_reconstruction_;
    NumericTextFile* coordinate_file_;

    // Parameters
    float threshold_;
    float min_peak_radius_squared_;
    float input_pixel_size_;
    float search_pixel_size_;
    float binned_3d_pixel_size_;
    bool  enable_peak_correction_;
    float peak_search_threshold_scale_;
    float fourier_scaling_factor_;

    // Peak correction members (only allocated if enabled)
    Image base_peak_;
    Image resampled_peak_;
    int   base_peak_size_;
    int   resampled_peak_size_;
    int   base_peak_first_element_offset_;
};

#endif
