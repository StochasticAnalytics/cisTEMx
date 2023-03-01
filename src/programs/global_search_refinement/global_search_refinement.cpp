#include <cistem_config.h>
#include "../refine3d/refine3d_defines.h"

#ifdef ENABLEGPU
#warning "GPU enabled in refine3d"
#include "../../gpu/gpu_core_headers.h"
#include "../../gpu/GpuImage.h"
#else
#include "../../core/core_headers.h"
#endif

#include "../refine3d/ProjectionComparisonObjects.h"

#ifdef CISTEM_PROFILING
using namespace cistem_timer;
#else
using namespace cistem_timer_noop;
#endif

//#define PRINT_GLOBAL_SEARCH_REFINEMENT_EXTRA_INFO

#define SHIFT_AND_RECALCULATE_SCORE

class GlobalSearchRefinementApp : public MyApp {
  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

    wxString input_search_images;
    wxString input_reconstruction;

    cisTEMParameters input_star_file;
    cisTEMParameters output_star_file;
    wxString         input_star_filename;
    wxString         output_star_filename;

    float low_resolution_limit    = 300.0f;
    float high_resolution_limit   = 8.0f;
    float angular_range           = 2.0f;
    float angular_step            = 5.0f;
    int   best_parameters_to_keep = 20;
    float defocus_search_range    = 1000;
    float defocus_search_step     = 10;

    //	float		pixel_size_refine_step = 0.001f;
    float    padding               = 1.0;
    bool     do_defocus_refinement = false;
    float    wanted_mask_radius    = 0.0f;
    wxString my_symmetry           = "C1";
    float    in_plane_angular_step = 0;
    float    wanted_threshold;
    float    min_peak_radius;

    int max_threads;

  private:
};

// TODO: replace this with the ProjectionComparisonObject class and setup for GPU
class TemplateComparisonObject {

    long   _n_pixels_in_variance_estimate;
    double _sum_of_pixel_values;
    double _sum_of_pixel_values_squared;
    long   _number_of_real_space_pixels;
    double _variance_estimate;
    bool   _is_score_adjusted_by_size;
    int    _n_updates;

    EmpiricalDistribution _score_distribution;
    float                 _mask_radius;

  public:
    Image* input_reconstruction;
    Image* windowed_particle;
    Image* projection_filter;
    Image  current_projection;
#ifdef SHIFT_AND_RECALCULATE_SCORE
    Image copy_of_current_projection;
#endif
    Image padded_projection;

    AnglesAndShifts* angles;
    float            pixel_size_factor;

    void ZeroVarianceEstimate( ) {
        _n_pixels_in_variance_estimate = 0;
        _sum_of_pixel_values           = 0;
        _sum_of_pixel_values_squared   = 0;
        _number_of_real_space_pixels   = projection_filter->number_of_real_space_pixels;
        _is_score_adjusted_by_size     = false;
        _n_updates                     = 0;
        _variance_estimate             = sqrt(double(_number_of_real_space_pixels));
    }

    void ZeroImages( ) {
        current_projection.Allocate(projection_filter->logical_x_dimension, projection_filter->logical_x_dimension, false);
        current_projection.SetToConstant(0.0f);
        // just in case we didn't allocate, make sure we don't trip up the backward FFT routine
        current_projection.is_in_real_space = false;
#ifdef SHIFT_AND_RECALCULATE_SCORE
        copy_of_current_projection.Allocate(projection_filter->logical_x_dimension, projection_filter->logical_x_dimension, false);
        copy_of_current_projection.SetToConstant(0.0f);
        // just in case we didn't allocate, make sure we don't trip up the backward FFT routine
        copy_of_current_projection.is_in_real_space = false;
#endif
        if ( input_reconstruction->logical_x_dimension != current_projection.logical_x_dimension ) {
            MyAssertTrue(false, "input_reconstruction->logical_x_dimension != current_projection.logical_x_dimension");
            // padded_projection.Allocate(->input_reconstruction->logical_x_dimension, input_reconstruction->logical_x_dimension, false);
            // padded_projection.SetToConstant(0.0f);
            // padded_projection.is_in_real_space = false;
        }
    }

    void Zero(float wanted_mask_radius) {
        // We need to call ZeroImages first, or else there is a change we get zero image size
        ZeroImages( );
        ZeroVarianceEstimate( );
        _score_distribution.Reset( );
        _mask_radius = wanted_mask_radius;
    }

    float GetSigma( ) {
        return _score_distribution.GetSampleStandardDeviation( );
    }

    float GetMean( ) {
        return _score_distribution.GetSampleMean( );
    }

    void UpdateVarianceEstimate( ) {
        // FIXME this ignores the possible use of the non copy

        copy_of_current_projection.UpdateDistributionOfRealValues(&_score_distribution, _mask_radius);
        // wxPrintf("Update %i: %f %f %f %ld\n", _n_updates, _variance_estimate, _sum_of_pixel_values, _sum_of_pixel_values_squared, _number_of_real_space_pixels);
    }

    void AdjustScoreByNumberOfPixels(float& score) {
        MyDebugAssertTrue(_number_of_real_space_pixels > 0, "Number of real space pixels is zero");
        score *= _variance_estimate;
        _is_score_adjusted_by_size = true;
    }

    void GetPeak(Peak& my_peak, bool is_copy_of_prj) {
        if ( is_copy_of_prj )
            my_peak = copy_of_current_projection.FindPeakWithParabolaFit( );
        else
            my_peak = current_projection.FindPeakWithParabolaFit( );
    }

    void AdjustScoreByVarianceEstimate(float& score) {
        float undo_prior_adjustment = 1.0f;
        if ( _is_score_adjusted_by_size ) {
            undo_prior_adjustment = _variance_estimate;
        }
        // wxPrintf("Variance estimate is %f, bool is %i %f\n", _score_distribution.GetSampleStandardDeviation( ), _is_score_adjusted_by_size, undo_prior_adjustment);

        score /= undo_prior_adjustment;
        score = (score - _score_distribution.GetSampleMean( )) / _score_distribution.GetSampleStandardDeviation( );
    }

    //	int							slice = 1;
};

// This is the function which will be minimized
Peak TemplateScore(void* scoring_parameters, StopWatch& timer) {
    TemplateComparisonObject* comparison_object = reinterpret_cast<TemplateComparisonObject*>(scoring_parameters);
    //	Peak box_peak;
    // FIXME: ALlocating this on every loop is bonkers
    timer.start("zero images");
    comparison_object->ZeroImages( );
    timer.lap("zero images");

    if ( comparison_object->input_reconstruction->logical_x_dimension != comparison_object->current_projection.logical_x_dimension ) {
        MyAssertTrue(false, "Not implemented");
        // comparison_object->input_reconstruction->ExtractSlice(&comparison_object->padded_projection, *comparison_object->angles, 1.0f, false);
        // padded_projection.SwapRealSpaceQuadrants( );
    }
    else {
        timer.start("to projection");
        comparison_object->input_reconstruction->ExtractSlice(comparison_object->current_projection, *comparison_object->angles, 1.0f, false);
        timer.lap("to projection");
        timer.start("swap quadrants");
        comparison_object->current_projection.SwapRealSpaceQuadrants( );
        timer.lap("swap quadrants");
    }

    timer.start("apply filter");
    comparison_object->current_projection.MultiplyPixelWise(*comparison_object->projection_filter);

    comparison_object->current_projection.ZeroCentralPixel( );
    comparison_object->current_projection.DivideByConstant(sqrtf(comparison_object->current_projection.ReturnSumOfSquares( )));
    timer.lap("apply filter");
#ifdef SHIFT_AND_RECALCULATE_SCORE
    timer.start("clean copy");
    comparison_object->copy_of_current_projection.CopyFrom(&comparison_object->current_projection);
    timer.lap("clean copy");
#endif

    timer.start("complex conj mul");
#ifdef MKL
    // Use the MKL
    vmcMulByConj(comparison_object->current_projection.real_memory_allocated / 2, reinterpret_cast<MKL_Complex8*>(comparison_object->windowed_particle->complex_values), reinterpret_cast<MKL_Complex8*>(comparison_object->current_projection.complex_values), reinterpret_cast<MKL_Complex8*>(comparison_object->current_projection.complex_values), VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
#else
    for ( long pixel_counter = 0; pixel_counter < comparison_object->current_projection.real_memory_allocated / 2; pixel_counter++ ) {
        comparison_object->current_projection.complex_values[pixel_counter] = std::conj(comparison_object->current_projection.complex_values[pixel_counter]) * comparison_object->windowed_particle->complex_values[pixel_counter];
    }
#endif
    timer.lap("complex conj mul");
    timer.start("fft1");
    comparison_object->current_projection.BackwardFFT( );
    timer.lap("fft1");

    //	wxPrintf("ping");

    timer.start("Get peak");
    // FIXME: This is a hack to get the peak to work
    Peak tmp_peak;
    comparison_object->GetPeak(tmp_peak, false);
    comparison_object->AdjustScoreByNumberOfPixels(tmp_peak.value);

    float initial_x = tmp_peak.x;
    float initial_y = tmp_peak.y;
    timer.lap("Get peak");

#ifdef SHIFT_AND_RECALCULATE_SCORE
    timer.start("phse shift");
    comparison_object->copy_of_current_projection.PhaseShift(initial_x, initial_y);
    timer.lap("phse shift");

    timer.start("conj mul 2");
#ifdef MKL
    // Use the MKL
    vmcMulByConj(comparison_object->copy_of_current_projection.real_memory_allocated / 2, reinterpret_cast<MKL_Complex8*>(comparison_object->windowed_particle->complex_values), reinterpret_cast<MKL_Complex8*>(comparison_object->copy_of_current_projection.complex_values), reinterpret_cast<MKL_Complex8*>(comparison_object->copy_of_current_projection.complex_values), VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
#else
    for ( long pixel_counter = 0; pixel_counter < current_projection.real_memory_allocated / 2; pixel_counter++ ) {
        comparison_object->copy_of_current_projection.complex_values[pixel_counter] = std::conj(comparison_object->copy_of_current_projection.complex_values[pixel_counter]) * comparison_object->windowed_particle->complex_values[pixel_counter];
    }

#endif
    //	wxPrintf("ping");
    timer.lap("conj mul 2");
    timer.start("fft2");
    comparison_object->copy_of_current_projection.BackwardFFT( );
    timer.lap("fft2");
    timer.start("Update variance estimate");
    comparison_object->UpdateVarianceEstimate( );
    timer.lap("Update variance estimate");

    timer.start("Get peak 2");
    // FIXME: This is a hack to get the peak to work
    comparison_object->GetPeak(tmp_peak, true);
    comparison_object->AdjustScoreByNumberOfPixels(tmp_peak.value);
    tmp_peak.x += initial_x;
    tmp_peak.y += initial_y;
    timer.lap("Get peak 2");

#endif
    return tmp_peak;
}

IMPLEMENT_APP(GlobalSearchRefinementApp)

// override the DoInteractiveUserInput

void GlobalSearchRefinementApp::DoInteractiveUserInput( ) {

    UserInput* my_input = new UserInput("RefineTemplate", 1.00);

    // This block is the same as in globale_search.cpp
    input_search_images  = my_input->GetFilenameFromUser("Input images to be searched", "The input image stack, containing the images that should be searched", "image_stack.mrc", true);
    input_reconstruction = my_input->GetFilenameFromUser("Input template reconstruction", "The 3D reconstruction from which projections are calculated", "reconstruction.mrc", true);

    input_star_filename  = my_input->GetFilenameFromUser("Input star file", "The input star file, containing the images that should be searched", "input_particles.star", true);
    output_star_filename = my_input->GetFilenameFromUser("output star file", "The output star file, containing the images that should be searched", "input_particles.star", false);

    wxFileName directory_for_results(my_input->GetFilenameFromUser("Output directory for results, subdirectory using image name will be created.", "", "./", false));
    MyDebugAssertFalse(directory_for_results.HasExt( ), "Output directory should not have an extension");
    if ( ! directory_for_results.DirExists( ) ) {
        MyDebugPrint("Output directory does not exist, creating it");
        directory_for_results.Mkdir(0777, wxPATH_MKDIR_FULL);
    }
    wxFileName input_search_image_file_name_full(input_search_images);
    wxString   directory_for_results_string = directory_for_results.GetFullPath( );

    low_resolution_limit  = my_input->GetFloatFromUser("Low resolution limit (A)", "Low resolution limit of the data used for alignment in Angstroms", "300.0", 0.0);
    high_resolution_limit = my_input->GetFloatFromUser("High resolution limit (A)", "High resolution limit of the data used for alignment in Angstroms", "8.0", 0.0);
    //	angular_range = my_input->GetFloatFromUser("Angular refinement range", "AAngular range to refine", "2.0", 0.1);
    angular_step          = my_input->GetFloatFromUser("Out of plane angular step", "Angular step size for global grid search", "0.2", 0.00);
    in_plane_angular_step = my_input->GetFloatFromUser("In plane angular step", "Angular step size for in-plane rotations during the search", "0.1", 0.00);
    //	best_parameters_to_keep = my_input->GetIntFromUser("Number of top hits to refine", "The number of best global search orientations to refine locally", "20", 1);
    do_defocus_refinement = my_input->GetYesNoFromUser("Refine defocus", "Should the particle defocus be refined?", "No");
    defocus_search_range  = my_input->GetFloatFromUser("Defocus search range (A) (0.0 = no search)", "Search range (-value ... + value) around current defocus", "200.0", 0.0);
    defocus_search_step   = my_input->GetFloatFromUser("Desired defocus accuracy (A)", "Accuracy to be achieved in defocus search", "10.0", 0.0);

    //	pixel_size_refine_step = my_input->GetFloatFromUser("Pixel size refine step (A) (0.0 = no refinement)", "Step size used in the pixel size refinement", "0.001", 0.0);
    padding            = my_input->GetFloatFromUser("Padding factor", "Factor determining how much the input volume is padded to improve projections", "2.0", 1.0);
    wanted_mask_radius = my_input->GetFloatFromUser("Mask radius (A) (0.0 = no mask)", "Radius of a circular mask to be applied to the input particles during refinement", "0.0", 0.0);
    //	my_symmetry = my_input->GetSymmetryFromUser("Template symmetry", "The symmetry of the template reconstruction", "C1");

#ifdef _OPENMP
    max_threads = my_input->GetIntFromUser("Max. threads to use for calculation", "When threading, what is the max threads to run", "1", 1);
#else
    max_threads = 1;
#endif

    // TODO: add a minimum number of detections threshold to reject likely bad images.

    int      first_search_position           = -1;
    int      last_search_position            = -1;
    int      image_number_for_gui            = 0;
    int      number_of_jobs_per_image_in_gui = 0;
    float    threshold_for_result_plotting   = 0.0f;
    wxString filename_for_gui_result_image;

    delete my_input;

    //	my_current_job.Reset(42);
    my_current_job.ManualSetArguments("ttffffibfffffiiiiitftt",
                                      input_search_images.ToUTF8( ).data( ), // 0
                                      input_reconstruction.ToUTF8( ).data( ), // 1
                                      low_resolution_limit, // 2
                                      high_resolution_limit, // 3
                                      angular_range, // 4
                                      angular_step, // 5
                                      best_parameters_to_keep, // 6
                                      do_defocus_refinement, // 10
                                      defocus_search_range, // 7
                                      defocus_search_step, // 8
                                      padding, // 9
                                      wanted_mask_radius, // 11
                                      in_plane_angular_step, // 14
                                      first_search_position, // 15
                                      last_search_position, //  16
                                      image_number_for_gui, // 17
                                      number_of_jobs_per_image_in_gui, // 18
                                      max_threads, // 19
                                      directory_for_results_string.ToUTF8( ).data( ), // 20
                                      threshold_for_result_plotting, // 21
                                      filename_for_gui_result_image.ToUTF8( ).data( ), // 22
                                      input_star_filename.ToUTF8( ).data( )); // 23
}

// override the do calculation method which will be what is actually run..

bool GlobalSearchRefinementApp::DoCalculation( ) {

    StopWatch timer;
    StopWatch refine_timer;
    timer.start("Initialize");
    wxDateTime start_time = wxDateTime::Now( );

    input_search_images                    = my_current_job.arguments[0].ReturnStringArgument( );
    input_reconstruction                   = my_current_job.arguments[1].ReturnStringArgument( );
    low_resolution_limit                   = my_current_job.arguments[2].ReturnFloatArgument( );
    high_resolution_limit                  = my_current_job.arguments[3].ReturnFloatArgument( );
    angular_range                          = my_current_job.arguments[4].ReturnFloatArgument( );
    angular_step                           = my_current_job.arguments[5].ReturnFloatArgument( );
    best_parameters_to_keep                = my_current_job.arguments[6].ReturnIntegerArgument( );
    do_defocus_refinement                  = my_current_job.arguments[7].ReturnBoolArgument( );
    defocus_search_range                   = my_current_job.arguments[8].ReturnFloatArgument( );
    defocus_search_step                    = my_current_job.arguments[9].ReturnFloatArgument( );
    padding                                = my_current_job.arguments[10].ReturnFloatArgument( );
    wanted_mask_radius                     = my_current_job.arguments[11].ReturnFloatArgument( );
    in_plane_angular_step                  = my_current_job.arguments[12].ReturnFloatArgument( );
    int first_search_position              = my_current_job.arguments[13].ReturnIntegerArgument( );
    int last_search_position               = my_current_job.arguments[14].ReturnIntegerArgument( );
    int image_number_for_gui               = my_current_job.arguments[15].ReturnIntegerArgument( );
    int number_of_jobs_per_image_in_gui    = my_current_job.arguments[16].ReturnIntegerArgument( );
    max_threads                            = my_current_job.arguments[17].ReturnIntegerArgument( );
    wxString directory_for_results         = my_current_job.arguments[18].ReturnStringArgument( );
    float    threshold_for_result_plotting = my_current_job.arguments[19].ReturnFloatArgument( );
    wxString filename_for_gui_result_image = my_current_job.arguments[20].ReturnStringArgument( );
    input_star_filename                    = my_current_job.arguments[21].ReturnStringArgument( );

    if ( is_running_locally == false )
        max_threads = number_of_threads_requested_on_command_line; // OVERRIDE FOR THE GUI, AS IT HAS TO BE SET ON THE COMMAND LINE...

    int i, j;

    float outer_mask_radius;

    int   number_of_rotations;
    long  total_correlation_positions;
    long  current_correlation_position;
    long  pixel_counter;
    float sq_dist_x, sq_dist_y;
    long  address;

    int current_x;
    int current_y;

    int defocus_i;
    int size_i;
    int size_is = 0;

    AnglesAndShifts angles;

    ImageFile input_search_image_file;
    ImageFile input_reconstruction_file;

    Curve whitening_filter;
    Curve number_of_terms;

    Curve whitening_filter_vol;
    Curve number_of_terms_vol;

    input_search_image_file.OpenFile(input_search_images.ToStdString( ), false);
    input_reconstruction_file.OpenFile(input_reconstruction.ToStdString( ), false);

    Image input_image;
    Image input_reconstruction;

    Peak current_peak;
    Peak template_peak;

    float starting_score;
    bool  first_score;

    float best_phi_score;
    float best_theta_score;
    float best_psi_score;
    float best_defocus_score;

    float score_adjustment;
    //	float offset_warning_threshold = 10.0f;

    int   number_of_peaks_found = 0;
    int   peak_number;
    float mask_falloff     = 20.0;
    float min_peak_radius2 = powf(min_peak_radius, 2);

    // if ( (input_search_image_file.ReturnZSize( ) < 1) || (mip_input_file.ReturnZSize( ) < 1) || (scaled_mip_input_file.ReturnZSize( ) < 1) || (best_psi_input_file.ReturnZSize( ) < 1) || (best_theta_input_file.ReturnZSize( ) < 1) || (best_phi_input_file.ReturnZSize( ) < 1) ) {
    //     SendErrorAndCrash("Error: Input files do not contain selected result\n");
    // }

    //
    timer.lap("Initialize");

    timer.start("read in");
    input_image.ReadSlice(&input_search_image_file, 1);
    input_star_file.ReadFromcisTEMStarFile(input_star_filename);
    cisTEMParameterLine input_parameters_ = input_star_file.ReturnLine(0);
    float               pixel_size        = input_parameters_.pixel_size;
    // As with everywhere, we are assuming acubic volume.
    input_reconstruction.ReadSlices(&input_reconstruction_file, 1, input_reconstruction_file.ReturnNumberOfSlices( ));

    float binning_factor_refine;
    float original_pixel_size;
    float binned_pixel_size;
    int   original_x_size;
    int   binned_x_size;

    {
        // We are done with the reconstructed volume, so let's get rid of it (something wonky was going on with the destrcutor)

        ReconstructedVolume input_volume;
        input_volume.InitWithDimensions(input_reconstruction_file.ReturnXSize( ), input_reconstruction_file.ReturnYSize( ), input_reconstruction_file.ReturnZSize( ), pixel_size, my_symmetry);
        input_volume.density_map = &input_reconstruction;
        input_volume.PrepareForProjections(low_resolution_limit, high_resolution_limit, false, true, false);
        binning_factor_refine = input_volume.pixel_size / pixel_size;
        original_pixel_size   = pixel_size;
        binned_pixel_size     = input_volume.pixel_size;
        original_x_size       = input_reconstruction_file.ReturnXSize( );
        binned_x_size         = input_volume.density_map->logical_x_dimension;

        input_volume.density_map            = nullptr;
        input_volume.projection_initialized = false;
    }

    wxPrintf("\nBinning factor for refinement = %f, new pixel size = %f\n", binning_factor_refine, binned_pixel_size);

    timer.lap("read in");

    timer.start("setup 3d");
    // TODO: this should also include any scalling needed based on resolution, this can be added for efficiency, first use full size and just filter to limit resolution.
    int padded_size;
    std::cerr << "padding is " << padding << std::endl;
    if ( padding != 1.0f ) {
        float tmp_lower = ReturnClosestFactorizedLower(input_reconstruction.logical_x_dimension * padding, 5);
        float tmp_upper = ReturnClosestFactorizedUpper(input_reconstruction.logical_x_dimension * padding, 5);
        std::cerr << "factorized sizes are " << tmp_lower << " " << tmp_upper << std::endl;
        padded_size = abs(tmp_lower - input_reconstruction.logical_x_dimension * padding) < abs(tmp_upper - input_reconstruction.logical_x_dimension * padding) ? tmp_lower : tmp_upper;

        std::cerr << "padded size: " << padded_size << std::endl;
        input_reconstruction.Resize(padded_size, padded_size, padded_size, input_reconstruction.ReturnAverageOfRealValuesOnEdges( ));
    }
    else {
        padded_size = input_reconstruction.logical_x_dimension;
    }
    // These are handled in input_volume.prepareforprojections
    // input_reconstruction.ForwardFFT( );
    // input_reconstruction.ZeroCentralPixel( );
    // input_reconstruction.SwapRealSpaceQuadrants( );
    timer.lap("setup 3d");

    CTF input_ctf;

    // work out the filter to just whiten the image..
    // we will cound on "local" whitening when getting into using refin3d for the final refinement.
    timer.start("setup whitening");
    whitening_filter.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((input_image.logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));
    number_of_terms.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((input_image.logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));

    //
    whitening_filter_vol.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((input_reconstruction.logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));
    number_of_terms_vol.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((input_reconstruction.logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));
    timer.lap("setup whitening");
    wxDateTime my_time_out;
    wxDateTime my_time_in;

    // remove outliers
    // FIXME: fixed value should be in constants.h
    timer.start("outliers");
    input_image.ReplaceOutliersWithMean(5.0f);
    timer.lap("outliers");

    timer.start("2d prep");
    input_image.ForwardFFT( );

    input_image.ZeroCentralPixel( );
    input_image.Compute1DPowerSpectrumCurve(&whitening_filter, &number_of_terms);
    whitening_filter.SquareRoot( );
    whitening_filter.Reciprocal( );
    whitening_filter.MultiplyByConstant(1.0f / whitening_filter.ReturnMaximumValue( ));

    input_image.ApplyCurveFilter(&whitening_filter);
    input_image.ZeroCentralPixel( );
    input_image.DivideByConstant(sqrt(input_image.ReturnSumOfSquares( )));
    input_image.BackwardFFT( );
    timer.lap("2d prep");

    // I wonder if it makes sense to do outlier removal here as well..
    //	long *addresses = new long[input_image.logical_x_dimension * input_image.logical_y_dimension / 100];
    // count total searches (lazy)

    total_correlation_positions  = 0;
    current_correlation_position = 0;

    // if running locally, search over all of them

    current_peak.value = std::numeric_limits<float>::max( );
    timer.start("read star");
    // input_star_file.ReadFromcisTEMStarFile(input_star_filename);

    // TODO: would ensuring constness for the input parameters make any sense?
    output_star_file = input_star_file;
    timer.lap("read star");
    // To make the transition easier, first keep these unnecessary columns in the output file
    number_of_peaks_found      = input_star_file.ReturnNumberofLines( );
    int number_of_active_peaks = 0;
    for ( int i = 0; i < number_of_peaks_found; i++ ) {
        if ( input_star_file.ReturnImageIsActive(i) > 0 ) {
            number_of_active_peaks++;
        }
    }

    if ( is_running_locally ) {
        wxPrintf("\nRefining %i active positions of %i in the MIP.\n", number_of_active_peaks, number_of_peaks_found);

        wxPrintf("\nPerforming refinement...\n\n");
        //		my_progress = new ProgressBar(total_correlation_positions);
    }

    timer.start("alloc template results");
    ArrayOfTemplateMatchFoundPeakInfos all_peak_changes;
    ArrayOfTemplateMatchFoundPeakInfos all_peak_infos;

    TemplateMatchFoundPeakInfo temp_peak;
    all_peak_changes.Alloc(number_of_peaks_found);
    all_peak_changes.Add(temp_peak, number_of_peaks_found);

    all_peak_infos.Alloc(number_of_peaks_found);
    all_peak_infos.Add(temp_peak, number_of_peaks_found);
    timer.lap("alloc template results");

    if ( max_threads > number_of_peaks_found )
        max_threads = number_of_peaks_found;

#pragma omp parallel num_threads(max_threads) default(none) shared(timer, refine_timer, padded_size, std::cerr, number_of_peaks_found, input_image, mask_falloff, wanted_mask_radius, input_star_file, output_star_file, \
                                                                   defocus_search_range, angular_step, in_plane_angular_step, whitening_filter, input_reconstruction, min_peak_radius2,                                  \
                                                                   input_reconstruction_file, max_threads, low_resolution_limit, high_resolution_limit,                                                                  \
                                                                   all_peak_changes, all_peak_infos, original_pixel_size, binned_pixel_size, binning_factor_refine,                                                      \
                                                                   original_x_size, binned_x_size) private(current_peak, sq_dist_x, sq_dist_y, address,                                                                  \
                                                                                                           defocus_i, size_i,                                                                                            \
                                                                                                           best_defocus_score, best_phi_score, best_theta_score, best_psi_score,                                         \
                                                                                                           angles, template_peak, i, j, peak_number,                                                                     \
                                                                                                           first_score, starting_score, size_is, score_adjustment)
    {

        timer.start("alloc local imgs");
        Image windowed_particle_;
        Image projection_filter_;
        Image binned_particle_;

        //
        cisTEMParameterLine input_parameters_;

        CTF  ctf_;
        Peak best_peak_;

        float pixel_size_; // Set from the input parameters
        float mask_radius_ = wanted_mask_radius;

        // TODO: switch to me for gpu access, requires going from input_reconstruction (Image) to reconstructed volume
        // requires a particle object as well
        // ProjectionComparisonObjects comparison_object_;
        TemplateComparisonObject template_object;

        windowed_particle_.Allocate(original_x_size, original_x_size, 1, true);
        binned_particle_.Allocate(binned_x_size, binned_x_size, 1, false);
        projection_filter_.Allocate(binned_x_size, binned_x_size, 1, false);

        current_peak.value = std::numeric_limits<float>::max( );

        //	number_of_peaks_found = 0;

        template_object.input_reconstruction = &input_reconstruction;
        template_object.windowed_particle    = &binned_particle_;
        template_object.projection_filter    = &projection_filter_;
        template_object.angles               = &angles;
        timer.lap("alloc local imgs");
//	while (current_peak.value >= wanted_threshold)
#pragma omp for schedule(dynamic, 1)
        for ( peak_number = 0; peak_number < number_of_peaks_found; peak_number++ ) {

            // Grab a local copy of input parameters from the shared starfile
            input_parameters_ = input_star_file.ReturnLine(peak_number);
            if ( input_parameters_.image_is_active < 0 )
                continue;

            pixel_size_ = binned_pixel_size; //input_parameters_.pixel_size;

            // assume cube in determining the maximum radius.

            float maximum_mask_radius = (float(original_x_size) / 2.0f - 1.0f) * original_pixel_size;
            if ( mask_radius_ > maximum_mask_radius )
                mask_radius_ = maximum_mask_radius;
            // TODO: adjust pixel size here?

            timer.start("init ctf");
            // I don't think there is any significant overhead to intializing the CTF here, vs setting individual values
            // ctf_.Init(voltage_kV, spherical_aberration_mm, amplitude_contrast, defocus1, defocus2, defocus_angle, 0.0, 0.0, 0.0, pixel_size, deg_2_rad(phase_shift));
            ctf_.Init(input_parameters_.microscope_voltage_kv,
                      input_parameters_.microscope_spherical_aberration_mm,
                      input_parameters_.amplitude_contrast,
                      input_parameters_.defocus_1,
                      input_parameters_.defocus_2,
                      input_parameters_.defocus_angle,
                      0.0,
                      0.0,
                      0.0,
                      pixel_size_,
                      deg_2_rad(input_parameters_.phase_shift)); // NOTE: no beam tilt here. If you want to adapt this to high resolution, you will need to add beam tilt
            timer.lap("init ctf");
            timer.start("window particle");

            // Make sure the windowed particle is set to real space
            windowed_particle_.is_in_real_space = true;
            // FIXME: confirm how the fractional shifts should look, but for now it prob doesn't matter too much.
            input_image.ClipInto(&windowed_particle_, 0.0f, false, 1.0f,
                                 myroundint(input_parameters_.x_shift / original_pixel_size - input_image.physical_address_of_box_center_x),
                                 myroundint(input_parameters_.y_shift / original_pixel_size - input_image.physical_address_of_box_center_y), 0);
            timer.lap("window particle");

            timer.start("filter particle");
            if ( mask_radius_ > 0.0f )
                windowed_particle_.CosineMask(mask_radius_ / original_pixel_size, mask_falloff / original_pixel_size);
            windowed_particle_.ForwardFFT( );
            windowed_particle_.SwapRealSpaceQuadrants( );
            timer.lap("filter particle");

            timer.start("resize particle");
            // Make sure the binned particle is set to fourier space
            binned_particle_.is_in_real_space = false;
            windowed_particle_.ClipInto(&binned_particle_, 0.0f);
            timer.lap("resize particle");

            // TODO: what is pixel_size_factor? If not immediately switching to Projection compairson object, then add a Reset() method
            template_object.pixel_size_factor = 1.0f;
            first_score                       = false;

            //		number_of_peaks_found++;

            float initial_phi      = input_parameters_.phi;
            float initial_theta    = input_parameters_.theta;
            float initial_psi      = input_parameters_.psi;
            float initial_defocus1 = input_parameters_.defocus_1;
            float initial_defocus2 = input_parameters_.defocus_2;

            float best_phi   = initial_phi;
            float best_theta = initial_theta;
            float best_psi   = initial_psi;

            // Get a starting score for the input position
            // TODO: should the AnglesAndShifts object have the residual shifts from x/y
            angles.Init(initial_phi, initial_theta, initial_psi, 0.0, 0.0);
            // NOTE: Astigmatism angle is fixed, but could be fit and smoothed over the image (same as defocus refinement)
            ctf_.SetDefocus(initial_defocus1 / pixel_size_, initial_defocus2 / pixel_size_, deg_2_rad(input_parameters_.defocus_angle));
            timer.start("prj filter");
            projection_filter_.CalculateCTFImage(ctf_);
            projection_filter_.ApplyCurveFilter(&whitening_filter);
            if ( high_resolution_limit > 0.0 )
                projection_filter_.CosineMask(pixel_size_ / high_resolution_limit, pixel_size_ / 100.0);
            if ( low_resolution_limit > 0.0 )
                projection_filter_.CosineMask(pixel_size_ / low_resolution_limit, pixel_size_ / 100.0, true);
            timer.lap("prj filter");

            timer.start("zero template object");
            template_object.Zero(mask_radius_ / pixel_size_);
            timer.lap("zero template object");

            // The score change is currently holding the score adjusted based on local noise FIXME
            float score_adjustment = input_parameters_.score_change / input_parameters_.score;
            timer.start("score 1");
            best_peak_              = TemplateScore(&template_object, refine_timer);
            input_parameters_.score = best_peak_.value;
            timer.lap("score 1");

            if ( do_defocus_refinement && defocus_search_range == 0 ) {
                // FIXME hack to get a score with a new template but no refinement
                input_parameters_.score_change                    = input_parameters_.score * score_adjustment;
                output_star_file.all_parameters.Item(peak_number) = input_parameters_;
                continue;
            }

#ifdef PRINT_GLOBAL_SEARCH_REFINEMENT_EXTRA_INFO
#pragma omp critical
            {
                wxPrintf("\nRefining peak %i at x, y =  %6i, %6i with starting score: %3.3e\n", peak_number + 1, myroundint(current_peak.x), myroundint(current_peak.y), best_peak_.value);
            }
#endif

            // Do a brute force search over the defocus range

            // TODO: right now, defocus is also searched in the refinement loop so this is redundant
            int best_iDefocus = 0;
            if ( do_defocus_refinement ) {
#ifdef PRINT_GLOBAL_SEARCH_REFINEMENT_EXTRA_INFO
#pragma omp critical
                {
                    wxPrintf("Refining defocus range: %f in step size %f\n", defocus_search_range, defocus_search_step);
                }
#endif
                for ( int iDefocus = -myroundint(float(defocus_search_range) / float(defocus_search_step)); iDefocus <= myroundint(float(defocus_search_range) / float(defocus_search_step)); iDefocus++ ) {

                    timer.start("def_refine prj filter");
                    ctf_.SetDefocus((initial_defocus1 + iDefocus * defocus_search_step) / pixel_size_, (initial_defocus2 + iDefocus * defocus_search_step) / pixel_size_, deg_2_rad(input_parameters_.defocus_angle));
                    projection_filter_.CalculateCTFImage(ctf_);
                    // FIXME: we shoulid be applying the image curve to the volume projection, not this.

                    projection_filter_.ApplyCurveFilter(&whitening_filter);
                    if ( high_resolution_limit > 0.0 )
                        projection_filter_.CosineMask(pixel_size_ / high_resolution_limit, pixel_size_ / 100.0);
                    if ( low_resolution_limit > 0.0 )
                        projection_filter_.CosineMask(pixel_size_ / low_resolution_limit, pixel_size_ / 100.0, true);

                    timer.lap("def_refine prj filter");
                    timer.start("score def_refine");
                    template_peak = TemplateScore(&template_object, refine_timer);
                    timer.lap("score def_refine");

                    // wxPrintf("For defocus1 %f, offset %i, score is %f\n", initial_defocus1 + iDefocus * defocus_search_step, iDefocus, template_peak.value);

                    if ( template_peak.value > best_peak_.value ) {
                        best_peak_    = template_peak;
                        best_iDefocus = iDefocus;
                    }
                }
#ifdef PRINT_GLOBAL_SEARCH_REFINEMENT_EXTRA_INFO
#pragma omp critical
                {
                    wxPrintf("Best defocus was: %i \n", best_iDefocus);
                }
#endif
            }
            else {

                int n_phi_steps   = 0;
                int n_theta_steps = 0;
                int n_psi_steps   = 0;
                // defocus_i = 0;

                // TODO: This would be a place to constrain the fit defocus across the image before moving on, but this would require joining all threads.
                // For now, I am just removing the outermost loop which was over the defocus range. (def, euler sphere, in plane)
                // for ( ll = 0; ll < 2; ll = -2 * ll + 1 ) {
                //     if ( (ll != 0) && (! do_defocus_refinement) )
                //         break;

                do { // while ( best_peak_.value > best_defocus_score );
                    best_defocus_score = best_peak_.value;
                    // if ( do_defocus_refinement )
                    //     defocus_i += 1;
                    // make the projection filter, which will be CTF * whitening filter
                    timer.start("ang_refine prj filter");
                    ctf_.SetDefocus((initial_defocus1 + best_iDefocus * defocus_search_step) / pixel_size_, (initial_defocus2 + best_iDefocus * defocus_search_step) / pixel_size_, deg_2_rad(input_parameters_.defocus_angle));
                    projection_filter_.CalculateCTFImage(ctf_);
                    projection_filter_.ApplyCurveFilter(&whitening_filter);
                    if ( high_resolution_limit > 0.0 )
                        projection_filter_.CosineMask(pixel_size_ / high_resolution_limit, pixel_size_ / 100.0);
                    if ( low_resolution_limit > 0.0 )
                        projection_filter_.CosineMask(pixel_size_ / low_resolution_limit, pixel_size_ / 100.0, true);
                    timer.lap("ang_refine prj filter");

                    for ( int i_phi = 0; i_phi < 2; i_phi = -2 * i_phi + 1 ) {
                        do { // while ( best_peak_.value > best_phi_score );
                            best_phi_score = best_peak_.value;
                            n_phi_steps += i_phi;
                            for ( int i_theta = 0; i_theta < 2; i_theta = -2 * i_theta + 1 ) {
                                do { // while ( best_peak_.value > best_theta_score );
                                    best_theta_score = best_peak_.value;
                                    n_theta_steps += i_theta;
                                    for ( int i_psi = 0; i_psi < 2; i_psi = -2 * i_psi + 1 ) {
                                        do { // while ( best_peak_.value > best_psi_score );
                                            best_psi_score = best_peak_.value;
                                            n_psi_steps += i_psi;
                                            // FIXME: In the First loop we should have our starting score = best score, which should break out of the while loop (which is a waste)
                                            timer.start("score ang_refine");
                                            angles.Init(initial_phi + n_phi_steps * angular_step, initial_theta + n_theta_steps * angular_step, initial_psi + n_psi_steps * in_plane_angular_step, 0.0, 0.0);
                                            template_peak = TemplateScore(&template_object, refine_timer);
                                            timer.lap("score ang_refine");
                                            // std::cerr << "n_psi and i_psi: " << n_psi_steps << " " << i_psi << std::endl;
                                            // std::cerr << "testing angles " << angles.ReturnPhiAngle( ) << " " << angles.ReturnThetaAngle( ) << " " << angles.ReturnPsiAngle( ) << " score " << template_peak.value << std::endl;

                                            if ( template_peak.value > best_peak_.value ) {
                                                best_peak_ = template_peak;
                                                best_phi   = initial_phi + n_phi_steps * angular_step;
                                                best_theta = initial_theta + n_theta_steps * angular_step;
                                                best_psi   = initial_psi + n_psi_steps * in_plane_angular_step;
                                            }

                                        } while ( best_peak_.value > best_psi_score );
                                        n_psi_steps -= i_psi;
                                    }
                                } while ( best_peak_.value > best_theta_score );
                                n_theta_steps -= i_theta;
                            }
                        } while ( best_peak_.value > best_phi_score );
                        n_phi_steps -= i_phi;
                    }
                } while ( best_peak_.value > best_defocus_score );
                // if ( do_defocus_refinement )
                //     defocus_i -= ll;
                // }
            }

            timer.start("adjust scores");
            float tmp_to_print = best_peak_.value;
            template_object.AdjustScoreByVarianceEstimate(tmp_to_print);
            timer.lap("adjust scores");
            // wxPrintf("Estimated sigma %f, score %f, re-adjusted score %f\n", template_object.GetSigma( ), best_peak_.value, tmp_to_print);

            timer.start("update output star");
            output_star_file.all_parameters.Item(peak_number).x_shift = best_peak_.x * binned_pixel_size + input_parameters_.x_shift;
            output_star_file.all_parameters.Item(peak_number).y_shift = best_peak_.y * binned_pixel_size + input_parameters_.y_shift;

            output_star_file.all_parameters.Item(peak_number).phi          = best_phi;
            output_star_file.all_parameters.Item(peak_number).theta        = best_theta;
            output_star_file.all_parameters.Item(peak_number).psi          = best_psi;
            output_star_file.all_parameters.Item(peak_number).defocus_1    = (initial_defocus1 + best_iDefocus * defocus_search_step);
            output_star_file.all_parameters.Item(peak_number).defocus_2    = (initial_defocus2 + best_iDefocus * defocus_search_step);
            output_star_file.all_parameters.Item(peak_number).score_change = tmp_to_print;
            output_star_file.all_parameters.Item(peak_number).score        = best_peak_.value;
            // output_star_file.all_parameters.Item(peak_number).score_change = best_peak_.value - input_parameters_.score;
            timer.lap("update output star");

        } // end omp for loop over peaks
    } // end omp section

    // The filenames in particular make the star file harder to read and for auto_functionality are not necessary.
    output_star_file.parameters_to_write.stack_filename          = false;
    output_star_file.parameters_to_write.original_image_filename = false;
    output_star_file.parameters_to_write.reference_3d_filename   = false;
    output_star_file.parameters_to_write.best_2d_class           = false;

    timer.start("write output star");
    output_star_file.WriteTocisTEMStarFile(output_star_filename.ToStdString( ));
    timer.lap("write output star");

    timer.print_times( );
    refine_timer.print_times( );
    if ( is_running_locally == true ) {
        wxPrintf("\nRefine Template: Normal termination\n");
        wxDateTime finish_time = wxDateTime::Now( );
        wxPrintf("Total Run Time : %s\n\n", finish_time.Subtract(start_time).Format("%Hh:%Mm:%Ss"));
    }
    else // find peaks, write and write a result image, then send result..
    {

        MyAssertTrue(false, "RefineTemplate: This should not be called in a distributed environment.");
        for ( int counter = 0; counter < all_peak_infos.GetCount( ); counter++ ) {

            // FIXME: paramters

            if ( all_peak_infos[counter].peak_height < threshold_for_result_plotting ) {
                // all_peak_infos[counter].x_pos = (all_peak_infos[counter].x_pos + best_scaled_mip.physical_address_of_box_center_x) * pixel_size;
                // all_peak_infos[counter].y_pos = (all_peak_infos[counter].y_pos + best_scaled_mip.physical_address_of_box_center_y) * pixel_size;
                continue;
            }

            // ok we have peak..

            number_of_peaks_found++;

            // get angles and mask out the local area so it won't be picked again..

            // scale the shifts by the pixel size..

            // all_peak_infos[counter].x_pos = (all_peak_infos[counter].x_pos + best_scaled_mip.physical_address_of_box_center_x) * pixel_size;
            // all_peak_infos[counter].y_pos = (all_peak_infos[counter].y_pos + best_scaled_mip.physical_address_of_box_center_y) * pixel_size;
        }

        // tell the gui that this result is available...

        SendTemplateMatchingResultToSocket(controller_socket, image_number_for_gui, threshold_for_result_plotting, all_peak_infos, all_peak_changes);
    }

    return true;
}

#ifdef PRINT_GLOBAL_SEARCH_REFINEMENT_EXTRA_INFO
#undef PRINT_GLOBAL_SEARCH_REFINEMENT_EXTRA_INFO
#endif