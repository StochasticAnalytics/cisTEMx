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

#define PRINT_GLOBAL_SEARCH_REFINEMENT_EXTRA_INFO

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
    float    xy_change_threshold        = 10.0f;
    bool     exclude_above_xy_threshold = false;
    int      result_number              = 1;

    int max_threads;

  private:
};

// TODO: replace this with the ProjectionComparisonObject class and setup for GPU
class TemplateComparisonObject {
  public:
    Image* input_reconstruction;
    Image* windowed_particle;
    Image* projection_filter;

    AnglesAndShifts* angles;
    float            pixel_size_factor;

    //	int							slice = 1;
};

// This is the function which will be minimized
Peak TemplateScore(void* scoring_parameters) {
    TemplateComparisonObject* comparison_object = reinterpret_cast<TemplateComparisonObject*>(scoring_parameters);
    Image                     current_projection;
    //	Peak box_peak;

    // FIXME: ALlocating this on every loop is bonkers
    current_projection.Allocate(comparison_object->projection_filter->logical_x_dimension, comparison_object->projection_filter->logical_x_dimension, false);
    if ( comparison_object->input_reconstruction->logical_x_dimension != current_projection.logical_x_dimension ) {
        Image padded_projection;
        padded_projection.Allocate(comparison_object->input_reconstruction->logical_x_dimension, comparison_object->input_reconstruction->logical_x_dimension, false);
        comparison_object->input_reconstruction->ExtractSlice(padded_projection, *comparison_object->angles, 1.0f, false);
        padded_projection.SwapRealSpaceQuadrants( );
        padded_projection.BackwardFFT( );
        padded_projection.ChangePixelSize(&current_projection, comparison_object->pixel_size_factor, 0.001f, true);
    }
    else {
        comparison_object->input_reconstruction->ExtractSlice(current_projection, *comparison_object->angles, 1.0f, false);
        current_projection.SwapRealSpaceQuadrants( );
        current_projection.BackwardFFT( );
        current_projection.ChangePixelSize(&current_projection, comparison_object->pixel_size_factor, 0.001f, true);
    }

    current_projection.MultiplyPixelWise(*comparison_object->projection_filter);

    current_projection.ZeroCentralPixel( );
    current_projection.DivideByConstant(sqrtf(current_projection.ReturnSumOfSquares( )));
#ifdef MKL
    // Use the MKL
    vmcMulByConj(current_projection.real_memory_allocated / 2, reinterpret_cast<MKL_Complex8*>(comparison_object->windowed_particle->complex_values), reinterpret_cast<MKL_Complex8*>(current_projection.complex_values), reinterpret_cast<MKL_Complex8*>(current_projection.complex_values), VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
#else
    for ( long pixel_counter = 0; pixel_counter < current_projection.real_memory_allocated / 2; pixel_counter++ ) {
        current_projection.complex_values[pixel_counter] = std::conj(current_projection.complex_values[pixel_counter]) * comparison_object->windowed_particle->complex_values[pixel_counter];
    }
#endif
    current_projection.BackwardFFT( );
    //	wxPrintf("ping");

    return current_projection.FindPeakWithIntegerCoordinates( );
}

IMPLEMENT_APP(GlobalSearchRefinementApp)

// override the DoInteractiveUserInput

void GlobalSearchRefinementApp::DoInteractiveUserInput( ) {

    UserInput* my_input = new UserInput("RefineTemplate", 1.00);

    // This block is the same as in globale_search.cpp
    input_search_images  = my_input->GetFilenameFromUser("Input images to be searched", "The input image stack, containing the images that should be searched", "image_stack.mrc", true);
    input_star_filename  = my_input->GetFilenameFromUser("Input star file, ignored if prev false", "The input star file, containing the images that should be searched", "input_particles.star", true);
    input_reconstruction = my_input->GetFilenameFromUser("Input template reconstruction", "The 3D reconstruction from which projections are calculated", "reconstruction.mrc", true);
    wxFileName directory_for_results(my_input->GetFilenameFromUser("Output directory for results, subdirectory using image name will be created.", "", "./", false));
    MyDebugAssertFalse(directory_for_results.HasExt( ), "Output directory should not have an extension");
    if ( ! directory_for_results.DirExists( ) ) {
        MyDebugPrint("Output directory does not exist, creating it");
        directory_for_results.Mkdir(0777, wxPATH_MKDIR_FULL);
    }
    wxFileName input_search_image_file_name_full(input_search_images);
    wxString   directory_for_results_string = directory_for_results.GetFullPath( );

    // For now, we'll just use the scaled_mip as supplied, but may want to be able to adjust the scalling later
    // Then instead of applying the smoothing in global_search, we have the raw output to save here.
    // correlation_avg_input_filename = input_search_image_file_name_full.GetName( ) + "_avg.mrc";
    // correlation_std_input_filename = input_search_image_file_name_full.GetName( ) + "_std.mrc";

    wanted_threshold = my_input->GetFloatFromUser("Peak threshold", "Peaks over this size will be taken", "7.5", 0.0);
    min_peak_radius  = my_input->GetFloatFromUser("Min peak radius (px.)", "Essentially the minimum closeness for peaks", "10.0", 0.0);

    //	low_resolution_limit = my_input->GetFloatFromUser("Low resolution limit (A)", "Low resolution limit of the data used for alignment in Angstroms", "300.0", 0.0);
    //	high_resolution_limit = my_input->GetFloatFromUser("High resolution limit (A)", "High resolution limit of the data used for alignment in Angstroms", "8.0", 0.0);
    //	angular_range = my_input->GetFloatFromUser("Angular refinement range", "AAngular range to refine", "2.0", 0.1);
    angular_step          = my_input->GetFloatFromUser("Out of plane angular step", "Angular step size for global grid search", "0.2", 0.00);
    in_plane_angular_step = my_input->GetFloatFromUser("In plane angular step", "Angular step size for in-plane rotations during the search", "0.1", 0.00);
    //	best_parameters_to_keep = my_input->GetIntFromUser("Number of top hits to refine", "The number of best global search orientations to refine locally", "20", 1);
    defocus_search_range  = my_input->GetFloatFromUser("Defocus search range (A) (0.0 = no search)", "Search range (-value ... + value) around current defocus", "200.0", 0.0);
    defocus_search_step   = my_input->GetFloatFromUser("Desired defocus accuracy (A)", "Accuracy to be achieved in defocus search", "10.0", 0.0);
    do_defocus_refinement = my_input->GetYesNoFromUser("Refine defocus", "Should the particle defocus be refined?", "No");

    //	pixel_size_refine_step = my_input->GetFloatFromUser("Pixel size refine step (A) (0.0 = no refinement)", "Step size used in the pixel size refinement", "0.001", 0.0);
    padding            = my_input->GetFloatFromUser("Padding factor", "Factor determining how much the input volume is padded to improve projections", "2.0", 1.0);
    wanted_mask_radius = my_input->GetFloatFromUser("Mask radius (A) (0.0 = no mask)", "Radius of a circular mask to be applied to the input particles during refinement", "0.0", 0.0);
    //	my_symmetry = my_input->GetSymmetryFromUser("Template symmetry", "The symmetry of the template reconstruction", "C1");
    xy_change_threshold        = my_input->GetFloatFromUser("Moved peak warning (A)", "Threshold for displaying warning of peak location changes during refinement", "10.0", 0.0);
    exclude_above_xy_threshold = my_input->GetYesNoFromUser("Exclude moving peaks", "Should the peaks that move more than the threshold be excluded from the output MIPs?", "No");
    result_number              = my_input->GetIntFromUser("Result number to refine", "If input files contain results from several searches, which one should be refined?", "1", 1);

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
    my_current_job.ManualSetArguments("ttffffifffbffffbtfiiiiiitftbt",
                                      input_search_images.ToUTF8( ).data( ), // 0
                                      input_reconstruction.ToUTF8( ).data( ), // 1
                                      low_resolution_limit, // 2
                                      high_resolution_limit, // 3
                                      angular_range, // 4
                                      angular_step, // 5
                                      best_parameters_to_keep, // 6
                                      defocus_search_range, // 7
                                      defocus_search_step, // 8
                                      padding, // 9
                                      do_defocus_refinement, // 10
                                      wanted_mask_radius, // 11
                                      wanted_threshold, // 12
                                      min_peak_radius, // 13
                                      xy_change_threshold, // 14
                                      exclude_above_xy_threshold, // 15
                                      my_symmetry.ToUTF8( ).data( ), // 16
                                      in_plane_angular_step, // 17
                                      first_search_position, // 18
                                      last_search_position, // 19
                                      image_number_for_gui, // 20
                                      number_of_jobs_per_image_in_gui, // 21
                                      result_number, // 22
                                      max_threads, // 23
                                      directory_for_results_string.ToUTF8( ).data( ), // 24
                                      threshold_for_result_plotting, // 25
                                      filename_for_gui_result_image.ToUTF8( ).data( ), // 26
                                      input_star_filename.ToUTF8( ).data( )); // 27
}

// override the do calculation method which will be what is actually run..

bool GlobalSearchRefinementApp::DoCalculation( ) {
    wxDateTime start_time = wxDateTime::Now( );

    input_search_images                    = my_current_job.arguments[0].ReturnStringArgument( );
    input_search_images                    = my_current_job.arguments[1].ReturnStringArgument( );
    low_resolution_limit                   = my_current_job.arguments[2].ReturnFloatArgument( );
    high_resolution_limit                  = my_current_job.arguments[3].ReturnFloatArgument( );
    angular_range                          = my_current_job.arguments[4].ReturnFloatArgument( );
    angular_step                           = my_current_job.arguments[5].ReturnFloatArgument( );
    best_parameters_to_keep                = my_current_job.arguments[6].ReturnIntegerArgument( );
    defocus_search_range                   = my_current_job.arguments[7].ReturnFloatArgument( );
    defocus_search_step                    = my_current_job.arguments[8].ReturnFloatArgument( );
    padding                                = my_current_job.arguments[9].ReturnFloatArgument( );
    do_defocus_refinement                  = my_current_job.arguments[10].ReturnBoolArgument( );
    wanted_mask_radius                     = my_current_job.arguments[11].ReturnFloatArgument( );
    wanted_threshold                       = my_current_job.arguments[12].ReturnFloatArgument( );
    min_peak_radius                        = my_current_job.arguments[13].ReturnFloatArgument( );
    xy_change_threshold                    = my_current_job.arguments[14].ReturnFloatArgument( );
    exclude_above_xy_threshold             = my_current_job.arguments[15].ReturnBoolArgument( );
    wxString my_symmetry                   = my_current_job.arguments[16].ReturnStringArgument( );
    in_plane_angular_step                  = my_current_job.arguments[17].ReturnFloatArgument( );
    int first_search_position              = my_current_job.arguments[18].ReturnIntegerArgument( );
    int last_search_position               = my_current_job.arguments[19].ReturnIntegerArgument( );
    int image_number_for_gui               = my_current_job.arguments[20].ReturnIntegerArgument( );
    int number_of_jobs_per_image_in_gui    = my_current_job.arguments[21].ReturnIntegerArgument( );
    result_number                          = my_current_job.arguments[22].ReturnIntegerArgument( );
    max_threads                            = my_current_job.arguments[23].ReturnIntegerArgument( );
    wxString directory_for_results         = my_current_job.arguments[24].ReturnStringArgument( );
    float    threshold_for_result_plotting = my_current_job.arguments[25].ReturnFloatArgument( );
    wxString filename_for_gui_result_image = my_current_job.arguments[26].ReturnStringArgument( );
    input_star_filename                    = my_current_job.arguments[27].ReturnStringArgument( );

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

    input_search_image_file.OpenFile(input_search_images.ToStdString( ), false);
    input_reconstruction_file.OpenFile(input_search_images.ToStdString( ), false);

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

    float defocus_step;
    float score_adjustment;
    //	float offset_warning_threshold = 10.0f;

    int   number_of_peaks_found = 0;
    int   peak_number;
    float mask_falloff     = 20.0;
    float min_peak_radius2 = powf(min_peak_radius, 2);

    // if ( (input_search_image_file.ReturnZSize( ) < result_number) || (mip_input_file.ReturnZSize( ) < result_number) || (scaled_mip_input_file.ReturnZSize( ) < result_number) || (best_psi_input_file.ReturnZSize( ) < result_number) || (best_theta_input_file.ReturnZSize( ) < result_number) || (best_phi_input_file.ReturnZSize( ) < result_number) ) {
    //     SendErrorAndCrash("Error: Input files do not contain selected result\n");
    // }

    //

    input_image.ReadSlice(&input_search_image_file, result_number);

    input_reconstruction.ReadSlices(&input_reconstruction_file, 1, input_reconstruction_file.ReturnNumberOfSlices( ));
    // TODO: this should also include any scalling needed based on resolution, this can be added for efficiency, first use full size and just filter to limit resolution.
    if ( padding != 1.0f ) {
        input_reconstruction.Resize(input_reconstruction.logical_x_dimension * padding, input_reconstruction.logical_y_dimension * padding, input_reconstruction.logical_z_dimension * padding, input_reconstruction.ReturnAverageOfRealValuesOnEdges( ));
    }
    input_reconstruction.ForwardFFT( );
    input_reconstruction.ZeroCentralPixel( );
    input_reconstruction.SwapRealSpaceQuadrants( );

    CTF input_ctf;

    // work out the filter to just whiten the image..
    // we will cound on "local" whitening when getting into using refin3d for the final refinement.

    whitening_filter.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((input_image.logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));
    number_of_terms.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((input_image.logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));

    wxDateTime my_time_out;
    wxDateTime my_time_in;

    // remove outliers
    // FIXME: fixed value should be in constants.h
    input_image.ReplaceOutliersWithMean(5.0f);
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

    // I wonder if it makes sense to do outlier removal here as well..
    //	long *addresses = new long[input_image.logical_x_dimension * input_image.logical_y_dimension / 100];
    // count total searches (lazy)

    total_correlation_positions  = 0;
    current_correlation_position = 0;

    // if running locally, search over all of them

    current_peak.value = std::numeric_limits<float>::max( );

    input_star_file.ReadFromcisTEMStarFile(input_star_filename);

    // TODO: would ensuring constness for the input parameters make any sense?
    output_star_file = input_star_file;

    // To make the transition easier, first keep these unnecessary columns in the output file
    number_of_peaks_found = input_star_file.ReturnNumberofLines( );

    if ( is_running_locally ) {
        wxPrintf("\nRefining %i positions in the MIP.\n", number_of_peaks_found);

        wxPrintf("\nPerforming refinement...\n\n");
        //		my_progress = new ProgressBar(total_correlation_positions);
    }

    ArrayOfTemplateMatchFoundPeakInfos all_peak_changes;
    ArrayOfTemplateMatchFoundPeakInfos all_peak_infos;

    TemplateMatchFoundPeakInfo temp_peak;
    all_peak_changes.Alloc(number_of_peaks_found);
    all_peak_changes.Add(temp_peak, number_of_peaks_found);

    all_peak_infos.Alloc(number_of_peaks_found);
    all_peak_infos.Add(temp_peak, number_of_peaks_found);

    if ( max_threads > number_of_peaks_found )
        max_threads = number_of_peaks_found;

#pragma omp parallel num_threads(max_threads) default(none) shared(number_of_peaks_found, input_image, mask_falloff, wanted_mask_radius, input_star_file, output_star_file,             \
                                                                   defocus_search_range, angular_step, in_plane_angular_step, whitening_filter, input_reconstruction, min_peak_radius2, \
                                                                   input_reconstruction_file, max_threads, defocus_step, xy_change_threshold, exclude_above_xy_threshold,               \
                                                                   all_peak_changes, all_peak_infos) private(current_peak, sq_dist_x, sq_dist_y, address,                               \
                                                                                                             defocus_i, size_i,                                                         \
                                                                                                             best_defocus_score, best_phi_score, best_theta_score, best_psi_score,      \
                                                                                                             angles, template_peak, i, j, peak_number,                                  \
                                                                                                             first_score, starting_score, size_is, score_adjustment)
    {

        Image windowed_particle_;
        Image projection_filter_;

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

        windowed_particle_.Allocate(input_reconstruction_file.ReturnXSize( ), input_reconstruction_file.ReturnXSize( ), true);
        projection_filter_.Allocate(input_reconstruction_file.ReturnXSize( ), input_reconstruction_file.ReturnXSize( ), false);

        current_peak.value = std::numeric_limits<float>::max( );

        //	number_of_peaks_found = 0;

        template_object.input_reconstruction = &input_reconstruction;
        template_object.windowed_particle    = &windowed_particle_;
        template_object.projection_filter    = &projection_filter_;
        template_object.angles               = &angles;

//	while (current_peak.value >= wanted_threshold)
#pragma omp for schedule(dynamic, 1)
        for ( peak_number = 0; peak_number < number_of_peaks_found; peak_number++ ) {

            // Grab a local copy of input parameters from the shared starfile
            input_parameters_ = input_star_file.ReturnLine(peak_number);

            pixel_size_ = input_parameters_.pixel_size;

            // assume cube in determining the maximum radius.

            float maximum_mask_radius = (float(input_reconstruction_file.ReturnXSize( )) / 2.0f - 1.0f) * pixel_size_;
            if ( mask_radius_ > maximum_mask_radius )
                mask_radius_ = maximum_mask_radius;
            // TODO: adjust pixel size here?

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
                      pixel_size_, // TODO: if we are downsampling, this pixel size is wrong
                      deg_2_rad(input_parameters_.phase_shift)); // NOTE: no beam tilt here. If you want to adapt this to high resolution, you will need to add beam tilt

            // FIXME: confirm how the fractional shifts should look, but for now it prob doesn't matter too much.
            input_image.ClipInto(&windowed_particle_, 0.0f, false, 1.0f, myroundint(input_parameters_.x_shift), myroundint(input_parameters_.y_shift), 0);
            if ( mask_radius_ > 0.0f )
                windowed_particle_.CosineMask(mask_radius_ / pixel_size_, mask_falloff / pixel_size_);
            windowed_particle_.ForwardFFT( );
            windowed_particle_.SwapRealSpaceQuadrants( );

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
            projection_filter_.CalculateCTFImage(ctf_);
            projection_filter_.ApplyCurveFilter(&whitening_filter);
            best_peak_ = TemplateScore(&template_object);

#ifdef PRINT_GLOBAL_SEARCH_REFINEMENT_EXTRA_INFO
#pragma omp critical
            {
                wxPrintf("\nRefining peak %i at x, y =  %6i, %6i with starting score: %3.3f\n", peak_number + 1, myroundint(current_peak.x), myroundint(current_peak.y), best_peak_.value);
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
                for ( int iDefocus = -myroundint(float(defocus_search_range) / float(defocus_step)); iDefocus <= myroundint(float(defocus_search_range) / float(defocus_step)); iDefocus++ ) {

                    ctf_.SetDefocus((initial_defocus1 + iDefocus * defocus_step) / pixel_size_, (initial_defocus2 + iDefocus * defocus_step) / pixel_size_, deg_2_rad(input_parameters_.defocus_angle));
                    projection_filter_.CalculateCTFImage(ctf_);
                    projection_filter_.ApplyCurveFilter(&whitening_filter);
                    template_peak = TemplateScore(&template_object);

                    if ( template_peak.value > best_peak_.value ) {
                        best_peak_    = template_peak;
                        best_iDefocus = iDefocus;
                    }
                }
            }

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
                ctf_.SetDefocus((initial_defocus1 + best_iDefocus * defocus_step) / pixel_size_, (initial_defocus2 + best_iDefocus * defocus_step) / pixel_size_, deg_2_rad(input_parameters_.defocus_angle));
                projection_filter_.CalculateCTFImage(ctf_);
                projection_filter_.ApplyCurveFilter(&whitening_filter);

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
                                        angles.Init(initial_phi + n_phi_steps * angular_step, initial_theta + n_theta_steps * angular_step, initial_psi + n_psi_steps * in_plane_angular_step, 0.0, 0.0);
                                        template_peak = TemplateScore(&template_object);
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

            all_peak_changes[peak_number].x_pos = best_peak_.x * pixel_size_;
            all_peak_changes[peak_number].y_pos = best_peak_.y * pixel_size_;

            // all_peak_infos[peak_number].x_pos = found_peaks[peak_number].x + best_peak_.x; // NOT SCALING BY PIXEL SIZE - DO AFTER MAKING RESULT IMAGE
            // all_peak_infos[peak_number].y_pos = found_peaks[peak_number].y + best_peak_.y; // NOT SCALING BY PIXEL SIZE - DO AFTER MAKING RESULT IMAGE
        } // end omp for loop
    } // end omp section

    if ( is_running_locally == true ) {
        wxPrintf("\nRefine Template: Normal termination\n");
        wxDateTime finish_time = wxDateTime::Now( );
        wxPrintf("Total Run Time : %s\n\n", finish_time.Subtract(start_time).Format("%Hh:%Mm:%Ss"));
    }
    else // find peaks, write and write a result image, then send result..
    {

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

#undef PRINT_GLOBAL_SEARCH_REFINEMENT_EXTRA_INFO