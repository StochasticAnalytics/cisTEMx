#include "../../core/core_headers.h"

#include "../refine3d/refine3d_defines.h"
#include "../refine3d/ProjectionComparisonObjects.h"

class GlobalSearchRefinementApp : public MyApp {
  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

    wxString input_search_images;
    wxString input_reconstruction;

    wxString mip_input_filename;
    wxString scaled_mip_input_filename;
    wxString best_psi_input_filename;
    wxString best_theta_input_filename;
    wxString best_phi_input_filename;

    cisTEMParameterLine input_parameters;
    cisTEMParameters    input_star_file;
    cisTEMParameterLine output_parameters;
    cisTEMParameters    output_star_file;
    wxString            input_star_filename;
    wxString            output_star_filename;

    float pixel_size              = 1.0f;
    float voltage_kV              = 300.0f;
    float spherical_aberration_mm = 2.7f;
    float amplitude_contrast      = 0.07f;
    float defocus1                = 10000.0f;
    float defocus2                = 10000.0f;
    float defocus_angle;
    float phase_shift;
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
    float    mask_radius           = 0.0f;
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

    mip_input_filename        = input_search_image_file_name_full.GetName( ) + "_mip.mrc";
    scaled_mip_input_filename = input_search_image_file_name_full.GetName( ) + "_scaled_mip.mrc";
    best_psi_input_filename   = input_search_image_file_name_full.GetName( ) + "_psi.mrc";
    best_theta_input_filename = input_search_image_file_name_full.GetName( ) + "_theta.mrc";
    best_phi_input_filename   = input_search_image_file_name_full.GetName( ) + "_phi.mrc";
    // For now, we'll just use the scaled_mip as supplied, but may want to be able to adjust the scalling later
    // Then instead of applying the smoothing in global_search, we have the raw output to save here.
    // correlation_avg_input_filename = input_search_image_file_name_full.GetName( ) + "_avg.mrc";
    // correlation_std_input_filename = input_search_image_file_name_full.GetName( ) + "_std.mrc";

    wanted_threshold        = my_input->GetFloatFromUser("Peak threshold", "Peaks over this size will be taken", "7.5", 0.0);
    min_peak_radius         = my_input->GetFloatFromUser("Min peak radius (px.)", "Essentially the minimum closeness for peaks", "10.0", 0.0);
    pixel_size              = my_input->GetFloatFromUser("Pixel size of images (A)", "Pixel size of input images in Angstroms", "1.0", 0.0);
    voltage_kV              = my_input->GetFloatFromUser("Beam energy (keV)", "The energy of the electron beam used to image the sample in kilo electron volts", "300.0", 0.0);
    spherical_aberration_mm = my_input->GetFloatFromUser("Spherical aberration (mm)", "Spherical aberration of the objective lens in millimeters", "2.7");
    amplitude_contrast      = my_input->GetFloatFromUser("Amplitude contrast", "Assumed amplitude contrast", "0.07", 0.0, 1.0);
    defocus1                = my_input->GetFloatFromUser("Defocus1 (angstroms)", "Defocus1 for the input image", "10000", 0.0);
    defocus2                = my_input->GetFloatFromUser("Defocus2 (angstroms)", "Defocus2 for the input image", "10000", 0.0);
    defocus_angle           = my_input->GetFloatFromUser("Defocus angle (degrees)", "Defocus Angle for the input image", "0.0");
    phase_shift             = my_input->GetFloatFromUser("Phase shift (degrees)", "Additional phase shift in degrees", "0.0");
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
    padding     = my_input->GetFloatFromUser("Padding factor", "Factor determining how much the input volume is padded to improve projections", "2.0", 1.0);
    mask_radius = my_input->GetFloatFromUser("Mask radius (A) (0.0 = no mask)", "Radius of a circular mask to be applied to the input particles during refinement", "0.0", 0.0);
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
    my_current_job.ManualSetArguments("ttfffffffffffifffbfftttttfffbtfiiiiiitftbt",
                                      input_search_images.ToUTF8( ).data( ), // 0
                                      input_reconstruction.ToUTF8( ).data( ), // 1
                                      pixel_size, // 2
                                      voltage_kV, // 3
                                      spherical_aberration_mm, // 4
                                      amplitude_contrast, // 5
                                      defocus1, // 6
                                      defocus2, // 7
                                      defocus_angle, // 8
                                      low_resolution_limit, // 9
                                      high_resolution_limit, // 10
                                      angular_range, // 11
                                      angular_step, // 12
                                      best_parameters_to_keep, // 13
                                      defocus_search_range, // 14
                                      defocus_search_step, // 15
                                      padding, // 16
                                      do_defocus_refinement, // 17
                                      mask_radius, // 18
                                      phase_shift, // 19
                                      mip_input_filename.ToUTF8( ).data( ), // 20
                                      scaled_mip_input_filename.ToUTF8( ).data( ), // 21
                                      best_psi_input_filename.ToUTF8( ).data( ), // 22
                                      best_theta_input_filename.ToUTF8( ).data( ), // 23
                                      best_phi_input_filename.ToUTF8( ).data( ), // 24
                                      wanted_threshold, // 25
                                      min_peak_radius, // 26
                                      xy_change_threshold, // 27
                                      exclude_above_xy_threshold, // 28
                                      my_symmetry.ToUTF8( ).data( ), // 29
                                      in_plane_angular_step, // 30
                                      first_search_position, // 31
                                      last_search_position, // 32
                                      image_number_for_gui, // 33
                                      number_of_jobs_per_image_in_gui, // 34
                                      result_number, // 35
                                      max_threads, // 36
                                      directory_for_results_string.ToUTF8( ).data( ), // 37
                                      threshold_for_result_plotting, // 38
                                      filename_for_gui_result_image.ToUTF8( ).data( ), // 39
                                      input_star_filename.ToUTF8( ).data( )); // 40
}

// override the do calculation method which will be what is actually run..

bool GlobalSearchRefinementApp::DoCalculation( ) {
    wxDateTime start_time = wxDateTime::Now( );

    input_search_images       = my_current_job.arguments[0].ReturnStringArgument( );
    input_search_images       = my_current_job.arguments[1].ReturnStringArgument( );
    pixel_size                = my_current_job.arguments[2].ReturnFloatArgument( );
    voltage_kV                = my_current_job.arguments[3].ReturnFloatArgument( );
    spherical_aberration_mm   = my_current_job.arguments[4].ReturnFloatArgument( );
    amplitude_contrast        = my_current_job.arguments[5].ReturnFloatArgument( );
    defocus1                  = my_current_job.arguments[6].ReturnFloatArgument( );
    defocus2                  = my_current_job.arguments[7].ReturnFloatArgument( );
    defocus_angle             = my_current_job.arguments[8].ReturnFloatArgument( );
    low_resolution_limit      = my_current_job.arguments[9].ReturnFloatArgument( );
    high_resolution_limit     = my_current_job.arguments[10].ReturnFloatArgument( );
    angular_range             = my_current_job.arguments[11].ReturnFloatArgument( );
    angular_step              = my_current_job.arguments[12].ReturnFloatArgument( );
    best_parameters_to_keep   = my_current_job.arguments[13].ReturnIntegerArgument( );
    defocus_search_range      = my_current_job.arguments[14].ReturnFloatArgument( );
    defocus_search_step       = my_current_job.arguments[15].ReturnFloatArgument( );
    padding                   = my_current_job.arguments[16].ReturnFloatArgument( );
    do_defocus_refinement     = my_current_job.arguments[17].ReturnBoolArgument( );
    mask_radius               = my_current_job.arguments[18].ReturnFloatArgument( );
    phase_shift               = my_current_job.arguments[19].ReturnFloatArgument( );
    mip_input_filename        = my_current_job.arguments[20].ReturnStringArgument( );
    scaled_mip_input_filename = my_current_job.arguments[21].ReturnStringArgument( );
    best_psi_input_filename   = my_current_job.arguments[22].ReturnStringArgument( );
    best_theta_input_filename = my_current_job.arguments[23].ReturnStringArgument( );
    best_phi_input_filename   = my_current_job.arguments[24].ReturnStringArgument( );

    wanted_threshold           = my_current_job.arguments[25].ReturnFloatArgument( );
    min_peak_radius            = my_current_job.arguments[26].ReturnFloatArgument( );
    xy_change_threshold        = my_current_job.arguments[27].ReturnFloatArgument( );
    exclude_above_xy_threshold = my_current_job.arguments[28].ReturnBoolArgument( );
    wxString my_symmetry       = my_current_job.arguments[29].ReturnStringArgument( );
    in_plane_angular_step      = my_current_job.arguments[30].ReturnFloatArgument( );

    int first_search_position              = my_current_job.arguments[31].ReturnIntegerArgument( );
    int last_search_position               = my_current_job.arguments[32].ReturnIntegerArgument( );
    int image_number_for_gui               = my_current_job.arguments[33].ReturnIntegerArgument( );
    int number_of_jobs_per_image_in_gui    = my_current_job.arguments[34].ReturnIntegerArgument( );
    result_number                          = my_current_job.arguments[35].ReturnIntegerArgument( );
    max_threads                            = my_current_job.arguments[36].ReturnIntegerArgument( );
    wxString directory_for_results         = my_current_job.arguments[37].ReturnStringArgument( );
    float    threshold_for_result_plotting = my_current_job.arguments[38].ReturnFloatArgument( );
    wxString filename_for_gui_result_image = my_current_job.arguments[39].ReturnStringArgument( );
    input_star_filename                    = my_current_job.arguments[40].ReturnStringArgument( );

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
    long  best_address;

    int current_x;
    int current_y;

    int phi_i;
    int theta_i;
    int psi_i;
    int defocus_i;
    int defocus_is = 0;
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

    Peak* found_peaks = nullptr;
    Peak  current_peak;
    Peak  template_peak;
    Peak  best_peak;
    long  current_address;
    long  address_offset;

    float current_phi;
    float current_theta;
    float current_psi;
    float current_defocus1;
    float current_defocus2;

    float current_pixel_size;
    float best_score;
    float score;
    float starting_score;
    bool  first_score;

    float best_phi_score;
    float best_theta_score;
    float best_psi_score;
    float best_defocus_score;

    int   ii, jj, kk, ll;
    float mult_i;
    float mult_i_start;
    float defocus_step;
    float score_adjustment;
    float offset_distance;
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

    // assume cube in determining the maximum radius.

    float maximum_mask_radius = (float(input_reconstruction_file.ReturnXSize( )) / 2.0f - 1.0f) * pixel_size;
    if ( mask_radius > maximum_mask_radius )
        mask_radius = maximum_mask_radius;

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

    best_scaled_mip.CopyFrom(&scaled_mip_image);
    current_peak.value = std::numeric_limits<float>::max( );

    wxFileName star_filename(ReadFromcisTEMStarFile);
    if ( star_filename.GetExt( ) == "cistem" )
        input_star_file.ReadFromcisTEMBinaryFile(input_star_filename);
    else
        input_star_file.ReadFromcisTEMStarFile(input_star_filename);

    // To make the transition easier, first keep these unnecessary columns in the output file
    number_of_peaks_found = input_star_file.ReturnNumberofLines( );

    // FIXME: replace this logic with an actual bool to refine or not to refine and then instead have a sanity check here.
    // FIXME: There is a defocus_search_step, defocus_refine_step, and a defocus_step - wtf

    if ( is_running_locally == true ) {
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

#pragma omp parallel num_threads(max_threads) default(none) shared(number_of_peaks_found, found_peaks, input_image, mask_radius, pixel_size, mask_falloff,                                                                                                                                                       \
                                                                   mip_image, scaled_mip_image, phi_image, theta_image, psi_image, defocus_search_range, defocus_refine_step,                                                                                                                                    \
                                                                   defocus1, defocus2, defocus_angle, angular_step, in_plane_angular_step, whitening_filter, input_reconstruction, min_peak_radius2,                                                                                                             \
                                                                   input_reconstruction_file, voltage_kV, spherical_aberration_mm, amplitude_contrast,                                                                                                                                                           \
                                                                   phase_shift, max_threads, defocus_step, xy_change_threshold, exclude_above_xy_threshold, all_peak_changes, all_peak_infos) private(current_peak, sq_dist_x, sq_dist_y, address, current_address, current_phi, current_theta, current_psi,     \
                                                                                                                                                                                                      current_defocus1, current_defocus2, best_score, phi_i, theta_i, psi_i, defocus_i, size_i,                  \
                                                                                                                                                                                                      mult_i_start, mult_i, ll, input_ctf, best_defocus_score, best_phi_score, best_theta_score, best_psi_score, \
                                                                                                                                                                                                      kk, jj, ii, angles, score, address_offset, template_peak, i, j, best_address, peak_number,                 \
                                                                                                                                                                                                      first_score, starting_score, size_is, defocus_is, score_adjustment, offset_distance, best_peak)
    {

        Image windowed_particle_;
        Image projection_filter_;

        // TODO: switch to me for gpu access, requires going from input_reconstruction (Image) to reconstructed volume
        // requires a particle object as well
        // ProjectionComparisonObjects comparison_object_;
        TemplateComparisonObject template_object;

        input_ctf.Init(voltage_kV, spherical_aberration_mm, amplitude_contrast, defocus1, defocus2, defocus_angle, 0.0, 0.0, 0.0, pixel_size, deg_2_rad(phase_shift));
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
            // look for a peak..

            current_peak.x     = input_star_file.ReturnXShift(peak_number) / pixel_size;
            current_peak.y     = input_star_file.ReturnYShift(peak_number) / pixel_size;
            current_peak.value = input_star_file.ReturnScore(peak_number);

            // ok we have peak..

            // FIXME: confirm how the fractional shifts should look, but for now it prob doesn't matter too much.
            input_image.ClipInto(&windowed_particle_, 0.0f, false, 1.0f, myroundint(current_peak.x), myroundint(current_peak.y), 0);
            if ( mask_radius > 0.0f )
                windowed_particle_.CosineMask(mask_radius / pixel_size, mask_falloff / pixel_size);
            windowed_particle_.ForwardFFT( );
            windowed_particle_.SwapRealSpaceQuadrants( );

            // TODO: what is pixel_size_factor? If not immediately switching to Projection compairson object, then add a Reset() method
            template_object.pixel_size_factor = 1.0f;
            first_score                       = false;

            //		number_of_peaks_found++;

            // get angles and mask out the local area so it won't be picked again..

            address = 0;

            // FIXME we should already have all of this info in the parameters object
            // current_peak.x = current_peak.x + scaled_mip_image_local.physical_address_of_box_center_x;
            // current_peak.y = current_peak.y + scaled_mip_image_local.physical_address_of_box_center_y;

            // FIXME: from parameters
            // current_phi        = best_phi_local.real_values[address];
            // current_theta      = best_theta_local.real_values[address];
            // current_psi        = best_psi_local.real_values[address];
            // current_defocus    = best_defocus_local.real_values[address];
            current_phi      = input_star_file.ReturnPhi(peak_number);
            current_theta    = input_star_file.ReturnTheta(peak_number);
            current_psi      = input_star_file.ReturnPsi(peak_number);
            current_defocus1 = input_star_file.ReturnDefocus1(peak_number);
            current_defocus2 = input_star_file.ReturnDefocus2(peak_number);

            best_score = -std::numeric_limits<float>::max( );
            angles.Init(current_phi, current_theta, current_psi, 0.0, 0.0);

            // FIXME: current_defocus will be zero and pixel_size should be replaced with either pixel_size_search or something similar to account for resampling
            input_ctf.SetDefocus(current_defocus1 / pixel_size, current_defocus2 / pixel_size, deg_2_rad(input_star_file.ReturnDefocusAngle(peak_number)));
            projection_filter_.CalculateCTFImage(input_ctf);
            projection_filter_.ApplyCurveFilter(&whitening_filter);

            template_peak = TemplateScore(&template_object);

            if ( max_threads == 1 )
                wxPrintf("\nRefining peak %i at x, y =  %6i, %6i\n", peak_number + 1, myroundint(current_peak.x), myroundint(current_peak.y));

            // TODO: resume here - go back and get rid of microscope parameters on input and replace them from the parameter file.
            // FIXME: This should probably be after the angular search (perhaps test both or possible make it exclusive (can't be done with angular search))
            if ( do_defocus_refinement ) {
                for ( defocus_is = -myroundint(float(defocus_search_range) / float(defocus_step)); defocus_is <= myroundint(float(defocus_search_range) / float(defocus_step)); defocus_is++ ) {
                    input_ctf.SetDefocus((defocus1 + current_defocus + defocus_is * defocus_step) / pixel_size, (defocus2 + current_defocus + defocus_is * defocus_step) / pixel_size, deg_2_rad(defocus_angle));
                    projection_filter_.CalculateCTFImage(input_ctf);
                    projection_filter_.ApplyCurveFilter(&whitening_filter);

                    //							angles.Init(current_phi, current_theta, current_psi, 0.0, 0.0);

                    template_peak = TemplateScore(&template_object);
                    score         = template_peak.value;
                    if ( score > best_score ) {
                        best_peak       = template_peak;
                        best_score      = score;
                        address_offset  = (scaled_mip_image.logical_x_dimension + scaled_mip_image.padding_jump_value) * myroundint(template_peak.y) + myroundint(template_peak.x);
                        best_address    = current_address + address_offset;
                        offset_distance = sqrtf(powf(template_peak.x, 2) + powf(template_peak.y, 2));
                    }
                }
                // current_defocus = best_defocus_local.real_values[best_address];
            }

            phi_i     = 0;
            theta_i   = 0;
            psi_i     = 0;
            defocus_i = 0;
            //					score = best_score;

            mult_i_start = defocus_step / defocus_refine_step;
            for ( mult_i = mult_i_start; mult_i > 0.5f; mult_i /= 2.0f ) {
                for ( ll = 0; ll < 2; ll = -2 * ll + 1 ) {
                    if ( (ll != 0) && (defocus_refine_step == 0.0f) )
                        break;
                    do { // while ( best_score > best_defocus_score );
                        best_defocus_score = best_score;
                        if ( defocus_search_range != 0.0f )
                            defocus_i += myroundint(mult_i * ll);

                        // make the projection filter, which will be CTF * whitening filter
                        input_ctf.SetDefocus((defocus1 + current_defocus + defocus_i * defocus_refine_step) / pixel_size, (defocus2 + current_defocus + defocus_i * defocus_refine_step) / pixel_size, deg_2_rad(defocus_angle));
                        projection_filter_.CalculateCTFImage(input_ctf);
                        projection_filter_.ApplyCurveFilter(&whitening_filter);

                        for ( kk = 0; kk < 2; kk = -2 * kk + 1 ) {
                            do { // while ( best_score > best_phi_score );
                                best_phi_score = best_score;
                                phi_i += kk;
                                for ( jj = 0; jj < 2; jj = -2 * jj + 1 ) {
                                    do { // while ( best_score > best_theta_score );
                                        best_theta_score = best_score;
                                        theta_i += jj;
                                        for ( ii = 0; ii < 2; ii = -2 * ii + 1 ) {
                                            do { // while ( best_score > best_psi_score );
                                                best_psi_score = best_score;
                                                psi_i += ii;

                                                angles.Init(current_phi + phi_i * angular_step, current_theta + theta_i * angular_step, current_psi + psi_i * in_plane_angular_step, 0.0, 0.0);

                                                template_peak = TemplateScore(&template_object);
                                                score         = template_peak.value;
                                                if ( score > best_score ) {
                                                    best_peak       = template_peak;
                                                    best_score      = score;
                                                    address_offset  = (scaled_mip_image.logical_x_dimension + scaled_mip_image.padding_jump_value) * myroundint(template_peak.y) + myroundint(template_peak.x);
                                                    best_address    = current_address + address_offset;
                                                    offset_distance = sqrtf(powf(template_peak.x, 2) + powf(template_peak.y, 2));
                                                    // best_psi_local.real_values[best_address]   = current_psi + psi_i * in_plane_angular_step;
                                                    // best_theta_local.real_values[best_address] = current_theta + theta_i * angular_step;
                                                    // best_phi_local.real_values[best_address]   = current_phi + phi_i * angular_step;
                                                }
                                            } while ( best_score > best_psi_score );
                                            psi_i -= ii;
                                        }
                                    } while ( best_score > best_theta_score );
                                    theta_i -= jj;
                                }
                            } while ( best_score > best_phi_score );
                            phi_i -= kk;
                        }
                    } while ( best_score > best_defocus_score );
                    if ( defocus_search_range != 0.0f )
                        defocus_i -= ll;
                }
            }

            all_peak_changes[peak_number].x_pos = best_peak.x * pixel_size;
            all_peak_changes[peak_number].y_pos = best_peak.y * pixel_size;
            // all_peak_changes[peak_number].psi   = best_psi_local.real_values[best_address] - psi_image.real_values[current_address];
            // all_peak_changes[peak_number].theta = best_theta_local.real_values[best_address] - theta_image.real_values[current_address];

            // all_peak_changes[peak_number].peak_height = best_scaled_mip_local.real_values[best_address] - starting_score;

            all_peak_infos[peak_number].x_pos = found_peaks[peak_number].x + best_peak.x; // NOT SCALING BY PIXEL SIZE - DO AFTER MAKING RESULT IMAGE
            all_peak_infos[peak_number].y_pos = found_peaks[peak_number].y + best_peak.y; // NOT SCALING BY PIXEL SIZE - DO AFTER MAKING RESULT IMAGE
            // all_peak_infos[peak_number].psi   = best_psi_local.real_values[best_address];
            // all_peak_infos[peak_number].theta = best_theta_local.real_values[best_address];
            // all_peak_infos[peak_number].phi   = best_phi_local.real_values[best_address];
            // all_peak_infos[peak_number].defocus = best_defocus_local.real_values[best_address];

            // all_peak_infos[peak_number].peak_height = best_scaled_mip_local.real_values[best_address];

        NEXTPEAK:
            if ( angular_step == 0.0 && in_plane_angular_step == 0.0 )
                wxPrintf("Stopping refinement now\n");
        }

    } // end omp section

    //	delete my_progress;

    delete[] found_peaks;
    //	delete [] addresses;

    if ( is_running_locally == true ) {
        wxPrintf("\nRefine Template: Normal termination\n");
        wxDateTime finish_time = wxDateTime::Now( );
        wxPrintf("Total Run Time : %s\n\n", finish_time.Subtract(start_time).Format("%Hh:%Mm:%Ss"));
    }
    else // find peaks, write and write a result image, then send result..
    {

        for ( int counter = 0; counter < all_peak_infos.GetCount( ); counter++ ) {

            if ( all_peak_infos[counter].peak_height < threshold_for_result_plotting ) {
                all_peak_infos[counter].x_pos = (all_peak_infos[counter].x_pos + best_scaled_mip.physical_address_of_box_center_x) * pixel_size;
                all_peak_infos[counter].y_pos = (all_peak_infos[counter].y_pos + best_scaled_mip.physical_address_of_box_center_y) * pixel_size;
                continue;
            }

            offset_distance = sqrtf(powf(all_peak_changes[counter].x_pos, 2) + powf(all_peak_changes[counter].y_pos, 2));
            if ( offset_distance * pixel_size >= xy_change_threshold && exclude_above_xy_threshold == true )
                continue;

            // ok we have peak..

            number_of_peaks_found++;

            // get angles and mask out the local area so it won't be picked again..

            // scale the shifts by the pixel size..

            all_peak_infos[counter].x_pos = (all_peak_infos[counter].x_pos + best_scaled_mip.physical_address_of_box_center_x) * pixel_size;
            all_peak_infos[counter].y_pos = (all_peak_infos[counter].y_pos + best_scaled_mip.physical_address_of_box_center_y) * pixel_size;
        }

        // tell the gui that this result is available...

        SendTemplateMatchingResultToSocket(controller_socket, image_number_for_gui, threshold_for_result_plotting, all_peak_infos, all_peak_changes);
    }

    return true;
}
