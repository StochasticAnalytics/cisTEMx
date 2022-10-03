
#include <cistem_config.h>
#include "refine3d_defines.h"

#ifdef ENABLEGPU
#warning "GPU enabled in refine3d"
#include "../../gpu/gpu_core_headers.h"
#include "../../gpu/GpuImage.h"
#else
#include "../../core/core_headers.h"
#endif

#ifdef CISTEM_PROFILING
using namespace cistem_timer;
#else
using namespace cistem_timer_noop;
#endif

#include "ProjectionComparisonObjects.h"
#include "batch_size_optimizer.h"

StopWatch global_timer;

class
        Refine3DApp : public MyApp {
  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );
    void AddCommandLineOptions( );

    // Depends on compilation mode --enable-profiling
    StopWatch timer;
    int       wanted_batch_size = 1;

  private:
};

//float		beam_tilt_x = 0.0f;
//float		beam_tilt_y = 0.0f;
// This is the function which will be minimized
float FrealignObjectiveFunction(void* scoring_parameters, float* array_of_values) {

    global_timer.start("re_cast");
    ProjectionComparisonObjects* comparison_object = reinterpret_cast<ProjectionComparisonObjects*>(scoring_parameters);
    global_timer.lap("re_cast");
    //for (int i = 0; i < comparison_object->particle->number_of_parameters; i++) {comparison_object->particle->temp_float[i] = comparison_object->particle->current_parameters[i];}
    global_timer.start("copy_params");
    comparison_object->particle->temp_parameters = comparison_object->particle->current_parameters;
    global_timer.lap("copy_params");
    global_timer.start("update_params");
    comparison_object->particle->UnmapParameters(array_of_values);
    global_timer.lap("update_params");

    bool calculate_projection = true; // normal default for cpu refinement, over-ride when calculating gpu projection
    bool use_gpu_projection   = false; // normal default for cpu refinement, used to selectivley override image methods in the comparison_object->reference_volume->CalculateProjection() method

    if ( comparison_object->particle->no_ctf_weighting ) {
        comparison_object->swap_quadrants = false;
        comparison_object->apply_shifts   = false;
        comparison_object->whiten         = false;
        comparison_object->apply_ctf      = false;
        comparison_object->absolute_ctf   = false;
    }
    else {
        if ( comparison_object->particle->includes_reference_ssnr_weighting ) {
            comparison_object->swap_quadrants = false;
            comparison_object->apply_shifts   = true;
            comparison_object->whiten         = true;
            comparison_object->apply_ctf      = false;
            comparison_object->absolute_ctf   = false;
        }
        // Case for normal parameter refinement with weighting applied only to particle images
        else {
            comparison_object->swap_quadrants = false;
            comparison_object->apply_shifts   = true;
            comparison_object->whiten         = false;
            comparison_object->apply_ctf      = true;
            comparison_object->absolute_ctf   = true;
        }
    }

    // FIXME: testing
    // comparison_object->whiten = false;
#ifdef ENABLEGPU

    // Flip the overrides;
    calculate_projection = false;
    use_gpu_projection   = true;
    // First get the projection, optionally shifted and multiplied by the CTF
    global_timer.start("gpu_projection");
    float gpu_score = comparison_object->DoGpuProjection( );
    global_timer.lap("gpu_projection");

#ifdef SAVE_DEBUG_IMAGES
    if ( comparison_object->nprj < N_DEBUG_IMAGES ) {
        comparison_object->gpu_particle_image.QuickAndDirtyWriteSlices("gpu_particle_" + std::to_string(comparison_object->nprj) + ".mrc", 1, 1);
        comparison_object->particle->particle_image->QuickAndDirtyWriteSlices("cpu_particle_" + std::to_string(comparison_object->nprj) + ".mrc", 1, 1);
        comparison_object->gpu_projection.SwapRealSpaceQuadrants( );
        comparison_object->gpu_projection.QuickAndDirtyWriteSlices("gpu_projection_fromGPU_" + std::to_string(comparison_object->nprj) + ".mrc", 1, 1);
        cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
        comparison_object->projection_image->SwapRealSpaceQuadrants( );
        comparison_object->projection_image->QuickAndDirtyWriteSlice("gpu_projection_" + std::to_string(comparison_object->nprj) + ".mrc", 1);
        comparison_object->projection_image->SwapRealSpaceQuadrants( );
        comparison_object->nprj++;
        if ( comparison_object->nprj == N_DEBUG_IMAGES ) {
            wxPrintf("GPU projection is done\n");
            exit(0);
        }
    }
#endif

#endif

    global_timer.start("calc_proj cpu");
    comparison_object->reference_volume->CalculateProjection(*comparison_object->projection_image,
                                                             *comparison_object->particle->ctf_image, comparison_object->particle->alignment_parameters,
                                                             comparison_object->mask_radius, comparison_object->mask_falloff,
                                                             comparison_object->particle->pixel_size / comparison_object->particle->filter_radius_high,
                                                             comparison_object->swap_quadrants,
                                                             comparison_object->apply_shifts,
                                                             comparison_object->whiten,
                                                             comparison_object->apply_ctf,
                                                             comparison_object->absolute_ctf,
                                                             calculate_projection,
                                                             use_gpu_projection);

    global_timer.lap("calc_proj cpu");

#if defined(SAVE_DEBUG_IMAGES) && ! defined(ENABLEGPU)
    if ( comparison_object->nprj < 10 ) {
        comparison_object->projection_image->SwapRealSpaceQuadrants( );
        comparison_object->projection_image->QuickAndDirtyWriteSlice("cpu_projection_" + std::to_string(comparison_object->nprj) + ".mrc", 1);
        comparison_object->projection_image->SwapRealSpaceQuadrants( );
        comparison_object->nprj++;
        if ( comparison_object->nprj == N_DEBUG_IMAGES ) {
            wxPrintf("CPU projection is done\n");
            exit(0);
        }
    }
#endif

    // The minimizer sometimes tries weird values
    if ( isnan(comparison_object->particle->alignment_parameters.ReturnShiftX( )) || fabsf(comparison_object->particle->alignment_parameters.ReturnShiftX( ) - comparison_object->initial_x_shift) > comparison_object->x_shift_limit )
        return 1;
    if ( isnan(comparison_object->particle->alignment_parameters.ReturnShiftY( )) || fabsf(comparison_object->particle->alignment_parameters.ReturnShiftY( ) - comparison_object->initial_y_shift) > comparison_object->y_shift_limit )
        return 1;
    if ( isnan(comparison_object->particle->alignment_parameters.ReturnPsiAngle( )) || fabsf(comparison_object->particle->alignment_parameters.ReturnPsiAngle( ) - comparison_object->initial_psi_angle) > comparison_object->angle_change_limit )
        return 1;
    if ( isnan(comparison_object->particle->alignment_parameters.ReturnPhiAngle( )) || fabsf(comparison_object->particle->alignment_parameters.ReturnPhiAngle( ) - comparison_object->initial_phi_angle) > comparison_object->angle_change_limit )
        return 1;
    if ( isnan(comparison_object->particle->alignment_parameters.ReturnThetaAngle( )) || fabsf(comparison_object->particle->alignment_parameters.ReturnThetaAngle( ) - comparison_object->initial_theta_angle) > comparison_object->angle_change_limit )
        return 1;

    if ( isnan(comparison_object->particle->temp_parameters.x_shift) || fabsf(comparison_object->particle->temp_parameters.x_shift - comparison_object->initial_x_shift) > comparison_object->x_shift_limit )
        return 1;
    if ( isnan(comparison_object->particle->temp_parameters.y_shift) || fabsf(comparison_object->particle->temp_parameters.y_shift - comparison_object->initial_y_shift) > comparison_object->y_shift_limit )
        return 1;
    if ( isnan(comparison_object->particle->temp_parameters.psi) || fabsf(comparison_object->particle->temp_parameters.psi - comparison_object->initial_psi_angle) > comparison_object->angle_change_limit )
        return 1;
    if ( isnan(comparison_object->particle->temp_parameters.phi) || fabsf(comparison_object->particle->temp_parameters.phi - comparison_object->initial_phi_angle) > comparison_object->angle_change_limit )
        return 1;
    if ( isnan(comparison_object->particle->temp_parameters.theta) || fabsf(comparison_object->particle->temp_parameters.theta - comparison_object->initial_theta_angle) > comparison_object->angle_change_limit )
        return 1;

    global_timer.start("ReturnScore");

#if defined(CALCULATE_SCORE_ON_CPU_DISABLE_GPU_PARTICLE) || ! defined(ENABLEGPU)
#ifdef USE_OPTIMIZED_CPU_SCORE_CALCULATION
    float filter_radius_high = fminf(powf(comparison_object->particle->pixel_size / comparison_object->particle->filter_radius_high, 2), 0.25);
    float filter_radius_low  = 0.0f;
    if ( comparison_object->particle->filter_radius_low != 0.0 )
        filter_radius_low = powf(comparison_object->particle->pixel_size / comparison_object->particle->filter_radius_low, 2);

    float tmp_corr = comparison_object->particle->particle_image->GetWeightedCorrelationWithImage(*comparison_object->projection_image,
                                                                                                  filter_radius_low, filter_radius_high,
                                                                                                  powf(comparison_object->particle->pixel_size / comparison_object->particle->signed_CC_limit, 2));
#else
    float tmp_corr = comparison_object->particle->particle_image->GetWeightedCorrelationWithImage(*comparison_object->projection_image, comparison_object->particle->bin_index,
                                                                                                  comparison_object->particle->pixel_size / comparison_object->particle->signed_CC_limit);

#endif
#else
#ifdef ENABLEGPU
    float tmp_corr = gpu_score;
#else
#error "We need to calculate the score somewhere!!"
#endif

#endif

    global_timer.lap("ReturnScore");

    // wxPrintf("Gpu score: gpu_score = %f, cpu_score = %f\n", gpu_score, tmp_corr);
    global_timer.start("ReturnPenalty");

    float tmp_penalty = comparison_object->particle->ReturnParameterPenalty(comparison_object->particle->temp_parameters);
    global_timer.lap("ReturnPenalty");

#ifdef CISTEM_DEBUG
    if ( isnan(tmp_corr) || isnan(tmp_penalty) ) {
        MyPrintWithDetails("FrealignObjectiveFunction about to return NaN. Details to follow.\n");
        wxPrintf("shift x = %f\nshift y = %f\npsi   = %f\nphi   = %f\ntheta = %f\n",
                 comparison_object->particle->alignment_parameters.ReturnShiftX( ),
                 comparison_object->particle->alignment_parameters.ReturnShiftY( ),
                 comparison_object->particle->alignment_parameters.ReturnPsiAngle( ),
                 comparison_object->particle->alignment_parameters.ReturnPhiAngle( ),
                 comparison_object->particle->alignment_parameters.ReturnThetaAngle( ));
    }

    MyDebugAssertFalse(isnan(tmp_corr), "Frealign score function: correlation term is NaN");
    MyDebugAssertFalse(isnan(tmp_penalty), "Frealign score function: penalty term is NaN");
#endif

#ifdef PRINT_SCORES
    wxPrintf("Current score is %g, from line %i\n", -100.f * (-tmp_corr - tmp_penalty), __LINE__);
#endif

    return -tmp_corr - tmp_penalty;

    // This penalty term assumes a Gaussian x,y distribution that is probably not correct in most cases. It might be better to leave it out.
}

IMPLEMENT_APP(Refine3DApp)

// override the DoInteractiveUserInput
void Refine3DApp::AddCommandLineOptions( ) {

    // TODO consider short vs long switches.

    // Options for saving diagnostic images
    command_line_parser.AddOption("", "batch-size", "Requested batch size for in-plane search in global refinement. Default is 1", wxCMD_LINE_VAL_NUMBER);
}

void Refine3DApp::DoInteractiveUserInput( ) {

    long temp_long;
    if ( command_line_parser.Found("batch-size", &temp_long) ) {
        wanted_batch_size = int(temp_long);
    }

    wxString input_particle_images;
    wxString input_star_filename;
    wxString input_reconstruction;
    wxString input_reconstruction_statistics;
    bool     use_statistics;
    wxString ouput_matching_projections;
    wxString ouput_star_filename;
    wxString ouput_shift_filename;
    wxString my_symmetry    = "C1";
    int      first_particle = 1;
    int      last_particle  = 0;
    float    percent_used   = 1.0;
    float    pixel_size     = 1.0;
    //	float		voltage_kV = 300.0;
    //	float		spherical_aberration_mm = 2.7;
    //	float		amplitude_contrast = 0.07;
    float molecular_mass_kDa              = 1000.0;
    float inner_mask_radius               = 0.0;
    float outer_mask_radius               = 100.0;
    float low_resolution_limit            = 300.0;
    float high_resolution_limit           = 8.0;
    float signed_CC_limit                 = 0.0;
    float classification_resolution_limit = 0.0;
    float mask_radius_search              = 0.0;
    float high_resolution_limit_search    = 20.0;
    float angular_step                    = 5.0;
    int   best_parameters_to_keep         = 20;
    float max_search_x                    = 0;
    float max_search_y                    = 0;
    float mask_center_2d_x                = 100.0;
    float mask_center_2d_y                = 100.0;
    float mask_center_2d_z                = 100.0;
    float mask_radius_2d                  = 100.0;
    float defocus_search_range            = 500;
    float defocus_step                    = 50;
    float padding                         = 1.0;
    //	float		filter_constant = 1.0;
    bool do_global_search               = false;
    bool do_local_refinement            = true;
    bool refine_psi                     = true;
    bool refine_theta                   = true;
    bool refine_phi                     = true;
    bool refine_x                       = true;
    bool refine_y                       = true;
    bool calculate_matching_projections = false;
    bool apply_2D_masking               = false;
    bool ctf_refinement                 = false;
    bool normalize_particles            = true;
    bool invert_contrast                = false;
    bool exclude_blank_edges            = true;
    bool normalize_input_3d             = true;
    bool threshold_input_3d             = true;
    int  max_threads;

    UserInput* my_input = new UserInput("Refine3D", 1.02);

    input_particle_images           = my_input->GetFilenameFromUser("Input particle images", "The input image stack, containing the experimental particle images", "my_image_stack.mrc", true);
    input_star_filename             = my_input->GetFilenameFromUser("Input cisTEM star file", "The input star file, containing your particle alignment parameters", "my_parameters.star", true);
    input_reconstruction            = my_input->GetFilenameFromUser("Input reconstruction", "The 3D reconstruction from which projections are calculated", "my_reconstruction.mrc", true);
    input_reconstruction_statistics = my_input->GetFilenameFromUser("Input data statistics", "The table listing FSC, Part_FSC, Part_SSNR and Rec_SSNR", "my_statistics.txt", false);
    use_statistics                  = my_input->GetYesNoFromUser("Use statistics", "Answer No if no statistics are available?", "Yes");
    ouput_matching_projections      = my_input->GetFilenameFromUser("Output matching projections", "The output image stack, containing the matching projections", "my_projection_stack.mrc", false);
    ouput_star_filename             = my_input->GetFilenameFromUser("Output cisTEM star file", "The output star file, containing your refined particle alignment parameters", "my_refined_parameters.star", false);
    ouput_shift_filename            = my_input->GetFilenameFromUser("Output parameter changes", "The changes in the alignment parameters compared to the input parameters", "my_parameter_changes.star", false);
    my_symmetry                     = my_input->GetSymmetryFromUser("Particle symmetry", "The assumed symmetry of the particle to be reconstructed", "C1");
    first_particle                  = my_input->GetIntFromUser("First particle to refine (0 = first in stack)", "The first particle in the stack that should be refined", "1", 0);
    last_particle                   = my_input->GetIntFromUser("Last particle to refine (0 = last in stack)", "The last particle in the stack that should be refined", "0", 0);
    percent_used                    = my_input->GetFloatFromUser("Percent of particles to use (1 = all)", "The percentage of randomly selected particles that will be refined", "1.0", 0.0, 1.0);
    pixel_size                      = my_input->GetFloatFromUser("Pixel size of reconstruction (A)", "Pixel size of input reconstruction in Angstroms", "1.0", 0.0);
    //	voltage_kV = my_input->GetFloatFromUser("Beam energy (keV)", "The energy of the electron beam used to image the sample in kilo electron volts", "300.0", 0.0);
    //	spherical_aberration_mm = my_input->GetFloatFromUser("Spherical aberration (mm)", "Spherical aberration of the objective lens in millimeters", "2.7", 0.0);
    //	amplitude_contrast = my_input->GetFloatFromUser("Amplitude contrast", "Assumed amplitude contrast", "0.07", 0.0, 1.0);
    //	beam_tilt_x = my_input->GetFloatFromUser("Beam tilt along x (mrad)", "Beam tilt present in data along the x axis in mrad", "0.0", -100.0, 100.0);
    //	beam_tilt_y = my_input->GetFloatFromUser("Beam tilt along y (mrad)", "Beam tilt present in data along the y axis in mrad", "0.0", -100.0, 100.0);
    molecular_mass_kDa              = my_input->GetFloatFromUser("Molecular mass of particle (kDa)", "Total molecular mass of the particle to be reconstructed in kilo Daltons", "1000.0", 0.0);
    inner_mask_radius               = my_input->GetFloatFromUser("Inner mask radius (A)", "Radius of a circular mask to be applied to the center of the input reconstruction in Angstroms", "0.0", 0.0);
    outer_mask_radius               = my_input->GetFloatFromUser("Outer mask radius (A)", "Radius of a circular mask to be applied to the input reconstruction and images during refinement, in Angstroms", "100.0", inner_mask_radius);
    low_resolution_limit            = my_input->GetFloatFromUser("Low resolution limit (A)", "Low resolution limit of the data used for alignment in Angstroms", "300.0", 0.0);
    high_resolution_limit           = my_input->GetFloatFromUser("High resolution limit (A)", "High resolution limit of the data used for alignment in Angstroms", "8.0", 0.0);
    signed_CC_limit                 = my_input->GetFloatFromUser("Resolution limit for signed CC (A) (0.0 = max)", "The absolute value of the weighted Fourier ring correlation will be used beyond this limit", "0.0", 0.0);
    classification_resolution_limit = my_input->GetFloatFromUser("Res limit for classification (A) (0.0 = max)", "Resolution limit of the data used for calculating LogP", "0.0", 0.0);
    mask_radius_search              = my_input->GetFloatFromUser("Mask radius for global search (A) (0.0 = max)", "Radius of a circular mask to be applied to the input images during global search", "100.0", 0.0);
    high_resolution_limit_search    = my_input->GetFloatFromUser("Approx. resolution limit for search (A)", "High resolution limit of the data used in the global search in Angstroms", "20.0", 0.0);
    angular_step                    = my_input->GetFloatFromUser("Angular step (0.0 = set automatically)", "Angular step size for global grid search", "0.0", 0.0);
    best_parameters_to_keep         = my_input->GetIntFromUser("Number of top hits to refine", "The number of best global search orientations to refine locally", "20", 1);
    max_search_x                    = my_input->GetFloatFromUser("Search range in X (A) (0.0 = 0.5 * mask radius)", "The maximum global peak search distance along X from the particle box center", "0.0", 0.0);
    max_search_y                    = my_input->GetFloatFromUser("Search range in Y (A) (0.0 = 0.5 * mask radius)", "The maximum global peak search distance along Y from the particle box center", "0.0", 0.0);
    mask_center_2d_x                = my_input->GetFloatFromUser("2D mask X coordinate (A)", "X coordinate of 2D mask center", "100.0", 0.0);
    mask_center_2d_y                = my_input->GetFloatFromUser("2D mask Y coordinate (A)", "Y coordinate of 2D mask center", "100.0", 0.0);
    mask_center_2d_z                = my_input->GetFloatFromUser("2D mask Z coordinate (A)", "Z coordinate of 2D mask center", "100.0", 0.0);
    mask_radius_2d                  = my_input->GetFloatFromUser("2D mask radius (A)", "Radius of a circular mask to be used for likelihood calculation", "100.0", 0.0);
    defocus_search_range            = my_input->GetFloatFromUser("Defocus search range (A)", "Search range (-value ... + value) around current defocus", "500.0", 0.0);
    defocus_step                    = my_input->GetFloatFromUser("Defocus step (A)", "Step size used in the defocus search", "50.0", 0.0);
    padding                         = my_input->GetFloatFromUser("Tuning parameters: padding factor", "Factor determining how much the input volume is padded to improve projections", "1.0", 1.0);
    //	filter_constant = my_input->GetFloatFromUser("Tuning parameters: filter constant", "Constant determining how strongly data with small CTF values is suppressed during particle alignment", "1.0", 1.0);
    do_global_search               = my_input->GetYesNoFromUser("Global search", "Should a global search be performed before local refinement?", "No");
    do_local_refinement            = my_input->GetYesNoFromUser("Local refinement", "Should a local parameter refinement be performed?", "Yes");
    refine_psi                     = my_input->GetYesNoFromUser("Refine Psi", "Should the Psi (in-plane) Euler angle be refined?", "Yes");
    refine_theta                   = my_input->GetYesNoFromUser("Refine Theta", "Should the Theta (out-of-plane) Euler angle be refined?", "Yes");
    refine_phi                     = my_input->GetYesNoFromUser("Refine Phi", "Should the Phi Euler angle be refined?", "Yes");
    refine_x                       = my_input->GetYesNoFromUser("Refine ShiftX", "Should the X shift be refined?", "Yes");
    refine_y                       = my_input->GetYesNoFromUser("Refine ShiftY", "Should the Y shift be refined?", "Yes");
    calculate_matching_projections = my_input->GetYesNoFromUser("Calculate matching projections", "Should matching projections be calculated?", "No");
    apply_2D_masking               = my_input->GetYesNoFromUser("Apply 2D masking", "Should 2D masking be used for the likelihood calculation?", "No");
    ctf_refinement                 = my_input->GetYesNoFromUser("Refine defocus", "Should the particle defocus be refined?", "No");
    normalize_particles            = my_input->GetYesNoFromUser("Normalize particles", "The input particle images should always be normalized unless they were pre-processed", "Yes");
    invert_contrast                = my_input->GetYesNoFromUser("Invert particle contrast", "Should the contrast in the particle images be inverted?", "No");
    exclude_blank_edges            = my_input->GetYesNoFromUser("Exclude images with blank edges", "Should particle images with blank edges be excluded from processing?", "Yes");
    normalize_input_3d             = my_input->GetYesNoFromUser("Normalize input reconstruction", "The input reconstruction should always be normalized unless it was generated by reconstruct3d with normalized particles", "Yes");
    threshold_input_3d             = my_input->GetYesNoFromUser("Threshold input reconstruction", "Should the input reconstruction thresholded to suppress some of the background noise", "No");
    // Add phase flip option, normalize option, remove input statistics & use statistics

#ifdef _OPENMP
    max_threads = my_input->GetIntFromUser("Max. threads to use for calculation", "When threading, what is the max threads to run", "1", 1);
#else
    max_threads    = 1;
#endif

    delete my_input;

    bool do_global_search_and_local_refinement = false;
    int  current_class                         = 0;
    bool ignore_input_angles                   = false;
    bool defocus_bias                          = false;
    //	my_current_job.Reset(53);
    my_current_job.ManualSetArguments("ttttbttttiiffffffffffffifffffffffbbbbbbbbbbbbbbbibibb", input_particle_images.ToUTF8( ).data( ),
                                      input_star_filename.ToUTF8( ).data( ),
                                      input_reconstruction.ToUTF8( ).data( ),
                                      input_reconstruction_statistics.ToUTF8( ).data( ), use_statistics,
                                      ouput_matching_projections.ToUTF8( ).data( ),
                                      ouput_star_filename.ToUTF8( ).data( ),
                                      ouput_shift_filename.ToUTF8( ).data( ),
                                      my_symmetry.ToUTF8( ).data( ),
                                      first_particle, last_particle, percent_used, pixel_size,
                                      molecular_mass_kDa, inner_mask_radius, outer_mask_radius, low_resolution_limit,
                                      high_resolution_limit, signed_CC_limit, classification_resolution_limit,
                                      mask_radius_search, high_resolution_limit_search, angular_step, best_parameters_to_keep,
                                      max_search_x, max_search_y,
                                      mask_center_2d_x, mask_center_2d_y, mask_center_2d_z, mask_radius_2d,
                                      defocus_search_range, defocus_step, padding,
                                      do_global_search, do_local_refinement,
                                      refine_psi, refine_theta, refine_phi, refine_x, refine_y,
                                      calculate_matching_projections, apply_2D_masking, ctf_refinement, normalize_particles,
                                      invert_contrast, exclude_blank_edges, normalize_input_3d, threshold_input_3d,
                                      max_threads, do_global_search_and_local_refinement, current_class, ignore_input_angles, defocus_bias);
}

// override the do calculation method which will be what is actually run..

bool Refine3DApp::DoCalculation( ) {

    timer.start("refine3d setup");
    Particle refine_particle;
    Particle refine_particle_;
    Particle search_particle;
    Particle search_particle_;

    wxString input_particle_images           = my_current_job.arguments[0].ReturnStringArgument( ); // global
    wxString input_star_filename             = my_current_job.arguments[1].ReturnStringArgument( ); // not sure
    wxString input_reconstruction            = my_current_job.arguments[2].ReturnStringArgument( ); // global
    wxString input_reconstruction_statistics = my_current_job.arguments[3].ReturnStringArgument( ); // global
    bool     use_statistics                  = my_current_job.arguments[4].ReturnBoolArgument( ); // global
    wxString output_matching_projections     = my_current_job.arguments[5].ReturnStringArgument( ); // ignore (always false)
    wxString output_star_filename            = my_current_job.arguments[6].ReturnStringArgument( ); // not sure par file
    wxString output_shift_filename           = my_current_job.arguments[7].ReturnStringArgument( ); // not sure output
    wxString my_symmetry                     = my_current_job.arguments[8].ReturnStringArgument( ); // global
    int      first_particle                  = my_current_job.arguments[9].ReturnIntegerArgument( ); // local (effectively ignore)
    int      last_particle                   = my_current_job.arguments[10].ReturnIntegerArgument( ); // local (effectively ignore)
    float    percent_used                    = my_current_job.arguments[11].ReturnFloatArgument( );
    float    pixel_size                      = my_current_job.arguments[12].ReturnFloatArgument( ); // local
    //	float    voltage_kV							= my_current_job.arguments[13].ReturnFloatArgument(); // local
    //	float 	 spherical_aberration_mm			= my_current_job.arguments[14].ReturnFloatArgument(); // local
    //	float    amplitude_contrast					= my_current_job.arguments[15].ReturnFloatArgument(); // local
    float molecular_mass_kDa              = my_current_job.arguments[13].ReturnFloatArgument( ); // global
    float inner_mask_radius               = my_current_job.arguments[14].ReturnFloatArgument( ); // global
    float outer_mask_radius               = my_current_job.arguments[15].ReturnFloatArgument( ); // global
    float low_resolution_limit            = my_current_job.arguments[16].ReturnFloatArgument( ); // global
    float high_resolution_limit           = my_current_job.arguments[17].ReturnFloatArgument( ); // global
    float signed_CC_limit                 = my_current_job.arguments[18].ReturnFloatArgument( ); // global
    float classification_resolution_limit = my_current_job.arguments[19].ReturnFloatArgument( ); // global
    float mask_radius_search              = my_current_job.arguments[20].ReturnFloatArgument( ); // global
    float high_resolution_limit_search    = my_current_job.arguments[21].ReturnFloatArgument( ); // global
    float angular_step                    = my_current_job.arguments[22].ReturnFloatArgument( ); // global
    int   best_parameters_to_keep         = my_current_job.arguments[23].ReturnIntegerArgument( ); // global
    float max_search_x                    = my_current_job.arguments[24].ReturnFloatArgument( ); // global
    float max_search_y                    = my_current_job.arguments[25].ReturnFloatArgument( ); // global
    refine_particle.mask_center_2d_x      = my_current_job.arguments[26].ReturnFloatArgument( ); // global
    refine_particle.mask_center_2d_y      = my_current_job.arguments[27].ReturnFloatArgument( ); // global
    refine_particle.mask_center_2d_z      = my_current_job.arguments[28].ReturnFloatArgument( ); // global
    refine_particle.mask_radius_2d        = my_current_job.arguments[29].ReturnFloatArgument( ); // global
    float defocus_search_range            = my_current_job.arguments[30].ReturnFloatArgument( ); // global
    float defocus_step                    = my_current_job.arguments[31].ReturnFloatArgument( ); // global
    float padding                         = my_current_job.arguments[32].ReturnFloatArgument( ); // global
    //	float	 filter_constant					= my_current_job.arguments[35].ReturnFloatArgument();
    bool do_global_search    = my_current_job.arguments[33].ReturnBoolArgument( ); // global
    bool do_local_refinement = my_current_job.arguments[34].ReturnBoolArgument( ); // global
    // Psi, Theta, Phi, ShiftX, ShiftY
    refine_particle.parameter_map.psi          = my_current_job.arguments[35].ReturnBoolArgument( ); // global
    refine_particle.parameter_map.theta        = my_current_job.arguments[36].ReturnBoolArgument( ); // global
    refine_particle.parameter_map.phi          = my_current_job.arguments[37].ReturnBoolArgument( ); // global
    refine_particle.parameter_map.x_shift      = my_current_job.arguments[38].ReturnBoolArgument( ); // global
    refine_particle.parameter_map.y_shift      = my_current_job.arguments[39].ReturnBoolArgument( ); // global
    bool calculate_matching_projections        = my_current_job.arguments[40].ReturnBoolArgument( ); // global - but ignore
    refine_particle.apply_2D_masking           = my_current_job.arguments[41].ReturnBoolArgument( ); // global
    bool ctf_refinement                        = my_current_job.arguments[42].ReturnBoolArgument( ); // global
    bool normalize_particles                   = my_current_job.arguments[43].ReturnBoolArgument( );
    bool invert_contrast                       = my_current_job.arguments[44].ReturnBoolArgument( ); // global - but ignore
    bool exclude_blank_edges                   = my_current_job.arguments[45].ReturnBoolArgument( );
    bool normalize_input_3d                    = my_current_job.arguments[46].ReturnBoolArgument( );
    bool threshold_input_3d                    = my_current_job.arguments[47].ReturnBoolArgument( );
    int  max_threads                           = my_current_job.arguments[48].ReturnIntegerArgument( );
    bool do_global_search_and_local_refinement = my_current_job.arguments[49].ReturnBoolArgument( );
    int  current_class                         = my_current_job.arguments[50].ReturnIntegerArgument( ); // global - but ignore
    bool ignore_input_angles                   = my_current_job.arguments[51].ReturnBoolArgument( ); // during global search, ignore the starting parameters (this helps reduce bias)
    bool defocus_bias                          = my_current_job.arguments[52].ReturnBoolArgument( ); // during ab-initio 3D, biases random selection of particles towards higher defocus values

    if ( is_running_locally == false )
        max_threads = number_of_threads_requested_on_command_line; // OVERRIDE FOR THE GUI, AS IT HAS TO BE SET ON THE COMMAND LINE...

    // Constraints for phi, theta, psi not yet implemented
    refine_particle.constraints_used.phi     = false;
    refine_particle.constraints_used.theta   = false;
    refine_particle.constraints_used.psi     = false;
    refine_particle.constraints_used.x_shift = true;
    refine_particle.constraints_used.y_shift = true;

    Image                 input_image_;
    Image                 projection_image_;
    Image                 search_projection_image;
    Image                 binned_image;
    Image                 final_image;
    Image                 temp_image_;
    Image                 temp_image2_;
    Image                 sum_power;
    Image                 sum_power_local;
    Image*                projection_cache = NULL;
    CTF                   input_ctf;
    Image                 snr_image;
    ReconstructedVolume   input_3d;
    ReconstructedVolume   input_3d_;
    ReconstructedVolume   search_reference_3d;
    ReconstructedVolume   search_reference_3d_;
    ConjugateGradient     conjugate_gradient_minimizer;
    EulerSearch           global_euler_search, euler_search_;
    Curve                 noise_power_spectrum;
    Curve                 number_of_terms;
    RandomNumberGenerator random_particle(true);
    ProgressBar*          my_progress;

#ifdef ENABLEGPU
    GpuImage* gpu_projection_cache;
#else
    // Dummy for the OMP shared cla
    int gpu_projection_cache = 0;
#endif

    JobResult* intermediate_result;

    int i;
    int current_projection;
    int current_line;
    int current_line_;
    int images_to_process = 0;
    int image_counter;
    int defocus_i;
    int best_defocus_i;
    int random_reset_counter;
    int random_reset_count = 10;
    int binned_image_box_size;
    int search_box_size;
    int binned_search_image_box_size;
    int number_of_blank_edges;
    int number_of_blank_edges_local;
    int max_samples = 2000;
    int istart;
    int parameter_to_keep;

    cisTEMParameterLine input_parameters;
    cisTEMParameterLine output_parameters;
    cisTEMParameterLine search_parameters;
    cisTEMParameterLine parameter_average;
    cisTEMParameterLine parameter_variance;
    cisTEMParameterLine output_parameter_change;

    float cg_starting_point[17];
    float cg_accuracy[17];
    float gui_result_parameters[26];

    float binning_factor_refine;
    float binning_factor_search;
    float mask_falloff = 20.0; // in Angstrom
    float temp_float;
    float psi_max;
    float psi_step;
    float psi_start;
    float score;
    float best_score;
    float frealign_score_local;
    float mask_radius_for_noise;
    float percentage;
    float variance;
    float average;
    float average_density_max;
    float defocus_lower_limit;
    float defocus_upper_limit;
    float defocus_range_mean2;
    float defocus_range_std;
    float defocus_mean_score = 0.0;
    float defocus_score;
    float image_shift_x;
    float image_shift_y;
    //	float low_resolution_contrast = 0.5f;
    bool file_read;
    bool skip_local_refinement = false;

    bool take_random_best_parameter;

    if ( best_parameters_to_keep < 0 ) {
        best_parameters_to_keep    = -best_parameters_to_keep;
        take_random_best_parameter = true;
    }
    else {
        take_random_best_parameter = false;
    }

    wxDateTime my_time_in;

    ZeroFloatArray(cg_starting_point, 17);
    ZeroFloatArray(cg_accuracy, 17);
    long number_of_calls_to_score_function = 0;

    if ( (is_running_locally && ! DoesFileExist(input_star_filename)) || (! is_running_locally && ! DoesFileExistWithWait(input_star_filename, 90)) ) {
        SendErrorAndCrash(wxString::Format("Error: Input star file %s not found\n", input_star_filename));
    }
    if ( (is_running_locally && ! DoesFileExist(input_particle_images)) || (! is_running_locally && ! DoesFileExistWithWait(input_particle_images, 90)) ) {
        SendErrorAndCrash(wxString::Format("Error: Input particle stack %s not found\n", input_particle_images));
    }
    if ( (is_running_locally && ! DoesFileExist(input_reconstruction)) || (! is_running_locally && ! DoesFileExistWithWait(input_reconstruction, 90)) ) {
        SendErrorAndCrash(wxString::Format("Error: Input reconstruction %s not found\n", input_reconstruction));
    }
    //	wxPrintf("\nOpening input file %s.\n", input_parameter_file);

    //FrealignParameterFile input_par_file(input_parameter_file, OPEN_TO_READ);
    cisTEMParameters input_star_file;

    wxFileName star_filename(input_star_filename);
    if ( star_filename.GetExt( ) == "cistem" )
        input_star_file.ReadFromcisTEMBinaryFile(input_star_filename);
    else
        input_star_file.ReadFromcisTEMStarFile(input_star_filename);

    // Read whole parameter file to work out average values and variances

    parameter_average  = input_star_file.ReturnParameterAverages( );
    parameter_variance = input_star_file.ReturnParameterVariances( );

    defocus_lower_limit = 15000.0 * sqrtf(parameter_average.microscope_voltage_kv / 300.0);
    defocus_upper_limit = 25000.0 * sqrtf(parameter_average.microscope_voltage_kv / 300.0);
    defocus_range_mean2 = defocus_upper_limit + defocus_lower_limit;
    defocus_range_std   = 0.5 * (defocus_upper_limit - defocus_lower_limit);

    MRCFile input_stack(input_particle_images.ToStdString( ), false);

    if ( last_particle == 0 )
        last_particle = input_stack.ReturnZSize( );
    if ( first_particle == 0 )
        first_particle = 1;
    if ( last_particle > input_stack.ReturnZSize( ) )
        last_particle = input_stack.ReturnZSize( );

    for ( current_line = 0; current_line < input_star_file.ReturnNumberofLines( ); current_line++ ) {
        if ( input_star_file.ReturnPositionInStack(current_line) >= first_particle && input_star_file.ReturnPositionInStack(current_line) <= last_particle )
            images_to_process++;
    }

    //input_par_file.ReadFile(false, input_stack.ReturnZSize());'
#ifdef CISTEM_DETERMINISTIC_OUTCOME
    random_particle.SetSeed(0);
#else
    random_particle.SetSeed(int(10000.0 * fabsf(input_star_file.ReturnAverageSigma(true))) % 10000);
#endif
    if ( defocus_bias ) {
        float* buffer_array = new float[input_star_file.ReturnNumberofLines( )];
        for ( current_line = 0; current_line < input_star_file.ReturnNumberofLines( ); current_line++ ) {
            buffer_array[current_line] = expf(-powf(0.25 * (fabsf(input_star_file.ReturnDefocus1(current_line)) + fabsf(input_star_file.ReturnDefocus2(current_line)) - defocus_range_mean2) / defocus_range_std, 2.0));
            //			defocus_mean_score += expf(- powf(0.25 * (fabsf(input_par_file.ReadParameter(current_line, 8)) + fabsf(input_par_file.ReadParameter(current_line, 9)) - defocus_range_mean2) / defocus_range_std, 2.0));
            //			wxPrintf("df, score = %i %g %g\n", current_line, input_par_file.ReadParameter(current_line, 8), buffer_array[current_line]);
        }
        std::sort(buffer_array, buffer_array + input_star_file.ReturnNumberofLines( ) - 1);
        defocus_mean_score = buffer_array[input_star_file.ReturnNumberofLines( ) / 2];
        //		wxPrintf("median = %g\n", defocus_mean_score);
        //		defocus_mean_score /= current_line;
        delete[] buffer_array;
    }

    MRCFile  input_file(input_reconstruction.ToStdString( ), false, true);
    MRCFile* output_file;
    if ( percent_used < 1.0 && calculate_matching_projections ) {
        calculate_matching_projections = false;
        wxPrintf("\nPercent of particles used < 1, matching projections not calculated.\n");
    }
    if ( max_threads > 1 && calculate_matching_projections ) {
        calculate_matching_projections = false;
        wxPrintf("\nMatching projections not calculated when multi-threading.\n");
    }
    if ( calculate_matching_projections )
        output_file = new MRCFile(output_matching_projections.ToStdString( ), true);

    //FrealignParameterFile my_output_par_file(ouput_parameter_file, OPEN_TO_WRITE);
    //FrealignParameterFile my_output_par_shifts_file(ouput_shift_file, OPEN_TO_WRITE, 16);

    cisTEMParameters output_star_file;
    cisTEMParameters output_shifts_file;

    // allocate memory for output files..

    output_star_file.PreallocateMemoryAndBlank(input_star_file.ReturnNumberofLines( ));
    output_shifts_file.PreallocateMemoryAndBlank(input_star_file.ReturnNumberofLines( ));

    if ( input_stack.ReturnXSize( ) != input_stack.ReturnYSize( ) ) {
        input_stack.PrintInfo( );
        SendErrorAndCrash("Error: Particles are not square\n");
    }
    if ( (input_file.ReturnXSize( ) != input_file.ReturnYSize( )) || (input_file.ReturnXSize( ) != input_file.ReturnZSize( )) ) {
        input_file.PrintInfo( );
        SendErrorAndCrash("Error: Input reconstruction is not cubic\n");
    }
    if ( input_file.ReturnXSize( ) != input_stack.ReturnXSize( ) ) {
        input_file.PrintInfo( );
        input_stack.PrintInfo( );
        SendErrorAndCrash("Error: Dimension of particles and input reconstruction differ\n");
    }
    if ( last_particle < first_particle && last_particle != 0 ) {
        SendErrorAndCrash("Error: Number of last particle to refine smaller than number of first particle to refine\n");
    }
    // FIXME mask radius = 0 is supposed to give max, but that is not set. In Autorefine3d it is estimated as 0.8 * particle size which I guess is generally something like 0.8 * 1/2 box size so ~ 0.4 * box size
    if ( mask_radius_search == 0 ) {
        mask_radius_search = 0.4 * input_file.ReturnXSize( ) * pixel_size;
    }
    if ( max_search_x == 0.0 )
        max_search_x = mask_radius_search;
    if ( max_search_y == 0.0 )
        max_search_y = mask_radius_search;

    my_time_in = wxDateTime::Now( );

    output_star_file.AddCommentToHeader("# Refine3D run date and time:              " + my_time_in.FormatISOCombined(' '));
    output_star_file.AddCommentToHeader("# Input particle images:                   " + input_particle_images);
    output_star_file.AddCommentToHeader("# Input cisTEM parameter filename:         " + input_star_filename);
    output_star_file.AddCommentToHeader("# Input reconstruction:                    " + input_reconstruction);
    output_star_file.AddCommentToHeader("# Input data statistics:                   " + input_reconstruction_statistics);
    output_star_file.AddCommentToHeader("# Use statistics:                          " + BoolToYesNo(use_statistics));
    output_star_file.AddCommentToHeader("# Output matching projections:             " + output_matching_projections);
    output_star_file.AddCommentToHeader("# Output cisTEM parameter file:            " + output_star_filename);
    output_star_file.AddCommentToHeader("# Output cisTEM parameter changes:         " + output_shift_filename);
    output_star_file.AddCommentToHeader("# Particle symmetry:                       " + my_symmetry);
    output_star_file.AddCommentToHeader("# First particle to refine:                " + wxString::Format("%i", first_particle));
    output_star_file.AddCommentToHeader("# Last particle to refine:                 " + wxString::Format("%i", last_particle));
    output_star_file.AddCommentToHeader("# Percent of particles to refine:          " + wxString::Format("%f", percent_used));
    output_star_file.AddCommentToHeader("# Pixel size of reconstruction (A):        " + wxString::Format("%f", pixel_size));
    //	output_star_file.AddCommentToHeader("# Beam energy (keV):                       " + wxString::Format("%f", voltage_kV));
    //	output_star_file.AddCommentToHeader("# Spherical aberration (mm):               " + wxString::Format("%f", spherical_aberration_mm));
    //	output_star_file.AddCommentToHeader("# Amplitude contrast:                      " + wxString::Format("%f", amplitude_contrast));
    //	output_star_file.AddCommentToHeader("# Beam tilt in x (mrad):                   " + wxString::Format("%f", beam_tilt_x));
    //	output_star_file.AddCommentToHeader("# Beam tilt in y (mrad):                   " + wxString::Format("%f", beam_tilt_y));
    output_star_file.AddCommentToHeader("# Molecular mass of particle (kDa):        " + wxString::Format("%f", molecular_mass_kDa));
    output_star_file.AddCommentToHeader("# Inner mask radius for refinement (A):    " + wxString::Format("%f", inner_mask_radius));
    output_star_file.AddCommentToHeader("# Outer mask radius for refinement (A):    " + wxString::Format("%f", outer_mask_radius));
    output_star_file.AddCommentToHeader("# Low resolution limit (A):                " + wxString::Format("%f", low_resolution_limit));
    output_star_file.AddCommentToHeader("# High resolution limit (A):               " + wxString::Format("%f", high_resolution_limit));
    output_star_file.AddCommentToHeader("# Resolution limit for signed CC (A):      " + wxString::Format("%f", signed_CC_limit));
    output_star_file.AddCommentToHeader("# Res limit for classification (A):        " + wxString::Format("%f", classification_resolution_limit));
    output_star_file.AddCommentToHeader("# Mask radius for global search (A):       " + wxString::Format("%f", mask_radius_search));
    output_star_file.AddCommentToHeader("# Approx. resolution limit for search (A): " + wxString::Format("%f", high_resolution_limit_search));
    output_star_file.AddCommentToHeader("# Angular step:                            " + wxString::Format("%f", angular_step));
    output_star_file.AddCommentToHeader("# Number of top hits to refine:            " + wxString::Format("%i", best_parameters_to_keep));
    output_star_file.AddCommentToHeader("# Search range in X (A):                   " + wxString::Format("%f", max_search_x));
    output_star_file.AddCommentToHeader("# Search range in Y (A):                   " + wxString::Format("%f", max_search_y));
    output_star_file.AddCommentToHeader("# 2D mask X coordinate (A):                " + wxString::Format("%f", refine_particle.mask_center_2d_x));
    output_star_file.AddCommentToHeader("# 2D mask Y coordinate (A):                " + wxString::Format("%f", refine_particle.mask_center_2d_y));
    output_star_file.AddCommentToHeader("# 2D mask Z coordinate (A):                " + wxString::Format("%f", refine_particle.mask_center_2d_z));
    output_star_file.AddCommentToHeader("# 2D mask radius (A):                      " + wxString::Format("%f", refine_particle.mask_radius_2d));
    output_star_file.AddCommentToHeader("# Defocus search range (A):                " + wxString::Format("%f", defocus_search_range));
    output_star_file.AddCommentToHeader("# Defocus step (A):                        " + wxString::Format("%f", defocus_step));
    output_star_file.AddCommentToHeader("# Padding factor:                          " + wxString::Format("%f", padding));
    //	output_star_file.AddCommentToHeader("# Filter constant:                         " + wxString::Format("%f", filter_constant));
    output_star_file.AddCommentToHeader("# Global search:                           " + BoolToYesNo(do_global_search));
    output_star_file.AddCommentToHeader("# Local refinement:                        " + BoolToYesNo(do_local_refinement));
    output_star_file.AddCommentToHeader("# Refine Psi:                              " + BoolToYesNo(refine_particle.parameter_map.psi));
    output_star_file.AddCommentToHeader("# Refine Theta:                            " + BoolToYesNo(refine_particle.parameter_map.theta));
    output_star_file.AddCommentToHeader("# Refine Phi:                              " + BoolToYesNo(refine_particle.parameter_map.phi));
    output_star_file.AddCommentToHeader("# Refine ShiftX:                           " + BoolToYesNo(refine_particle.parameter_map.x_shift));
    output_star_file.AddCommentToHeader("# Refine ShiftY:                           " + BoolToYesNo(refine_particle.parameter_map.y_shift));
    output_star_file.AddCommentToHeader("# Calculate matching projections:          " + BoolToYesNo(calculate_matching_projections));
    output_star_file.AddCommentToHeader("# Apply 2D masking:                        " + BoolToYesNo(refine_particle.apply_2D_masking));
    output_star_file.AddCommentToHeader("# Refine defocus:                          " + BoolToYesNo(ctf_refinement));
    output_star_file.AddCommentToHeader("# Normalize particles:                     " + BoolToYesNo(normalize_particles));
    output_star_file.AddCommentToHeader("# Invert particle contrast:                " + BoolToYesNo(invert_contrast));
    output_star_file.AddCommentToHeader("# Exclude images with blank edges:         " + BoolToYesNo(exclude_blank_edges));
    output_star_file.AddCommentToHeader("# Normalize input reconstruction:          " + BoolToYesNo(normalize_input_3d));
    output_star_file.AddCommentToHeader("# Threshold input reconstruction:          " + BoolToYesNo(threshold_input_3d));
    output_star_file.AddCommentToHeader("#");

    if ( ! refine_particle.parameter_map.phi && ! refine_particle.parameter_map.theta && ! refine_particle.parameter_map.psi && ! refine_particle.parameter_map.x_shift && ! refine_particle.parameter_map.y_shift ) {
        // These are already set to false coming from the Autorefine3d panel
        // This overrides the yes/no from user input when run interactively.
        // This would also override the selection of global/local refinement int the manual Refine3d panel. FIXME: is this intended behavior?
        do_local_refinement = false;
        do_global_search    = false;
    }

    if ( do_global_search_and_local_refinement ) {
        // This would override the previous conditional on a nosearch run. As configured this would only happen when called from Autorefine3d.
        // FIXME: is this intended behavior?
        refine_particle.parameter_map.psi     = true;
        refine_particle.parameter_map.theta   = true;
        refine_particle.parameter_map.phi     = true;
        refine_particle.parameter_map.x_shift = true;
        refine_particle.parameter_map.y_shift = true;
        do_local_refinement                   = true;
        do_global_search                      = true;
    }

    if ( high_resolution_limit < 2.0 * pixel_size )
        high_resolution_limit = 2.0 * pixel_size;
    if ( classification_resolution_limit < 2.0 * pixel_size )
        classification_resolution_limit = 2.0 * pixel_size;
    if ( high_resolution_limit_search < 2.0 * pixel_size )
        high_resolution_limit_search = 2.0 * pixel_size;
    if ( signed_CC_limit == 0.0 )
        signed_CC_limit = pixel_size;

    if ( outer_mask_radius > float(input_stack.ReturnXSize( )) / 2.0 * pixel_size - mask_falloff )
        outer_mask_radius = float(input_stack.ReturnXSize( )) / 2.0 * pixel_size - mask_falloff;
    if ( mask_radius_search > float(input_stack.ReturnXSize( )) / 2.0 * pixel_size - mask_falloff )
        mask_radius_search = float(input_stack.ReturnXSize( )) / 2.0 * pixel_size - mask_falloff;

    input_3d.InitWithDimensions(input_file.ReturnXSize( ), input_file.ReturnYSize( ), input_file.ReturnZSize( ), pixel_size, my_symmetry);
    input_3d.molecular_mass_in_kDa = molecular_mass_kDa;
    ResolutionStatistics input_statistics(pixel_size, input_3d.density_map->logical_y_dimension);
    ResolutionStatistics search_statistics;
    ResolutionStatistics refine_statistics;
    if ( use_statistics ) {
        if ( ! DoesFileExist(input_reconstruction_statistics) ) {
            SendError(wxString::Format("Error: Input statistics %s not found\n", input_reconstruction_statistics));
            exit(-1);
        }
        input_statistics.ReadStatisticsFromFile(input_reconstruction_statistics);
        //		input_statistics.part_SSNR.PrintToStandardOut();
        input_statistics.RestrainParticleSSNR(10.0f);
        //		input_statistics.part_SSNR.PrintToStandardOut();
        //		exit(0);
    }
    else {
        wxPrintf("\nUsing default statistics\n");
        input_statistics.GenerateDefaultStatistics(molecular_mass_kDa);
    }
    refine_statistics = input_statistics;
    input_3d.density_map->ReadSlices(&input_file, 1, input_3d.density_map->logical_z_dimension);
    //!!! This line is incompatible with ML !!!
    //	input_3d.density_map->CosineMask(outer_mask_radius / pixel_size, mask_falloff / pixel_size);
    //	input_3d.density_map->AddConstant(- input_3d.density_map->ReturnAverageOfRealValuesOnEdges());
    // Remove masking here to avoid edge artifacts later
    input_3d.density_map->CosineMask(outer_mask_radius / pixel_size, mask_falloff / pixel_size, false, true, 0.0);
    if ( inner_mask_radius > 0.0 )
        input_3d.density_map->CosineMask(inner_mask_radius / pixel_size, mask_falloff / pixel_size, true);
    //	for (i = 0; i < input_3d.density_map->real_memory_allocated; i++) if (input_3d.density_map->real_values[i] < 0.0) input_3d.density_map->real_values[i] = -log(-input_3d.density_map->real_values[i] + 1.0);
    if ( threshold_input_3d ) {
        average_density_max = input_3d.density_map->ReturnAverageOfMaxN(100, outer_mask_radius / pixel_size);
        input_3d.density_map->SetMinimumValue(-0.3 * average_density_max);
        //		input_3d.density_map->SetMinimumValue(0.0);
    }

    //	input_image.Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), true);
    //	if (outer_mask_radius > input_image.physical_address_of_box_center_x * pixel_size- mask_falloff) outer_mask_radius = input_image.physical_address_of_box_center_x * pixel_size - mask_falloff;
    //	if (mask_radius_search > input_image.physical_address_of_box_center_x * pixel_size- mask_falloff) mask_radius_search = input_image.physical_address_of_box_center_x * pixel_size - mask_falloff;
    input_3d.mask_radius = outer_mask_radius;

    if ( do_global_search ) {
        if ( best_parameters_to_keep == 0 ) {
            best_parameters_to_keep = 1;
            skip_local_refinement   = true;
        }
        // Assume square particles
        search_reference_3d = input_3d;
        search_statistics   = input_statistics;
        search_box_size     = ReturnClosestFactorizedUpper(myroundint(2.0 / pixel_size * (std::max(max_search_x, max_search_y) + mask_radius_search)), 3, true);
        if ( search_box_size > search_reference_3d.density_map->logical_x_dimension )
            search_box_size = search_reference_3d.density_map->logical_x_dimension;
        if ( search_box_size != search_reference_3d.density_map->logical_x_dimension * padding )
            search_reference_3d.density_map->Resize(search_box_size * padding, search_box_size * padding, search_box_size * padding);
        if ( mask_radius_search > float(search_box_size) / 2.0 * pixel_size - mask_falloff )
            mask_radius_search = float(search_box_size) / 2.0 * pixel_size - mask_falloff;
        //		search_reference_3d.PrepareForProjections(high_resolution_limit_search, true);
        search_reference_3d.PrepareForProjections(low_resolution_limit, high_resolution_limit_search, true);
        //		search_statistics.Init(search_reference_3d.pixel_size, search_reference_3d.density_map->logical_y_dimension / 2 + 1);
        binning_factor_search        = search_reference_3d.pixel_size / pixel_size;
        binned_search_image_box_size = myroundint(search_reference_3d.density_map->logical_x_dimension / padding);
        //		search_particle.Allocate(binned_search_image_box_size, binned_search_image_box_size);
        //		search_projection_image.Allocate(search_reference_3d.density_map->logical_x_dimension, search_reference_3d.density_map->logical_y_dimension, false);
        //		temp_image2.Allocate(search_box_size, search_box_size, true);
        //Scale to make projections compatible with images for ML calculation
        // FIXME: this is just binning ^ 2/3 why is this, and why is it written this way
        search_reference_3d.density_map->MultiplyByConstant(powf(powf(binning_factor_search, 1.0 / 3.0), 2));
        //if (angular_step <= 0) angular_step = 360.0 * high_resolution_limit_search / PI / outer_mask_radius;
        if ( angular_step <= 0 )
            angular_step = CalculateAngularStep(high_resolution_limit_search, outer_mask_radius);
        // FIXME: This is in-principle doing the exact same thing as CalculateAngularStep, at least if we've binned to a pixel size = 2 * high_resolution_limit_search
        psi_step = rad_2_deg(search_reference_3d.pixel_size / outer_mask_radius);
        psi_step = 360.0 / int(360.0 / psi_step + 0.5);
#ifdef CISTEM_DETERMINISTIC_OUTCOME
        psi_start = 0.0;
#else
        psi_start             = psi_step / 2.0 * global_random_number_generator.GetUniformRandom( );
#endif

        psi_max = 0.0;
        if ( refine_particle.parameter_map.psi )
            psi_max = 360.0;
        wxPrintf("\nBox size for search = %i, binning factor = %f, new pixel size = %f, resolution limit = %f\nAngular step size = %f, in-plane = %f\n", search_reference_3d.density_map->logical_x_dimension, binning_factor_search, search_reference_3d.pixel_size, search_reference_3d.pixel_size * 2.0, angular_step, psi_step);
    }

    if ( padding != 1.0 ) {
        input_3d.density_map->Resize(input_3d.density_map->logical_x_dimension * padding, input_3d.density_map->logical_y_dimension * padding, input_3d.density_map->logical_z_dimension * padding, input_3d.density_map->ReturnAverageOfRealValuesOnEdges( ));
        //		refine_statistics.part_SSNR.ResampleCurve(&refine_statistics.part_SSNR, refine_statistics.part_SSNR.number_of_points * padding);
    }

    //	input_3d.PrepareForProjections(high_resolution_limit);
    input_3d.PrepareForProjections(low_resolution_limit, high_resolution_limit);
    // So we can share the global pointer

    binning_factor_refine = input_3d.pixel_size / pixel_size;
    binned_image_box_size = myroundint(input_stack.ReturnXSize( ) / binning_factor_refine);
    //Scale to make projections compatible with images for ML calculation
    //	input_3d.density_map->MultiplyByConstant(binning_factor_refine);
    //	input_3d.density_map->MultiplyByConstant(powf(powf(binning_factor_refine, 1.0 / 3.0), 2));
    wxPrintf("\nBinning factor for refinement = %f, new pixel size = %f\n", binning_factor_refine, input_3d.pixel_size);

    //	temp_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), true);
    sum_power.Allocate(input_stack.ReturnXSize( ), input_stack.ReturnYSize( ), false);
    //	refine_particle.Allocate(input_3d.density_map->logical_x_dimension, input_3d.density_map->logical_y_dimension);
    //	ctf_input_image.Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), false);
    //	projection_image.Allocate(input_3d.density_map->logical_x_dimension, input_3d.density_map->logical_y_dimension, false);
    //	unbinned_image.Allocate(input_file.ReturnXSize() * padding, input_file.ReturnYSize() * padding, true);
    //	if (ctf_refinement) binned_image.Allocate(input_3d.density_map->logical_x_dimension, input_3d.density_map->logical_y_dimension, false);
    //	final_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), true);

    if ( parameter_variance.phi < 0.001 )
        refine_particle.constraints_used.phi = false;
    if ( parameter_variance.theta < 0.001 )
        refine_particle.constraints_used.theta = false;
    if ( parameter_variance.psi < 0.001 )
        refine_particle.constraints_used.psi = false;
    if ( parameter_variance.x_shift < 0.001 )
        refine_particle.constraints_used.x_shift = false;
    if ( parameter_variance.y_shift < 0.001 )
        refine_particle.constraints_used.y_shift = false;

    refine_particle.SetParameterStatistics(parameter_average, parameter_variance);

    if ( normalize_particles ) {
        wxPrintf("Calculating noise power spectrum...\n\n");
        random_reset_count = std::max(random_reset_count, max_threads);
        percentage         = float(max_samples) / float(images_to_process) / random_reset_count;
        sum_power.SetToConstant(0.0f);
        number_of_blank_edges = 0;
        noise_power_spectrum.SetupXAxis(0.0f, 0.5f * sqrtf(2.0f), int((sum_power.logical_x_dimension / 2.0f + 1.0f) * sqrtf(2.0f) + 1.0f));
        number_of_terms.SetupXAxis(0.0f, 0.5f * sqrtf(2.0f), int((sum_power.logical_x_dimension / 2.0f + 1.0f) * sqrtf(2.0f) + 1.0f));
        if ( is_running_locally == true )
            my_progress = new ProgressBar(images_to_process / max_threads);
        current_line         = 0;
        random_reset_counter = 0;

#pragma omp parallel num_threads(max_threads) default(none) shared(timer, input_star_file, first_particle, last_particle, my_progress, percentage, exclude_blank_edges, input_stack,                                                                                                                                                                                                            \
                                                                   outer_mask_radius, mask_falloff, number_of_blank_edges, sum_power, current_line, global_random_number_generator, random_reset_count, random_reset_counter) private(current_line_, input_parameters, image_counter, number_of_blank_edges_local, variance, temp_image_, sum_power_local, input_image_, temp_float, file_read, \
                                                                                                                                                                                                                                      mask_radius_for_noise)
        {

            image_counter               = 0;
            number_of_blank_edges_local = 0;
            input_image_.Allocate(input_stack.ReturnXSize( ), input_stack.ReturnYSize( ), true);
            temp_image_.Allocate(input_stack.ReturnXSize( ), input_stack.ReturnYSize( ), true);
            sum_power_local.Allocate(input_stack.ReturnXSize( ), input_stack.ReturnYSize( ), false);
            sum_power_local.SetToConstant(0.0f);
            file_read = false;

#pragma omp for schedule(static, 1)
            for ( current_line_ = 0; current_line_ < input_star_file.ReturnNumberofLines( ); current_line_++ ) {
#pragma omp critical
                {
                    input_parameters = input_star_file.ReturnLine(current_line);

                    current_line++;
                    if ( input_parameters.position_in_stack >= first_particle && input_parameters.position_in_stack <= last_particle ) {
                        file_read = false;
                        if ( random_reset_counter == 0 ) {
                            // FIXME: I don't know what this is supposed to be doing, but if non-zero, it causes huge slow downs.
                            temp_float = 0.0f;
                            // #ifdef CISTEM_DETERMINISTIC_OUTCOME
                            //                             temp_float = 0.0f;
                            // #else
                            //                             temp_float = global_random_number_generator.GetUniformRandom( );
                        }
                        // #endif
                        if ( (temp_float >= 1.0 - 2.0f * percentage) || (random_reset_counter != 0) ) {
                            random_reset_counter++;
                            if ( random_reset_counter == random_reset_count )
                                random_reset_counter = 0;
                            //						wxPrintf("reading %i\n", int(input_parameters[0] + 0.5f));
                            input_image_.ReadSlice(&input_stack, input_parameters.position_in_stack);
                            file_read = true;
                        }
                    }
                }
                if ( input_parameters.position_in_stack < first_particle || input_parameters.position_in_stack > last_particle )
                    continue;
                image_counter++;

                if ( is_running_locally == true && ReturnThreadNumberOfCurrentThread( ) == 0 )
                    my_progress->Update(image_counter);

                if ( ! file_read )
                    continue;
                //			if ((temp_float < 1.0 - 2.0f * percentage) && (random_reset_counter == 0)) continue;
                //			if ((global_random_number_generator.GetUniformRandom() < 1.0 - 2.0f * percentage)) continue;
                //			input_image_.ReadSlice(&input_stack, int(input_parameters[0] + 0.5f));
                mask_radius_for_noise = outer_mask_radius / input_parameters.pixel_size;
                if ( 2.0 * mask_radius_for_noise + mask_falloff / input_parameters.pixel_size > 0.95f * input_stack.ReturnXSize( ) ) {
                    mask_radius_for_noise = 0.95f * input_stack.ReturnXSize( ) / 2.0f - mask_falloff / 2.0f / input_parameters.pixel_size;
                }
                if ( exclude_blank_edges && input_image_.ContainsBlankEdges(mask_radius_for_noise) ) {
                    number_of_blank_edges_local++;
                    continue;
                }
                variance = input_image_.ReturnVarianceOfRealValues(mask_radius_for_noise, 0.0f, 0.0f, 0.0f, true);
                if ( variance == 0.0f )
                    continue;
                input_image_.MultiplyByConstant(1.0f / sqrtf(variance));
                input_image_.CosineMask(mask_radius_for_noise, mask_falloff / input_parameters.pixel_size, true);
                input_image_.ForwardFFT( );
                temp_image_.CopyFrom(&input_image_);
                temp_image_.ConjugateMultiplyPixelWise(input_image_);
                sum_power_local.AddImage(&temp_image_);
            }

#pragma omp critical
            {
                number_of_blank_edges += number_of_blank_edges_local;
                sum_power.AddImage(&sum_power_local);
            }

            input_image_.Deallocate( );
            sum_power_local.Deallocate( );
            temp_image_.Deallocate( );

        } // end omp section

        if ( is_running_locally == true )
            delete my_progress;
        sum_power.Compute1DRotationalAverage(noise_power_spectrum, number_of_terms);
        noise_power_spectrum.SquareRoot( );
        noise_power_spectrum.Reciprocal( );

        if ( exclude_blank_edges ) {
            wxPrintf("\nImages with blank edges excluded from noise power calculation = %i\n", number_of_blank_edges);
        }
    }

    timer.lap("refine3d setup");

    if ( do_global_search ) {
        timer.start("generate projection cache for global search");
        //for (i = 0; i < search_particle.number_of_parameters; i++) {search_particle.parameter_map[i] = refine_particle.parameter_map[i];}
        search_particle.parameter_map = refine_particle.parameter_map;
        // Set parameter_map for x,y translations to true since they will always be searched and refined in a global search
        // Decided not to do this to honor user request
        //		search_particle.parameter_map[4] = true;
        //		search_particle.parameter_map[5] = true;
        search_particle.constraints_used = refine_particle.constraints_used;
        //for (i = 0; i < search_particle.number_of_parameters; i++) {search_particle.constraints_used[i] = refine_particle.constraints_used[i];}
        search_particle.SetParameterStatistics(parameter_average, parameter_variance);

        // Use projection_cache only if both phi and theta are searched; otherwise calculate projections on the fly
        if ( search_particle.parameter_map.phi && search_particle.parameter_map.theta ) {
            global_euler_search.InitGrid(my_symmetry, angular_step, 0.0, 0.0, psi_max, psi_step, psi_start, search_reference_3d.pixel_size / high_resolution_limit_search, search_particle.parameter_map, best_parameters_to_keep);
            if ( global_euler_search.best_parameters_to_keep != best_parameters_to_keep )
                best_parameters_to_keep = global_euler_search.best_parameters_to_keep;
            projection_cache = new Image[global_euler_search.number_of_search_positions];
            for ( i = 0; i < global_euler_search.number_of_search_positions; i++ ) {
                projection_cache[i].Allocate(binned_search_image_box_size, binned_search_image_box_size, false);
            }
            search_reference_3d.density_map->GenerateReferenceProjections(projection_cache, global_euler_search, search_reference_3d.pixel_size / high_resolution_limit_search);
            wxPrintf("\nNumber of global search views = %i (best_parameters to keep = %i)\n", global_euler_search.number_of_search_positions, global_euler_search.best_parameters_to_keep);

#ifdef ENABLEGPU
            gpu_projection_cache = new GpuImage[global_euler_search.number_of_search_positions];
            for ( i = 0; i < global_euler_search.number_of_search_positions; i++ ) {
                gpu_projection_cache[i].Init(projection_cache[i], true, true);
                if ( i == global_euler_search.number_of_search_positions - 1 ) {
                    // block on the last copy to let everything complete.
                    gpu_projection_cache[i].CopyHostToDevice(true);
                }
                else {
                    gpu_projection_cache[i].CopyHostToDevice(false);
                }
            }
#endif
        }

        //		search_projection_image.RotateFourier2DGenerateIndex(kernel_index, psi_max, psi_step, psi_start);

        if ( search_particle.parameter_map.x_shift )
            global_euler_search.max_search_x = max_search_x;
        else
            global_euler_search.max_search_x = 0.0;
        if ( search_particle.parameter_map.y_shift )
            global_euler_search.max_search_y = max_search_y;
        else
            global_euler_search.max_search_y = 0.0;
        timer.lap("generate projection cache for global search");
    }

    wxPrintf("\nAverage sigma noise = %f, average LogP = %f\nAverage ShiftX = %f, average ShiftY = %f\nSigma ShiftX = %f, sigma ShiftY = %f\n\nNumber of particles to refine = %i\n\n",
             parameter_average.sigma, parameter_average.score, parameter_average.x_shift, parameter_average.y_shift, sqrtf(parameter_variance.x_shift), sqrtf(parameter_variance.y_shift), images_to_process);

    if ( is_running_locally == true )
        my_progress = new ProgressBar(images_to_process / max_threads);

    current_projection = 0;

#pragma omp parallel num_threads(max_threads) default(none) shared(timer, parameter_average, input_3d, input_star_file, input_stack, max_threads, search_particle,                                                                                                                                                                                                   \
                                                                   first_particle, last_particle, invert_contrast, normalize_particles, noise_power_spectrum, padding, ctf_refinement, defocus_search_range, defocus_step, normalize_input_3d,                                                                                                                       \
                                                                   refine_statistics, pixel_size, my_progress, outer_mask_radius, mask_falloff, high_resolution_limit, molecular_mass_kDa, percent_used, output_shifts_file, do_local_refinement,                                                                                                                    \
                                                                   binning_factor_refine, low_resolution_limit, input_statistics, output_star_file, current_projection, do_global_search_and_local_refinement, signed_CC_limit, defocus_bias,                                                                                                                        \
                                                                   random_particle, defocus_range_mean2, defocus_range_std, defocus_mean_score, current_class, mask_radius_search, search_reference_3d, high_resolution_limit_search,                                                                                                                                \
                                                                   binning_factor_search, search_statistics, search_box_size, projection_cache, gpu_projection_cache, my_symmetry, angular_step, psi_max, psi_step, psi_start, take_random_best_parameter, refine_particle,                                                                                          \
                                                                   skip_local_refinement, calculate_matching_projections, classification_resolution_limit, output_file, best_parameters_to_keep, ignore_input_angles, global_random_number_generator,                                                                                                                \
                                                                   global_euler_search, binned_image_box_size, binned_search_image_box_size, do_global_search, number_of_calls_to_score_function) private(image_counter, refine_particle_, current_line_, input_parameters, temp_float, output_parameters, input_ctf, variance, average,                             \
                                                                                                                                                                                                          best_score, defocus_i, score, cg_starting_point, input_image_, search_particle_, intermediate_result, gui_result_parameters, image_shift_x, image_shift_y, \
                                                                                                                                                                                                          binned_image, projection_image_, best_defocus_i, defocus_score, temp_image2_, search_projection_image, cg_accuracy, search_reference_3d_,                  \
                                                                                                                                                                                                          temp_image_, search_parameters, istart, parameter_to_keep, conjugate_gradient_minimizer, i, final_image, input_3d_, euler_search_, frealign_score_local)
    { // for omp

        timer.start("omp copy to local variables");

        ProjectionComparisonObjects comparison_object;
        BatchSizeOptimizer          batch_size_optimizer;
#ifdef ENABLEGPU
        GpuImage* use_this_cache = gpu_projection_cache;
#else
        Image* use_this_cache = projection_cache;
#endif

        int  new_buffer_size                         = 0;
        long number_of_calls_to_score_function_local = 0;

        //	input_3d_ = input_3d;
        input_3d_.CopyAllButVolume(&input_3d);
        input_3d_.density_map = input_3d.density_map;

        // TP = Thread Private, originally called "local" which was super confusing with other booleans
        bool do_global_search_    = do_global_search;
        bool do_local_refinement_ = do_local_refinement;

        refine_particle_.CopyAllButImages(&refine_particle);
        refine_particle_.Allocate(binned_image_box_size, binned_image_box_size);
        //	refine_particle_.SetParameterStatistics(parameter_average, parameter_variance);

        input_image_.Allocate(input_stack.ReturnXSize( ), input_stack.ReturnYSize( ), true);
        temp_image_.Allocate(input_stack.ReturnXSize( ), input_stack.ReturnYSize( ), true);
        projection_image_.Allocate(binned_image_box_size, binned_image_box_size, false);

        if ( ctf_refinement && high_resolution_limit <= 20.0 )
            binned_image.Allocate(binned_image_box_size, binned_image_box_size, false);

        if ( do_global_search_ || do_global_search_and_local_refinement ) {
            search_reference_3d_.CopyAllButVolume(&search_reference_3d);
            search_reference_3d_.density_map = search_reference_3d.density_map;
            comparison_object.PrepareGpuVolumeProjection(search_reference_3d_, true);

            search_particle_.CopyAllButImages(&search_particle);
            search_particle_.Allocate(binned_search_image_box_size, binned_search_image_box_size);
            search_projection_image.Allocate(binned_search_image_box_size, binned_search_image_box_size, false);
            temp_image2_.Allocate(search_box_size, search_box_size, true);

            euler_search_ = global_euler_search;
        }

        if ( calculate_matching_projections )
            final_image.Allocate(input_stack.ReturnXSize( ), input_stack.ReturnYSize( ), true);

        // ifndef ENABLEGPU this is a noop

        // Try only thread 0 and sharing the object
        comparison_object.PrepareGpuVolumeProjection(input_3d_, false);
        // #pragma omp barrier

        image_counter = 0;
        timer.lap("omp copy to local variables");
#pragma omp for schedule(dynamic, 1)
        for ( current_line_ = 0; current_line_ < input_star_file.ReturnNumberofLines( ); current_line_++ ) {

            input_parameters = input_star_file.ReturnLine(current_line_);
            if ( input_parameters.position_in_stack < first_particle || input_parameters.position_in_stack > last_particle )
                continue;

// FIXME: This comment does not make sense, the position in the stack is determined by current_line_, which is assigned to given thread, how could this create a problem?
// ReadSlice requires omp critical to avoid parallel reads, which may lead to the wrong slice being read
#pragma omp critical
            input_image_.ReadSlice(&input_stack, input_parameters.position_in_stack);
            MyDebugAssertFalse(input_image_.HasNan( ), "Input image read from disk has NaN. Position in stack = %i\n", input_parameters.position_in_stack);

            image_counter++;

            output_parameters = input_parameters;
#ifdef CISTEM_DETERMINISTIC_OUTCOME
            temp_float = 0.0f;
#else
            temp_float = random_particle.GetUniformRandom( );
#endif
            if ( defocus_bias ) {
                defocus_score = expf(-powf(0.25 * (fabsf(input_parameters.defocus_1) + fabsf(input_parameters.defocus_2) - defocus_range_mean2) / defocus_range_std, 2.0));
                temp_float *= defocus_score / defocus_mean_score;
            }

            if ( temp_float < 1.0 - 2.0 * percent_used ) {
                input_parameters.image_is_active = -1; //- fabsf(input_parameters[7]);
                input_parameters.score_change    = 0.0;

                //my_output_par_file.WriteLine(input_parameters);
                output_star_file.all_parameters[current_line_] = input_parameters;

                if ( is_running_locally == false ) // send results back to the gui..
                {
                    intermediate_result             = new JobResult;
                    intermediate_result->job_number = my_current_job.job_number;

                    gui_result_parameters[0]  = current_class;
                    gui_result_parameters[1]  = input_parameters.position_in_stack;
                    gui_result_parameters[2]  = input_parameters.image_is_active;
                    gui_result_parameters[3]  = input_parameters.psi;
                    gui_result_parameters[4]  = input_parameters.theta;
                    gui_result_parameters[5]  = input_parameters.phi;
                    gui_result_parameters[6]  = input_parameters.x_shift;
                    gui_result_parameters[7]  = input_parameters.y_shift;
                    gui_result_parameters[8]  = input_parameters.defocus_1;
                    gui_result_parameters[9]  = input_parameters.defocus_2;
                    gui_result_parameters[10] = input_parameters.defocus_angle;
                    gui_result_parameters[11] = input_parameters.phase_shift;
                    gui_result_parameters[12] = input_parameters.occupancy;
                    gui_result_parameters[13] = input_parameters.logp;
                    gui_result_parameters[14] = input_parameters.sigma;
                    gui_result_parameters[15] = input_parameters.score;
                    gui_result_parameters[16] = input_parameters.score_change;
                    gui_result_parameters[17] = input_parameters.pixel_size;
                    gui_result_parameters[18] = input_parameters.microscope_voltage_kv;
                    gui_result_parameters[19] = input_parameters.microscope_spherical_aberration_mm;
                    gui_result_parameters[20] = input_parameters.beam_tilt_x;
                    gui_result_parameters[21] = input_parameters.beam_tilt_y;
                    gui_result_parameters[22] = input_parameters.image_shift_x;
                    gui_result_parameters[23] = input_parameters.image_shift_y;
                    gui_result_parameters[24] = input_parameters.amplitude_contrast;
                    gui_result_parameters[25] = input_parameters.assigned_subset;

                    intermediate_result->SetResult(26, gui_result_parameters);
                    AddJobToResultQueue(intermediate_result);
                }

                output_parameters.SetAllToZero( );
                //for (i = 1; i < refine_particle_.number_of_parameters; i++) output_parameters[i] = 0.0;
                output_parameters.position_in_stack              = input_parameters.position_in_stack;
                output_shifts_file.all_parameters[current_line_] = output_parameters;
                if ( is_running_locally == true && ReturnThreadNumberOfCurrentThread( ) == 0 )
                    my_progress->Update(image_counter);
                continue;
            }
            //		output_parameters = input_parameters;

            // This seems to be hack to toggle global search on/off from Autorefine3d
            if ( do_global_search_and_local_refinement ) {
                if ( input_parameters.image_is_active == 0.0 ) {
                    do_local_refinement_ = false;
                    do_global_search_    = true;
                }
                else {
                    do_local_refinement_ = true;
                    do_global_search_    = false;
                }
            }

            // Set up Particle object
            refine_particle_.ResetImageFlags( );
            refine_particle_.mask_radius  = outer_mask_radius;
            refine_particle_.mask_falloff = mask_falloff;
            //		refine_particle_.filter_radius_low = low_resolution_limit;
            refine_particle_.filter_radius_high = high_resolution_limit;
            refine_particle_.molecular_mass_kDa = molecular_mass_kDa;

            if ( do_global_search_and_local_refinement == true && do_global_search_ == true )
                refine_particle_.signed_CC_limit = pixel_size;
            else
                refine_particle_.signed_CC_limit = signed_CC_limit;

            // The following line would allow using particles with different pixel sizes
            refine_particle_.pixel_size    = input_3d_.pixel_size;
            refine_particle_.is_normalized = normalize_particles;
            refine_particle_.sigma_noise   = input_parameters.sigma / binning_factor_refine;
            //		refine_particle_.logp = -std::numeric_limits<float>::max();
            refine_particle_.SetParameters(input_parameters);

            comparison_object.SetInitialAnglesAndShifts(refine_particle_);

            if ( refine_particle_.parameter_map.x_shift )
                image_shift_x = 0.0f;
            else
                image_shift_x = input_parameters.image_shift_x;
            if ( refine_particle_.parameter_map.y_shift )
                image_shift_y = 0.0f;
            else
                image_shift_y = input_parameters.image_shift_y;

            refine_particle_.MapParameterAccuracy(cg_accuracy);
            //		refine_particle_.SetIndexForWeightedCorrelation();
            refine_particle_.SetParameterConstraints(powf(parameter_average.sigma, 2));

            input_ctf.Init(input_parameters.microscope_voltage_kv, input_parameters.microscope_spherical_aberration_mm, input_parameters.amplitude_contrast, input_parameters.defocus_1, input_parameters.defocus_2, input_parameters.defocus_angle, 0.0, 0.0, 0.0, pixel_size, input_parameters.phase_shift, input_parameters.beam_tilt_x / 1000.0f, input_parameters.beam_tilt_y / 1000.0f, image_shift_x, image_shift_y);
            //		input_ctf.SetLowResolutionContrast(low_resolution_contrast);
            //		ctf_input_image.CalculateCTFImage(input_ctf);
            //		refine_particle_.is_phase_flipped = true;

            //		input_image_.ReadSlice(&input_stack, int(input_parameters.position_in_stack + 0.5));
            timer.start("normalize particles");
            // FIXME: This std dev should be in cistem_constants.h
            input_image_.ReplaceOutliersWithMean(5.0);
            if ( invert_contrast )
                input_image_.InvertRealValues( );
            if ( normalize_particles ) {
                input_image_.ChangePixelSize(&input_image_, pixel_size / input_parameters.pixel_size, 0.001f, true);
                //			input_image_.ForwardFFT();
                // Whiten noise
                input_image_.ApplyCurveFilterUninterpolated(&noise_power_spectrum);
                //			input_image_.ApplyCurveFilter(&noise_power_spectrum);
                // Apply cosine filter to reduce ringing
                //			input_image_.CosineMask(std::max(pixel_size / high_resolution_limit, pixel_size / 7.0f + pixel_size / mask_falloff) - pixel_size / (2.0 * mask_falloff), pixel_size / mask_falloff);
                input_image_.BackwardFFT( );
                // Normalize background variance and average
                variance = input_image_.ReturnVarianceOfRealValues(input_image_.physical_address_of_box_center_x - mask_falloff / pixel_size, 0.0, 0.0, 0.0, true);
                average  = input_image_.ReturnAverageOfRealValues(input_image_.physical_address_of_box_center_x - mask_falloff / pixel_size, true);

                if ( variance == 0.0f )
                    input_image_.SetToConstant(0.0f);
                else
                    input_image_.AddMultiplyConstant(-average, 1.0 / sqrtf(variance));
                // At this point, input_image_ should have white background with a variance of 1. The variance should therefore be about 1/binning_factor^2 after binning.
            }
            else
                input_image_.ChangePixelSize(&input_image_, pixel_size / input_parameters.pixel_size, 0.001f);

            timer.lap("normalize particles");
            // Option to add noise to images to get out of local optima
            //		input_image_.AddGaussianNoise(sqrtf(2.0 * input_image_.ReturnVarianceOfRealValues()));

            temp_image_.CopyFrom(&input_image_);
            temp_image_.ForwardFFT( );
            temp_image_.ClipInto(refine_particle_.particle_image);
            refine_particle_.RecordNewImageData( );
            search_particle_.RecordNewImageData( );
            refine_particle_.use_half_precision_where_possible = true;
            search_particle_.use_half_precision_where_possible = true;
            // Multiply by binning_factor so variance after binning is close to 1.
            //		refine_particle_.particle_image->MultiplyByConstant(binning_factor_refine);

            comparison_object.SetHostPointers(&input_3d_, &projection_image_, &refine_particle_, false);

            refine_particle_.MapParameters(cg_starting_point);

            refine_particle_.PhaseShiftInverse( );

            // FIXME: this should be in cistem constants
            if ( ctf_refinement && high_resolution_limit <= 20.0 ) {
                timer.start("ctf setup");
                //			wxPrintf("\nRefining defocus for parameter line %i\n", current_line);
                // FIXME: this should be in cistem constants
                refine_particle_.filter_radius_low = 30.0;

                new_buffer_size = refine_particle_.SetIndexForWeightedCorrelation( );
                comparison_object.AllocateBuffers(new_buffer_size);

                // FIXME: NAMING Binned image isn't really binned, but it is being used to keep a clean copy of the image
                binned_image.CopyFrom(refine_particle_.particle_image);
                comparison_object.GetCleanCopyOfParticleImage(false);
                refine_particle_.InitCTF(input_parameters.microscope_voltage_kv, input_parameters.microscope_spherical_aberration_mm, input_parameters.amplitude_contrast, input_parameters.defocus_1, input_parameters.defocus_2, input_parameters.defocus_angle, input_parameters.phase_shift, input_parameters.beam_tilt_x / 1000.0f, input_parameters.beam_tilt_y / 1000.0f, image_shift_x, image_shift_y);
                best_score = -std::numeric_limits<float>::max( );

                timer.lap("ctf setup");
                for ( defocus_i = -myround(float(defocus_search_range) / float(defocus_step)); defocus_i <= myround(float(defocus_search_range) / float(defocus_step)); defocus_i++ ) {

                    refine_particle_.SetDefocus(input_parameters.defocus_1 + defocus_i * defocus_step, input_parameters.defocus_2 + defocus_i * defocus_step, input_parameters.defocus_angle, input_parameters.phase_shift);
                    refine_particle_.InitCTFImage(input_parameters.microscope_voltage_kv, input_parameters.microscope_spherical_aberration_mm, input_parameters.amplitude_contrast, input_parameters.defocus_1 + defocus_i * defocus_step, input_parameters.defocus_2 + defocus_i * defocus_step, input_parameters.defocus_angle, input_parameters.phase_shift, input_parameters.beam_tilt_x / 1000.0f, input_parameters.beam_tilt_y / 1000.0f, image_shift_x, image_shift_y);

                    timer.start("ctf ssnr");
                    if ( normalize_input_3d )
                        refine_particle_.WeightBySSNR(refine_statistics.part_SSNR, 1);
                    //				// Apply SSNR weighting only to image since input 3D map assumed to be calculated from correctly whitened images
                    else
                        refine_particle_.WeightBySSNR(refine_statistics.part_SSNR, 0);
                    timer.lap("ctf ssnr");
                    timer.start("ctf image proc");
                    refine_particle_.PhaseFlipImage( );
                    refine_particle_.BeamTiltMultiplyImage( );
                    //				refine_particle_.CosineMask(false, true, 0.0);
                    refine_particle_.CosineMask( );
                    refine_particle_.PhaseShift( );
                    refine_particle_.CenterInCorner( );
                    timer.lap("ctf image proc");
                    //				refine_particle_.WeightBySSNR(input_3d_.statistics.part_SSNR, 1);

                    // ifndef ENABLEGPU this is a noop
                    // because this is the first encounter on this loop, we need to make sure the host

                    comparison_object.PrepareGpuImages(refine_particle_, projection_image_, false);
                    comparison_object.PrepareGpuCTFImages(refine_particle_, false);

                    timer.start("ctf score");
                    score = -100.0 * FrealignObjectiveFunction(&comparison_object, cg_starting_point);
                    timer.lap("ctf score");
#ifdef PRINT_SCORES
                    wxPrintf("Score is %f from line %i\n", score, __LINE__);
#endif
                    if ( score > best_score ) {
                        best_score     = score;
                        best_defocus_i = defocus_i;
                        //					wxPrintf("Parameter line = %i, Defocus = %f, score = %g\n", current_line, defocus_i * defocus_step, score);
                    }
                    refine_particle_.particle_image->CopyFrom(&binned_image);
                    comparison_object.ResetCleanCopyOfParticleImage(false);
                    refine_particle_.is_ssnr_filtered   = false;
                    refine_particle_.is_masked          = false;
                    refine_particle_.is_centered_in_box = true;
                    refine_particle_.shift_counter      = 1;
                }

                // Clean up the extra GPU memory for the clean copy
                comparison_object.DeallocateCleanCopyOfParticleImage( );
                output_parameters.defocus_1 = input_parameters.defocus_1 + best_defocus_i * defocus_step;
                output_parameters.defocus_2 = input_parameters.defocus_2 + best_defocus_i * defocus_step;
                refine_particle_.SetDefocus(output_parameters.defocus_1, output_parameters.defocus_2, input_parameters.defocus_angle, input_parameters.phase_shift);
                refine_particle_.InitCTFImage(input_parameters.microscope_voltage_kv, input_parameters.microscope_spherical_aberration_mm, input_parameters.amplitude_contrast, output_parameters.defocus_1, output_parameters.defocus_2, input_parameters.defocus_angle, input_parameters.phase_shift, input_parameters.beam_tilt_x / 1000.0f, input_parameters.beam_tilt_y / 1000.0f, image_shift_x, image_shift_y);
            }
            else {
                refine_particle_.InitCTFImage(input_parameters.microscope_voltage_kv, input_parameters.microscope_spherical_aberration_mm, input_parameters.amplitude_contrast, input_parameters.defocus_1, input_parameters.defocus_2, input_parameters.defocus_angle, input_parameters.phase_shift, input_parameters.beam_tilt_x / 1000.0f, input_parameters.beam_tilt_y / 1000.0f, image_shift_x, image_shift_y);
            }
            //		refine_particle_.SetLowResolutionContrast(low_resolution_contrast);

            timer.start("weight and prep particle");
            refine_particle_.filter_radius_low = low_resolution_limit;
            new_buffer_size                    = refine_particle_.SetIndexForWeightedCorrelation( );
            // FIXME: could this be done outside the loop? ALso double check that it is even being used.
            comparison_object.AllocateBuffers(new_buffer_size);

            if ( normalize_input_3d )
                refine_particle_.WeightBySSNR(refine_statistics.part_SSNR, 1);
            // Apply SSNR weighting only to image since input 3D map assumed to be calculated from correctly whitened images
            else
                refine_particle_.WeightBySSNR(refine_statistics.part_SSNR, 0);
            refine_particle_.PhaseFlipImage( );
            refine_particle_.BeamTiltMultiplyImage( );
            //		refine_particle_.CosineMask(false, true, 0.0);
            refine_particle_.CosineMask( );
            refine_particle_.PhaseShift( );
            refine_particle_.CenterInCorner( );
            timer.lap("weight and prep particle");
            //		refine_particle_.WeightBySSNR(input_3d_.statistics.part_SSNR, 1);

            //		input_parameters[15] = 10.0;
            //		input_parameters.score = - 100.0 * FrealignObjectiveFunction(&comparison_object, cg_starting_point);

            comparison_object.PrepareGpuImages(refine_particle_, projection_image_, false);
            comparison_object.PrepareGpuCTFImages(refine_particle_, false);

            if ( (refine_particle_.number_of_search_dimensions > 0) && (do_global_search_ || do_local_refinement_) ) {
                timer.start("refining search FrealignObjFunct 1");
                input_parameters.score = -100.0 * FrealignObjectiveFunction(&comparison_object, cg_starting_point);
#ifdef PRINT_SCORES
                wxPrintf("Starting score is %g\n", input_parameters.score);
#endif
                timer.lap("refining search FrealignObjFunct 1");
                if ( do_global_search_ ) {
                    timer.start("global search local");
                    //				my_time_in = wxDateTime::UNow();
                    search_particle_.ResetImageFlags( );
                    search_particle_.pixel_size = search_reference_3d_.pixel_size;
                    if ( mask_radius_search == 0.0 ) {
                        search_particle_.mask_radius = search_particle_.particle_image->logical_x_dimension / 2 * search_particle_.pixel_size - mask_falloff;
                    }
                    else {
                        search_particle_.mask_radius = mask_radius_search;
                    }
                    search_particle_.mask_falloff       = mask_falloff;
                    search_particle_.filter_radius_low  = 0.0;
                    search_particle_.filter_radius_high = high_resolution_limit_search;
                    search_particle_.molecular_mass_kDa = molecular_mass_kDa;
                    search_particle_.signed_CC_limit    = signed_CC_limit;
                    search_particle_.sigma_noise        = input_parameters.sigma / binning_factor_search;
                    //				search_particle_.logp = -std::numeric_limits<float>::max();
                    search_particle_.SetParameters(input_parameters);
                    search_particle_.number_of_search_dimensions = refine_particle_.number_of_search_dimensions;
                    search_particle_.InitCTFImage(input_parameters.microscope_voltage_kv, input_parameters.microscope_spherical_aberration_mm, input_parameters.amplitude_contrast, input_parameters.defocus_1, input_parameters.defocus_2, input_parameters.defocus_angle, input_parameters.phase_shift, input_parameters.beam_tilt_x / 1000.0f, input_parameters.beam_tilt_y / 1000.0f, image_shift_x, image_shift_y);
                    //				search_particle_.SetLowResolutionContrast(low_resolution_contrast);
                    temp_image_.CopyFrom(&input_image_);
                    // Multiply by binning_factor so variance after binning is close to 1.
                    //				temp_image_.MultiplyByConstant(binning_factor_search);
                    // Assume square images
                    if ( search_box_size != temp_image_.logical_x_dimension ) {
                        temp_image_.ClipInto(&temp_image2_);
                        temp_image2_.ForwardFFT( );
                        temp_image2_.ClipInto(search_particle_.particle_image);
                    }
                    else {
                        temp_image_.ForwardFFT( );
                        temp_image_.ClipInto(search_particle_.particle_image);
                    }

                    // TODO: is this why the conj is taken of the particle and not the reference during the grid search (which would invert this shift as well?)
                    search_particle_.PhaseShiftInverse( );
                    // Always apply particle SSNR weighting (i.e. whitening) reference normalization since reference
                    // projections will not have SSNR (i.e. CTF-dependent) weighting applied
                    search_particle_.WeightBySSNR(search_statistics.part_SSNR, 1);
                    search_particle_.PhaseFlipImage( );
                    search_particle_.BeamTiltMultiplyImage( );
                    //				search_particle_.CosineMask(false, true, 0.0);
                    search_particle_.CosineMask( );
                    search_particle_.PhaseShift( );
                    //				search_particle_.CenterInCorner();
                    //				search_particle_.WeightBySSNR(search_reference_3d_.statistics.part_SSNR);

                    if ( search_particle_.parameter_map.phi && ! search_particle_.parameter_map.theta ) {
                        euler_search_.InitGrid(my_symmetry, angular_step, 0.0, input_parameters.theta, psi_max, psi_step, psi_start, search_reference_3d_.pixel_size / high_resolution_limit_search, search_particle_.parameter_map, best_parameters_to_keep);
                        if ( euler_search_.best_parameters_to_keep != best_parameters_to_keep )
                            best_parameters_to_keep = euler_search_.best_parameters_to_keep;
                        if ( ! search_particle_.parameter_map.phi )
                            euler_search_.psi_start = 360.0 - input_parameters.phi;
                        wanted_batch_size = batch_size_optimizer.start( );
                        euler_search_.Run(search_particle_, *search_reference_3d_.density_map, use_this_cache, wanted_batch_size);
                        batch_size_optimizer.update_batch_size( );
                    }
                    else if ( ! search_particle_.parameter_map.phi && search_particle_.parameter_map.theta ) {
                        euler_search_.InitGrid(my_symmetry, angular_step, input_parameters.psi, 0.0, psi_max, psi_step, psi_start, search_reference_3d_.pixel_size / high_resolution_limit_search, search_particle_.parameter_map, best_parameters_to_keep);
                        if ( euler_search_.best_parameters_to_keep != best_parameters_to_keep )
                            best_parameters_to_keep = euler_search_.best_parameters_to_keep;
                        if ( ! search_particle_.parameter_map.psi )
                            euler_search_.psi_start = 360.0 - input_parameters.phi;
                        wanted_batch_size = batch_size_optimizer.start( );
                        euler_search_.Run(search_particle_, *search_reference_3d_.density_map, use_this_cache, wanted_batch_size);
                        batch_size_optimizer.update_batch_size( );
                    }
                    else if ( search_particle_.parameter_map.phi && search_particle_.parameter_map.theta ) {
                        if ( ! search_particle_.parameter_map.psi )
                            euler_search_.psi_start = 360.0 - input_parameters.phi;
                        if ( euler_search_.best_parameters_to_keep != best_parameters_to_keep )
                            best_parameters_to_keep = euler_search_.best_parameters_to_keep;
                        //					for (i = 0; i < euler_search_.number_of_search_positions; i++) {projection_cache[i].SwapRealSpaceQuadrants(); projection_cache[i].QuickAndDirtyWriteSlice("projection.mrc", i + 1);}
                        //					search_particle_.particle_image->QuickAndDirtyWriteSlice("particle_image.mrc", 1);
                        //					exit(0);
                        wanted_batch_size = batch_size_optimizer.start( );
                        euler_search_.Run(search_particle_, *search_reference_3d_.density_map, use_this_cache, wanted_batch_size);
                        batch_size_optimizer.update_batch_size( );
                    }
                    else if ( search_particle_.parameter_map.psi ) {
                        euler_search_.InitGrid(my_symmetry, angular_step, 0.0, 0.0, psi_max, psi_step, psi_start, search_reference_3d_.pixel_size / high_resolution_limit_search, search_particle_.parameter_map, best_parameters_to_keep);
                        if ( euler_search_.best_parameters_to_keep != best_parameters_to_keep )
                            best_parameters_to_keep = euler_search_.best_parameters_to_keep;
                        wanted_batch_size = batch_size_optimizer.start( );
                        euler_search_.Run(search_particle_, *search_reference_3d_.density_map, use_this_cache, wanted_batch_size);
                        batch_size_optimizer.update_batch_size( );
                    }
                    else {
                        euler_search_.InitGrid(my_symmetry, angular_step, 0.0, 0.0, psi_max, psi_step, psi_start, search_reference_3d_.pixel_size / high_resolution_limit_search, search_particle_.parameter_map, best_parameters_to_keep);
                        euler_search_.psi_start = 360.0 - input_parameters.phi;
                        best_parameters_to_keep = 1;
                    }

                    // FIXME: this should be configurable, for example, in adding a loop here to do branch and bound a local refinemnt doesn't make any sense.
                    // Do local refinement of the top hits to determine the best match

                    search_parameters                           = input_parameters;
                    euler_search_.list_of_best_parameters[0][0] = input_parameters.phi;
                    euler_search_.list_of_best_parameters[0][1] = input_parameters.theta;
                    euler_search_.list_of_best_parameters[0][2] = input_parameters.psi;
                    euler_search_.list_of_best_parameters[0][3] = input_parameters.x_shift;
                    euler_search_.list_of_best_parameters[0][4] = input_parameters.y_shift;

                    // for ( int i = 0; i < euler_search_.best_parameters_to_keep; i++ ) {
                    //     wxPrintf("Phi,Theta,Psi x,y score %f,%f,%f  %f,%f  %g\n", euler_search_.list_of_best_parameters[i][0], euler_search_.list_of_best_parameters[i][1], euler_search_.list_of_best_parameters[i][2], euler_search_.list_of_best_parameters[i][3], euler_search_.list_of_best_parameters[i][4], euler_search_.list_of_best_parameters[i][5]);
                    // }
                    // exit(1);
                    //for (i = 0; i < search_particle_.number_of_parameters; i++) {search_parameters[i] = input_parameters[i];}
                    //for (j = 1; j < 6; j++) {euler_search_.list_of_best_parameters[0][j - 1] = input_parameters[j];}

                    search_particle_.SetParameterConstraints(powf(parameter_average.sigma, 2));

                    comparison_object.SetHostPointers(&search_reference_3d_, &search_projection_image, &search_particle_, true);

                    search_particle_.CenterInCorner( );
                    new_buffer_size = search_particle_.SetIndexForWeightedCorrelation( );
                    comparison_object.AllocateBuffers(new_buffer_size);

                    comparison_object.PrepareGpuImages(search_particle_, search_projection_image, true);
                    comparison_object.PrepareGpuCTFImages(search_particle_, true);

                    search_particle_.SetParameters(input_parameters);
                    search_particle_.MapParameters(cg_starting_point);
                    search_particle_.mask_radius = outer_mask_radius;
                    //				output_parameters[15] = - 100.0 * FrealignObjectiveFunction(&comparison_object, cg_starting_point);
                    //				if (! do_local_refinement) input_parameters[15] = output_parameters[15];
                    search_particle_.UnmapParametersToExternal(output_parameters, cg_starting_point);
                    if ( ignore_input_angles && best_parameters_to_keep >= 1 )
                        istart = 1;
                    else
                        istart = 0;

                    if ( take_random_best_parameter == true ) {
                        float best_value  = euler_search_.list_of_best_parameters[1][5];
                        float worst_value = euler_search_.list_of_best_parameters[best_parameters_to_keep][5];

                        float diff           = best_value - worst_value;
                        float top_percent    = best_value - (diff * 0.15);
                        int   number_to_keep = 1;

                        for ( int counter = 2; counter <= best_parameters_to_keep; counter++ ) {
                            if ( euler_search_.list_of_best_parameters[counter][5] >= top_percent ) {
                                number_to_keep++;
                            }
                        }

#ifdef CISTEM_DETERMINISTIC_OUTCOME
                        parameter_to_keep = myroundint(((0.0f + 1.0) / 2.0) * float(number_to_keep - 1)) + 1;

#else
                        parameter_to_keep = myroundint(((global_random_number_generator.GetUniformRandom( ) + 1.0) / 2.0) * float(number_to_keep - 1)) + 1;
#endif
                        //wxPrintf("best_value = %f, worst_value = %f, top 10%% = %f, number_above = %i, number to take = %i\n", best_value, worst_value, top_percent, number_to_keep, parameter_to_keep);

                        /*for (j = 1; j < 6; j++)
					{
						search_parameters[j] = euler_search_local.list_of_best_parameters[parameter_to_keep][j - 1];
					}*/

                        search_parameters.psi     = euler_search_.list_of_best_parameters[parameter_to_keep][2];
                        search_parameters.theta   = euler_search_.list_of_best_parameters[parameter_to_keep][1];
                        search_parameters.phi     = euler_search_.list_of_best_parameters[parameter_to_keep][0];
                        search_parameters.x_shift = euler_search_.list_of_best_parameters[parameter_to_keep][3];
                        search_parameters.y_shift = euler_search_.list_of_best_parameters[parameter_to_keep][4];

                        if ( ! search_particle_.parameter_map.x_shift )
                            search_parameters.x_shift = input_parameters.x_shift;
                        if ( ! search_particle_.parameter_map.y_shift )
                            search_parameters.y_shift = input_parameters.y_shift;
                        search_particle_.SetParameters(search_parameters);
                        search_particle_.MapParameters(cg_starting_point);

                        search_parameters.score = -100.0 * conjugate_gradient_minimizer.Init(&FrealignObjectiveFunction, &comparison_object, search_particle_.number_of_search_dimensions, cg_starting_point, cg_accuracy);
                        output_parameters.score = search_parameters.score;
                        if ( ! do_local_refinement_ )
                            input_parameters.score = output_parameters.score;

                        temp_float = -100.0 * conjugate_gradient_minimizer.Run(50);
                        number_of_calls_to_score_function_local += conjugate_gradient_minimizer.GetNumFunctionCalls( );
#ifdef PRINT_SCORES
                        wxPrintf("Current score is %g, from line %i\n", temp_float, __LINE__);
#endif

                        search_particle_.UnmapParametersToExternal(output_parameters, conjugate_gradient_minimizer.GetPointerToBestValues( ));
                        output_parameters.score = temp_float;
                    }
                    else {

                        for ( i = istart; i <= best_parameters_to_keep; i++ ) {
                            search_parameters.psi     = euler_search_.list_of_best_parameters[i][2];
                            search_parameters.theta   = euler_search_.list_of_best_parameters[i][1];
                            search_parameters.phi     = euler_search_.list_of_best_parameters[i][0];
                            search_parameters.x_shift = euler_search_.list_of_best_parameters[i][3];
                            search_parameters.y_shift = euler_search_.list_of_best_parameters[i][4];

                            //					wxPrintf("parameters in  = %i %g, %g, %g, %g, %g %g\n", i, search_parameters[3], search_parameters[2],
                            //							search_parameters[1], search_parameters[4], search_parameters[5], euler_search_.list_of_best_parameters[i][5]);
                            if ( ! search_particle_.parameter_map.x_shift )
                                search_parameters.x_shift = input_parameters.x_shift;
                            if ( ! search_particle_.parameter_map.y_shift )
                                search_parameters.y_shift = input_parameters.y_shift;
                            search_particle_.SetParameters(search_parameters);
                            search_particle_.MapParameters(cg_starting_point);

                            search_parameters.score = -100.0 * conjugate_gradient_minimizer.Init(&FrealignObjectiveFunction, &comparison_object, search_particle_.number_of_search_dimensions, cg_starting_point, cg_accuracy);

                            if ( i == istart ) {
                                output_parameters.score = search_parameters.score;
                                if ( ! do_local_refinement_ )
                                    input_parameters.score = output_parameters.score;
                            }
                            if ( skip_local_refinement )
                                temp_float = search_parameters.score;
                            else {
                                temp_float = -100.0 * conjugate_gradient_minimizer.Run(50);
                                number_of_calls_to_score_function_local += conjugate_gradient_minimizer.GetNumFunctionCalls( );
                            }

#ifdef PRINT_SCORES
                            wxPrintf("Current score is %g, from line %i\n", temp_float, __LINE__);
#endif

                            // Uncomment the following line to skip local refinement.
                            //					temp_float = search_parameters[15];
                            //					wxPrintf("best, refine in, out, diff = %i %g %g %g %g\n", i, output_parameters[15], search_parameters[15], temp_float, temp_float - output_parameters[15]);
                            //					log_diff = output_parameters[15] - temp_float;
                            //					if (log_diff > log_range) log_diff = log_range;
                            //					if (log_diff < - log_range) log_diff = - log_range;
                            if ( temp_float > output_parameters.score )
                            // If log_diff >= 0, exp(log_diff) will always be larger than the random number and the search parameters will be kept.
                            // If log_diff < 0, there is an increasing chance that the random number is larger than exp(log_diff) and the new
                            // (worse) parameters will not be kept.
                            //					if ((global_random_number_generator.GetUniformRandom() + 1.0) / 2.0 < 1.0 / (1.0 + exp(log_diff)))
                            {
                                //						if (log_diff < 0.0) wxPrintf("log_diff = %g\n", log_diff);
                                //						wxPrintf("image_counter = %i, i = %i, score = %g\n", image_counter, i, temp_float);
                                search_particle_.UnmapParametersToExternal(output_parameters, conjugate_gradient_minimizer.GetPointerToBestValues( ));
                                output_parameters.score = temp_float;
                                //						wxPrintf("parameters out = %g, %g, %g, %g, %g\n", output_parameters[3], output_parameters[2],
                                //								output_parameters[1], output_parameters[4], output_parameters[5]);
                            }
                            //					wxPrintf("refine in, out, keep = %i %g %g %g\n", i, search_parameters[15], temp_float, output_parameters[15]);
                            //					wxPrintf("parameters out = %g, %g, %g, %g, %g\n", output_parameters[3], output_parameters[2],
                            //							output_parameters[1], output_parameters[4], output_parameters[5]);
                        }
                    }
                    refine_particle_.SetParameters(output_parameters, true);
                    output_parameters.score_change = output_parameters.score - input_parameters.score;
                    //				my_time_out = wxDateTime::UNow(); wxPrintf("global search done: ms taken = %li\n", my_time_out.Subtract(my_time_in).GetMilliseconds());
                    timer.lap("global search local");
                }

                if ( do_local_refinement_ ) {
                    timer.start("local refinement local");
                    //				my_time_in = wxDateTime::UNow();

                    comparison_object.SetHostPointers(&input_3d_, &projection_image_, &refine_particle_, false);

                    comparison_object.PrepareGpuImages(refine_particle_, projection_image_, false);
                    comparison_object.PrepareGpuCTFImages(refine_particle_, false);

                    refine_particle_.MapParameters(cg_starting_point);

                    temp_float = -100.0 * conjugate_gradient_minimizer.Init(&FrealignObjectiveFunction, &comparison_object, refine_particle_.number_of_search_dimensions, cg_starting_point, cg_accuracy);
                    //???				if (! do_global_search) input_parameters[15] = temp_float;
                    output_parameters.score = -100.0 * conjugate_gradient_minimizer.Run(50);
                    number_of_calls_to_score_function_local += conjugate_gradient_minimizer.GetNumFunctionCalls( );

#ifdef PRINT_SCORES
                    wxPrintf("Current score is %g, from line %i\n", output_parameters.score, __LINE__);
#endif
                    // wxPrintf("Output, input = %g, %g from line %i\n", output_parameters.score, input_parameters.score, __LINE__);

                    refine_particle_.UnmapParametersToExternal(output_parameters, conjugate_gradient_minimizer.GetPointerToBestValues( ));
                    timer.lap("local refinement local");
                    //				my_time_out = wxDateTime::UNow(); wxPrintf("local refinement done: ms taken = %li\n", my_time_out.Subtract(my_time_in).GetMilliseconds());
                }
                //			log_diff = input_parameters[15] - output_parameters[15];
                //			wxPrintf("in = %g out = %g log_diff = %g ratio = %g\n", input_parameters[15], output_parameters[15], log_diff, 1.0 / (1.0 + exp(log_diff)));
                //			if (log_diff > log_range) log_diff = log_range;
                //			if (log_diff < - log_range) log_diff = - log_range;
                // If log_diff >= 0, exp(log_diff) will never be smaller than the random number and the new parameters will be kept.
                // If log_diff < 0 (new parameters give worse likelihood), new parameters will only be kept if random number smaller than exp(log_diff).
                //			if ((global_random_number_generator.GetUniformRandom() + 1.0) / 2.0 >= 1.0 / (1.0 + exp(log_diff))) for (i = 0; i < refine_particle_.number_of_parameters; i++) {output_parameters[i] = input_parameters[i];}
                //			else output_parameters[16] = output_parameters[15] - input_parameters[15];
                output_parameters.score_change = output_parameters.score - input_parameters.score;
                //			wxPrintf("in, out, diff = %g %g %g\n", input_parameters.score, output_parameters.score, output_parameters.score_change);
                if ( output_parameters.score_change < 0.0f )
                    output_parameters = input_parameters;
            }
            //		else
            //		{
            ////			input_parameters.score = - 100.0 * FrealignObjectiveFunction(&comparison_object, cg_starting_point);
            ////			output_parameters.score = - 100.0 * FrealignObjectiveFunction(&comparison_object, cg_starting_point);
            //			output_parameters.score = input_parameters.score;
            //			output_parameters.score_change = 0.0f;
            //		}
            // refine_particle_.UnmapParametersToExternal(output_parameters, conjugate_gradient_minimizer.GetPointerToBestValues( ));

            refine_particle_.SetParameters(output_parameters);
            // refine_particle_.UnmapParametersToExternal(output_parameters, conjugate_gradient_minimizer.GetPointerToBestValues( ));

            //		refine_particle_.SetAlignmentParameters(output_parameters.phi, output_parameters.theta, output_parameters.psi, 0.0, 0.0);
            //		unbinned_image.ClipInto(refine_particle_.particle_image);
            //		refine_particle_.particle_image->MultiplyByConstant(binning_factor_refine);
            //		refine_particle_.particle_image->QuickAndDirtyWriteSlice("part3.mrc", 1);
            //		refine_particle_.PhaseFlipImage();
            //		refine_particle_.BeamTiltMultiplyImage();
            //		refine_particle_.CalculateProjection(projection_image_, input_3d_);
            //		projection_image_.ClipInto(&unbinned_image);
            //		unbinned_image.BackwardFFT();
            //		unbinned_image.ClipInto(&final_image);
            //		logp = refine_particle_.ReturnLogLikelihood(input_image_, final_image, pixel_size, classification_resolution_limit, alpha, sigma);

            timer.start("return log likelihood");
            if ( (refine_particle_.number_of_search_dimensions > 0) && (do_global_search_ || do_local_refinement_) ) {

                output_parameters.logp = refine_particle_.ReturnLogLikelihood(input_image_, input_ctf, input_3d_, input_statistics, classification_resolution_limit);
            }
            else {

                output_parameters.logp  = refine_particle_.ReturnLogLikelihood(input_image_, input_ctf, input_3d_, input_statistics, classification_resolution_limit, &frealign_score_local);
                output_parameters.score = -100.0f * frealign_score_local;
                // refine_particle_.UnmapParametersToExternal(output_parameters, conjugate_gradient_minimizer.GetPointerToBestValues( ));

                output_parameters.score_change = output_parameters.score - input_parameters.score;
            }
            timer.lap("return log likelihood");
            //		logp = refine_particle_.ReturnLogLikelihood(input_3d_, refine_statistics, classification_resolution_limit);
            //		output_parameters[14] = sigma * binning_factor_refine;

            //		refine_particle_.CalculateMaskedLogLikelihood(projection_image_, input_3d_, classification_resolution_limit);
            //		output_parameters[13] = refine_particle_.logp;
            if ( refine_particle_.snr > 0.0 )
                output_parameters.sigma = sqrtf(1.0 / refine_particle_.snr);

            //		output_parameters[14] = refine_particle_.sigma_noise * binning_factor_refine;
            //		wxPrintf("logp, sigma, score = %g %g %g\n", output_parameters[13], output_parameters[14], output_parameters[15]);
            //		refine_particle_.CalculateProjection(projection_image_, input_3d_);
            //		projection_image_.BackwardFFT();
            //		wxPrintf("snr = %g mask = %g var_A = %g\n", refine_particle_.snr, refine_particle_.mask_volume, projection_image_.ReturnVarianceOfRealValues());
            //		output_parameters[14] = sqrtf(refine_particle_.snr * refine_particle_.particle_image->number_of_real_space_pixels
            //				/ refine_particle_.mask_volume / projection_image_.ReturnVarianceOfRealValues()) * binning_factor_refine;

            if ( calculate_matching_projections ) {
                timer.start("calculate matching projections");
                refine_particle_.SetAlignmentParameters(output_parameters.phi, output_parameters.theta, output_parameters.psi, 0.0, 0.0);
                current_projection++;
                refine_particle_.CalculateProjection(projection_image_, input_3d_);
                projection_image_.ClipInto(&final_image);
                final_image.PhaseShift(output_parameters.x_shift / pixel_size, output_parameters.y_shift / pixel_size);
                final_image.BackwardFFT( );
                final_image.WriteSlice(output_file, current_projection);
                timer.lap("calculate matching projections");
            }

            //temp_float = input_parameters[1]; input_parameters[1] = input_parameters[3]; input_parameters[3] = temp_float;
            //temp_float = output_parameters[1]; output_parameters[1] = output_parameters[3]; output_parameters[3] = temp_float;

            //input_parameters.SwapPsiAndPhi();
            //output_parameters.SwapPsiAndPhi();
            output_parameters.ReplaceNanAndInfWithOther(input_parameters);
            /*
		for (i = 1; i < refine_particle_.number_of_parameters; i++)
		{
			if (std::isnan(output_parameters[i]) != 0)
			{
//				MyDebugAssertTrue(false, "NaN value for output parameter encountered");
				output_parameters[i] = input_parameters[i];
			}
		}
		*/
            input_parameters.image_is_active  = 1; //fabsf(input_parameters[7]);
            output_parameters.image_is_active = input_parameters.image_is_active;
            if ( output_parameters.score < 0.0 )
                output_parameters.score = 0.0;
            // wxPrintf("Score %g at line %d\n", output_parameters.score, __LINE__);
            output_star_file.all_parameters[current_line_] = output_parameters;

            if ( is_running_locally == false ) // send results back to the gui..
            {
                timer.start("writing results to the gui");
                intermediate_result             = new JobResult;
                intermediate_result->job_number = my_current_job.job_number;

                //			if (output_parameters.position_in_stack == 0) wxPrintf("HELP IT IS 0\n");
                gui_result_parameters[0]  = current_class;
                gui_result_parameters[1]  = output_parameters.position_in_stack;
                gui_result_parameters[2]  = output_parameters.image_is_active;
                gui_result_parameters[3]  = output_parameters.psi;
                gui_result_parameters[4]  = output_parameters.theta;
                gui_result_parameters[5]  = output_parameters.phi;
                gui_result_parameters[6]  = output_parameters.x_shift;
                gui_result_parameters[7]  = output_parameters.y_shift;
                gui_result_parameters[8]  = output_parameters.defocus_1;
                gui_result_parameters[9]  = output_parameters.defocus_2;
                gui_result_parameters[10] = output_parameters.defocus_angle;
                gui_result_parameters[11] = output_parameters.phase_shift;
                gui_result_parameters[12] = output_parameters.occupancy;
                gui_result_parameters[13] = output_parameters.logp;
                gui_result_parameters[14] = output_parameters.sigma;
                gui_result_parameters[15] = output_parameters.score;
                gui_result_parameters[16] = output_parameters.score_change;
                gui_result_parameters[17] = output_parameters.pixel_size;
                gui_result_parameters[18] = output_parameters.microscope_voltage_kv;
                gui_result_parameters[19] = output_parameters.microscope_spherical_aberration_mm;
                gui_result_parameters[20] = output_parameters.beam_tilt_x;
                gui_result_parameters[21] = output_parameters.beam_tilt_y;
                gui_result_parameters[22] = output_parameters.image_shift_x;
                gui_result_parameters[23] = output_parameters.image_shift_y;
                gui_result_parameters[24] = output_parameters.amplitude_contrast;
                gui_result_parameters[25] = output_parameters.assigned_subset;

                intermediate_result->SetResult(26, gui_result_parameters);
                AddJobToResultQueue(intermediate_result);
                timer.lap("writing results to the gui");
            }

            //for (i = 1; i < refine_particle_.number_of_parameters; i++) {output_parameters[i] -= input_parameters[i];}
            output_parameters.Subtract(input_parameters);
            output_parameters.position_in_stack                           = input_parameters.position_in_stack;
            output_shifts_file.all_parameters[current_line_]              = output_parameters;
            output_shifts_file.all_parameters[current_line_].score_change = 0.0f;

            //my_output_par_shifts_file.WriteLine(output_parameters);

            //fflush(my_output_par_file.parameter_file);
            //fflush(my_output_par_shifts_file.parameter_file);

            //		wxPrintf("thread = %i, line = %i\n", ReturnThreadNumberOfCurrentThread(), current_line_);
            if ( is_running_locally == true && ReturnThreadNumberOfCurrentThread( ) == 0 )
                my_progress->Update(image_counter);
        }

        timer.start("clean up");
        input_image_.Deallocate( );
        temp_image_.Deallocate( );
        projection_image_.Deallocate( );
        if ( ctf_refinement && high_resolution_limit <= 20.0 )
            binned_image.Deallocate( );
        refine_particle_.Deallocate( );
        if ( do_global_search_ || do_global_search_and_local_refinement == true ) {
            search_particle_.Deallocate( );
            search_projection_image.Deallocate( );
            temp_image2_.Deallocate( );
        }
        if ( calculate_matching_projections )
            final_image.Deallocate( );
        timer.lap("clean up");

#pragma omp critical
        {
            number_of_calls_to_score_function += number_of_calls_to_score_function_local;
        }

        euler_search_.timer.print_times( );

    } // end omp section

    if ( is_running_locally == true )
        delete my_progress;

    timer.start("write out star files");
    // write the files..

    output_star_file.WriteTocisTEMStarFile(output_star_filename, -1, -1, first_particle, last_particle);
    output_shifts_file.WriteTocisTEMStarFile(output_shift_filename, -1, -1, first_particle, last_particle);
    timer.lap("write out star files");
    //	delete global_euler_search;
    if ( do_global_search ) {
        delete[] projection_cache;
#ifdef ENABLEGPU
        delete[] gpu_projection_cache;
#endif
        //		search_projection_image.RotateFourier2DDeleteIndex(kernel_index, psi_max, psi_step);
    }
    if ( calculate_matching_projections )
        delete output_file;

    wxPrintf("\nRefine3D: Normal termination\n\n");

    timer.print_times( );
    global_timer.print_times( );
    wxPrintf("\nNumber of calls to scoring function %ld\n", number_of_calls_to_score_function);

    return true;
}
