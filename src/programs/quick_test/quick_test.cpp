

#include "../../core/core_headers.h"
#include "../../constants/constants.h"

#ifdef ENABLEGPU
#include "../../gpu/gpu_core_headers.h"
#include "../../gpu/DeviceManager.h"
#include "../../gpu/TemplateMatchingCore.h"
#include "quick_test_gpu.h"
#else
#include "../../core/core_headers.h"
#endif

#include "../../constants/constants.h"

#include "../../core/scattering_potential.h"

// Toggle which experiment to build — define exactly one:
//#define cisTEM_test_shifts
//#define cisTEM_test_astigmatism
#define cisTEM_test_scalloping

class
        QuickTestApp : public MyApp {

  public:
    bool     DoCalculation( );
    void     DoInteractiveUserInput( );
    void     AddCommandLineOptions( );
    wxString symmetry_symbol;
    bool     my_test_1 = false;
    bool     my_test_2 = true;
    int      idx;

    std::array<wxString, 2> input_starfile_filename;

  private:
};

IMPLEMENT_APP(QuickTestApp)

// Optional command-line stuff
void QuickTestApp::AddCommandLineOptions( ) {
    command_line_parser.AddLongSwitch("disable-user-input", "Disable interactive user input prompts. Default false");
    command_line_parser.AddOption("", "noise-power-before-ctf", "Scalloping experiment: noise:signal power ratio 1:N before the CTF (default 1); sigma = sqrt(N)", wxCMD_LINE_VAL_DOUBLE);
    command_line_parser.AddOption("", "noise-power-after-ctf", "Scalloping experiment: noise:signal power ratio 1:N after the CTF (default 10); sigma = sqrt(N)", wxCMD_LINE_VAL_DOUBLE);
    command_line_parser.AddOption("", "sweep-original-peak-size", "Scalloping experiment: peak window size in px (default 8, positive even)", wxCMD_LINE_VAL_NUMBER);
    command_line_parser.AddOption("", "sweep-padding-multiplier", "Scalloping experiment: padded_peak_size = N*window (default 2)", wxCMD_LINE_VAL_NUMBER);
    command_line_parser.AddOption("", "sweep-upsample-factor", "Scalloping experiment: upsample_peak_size = N*padded (default 8)", wxCMD_LINE_VAL_NUMBER);
    command_line_parser.AddOption("", "sweep-padding-mode", "Scalloping experiment: 0=mirror (default), 1=zero pad, 2=Hann+zero pad", wxCMD_LINE_VAL_NUMBER);
    command_line_parser.AddOption("", "sweep-width-fraction", "Scalloping experiment: FWHM width fraction above baseline (default 0.5)", wxCMD_LINE_VAL_DOUBLE);
    command_line_parser.AddOption("", "shift-mode", "Scalloping experiment: 0=diagonal (default), 1=x only, 2=y only, 3=random azimuth (fixed magnitude)", wxCMD_LINE_VAL_NUMBER);
    command_line_parser.AddOption("", "base-upsample-factor", "Scalloping experiment: Fourier-upsample the reference by this factor before the sweep (default 1; 2 -> 0.5 A/px)", wxCMD_LINE_VAL_DOUBLE);
    command_line_parser.AddOption("", "n-replicates", "Scalloping experiment: independent noise realizations per (pixel size, offset) cell (default 10)", wxCMD_LINE_VAL_NUMBER);
    command_line_parser.AddLongSwitch("probe-upsampler", "Scalloping experiment: run the synthetic-peak upsampler probe (recover a known sub-pixel shift of a clean Gaussian, swept over peak width and padding mode) and exit");
    command_line_parser.AddLongSwitch("warmup", "Scalloping experiment: run many replicates at one fixed condition, report running mean/SEM of corrected peak height vs N (to choose n-replicates), and exit");
    command_line_parser.AddLongSwitch("report-snr", "Scalloping experiment: compute and report the total power SNR (signal : pre-CTF-noise-through-CTF + post-CTF-noise) per pixel size, and exit");
    command_line_parser.AddLongSwitch("defocus-sweep", "Scalloping experiment: sweep defocus (Scherzer ~700 A to ~14200 A in 1500 A steps) against the coarse 1/3/5/7 A/px ladder instead of the single-defocus full pixel ladder; records defocus + per-peak FWHM columns");
}

// override the DoInteractiveUserInput

void QuickTestApp::DoInteractiveUserInput( ) {
    // This flag allows skipping interactive prompts, useful for automated testing with Copilot.
    if ( command_line_parser.FoundSwitch("disable-user-input") ) {
        std::cout << "Skipping interactive user input as per command line flag." << std::endl;
        return;
    }
    UserInput* my_input = new UserInput("QuickTest", 2.0);

    idx                           = my_input->GetIntFromUser("Index", "", "", 0, 1000);
    input_starfile_filename.at(0) = my_input->GetFilenameFromUser("Input starfile filename 1", "", "", false);
    input_starfile_filename.at(1) = my_input->GetFilenameFromUser("Input starfile filename 2", "", "", false);
    symmetry_symbol               = my_input->GetSymmetryFromUser("Particle symmetry", "The assumed symmetry of the particle to be reconstructed", "C1");

    delete my_input;
}

bool QuickTestApp::DoCalculation( ) {

#ifdef ENABLEGPU
    // DeviceManager gpuDev;
    // gpuDev.ListDevices( );

    // QuickTestGPU quick_test_gpu;3
    // quick_test_gpu.callHelloFromGPU(idx);
#endif

#ifdef cisTEM_test_shifts
    const int vol_size     = 512;
    const int prj_size     = 1024;
    const int n_replicates = 10;
    // vols at  0.5 1.0 2.0 4.0 /scratch/etna/sub_pixel_and_astig_exp/7A4M-assembly1-pixel_size_0.5.mrc

    // Let's actually make an vector of std pairs of pixel_size and peak_file_name
    std::vector<std::pair<float, std::string>> pixel_size_and_peak_file_names;
    pixel_size_and_peak_file_names.push_back(std::make_pair(0.5f, "/scratch/etna/sub_pixel_and_astig_exp/7A4M-assembly1-pixel_size_0.5.mrc"));
    pixel_size_and_peak_file_names.push_back(std::make_pair(1.0f, "/scratch/etna/sub_pixel_and_astig_exp/7A4M-assembly1-pixel_size_1.0.mrc"));
    pixel_size_and_peak_file_names.push_back(std::make_pair(2.0f, "/scratch/etna/sub_pixel_and_astig_exp/7A4M-assembly1-pixel_size_2.0.mrc"));
    pixel_size_and_peak_file_names.push_back(std::make_pair(4.0f, "/scratch/etna/sub_pixel_and_astig_exp/7A4M-assembly1-pixel_size_4.0.mrc"));

    Image prj        = Image(vol_size, vol_size, 1, true);
    Image padded_prj = Image(prj_size, prj_size, 1, true);
    // Over each vol we'll read in, prep for projection using an AnglesAndShifts object, and then project using the Image::ExtractSlice method.
    // Identity projection (no rotation) and we'll loop over shifts from 0.0 to sqrt(2)*0.5 in steps of 0.05 equal in x and y.
    std::vector<float> shifts = {0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7};

    for ( int i = 0; i < n_replicates; i++ ) {
        // For each replicate we'll make two noise images, one before and one after the CTF is applied. These will be  size prj and padded_prj.
        Image noise_before_ctf = Image(prj_size, prj_size, 1, true);
        Image noise_after_ctf  = Image(prj_size, prj_size, 1, true);
        noise_before_ctf.FillWithNoise(NoiseType::GAUSSIAN, 0.0f, 1.0f);
        noise_after_ctf.FillWithNoise(NoiseType::GAUSSIAN, 0.0f, 8.0f);

        for ( const auto& pixel_size_and_peak_file_name : pixel_size_and_peak_file_names ) {
            auto peak_file_name = pixel_size_and_peak_file_name.second;
            wxPrintf("Processing peak file: %s\n", peak_file_name.c_str( ));
            float pixel_size = pixel_size_and_peak_file_name.first;
            // For this experiment we'll use a single CTF for all replicates.
            CTF   ctf(300, 2.7, 0.07, 5200, 5000, 45, pixel_size, 0);
            Image vol;
            vol.QuickAndDirtyReadSlices(peak_file_name, 1, 512);
            vol.ForwardFFT( );

            vol.SwapRealSpaceQuadrants( );

            for ( const auto& shift : shifts ) {
                // Print shifts
                wxPrintf("Shift: %f\n", shift);
                AnglesAndShifts angles_and_shifts(0.0, 0.0, 0.0, shift, shift);
                padded_prj.SetToConstant(0.0f);

                vol.ExtractSlice(prj, angles_and_shifts, 0.0, false);
                prj.SwapRealSpaceQuadrants( );
                // Angles and shifts apparently does not shift on projection
                // If the shift is 0.0 add a tiny shift
                float shift_to_use = shift == 0.0f ? 0.005f : shift;

                prj.PhaseShift(shift_to_use, shift_to_use, 0.0f);
                prj.BackwardFFT( );
                prj.AddConstant(-1.0f * prj.ReturnAverageOfRealValuesOnEdges( ));
                prj.ClipInto(&padded_prj);

                padded_prj.ZeroFloatAndNormalize( );
                padded_prj.AddImage(&noise_before_ctf);
                padded_prj.ForwardFFT( );
                padded_prj.ApplyCTF(ctf);
                padded_prj.BackwardFFT( );
                padded_prj.AddImage(&noise_after_ctf);

                // output dir /scratch/etna/sub_pixel_and_astig_exp/pixel_size_4.0
                std::string pixel_size_str = wxString::Format("%.1f", pixel_size).ToStdString( );
                std::string shift_str      = wxString::Format("%.2f", shift).ToStdString( );
                std::string output_dir     = "/scratch/etna/sub_pixel_and_astig_exp/pixel_size_Z_" + pixel_size_str;
                padded_prj.QuickAndDirtyWriteSlice(output_dir + "/test_proj_shift_" + shift_str + "_replicate_" + std::to_string(i) + ".mrc", 1);
            }

        } // end of peak_file_names

    } // end of n_replicates
#endif // cisTEM_test_shifts

#ifdef cisTEM_test_astigmatism
    const int   vol_size     = 512;
    const int   img_size     = 4096;
    const int   n_replicates = 10;
    const float pixel_size   = 0.5f;

    // 8x8 grid of 512x512 cells tiling the 4096x4096 image = 64 particles
    const int   grid_n          = 8;
    const int   cell_size       = img_size / grid_n; // 512, same as vol_size
    const int   particle_radius = 110; // ~110 Ang / 0.5 Ang/pix = 220 px diameter, 110 px radius
    const int   edge_margin     = 10;
    const float max_nudge       = static_cast<float>(cell_size / 2 - particle_radius - edge_margin); // 136 px

    // Grid cell centers relative to image center
    // Cells at indices 0..7, centers at (i+0.5)*512 in image coords
    // Relative to image center (2048): (i+0.5)*512 - 2048 = (i-3.5)*512
    std::vector<std::pair<int, int>> grid_positions;
    for ( int gy = 0; gy < grid_n; gy++ ) {
        for ( int gx = 0; gx < grid_n; gx++ ) {
            int cx = static_cast<int>((gx - 3.5f) * cell_size);
            int cy = static_cast<int>((gy - 3.5f) * cell_size);
            grid_positions.push_back({cx, cy});
        }
    }

    // Defocus conditions: {def1, def2}, all with astigmatism angle = 45
    std::vector<std::pair<float, float>> defocus_conditions = {
            {2000.0f, 2000.0f},
            {4000.0f, 4000.0f},
            {6000.0f, 6000.0f},
            {9000.0f, 9000.0f},
            {12000.0f, 12000.0f},
            {6000.0f, 2000.0f},
            {12000.0f, 6000.0f}};
    const float defocus_angle = 45.0f;

    std::string vol_filename = "/scratch/etna/sub_pixel_and_astig_exp/7A4M-assembly1-pixel_size_0.5.mrc";
    Image       vol;
    vol.QuickAndDirtyReadSlices(vol_filename, 1, vol_size);
    vol.ForwardFFT( );
    vol.SwapRealSpaceQuadrants( );

    Image prj      = Image(vol_size, vol_size, 1, true);
    Image full_img = Image(img_size, img_size, 1, true);

    for ( int i = 0; i < n_replicates; i++ ) {
        Image noise_before_ctf = Image(img_size, img_size, 1, true);
        Image noise_after_ctf  = Image(img_size, img_size, 1, true);
        noise_before_ctf.FillWithNoise(NoiseType::GAUSSIAN, 0.0f, 1.0f);
        noise_after_ctf.FillWithNoise(NoiseType::GAUSSIAN, 0.0f, 4.0f);

        for ( const auto& defocus : defocus_conditions ) {
            wxPrintf("Replicate %d, defocus %.0f/%.0f\n", i, defocus.first, defocus.second);
            CTF ctf(300, 2.7, 0.07, defocus.first, defocus.second, defocus_angle, pixel_size, 0);

            full_img.SetToConstant(0.0f);

            for ( const auto& grid_pos : grid_positions ) {
                // Uniform random rotation on SO(3)
                float phi   = (global_random_number_generator.GetUniformRandom( ) + 1.0f) * 180.0f;
                float theta = rad_2_deg(acosf(global_random_number_generator.GetUniformRandom( )));
                float psi   = (global_random_number_generator.GetUniformRandom( ) + 1.0f) * 180.0f;

                // Random nudge within cell: ±136 px max, applied as phase shift before BackwardFFT
                float nudge_x = global_random_number_generator.GetUniformRandom( ) * max_nudge;
                float nudge_y = global_random_number_generator.GetUniformRandom( ) * max_nudge;

                AnglesAndShifts angles(phi, theta, psi, 0.0f, 0.0f);
                vol.ExtractSlice(prj, angles, 0.0, false);
                prj.SwapRealSpaceQuadrants( );
                prj.PhaseShift(nudge_x, nudge_y, 0.0f);
                prj.BackwardFFT( );
                prj.AddConstant(-1.0f * prj.ReturnAverageOfRealValuesOnEdges( ));

                full_img.InsertOtherImageAtSpecifiedPosition(&prj, grid_pos.first, grid_pos.second, 0);
            }

            full_img.ZeroFloatAndNormalize( );
            full_img.AddImage(&noise_before_ctf);
            full_img.ForwardFFT( );
            full_img.ApplyCTF(ctf);
            full_img.BackwardFFT( );
            full_img.AddImage(&noise_after_ctf);

            int         def1_int   = static_cast<int>(defocus.first);
            int         def2_int   = static_cast<int>(defocus.second);
            std::string def_str    = std::to_string(def1_int) + "_" + std::to_string(def2_int);
            std::string output_dir = "/scratch/etna/sub_pixel_and_astig_exp/astigmatism_def_" + def_str;
            full_img.QuickAndDirtyWriteSlice(output_dir + "/astigmatism_def_" + def_str + "_replicate_" + std::to_string(i) + ".mrc", 1);
        }

    } // end of n_replicates
#endif // cisTEM_test_astigmatism

#ifdef cisTEM_test_scalloping
    // ----------------------------------------------------------------------------------
    // Scalloping-loss recovery experiment.
    //
    // For a noisy, CTF-distorted copy of a reference, measure the cross-correlation peak
    // height across a range of sub-pixel offsets, both at integer-pixel sampling and after
    // Fourier-upsampling peak-sampling correction. The correction targets the "scalloping
    // loss": the peak-height deficit that grows as the true peak falls between samples.
    //
    // Peak extraction reuses the production algorithm exactly as make_template_result does:
    // Image::FindPeakWithIntegerCoordinatesForManyPeaks returns the uncorrected height
    // (peak_list) and the sampling-corrected height + sub-pixel offset (upsampled_peak_list).
    // ----------------------------------------------------------------------------------

    const bool debug_output = true; // save the reference and one noisy test image for inspection

    const std::string input_ref_filename     = "build/base_img.mrc";
    const std::string output_data_filename   = "build/scalloping_experiment.txt";
    const std::string debug_ref_ctf_filename = "build/scalloping_debug_ref_with_ctf.mrc"; // noiseless CTF-filtered reference
    const std::string debug_img_filename     = "build/scalloping_debug_img_ctf_noise.mrc"; // final target: CTF + noise

    // CTF: 300 kV, Cs 2.7 mm, 0.07 amplitude contrast, 8000 A defocus, no astigmatism, no phase shift.
    const float ctf_voltage      = 300.0f;
    const float ctf_cs           = 2.7f;
    const float ctf_amp_contrast = 0.07f;
    const float ctf_defocus      = 8000.0f;

    // Noise is specified as the noise:signal POWER (variance) ratio 1:N. The signal is normalized
    // to unit variance immediately before each noise add, so the noise variance equals N and its
    // standard deviation is sqrt(N). Defaults: 1:1 before the CTF, 1:10 after. Overridable from the
    // command line so noise can be swept and driven to ~0 as a positive control without rebuilding.
    float  noise_power_before_ctf = 1.0f;
    float  noise_power_after_ctf  = 10.0f;
    double temp_double;
    if ( command_line_parser.Found("noise-power-before-ctf", &temp_double) )
        noise_power_before_ctf = float(temp_double);
    if ( command_line_parser.Found("noise-power-after-ctf", &temp_double) )
        noise_power_after_ctf = float(temp_double);
    const float noise_sigma_before_ctf = sqrtf(noise_power_before_ctf);
    const float noise_sigma_after_ctf  = sqrtf(noise_power_after_ctf);
    wxPrintf("Noise power 1:N -> before CTF 1:%.3f (sigma %.4f), after CTF 1:%.3f (sigma %.4f)\n",
             noise_power_before_ctf, noise_sigma_before_ctf, noise_power_after_ctf, noise_sigma_after_ctf);

    // Peak-correction upsampling parameters, command-line configurable. Defaults match the production
    // peak finder: window 8, mirror pad (mode 0) to 2x (=16), upsample 8x (tile 128). Test window=16
    // pad=2 upsample=16 (tile 512) as an alternative; 128, 512, and the 2x-padded correlation box are
    // all 5-smooth even sizes for FFT efficiency.
    //
    // The sweep variant is used (not the production non-sweep call) because it bounds the sub-pixel
    // search to +/-0.5 px; the non-sweep method searches a (N/2)*sqrt(2) radius and can report
    // unphysical |offset| > 0.5.
    int   sweep_original_peak_size = 8;
    int   sweep_padding_multiplier = 2;
    int   sweep_upsample_factor    = 8;
    int   sweep_padding_mode       = 0;
    float sweep_width_fraction     = 0.5f;
    long  temp_long;
    if ( command_line_parser.Found("sweep-original-peak-size", &temp_long) )
        sweep_original_peak_size = int(temp_long);
    if ( command_line_parser.Found("sweep-padding-multiplier", &temp_long) )
        sweep_padding_multiplier = int(temp_long);
    if ( command_line_parser.Found("sweep-upsample-factor", &temp_long) )
        sweep_upsample_factor = int(temp_long);
    if ( command_line_parser.Found("sweep-padding-mode", &temp_long) )
        sweep_padding_mode = int(temp_long);
    if ( command_line_parser.Found("sweep-width-fraction", &temp_double) )
        sweep_width_fraction = float(temp_double);
    wxPrintf("Peak upsampling: window=%i padding=%ix upsample=%ix mode=%i width_fraction=%.3f\n",
             sweep_original_peak_size, sweep_padding_multiplier, sweep_upsample_factor,
             sweep_padding_mode, sweep_width_fraction);

    // Direction the sub-pixel displacement magnitude is applied. Axis-locked modes (x, y, diagonal)
    // probe for systematic per-axis bias; random mode (fixed magnitude, uniform random azimuth per
    // replicate) is the "real experiment" case — losses are not fully independent of direction
    // because a digital pixel is effectively square, so averaging over direction at fixed magnitude
    // characterizes that anisotropy.
    enum ScallopingShiftMode : int { SHIFT_DIAGONAL = 0,
                                     SHIFT_X        = 1,
                                     SHIFT_Y        = 2,
                                     SHIFT_RANDOM   = 3 };

    ScallopingShiftMode shift_mode = SHIFT_DIAGONAL;
    if ( command_line_parser.Found("shift-mode", &temp_long) )
        shift_mode = ScallopingShiftMode(int(temp_long));
    const char* shift_mode_name      = shift_mode == SHIFT_X ? "x" : shift_mode == SHIFT_Y ? "y"
                                                             : shift_mode == SHIFT_RANDOM  ? "random"
                                                                                           : "diagonal";
    float       base_upsample_factor = 1.0f;
    if ( command_line_parser.Found("base-upsample-factor", &temp_double) )
        base_upsample_factor = float(temp_double);
    // Replicates per (pixel size, offset) cell. Each draws fresh, independent noise (and, in random
    // shift mode, a fresh azimuth) so the noisy peak-height measurement can be averaged; a single
    // realization at 1:10 post-CTF noise is dominated by scatter.
    int n_replicates = 10;
    if ( command_line_parser.Found("n-replicates", &temp_long) )
        n_replicates = int(temp_long);
    wxPrintf("Shift mode: %s   base upsample factor: %.2f   replicates: %i\n", shift_mode_name, base_upsample_factor, n_replicates);

    // Defocus axis. Default: a single value (ctf_defocus). With --defocus-sweep the CTF defocus is
    // walked from Scherzer (~700 A) out to ~14200 A in 1500 A steps and paired with a coarse
    // 1/3/5/7 A/px ladder, to probe whether recovery degrades and the correlation peak broadens
    // with defocus while SNR and geometry are held fixed.
    const bool         defocus_sweep = command_line_parser.FoundSwitch("defocus-sweep");
    std::vector<float> defocus_values;
    if ( defocus_sweep ) {
        for ( float def = 700.0f; def <= 14200.0f + 1.0f; def += 1500.0f )
            defocus_values.push_back(def);
    }
    else {
        defocus_values.push_back(ctf_defocus);
    }
    wxPrintf("Defocus sweep: %s   (%i defocus value(s))\n", defocus_sweep ? "on" : "off", int(defocus_values.size( )));

    // Core-algo probe (positive control): does the sub-pixel upsampler recover a KNOWN shift of a
    // clean synthetic peak? This removes the reference, CTF, and noise entirely. A Gaussian of
    // controllable width sigma is built at the box center and shifted by a known sub-pixel amount via
    // an exact Fourier phase shift, then handed to the same FindPeakWithIntegerCoordinatesForManyPeaksSweep
    // the experiment uses. found should track the applied shift. Sweeping sigma isolates whether a
    // failure is width-dependent (which the window knob only masks) or intrinsic to the padding mode;
    // sweeping the mode tests the symmetrization hypothesis (mirror/Hann pad the tile into an even
    // function, and an even function has no sub-pixel position to recover).
    if ( command_line_parser.Found("probe-upsampler") ) {
        const int                probe_dim    = 128;
        const int                box_center   = probe_dim / 2;
        const std::vector<float> probe_sigmas = {0.8f, 1.5f, 3.0f, 6.0f};
        const std::vector<float> probe_shifts = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f};
        const std::vector<int>   probe_modes  = {0, 1, 2};
        wxPrintf("\n# synthetic upsampler probe: window=%i padding=%ix upsample=%ix\n",
                 sweep_original_peak_size, sweep_padding_multiplier, sweep_upsample_factor);
        wxPrintf("# mode sigma applied found_dx found_dy\n");
        for ( const int probe_mode : probe_modes ) {
            for ( const float sigma : probe_sigmas ) {
                for ( const float applied : probe_shifts ) {
                    Image peak;
                    peak.Allocate(probe_dim, probe_dim, 1, true);
                    const float two_sigma_sq = 2.0f * sigma * sigma;
                    for ( int j = 0; j < probe_dim; j++ ) {
                        for ( int i = 0; i < probe_dim; i++ ) {
                            const float dx                                                       = float(i - box_center);
                            const float dy                                                       = float(j - box_center);
                            peak.real_values[peak.ReturnReal1DAddressFromPhysicalCoord(i, j, 0)] = expf(-(dx * dx + dy * dy) / two_sigma_sq);
                        }
                    }
                    peak.is_in_real_space         = true;
                    peak.object_is_centred_in_box = true;
                    peak.ForwardFFT( );
                    peak.PhaseShift(applied, applied, 0.0f); // exact sub-pixel shift, in pixels
                    peak.BackwardFFT( );
                    peak.object_is_centred_in_box = true;

                    const Peak         integer_peak = peak.FindPeakWithIntegerCoordinates( );
                    std::vector<Peak>  peak_list, upsampled_peak_list;
                    std::vector<float> fwhm_x_px, fwhm_y_px;
                    std::vector<int>   upsample_status;
                    peak.FindPeakWithIntegerCoordinatesForManyPeaksSweep(
                            peak_list, upsampled_peak_list, fwhm_x_px, fwhm_y_px, upsample_status,
                            0.5f * integer_peak.value, cistem::match_template::PEAK_THRESHOLD_SCALE, 10.0f, 0,
                            sweep_original_peak_size, sweep_padding_multiplier, sweep_upsample_factor, probe_mode, sweep_width_fraction);
                    const float found_dx = (peak_list[0].x - float(peak.physical_address_of_box_center_x)) + upsampled_peak_list[0].x;
                    const float found_dy = (peak_list[0].y - float(peak.physical_address_of_box_center_y)) + upsampled_peak_list[0].y;
                    wxPrintf("%i %.2f %.3f %.4f %.4f\n", probe_mode, sigma, applied, found_dx, found_dy);
                }
            }
        }
        return true;
    }

    // Sub-pixel offset MAGNITUDES applied to the target, 0 -> 0.5 px so the peak stays inside the
    // origin pixel for any direction. shift_mode picks how the magnitude m is split into (dx, dy):
    //   diagonal -> (m/sqrt2, m/sqrt2)     x -> (m, 0)     y -> (0, m)     random -> m*(cos,sin) theta
    // All produce displacement magnitude m, so modes are compared at matched magnitude.
    // 0 -> 0.5 px in 0.025 steps (21 values) finely samples the offset axis of the recovery grid.
    std::vector<float> sub_pixel_offset_magnitudes;
    for ( int off_idx = 0; off_idx <= 20; off_idx++ )
        sub_pixel_offset_magnitudes.push_back(0.025f * float(off_idx));

    // Read the perfect reference and its native sampling.
    ImageFile ref_file(input_ref_filename, false);
    float     native_pixel_size = ref_file.ReturnPixelSize( );
    Image     perfect_ref;
    perfect_ref.ReadSlices(&ref_file, 1, 1);
    int native_dim = perfect_ref.logical_x_dimension;

    // Optional Fourier upsampling of the reference: pad its transform to a larger even box, giving a
    // finer effective pixel size. No real information is added (content stays band-limited to the
    // original Nyquist); this tests whether finer base sampling changes the sub-pixel behaviour.
    if ( base_upsample_factor > 1.0f ) {
        const int upsampled_dim = ReturnClosestFactorizedUpper(int(roundf(float(native_dim) * base_upsample_factor)), 5, true);
        perfect_ref.ForwardFFT( );
        perfect_ref.Resize(upsampled_dim, upsampled_dim, 1);
        perfect_ref.BackwardFFT( );
        native_pixel_size *= float(native_dim) / float(upsampled_dim);
        native_dim = upsampled_dim;
    }
    wxPrintf("Reference: %ix%i, effective pixel size %.4f A\n", native_dim, perfect_ref.logical_y_dimension, native_pixel_size);

    // Pre-binning: crop the Fourier transform to bin. Effective pixel = native * native_dim / binned_dim.
    // A ratio ladder 1..8 forms the pixel-size axis of the recovery grid; with --base-upsample-factor 2
    // (native 0.5 A) this spans 0.5-4.0 A/px. Each binned dim is the largest 5-smooth even size <= the
    // target (fast Fourier crop AND fast correlation FFT); duplicates from that rounding are dropped.
    // Pixel-size (binning) axis as ratios of the native sampling. The full sweep walks 1.0 .. 8.0;
    // --defocus-sweep uses the coarse 1/3/5/7 ladder (native ~1 A/px -> ~1/3/5/7 A/px), to be filled
    // in later if the defocus trend is interesting. Each ratio maps to the largest 5-smooth even
    // binned dim <= native/ratio; duplicates from that rounding are dropped.
    std::vector<float> pixel_ratios;
    if ( defocus_sweep ) {
        pixel_ratios = {1.0f, 3.0f, 5.0f, 7.0f};
    }
    else {
        const int n_pixel_sizes = 15;
        for ( int ps_idx = 0; ps_idx < n_pixel_sizes; ps_idx++ )
            pixel_ratios.push_back(1.0f + (7.0f / float(n_pixel_sizes - 1)) * float(ps_idx)); // 1.0 .. 8.0
    }
    std::vector<int> binned_dims;
    int              last_bd = -1;
    for ( const float ratio : pixel_ratios ) {
        const int bd = ReturnClosestFactorizedLower(int(roundf(float(native_dim) / ratio)), 5, true);
        if ( bd < 2 * sweep_original_peak_size ) // correlation map must comfortably hold the peak tile
            continue;
        if ( bd == last_bd ) // dedup after factorized rounding
            continue;
        binned_dims.push_back(bd);
        last_bd = bd;
    }

    const float pi_f = 3.14159265358979323846f;

    // ---- shared inner routines -------------------------------------------------------------------

    // Anti-alias preparation of a correlation input. real_img_normalized is a real-space, zero-mean,
    // unit-variance image; this tapers the box edge to remove the boundary discontinuity, zero-pads
    // to padded_dim (linearizing the cross-correlation so there is no circular wraparound), and
    // forward-transforms. Mutates real_img_normalized (taper is in place). This is the fix for the
    // aliasing that a raw (untapered, unpadded) cross-correlation would otherwise inject.
    auto prepare_correlation_input = [&](Image& real_img_normalized, Image& out_padded_fft, int padded_dim) {
        real_img_normalized.TaperEdges( );
        out_padded_fft.Allocate(padded_dim, padded_dim, 1, true);
        real_img_normalized.ClipInto(&out_padded_fft, 0.0f);
        out_padded_fft.ForwardFFT( );
    };

    // Build everything that depends only on the pixel size (binning): the clean reference FFT (reused
    // to seed every target), the CTF, the conjugated & prepared correlation template, the clean
    // post-CTF signal std (for the post-CTF noise scaling), and the 2x-padded correlation box size.
    auto build_pixel_context = [&](int binned_dim, float defocus_value, Image& ref_fft, CTF& ctf, Image& template_conj_fft,
                                   float& sigma_signal_post_ctf, int& padded_dim, float& pixel_size,
                                   bool save_debug_ref) {
        pixel_size = native_pixel_size * float(native_dim) / float(binned_dim);

        Image binned_ref;
        binned_ref.CopyFrom(&perfect_ref);
        if ( binned_dim != native_dim ) {
            binned_ref.ForwardFFT( );
            binned_ref.Resize(binned_dim, binned_dim, 1);
            binned_ref.BackwardFFT( );
        }
        // Zero mean, unit variance so template and target share a scale (raw base_img values ~1e-5
        // would otherwise make the correlation magnitude negligible and the noise ratios meaningless).
        binned_ref.ZeroFloatAndNormalize( );

        ref_fft.CopyFrom(&binned_ref);
        ref_fft.ForwardFFT( );

        ctf = CTF(ctf_voltage, ctf_cs, ctf_amp_contrast, defocus_value, defocus_value, 0.0f, pixel_size, 0.0f);

        // The perfect reference AS APPLIED in the cross-correlation: CTF-filtered, normalized, real.
        Image ctf_ref;
        ctf_ref.CopyFrom(&ref_fft);
        ctf_ref.ApplyCTF(ctf);
        ctf_ref.BackwardFFT( );
        ctf_ref.ZeroFloatAndNormalize( );
        if ( save_debug_ref )
            ctf_ref.QuickAndDirtyWriteSlice(debug_ref_ctf_filename, 1, true, pixel_size);

        // Correlation box = 2x the binned image, rounded up to a 5-smooth even size.
        padded_dim = ReturnClosestFactorizedUpper(2 * binned_dim, 5, true);

        // Template = conj(FFT of the prepared CTF reference). Reused for every replicate; never mutated.
        prepare_correlation_input(ctf_ref, template_conj_fft, padded_dim);
        template_conj_fft.Conj( );

        // Clean post-CTF signal std (dim D, no taper/pad). PhaseShift preserves power, so the post-CTF
        // signal power is independent of the applied sub-pixel shift.
        Image clean_ctf_signal;
        clean_ctf_signal.CopyFrom(&ref_fft);
        clean_ctf_signal.ApplyCTF(ctf);
        clean_ctf_signal.BackwardFFT( );
        sigma_signal_post_ctf = sqrtf(clean_ctf_signal.ReturnVarianceOfRealValues( ));
    };

    // One replicate: build a shifted, CTF-distorted, doubly-noised target from the clean reference
    // FFT, prepare it (zero-mean, taper, 2x pad), cross-correlate against the prepared conjugated
    // template, and read the uncorrected (integer) and sampling-corrected (upsampled) peak heights
    // and the measured offset from box center.
    auto run_one = [&](Image& ref_fft, CTF& ctf, Image& template_conj_fft, float sigma_signal_post_ctf,
                       int padded_dim, float pixel_size, float shift_x, float shift_y, bool save_debug_noisy,
                       float& found_dx, float& found_dy, float& integer_height, float& corrected_height,
                       float& box_center_value, float& fwhm_x_binned_px, float& fwhm_y_binned_px,
                       int& upsample_status_out) {
        Image target;
        target.CopyFrom(&ref_fft); // Fourier space
        target.PhaseShift(shift_x, shift_y, 0.0f);
        target.BackwardFFT( );
        // Pre-CTF noise on the pure signal normalized to unit variance -> genuine 1:1 (variance 1).
        // AddNoise draws a fresh, independent, time-seeded realization each call.
        target.ZeroFloatAndNormalize( );
        target.AddNoise(NoiseType::GAUSSIAN, 0.0f, noise_sigma_before_ctf);
        target.ForwardFFT( );
        target.ApplyCTF(ctf);
        target.BackwardFFT( );
        // Post-CTF noise scaled by the clean post-CTF SIGNAL std (not the composite std), so the signal
        // component sits at unit variance; the "1:N" add is then a true ratio relative to the signal.
        target.AddConstant(-target.ReturnAverageOfRealValues( ));
        target.DivideByConstant(sigma_signal_post_ctf);
        target.AddNoise(NoiseType::GAUSSIAN, 0.0f, noise_sigma_after_ctf);

        if ( save_debug_noisy )
            target.QuickAndDirtyWriteSlice(debug_img_filename, 1, true, pixel_size);

        // Prepare (zero-mean, taper, 2x pad) then cross-correlate. Multiply (not conjugate-multiply):
        // the template is already conjugated, so this yields fft(target) * conj(fft(ctf_ref)).
        target.ZeroFloatAndNormalize( );
        Image target_padded;
        prepare_correlation_input(target, target_padded, padded_dim);
        target_padded.MultiplyPixelWise(template_conj_fft);

        // Center the zero-lag peak: for even dims SwapRealSpaceQuadrants applies the same N/2 phase
        // shift regardless of the flag and toggles object_is_centred_in_box false -> true, which the
        // peak finders assert.
        target_padded.object_is_centred_in_box = false;
        target_padded.SwapRealSpaceQuadrants( );
        target_padded.BackwardFFT( );

        // Correlation value at the KNOWN zero-lag location (box center). At zero applied offset the
        // true signal peak sits here on-grid, so averaging this over replicates is an unbiased
        // (zero-mean-noise) estimator of the true peak height. Real-space, centered.
        box_center_value = target_padded.ReturnRealPixelFromPhysicalCoord(
                target_padded.physical_address_of_box_center_x, target_padded.physical_address_of_box_center_y, 0);

        const Peak         integer_peak = target_padded.FindPeakWithIntegerCoordinates( );
        std::vector<Peak>  peak_list;
        std::vector<Peak>  upsampled_peak_list;
        std::vector<float> fwhm_x_px;
        std::vector<float> fwhm_y_px;
        std::vector<int>   upsample_status;
        target_padded.FindPeakWithIntegerCoordinatesForManyPeaksSweep(
                peak_list, upsampled_peak_list, fwhm_x_px, fwhm_y_px, upsample_status,
                0.5f * integer_peak.value, cistem::match_template::PEAK_THRESHOLD_SCALE, 10.0f, 0,
                sweep_original_peak_size, sweep_padding_multiplier, sweep_upsample_factor,
                sweep_padding_mode, sweep_width_fraction, 0.0f); // fwhm_baseline 0 -> physical half-max above the ~0 correlation floor
        MyDebugAssertTrue(! peak_list.empty( ), "No peak crossed threshold in the correlation map");

        found_dx         = (peak_list[0].x - float(target_padded.physical_address_of_box_center_x)) + upsampled_peak_list[0].x;
        found_dy         = (peak_list[0].y - float(target_padded.physical_address_of_box_center_y)) + upsampled_peak_list[0].y;
        integer_height   = peak_list[0].value; // uncorrected, at the integer pixel
        corrected_height = upsampled_peak_list[0].value; // after sub-pixel upsampling correction
        // Per-peak FWHM of the winning peak (index 0), upsampled px -> binned px; the -1 sentinel
        // passes through when the walk hit the tile edge or upsampling did not run for that peak.
        fwhm_x_binned_px    = (! fwhm_x_px.empty( ) && fwhm_x_px[0] > 0.0f) ? fwhm_x_px[0] / float(sweep_upsample_factor) : -1.0f;
        fwhm_y_binned_px    = (! fwhm_y_px.empty( ) && fwhm_y_px[0] > 0.0f) ? fwhm_y_px[0] / float(sweep_upsample_factor) : -1.0f;
        upsample_status_out = upsample_status.empty( ) ? -1 : upsample_status[0];
    };

    // Split a displacement magnitude into per-axis shifts by mode (random draws a fresh azimuth).
    auto compute_shift = [&](float magnitude, float& shift_x, float& shift_y) {
        if ( shift_mode == SHIFT_X ) {
            shift_x = magnitude;
            shift_y = 0.0f;
        }
        else if ( shift_mode == SHIFT_Y ) {
            shift_x = 0.0f;
            shift_y = magnitude;
        }
        else if ( shift_mode == SHIFT_RANDOM ) {
            const float theta = (global_random_number_generator.GetUniformRandom( ) + 1.0f) * pi_f; // [0, 2pi]
            shift_x           = magnitude * cosf(theta);
            shift_y           = magnitude * sinf(theta);
        }
        else { // SHIFT_DIAGONAL
            const float d = magnitude / sqrtf(2.0f);
            shift_x       = d;
            shift_y       = d;
        }
    };

    // ---- warmup: how many replicates does the noisy measurement need? --------------------------
    if ( command_line_parser.Found("warmup") ) {
        if ( binned_dims.empty( ) ) {
            wxPrintf("No valid pixel sizes for warmup.\n");
            return true;
        }
        const int   warm_binned_dim = binned_dims[binned_dims.size( ) / 2];
        const float warm_magnitude  = 0.25f;
        const int   n_warm          = 300;

        Image ref_fft, template_conj_fft;
        CTF   ctf;
        float sigma_signal_post_ctf, pixel_size;
        int   padded_dim;
        build_pixel_context(warm_binned_dim, ctf_defocus, ref_fft, ctf, template_conj_fft, sigma_signal_post_ctf, padded_dim, pixel_size, false);

        NumericTextFile warm_file("build/scalloping_warmup.txt", OPEN_TO_WRITE, 4);
        warm_file.WriteCommentLine("warmup pixel_size %.4f magnitude %.4f shift_mode %s", pixel_size, warm_magnitude, shift_mode_name);
        warm_file.WriteCommentLine("replicate corrected_height running_mean running_sem");
        wxPrintf("\n# warmup px=%.4f magnitude=%.3f N=%i\n# n corrected running_mean running_sem\n", pixel_size, warm_magnitude, n_warm);

        double sum = 0.0, sum_sq = 0.0;
        for ( int rep = 0; rep < n_warm; rep++ ) {
            float shift_x, shift_y;
            compute_shift(warm_magnitude, shift_x, shift_y);
            float found_dx, found_dy, integer_height, corrected_height, box_center_value;
            float warm_fwhm_x, warm_fwhm_y;
            int   warm_status;
            run_one(ref_fft, ctf, template_conj_fft, sigma_signal_post_ctf, padded_dim, pixel_size,
                    shift_x, shift_y, false, found_dx, found_dy, integer_height, corrected_height,
                    box_center_value, warm_fwhm_x, warm_fwhm_y, warm_status);
            sum += corrected_height;
            sum_sq += double(corrected_height) * double(corrected_height);
            const int    n      = rep + 1;
            const double mean   = sum / n;
            const double var    = n > 1 ? (sum_sq - sum * sum / n) / (n - 1) : 0.0;
            const double sem    = n > 1 ? sqrt(var / n) : 0.0;
            float        row[4] = {float(n), corrected_height, float(mean), float(sem)};
            warm_file.WriteLine(row);
            if ( n % 10 == 0 || n == n_warm )
                wxPrintf("%i %.5f %.5f %.5f\n", n, corrected_height, float(mean), float(sem));
        }
        warm_file.Close( );
        wxPrintf("\nWrote build/scalloping_warmup.txt\n");
        return true;
    }

    // ---- report total SNR per pixel size (for figure annotation) --------------------------------
    // Total power SNR at the correlation input = signal_power : total_noise_power. The clean CTF
    // reference is normalized to unit variance, so the post-CTF noise contributes its nominal power
    // directly; the pre-CTF noise contributes only after passing through the CTF (which reshapes its
    // spectrum), so its post-CTF power is measured, not assumed. The pre/post split is a methods
    // detail; this single number characterizes how hard a cell is.
    if ( command_line_parser.Found("report-snr") ) {
        NumericTextFile snr_file("build/scalloping_snr.txt", OPEN_TO_WRITE, 2);
        snr_file.WriteCommentLine("noise_power_before_ctf %.4f noise_power_after_ctf %.4f", noise_power_before_ctf, noise_power_after_ctf);
        snr_file.WriteCommentLine("pixel_size total_power_snr");
        wxPrintf("\n# pixel_size total_power_snr\n");
        for ( const int bd : binned_dims ) {
            Image ref_fft, template_conj_fft;
            CTF   ctf;
            float sigma_signal_post_ctf, pixel_size;
            int   padded_dim;
            build_pixel_context(bd, ctf_defocus, ref_fft, ctf, template_conj_fft, sigma_signal_post_ctf, padded_dim, pixel_size, false);

            // Variance of a pre-CTF noise realization after the CTF, relative to the clean post-CTF
            // signal variance (which the target normalization sets to 1). PhaseShift/normalization make
            // the signal power 1; post-CTF noise adds noise_power_after directly.
            Image pre_noise;
            pre_noise.Allocate(bd, bd, 1, true);
            pre_noise.FillWithNoise(NoiseType::GAUSSIAN, 0.0f, noise_sigma_before_ctf);
            pre_noise.ForwardFFT( );
            pre_noise.ApplyCTF(ctf);
            pre_noise.BackwardFFT( );
            const float pre_noise_power = pre_noise.ReturnVarianceOfRealValues( ) / (sigma_signal_post_ctf * sigma_signal_post_ctf);
            const float total_power_snr = 1.0f / (pre_noise_power + noise_power_after_ctf);

            float row[2] = {pixel_size, total_power_snr};
            snr_file.WriteLine(row);
            wxPrintf("%.4f %.5f\n", pixel_size, total_power_snr);
        }
        snr_file.Close( );
        wxPrintf("\nWrote build/scalloping_snr.txt\n");
        return true;
    }

    // ---- main sweep: (defocus) x (pixel size) x (offset magnitude) x (replicate) ------------------
    const int       records_per_line = 14;
    NumericTextFile output_file(output_data_filename, OPEN_TO_WRITE, records_per_line);
    output_file.WriteCommentLine("noise_before_ctf %.4f noise_after_ctf %.4f", noise_sigma_before_ctf, noise_sigma_after_ctf);
    output_file.WriteCommentLine("shift_mode %s base_upsample_factor %.3f n_replicates %i", shift_mode_name, base_upsample_factor, n_replicates);
    output_file.WriteCommentLine("upsample_window %i padding_mult %i upsample_factor %i padding_mode %i width_fraction %.3f",
                                 sweep_original_peak_size, sweep_padding_multiplier, sweep_upsample_factor, sweep_padding_mode, sweep_width_fraction);
    output_file.WriteCommentLine("defocus_sweep %i n_defocus %i (fwhm columns are binned px; multiply by pixel_size for Angstroms; -1 = FWHM unavailable)",
                                 defocus_sweep ? 1 : 0, int(defocus_values.size( )));
    output_file.WriteCommentLine("pixel_size replicate applied_magnitude applied_dx applied_dy found_dx found_dy integer_peak_height corrected_peak_height box_center_value defocus fwhm_x_binned_px fwhm_y_binned_px upsample_status");

    bool debug_img_saved = false;

    for ( const float defocus_value : defocus_values ) {
        for ( size_t bin_idx = 0; bin_idx < binned_dims.size( ); bin_idx++ ) {
            Image      ref_fft, template_conj_fft;
            CTF        ctf;
            float      sigma_signal_post_ctf, pixel_size;
            int        padded_dim;
            const bool save_ref = debug_output && bin_idx == 0 && defocus_value == defocus_values.front( );
            build_pixel_context(binned_dims[bin_idx], defocus_value, ref_fft, ctf, template_conj_fft, sigma_signal_post_ctf, padded_dim, pixel_size, save_ref);

            for ( const float offset_magnitude : sub_pixel_offset_magnitudes ) {
                for ( int rep = 0; rep < n_replicates; rep++ ) {
                    float shift_x, shift_y;
                    compute_shift(offset_magnitude, shift_x, shift_y);
                    const bool save_this = debug_output && ! debug_img_saved;
                    float      found_dx, found_dy, integer_peak_height, corrected_height, box_center_value;
                    float      fwhm_x_binned_px, fwhm_y_binned_px;
                    int        upsample_status_val;
                    run_one(ref_fft, ctf, template_conj_fft, sigma_signal_post_ctf, padded_dim, pixel_size,
                            shift_x, shift_y, save_this, found_dx, found_dy, integer_peak_height, corrected_height,
                            box_center_value, fwhm_x_binned_px, fwhm_y_binned_px, upsample_status_val);
                    if ( save_this )
                        debug_img_saved = true;

                    float record[14] = {pixel_size, float(rep), offset_magnitude, shift_x, shift_y,
                                        found_dx, found_dy, integer_peak_height, corrected_height, box_center_value,
                                        defocus_value, fwhm_x_binned_px, fwhm_y_binned_px, float(upsample_status_val)};
                    output_file.WriteLine(record);
                }
                wxPrintf("defocus=%.0f px=%.3f magnitude=%.3f  (%i replicates)\n", defocus_value, pixel_size, offset_magnitude, n_replicates);
            }
        }
    }

    output_file.Close( );
    wxPrintf("\nWrote %s\n", output_data_filename.c_str( ));
#endif // cisTEM_test_scalloping

    return true;
}
