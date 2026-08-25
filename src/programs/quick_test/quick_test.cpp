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

// ---------------------------------------------------------------------------------------------
// Peak-sampling experiment: synthetic-image generator.
// Design: /results/FastFFT/scalloping_experiment_v2/DESIGN_v3_for_review.md, sections 2 and 7.1.
// Added 2026-08-25 (experiment branch peak_sampling_v3).
//
// Activated by --gen-volume <mrc>; without it quick_test behaves as before. Everything the
// generator needs is read from the command line (no interactive prompts), so a batch of images is
// reproducible from the recorded generation_params.txt alone.
// ---------------------------------------------------------------------------------------------
struct PeakSamplingGeneratorParams {
    wxString volume_filename;
    wxString output_directory;
    int      n_images       = 100;
    int      image_size     = 4096;
    int      cells_per_side = 8;
    int      cell_size      = 512;
    int      jitter         = 64; // integer jitter of the particle centre inside its cell, +/- this many pixels
    // sigma_pre / sigma_post are relative to a unit-variance clean projection box (pre-CTF) and to a
    // unit-std clean post-CTF projection box (post-CTF). No defensible default exists yet: the pair is
    // picked from 2-3 candidate runs at hrl 2.0 (design section 2), so both are REQUIRED on the command
    // line. Constraint from the design: the hrl 2.0 scaled-MIP peaks should sit around real apoferritin
    // values (~12-15) and the raw CC must stay below 22.5, because the GPU MIP kernel drops raw values
    // outside [-12.5, 22.5) from sum, sum^2 and the MIP max. Too little noise pushes raw CC over the cap.
    float    sigma_pre          = -1.0f;
    float    sigma_post         = -1.0f;
    float    defocus            = 6000.0f; // A, defocus1 = defocus2, astigmatism 0
    float    kv                 = 300.0f;
    float    cs                 = 2.7f; // mm
    float    ac                 = 0.07f;
    float    pixel_size         = 1.0f; // A
    wxString seed_prefix        = "";
    bool     first_image_stages = false;
};

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

    bool ReadPeakSamplingGeneratorOptions(PeakSamplingGeneratorParams& p);
    bool RunPeakSamplingImageGenerator( );

  private:
};

IMPLEMENT_APP(QuickTestApp)

// Optional command-line stuff
void QuickTestApp::AddCommandLineOptions( ) {
    command_line_parser.AddLongSwitch("disable-user-input", "Disable interactive user input prompts. Default false");

    // Peak-sampling generator (design v3 sections 2 / 7.1). Presence of --gen-volume switches the
    // program into generator mode.
    command_line_parser.AddLongOption("gen-volume", "Reference volume (MRC, cubic). Presence of this option runs the synthetic-image generator and returns.", wxCMD_LINE_VAL_STRING);
    command_line_parser.AddLongOption("gen-out-dir", "Output directory for img_<i>.mrc, img_<i>_truth.txt and generation_params.txt (created if missing).", wxCMD_LINE_VAL_STRING);
    command_line_parser.AddLongOption("gen-n-images", "Number of images to generate. Default 100.", wxCMD_LINE_VAL_NUMBER);
    command_line_parser.AddLongOption("gen-image-size", "Square image size in pixels. Default 4096.", wxCMD_LINE_VAL_NUMBER);
    command_line_parser.AddLongOption("gen-cells-per-side", "Cells per side (particles per image = cells^2). Default 8.", wxCMD_LINE_VAL_NUMBER);
    command_line_parser.AddLongOption("gen-cell-size", "Cell size in pixels; cells_per_side * cell_size must equal image_size. Default 512.", wxCMD_LINE_VAL_NUMBER);
    command_line_parser.AddLongOption("gen-jitter", "Integer jitter of the particle centre inside the cell, uniform in [-jitter, +jitter]^2. Default 64.", wxCMD_LINE_VAL_NUMBER);
    // See the raw-CC cap note on PeakSamplingGeneratorParams::sigma_pre for how these are chosen.
    command_line_parser.AddLongOption("gen-sigma-pre", "REQUIRED. Pre-CTF Gaussian noise sigma_b (relative to a unit-variance clean projection box).", wxCMD_LINE_VAL_DOUBLE);
    command_line_parser.AddLongOption("gen-sigma-post", "REQUIRED. Post-CTF Gaussian noise sigma_a (relative to a unit-std clean post-CTF projection box).", wxCMD_LINE_VAL_DOUBLE);
    command_line_parser.AddLongOption("gen-defocus", "Defocus in Angstroms (defocus1 = defocus2, no astigmatism). Default 6000.", wxCMD_LINE_VAL_DOUBLE);
    command_line_parser.AddLongOption("gen-kv", "Acceleration voltage in kV. Default 300.", wxCMD_LINE_VAL_DOUBLE);
    command_line_parser.AddLongOption("gen-cs", "Spherical aberration in mm. Default 2.7.", wxCMD_LINE_VAL_DOUBLE);
    command_line_parser.AddLongOption("gen-ac", "Amplitude contrast. Default 0.07.", wxCMD_LINE_VAL_DOUBLE);
    command_line_parser.AddLongOption("gen-pixel-size", "Pixel size in Angstroms (written to the MRC headers, used for the CTF). Default 1.0.", wxCMD_LINE_VAL_DOUBLE);
    command_line_parser.AddLongOption("gen-seed-prefix", "String prepended to every RNG seed (seed = prefix + image name + stage suffix). Default empty.", wxCMD_LINE_VAL_STRING);
    command_line_parser.AddLongSwitch("gen-first-image-stages", "For image 000 also write img_000_stages.mrc (6-slice stack) and img_000_points.txt (IMOD point2model input).");
}

// override the DoInteractiveUserInput

void QuickTestApp::DoInteractiveUserInput( ) {
    // This flag allows skipping interactive prompts, useful for automated testing with Copilot.
    if ( command_line_parser.FoundSwitch("disable-user-input") ) {
        std::cout << "Skipping interactive user input as per command line flag." << std::endl;
        return;
    }
    // Generator mode is fully command-line driven; no prompts.
    if ( command_line_parser.Found("gen-volume") ) {
        wxPrintf("Peak-sampling generator mode (--gen-volume present): skipping interactive user input.\n");
        return;
    }
    UserInput* my_input = new UserInput("QuickTest", 2.0);

    idx                           = my_input->GetIntFromUser("Index", "", "", 0, 1000);
    input_starfile_filename.at(0) = my_input->GetFilenameFromUser("Input starfile filename 1", "", "", false);
    input_starfile_filename.at(1) = my_input->GetFilenameFromUser("Input starfile filename 2", "", "", false);
    symmetry_symbol               = my_input->GetSymmetryFromUser("Particle symmetry", "The assumed symmetry of the particle to be reconstructed", "C1");

    delete my_input;
}

bool QuickTestApp::ReadPeakSamplingGeneratorOptions(PeakSamplingGeneratorParams& p) {
    wxString temp_string;
    long     temp_long;
    double   temp_double = 0.0;

    if ( ! command_line_parser.Found("gen-volume", &temp_string) )
        return false;
    p.volume_filename = temp_string;

    if ( ! command_line_parser.Found("gen-out-dir", &temp_string) ) {
        wxPrintf("Error: --gen-out-dir is required in generator mode.\n");
        return false;
    }
    p.output_directory = temp_string;

    if ( command_line_parser.Found("gen-n-images", &temp_long) )
        p.n_images = int(temp_long);
    if ( command_line_parser.Found("gen-image-size", &temp_long) )
        p.image_size = int(temp_long);
    if ( command_line_parser.Found("gen-cells-per-side", &temp_long) )
        p.cells_per_side = int(temp_long);
    if ( command_line_parser.Found("gen-cell-size", &temp_long) )
        p.cell_size = int(temp_long);
    if ( command_line_parser.Found("gen-jitter", &temp_long) )
        p.jitter = int(temp_long);

    bool have_sigma_pre  = command_line_parser.Found("gen-sigma-pre", &temp_double);
    p.sigma_pre          = float(temp_double);
    bool have_sigma_post = command_line_parser.Found("gen-sigma-post", &temp_double);
    p.sigma_post         = float(temp_double);
    if ( ! have_sigma_pre || ! have_sigma_post ) {
        wxPrintf("Error: --gen-sigma-pre and --gen-sigma-post are both required (no defensible default yet; see the comment on PeakSamplingGeneratorParams).\n");
        return false;
    }
    // > 0 (not >= 0): std::normal_distribution requires stddev > 0, and the design has no
    // noiseless images in any case.
    if ( p.sigma_pre <= 0.0f || p.sigma_post <= 0.0f ) {
        wxPrintf("Error: sigma values must be > 0 (got pre %f, post %f).\n", p.sigma_pre, p.sigma_post);
        return false;
    }

    if ( command_line_parser.Found("gen-defocus", &temp_double) )
        p.defocus = float(temp_double);
    if ( command_line_parser.Found("gen-kv", &temp_double) )
        p.kv = float(temp_double);
    if ( command_line_parser.Found("gen-cs", &temp_double) )
        p.cs = float(temp_double);
    if ( command_line_parser.Found("gen-ac", &temp_double) )
        p.ac = float(temp_double);
    if ( command_line_parser.Found("gen-pixel-size", &temp_double) )
        p.pixel_size = float(temp_double);
    if ( command_line_parser.Found("gen-seed-prefix", &temp_string) )
        p.seed_prefix = temp_string;
    p.first_image_stages = command_line_parser.FoundSwitch("gen-first-image-stages");

    if ( p.n_images < 1 || p.image_size < 2 || p.cells_per_side < 1 || p.cell_size < 1 || p.jitter < 0 ) {
        wxPrintf("Error: n_images, image_size, cells_per_side and cell_size must be positive and jitter >= 0.\n");
        return false;
    }
    if ( p.cells_per_side * p.cell_size != p.image_size ) {
        wxPrintf("Error: cells_per_side * cell_size (%i * %i = %i) must equal image_size (%i).\n", p.cells_per_side, p.cell_size, p.cells_per_side * p.cell_size, p.image_size);
        return false;
    }
    MyDebugAssertTrue(p.cells_per_side * p.cell_size == p.image_size, "cells_per_side * cell_size != image_size");
    return true;
}

bool QuickTestApp::RunPeakSamplingImageGenerator( ) {
    PeakSamplingGeneratorParams p;
    if ( ! ReadPeakSamplingGeneratorOptions(p) ) {
        SendErrorAndCrash("Peak-sampling generator: invalid or missing options (see messages above).");
        return false;
    }

    // ---- reference volume ---------------------------------------------------------------------
    ImageFile volume_file(p.volume_filename.ToStdString( ), false);
    Image     volume;
    volume.ReadSlices(&volume_file, 1, volume_file.ReturnZSize( ));
    const float header_pixel_size = volume_file.ReturnPixelSize( );
    if ( ! volume.IsCubic( ) ) {
        SendErrorAndCrash(wxString::Format("Reference volume must be cubic (got %i x %i x %i).", volume.logical_x_dimension, volume.logical_y_dimension, volume.logical_z_dimension));
        return false;
    }
    if ( fabsf(header_pixel_size - p.pixel_size) > 1e-3f )
        wxPrintf("Warning: volume header pixel size %.4f A differs from --gen-pixel-size %.4f A; the option is used (design decouples volume choice from the code).\n", header_pixel_size, p.pixel_size);

    // Projection box = volume box (design section 2).
    const int box_size = volume.logical_x_dimension;

    // ---- geometry checks -------------------------------------------------------------------
    // A projection box spans [centre - box/2, centre - box/2 + box). Two neighbouring particles are
    // closest when one jitters +jitter and the other -jitter, i.e. centres cell_size - 2*jitter apart;
    // boxes cannot overlap when that separation is >= box_size. For the defaults 512 - 128 = 384 = box:
    // boxes touch, no overlap. The outermost centre is cell_size/2 - jitter from the image edge, so the
    // box stays inside the image when cell_size/2 - jitter - box_size/2 >= 0 (defaults: 256-64-192 = 0).
    const bool boxes_cannot_overlap = (p.cell_size - 2 * p.jitter) >= box_size;
    const bool boxes_inside_image   = (p.cell_size / 2 - p.jitter - box_size / 2) >= 0;
    MyDebugAssertTrue(boxes_cannot_overlap, "Projection boxes can overlap: cell %i - 2*jitter %i < box %i", p.cell_size, p.jitter, box_size);
    MyDebugAssertTrue(boxes_inside_image, "Projection boxes can leave the image: cell/2 %i - jitter %i - box/2 %i < 0", p.cell_size / 2, p.jitter, box_size / 2);
    if ( ! boxes_cannot_overlap || ! boxes_inside_image ) {
        SendErrorAndCrash(wxString::Format("Geometry rejected: cell %i, jitter %i, box %i (overlap ok: %i, inside ok: %i).", p.cell_size, p.jitter, box_size, int(boxes_cannot_overlap), int(boxes_inside_image)));
        return false;
    }

    // ---- output directory ----------------------------------------------------------------------
    if ( ! wxFileName::DirExists(p.output_directory) ) {
        if ( ! wxFileName::Mkdir(p.output_directory, 0777, wxPATH_MKDIR_FULL) ) {
            SendErrorAndCrash(wxString::Format("Could not create output directory %s", p.output_directory));
            return false;
        }
    }
    const wxString out = p.output_directory + "/";

    // ---- volume preparation for ExtractSlice (once) -------------------------------------------
    // Same prep as the CPU search path in match_template (ZeroCentralPixel + SwapRealSpaceQuadrants on
    // the Fourier-space volume). ReadSlices leaves object_is_centred_in_box = true; the swap in Fourier
    // space applies a phase ramp of N/2 per axis and toggles the flag to false, which ExtractSlice
    // asserts. The z component of that swap is a phase on kz only and does not touch the kz = 0
    // central section read below.
    volume.ForwardFFT( );
    volume.ZeroCentralPixel( );
    volume.SwapRealSpaceQuadrants( );

    // One projection at theta = phi = psi = 0 with a sub-pixel shift (dx, dy) in pixels.
    // ExtractSlice at identity: RotateCoords maps (kx, ky, 0) -> (kx, ky, 0) exactly, so
    // ReturnLinearInterpolatedFourier is evaluated at integer coordinates on the kz = 0 plane with
    // weights (1, 0): the exact central section, except that samples whose +1 neighbour would lie
    // beyond the upper complex bound (the last kx column / last +ky row) return 0, and the DC term is
    // set to 0 by ExtractSlice itself. apply_resolution_limit = false fills every other sample.
    // FFT-state chain: ExtractSlice leaves prj in Fourier space with object_is_centred_in_box = false;
    // SwapRealSpaceQuadrants (Fourier) shifts by N/2 per axis and sets the flag true, so the density
    // centre sits at physical_address_of_box_center = box/2 after the BackwardFFT; PhaseShift (Fourier)
    // multiplies by exp(-i 2 pi (dx kx + dy ky)/N), which for the FFTW forward sign convention moves the
    // density by +dx along +x (increasing array index) and +dy along +y.
    auto make_projection = [&](float dx, float dy, Image& prj) {
        prj.Allocate(box_size, box_size, 1, false);
        AnglesAndShifts angles(0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
        volume.ExtractSlice(prj, angles, 1.0f, false);
        prj.SwapRealSpaceQuadrants( );
        prj.PhaseShift(dx, dy, 0.0f);
        prj.BackwardFFT( );
        prj.AddConstant(-prj.ReturnAverageOfRealValuesOnEdges( ));
    };

    // ---- normalisation constants (once; the clean box is identical for every particle) ---------
    // pre_scale: multiplies the clean pasted image so one clean projection box has unit variance.
    // post_ctf_std: std of that unit-variance clean box after the same CTF (applied at the box size, as
    // the prior generator did; the CTF is sampled on the box's Fourier grid rather than the full
    // image's, a normalisation-only difference that is constant across images and recorded).
    Image clean_box;
    make_projection(0.0f, 0.0f, clean_box);
    const float clean_box_variance = clean_box.ReturnVarianceOfRealValues( );
    if ( clean_box_variance <= 0.0f ) {
        SendErrorAndCrash("Clean projection box has zero variance; is the volume empty?");
        return false;
    }
    const float pre_scale = 1.0f / sqrtf(clean_box_variance);

    // CTF exactly as match_template initialises its input CTF (CTF::Init with the pixel size; the
    // class converts to pixel units internally). Image::ApplyCTF multiplies each Fourier sample by
    // CTF::Evaluate = -sin(chi), chi = pi lambda k^2 (df - 0.5 lambda^2 k^2 Cs) + amplitude-contrast
    // term, so positive defocus (underfocus) gives a NEGATIVE low-frequency transfer: positive
    // density comes out dark, the cisTEM / real-micrograph convention.
    CTF ctf;
    ctf.Init(p.kv, p.cs, p.ac, p.defocus, p.defocus, 0.0f, 0.0f, 0.0f, 0.0f, p.pixel_size, 0.0f);

    Image clean_box_ctf;
    clean_box_ctf.CopyFrom(&clean_box);
    clean_box_ctf.MultiplyByConstant(pre_scale);
    clean_box_ctf.ForwardFFT( );
    clean_box_ctf.ApplyCTF(ctf);
    clean_box_ctf.BackwardFFT( );
    const float post_ctf_std = sqrtf(clean_box_ctf.ReturnVarianceOfRealValues( ));
    if ( post_ctf_std <= 0.0f ) {
        SendErrorAndCrash("Clean post-CTF projection box has zero std.");
        return false;
    }

    const int n_particles = p.cells_per_side * p.cells_per_side;

    // ---- generation_params.txt -----------------------------------------------------------------
    {
        FILE* fp = fopen((out + "generation_params.txt").ToStdString( ).c_str( ), "w");
        if ( fp == NULL ) {
            SendErrorAndCrash("Could not open generation_params.txt for writing.");
            return false;
        }
        fprintf(fp, "# Peak-sampling experiment synthetic images. Design: /results/FastFFT/scalloping_experiment_v2/DESIGN_v3_for_review.md sections 2, 7.1\n");
        fprintf(fp, "volume_filename %s\n", p.volume_filename.ToStdString( ).c_str( ));
        fprintf(fp, "volume_header_pixel_size %.6f\n", header_pixel_size);
        fprintf(fp, "volume_box_size %i\n", box_size);
        fprintf(fp, "projection_box_size %i\n", box_size);
        fprintf(fp, "output_directory %s\n", p.output_directory.ToStdString( ).c_str( ));
        fprintf(fp, "n_images %i\n", p.n_images);
        fprintf(fp, "image_size %i\n", p.image_size);
        fprintf(fp, "cells_per_side %i\n", p.cells_per_side);
        fprintf(fp, "cell_size %i\n", p.cell_size);
        fprintf(fp, "particles_per_image %i\n", n_particles);
        fprintf(fp, "jitter %i\n", p.jitter);
        fprintf(fp, "sigma_pre %.6f\n", p.sigma_pre);
        fprintf(fp, "sigma_post %.6f\n", p.sigma_post);
        fprintf(fp, "defocus_1 %.3f\n", p.defocus);
        fprintf(fp, "defocus_2 %.3f\n", p.defocus);
        fprintf(fp, "astigmatism_angle 0.0\n");
        fprintf(fp, "additional_phase_shift 0.0\n");
        fprintf(fp, "kv %.3f\n", p.kv);
        fprintf(fp, "cs %.4f\n", p.cs);
        fprintf(fp, "amplitude_contrast %.4f\n", p.ac);
        fprintf(fp, "pixel_size %.6f\n", p.pixel_size);
        fprintf(fp, "seed_prefix %s\n", p.seed_prefix.ToStdString( ).c_str( ));
        fprintf(fp, "first_image_stages %i\n", int(p.first_image_stages));
        fprintf(fp, "clean_box_variance_before_scaling %.8e\n", clean_box_variance);
        fprintf(fp, "pre_scale %.8e\n", pre_scale);
        fprintf(fp, "post_ctf_std_of_unit_variance_clean_box %.8e\n", post_ctf_std);
        fprintf(fp, "# Processing chain per image: paste clean projections -> multiply by pre_scale -> add N(0, sigma_pre) -> FFT, ApplyCTF, IFFT -> divide by post_ctf_std -> add N(0, sigma_post).\n");
        fprintf(fp, "# Per particle: ExtractSlice(identity) on the FFT'd, quadrant-swapped volume; SwapRealSpaceQuadrants; PhaseShift(dx, dy); BackwardFFT; subtract edge mean; InsertOtherImageAtSpecifiedPosition (additive, integer).\n");
        fprintf(fp, "# Coordinate convention: 0-based array index, x = fastest axis (column), y = row index as stored in the MRC (row 0 first in the file). Pixel centre = integer index.\n");
        fprintf(fp, "# InsertOtherImageAtSpecifiedPosition places the box centre (box/2) at image pixel (image/2 + wanted_x, image/2 + wanted_y); the generator passes wanted = int - image/2 so the box centre lands on (int_x, int_y).\n");
        fprintf(fp, "# PhaseShift(dx, dy) multiplies by exp(-i 2 pi (dx kx + dy ky)/N): positive dx moves the density towards increasing x index. true_x = int_x + dx, true_y = int_y + dy.\n");
        fprintf(fp, "# RNG: std::mt19937 seeded with std::hash<std::string>(seed). Seeds: <prefix><image>_geometry (per particle, in order: jitter_x int, jitter_y int, dx float, dy float), <prefix><image>_noise_pre, <prefix><image>_noise_post.\n");
        fprintf(fp, "# img_000_points.txt (if written): 'x y z' per particle in the same 0-based index convention, z = 0; IMOD's half-pixel / y-direction handling was not verified here, check the overlay visually.\n");
        fprintf(fp, "# Stage stack img_000_stages.mrc (if written), slices: 1 clean box before sub-pixel shift, 2 after shift (both box-size images pasted at the image centre), 3 clean full image (after pre_scale), 4 + pre-CTF noise, 5 after CTF and division by post_ctf_std, 6 final (+ post-CTF noise).\n");
        fclose(fp);
    }

    wxPrintf("Peak-sampling generator: volume %s (box %i), %i images of %i px, %i particles each, box variance %.4e, pre_scale %.4e, post_ctf_std %.4e\n",
             p.volume_filename, box_size, p.n_images, p.image_size, n_particles, clean_box_variance, pre_scale, post_ctf_std);

    // ---- per-image loop --------------------------------------------------------------------------
    Image full;
    Image prj;
    full.Allocate(p.image_size, p.image_size, 1, true);
    const int image_centre = full.physical_address_of_box_center_x; // == image_size / 2 (square image)

    for ( int image_counter = 0; image_counter < p.n_images; image_counter++ ) {
        const wxString    image_name     = wxString::Format("img_%03i", image_counter);
        const std::string seed_base      = (p.seed_prefix + image_name).ToStdString( );
        const bool        capture_stages = p.first_image_stages && image_counter == 0;

        RandomNumberGenerator rng_geometry(seed_base + "_geometry");
        RandomNumberGenerator rng_noise_pre(seed_base + "_noise_pre");
        RandomNumberGenerator rng_noise_post(seed_base + "_noise_post");

        full.SetToConstant(0.0f);
        full.is_in_real_space         = true;
        full.object_is_centred_in_box = true;

        std::vector<int>   cell_ix(n_particles), cell_iy(n_particles), int_x(n_particles), int_y(n_particles);
        std::vector<float> dx(n_particles), dy(n_particles);

        Image stage_box_before, stage_box_after, stage_clean, stage_pre_noise, stage_post_ctf;

        for ( int particle = 0; particle < n_particles; particle++ ) {
            cell_ix[particle]  = particle % p.cells_per_side;
            cell_iy[particle]  = particle / p.cells_per_side;
            const int jitter_x = rng_geometry.GetUniformRandomSTD<int>(-p.jitter, p.jitter); // inclusive both ends
            const int jitter_y = rng_geometry.GetUniformRandomSTD<int>(-p.jitter, p.jitter);
            dx[particle]       = rng_geometry.GetUniformRandomSTD<float>(-0.5f, 0.5f); // [-0.5, 0.5)
            dy[particle]       = rng_geometry.GetUniformRandomSTD<float>(-0.5f, 0.5f);
            int_x[particle]    = cell_ix[particle] * p.cell_size + p.cell_size / 2 + jitter_x;
            int_y[particle]    = cell_iy[particle] * p.cell_size + p.cell_size / 2 + jitter_y;

            make_projection(dx[particle], dy[particle], prj);
            // Box centre (box/2) -> image pixel (int_x, int_y): see the offset convention comment above.
            full.InsertOtherImageAtSpecifiedPosition(&prj, int_x[particle] - image_centre, int_y[particle] - image_centre, 0);

            if ( capture_stages && particle == 0 ) {
                stage_box_before.CopyFrom(&clean_box);
                stage_box_after.CopyFrom(&prj);
            }
        }

        // ---- noise chain (design section 2), once at full sampling ----------------------------
        full.MultiplyByConstant(pre_scale);
        if ( capture_stages )
            stage_clean.CopyFrom(&full);

        full.AddNoiseUsingGenerator(rng_noise_pre, cistem::NoiseType::GAUSSIAN, 0.0f, p.sigma_pre); // (mean, sigma)
        if ( capture_stages )
            stage_pre_noise.CopyFrom(&full);

        full.ForwardFFT( );
        full.ApplyCTF(ctf);
        full.BackwardFFT( );
        full.DivideByConstant(post_ctf_std);
        if ( capture_stages )
            stage_post_ctf.CopyFrom(&full);

        full.AddNoiseUsingGenerator(rng_noise_post, cistem::NoiseType::GAUSSIAN, 0.0f, p.sigma_post);

        // ---- outputs ----------------------------------------------------------------------------
        full.QuickAndDirtyWriteSlice((out + image_name + ".mrc").ToStdString( ), 1, true, p.pixel_size);

        {
            FILE* fp = fopen((out + image_name + "_truth.txt").ToStdString( ).c_str( ), "w");
            if ( fp == NULL ) {
                SendErrorAndCrash(wxString::Format("Could not open %s_truth.txt for writing.", image_name));
                return false;
            }
            fprintf(fp, "# %s ground truth. Coordinates: 0-based array index, x = fastest axis (column), y = row as stored in the MRC. Pixel centre = integer index.\n", image_name.ToStdString( ).c_str( ));
            fprintf(fp, "# int_x int_y: integer paste position (projection box centre box/2 lands here). dx dy: sub-pixel Fourier shift in pixels, uniform on [-0.5, 0.5); positive dx moves density to larger x.\n");
            fprintf(fp, "# true_x = int_x + dx, true_y = int_y + dy: centre of the pasted density in image pixel coordinates.\n");
            fprintf(fp, "# particle_id cell_ix cell_iy int_x int_y dx dy true_x true_y\n");
            for ( int particle = 0; particle < n_particles; particle++ ) {
                fprintf(fp, "%i %i %i %i %i %.6f %.6f %.6f %.6f\n", particle, cell_ix[particle], cell_iy[particle], int_x[particle], int_y[particle],
                        dx[particle], dy[particle], float(int_x[particle]) + dx[particle], float(int_y[particle]) + dy[particle]);
            }
            fclose(fp);
        }

        if ( capture_stages ) {
            // Stage stack: every slice must share the image size, so the two box-size stages are pasted
            // (additively onto zeros) at the image centre with the same InsertOtherImage convention.
            MRCFile stack_file((out + image_name + "_stages.mrc").ToStdString( ), true);
            Image   padded;
            padded.Allocate(p.image_size, p.image_size, 1, true);

            padded.SetToConstant(0.0f);
            padded.InsertOtherImageAtSpecifiedPosition(&stage_box_before, 0, 0, 0);
            padded.WriteSlice(&stack_file, 1);
            padded.SetToConstant(0.0f);
            padded.InsertOtherImageAtSpecifiedPosition(&stage_box_after, 0, 0, 0);
            padded.WriteSlice(&stack_file, 2);
            stage_clean.WriteSlice(&stack_file, 3);
            stage_pre_noise.WriteSlice(&stack_file, 4);
            stage_post_ctf.WriteSlice(&stack_file, 5);
            full.WriteSlice(&stack_file, 6);
            stack_file.SetPixelSize(p.pixel_size);
            stack_file.WriteHeader( );
            stack_file.CloseFile( );

            // IMOD point2model input: plain "x y z" triplets, one per particle centre. Convention (0-based
            // array index, pixel centre = integer index, y = MRC row) is documented in generation_params.txt.
            FILE* fp = fopen((out + image_name + "_points.txt").ToStdString( ).c_str( ), "w");
            if ( fp == NULL ) {
                SendErrorAndCrash("Could not open the points file for writing.");
                return false;
            }
            for ( int particle = 0; particle < n_particles; particle++ )
                fprintf(fp, "%.4f %.4f 0\n", float(int_x[particle]) + dx[particle], float(int_y[particle]) + dy[particle]);
            fclose(fp);
        }

        wxPrintf("Wrote %s (%i particles)\n", image_name, n_particles);
    }

    // Q (design section 3, continuity CC of image 000 against the prior harness) is NOT implemented
    // here. It would attach at this point, operating on img_000.mrc and the clean_box / ctf objects
    // above, once the prior harness processing is available on this branch.

    return true;
}

bool QuickTestApp::DoCalculation( ) {

    if ( command_line_parser.Found("gen-volume") )
        return RunPeakSamplingImageGenerator( );

#ifdef ENABLEGPU
        // DeviceManager gpuDev;
        // gpuDev.ListDevices( );

        // QuickTestGPU quick_test_gpu;
        // quick_test_gpu.callHelloFromGPU(idx);
#endif

    return true;
}
