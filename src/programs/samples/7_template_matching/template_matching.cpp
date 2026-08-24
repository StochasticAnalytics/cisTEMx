/**
 * 7_template_matching: end-to-end pipeline tests on real data.
 *
 * Unlike the other samples (which exercise GpuImage/core methods in-process), these
 * tests drive the REAL match_template_gpu binary through its interactive input on the
 * reference micrographs in PLASMONLABS_REF_IMAGES/TM_tests. That is deliberate: the
 * production orchestration (TemplateMatchingDataSizer, whitening, FastFFT input prep,
 * TemplateMatchingCore, peak extraction, output writing) is what a regression must
 * guard, and re-implementing it here would test a copy instead of the product.
 *
 * Tests:
 *  1. Apoferritin O-symmetry search regression (EMPIAR-10568, 2048^2 @ 0.7896 A):
 *     a full production-settings search (~10 s on a modern GPU); MIP/scaled-MIP
 *     statistics and the always-on _peak_info_ output are compared against stored
 *     baselines.
 *  2. GPU batch-size invariance: the same search with --gpu-batch-size-multiplier 1
 *     must reproduce the multiplier-4 (default) scaled MIP to floating-point noise.
 *  3. K3 rotated-geometry consistency (Yeast 60S, native 5760x4092 vs the pre-rotated
 *     4092x5760 twin, with the TEMPLATE co-rotated so both searches sample identical
 *     specimen-frame orientations): two self-paired legs, FastFFT and classic, each
 *     compared on scaled-MIP field statistics and the top-10 matched peaks. Guards the
 *     rotated-coordinate/padding bookkeeping and the layout-dependent kernels; the
 *     always-on asserts in the search ride along. Coarse grid - consistency, not
 *     sensitivity, is under test (co-rotation makes that valid).
 *
 * Baseline contract:
 *  - Baselines live at $PLASMONLABS_REF_IMAGES/TM_tests/Baselines/<name>.txt as
 *    "key = value" lines with a provenance header.
 *  - With PLASMONLABS_WRITE_BASELINES set (any non-empty value), each comparing test
 *    RUNS the pipeline and WRITES its measured values to that directory instead of
 *    comparing, reporting SKIPPED(baseline written). Generation and comparison share
 *    the same measurement code, so there is nothing to hand-gather.
 *  - Absent a baseline file (and not in write mode), the comparison is reported
 *    SKIPPED so fresh checkouts do not hard-fail.
 *  - Tolerances are deliberately loose (percent-level) to absorb GPU-model and
 *    driver variation in fp16 accumulation; they are tight enough to catch the
 *    defect classes seen on this branch (variance-vs-std scaling, dropped peaks,
 *    coordinate corruption).
 */

#ifdef ENABLEGPU
#include "../../../gpu/gpu_core_headers.h"
#else
#error "GPU is not enabled"
#include "../../../core/core_headers.h"
#endif

#include "../../../gpu/GpuImage.h"

#include "../common/common.h"
#include "template_matching.h"

#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>

namespace {

// ---------------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------------

// The samples binary and the cisTEM programs are built into the same bin directory.
wxString ReturnSiblingBinary(const char* name) {
    wxString self = wxStandardPaths::Get( ).GetExecutablePath( );
    wxString dir  = wxFileName(self).GetPath( );
    return dir + "/" + name;
}

struct TmSearchSettings {
    wxString image_path;
    wxString template_path;
    wxString output_prefix; // <temp>/<prefix>_...
    float    pixel_size;
    float    voltage_kV         = 300.0f;
    float    cs_mm              = 2.7f;
    float    amplitude_contrast = 0.1f;
    float    defocus_1;
    float    defocus_2;
    float    defocus_angle;
    float    phase_shift       = 0.0f;
    float    high_res_limit    = 3.0f;
    float    out_of_plane_step = 2.5f;
    float    in_plane_step     = 1.5f;
    wxString symmetry          = "C1";
    int      max_threads       = 1;
    int      batch_multiplier  = 0; // 0 = do not pass the option (binary default)
    bool     use_fast_fft      = true;
};

wxString OutName(const TmSearchSettings& s, const char* what) {
    return s.output_prefix + "_" + what + ".mrc";
}

// Drives the interactive input of match_template_gpu. The prompt ORDER must track
// MatchTemplateApp::DoInteractiveUserInput; the padding prompt is gate-dependent, and
// this TU sees the same GpuImage.h / cistem_config.h gates as the binary built from
// this tree, so the #ifdefs below stay in lockstep by construction.
bool RunMatchTemplateSearch(const TmSearchSettings& s, const wxString& temp_directory) {
    wxString binary = ReturnSiblingBinary("match_template_gpu");
    if ( ! DoesFileExist(binary) ) {
        wxPrintf("  (match_template_gpu not found next to the samples binary: %s)\n", binary.ToUTF8( ).data( ));
        return false;
    }

    // The histogram name must contain "_histogram_": match_template derives the
    // always-on peak-info output path from it by replacing _histogram_ with _peak_info_.
    wxString      stdin_path = s.output_prefix + "_stdin.txt";
    std::ofstream in(stdin_path.ToStdString( ));
    in << s.image_path << "\n"
       << s.template_path << "\n"
       << OutName(s, "mip") << "\n"
       << OutName(s, "scaled_mip") << "\n"
       << OutName(s, "psi") << "\n"
       << OutName(s, "theta") << "\n"
       << OutName(s, "phi") << "\n"
       << OutName(s, "defocus") << "\n"
       << OutName(s, "pixel_size") << "\n"
       << OutName(s, "corr_avg") << "\n"
       << OutName(s, "corr_std") << "\n"
       << s.output_prefix << "_histogram_1.txt" << "\n"
       << s.pixel_size << "\n"
       << s.voltage_kV << "\n"
       << s.cs_mm << "\n"
       << s.amplitude_contrast << "\n"
       << s.defocus_1 << "\n"
       << s.defocus_2 << "\n"
       << s.defocus_angle << "\n"
       << s.phase_shift << "\n"
       << s.high_res_limit << "\n"
       << s.out_of_plane_step << "\n"
       << s.in_plane_step << "\n"
       << 0.0f << "\n" // defocus search range
       << 0.0f << "\n" // defocus step (0 = no search)
       << 0.0f << "\n" // pixel size search range
       << 0.0f << "\n" // pixel size step (0 = no search)
#if defined(cisTEM_EXPERIMENTAL_3d_TEXTURE_ENABLE) && defined(cisTEM_USING_FastFFT) && cisTEM_EXPERIMENTAL_3d_TEXTURE_TYPE != 0
       << 512 << "\n" // FastFFT texture prep: absolute output box edge, >= template extent (480/384)
#else
       << 1.0f << "\n" // classic padding factor
#endif
       << 0.0f << "\n" // mask radius (0 = max)
       << s.symmetry << "\n"
       << "Yes" << "\n" // use GPU
#ifdef cisTEM_USING_FastFFT
       << (s.use_fast_fft ? "Yes" : "No") << "\n" // use FastFFT
#endif
       << s.max_threads << "\n"
       << "Yes" << "\n"; // peak sampling correction
    in.close( );

    wxString multiplier_option = "";
    if ( s.batch_multiplier > 0 )
        multiplier_option = wxString::Format(" --gpu-batch-size-multiplier=%i", s.batch_multiplier);

    wxString log_path = s.output_prefix + "_run_log.txt";
    wxString cmd      = wxString::Format("cd %s && %s%s < %s > %s 2>&1",
                                         temp_directory, binary, multiplier_option, stdin_path, log_path);

    cistem_timer::StopWatch timer;
    timer.start("match_template_gpu");
    int ret = system(cmd.ToUTF8( ).data( ));
    timer.lap("match_template_gpu");

    if ( ret != 0 ) {
        wxPrintf("  match_template_gpu exited with %i; last log lines (%s):\n", ret, log_path.ToUTF8( ).data( ));
        int sysret = system(wxString::Format("tail -5 %s", log_path).ToUTF8( ).data( ));
        (void)sysret;
        return false;
    }
    return true;
}

struct MipStats {
    float max_value;
    float mean;
    float std;
};

MipStats MeasureMip(const wxString& filename) {
    MipStats stats{0.f, 0.f, 0.f};
    Image    mip;
    mip.QuickAndDirtyReadSlice(filename.ToStdString( ), 1);
    stats.max_value = mip.ReturnMaximumValue( );
    // Central 80% box: the padding frame written back by the post-search resize is
    // exactly-zero and would dilute mean/std with a value that depends on frame size.
    EmpiricalDistribution<double> d = mip.ReturnDistributionOfRealValues(0.4f * float(mip.logical_x_dimension));
    stats.mean                      = float(d.GetSampleMean( ));
    stats.std                       = sqrtf(float(d.GetSampleVariance( )));
    return stats;
}

// The _peak_info_ file is a NumericTextFile: '#'/'C'-prefixed comment lines, one row per peak.
int CountPeakInfoRows(const wxString& filename) {
    std::ifstream f(filename.ToStdString( ));
    if ( ! f.good( ) )
        return -1;
    int         n = 0;
    std::string line;
    while ( std::getline(f, line) ) {
        if ( line.empty( ) || line[0] == '#' || line[0] == 'C' || line[0] == 'c' )
            continue;
        n++;
    }
    return n;
}

// -------------------------- rotation helpers (K3 pair) ---------------------------

// The two 90-degree in-plane mappings between the native (W x H) frame and the
// rotated (H x W) frame, 0-based physical coordinates. Which one produced the
// _rotate90 file on disk is determined empirically (DetermineRotationDirection).
inline void MapRotatedToNative(const int direction, const int rot_x, const int rot_y,
                               const int rot_nx, const int rot_ny, int& native_x, int& native_y) {
    if ( direction == 0 ) {
        native_x = rot_y;
        native_y = (rot_nx - 1) - rot_x;
    }
    else {
        native_x = (rot_ny - 1) - rot_y;
        native_y = rot_x;
    }
}

// Sample pixels of the rotated file against the native file under both mappings and
// return the one that reproduces it exactly (rot90 on disk is a lossless permutation).
// Returns -1 if neither matches (unexpected pair).
int DetermineRotationDirection(Image& native, Image& rotated) {
    const int             trials = 2048;
    int                   hits[2]{0, 0};
    RandomNumberGenerator sampler(pi_v<float>);
    for ( int direction = 0; direction < 2; direction++ ) {
        for ( int t = 0; t < trials; t++ ) {
            int rx = myroundint(sampler.GetUniformRandomSTD(0.f, float(rotated.logical_x_dimension - 1)));
            int ry = myroundint(sampler.GetUniformRandomSTD(0.f, float(rotated.logical_y_dimension - 1)));
            int nx, ny;
            MapRotatedToNative(direction, rx, ry, rotated.logical_x_dimension, rotated.logical_y_dimension, nx, ny);
            if ( rotated.real_values[rotated.ReturnReal1DAddressFromPhysicalCoord(rx, ry, 0)] ==
                 native.real_values[native.ReturnReal1DAddressFromPhysicalCoord(nx, ny, 0)] )
                hits[direction]++;
        }
    }
    if ( hits[0] == trials && hits[1] < trials )
        return 0;
    if ( hits[1] == trials && hits[0] < trials )
        return 1;
    wxPrintf("  rotation direction ambiguous/unmatched (hits %i / %i of %i)\n", hits[0], hits[1], trials);
    return -1;
}

// Write a copy of a cubic volume rotated 90 degrees about z with the SAME in-plane
// mapping as the image pair - a pure index permutation, no interpolation - so the
// rotated-image leg samples identical specimen-frame orientations on the same grid.
bool WriteCoRotatedTemplate(const wxString& in_path, const wxString& out_path, const int direction) {
    ImageFile volume_file;
    if ( ! volume_file.OpenFile(in_path.ToStdString( ), false) )
        return false;
    Image volume;
    volume.ReadSlices(&volume_file, 1, volume_file.ReturnNumberOfSlices( ));
    MyAssertTrue(volume.logical_x_dimension == volume.logical_y_dimension, "template must be square in x/y");

    Image rotated;
    rotated.Allocate(volume.logical_x_dimension, volume.logical_y_dimension, volume.logical_z_dimension, true);
    for ( int k = 0; k < volume.logical_z_dimension; k++ ) {
        for ( int j = 0; j < volume.logical_y_dimension; j++ ) {
            for ( int i = 0; i < volume.logical_x_dimension; i++ ) {
                int src_x, src_y;
                MapRotatedToNative(direction, i, j, rotated.logical_x_dimension, rotated.logical_y_dimension, src_x, src_y);
                rotated.real_values[rotated.ReturnReal1DAddressFromPhysicalCoord(i, j, k)] =
                        volume.real_values[volume.ReturnReal1DAddressFromPhysicalCoord(src_x, src_y, k)];
            }
        }
    }
    rotated.WriteSlicesAndFillHeader(out_path.ToStdString( ), volume_file.ReturnPixelSize( ));
    return true;
}

// -------------------------- peak table comparison --------------------------------

struct PeakRow {
    double x, y, height;
};

// peak_info columns: x_pos y_pos defocus corrected_peak_height original_score above_threshold sub_pixel_x sub_pixel_y
std::vector<PeakRow> ReadTopPeaks(const wxString& filename, size_t wanted_number) {
    std::vector<PeakRow> rows;
    std::ifstream        f(filename.ToStdString( ));
    std::string          line;
    while ( std::getline(f, line) ) {
        if ( line.empty( ) || line[0] == '#' || line[0] == 'C' || line[0] == 'c' )
            continue;
        std::istringstream ss(line);
        PeakRow            r;
        double             defocus;
        if ( ss >> r.x >> r.y >> defocus >> r.height )
            rows.push_back(r);
    }
    std::sort(rows.begin( ), rows.end( ), [](const PeakRow& a, const PeakRow& b) { return a.height > b.height; });
    if ( rows.size( ) > wanted_number )
        rows.resize(wanted_number);
    return rows;
}

// Match the rotated run's top peaks to the native run's by position (mapped through the
// known rotation), then compare heights. Positions may be in pixels or Angstroms
// depending on the writer; units are inferred from the image extent.
bool CompareTopPeaks(const std::vector<PeakRow>& native_peaks, std::vector<PeakRow> rotated_peaks,
                     const int direction, const int native_nx, const int native_ny, const float pixel_size,
                     const double match_radius_px, const double height_rel_tol, const char* label) {
    if ( native_peaks.empty( ) || rotated_peaks.empty( ) ) {
        wxPrintf("  %s: no peaks to compare (native %zu, rotated %zu)\n", label, native_peaks.size( ), rotated_peaks.size( ));
        return false;
    }

    // Infer units: coordinates exceeding the pixel extent must be Angstroms.
    double max_coordinate = 0.0;
    for ( auto& p : native_peaks )
        max_coordinate = std::max(max_coordinate, std::max(p.x, p.y));
    const double to_px  = (max_coordinate > double(std::max(native_nx, native_ny))) ? 1.0 / pixel_size : 1.0;
    const int    rot_nx = native_ny, rot_ny = native_nx;

    int    matched          = 0;
    double worst_height_rel = 0.0;
    for ( auto& rp : rotated_peaks ) {
        int mapped_x, mapped_y;
        MapRotatedToNative(direction, myroundint(float(rp.x * to_px)), myroundint(float(rp.y * to_px)), rot_nx, rot_ny, mapped_x, mapped_y);
        double         best_d2 = 1e30;
        const PeakRow* best    = nullptr;
        for ( auto& np : native_peaks ) {
            const double dx = np.x * to_px - mapped_x, dy = np.y * to_px - mapped_y;
            const double d2 = dx * dx + dy * dy;
            if ( d2 < best_d2 ) {
                best_d2 = d2;
                best    = &np;
            }
        }
        if ( best != nullptr && best_d2 <= match_radius_px * match_radius_px ) {
            matched++;
            worst_height_rel = std::max(worst_height_rel, std::abs(best->height - rp.height) / std::max(std::abs(best->height), 1e-30));
        }
    }

    const size_t expected       = std::min(native_peaks.size( ), rotated_peaks.size( ));
    const bool   enough_matched = (size_t(matched) * 10 >= expected * 8); // >= 80%
    const bool   heights_agree  = (worst_height_rel <= height_rel_tol);
    wxPrintf("  %s: matched %i of %zu peaks within %.0f px; worst height rel diff %.4f (tol %.4f)\n",
             label, matched, expected, match_radius_px, worst_height_rel, height_rel_tol);
    return enough_matched && heights_agree;
}

// -------------------------- baseline read/write ----------------------------------

wxString BaselineDir(const wxString& cistem_ref_dir) {
    return cistem_ref_dir + "/TM_tests/Baselines";
}

bool InBaselineWriteMode(wxString& write_dir, const wxString& cistem_ref_dir) {
    wxString env_value;
    if ( wxGetEnv("PLASMONLABS_WRITE_BASELINES", &env_value) && ! env_value.IsEmpty( ) ) {
        // "1"/"yes" and friends write to the canonical location; a path writes there.
        if ( env_value.StartsWith("/") )
            write_dir = env_value;
        else
            write_dir = BaselineDir(cistem_ref_dir);
        return true;
    }
    return false;
}

bool WriteBaseline(const wxString& path, const std::map<std::string, double>& values) {
    int sysret = system(wxString::Format("mkdir -p %s", wxFileName(path).GetPath( )).ToUTF8( ).data( ));
    (void)sysret;
    std::ofstream f(path.ToStdString( ));
    if ( ! f.good( ) )
        return false;
    f << "# samples_functional_testing baseline, written " << wxDateTime::Now( ).FormatISOCombined( ) << "\n";
    f << "# branch/version: " << CISTEM_VERSION_TEXT << "\n";
    for ( auto& kv : values )
        f << kv.first << " = " << std::setprecision(9) << kv.second << "\n";
    return true;
}

bool ReadBaseline(const wxString& path, std::map<std::string, double>& values) {
    std::ifstream f(path.ToStdString( ));
    if ( ! f.good( ) )
        return false;
    std::string line;
    while ( std::getline(f, line) ) {
        if ( line.empty( ) || line[0] == '#' )
            continue;
        std::istringstream ss(line);
        std::string        key, equals;
        double             value;
        if ( ss >> key >> equals >> value )
            values[key] = value;
    }
    return true;
}

bool CompareWithTolerance(const char* what, double measured, double baseline, double relative_tolerance) {
    double denominator = std::max(std::abs(baseline), 1e-30);
    double relative    = std::abs(measured - baseline) / denominator;
    bool   ok          = relative <= relative_tolerance;
    if ( ! ok )
        wxPrintf("  %s: measured %g vs baseline %g (rel diff %.3f > tol %.3f)\n", what, measured, baseline, relative, relative_tolerance);
    return ok;
}

} // namespace

// ---------------------------------------------------------------------------------
// runner
// ---------------------------------------------------------------------------------

void TemplateMatchingPipelineRunner(const wxString& temp_directory) {

    SamplesPrintTestStartMessage("Starting template matching pipeline tests:", false);

    wxString cistem_ref_dir = CheckForReferenceImages( );

    if ( ! DoesFileExist(ReturnSiblingBinary("match_template_gpu")) ) {
        SamplesTestSkip("template matching pipeline", "match_template_gpu binary not found in the samples bin directory");
        SamplesPrintEndMessage( );
        return;
    }

    TEST(DoApoferritinSearchRegressionTest(cistem_ref_dir, temp_directory));
    TEST(DoBatchSizeInvarianceTest(cistem_ref_dir, temp_directory));
    TEST(DoK3RotatedGeometryConsistencyTest(cistem_ref_dir, temp_directory));

    SamplesPrintEndMessage( );
}

// ---------------------------------------------------------------------------------
// 1. Apoferritin O-symmetry production-settings search vs stored baseline
// ---------------------------------------------------------------------------------

bool DoApoferritinSearchRegressionTest(const wxString& cistem_ref_dir, const wxString& temp_directory) {
    bool passed     = true;
    bool all_passed = true;

    SamplesBeginTest("Apoferritin O-sym search (production settings)", passed);

    // Metadata from TM_tests/SPA/Apoferritin/MetaData/Apoferritin.toml (EMPIAR-10568).
    TmSearchSettings s;
    s.image_path    = cistem_ref_dir + "/TM_tests/SPA/Apoferritin/Images/apoferritin_6000.mrc";
    s.template_path = cistem_ref_dir + "/TM_tests/SPA/Apoferritin/Templates/apo_ref_no_C_T_480.mrc";
    s.output_prefix = temp_directory + "/tm_apo";
    s.pixel_size    = 0.7896f;
    s.defocus_1     = 5252.0f;
    s.defocus_2     = 4965.0f;
    s.defocus_angle = -31.2f;
    s.symmetry      = "O";

    if ( ! DoesFileExist(s.image_path) ) {
        SamplesTestResultSkipped("apoferritin reference data not present under PLASMONLABS_REF_IMAGES");
        return true;
    }

    passed = RunMatchTemplateSearch(s, temp_directory);
    passed = passed && DoesFileExist(OutName(s, "scaled_mip"));
    all_passed &= passed;
    SamplesTestResult(passed);
    if ( ! passed )
        return false;

    SamplesBeginTest("Apoferritin search vs stored baseline", passed);

    MipStats mip        = MeasureMip(OutName(s, "mip"));
    MipStats scaled_mip = MeasureMip(OutName(s, "scaled_mip"));
    int      n_peaks    = CountPeakInfoRows(s.output_prefix + "_peak_info_1.txt");

    std::map<std::string, double> measured = {
            {"mip_max", mip.max_value},
            {"mip_central_mean", mip.mean},
            {"mip_central_std", mip.std},
            {"scaled_mip_max", scaled_mip.max_value},
            {"scaled_mip_central_std", scaled_mip.std},
            {"peak_info_rows", double(n_peaks)},
    };

    wxString write_dir;
    wxString baseline_path = BaselineDir(cistem_ref_dir) + "/samples_tm_apoferritin_6000.txt";
    if ( InBaselineWriteMode(write_dir, cistem_ref_dir) ) {
        bool written = WriteBaseline(write_dir + "/samples_tm_apoferritin_6000.txt", measured);
        SamplesTestResultSkipped(written ? "baseline written (PLASMONLABS_WRITE_BASELINES)" : "baseline WRITE FAILED");
        return written;
    }

    std::map<std::string, double> baseline;
    if ( ! ReadBaseline(baseline_path, baseline) ) {
        SamplesTestResultSkipped("no stored baseline (run once with PLASMONLABS_WRITE_BASELINES=1)");
        return true;
    }

    // Percent-level tolerances absorb GPU/driver fp16 variation; the defect classes this
    // guards (variance-vs-std MIP scaling, silently dropped peaks, corrupted statistics)
    // move these numbers by far more.
    passed = true;
    passed &= CompareWithTolerance("mip_max", measured["mip_max"], baseline["mip_max"], 0.02);
    passed &= CompareWithTolerance("mip_central_mean", measured["mip_central_mean"], baseline["mip_central_mean"], 0.05);
    passed &= CompareWithTolerance("mip_central_std", measured["mip_central_std"], baseline["mip_central_std"], 0.02);
    passed &= CompareWithTolerance("scaled_mip_max", measured["scaled_mip_max"], baseline["scaled_mip_max"], 0.02);
    passed &= CompareWithTolerance("scaled_mip_central_std", measured["scaled_mip_central_std"], baseline["scaled_mip_central_std"], 0.02);
    passed &= CompareWithTolerance("peak_info_rows", measured["peak_info_rows"], baseline["peak_info_rows"], 0.05);

    all_passed &= passed;
    SamplesTestResult(passed);

    return all_passed;
}

// ---------------------------------------------------------------------------------
// 2. --gpu-batch-size-multiplier must not change results
// ---------------------------------------------------------------------------------

bool DoBatchSizeInvarianceTest(const wxString& cistem_ref_dir, const wxString& temp_directory) {
    bool passed = true;

    SamplesBeginTest("GPU batch-size invariance (multiplier 1 vs 4)", passed);

    // Re-uses the apoferritin search from test 1 (multiplier 4 = binary default) and
    // repeats it with multiplier 1; the batch size only changes GPU-side work grouping,
    // so the scaled MIP must agree to fp16 accumulation noise.
    TmSearchSettings s;
    s.image_path       = cistem_ref_dir + "/TM_tests/SPA/Apoferritin/Images/apoferritin_6000.mrc";
    s.template_path    = cistem_ref_dir + "/TM_tests/SPA/Apoferritin/Templates/apo_ref_no_C_T_480.mrc";
    s.output_prefix    = temp_directory + "/tm_apo_bs1";
    s.pixel_size       = 0.7896f;
    s.defocus_1        = 5252.0f;
    s.defocus_2        = 4965.0f;
    s.defocus_angle    = -31.2f;
    s.symmetry         = "O";
    s.batch_multiplier = 1;

    wxString reference_scaled_mip = temp_directory + "/tm_apo_scaled_mip.mrc";
    if ( ! DoesFileExist(s.image_path) || ! DoesFileExist(reference_scaled_mip) ) {
        SamplesTestResultSkipped("requires the apoferritin data and a completed test-1 run");
        return true;
    }

    passed = RunMatchTemplateSearch(s, temp_directory);
    if ( passed ) {
        Image reference, repeat;
        reference.QuickAndDirtyReadSlice(reference_scaled_mip.ToStdString( ), 1);
        repeat.QuickAndDirtyReadSlice(OutName(s, "scaled_mip").ToStdString( ), 1);

        float mask_radius = 0.4f * float(reference.logical_x_dimension);
        reference.ZeroFloatAndNormalize(1.f, mask_radius);
        repeat.ZeroFloatAndNormalize(1.f, mask_radius);
        float score = reference.ReturnCorrelationCoefficientUnnormalized(repeat, mask_radius);
        passed      = (score > 0.999f);
        if ( ! passed )
            wxPrintf("  scaled MIP correlation multiplier-1 vs multiplier-4: %f\n", score);
    }

    SamplesTestResult(passed);
    return passed;
}

// ---------------------------------------------------------------------------------
// 3. K3 rotated-geometry consistency (Yeast 60S)
// ---------------------------------------------------------------------------------

bool DoK3RotatedGeometryConsistencyTest(const wxString& cistem_ref_dir, const wxString& temp_directory) {
    bool passed     = true;
    bool all_passed = true;

    // DESIGN NOTE (2026-08-24). The first version of this test rotated ONLY the image and
    // compared results - flawed at a coarse grid: an image-only rotation shifts every
    // particle's in-plane angle by 90 degrees, so the two searches sample DIFFERENT
    // specimen-frame orientations (and traverse near-ties in a different order), which
    // produced a deterministic ~3% max difference on the v1.2.26 A100 gate that was an
    // artifact of the test, not necessarily of the code. Now the TEMPLATE is co-rotated by
    // the same 90 degrees (lossless index permutation, direction determined empirically
    // from the image pair), so both searches sample identical specimen-frame orientations
    // on the same grid and must agree to floating point. Residual disagreement isolates
    // the layout-dependent paths: RotateForSpeed, padding bookkeeping, and (FastFFT leg
    // only) the decomposed-6144 kernels. The classic (non-FastFFT) leg pairs with itself
    // at its own prime-factored search size - never compared across paths, since the
    // sizes differ (5842-class vs 6144).
    SamplesBeginTest("K3 pair: rotation direction + co-rotated template", passed);

    const wxString native_image_path  = cistem_ref_dir + "/TM_tests/Yeast/Images/147_Mar12_12.21.27_159_0.mrc";
    const wxString rotated_image_path = cistem_ref_dir + "/TM_tests/Yeast/Images/147_Mar12_12.21.27_159_0_rotate90.mrc";
    const wxString template_path      = cistem_ref_dir + "/TM_tests/Yeast/Templates/6Q8Y_mature_60S.mrc";

    if ( ! DoesFileExist(native_image_path) || ! DoesFileExist(rotated_image_path) ) {
        SamplesTestResultSkipped("yeast K3 reference pair not present under PLASMONLABS_REF_IMAGES");
        return true;
    }

    Image native_image, rotated_image;
    native_image.QuickAndDirtyReadSlice(native_image_path.ToStdString( ), 1);
    rotated_image.QuickAndDirtyReadSlice(rotated_image_path.ToStdString( ), 1);
    const int native_nx = native_image.logical_x_dimension;
    const int native_ny = native_image.logical_y_dimension;

    const int direction = DetermineRotationDirection(native_image, rotated_image);
    native_image.Deallocate( );
    rotated_image.Deallocate( );

    const wxString rotated_template_path = temp_directory + "/tm_k3_template_rot90.mrc";
    passed                               = (direction >= 0) && WriteCoRotatedTemplate(template_path, rotated_template_path, direction);
    all_passed &= passed;
    SamplesTestResult(passed);
    if ( ! passed )
        return false;

    // Base settings: metadata from TM_tests/Yeast/MetaData/Yeast.toml. C1 on a 24 Mpx K3
    // frame is expensive and this guards coordinate handling, not sensitivity - coarse grid.
    TmSearchSettings base;
    base.image_path        = native_image_path;
    base.template_path     = template_path;
    base.pixel_size        = 1.06f;
    base.defocus_1         = 4147.0f;
    base.defocus_2         = 4147.0f;
    base.defocus_angle     = 49.2f;
    base.high_res_limit    = 4.0f;
    base.out_of_plane_step = 20.0f;
    base.in_plane_step     = 15.0f;

    struct Leg {
        const char* name;
        bool        fast_fft;
    };

    const Leg legs[] = {{"fastfft", true}, {"classic", false}};

    for ( auto& leg : legs ) {
        std::string search_name = std::string("K3 pair search (") + leg.name + ")";
        SamplesBeginTest(search_name.c_str( ), passed);

        TmSearchSettings native_leg = base;
        native_leg.use_fast_fft     = leg.fast_fft;
        native_leg.output_prefix    = temp_directory + "/tm_k3_native_" + leg.name;

        TmSearchSettings rotated_leg = native_leg;
        rotated_leg.image_path       = rotated_image_path;
        rotated_leg.template_path    = rotated_template_path;
        rotated_leg.output_prefix    = temp_directory + "/tm_k3_rotated_" + leg.name;

        passed = RunMatchTemplateSearch(native_leg, temp_directory) && RunMatchTemplateSearch(rotated_leg, temp_directory);
        all_passed &= passed;
        SamplesTestResult(passed);
        if ( ! passed )
            continue;

        std::string agree_name = std::string("K3 pair agreement (") + leg.name + ")";
        SamplesBeginTest(agree_name.c_str( ), passed);

        // Field check (std of the scaled MIP) plus the top-10 matched-peak comparison:
        // positions mapped through the known rotation, heights compared per matched pair.
        MipStats native_stats  = MeasureMip(OutName(native_leg, "scaled_mip"));
        MipStats rotated_stats = MeasureMip(OutName(rotated_leg, "scaled_mip"));

        std::vector<PeakRow> native_peaks  = ReadTopPeaks(native_leg.output_prefix + "_peak_info_1.txt", 10);
        std::vector<PeakRow> rotated_peaks = ReadTopPeaks(rotated_leg.output_prefix + "_peak_info_1.txt", 10);

        passed = true;
        passed &= CompareWithTolerance((agree_name + ": scaled_mip_std").c_str( ), native_stats.std, rotated_stats.std, 0.02);
        passed &= CompareTopPeaks(native_peaks, rotated_peaks, direction, native_nx, native_ny, base.pixel_size,
                                  /* match_radius_px = */ 10.0, /* height_rel_tol = */ 0.01, agree_name.c_str( ));

        all_passed &= passed;
        SamplesTestResult(passed);
    }

    return all_passed;
}
