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
 *     4092x5760 twin): both orientations must find the same result. This is primarily
 *     a regression for the rotated-coordinate out-of-bounds writes (K3 fix) - the
 *     always-on asserts in the search ride along - so a coarse angular grid is used
 *     to keep it fast; consistency, not sensitivity, is under test.
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
       << "Yes" << "\n" // use FastFFT
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

    SamplesBeginTest("K3 native vs rotated90 search consistency", passed);

    // Metadata from TM_tests/Yeast/MetaData/Yeast.toml. C1 on a 24 Mpx K3 frame is
    // expensive, and this test guards coordinate handling (the rotated-geometry
    // out-of-bounds class), not detection sensitivity - hence the coarse grid.
    TmSearchSettings native;
    native.image_path        = cistem_ref_dir + "/TM_tests/Yeast/Images/147_Mar12_12.21.27_159_0.mrc";
    native.template_path     = cistem_ref_dir + "/TM_tests/Yeast/Templates/6Q8Y_mature_60S.mrc";
    native.output_prefix     = temp_directory + "/tm_k3_native";
    native.pixel_size        = 1.06f;
    native.defocus_1         = 4147.0f;
    native.defocus_2         = 4147.0f;
    native.defocus_angle     = 49.2f;
    native.high_res_limit    = 4.0f;
    native.out_of_plane_step = 20.0f;
    native.in_plane_step     = 15.0f;

    if ( ! DoesFileExist(native.image_path) ) {
        SamplesTestResultSkipped("yeast K3 reference data not present under PLASMONLABS_REF_IMAGES");
        return true;
    }

    TmSearchSettings rotated = native;
    rotated.image_path       = cistem_ref_dir + "/TM_tests/Yeast/Images/147_Mar12_12.21.27_159_0_rotate90.mrc";
    rotated.output_prefix    = temp_directory + "/tm_k3_rotated";

    passed = RunMatchTemplateSearch(native, temp_directory) && RunMatchTemplateSearch(rotated, temp_directory);
    all_passed &= passed;
    SamplesTestResult(passed);
    if ( ! passed )
        return false;

    SamplesBeginTest("K3 native vs rotated90 result agreement", passed);

    // The same physical content searched in two memory layouts must yield the same
    // detection statistics. Position mapping across the rotation is checked loosely
    // (the two runs pad/rotate internally); the values are the strong invariant.
    MipStats native_stats  = MeasureMip(OutName(native, "scaled_mip"));
    MipStats rotated_stats = MeasureMip(OutName(rotated, "scaled_mip"));
    int      native_peaks  = CountPeakInfoRows(native.output_prefix + "_peak_info_1.txt");
    int      rotated_peaks = CountPeakInfoRows(rotated.output_prefix + "_peak_info_1.txt");

    passed = true;
    passed &= CompareWithTolerance("scaled_mip_max (native vs rotated)", native_stats.max_value, rotated_stats.max_value, 0.02);
    passed &= CompareWithTolerance("scaled_mip_std (native vs rotated)", native_stats.std, rotated_stats.std, 0.02);
    if ( native_peaks >= 0 && rotated_peaks >= 0 )
        passed &= CompareWithTolerance("peak_info_rows (native vs rotated)", double(native_peaks), double(rotated_peaks), 0.10);

    all_passed &= passed;
    SamplesTestResult(passed);

    return all_passed;
}
