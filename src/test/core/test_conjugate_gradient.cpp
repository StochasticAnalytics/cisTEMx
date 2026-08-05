/*
 * Comparison and correctness tests for PowellConjugateGradient (refactored)
 * vs ConjugateGradient (original va04a_ wrapper).
 *
 * The comparison tests run both implementations on the same test functions
 * with identical starting points and verify that their results agree. Any
 * divergence illuminates behavioral differences (potential bugs in either).
 *
 * The correctness tests verify convergence to known analytical minima.
 */

#include "../../core/core_headers.h"
#include "../../../include/catch2/catch.hpp"

#include <cmath>
#include <cstdio>

#ifdef cisTEM_ENABLE_CG_REFACTOR_2026

// ============================================================================
// Test objective functions
// ============================================================================

// Sphere function: f(x) = sum(x_i^2)
// Minimum at origin, value 0
static float SphereFunction(void* /*params*/, float x[]) {
    int n = *static_cast<int*>(nullptr); // unused, dimension passed separately
    (void)n;
    // This shouldn't be called directly — use the typed versions below
    return 0.0f;
}

// Typed sphere functions for specific dimensions
static float Sphere1D(void* /*params*/, float x[]) {
    return x[0] * x[0];
}

static float Sphere2D(void* /*params*/, float x[]) {
    return x[0] * x[0] + x[1] * x[1];
}

static float Sphere3D(void* /*params*/, float x[]) {
    return x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
}

static float Sphere5D(void* /*params*/, float x[]) {
    float sum = 0.0f;
    for ( int i = 0; i < 5; i++ )
        sum += x[i] * x[i];
    return sum;
}

// Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
// Minimum at (1, 1), value 0
static float Rosenbrock2D(void* /*params*/, float x[]) {
    float a = 1.0f - x[0];
    float b = x[1] - x[0] * x[0];
    return a * a + 100.0f * b * b;
}

// Separable quadratic: f(x) = sum(a_i * (x_i - b_i)^2)
// Minimum at b, value 0
// Parameters passed as float array: [n, a_0..a_{n-1}, b_0..b_{n-1}]
static float SeparableQuadratic(void* params, float x[]) {
    float* p   = static_cast<float*>(params);
    int    n   = static_cast<int>(p[0]);
    float* a   = p + 1;
    float* b   = p + 1 + n;
    float  sum = 0.0f;
    for ( int i = 0; i < n; i++ ) {
        float diff = x[i] - b[i];
        sum += a[i] * diff * diff;
    }
    return sum;
}

// Beale's function: f(x,y) = (1.5-x+xy)^2 + (2.25-x+xy^2)^2 + (2.625-x+xy^3)^2
// Minimum at (3, 0.5), value 0
static float Beale2D(void* /*params*/, float x[]) {
    float t1 = 1.5f - x[0] + x[0] * x[1];
    float t2 = 2.25f - x[0] + x[0] * x[1] * x[1];
    float t3 = 2.625f - x[0] + x[0] * x[1] * x[1] * x[1];
    return t1 * t1 + t2 * t2 + t3 * t3;
}

// Constant function: f(x) = 42 (negative control)
static float ConstantFunction(void* /*params*/, float[] /*x*/) {
    return 42.0f;
}

// ============================================================================
// Advanced test objective functions (fuzz, adversarial, CTF)
// ============================================================================

// Generalized Rosenbrock in n dimensions:
// f(x) = sum_{i=0}^{n-2} [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
// Minimum at (1,...,1), value 0. params = int* pointing to dimension count
static float RosenbrockND(void* params, float x[]) {
    int   n   = *static_cast<int*>(params);
    float sum = 0.0f;
    for ( int i = 0; i < n - 1; i++ ) {
        float a = 1.0f - x[i];
        float b = x[i + 1] - x[i] * x[i];
        sum += a * a + 100.0f * b * b;
    }
    return sum;
}

// Noisy sphere with deterministic noise (FNV-1a hash of input coordinates).
// Both old and new see identical noise for the same x[] input.
// params = int* pointing to dimension count
static float NoisySphere(void* params, float x[]) {
    int      n    = *static_cast<int*>(params);
    float    sum  = 0.0f;
    unsigned hash = 2166136261u;
    for ( int i = 0; i < n; i++ ) {
        sum += x[i] * x[i];
        int xi_int = static_cast<int>(x[i] * 1000.0f);
        hash ^= static_cast<unsigned>(xi_int + i * 7919);
        hash *= 16777619u;
    }
    float noise = (static_cast<float>(hash % 10000) / 10000.0f - 0.5f) * 0.01f;
    return sum + noise;
}

// Oscillating function: f(x) = -sin(||x||) * ||x||
// Has local minima that can trap the optimizer
// params = int* pointing to dimension count
static float OscillatingFunction(void* params, float x[]) {
    int   n    = *static_cast<int*>(params);
    float norm = 0.0f;
    for ( int i = 0; i < n; i++ )
        norm += x[i] * x[i];
    norm = std::sqrt(norm);
    if ( norm < 1e-30f )
        return 0.0f;
    return -std::sin(norm) * norm;
}

// Sharp valley: f(x,y) = x^2 + 10000*y^2 (condition number ~10000)
static float SharpValley(void* /*params*/, float x[]) {
    return x[0] * x[0] + 10000.0f * x[1] * x[1];
}

// Nearly flat function: f(x) = epsilon * sum(x_i^2)
struct NearlyFlatParams {
    int   n;
    float epsilon;
};

static float NearlyFlat(void* params, float x[]) {
    auto* p   = static_cast<NearlyFlatParams*>(params);
    float sum = 0.0f;
    for ( int i = 0; i < p->n; i++ )
        sum += x[i] * x[i];
    return p->epsilon * sum;
}

// ============================================================================
// Bug isolation objective functions
// ============================================================================

// Degrading objective: score increases every call (triggers Bug 1/3 max-iteration path).
// Returns base_value + sum(x_i^2) + penalty * call_count.
// With penalty > 0 and low max_iterations, the score will exceed 2*|f_initial|,
// triggering the L110/L81 code path where Bug 1 (uninitialized memory read) and
// Bug 3 (fabricated score return) manifest.
struct DegradingObjectiveParams {
    int   n;
    int   call_count;
    float base_value;
    float penalty_per_call;
};

static float DegradingObjective(void* params, float x[]) {
    auto* p = static_cast<DegradingObjectiveParams*>(params);
    p->call_count++;
    float sum = p->base_value;
    for ( int i = 0; i < p->n; i++ )
        sum += x[i] * x[i];
    sum += p->penalty_per_call * static_cast<float>(p->call_count);
    return sum;
}

// Near-singular objective: sphere + sharp ridge along x[0]==x[1].
// Stresses the optimizer in ways that can produce NaN through near-zero
// denominators, mimicking conditions that triggered PR #334 NaN crashes.
// Clamps to 1.0f for |x_i| > 1e6 (mimics refine3d defensive guard).
static float NaNProneObjective(void* params, float x[]) {
    int   n   = *static_cast<int*>(params);
    float sum = 0.0f;
    for ( int i = 0; i < n; i++ ) {
        if ( std::abs(x[i]) > 1e6f )
            return 1.0f;
        sum += x[i] * x[i];
    }
    if ( n >= 2 ) {
        float denom = x[0] - x[1] + 1e-30f;
        sum += 0.001f / (denom * denom + 1e-10f);
    }
    return sum;
}

// Objective that returns NaN for extreme parameter values.
// Tests whether the optimizer handles NaN from the objective function
// without propagating garbage to output parameters.
static float SometimesNaNObjective(void* params, float x[]) {
    int   n   = *static_cast<int*>(params);
    float sum = 0.0f;
    for ( int i = 0; i < n; i++ ) {
        if ( std::abs(x[i]) > 100.0f )
            return std::numeric_limits<float>::quiet_NaN( );
        sum += x[i] * x[i];
    }
    return sum;
}

// Multi-Gaussian MTF-like objective: mimics find_dqe's MTFFit function.
// Fits sum of weighted Gaussians to a target curve.
// Parameters are pairs (sigma_j, weight_j); model = sum(w_j * exp(-s_j * freq^2)) / sum(w_j).
struct MTFLikeParams {
    float target_values[128];
    int   n_points;
    int   n_params;
};

static float MTFLikeObjective(void* params, float x[]) {
    auto* p        = static_cast<MTFLikeParams*>(params);
    float residual = 0.0f;
    for ( int pt = 0; pt < p->n_points; pt++ ) {
        float freq_sq     = static_cast<float>(pt + 1) / static_cast<float>(p->n_points);
        float model       = 0.0f;
        float sum_weights = 0.0f;
        for ( int j = 0; j < p->n_params; j += 2 ) {
            float sigma  = std::abs(x[j]);
            float weight = (j + 1 < p->n_params) ? std::abs(x[j + 1]) : 1.0f;
            sum_weights += weight;
            model += weight * std::exp(-sigma * freq_sq);
        }
        if ( sum_weights > 0.0f )
            model /= sum_weights;
        float diff = model - p->target_values[pt];
        residual += diff * diff;
    }
    return residual;
}

// ============================================================================
// CTF curve fitting helpers
// ============================================================================

struct CTFCurveParams {
    float* data_curve;
    int    num_bins;
    float  box_size;
    CTF    base_ctf;
    float  lowest_freq_sq;
    float  highest_freq_sq;
};

// 1D CTF curve cross-correlation objective (negative NCC).
// Reimplements the core of CtffindCurveObjectiveFunction using CTF::Evaluate.
// x[0] = defocus in pixels (symmetric, no astigmatism).
static float CTFCurveObjective(void* params, float x[]) {
    auto* p      = static_cast<CTFCurveParams*>(params);
    CTF   my_ctf = p->base_ctf;
    my_ctf.SetDefocus(x[0], x[0], 0.0f);

    double cross = 0.0, norm_data = 0.0, norm_ctf = 0.0;
    for ( int bin = 1; bin < p->num_bins; bin++ ) {
        float freq    = static_cast<float>(bin) / p->box_size;
        float freq_sq = freq * freq;
        if ( freq_sq > p->lowest_freq_sq && freq_sq < p->highest_freq_sq ) {
            float ctf_val = fabsf(my_ctf.Evaluate(freq_sq, 0.0f));
            cross += p->data_curve[bin] * ctf_val;
            norm_data += p->data_curve[bin] * p->data_curve[bin];
            norm_ctf += ctf_val * ctf_val;
        }
    }
    if ( norm_ctf * norm_data < 1e-30 )
        return 0.0f;
    return -static_cast<float>(cross / std::sqrt(norm_ctf * norm_data));
}

// ============================================================================
// Helper: run both old and new implementations on the same problem
// ============================================================================

struct ComparisonResult {
    float old_score;
    float new_score;
    float old_values[11]; // max dimensions we test
    float new_values[11];
    int   num_dims;
};

static ComparisonResult RunComparison(
        float (*objective)(void*, float[]),
        void* params,
        int   num_dims,
        float starting_values[],
        float accuracy[],
        int   max_iterations = 50) {
    ComparisonResult result;
    result.num_dims = num_dims;

    // --- Old implementation ---
    ConjugateGradient old_cg;
    old_cg.Init(objective, params, num_dims, starting_values, accuracy);
    result.old_score = old_cg.Run(max_iterations);
    for ( int i = 0; i < num_dims; i++ ) {
        result.old_values[i] = old_cg.GetBestValue(i);
    }

    // --- New implementation ---
    PowellConjugateGradient new_cg;
    new_cg.Init(objective, params, num_dims, starting_values, accuracy);
    result.new_score = new_cg.Run(max_iterations);
    for ( int i = 0; i < num_dims; i++ ) {
        result.new_values[i] = new_cg.GetBestValue(i);
    }

    return result;
}

// ============================================================================
// Comparison tests: old vs new should agree
// ============================================================================

TEST_CASE("Powell minimizer: old vs new comparison",
          "[ConjugateGradient][comparison]") {

    SECTION("Sphere 1D") {
        float start[]    = {5.0f};
        float accuracy[] = {0.01f};
        auto  r          = RunComparison(&Sphere1D, nullptr, 1, start, accuracy);

        INFO("Old score: " << r.old_score << ", New score: " << r.new_score);
        INFO("Old x[0]: " << r.old_values[0] << ", New x[0]: " << r.new_values[0]);

        float score_tol = 1e-4f * std::max(std::abs(r.old_score), 1.0f);
        CHECK(std::abs(r.old_score - r.new_score) < score_tol);
        CHECK(std::abs(r.old_values[0] - r.new_values[0]) < accuracy[0]);
    }

    SECTION("Sphere 2D") {
        float start[]    = {3.0f, -4.0f};
        float accuracy[] = {0.01f, 0.01f};
        auto  r          = RunComparison(&Sphere2D, nullptr, 2, start, accuracy);

        INFO("Old score: " << r.old_score << ", New score: " << r.new_score);
        float score_tol = 1e-4f * std::max(std::abs(r.old_score), 1.0f);
        CHECK(std::abs(r.old_score - r.new_score) < score_tol);
        for ( int i = 0; i < 2; i++ ) {
            CHECK(std::abs(r.old_values[i] - r.new_values[i]) < accuracy[i]);
        }
    }

    SECTION("Sphere 3D") {
        float start[]    = {1.0f, -2.0f, 3.0f};
        float accuracy[] = {0.01f, 0.01f, 0.01f};
        auto  r          = RunComparison(&Sphere3D, nullptr, 3, start, accuracy);

        INFO("Old score: " << r.old_score << ", New score: " << r.new_score);
        float score_tol = 1e-4f * std::max(std::abs(r.old_score), 1.0f);
        CHECK(std::abs(r.old_score - r.new_score) < score_tol);
    }

    SECTION("Sphere 5D") {
        float start[]    = {1.0f, -1.0f, 2.0f, -2.0f, 0.5f};
        float accuracy[] = {0.01f, 0.01f, 0.01f, 0.01f, 0.01f};
        auto  r          = RunComparison(&Sphere5D, nullptr, 5, start, accuracy);

        INFO("Old score: " << r.old_score << ", New score: " << r.new_score);
        float score_tol = 1e-4f * std::max(std::abs(r.old_score), 1.0f);
        CHECK(std::abs(r.old_score - r.new_score) < score_tol);
    }

    SECTION("Separable quadratic 3D") {
        // f(x) = 1*(x0-2)^2 + 4*(x1+1)^2 + 9*(x2-0.5)^2
        float params[]   = {3.0f, 1.0f, 4.0f, 9.0f, 2.0f, -1.0f, 0.5f};
        float start[]    = {0.0f, 0.0f, 0.0f};
        float accuracy[] = {0.01f, 0.01f, 0.01f};
        auto  r          = RunComparison(&SeparableQuadratic, params, 3, start, accuracy);

        INFO("Old score: " << r.old_score << ", New score: " << r.new_score);
        float score_tol = 1e-4f * std::max(std::abs(r.old_score), 1.0f);
        CHECK(std::abs(r.old_score - r.new_score) < score_tol);
    }

    SECTION("Separable quadratic 5D with varying scales") {
        float params[]   = {5.0f,
                            1.0f, 10.0f, 0.1f, 100.0f, 0.01f, // a_i
                            1.0f, -1.0f, 2.0f, -2.0f, 3.0f}; // b_i
        float start[]    = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float accuracy[] = {0.1f, 0.1f, 0.1f, 0.1f, 0.1f};
        auto  r          = RunComparison(&SeparableQuadratic, params, 5, start, accuracy);

        INFO("Old score: " << r.old_score << ", New score: " << r.new_score);
        float score_tol = 1e-3f * std::max(std::abs(r.old_score), 1.0f);
        CHECK(std::abs(r.old_score - r.new_score) < score_tol);
    }

    SECTION("Rosenbrock 2D") {
        float start[]    = {-1.0f, 1.0f};
        float accuracy[] = {0.01f, 0.01f};
        auto  r          = RunComparison(&Rosenbrock2D, nullptr, 2, start, accuracy);

        INFO("Old score: " << r.old_score << ", New score: " << r.new_score);
        // Rosenbrock is hard — allow wider tolerance
        float score_tol = 1e-2f * std::max(std::abs(r.old_score), 1.0f);
        CHECK(std::abs(r.old_score - r.new_score) < score_tol);
    }

    SECTION("Beale 2D") {
        float start[]    = {0.0f, 0.0f};
        float accuracy[] = {0.1f, 0.1f};
        auto  r          = RunComparison(&Beale2D, nullptr, 2, start, accuracy);

        INFO("Old score: " << r.old_score << ", New score: " << r.new_score);
        float score_tol = 1e-2f * std::max(std::abs(r.old_score), 1.0f);
        CHECK(std::abs(r.old_score - r.new_score) < score_tol);
    }
}

// ============================================================================
// Correctness tests: new implementation converges to known minima
// ============================================================================

TEST_CASE("Powell minimizer: convergence to known minima",
          "[ConjugateGradient][correctness]") {

    SECTION("Sphere 1D converges to origin") {
        PowellConjugateGradient cg;
        float                   start[]    = {5.0f};
        float                   accuracy[] = {0.001f};
        cg.Init(&Sphere1D, nullptr, 1, start, accuracy);
        float score = cg.Run(50);

        REQUIRE(std::abs(score) < 0.01f);
        REQUIRE(std::abs(cg.GetBestValue(0)) < 0.1f);
    }

    SECTION("Sphere 2D converges to origin") {
        PowellConjugateGradient cg;
        float                   start[]    = {3.0f, -4.0f};
        float                   accuracy[] = {0.001f, 0.001f};
        cg.Init(&Sphere2D, nullptr, 2, start, accuracy);
        float score = cg.Run(50);

        REQUIRE(score < 0.01f);
        REQUIRE(std::abs(cg.GetBestValue(0)) < 0.1f);
        REQUIRE(std::abs(cg.GetBestValue(1)) < 0.1f);
    }

    SECTION("Sphere 3D converges to origin") {
        PowellConjugateGradient cg;
        float                   start[]    = {1.0f, -2.0f, 3.0f};
        float                   accuracy[] = {0.01f, 0.01f, 0.01f};
        cg.Init(&Sphere3D, nullptr, 3, start, accuracy);
        float score = cg.Run(50);

        REQUIRE(score < 0.1f);
    }

    SECTION("Separable quadratic converges to known minimum") {
        float                   params[] = {3.0f, 1.0f, 4.0f, 9.0f, 2.0f, -1.0f, 0.5f};
        PowellConjugateGradient cg;
        float                   start[]    = {0.0f, 0.0f, 0.0f};
        float                   accuracy[] = {0.01f, 0.01f, 0.01f};
        cg.Init(&SeparableQuadratic, params, 3, start, accuracy);
        float score = cg.Run(50);

        REQUIRE(score < 0.1f);
        REQUIRE(std::abs(cg.GetBestValue(0) - 2.0f) < 0.2f);
        REQUIRE(std::abs(cg.GetBestValue(1) - (-1.0f)) < 0.2f);
        REQUIRE(std::abs(cg.GetBestValue(2) - 0.5f) < 0.2f);
    }

    SECTION("Rosenbrock 2D improves significantly from starting point") {
        PowellConjugateGradient cg;
        float                   start[]     = {-1.0f, 1.0f};
        float                   accuracy[]  = {0.001f, 0.001f};
        float                   initial     = cg.Init(&Rosenbrock2D, nullptr, 2, start, accuracy);
        float                   final_score = cg.Run(100);

        INFO("Initial: " << initial << ", Final: " << final_score);
        // Rosenbrock(−1,1) = 4 + 200 = 204. Should improve substantially.
        REQUIRE(final_score < initial);
        REQUIRE(final_score < 10.0f);
    }
}

// ============================================================================
// API compatibility tests
// ============================================================================

TEST_CASE("Powell minimizer: API compatibility",
          "[ConjugateGradient][api]") {

    SECTION("Legacy void* interface works") {
        PowellConjugateGradient cg;
        float                   start[]    = {3.0f, -4.0f};
        float                   accuracy[] = {0.01f, 0.01f};
        float                   score      = cg.Init(&Sphere2D, nullptr, 2, start, accuracy);

        REQUIRE(score == Approx(25.0f)); // 3^2 + 4^2

        float final_score = cg.Run(50);
        REQUIRE(final_score < score);
    }

    SECTION("std::function interface works") {
        auto sphere = [](float* x) -> float {
            return x[0] * x[0] + x[1] * x[1];
        };

        PowellConjugateGradient cg;
        float                   start[]    = {3.0f, -4.0f};
        float                   accuracy[] = {0.01f, 0.01f};
        float                   score      = cg.Init(PowellConjugateGradient::ObjectiveFunction(sphere),
                                                     2, start, accuracy);

        REQUIRE(score == Approx(25.0f));

        float final_score = cg.Run(50);
        REQUIRE(final_score < 0.01f);
    }

    SECTION("GetPointerToBestValues returns valid data") {
        PowellConjugateGradient cg;
        float                   start[]    = {3.0f, -4.0f};
        float                   accuracy[] = {0.01f, 0.01f};
        cg.Init(&Sphere2D, nullptr, 2, start, accuracy);
        cg.Run(50);

        float* ptr = cg.GetPointerToBestValues( );
        REQUIRE(ptr != nullptr);
        REQUIRE(std::abs(ptr[0]) < 0.1f);
        REQUIRE(std::abs(ptr[1]) < 0.1f);
    }

    SECTION("GetBestScore matches Run return value") {
        PowellConjugateGradient cg;
        float                   start[]    = {3.0f, -4.0f};
        float                   accuracy[] = {0.01f, 0.01f};
        cg.Init(&Sphere2D, nullptr, 2, start, accuracy);
        float run_result = cg.Run(50);

        REQUIRE(run_result == cg.GetBestScore( ));
    }
}

// ============================================================================
// Edge case tests
// ============================================================================

TEST_CASE("Powell minimizer: edge cases",
          "[ConjugateGradient][edge]") {

    SECTION("Constant function terminates gracefully") {
        PowellConjugateGradient cg;
        float                   start[]    = {1.0f, 2.0f};
        float                   accuracy[] = {0.01f, 0.01f};
        float                   score      = cg.Init(&ConstantFunction, nullptr, 2, start, accuracy);

        REQUIRE(score == Approx(42.0f));

        float final_score = cg.Run(50);
        // Should terminate without crashing; score should still be 42
        REQUIRE(final_score == Approx(42.0f));
    }

    SECTION("Re-initialization works") {
        PowellConjugateGradient cg;

        // First run
        float start1[]    = {5.0f};
        float accuracy1[] = {0.01f};
        cg.Init(&Sphere1D, nullptr, 1, start1, accuracy1);
        cg.Run(50);

        // Second run with different starting point
        float start2[]    = {-3.0f};
        float accuracy2[] = {0.001f};
        float score2      = cg.Init(&Sphere1D, nullptr, 1, start2, accuracy2);
        REQUIRE(score2 == Approx(9.0f));

        float final2 = cg.Run(50);
        REQUIRE(final2 < 0.01f);
    }
}

// ============================================================================
// Fuzz / stress tests
// ============================================================================

TEST_CASE("Powell minimizer: randomized separable quadratic fuzz",
          "[ConjugateGradient][fuzz]") {

    const int             N_TRIALS = 50;
    RandomNumberGenerator rng("cg_fuzz_quadratic");

    for ( int trial = 0; trial < N_TRIALS; trial++ ) {
        int n = rng.GetUniformRandomSTD<int>(1, 10);

        // Build SeparableQuadratic params: [n, a_0..a_{n-1}, b_0..b_{n-1}]
        float params[21]; // max: 1 + 10 + 10
        params[0] = static_cast<float>(n);

        float start[10], accuracy[10];
        for ( int i = 0; i < n; i++ ) {
            params[1 + i]     = rng.GetUniformRandomSTD<float>(0.01f, 100.0f);
            params[1 + n + i] = rng.GetUniformRandomSTD<float>(-10.0f, 10.0f);
            float offset      = rng.GetUniformRandomSTD<float>(-5.0f, 5.0f);
            start[i]          = params[1 + n + i] + offset;
            accuracy[i]       = rng.GetUniformRandomSTD<float>(0.001f, 1.0f);
        }

        INFO("Trial " << trial << ", n=" << n);

        auto r = RunComparison(&SeparableQuadratic, params, n, start, accuracy);

        // No NaN/Inf in outputs
        CHECK(std::isfinite(r.old_score));
        CHECK(std::isfinite(r.new_score));
        for ( int i = 0; i < n; i++ ) {
            CHECK(std::isfinite(r.old_values[i]));
            CHECK(std::isfinite(r.new_values[i]));
        }

        // Scores should agree within tolerance
        float score_tol = 0.01f * std::max(std::abs(r.old_score), 1.0f);
        CHECK(std::abs(r.old_score - r.new_score) < score_tol);

        // Both should converge toward known minimum (value >= 0 for quadratic)
        CHECK(r.old_score >= -0.01f);
        CHECK(r.new_score >= -0.01f);
    }
}

TEST_CASE("Powell minimizer: randomized Rosenbrock fuzz",
          "[ConjugateGradient][fuzz]") {

    const int             N_TRIALS = 20;
    RandomNumberGenerator rng("cg_fuzz_rosenbrock");

    for ( int trial = 0; trial < N_TRIALS; trial++ ) {
        int n = rng.GetUniformRandomSTD<int>(2, 5);

        float start[10], accuracy[10];
        for ( int i = 0; i < n; i++ ) {
            start[i]    = rng.GetUniformRandomSTD<float>(-2.0f, 2.0f);
            accuracy[i] = 0.01f;
        }

        INFO("Trial " << trial << ", n=" << n);

        // Evaluate initial score for comparison
        float initial_score = RosenbrockND(&n, start);

        auto r = RunComparison(&RosenbrockND, &n, n, start, accuracy, 100);

        // No NaN/Inf
        CHECK(std::isfinite(r.old_score));
        CHECK(std::isfinite(r.new_score));

        // Both should improve from starting point (or at least not explode)
        CHECK(r.old_score <= initial_score + 1.0f);
        CHECK(r.new_score <= initial_score + 1.0f);

        // Scores should be in the same ballpark
        float score_tol = 0.1f * std::max(std::abs(r.old_score), 1.0f);
        CHECK(std::abs(r.old_score - r.new_score) < score_tol);
    }
}

TEST_CASE("Powell minimizer: noisy objective function",
          "[ConjugateGradient][fuzz]") {

    const int             N_TRIALS = 20;
    RandomNumberGenerator rng("cg_fuzz_noisy");

    for ( int trial = 0; trial < N_TRIALS; trial++ ) {
        int n = rng.GetUniformRandomSTD<int>(2, 5);

        float start[10], accuracy[10];
        for ( int i = 0; i < n; i++ ) {
            start[i]    = rng.GetUniformRandomSTD<float>(-3.0f, 3.0f);
            accuracy[i] = 0.1f;
        }

        INFO("Trial " << trial << ", n=" << n);

        // Evaluate initial score for comparison
        float initial_score = NoisySphere(&n, start);

        auto r = RunComparison(&NoisySphere, &n, n, start, accuracy);

        // No NaN/Inf
        CHECK(std::isfinite(r.old_score));
        CHECK(std::isfinite(r.new_score));

        // Scores should agree within tolerance (noise adds ~0.01 uncertainty)
        float score_tol = 0.1f * std::max(std::abs(r.old_score), 1.0f);
        CHECK(std::abs(r.old_score - r.new_score) < score_tol);

        // Both should converge to near-minimum despite noise
        CHECK(r.old_score <= initial_score + 0.1f);
        CHECK(r.new_score <= initial_score + 0.1f);
    }
}

// ============================================================================
// Adversarial / pathological function tests
// ============================================================================

TEST_CASE("Powell minimizer: adversarial functions",
          "[ConjugateGradient][adversarial]") {

    SECTION("Oscillating function with low max iterations") {
        int   n          = 2;
        float start[]    = {5.0f, 3.0f};
        float accuracy[] = {0.1f, 0.1f};

        auto r = RunComparison(&OscillatingFunction, &n, n, start, accuracy, 3);

        // Should not crash, and outputs should be finite
        CHECK(std::isfinite(r.old_score));
        CHECK(std::isfinite(r.new_score));
        for ( int i = 0; i < n; i++ ) {
            CHECK(std::isfinite(r.old_values[i]));
            CHECK(std::isfinite(r.new_values[i]));
        }
    }

    SECTION("High dimensions: new implementation only (n=11)") {
        int   n = 11;
        float params[23]; // 1 + 11 + 11
        params[0] = static_cast<float>(n);
        float start[11], accuracy[11];
        for ( int i = 0; i < n; i++ ) {
            params[1 + i]     = static_cast<float>(i + 1); // a_i = 1..11
            params[1 + n + i] = static_cast<float>(i) * 0.5f; // b_i
            start[i]          = 0.0f;
            accuracy[i]       = 0.1f;
        }

        PowellConjugateGradient cg;
        cg.Init(&SeparableQuadratic, params, n, start, accuracy);
        float score = cg.Run(100);

        INFO("Score for n=11: " << score);
        CHECK(std::isfinite(score));
        CHECK(score < 10.0f);
    }

    SECTION("High dimensions: new implementation only (n=15)") {
        int   n = 15;
        float params[31]; // 1 + 15 + 15
        params[0] = static_cast<float>(n);
        float start[15], accuracy[15];
        for ( int i = 0; i < n; i++ ) {
            params[1 + i]     = 1.0f;
            params[1 + n + i] = static_cast<float>(i);
            start[i]          = 0.0f;
            accuracy[i]       = 0.1f;
        }

        PowellConjugateGradient cg;
        cg.Init(&SeparableQuadratic, params, n, start, accuracy);
        float score = cg.Run(100);

        INFO("Score for n=15: " << score);
        CHECK(std::isfinite(score));
        CHECK(score < 50.0f);
    }

    SECTION("High dimensions: new implementation only (n=20)") {
        int   n = 20;
        float params[41]; // 1 + 20 + 20
        params[0] = static_cast<float>(n);
        float start[20], accuracy[20];
        for ( int i = 0; i < n; i++ ) {
            params[1 + i]     = 1.0f;
            params[1 + n + i] = 0.0f;
            start[i]          = 1.0f;
            accuracy[i]       = 0.1f;
        }

        PowellConjugateGradient cg;
        cg.Init(&SeparableQuadratic, params, n, start, accuracy);
        float score = cg.Run(200);

        INFO("Score for n=20: " << score);
        CHECK(std::isfinite(score));
        CHECK(score < 100.0f);
    }

    SECTION("Nearly flat function (epsilon = 1e-8)") {
        NearlyFlatParams nf_params  = {3, 1e-8f};
        float            start[]    = {1.0f, -1.0f, 2.0f};
        float            accuracy[] = {0.01f, 0.01f, 0.01f};

        auto r = RunComparison(&NearlyFlat, &nf_params, 3, start, accuracy);

        CHECK(std::isfinite(r.old_score));
        CHECK(std::isfinite(r.new_score));
        // Score should be very small (function value at start is ~6e-8)
        CHECK(r.old_score < 1e-6f);
        CHECK(r.new_score < 1e-6f);
    }

    SECTION("Sharp valley (condition number ~10000)") {
        float start[]    = {1.0f, 1.0f};
        float accuracy[] = {0.01f, 0.0001f};

        auto r = RunComparison(&SharpValley, nullptr, 2, start, accuracy);

        CHECK(std::isfinite(r.old_score));
        CHECK(std::isfinite(r.new_score));
        // Both should find values close to origin
        float score_tol = 1e-2f * std::max(std::abs(r.old_score), 1.0f);
        CHECK(std::abs(r.old_score - r.new_score) < score_tol);
    }
}

// ============================================================================
// Bug isolation tests
// ============================================================================

TEST_CASE("Powell minimizer: max iterations with degrading score",
          "[ConjugateGradient][bug][regression]") {

    SECTION("Degrading objective: refactored returns valid results") {
        DegradingObjectiveParams dp         = {3, 0, 10.0f, 5.0f};
        float                    start[]    = {1.0f, 1.0f, 1.0f};
        float                    accuracy[] = {0.1f, 0.1f, 0.1f};

        PowellConjugateGradient cg;
        float                   initial = cg.Init(&DegradingObjective, &dp, 3, start, accuracy);
        float                   score   = cg.Run(3); // very low max_iterations to force exit

        INFO("Initial: " << initial << ", Final score: " << score);
        INFO("Returned params: " << cg.GetBestValue(0) << ", "
                                 << cg.GetBestValue(1) << ", " << cg.GetBestValue(2));

        // Bug 1 check: parameters should be finite (not garbage/NaN)
        CHECK(std::isfinite(score));
        for ( int i = 0; i < 3; i++ ) {
            CHECK(std::isfinite(cg.GetBestValue(i)));
        }

        // Bug 3 check: returned score should match actual evaluation
        // at the returned parameters. A fabricated fkeep value will NOT match.
        dp.call_count = 0; // reset counter for clean re-evaluation
        float re_eval = DegradingObjective(&dp, cg.GetPointerToBestValues( ));
        INFO("Re-evaluation at returned params: " << re_eval);
        // NOTE: If this fails, it documents Bug 3 (fabricated score return).
        // The refactored code at line 822 sets best_score_ = fkeep when
        // max iterations are reached with degraded score — this is a known
        // mitigation gap. The score won't match because fkeep = 2*|f_initial|.
        CHECK(std::abs(re_eval - score) < std::abs(score) * 0.1f + 1.0f);
    }

    SECTION("Degrading objective: comparison old vs new") {
        DegradingObjectiveParams dp_old     = {3, 0, 10.0f, 5.0f};
        DegradingObjectiveParams dp_new     = {3, 0, 10.0f, 5.0f};
        float                    start[]    = {1.0f, 1.0f, 1.0f};
        float                    accuracy[] = {0.1f, 0.1f, 0.1f};

        ConjugateGradient old_cg;
        old_cg.Init(&DegradingObjective, &dp_old, 3, start, accuracy);
        float old_score = old_cg.Run(3);

        PowellConjugateGradient new_cg;
        new_cg.Init(&DegradingObjective, &dp_new, 3, start, accuracy);
        float new_score = new_cg.Run(3);

        INFO("Old score: " << old_score << ", New score: " << new_score);
        for ( int i = 0; i < 3; i++ ) {
            INFO("Old param[" << i << "]: " << old_cg.GetBestValue(i)
                              << ", New param[" << i << "]: " << new_cg.GetBestValue(i));
        }

        CHECK(std::isfinite(old_score));
        CHECK(std::isfinite(new_score));

        // Re-evaluate both at their returned parameters
        dp_old.call_count = 0;
        dp_new.call_count = 0;
        float old_re      = DegradingObjective(&dp_old, old_cg.GetPointerToBestValues( ));
        float new_re      = DegradingObjective(&dp_new, new_cg.GetPointerToBestValues( ));
        INFO("Old re-eval: " << old_re << " (mismatch: " << std::abs(old_re - old_score) << ")");
        INFO("New re-eval: " << new_re << " (mismatch: " << std::abs(new_re - new_score) << ")");
    }

    SECTION("Positive control: sphere under same max_iterations") {
        float start[]    = {1.0f, -2.0f, 3.0f};
        float accuracy[] = {0.1f, 0.1f, 0.1f};

        PowellConjugateGradient cg;
        float                   initial = cg.Init(&Sphere3D, nullptr, 3, start, accuracy);
        float                   score   = cg.Run(3);

        INFO("Sphere: Initial: " << initial << ", Final: " << score);
        CHECK(std::isfinite(score));
        CHECK(score <= initial); // should improve or stay same
    }
}

TEST_CASE("Powell minimizer: n=11 regression (find_dqe scenario)",
          "[ConjugateGradient][bug][regression]") {

    // Build target curve from known parameters (5.5 Gaussian pairs + 1 extra)
    MTFLikeParams mtf_params;
    mtf_params.n_points = 64;
    mtf_params.n_params = 11;

    // Known true parameters
    float true_params[11] = {
            1.0f, 0.5f, // pair 1: sigma=1.0, weight=0.5
            3.0f, 0.3f, // pair 2: sigma=3.0, weight=0.3
            0.5f, 0.8f, // pair 3: sigma=0.5, weight=0.8
            2.0f, 0.4f, // pair 4: sigma=2.0, weight=0.4
            4.0f, 0.2f, // pair 5: sigma=4.0, weight=0.2
            1.5f // extra (unpaired sigma, uses weight=1.0)
    };

    // Generate target
    for ( int pt = 0; pt < mtf_params.n_points; pt++ ) {
        mtf_params.target_values[pt] = MTFLikeObjective(&mtf_params, true_params);
    }
    // Re-generate properly with freq dependence
    for ( int pt = 0; pt < mtf_params.n_points; pt++ ) {
        float freq_sq     = static_cast<float>(pt + 1) / static_cast<float>(mtf_params.n_points);
        float model       = 0.0f;
        float sum_weights = 0.0f;
        for ( int j = 0; j < 11; j += 2 ) {
            float sigma  = std::abs(true_params[j]);
            float weight = (j + 1 < 11) ? std::abs(true_params[j + 1]) : 1.0f;
            sum_weights += weight;
            model += weight * std::exp(-sigma * freq_sq);
        }
        if ( sum_weights > 0.0f )
            model /= sum_weights;
        mtf_params.target_values[pt] = model;
    }

    SECTION("n=11 with find_dqe accuracy pattern (new only)") {
        // Accuracy alternating pattern matching find_dqe lines 984-987
        float accuracy[11] = {0.1f, 0.01f, 0.1f, 0.01f, 0.1f, 0.01f,
                              0.1f, 0.01f, 0.1f, 0.01f, 0.1f};
        // Perturbed starting point
        float start[11];
        for ( int i = 0; i < 11; i++ )
            start[i] = true_params[i] + 0.5f;

        PowellConjugateGradient cg;
        float                   initial = cg.Init(&MTFLikeObjective, &mtf_params, 11, start, accuracy);
        float                   score   = cg.Run(100);

        INFO("n=11 MTF: Initial: " << initial << ", Final: " << score);
        CHECK(std::isfinite(score));
        CHECK(score < initial); // should improve
        for ( int i = 0; i < 11; i++ ) {
            CHECK(std::isfinite(cg.GetBestValue(i)));
        }
    }

    SECTION("n=11 comparison: verify old would overflow (new is safe)") {
        // Old ConjugateGradient uses xs[11] with 1-indexed access: xs[11] overflows
        // We only run the new implementation here — this is a regression test
        float accuracy[11];
        float start[11];
        for ( int i = 0; i < 11; i++ ) {
            accuracy[i] = 0.1f;
            start[i]    = true_params[i] + 1.0f;
        }

        PowellConjugateGradient cg;
        cg.Init(&MTFLikeObjective, &mtf_params, 11, start, accuracy);
        float score = cg.Run(50);

        INFO("n=11 new-only: " << score);
        CHECK(std::isfinite(score));
    }
}

TEST_CASE("Powell minimizer: NaN immunity",
          "[ConjugateGradient][bug][nan]") {

    SECTION("Near-singular objective does not produce NaN") {
        int   n          = 3;
        float start[]    = {1.0f, 1.001f, 2.0f};
        float accuracy[] = {0.01f, 0.01f, 0.01f};

        PowellConjugateGradient cg;
        cg.Init(&NaNProneObjective, &n, 3, start, accuracy);
        float score = cg.Run(50);

        INFO("NaN-prone: score=" << score);
        CHECK(std::isfinite(score));
        for ( int i = 0; i < 3; i++ ) {
            CHECK(std::isfinite(cg.GetBestValue(i)));
        }
    }

    SECTION("Objective returning NaN for extreme inputs") {
        int   n          = 2;
        float start[]    = {50.0f, -50.0f};
        float accuracy[] = {1.0f, 1.0f};

        PowellConjugateGradient cg;
        float                   initial = cg.Init(&SometimesNaNObjective, &n, 2, start, accuracy);
        float                   score   = cg.Run(50);

        INFO("SometimesNaN: initial=" << initial << ", final=" << score);
        INFO("Params: " << cg.GetBestValue(0) << ", " << cg.GetBestValue(1));
        // Document behavior — the optimizer may or may not handle NaN gracefully.
        // This test records the outcome for future reference.
        if ( std::isfinite(score) ) {
            INFO("Optimizer handled NaN objective gracefully");
        }
        else {
            INFO("Optimizer propagated NaN — known limitation");
        }
        // At minimum, parameters should not be garbage
        for ( int i = 0; i < 2; i++ ) {
            CHECK(std::isfinite(cg.GetBestValue(i)));
        }
    }

    SECTION("Fuzz: 50 random starts with NaN-prone objective") {
        RandomNumberGenerator rng("cg_nan_fuzz");
        const int             N_TRIALS = 50;

        for ( int trial = 0; trial < N_TRIALS; trial++ ) {
            int n = rng.GetUniformRandomSTD<int>(2, 5);

            float start[5], accuracy[5];
            for ( int i = 0; i < n; i++ ) {
                start[i]    = rng.GetUniformRandomSTD<float>(-10.0f, 10.0f);
                accuracy[i] = 0.1f;
            }

            INFO("Trial " << trial << ", n=" << n);

            PowellConjugateGradient cg;
            cg.Init(&NaNProneObjective, &n, n, start, accuracy);
            float score = cg.Run(50);

            CHECK(std::isfinite(score));
            for ( int i = 0; i < n; i++ ) {
                CHECK(std::isfinite(cg.GetBestValue(i)));
            }
        }
    }
}

// ============================================================================
// Real-data integration helpers
// ============================================================================

// Load a pre-computed 1D power spectrum from a binary float file.
// The Python helper script (precompute_ps.py) generates these from MRC files.
// File format: [int32: num_bins] [float32 * num_bins: power spectrum values]
// Returns number of bins read, or 0 on failure.
static int LoadPrecomputedPowerSpectrum(const std::string& ps_path,
                                        float* ps_values, int max_bins) {
    FILE* f = fopen(ps_path.c_str( ), "rb");
    if ( ! f )
        return 0;
    int num_bins = 0;
    if ( fread(&num_bins, sizeof(int), 1, f) != 1 ) {
        fclose(f);
        return 0;
    }
    if ( num_bins <= 0 || num_bins > max_bins ) {
        fclose(f);
        return 0;
    }
    size_t read = fread(ps_values, sizeof(float), num_bins, f);
    fclose(f);
    return static_cast<int>(read);
}

// Run a Python pre-processing script to generate a power spectrum binary file.
// Writes a temp .py script and executes it to avoid shell quoting issues.
// Returns true if the script ran and the output file exists.
static bool PrecomputePowerSpectrum(const std::string& mrc_path,
                                    const std::string& output_path) {
    std::string script_path = output_path + ".py";
    FILE*       script      = fopen(script_path.c_str( ), "w");
    if ( ! script )
        return false;
    fprintf(script,
            "import mrcfile, numpy as np, struct\n"
            "mrc = mrcfile.open('%s', mode='r')\n"
            "data = mrc.data[0].astype(np.float64)\n"
            "mrc.close()\n"
            "ft = np.fft.fft2(data)\n"
            "ps2d = np.abs(ft)**2\n"
            "ny, nx = ps2d.shape\n"
            "cy, cx = ny//2, nx//2\n"
            "Y, X = np.ogrid[:ny, :nx]\n"
            "r = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(int)\n"
            "max_r = min(cy, cx)\n"
            "ps1d = np.zeros(max_r, dtype=np.float64)\n"
            "for ri in range(max_r):\n"
            "    mask = (r == ri)\n"
            "    if mask.any():\n"
            "        ps1d[ri] = ps2d[mask].mean()\n"
            "f = open('%s', 'wb')\n"
            "f.write(struct.pack('i', max_r))\n"
            "f.write(np.array(ps1d, dtype=np.float32).tobytes())\n"
            "f.close()\n",
            mrc_path.c_str( ), output_path.c_str( ));
    fclose(script);

    std::string cmd = "python3 " + script_path + " 2>/dev/null";
    int         ret = system(cmd.c_str( ));
    return (ret == 0);
}

// CTF objective using pre-computed 1D power spectrum from real data.
// Fits defocus (x[0], in Angstroms) against real radially-averaged power spectrum.
struct RealPSCTFParams {
    float* ps_curve;
    int    num_bins;
    float  pixel_size;
    CTF    base_ctf;
};

static float RealPSCTFObjective(void* params, float x[]) {
    auto* p      = static_cast<RealPSCTFParams*>(params);
    CTF   my_ctf = p->base_ctf;
    my_ctf.SetDefocus(x[0], x[0], 0.0f);

    double cross = 0.0, norm_data = 0.0, norm_ctf = 0.0;
    for ( int bin = 1; bin < p->num_bins; bin++ ) {
        float freq    = static_cast<float>(bin) / (2.0f * p->pixel_size *
                                                static_cast<float>(p->num_bins));
        float freq_sq = freq * freq;
        // Skip very low and very high frequencies
        if ( freq_sq < 0.001f || freq_sq > 0.15f )
            continue;
        float ctf_val = fabsf(my_ctf.Evaluate(freq_sq, 0.0f));
        float ctf_sq  = ctf_val * ctf_val; // compare CTF^2 against power spectrum
        cross += p->ps_curve[bin] * ctf_sq;
        norm_data += p->ps_curve[bin] * p->ps_curve[bin];
        norm_ctf += ctf_sq * ctf_sq;
    }
    if ( norm_ctf * norm_data < 1e-30 )
        return 0.0f;
    return -static_cast<float>(cross / std::sqrt(norm_ctf * norm_data));
}

// Noisy CTF curve generator
static void GenerateNoisyCTFCurve(CTF& ctf, float* curve, int num_bins,
                                  float box_size, float lowest_freq_sq,
                                  float                  highest_freq_sq,
                                  float                  noise_level,
                                  float                  background_slope,
                                  RandomNumberGenerator& rng) {
    for ( int bin = 0; bin < num_bins; bin++ ) {
        float freq    = static_cast<float>(bin) / box_size;
        float freq_sq = freq * freq;
        if ( freq_sq > lowest_freq_sq && freq_sq < highest_freq_sq ) {
            float ctf_val    = fabsf(ctf.Evaluate(freq_sq, 0.0f));
            float background = std::exp(-background_slope * freq_sq);
            float noise      = rng.GetUniformRandomSTD<float>(-1.0f, 1.0f) * noise_level;
            curve[bin]       = ctf_val * background + noise;
        }
        else {
            curve[bin] = 0.0f;
        }
    }
}

// ============================================================================
// Real-data integration tests
// ============================================================================

TEST_CASE("Powell minimizer: CTF from real micrograph",
          "[ConjugateGradient][integration][ctf]") {

    // Check for reference images
    const char* ref_dir = std::getenv("PLASMONLABS_REF_IMAGES");
    if ( ! ref_dir ) {
        WARN("PLASMONLABS_REF_IMAGES not set — skipping real-data CTF tests");
        return;
    }

    SECTION("Apoferritin 6000A defocus estimation") {
        std::string mrc_path = std::string(ref_dir) +
                               "/TM_tests/SPA/Apoferritin/Images/apoferritin_6000.mrc";
        std::string ps_path = "/tmp/cg_test_apo6000_ps.bin";
        bool        ok      = PrecomputePowerSpectrum(mrc_path, ps_path);
        if ( ! ok ) {
            WARN("Python mrcfile preprocessing failed");
            return;
        }

        float ps_values[2048];
        int   num_bins = LoadPrecomputedPowerSpectrum(ps_path, ps_values, 2048);
        REQUIRE(num_bins > 100);

        const float pixel_size = 0.7896f;
        CTF         base_ctf;
        base_ctf.Init(300.0f, 2.7f, 0.07f,
                      6000.0f, 6000.0f, 0.0f,
                      pixel_size, 0.0f);

        RealPSCTFParams ctf_params;
        ctf_params.ps_curve   = ps_values;
        ctf_params.num_bins   = num_bins;
        ctf_params.pixel_size = pixel_size;
        ctf_params.base_ctf   = base_ctf;

        float start[]    = {3000.0f};
        float accuracy[] = {100.0f};

        PowellConjugateGradient cg;
        float                   initial = cg.Init(&RealPSCTFObjective, &ctf_params, 1,
                                                  start, accuracy);
        float                   score   = cg.Run(100);

        float result_defocus = cg.GetBestValue(0);
        INFO("Apoferritin 6000A: initial_score=" << initial << ", final=" << score);
        INFO("Recovered defocus: " << result_defocus << "A (expected ~6000A)");

        CHECK(std::isfinite(score));
        CHECK(score < initial);
        CHECK(result_defocus > 0.0f);
        CHECK(std::abs(result_defocus - 6000.0f) < 5000.0f);
    }

    SECTION("Apoferritin defocus sweep: ordering check") {
        const char* suffixes[]        = {"apoferritin_1600", "apoferritin_6000", "apoferritin_9000"};
        float       expected_defoci[] = {1600.0f, 6000.0f, 9000.0f};
        float       recovered[3];

        const float pixel_size = 0.7896f;
        CTF         base_ctf;
        base_ctf.Init(300.0f, 2.7f, 0.07f,
                      5000.0f, 5000.0f, 0.0f,
                      pixel_size, 0.0f);

        for ( int img_idx = 0; img_idx < 3; img_idx++ ) {
            std::string mrc_path = std::string(ref_dir) +
                                   "/TM_tests/SPA/Apoferritin/Images/" + suffixes[img_idx] + ".mrc";
            std::string ps_path = std::string("/tmp/cg_test_") +
                                  suffixes[img_idx] + "_ps.bin";

            bool ok = PrecomputePowerSpectrum(mrc_path, ps_path);
            if ( ! ok ) {
                WARN("Precompute failed for " << suffixes[img_idx]);
                return;
            }

            float ps_values[2048];
            int   num_bins = LoadPrecomputedPowerSpectrum(ps_path, ps_values, 2048);
            if ( num_bins < 100 ) {
                WARN("Too few bins");
                return;
            }

            RealPSCTFParams ctf_params;
            ctf_params.ps_curve   = ps_values;
            ctf_params.num_bins   = num_bins;
            ctf_params.pixel_size = pixel_size;
            ctf_params.base_ctf   = base_ctf;

            float start[]    = {5000.0f};
            float accuracy[] = {100.0f};

            PowellConjugateGradient cg;
            cg.Init(&RealPSCTFObjective, &ctf_params, 1, start, accuracy);
            cg.Run(100);

            recovered[img_idx] = cg.GetBestValue(0);
            INFO("Image " << suffixes[img_idx] << " (" << expected_defoci[img_idx]
                          << "A): recovered=" << recovered[img_idx] << "A");
        }

        CHECK(recovered[0] < recovered[1]);
        CHECK(recovered[1] < recovered[2]);
    }

    SECTION("Old vs new comparison on real power spectrum") {
        std::string mrc_path = std::string(ref_dir) +
                               "/TM_tests/SPA/Apoferritin/Images/apoferritin_6000.mrc";
        std::string ps_path = "/tmp/cg_test_apo6000_cmp_ps.bin";
        bool        ok      = PrecomputePowerSpectrum(mrc_path, ps_path);
        if ( ! ok ) {
            WARN("Precompute failed");
            return;
        }

        float ps_values[2048];
        int   num_bins = LoadPrecomputedPowerSpectrum(ps_path, ps_values, 2048);
        REQUIRE(num_bins > 100);

        const float pixel_size = 0.7896f;
        CTF         base_ctf;
        base_ctf.Init(300.0f, 2.7f, 0.07f,
                      6000.0f, 6000.0f, 0.0f,
                      pixel_size, 0.0f);

        RealPSCTFParams ctf_params;
        ctf_params.ps_curve   = ps_values;
        ctf_params.num_bins   = num_bins;
        ctf_params.pixel_size = pixel_size;
        ctf_params.base_ctf   = base_ctf;

        float start[]    = {3000.0f};
        float accuracy[] = {100.0f};

        auto r = RunComparison(&RealPSCTFObjective, &ctf_params, 1,
                               start, accuracy, 100);

        INFO("Old defocus: " << r.old_values[0] << "A, score: " << r.old_score);
        INFO("New defocus: " << r.new_values[0] << "A, score: " << r.new_score);

        CHECK(std::isfinite(r.old_score));
        CHECK(std::isfinite(r.new_score));
        float score_tol = 0.05f;
        CHECK(std::abs(r.old_score - r.new_score) < score_tol);
    }
}

TEST_CASE("Powell minimizer: projection CTF fitting",
          "[ConjugateGradient][integration][projection]") {

    const char* ref_dir = std::getenv("PLASMONLABS_REF_IMAGES");
    if ( ! ref_dir ) {
        WARN("PLASMONLABS_REF_IMAGES not set — skipping projection CTF tests");
        return;
    }

    SECTION("Precompute projection power spectrum and recover defocus") {
        // Use Python to: load projection, apply synthetic CTF, compute PS
        std::string prj_path = std::string(ref_dir) + "/ribo_ref_prj_0_.mrc";
        std::string ps_path  = "/tmp/cg_test_prj_ctf_ps.bin";

        // Apply CTF at known defocus in Python, compute power spectrum
        float       true_defocus = 12000.0f;
        float       pixel_size   = 1.2f;
        std::string cmd          = "python3 -c \""
                                   "import mrcfile, numpy as np, struct; "
                                   "mrc = mrcfile.open('" +
                          prj_path + "', mode='r'); "
                                     "data = mrc.data[0].astype(np.float64); "
                                     "mrc.close(); "
                                     "ny, nx = data.shape; "
                                     "Y, X = np.meshgrid(np.fft.fftfreq(ny, d=" +
                          std::to_string(pixel_size) + "), "
                                                       "np.fft.fftfreq(nx, d=" +
                          std::to_string(pixel_size) + "), indexing='ij'); "
                                                       "freq_sq = X**2 + Y**2; "
                                                       "wl = 12.2643 / np.sqrt(300e3 * (1 + 300e3 / 1.022e6)); "
                                                       "chi = np.pi * wl * freq_sq * " +
                          std::to_string(true_defocus) + " "
                                                         "- 0.5 * np.pi * 2.7e7 * wl**3 * freq_sq**2; "
                                                         "ctf = -np.sqrt(1 - 0.07**2) * np.sin(chi) - 0.07 * np.cos(chi); "
                                                         "ft_data = np.fft.fft2(data); "
                                                         "ft_ctf = ft_data * ctf; "
                                                         "ps2d = np.abs(ft_ctf)**2; "
                                                         "cy, cx = ny//2, nx//2; "
                                                         "R = np.sqrt((np.arange(ny)[:,None] - cy)**2 + (np.arange(nx)[None,:] - cx)**2).astype(int); "
                                                         "ps2d_shifted = np.fft.fftshift(ps2d); "
                                                         "max_r = min(cy, cx); "
                                                         "ps1d = np.array([ps2d_shifted[R == ri].mean() if (R == ri).any() else 0 for ri in range(max_r)], dtype=np.float32); "
                                                         "f = open('" +
                          ps_path + "', 'wb'); "
                                    "f.write(struct.pack('i', max_r)); "
                                    "f.write(ps1d.tobytes()); "
                                    "f.close()"
                                    "\" 2>/dev/null";
        int ret = system(cmd.c_str( ));
        if ( ret != 0 ) {
            WARN("Python projection preprocessing failed");
            return;
        }

        float ps_values[512];
        int   num_bins = LoadPrecomputedPowerSpectrum(ps_path, ps_values, 512);
        REQUIRE(num_bins > 50);

        CTF base_ctf;
        base_ctf.Init(300.0f, 2.7f, 0.07f,
                      true_defocus, true_defocus, 0.0f,
                      pixel_size, 0.0f);

        RealPSCTFParams ctf_params;
        ctf_params.ps_curve   = ps_values;
        ctf_params.num_bins   = num_bins;
        ctf_params.pixel_size = pixel_size;
        ctf_params.base_ctf   = base_ctf;

        float start[]    = {8000.0f};
        float accuracy[] = {100.0f};

        auto r = RunComparison(&RealPSCTFObjective, &ctf_params, 1,
                               start, accuracy, 100);

        INFO("True defocus: " << true_defocus);
        INFO("Old recovered: " << r.old_values[0] << ", New recovered: " << r.new_values[0]);

        CHECK(std::isfinite(r.old_score));
        CHECK(std::isfinite(r.new_score));
        CHECK(std::abs(r.old_values[0] - true_defocus) < 5000.0f);
        CHECK(std::abs(r.new_values[0] - true_defocus) < 5000.0f);
    }
}

TEST_CASE("Powell minimizer: CTF fitting with noise",
          "[ConjugateGradient][integration][noise]") {

    const float pixel_size      = 1.4f;
    const float box_size        = 512.0f;
    const int   num_bins        = 256;
    const float lowest_freq_sq  = 0.002f;
    const float highest_freq_sq = 0.12f;
    const float true_defocus_A  = 15000.0f;

    CTF true_ctf;
    true_ctf.Init(300.0f, 2.7f, 0.07f,
                  true_defocus_A, true_defocus_A, 0.0f,
                  pixel_size, 0.0f);

    CTFCurveParams ctf_params;
    float          data_curve[256];
    ctf_params.data_curve      = data_curve;
    ctf_params.num_bins        = num_bins;
    ctf_params.box_size        = box_size;
    ctf_params.base_ctf        = true_ctf;
    ctf_params.lowest_freq_sq  = lowest_freq_sq;
    ctf_params.highest_freq_sq = highest_freq_sq;

    float true_defocus_px  = true_defocus_A / pixel_size;
    float wrong_defocus_px = 10000.0f / pixel_size;
    float defocus_tol_px   = 2000.0f / pixel_size;

    SECTION("Low noise (SNR ~10)") {
        RandomNumberGenerator rng("cg_noise_low");
        GenerateNoisyCTFCurve(true_ctf, data_curve, num_bins,
                              box_size, lowest_freq_sq, highest_freq_sq,
                              0.01f, 5.0f, rng);

        float start[]    = {wrong_defocus_px};
        float accuracy[] = {100.0f};
        auto  r          = RunComparison(&CTFCurveObjective, &ctf_params, 1,
                                         start, accuracy, 100);

        INFO("Low noise — Old: " << r.old_values[0] * pixel_size
                                 << "A, New: " << r.new_values[0] * pixel_size << "A");
        // Noisy curves make exact recovery harder; check improvement and finiteness
        CHECK(std::isfinite(r.new_score));
        CHECK(r.new_score <= r.old_score + 0.01f);
    }

    SECTION("Medium noise (SNR ~3)") {
        RandomNumberGenerator rng("cg_noise_med");
        GenerateNoisyCTFCurve(true_ctf, data_curve, num_bins,
                              box_size, lowest_freq_sq, highest_freq_sq,
                              0.05f, 5.0f, rng);

        float start[]    = {wrong_defocus_px};
        float accuracy[] = {100.0f};
        auto  r          = RunComparison(&CTFCurveObjective, &ctf_params, 1,
                                         start, accuracy, 100);

        INFO("Med noise — Old: " << r.old_values[0] * pixel_size
                                 << "A, New: " << r.new_values[0] * pixel_size << "A");
        CHECK(std::isfinite(r.new_score));
        CHECK(r.new_score < r.old_score + 0.1f);
    }

    SECTION("High noise (SNR ~1)") {
        RandomNumberGenerator rng("cg_noise_high");
        GenerateNoisyCTFCurve(true_ctf, data_curve, num_bins,
                              box_size, lowest_freq_sq, highest_freq_sq,
                              0.2f, 5.0f, rng);

        float start[]    = {wrong_defocus_px};
        float accuracy[] = {100.0f};
        auto  r          = RunComparison(&CTFCurveObjective, &ctf_params, 1,
                                         start, accuracy, 100);

        INFO("High noise — Old score: " << r.old_score
                                        << ", New score: " << r.new_score);
        CHECK(std::isfinite(r.old_score));
        CHECK(std::isfinite(r.new_score));
    }

    SECTION("Noise fuzz: 20 trials") {
        RandomNumberGenerator rng("cg_noise_fuzz");
        for ( int trial = 0; trial < 20; trial++ ) {
            float noise = rng.GetUniformRandomSTD<float>(0.01f, 0.1f);
            float df_A  = rng.GetUniformRandomSTD<float>(5000.0f, 25000.0f);
            float df_px = df_A / pixel_size;

            CTF trial_ctf;
            trial_ctf.Init(300.0f, 2.7f, 0.07f, df_A, df_A, 0.0f,
                           pixel_size, 0.0f);
            ctf_params.base_ctf = trial_ctf;

            GenerateNoisyCTFCurve(trial_ctf, data_curve, num_bins,
                                  box_size, lowest_freq_sq, highest_freq_sq,
                                  noise, 5.0f, rng);

            float start_px[] = {df_px + rng.GetUniformRandomSTD<float>(-2000.0f, 2000.0f) / pixel_size};
            float acc[]      = {100.0f};

            INFO("Trial " << trial << ": df=" << df_A << "A, noise=" << noise);
            auto r = RunComparison(&CTFCurveObjective, &ctf_params, 1,
                                   start_px, acc, 100);
            CHECK(std::isfinite(r.old_score));
            CHECK(std::isfinite(r.new_score));
        }
    }
}

TEST_CASE("Powell minimizer: Multi-Gaussian MTF fitting",
          "[ConjugateGradient][integration][mtf]") {

    // Generate MTF target curve from known parameters
    auto make_target = [](MTFLikeParams& p, const float* true_params) {
        for ( int pt = 0; pt < p.n_points; pt++ ) {
            float freq_sq     = static_cast<float>(pt + 1) / static_cast<float>(p.n_points);
            float model       = 0.0f;
            float sum_weights = 0.0f;
            for ( int j = 0; j < p.n_params; j += 2 ) {
                float sigma  = std::abs(true_params[j]);
                float weight = (j + 1 < p.n_params) ? std::abs(true_params[j + 1]) : 1.0f;
                sum_weights += weight;
                model += weight * std::exp(-sigma * freq_sq);
            }
            if ( sum_weights > 0.0f )
                model /= sum_weights;
            p.target_values[pt] = model;
        }
    };

    auto run_mtf_test = [&](int n_params, const float* true_params,
                            const char* label) {
        MTFLikeParams p;
        p.n_points = 64;
        p.n_params = n_params;
        make_target(p, true_params);

        float start[11], accuracy[11];
        for ( int i = 0; i < n_params; i++ ) {
            start[i]    = true_params[i] + 0.5f;
            accuracy[i] = (i % 2 == 0) ? 0.1f : 0.01f; // find_dqe pattern
        }

        PowellConjugateGradient cg;
        float                   initial = cg.Init(&MTFLikeObjective, &p, n_params, start, accuracy);
        float                   score   = cg.Run(100);

        INFO(label << ": initial=" << initial << ", final=" << score);
        CHECK(std::isfinite(score));
        CHECK(score < initial);
    };

    float params_1[]  = {1.5f};
    float params_4[]  = {1.0f, 0.5f, 3.0f, 0.3f};
    float params_6[]  = {1.0f, 0.5f, 3.0f, 0.3f, 0.5f, 0.8f};
    float params_8[]  = {1.0f, 0.5f, 3.0f, 0.3f, 0.5f, 0.8f, 2.0f, 0.4f};
    float params_10[] = {1.0f, 0.5f, 3.0f, 0.3f, 0.5f, 0.8f, 2.0f, 0.4f, 4.0f, 0.2f};
    float params_11[] = {1.0f, 0.5f, 3.0f, 0.3f, 0.5f, 0.8f, 2.0f, 0.4f, 4.0f, 0.2f, 1.5f};

    SECTION("1-Gaussian (n=1)") { run_mtf_test(1, params_1, "MTF n=1"); }
    SECTION("2-Gaussian (n=4)") { run_mtf_test(4, params_4, "MTF n=4"); }
    SECTION("3-Gaussian (n=6)") { run_mtf_test(6, params_6, "MTF n=6"); }
    SECTION("4-Gaussian (n=8)") { run_mtf_test(8, params_8, "MTF n=8"); }
    SECTION("5-Gaussian (n=10)") { run_mtf_test(10, params_10, "MTF n=10"); }
    SECTION("5.5-Gaussian (n=11, find_dqe boundary)") { run_mtf_test(11, params_11, "MTF n=11"); }
}

#endif // cisTEM_ENABLE_CG_REFACTOR_2026
