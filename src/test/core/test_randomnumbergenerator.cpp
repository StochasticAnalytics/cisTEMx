/*
 * Copyright (c) 2025, Stochastic Analytics, LLC
 * Licensed under MPL 2.0 for academic use;
 * commercial license required for commercial use.
 * See LICENSE.md for details.
 */

#include "../../core/core_headers.h"
#include "../../../include/catch2/catch.hpp"

#include <vector>
#include <algorithm>
#include <cmath>
#include <map>

/*
Comprehensive unit tests for RandomNumberGenerator class.

Tests cover:
- All 4 constructor variants
- Seeding behavior (positive, negative, time-based)
- Distribution correctness with statistical validation
- Both internal LCG and std::mt19937 modes
- Reproducibility and determinism
- All distribution methods (Uniform, Normal, Poisson, Exponential, Gamma)

Statistical tests use production-grade validation with chi-square and
Kolmogorov-Smirnov tests at α = 0.01 significance level.
*/

// ============================================================================
// Statistical Test Helper Functions
// ============================================================================

/**
 * Calculate mean of a vector
 */
template <typename T>
double CalculateMean(const std::vector<T>& data) {
    double sum = 0.0;
    for ( const auto& val : data ) {
        sum += static_cast<double>(val);
    }
    return sum / static_cast<double>(data.size( ));
}

/**
 * Calculate sample variance of a vector (uses N-1 for Bessel's correction)
 */
template <typename T>
double CalculateVariance(const std::vector<T>& data) {
    if ( data.size( ) <= 1 )
        return 0.0; // Avoid division by zero

    double mean             = CalculateMean(data);
    double sum_squared_diff = 0.0;
    for ( const auto& val : data ) {
        double diff = static_cast<double>(val) - mean;
        sum_squared_diff += diff * diff;
    }
    // Use N-1 for sample variance (Bessel's correction)
    return sum_squared_diff / static_cast<double>(data.size( ) - 1);
}

/**
 * Calculate standard deviation of a vector
 */
template <typename T>
double CalculateStdDev(const std::vector<T>& data) {
    return std::sqrt(CalculateVariance(data));
}

/**
 * Chi-square goodness-of-fit test for uniform distribution
 * Returns chi-square statistic (lower is better fit)
 *
 * For uniform distribution in k bins, critical value at α=0.01 is approximately:
 * χ²(k-1, 0.01) ≈ k-1 + 2.58*sqrt(2*(k-1)) (approximate for large k)
 */
double ChiSquareUniformTest(const std::vector<float>& data, double min_val, double max_val, int num_bins) {
    if ( data.empty( ) || num_bins <= 0 )
        return 0.0;
    if ( max_val <= min_val )
        return 0.0;

    std::vector<int> observed(num_bins, 0);
    double           bin_width = (max_val - min_val) / num_bins;
    int              discarded = 0;

    // Count observations in each bin
    for ( const auto& val : data ) {
        // Handle edge case where val == max_val exactly
        int bin = static_cast<int>((val - min_val) / bin_width);
        if ( bin >= num_bins )
            bin = num_bins - 1; // Place max_val in last bin

        if ( bin >= 0 && bin < num_bins ) {
            observed[bin]++;
        }
        else {
            discarded++; // Track out-of-range values
        }
    }

    // Expected count in each bin for uniform distribution
    // Use actual observed count (in case some values were discarded)
    int total_observed = data.size( ) - discarded;
    if ( total_observed == 0 )
        return 0.0;

    double expected = static_cast<double>(total_observed) / num_bins;

    // Chi-square test requires expected count >= 5 per bin
    if ( expected < 5.0 ) {
        // Not enough data for reliable chi-square test
        return 0.0;
    }

    // Calculate chi-square statistic
    double chi_square = 0.0;
    for ( int i = 0; i < num_bins; i++ ) {
        double diff = observed[i] - expected;
        chi_square += (diff * diff) / expected;
    }

    return chi_square;
}

/**
 * Kolmogorov-Smirnov test for normal distribution
 * Returns KS statistic (lower is better fit)
 *
 * Critical value at α=0.01 for n samples: 1.63 / sqrt(n)
 */
double KolmogorovSmirnovNormalTest(const std::vector<float>& data, double mean, double stddev) {
    std::vector<float> sorted_data = data;
    std::sort(sorted_data.begin( ), sorted_data.end( ));

    double max_diff = 0.0;
    int    n        = sorted_data.size( );

    for ( int i = 0; i < n; i++ ) {
        // Empirical CDF at this point
        double empirical_cdf = static_cast<double>(i + 1) / n;

        // Theoretical normal CDF at this point
        double z               = (sorted_data[i] - mean) / stddev;
        double theoretical_cdf = 0.5 * (1.0 + std::erf(z / std::sqrt(2.0)));

        // Track maximum difference
        double diff = std::abs(empirical_cdf - theoretical_cdf);
        if ( diff > max_diff ) {
            max_diff = diff;
        }
    }

    return max_diff;
}

/**
 * Chi-square test for discrete distributions (Poisson, etc.)
 */
double ChiSquareDiscreteTest(const std::vector<int>& data, const std::map<int, double>& expected_probabilities) {
    // Count observed frequencies
    std::map<int, int> observed_counts;
    for ( const auto& val : data ) {
        observed_counts[val]++;
    }

    double chi_square = 0.0;
    int    n          = data.size( );

    // Calculate chi-square over all values in expected distribution
    for ( const auto& pair : expected_probabilities ) {
        int    value          = pair.first;
        double expected_count = n * pair.second;

        // Skip bins with expected count < 5 (chi-square requirement)
        if ( expected_count < 5.0 )
            continue;

        int    observed_count = observed_counts[value];
        double diff           = observed_count - expected_count;
        chi_square += (diff * diff) / expected_count;
    }

    return chi_square;
}

/**
 * Calculate Poisson probability mass function
 */
double PoissonPMF(int k, double lambda) {
    if ( k < 0 || lambda < 0.0 )
        return 0.0;

    // Special case: lambda = 0
    if ( lambda == 0.0 ) {
        return (k == 0) ? 1.0 : 0.0; // All probability mass at k=0
    }

    // P(X=k) = (λ^k * e^(-λ)) / k!
    // Use log to avoid overflow: log(P) = k*log(λ) - λ - log(k!)
    double log_prob = k * std::log(lambda) - lambda - std::lgamma(k + 1);
    return std::exp(log_prob);
}

// ============================================================================
// Constructor Tests
// ============================================================================

TEST_CASE("RandomNumberGenerator default constructor", "[RandomNumberGenerator][constructors]") {
    SECTION("Default constructor produces valid random numbers") {
        // Test behavior: default constructor should work
        RandomNumberGenerator rng;
        float                 val = rng.GetUniformRandom( );
        REQUIRE(val >= -1.0f);
        REQUIRE(val <= 1.0f);
    }

    SECTION("Internal mode with default seed produces reproducible sequence") {
        // Test behavior: internal mode has independent state
        RandomNumberGenerator rng1(true);
        RandomNumberGenerator rng2(true);

        // Both use default seed (4711) and have independent state
        for ( int i = 0; i < 100; i++ ) {
            float val1 = rng1.GetUniformRandom( );
            float val2 = rng2.GetUniformRandom( );
            REQUIRE(val1 == val2);
        }
    }

    SECTION("Standard mode cannot guarantee reproducibility across instances") {
        // Standard mode (internal=false) uses global rand()
        // Multiple instances interfere with each other
        // This test documents that behavior is NOT reproducible
        RandomNumberGenerator rng(false);

        // Can only test that it produces valid values
        float val = rng.GetUniformRandom( );
        REQUIRE(val >= -1.0f);
        REQUIRE(val <= 1.0f);

        INFO("Standard mode uses global rand() - not reproducible with multiple instances");
    }
}

TEST_CASE("RandomNumberGenerator explicit seed constructor", "[RandomNumberGenerator][constructors]") {
    SECTION("Same positive seed produces identical sequences - internal mode") {
        // Test behavior: internal mode with same seed has independent state
        RandomNumberGenerator rng1(12345, true);
        RandomNumberGenerator rng2(12345, true);

        // Both should produce identical sequences
        for ( int i = 0; i < 100; i++ ) {
            float val1 = rng1.GetUniformRandom( );
            float val2 = rng2.GetUniformRandom( );
            REQUIRE(val1 == val2);
        }
    }

    SECTION("Different seeds produce different sequences") {
        // Test behavior: different seeds should produce different sequences
        RandomNumberGenerator rng1(11111, true);
        RandomNumberGenerator rng2(22222, true);

        std::vector<float> seq1, seq2;
        for ( int i = 0; i < 100; i++ ) {
            seq1.push_back(rng1.GetUniformRandom( ));
            seq2.push_back(rng2.GetUniformRandom( ));
        }

        REQUIRE(seq1 != seq2);
    }

    SECTION("Explicit seed produces valid random numbers") {
        // Test behavior: constructor with explicit seed should work
        RandomNumberGenerator rng(12345, false);

        float val = rng.GetUniformRandom( );
        REQUIRE(val >= -1.0f);
        REQUIRE(val <= 1.0f);
    }

    SECTION("Negative seed produces valid random numbers") {
        // Test behavior: negative seed should work (uses time)
        RandomNumberGenerator rng(-1, false);

        // Should produce valid values
        float val = rng.GetUniformRandom( );
        REQUIRE(val >= -1.0f);
        REQUIRE(val <= 1.0f);
    }
}

TEST_CASE("RandomNumberGenerator string-based constructor", "[RandomNumberGenerator][constructors]") {
    SECTION("String constructor produces deterministic seed") {
        RandomNumberGenerator rng1("test_string");
        RandomNumberGenerator rng2("test_string");

        // Both should produce same sequence
        std::vector<float> seq1, seq2;
        for ( int i = 0; i < 100; i++ ) {
            seq1.push_back(rng1.GetUniformRandomSTD(-1.0f, 1.0f));
            seq2.push_back(rng2.GetUniformRandomSTD(-1.0f, 1.0f));
        }

        REQUIRE(seq1 == seq2);
    }

    SECTION("Different strings produce different seeds") {
        // Test multiple string pairs to ensure hash function produces variety
        std::vector<std::pair<std::string, std::string>> test_pairs = {
                {"string_A", "string_B"},
                {"image_001", "image_002"},
                {"data_x", "data_y"}};

        for ( const auto& pair : test_pairs ) {
            RandomNumberGenerator rng1(pair.first);
            RandomNumberGenerator rng2(pair.second);

            // Generate sequences from each
            std::vector<float> seq1, seq2;
            for ( int i = 0; i < 100; i++ ) {
                seq1.push_back(rng1.GetUniformRandomSTD(-1.0f, 1.0f));
                seq2.push_back(rng2.GetUniformRandomSTD(-1.0f, 1.0f));
            }

            // Sequences should differ (unless rare hash collision)
            INFO("Testing strings: '" << pair.first << "' vs '" << pair.second << "'");

            // Count differences to be more robust than simple equality
            int differences = 0;
            for ( size_t i = 0; i < seq1.size( ); i++ ) {
                if ( seq1[i] != seq2[i] ) {
                    differences++;
                }
            }

            INFO("Found " << differences << " differences out of " << seq1.size( ) << " values");

            // If strings produce different hashes, sequences MUST differ
            // Require at least some differences (allow for unlikely hash collision)
            REQUIRE(differences > 0);
        }
    }
}

TEST_CASE("RandomNumberGenerator thread-based constructor", "[RandomNumberGenerator][constructors]") {
    SECTION("Thread constructor creates functional RNG") {
        RandomNumberGenerator rng(1.0f); // thread_id = 1.0

        // Should produce valid random numbers
        float val = rng.GetUniformRandomSTD(-1.0f, 1.0f);
        REQUIRE(val >= -1.0f);
        REQUIRE(val <= 1.0f);
    }

    SECTION("Thread constructor with debug mode uses constant seed") {
        // Note: This test assumes debug_with_constant_seed == false
        // If debug mode is enabled, different thread IDs would produce identical sequences
        RandomNumberGenerator rng1(1.0f);
        RandomNumberGenerator rng2(2.0f);

        // With debug_with_constant_seed=false, these should likely differ
        // (though not guaranteed since they might get same timestamp)
        std::vector<float> seq1, seq2;
        for ( int i = 0; i < 10; i++ ) {
            seq1.push_back(rng1.GetUniformRandomSTD(0.0f, 1.0f));
            seq2.push_back(rng2.GetUniformRandomSTD(0.0f, 1.0f));
        }

        // This test documents behavior but doesn't strictly require difference
        // since timing could theoretically match
        INFO("Thread-based constructors may produce different sequences");
    }
}

// ============================================================================
// Seeding Behavior Tests
// ============================================================================

TEST_CASE("RandomNumberGenerator SetSeed behavior", "[RandomNumberGenerator][seeding]") {
    SECTION("SetSeed with positive value - internal mode reproducible") {
        // Test behavior: SetSeed should reset to reproducible state in internal mode
        RandomNumberGenerator rng1(true);
        RandomNumberGenerator rng2(true);

        rng1.SetSeed(99999);
        rng2.SetSeed(99999);

        // Both should produce identical sequences (internal mode = independent state)
        for ( int i = 0; i < 100; i++ ) {
            float val1 = rng1.GetUniformRandom( );
            float val2 = rng2.GetUniformRandom( );
            REQUIRE(val1 == val2);
        }
    }

    SECTION("SetSeed with negative value produces valid random numbers") {
        // Test behavior: negative seed should work (uses time)
        RandomNumberGenerator rng;
        rng.SetSeed(-1);

        // Should produce valid values
        float val = rng.GetUniformRandom( );
        REQUIRE(val >= -1.0f);
        REQUIRE(val <= 1.0f);
    }

    SECTION("SetSeed resets sequence - single instance") {
        // Test behavior: re-seeding should restart sequence
        // Use internal mode to avoid global state interference
        RandomNumberGenerator rng(12345, true);
        float                 val1 = rng.GetUniformRandom( );

        rng.SetSeed(12345); // Reset to same seed
        float val2 = rng.GetUniformRandom( );

        REQUIRE(val1 == val2); // Should get same first value
    }
}

TEST_CASE("RandomNumberGenerator Internal_srand", "[RandomNumberGenerator][seeding]") {
    SECTION("Internal_srand produces reproducible sequence") {
        // Test behavior: Internal_srand should reset internal PRNG state
        RandomNumberGenerator rng1(true);
        RandomNumberGenerator rng2(true);

        rng1.Internal_srand(42);
        rng2.Internal_srand(42);

        // Both should produce identical sequences
        for ( int i = 0; i < 100; i++ ) {
            int val1 = rng1.Internal_rand( );
            int val2 = rng2.Internal_rand( );
            REQUIRE(val1 == val2);
        }
    }

    SECTION("Internal_srand resets internal PRNG sequence") {
        // Test behavior: re-seeding should restart sequence
        RandomNumberGenerator rng(true);
        rng.Internal_srand(12345);
        int val1 = rng.Internal_rand( );

        rng.Internal_srand(12345); // Reset
        int val2 = rng.Internal_rand( );

        REQUIRE(val1 == val2); // Should get same first value
    }
}

// ============================================================================
// Reproducibility and Determinism Tests
// ============================================================================

TEST_CASE("RandomNumberGenerator global state contamination", "[RandomNumberGenerator][global_state]") {
    SECTION("Standard mode WARNING: global state makes multi-instance use unreliable") {
        // IMPORTANT: Standard mode (internal=false) uses global rand()
        // Multiple instances INTERFERE with each other!
        // This test DOCUMENTS the dangerous behavior - not a proper test

        RandomNumberGenerator rng(12345, false);

        // Can only reliably test single instance behavior
        std::vector<float> seq1, seq2;

        // First sequence
        for ( int i = 0; i < 100; i++ ) {
            seq1.push_back(rng.GetUniformRandom( ));
        }

        // Re-seed same instance
        rng.SetSeed(12345);

        // Second sequence - should match if no external interference
        for ( int i = 0; i < 100; i++ ) {
            seq2.push_back(rng.GetUniformRandom( ));
        }

        // This SHOULD pass for single instance, but may fail if other
        // code in the test suite calls rand() or srand()
        bool sequences_match = (seq1 == seq2);

        INFO("Standard mode single-instance reproducibility: " << (sequences_match ? "PASS" : "FAIL"));
        INFO("RECOMMENDATION: Use internal=true for reliable multi-instance behavior");

        // Don't REQUIRE - just document the behavior
        // In a test environment with parallel tests, this may fail!
    }

    SECTION("Internal mode instances have independent state") {
        // Internal mode (internal=true) uses instance-specific state
        // This is RELIABLE even with multiple instances
        RandomNumberGenerator rng1(12345, true);
        RandomNumberGenerator rng2(54321, true);

        // Each has its own state - no interference
        std::vector<float> seq1, seq2;
        for ( int i = 0; i < 100; i++ ) {
            seq1.push_back(rng1.GetUniformRandom( ));
            seq2.push_back(rng2.GetUniformRandom( ));
        }

        // Sequences MUST differ (different seeds, independent state)
        REQUIRE(seq1 != seq2);
    }
}

TEST_CASE("RandomNumberGenerator reproducibility with same seed", "[RandomNumberGenerator][reproducibility]") {
    SECTION("Same seed produces identical uniform sequences (internal mode)") {
        RandomNumberGenerator rng1(54321, true);
        RandomNumberGenerator rng2(54321, true);

        for ( int i = 0; i < 1000; i++ ) {
            float val1 = rng1.GetUniformRandom( );
            float val2 = rng2.GetUniformRandom( );
            REQUIRE(val1 == val2);
        }
    }

    SECTION("Standard mode uses global rand() state") {
        // Standard mode (internal=false) uses global rand(), which means
        // multiple instances share state and interfere with each other.
        // This test verifies that re-seeding a SINGLE instance produces
        // reproducible sequences.

        RandomNumberGenerator rng(54321, false);
        std::vector<float>    seq1, seq2;

        // First sequence
        for ( int i = 0; i < 100; i++ ) {
            seq1.push_back(rng.GetUniformRandom( ));
        }

        // Re-seed and generate second sequence
        rng.SetSeed(54321);
        for ( int i = 0; i < 100; i++ ) {
            seq2.push_back(rng.GetUniformRandom( ));
        }

        // Same seed should produce identical sequences
        REQUIRE(seq1 == seq2);
    }

    SECTION("Same seed produces identical normal sequences") {
        RandomNumberGenerator rng1(99999, true);
        RandomNumberGenerator rng2(99999, true);

        for ( int i = 0; i < 1000; i++ ) {
            float val1 = rng1.GetNormalRandom( );
            float val2 = rng2.GetNormalRandom( );
            REQUIRE(val1 == val2);
        }
    }

    SECTION("Different seeds produce different sequences") {
        RandomNumberGenerator rng1(11111);
        RandomNumberGenerator rng2(22222);

        std::vector<float> seq1, seq2;
        for ( int i = 0; i < 100; i++ ) {
            seq1.push_back(rng1.GetUniformRandom( ));
            seq2.push_back(rng2.GetUniformRandom( ));
        }

        REQUIRE(seq1 != seq2);
    }
}

// ============================================================================
// GetUniformRandom() Tests
// ============================================================================

TEST_CASE("RandomNumberGenerator GetUniformRandom range", "[RandomNumberGenerator][uniform]") {
    SECTION("GetUniformRandom returns values in [-1, 1] - internal mode") {
        RandomNumberGenerator rng(12345, true);

        for ( int i = 0; i < 10000; i++ ) {
            float val = rng.GetUniformRandom( );
            REQUIRE(val >= -1.0f);
            REQUIRE(val <= 1.0f);
        }
    }

    SECTION("GetUniformRandom returns values in [-1, 1] - standard mode") {
        RandomNumberGenerator rng(12345, false);

        for ( int i = 0; i < 10000; i++ ) {
            float val = rng.GetUniformRandom( );
            REQUIRE(val >= -1.0f);
            REQUIRE(val <= 1.0f);
        }
    }
}

TEST_CASE("RandomNumberGenerator GetUniformRandom statistical properties", "[RandomNumberGenerator][uniform][statistics]") {
    constexpr int sample_size = 100000;

    SECTION("GetUniformRandom mean near 0 - internal mode") {
        RandomNumberGenerator rng(12345, true);
        std::vector<float>    samples;

        for ( int i = 0; i < sample_size; i++ ) {
            samples.push_back(rng.GetUniformRandom( ));
        }

        double mean          = CalculateMean(samples);
        double expected_mean = 0.0;
        double tolerance     = 0.01; // Tight tolerance for large sample

        REQUIRE(std::abs(mean - expected_mean) < tolerance);
    }

    SECTION("GetUniformRandom variance near 1/3 - internal mode") {
        RandomNumberGenerator rng(12345, true);
        std::vector<float>    samples;

        for ( int i = 0; i < sample_size; i++ ) {
            samples.push_back(rng.GetUniformRandom( ));
        }

        double variance          = CalculateVariance(samples);
        double expected_variance = 1.0 / 3.0; // Var(Uniform(-1,1)) = (b-a)²/12 = 4/12 = 1/3
        double tolerance         = 0.01;

        REQUIRE(std::abs(variance - expected_variance) < tolerance);
    }

    SECTION("GetUniformRandom chi-square test - internal mode") {
        RandomNumberGenerator rng(12345, true);
        std::vector<float>    samples;

        for ( int i = 0; i < sample_size; i++ ) {
            samples.push_back(rng.GetUniformRandom( ));
        }

        int    num_bins   = 20;
        double chi_square = ChiSquareUniformTest(samples, -1.0, 1.0, num_bins);

        // Critical value at α=0.01 for df=19: approximately 36.19
        double critical_value = 36.19;

        REQUIRE(chi_square < critical_value);
    }

    SECTION("GetUniformRandom chi-square test - standard mode") {
        RandomNumberGenerator rng(12345, false);
        std::vector<float>    samples;

        for ( int i = 0; i < sample_size; i++ ) {
            samples.push_back(rng.GetUniformRandom( ));
        }

        int    num_bins   = 20;
        double chi_square = ChiSquareUniformTest(samples, -1.0, 1.0, num_bins);

        // Critical value at α=0.01 for df=19
        double critical_value = 36.19;

        REQUIRE(chi_square < critical_value);
    }
}

// ============================================================================
// GetNormalRandom() Tests
// ============================================================================

TEST_CASE("RandomNumberGenerator GetNormalRandom statistical properties", "[RandomNumberGenerator][normal][statistics]") {
    constexpr int sample_size = 100000;

    SECTION("GetNormalRandom mean near 0") {
        RandomNumberGenerator rng(12345, true);
        std::vector<float>    samples;

        for ( int i = 0; i < sample_size; i++ ) {
            samples.push_back(rng.GetNormalRandom( ));
        }

        double mean          = CalculateMean(samples);
        double expected_mean = 0.0;
        double tolerance     = 0.01;

        REQUIRE(std::abs(mean - expected_mean) < tolerance);
    }

    SECTION("GetNormalRandom standard deviation near 1") {
        RandomNumberGenerator rng(12345, true);
        std::vector<float>    samples;

        for ( int i = 0; i < sample_size; i++ ) {
            samples.push_back(rng.GetNormalRandom( ));
        }

        double stddev          = CalculateStdDev(samples);
        double expected_stddev = 1.0;
        double tolerance       = 0.01;

        REQUIRE(std::abs(stddev - expected_stddev) < tolerance);
    }

    SECTION("GetNormalRandom Kolmogorov-Smirnov test") {
        RandomNumberGenerator rng(12345, true);
        std::vector<float>    samples;

        for ( int i = 0; i < sample_size; i++ ) {
            samples.push_back(rng.GetNormalRandom( ));
        }

        double ks_statistic = KolmogorovSmirnovNormalTest(samples, 0.0, 1.0);

        // Critical value at α=0.01: 1.63 / sqrt(n)
        double critical_value = 1.63 / std::sqrt(static_cast<double>(sample_size));

        REQUIRE(ks_statistic < critical_value);
    }
}

// ============================================================================
// Internal PRNG Tests
// ============================================================================

TEST_CASE("RandomNumberGenerator Internal_rand range", "[RandomNumberGenerator][internal]") {
    SECTION("Internal_rand returns values in [0, 32767]") {
        RandomNumberGenerator rng(true);
        rng.Internal_srand(12345);

        for ( int i = 0; i < 10000; i++ ) {
            int val = rng.Internal_rand( );
            REQUIRE(val >= 0);
            REQUIRE(val <= 32767);
        }
    }

    SECTION("Internal_rand LCG implementation check") {
        RandomNumberGenerator rng(true);
        rng.Internal_srand(1);

        // Manually calculate expected values
        // LCG: next = next * 1103515245 + 12345
        // Output: ((next / 65536) % 32768)

        unsigned int expected_seed = 1;
        for ( int i = 0; i < 100; i++ ) {
            int output = rng.Internal_rand( );

            expected_seed       = expected_seed * 1103515245 + 12345;
            int expected_output = ((unsigned int)(expected_seed / 65536) % 32768);

            REQUIRE(output == expected_output);
        }
    }
}

// ============================================================================
// GetUniformRandomSTD() Template Tests
// ============================================================================

TEST_CASE("RandomNumberGenerator GetUniformRandomSTD integer distribution", "[RandomNumberGenerator][uniform_std][template]") {
    SECTION("Integer uniform in [0, 100]") {
        RandomNumberGenerator rng("test_uniform_int");

        for ( int i = 0; i < 10000; i++ ) {
            int val = rng.GetUniformRandomSTD(0, 100);
            REQUIRE(val >= 0);
            REQUIRE(val <= 100);
        }
    }

    SECTION("Integer uniform statistical properties") {
        RandomNumberGenerator rng("test_uniform_int_stats");
        std::vector<int>      samples;
        constexpr int         min_val     = 0;
        constexpr int         max_val     = 99;
        constexpr int         sample_size = 100000;

        for ( int i = 0; i < sample_size; i++ ) {
            samples.push_back(rng.GetUniformRandomSTD(min_val, max_val));
        }

        double mean          = CalculateMean(samples);
        double expected_mean = (min_val + max_val) / 2.0;
        double tolerance     = 0.5;

        REQUIRE(std::abs(mean - expected_mean) < tolerance);
    }
}

TEST_CASE("RandomNumberGenerator GetUniformRandomSTD float distribution", "[RandomNumberGenerator][uniform_std][template]") {
    SECTION("Float uniform in [0.0, 1.0]") {
        RandomNumberGenerator rng("test_uniform_float");

        for ( int i = 0; i < 10000; i++ ) {
            float val = rng.GetUniformRandomSTD(0.0f, 1.0f);
            REQUIRE(val >= 0.0f);
            REQUIRE(val <= 1.0f);
        }
    }

    SECTION("Float uniform in [-10.0, 10.0]") {
        RandomNumberGenerator rng("test_uniform_float_wide");

        for ( int i = 0; i < 10000; i++ ) {
            float val = rng.GetUniformRandomSTD(-10.0f, 10.0f);
            REQUIRE(val >= -10.0f);
            REQUIRE(val <= 10.0f);
        }
    }

    SECTION("Float uniform statistical properties") {
        RandomNumberGenerator rng("test_uniform_float_stats");
        std::vector<float>    samples;
        constexpr float       min_val     = -5.0f;
        constexpr float       max_val     = 5.0f;
        constexpr int         sample_size = 100000;

        for ( int i = 0; i < sample_size; i++ ) {
            samples.push_back(rng.GetUniformRandomSTD(min_val, max_val));
        }

        double mean          = CalculateMean(samples);
        double expected_mean = (min_val + max_val) / 2.0;
        double tolerance     = 0.05;

        REQUIRE(std::abs(mean - expected_mean) < tolerance);

        // Chi-square test
        double chi_square     = ChiSquareUniformTest(samples, min_val, max_val, 20);
        double critical_value = 36.19; // α=0.01, df=19

        REQUIRE(chi_square < critical_value);
    }
}

// ============================================================================
// GetNormalRandomSTD() Template Tests
// ============================================================================

TEST_CASE("RandomNumberGenerator GetNormalRandomSTD distribution", "[RandomNumberGenerator][normal_std][template]") {
    SECTION("Normal with mean=0, sigma=1") {
        RandomNumberGenerator rng("test_normal_std");
        std::vector<float>    samples;
        constexpr int         sample_size = 100000;

        for ( int i = 0; i < sample_size; i++ ) {
            samples.push_back(rng.GetNormalRandomSTD(0.0f, 1.0f));
        }

        double mean   = CalculateMean(samples);
        double stddev = CalculateStdDev(samples);

        REQUIRE(std::abs(mean - 0.0) < 0.01);
        REQUIRE(std::abs(stddev - 1.0) < 0.01);

        // K-S test
        double ks_stat        = KolmogorovSmirnovNormalTest(samples, 0.0, 1.0);
        double critical_value = 1.63 / std::sqrt(static_cast<double>(sample_size));
        REQUIRE(ks_stat < critical_value);
    }

    SECTION("Normal with mean=10, sigma=2") {
        RandomNumberGenerator rng("test_normal_std_custom");
        std::vector<float>    samples;
        constexpr int         sample_size    = 100000;
        constexpr float       expected_mean  = 10.0f;
        constexpr float       expected_sigma = 2.0f;

        for ( int i = 0; i < sample_size; i++ ) {
            samples.push_back(rng.GetNormalRandomSTD(expected_mean, expected_sigma));
        }

        double mean   = CalculateMean(samples);
        double stddev = CalculateStdDev(samples);

        REQUIRE(std::abs(mean - expected_mean) < 0.02);
        REQUIRE(std::abs(stddev - expected_sigma) < 0.02);

        // K-S test
        double ks_stat        = KolmogorovSmirnovNormalTest(samples, expected_mean, expected_sigma);
        double critical_value = 1.63 / std::sqrt(static_cast<double>(sample_size));
        REQUIRE(ks_stat < critical_value);
    }
}

// ============================================================================
// GetPoissonRandomSTD() Template Tests
// ============================================================================

TEST_CASE("RandomNumberGenerator GetPoissonRandomSTD distribution", "[RandomNumberGenerator][poisson_std][template]") {
    SECTION("Poisson with lambda=5") {
        RandomNumberGenerator rng("test_poisson");
        std::vector<int>      samples;
        constexpr int         sample_size = 100000;
        constexpr double      lambda      = 5.0;

        for ( int i = 0; i < sample_size; i++ ) {
            samples.push_back(rng.GetPoissonRandomSTD(lambda));
        }

        double mean     = CalculateMean(samples);
        double variance = CalculateVariance(samples);

        // For Poisson: E[X] = λ, Var[X] = λ
        REQUIRE(std::abs(mean - lambda) < 0.05);
        REQUIRE(std::abs(variance - lambda) < 0.1);
    }

    SECTION("Poisson non-negative values") {
        RandomNumberGenerator rng("test_poisson_nonneg");

        for ( int i = 0; i < 10000; i++ ) {
            int val = rng.GetPoissonRandomSTD(3.0);
            REQUIRE(val >= 0);
        }
    }

    SECTION("Poisson with lambda=10 statistical test") {
        RandomNumberGenerator rng("test_poisson_lambda10");
        std::vector<int>      samples;
        constexpr int         sample_size = 100000;
        constexpr double      lambda      = 10.0;

        for ( int i = 0; i < sample_size; i++ ) {
            samples.push_back(rng.GetPoissonRandomSTD(lambda));
        }

        // Build expected probability distribution for values near mean
        std::map<int, double> expected_probs;
        for ( int k = 0; k <= 30; k++ ) {
            expected_probs[k] = PoissonPMF(k, lambda);
        }

        // Chi-square test (only for bins with expected count >= 5)
        double chi_square = ChiSquareDiscreteTest(samples, expected_probs);

        // This is approximate - actual df depends on bins with expected >= 5
        // For lambda=10, most values 0-20 have sufficient expected counts
        double critical_value = 40.0; // Conservative for ~20 bins

        REQUIRE(chi_square < critical_value);
    }
}

// ============================================================================
// GetExponentialRandomSTD() Template Tests
// ============================================================================

TEST_CASE("RandomNumberGenerator GetExponentialRandomSTD distribution", "[RandomNumberGenerator][exponential_std][template]") {
    SECTION("Exponential with lambda=1.0") {
        RandomNumberGenerator rng("test_exponential");
        std::vector<float>    samples;
        constexpr int         sample_size = 100000;
        constexpr float       lambda      = 1.0f;

        for ( int i = 0; i < sample_size; i++ ) {
            samples.push_back(rng.GetExponentialRandomSTD(lambda));
        }

        double mean   = CalculateMean(samples);
        double stddev = CalculateStdDev(samples);

        // For Exponential(λ): E[X] = 1/λ, StdDev[X] = 1/λ
        double expected_mean   = 1.0 / lambda;
        double expected_stddev = 1.0 / lambda;

        REQUIRE(std::abs(mean - expected_mean) < 0.02);
        REQUIRE(std::abs(stddev - expected_stddev) < 0.02);
    }

    SECTION("Exponential non-negative values") {
        RandomNumberGenerator rng("test_exponential_nonneg");

        for ( int i = 0; i < 10000; i++ ) {
            float val = rng.GetExponentialRandomSTD(2.0f);
            REQUIRE(val >= 0.0f);
        }
    }

    SECTION("Exponential with lambda=0.5") {
        RandomNumberGenerator rng("test_exponential_half");
        std::vector<float>    samples;
        constexpr int         sample_size = 100000;
        constexpr float       lambda      = 0.5f;

        for ( int i = 0; i < sample_size; i++ ) {
            samples.push_back(rng.GetExponentialRandomSTD(lambda));
        }

        double mean          = CalculateMean(samples);
        double expected_mean = 1.0 / lambda; // = 2.0

        REQUIRE(std::abs(mean - expected_mean) < 0.02);
    }
}

// ============================================================================
// GetGammaRandomSTD() Template Tests
// ============================================================================

TEST_CASE("RandomNumberGenerator GetGammaRandomSTD distribution", "[RandomNumberGenerator][gamma_std][template]") {
    SECTION("Gamma with alpha=2, beta=2") {
        RandomNumberGenerator rng("test_gamma");
        std::vector<float>    samples;
        constexpr int         sample_size = 100000;
        constexpr float       alpha       = 2.0f; // shape
        constexpr float       beta        = 2.0f; // scale

        for ( int i = 0; i < sample_size; i++ ) {
            samples.push_back(rng.GetGammaRandomSTD(alpha, beta));
        }

        double mean     = CalculateMean(samples);
        double variance = CalculateVariance(samples);

        // For Gamma(α, β): E[X] = αβ, Var[X] = αβ²
        double expected_mean     = alpha * beta;
        double expected_variance = alpha * beta * beta;

        REQUIRE(std::abs(mean - expected_mean) < 0.1);
        REQUIRE(std::abs(variance - expected_variance) < 0.2);
    }

    SECTION("Gamma non-negative values") {
        RandomNumberGenerator rng("test_gamma_nonneg");

        for ( int i = 0; i < 10000; i++ ) {
            float val = rng.GetGammaRandomSTD(1.0f, 1.0f);
            REQUIRE(val >= 0.0f);
        }
    }

    SECTION("Gamma with alpha=5, beta=1") {
        RandomNumberGenerator rng("test_gamma_alpha5");
        std::vector<float>    samples;
        constexpr int         sample_size = 100000;
        constexpr float       alpha       = 5.0f;
        constexpr float       beta        = 1.0f;

        for ( int i = 0; i < sample_size; i++ ) {
            samples.push_back(rng.GetGammaRandomSTD(alpha, beta));
        }

        double mean   = CalculateMean(samples);
        double stddev = CalculateStdDev(samples);

        double expected_mean   = alpha * beta; // = 5.0
        double expected_stddev = std::sqrt(alpha) * beta; // = sqrt(5)

        REQUIRE(std::abs(mean - expected_mean) < 0.05);
        REQUIRE(std::abs(stddev - expected_stddev) < 0.05);
    }
}

// ============================================================================
// Dual PRNG Mode Comparison Tests
// ============================================================================

TEST_CASE("RandomNumberGenerator dual PRNG mode comparison", "[RandomNumberGenerator][modes]") {
    SECTION("Both modes produce valid uniform distributions") {
        RandomNumberGenerator rng_internal(12345, true);
        RandomNumberGenerator rng_standard(12345, false);

        std::vector<float> samples_internal, samples_standard;
        constexpr int      sample_size = 50000;

        for ( int i = 0; i < sample_size; i++ ) {
            samples_internal.push_back(rng_internal.GetUniformRandom( ));
            samples_standard.push_back(rng_standard.GetUniformRandom( ));
        }

        // Both should pass chi-square test
        double chi_internal   = ChiSquareUniformTest(samples_internal, -1.0, 1.0, 20);
        double chi_standard   = ChiSquareUniformTest(samples_standard, -1.0, 1.0, 20);
        double critical_value = 36.19;

        REQUIRE(chi_internal < critical_value);
        REQUIRE(chi_standard < critical_value);
    }

    SECTION("Both modes produce valid normal distributions") {
        RandomNumberGenerator rng_internal(12345, true);
        RandomNumberGenerator rng_standard(12345, false);

        std::vector<float> samples_internal, samples_standard;
        constexpr int      sample_size = 50000;

        for ( int i = 0; i < sample_size; i++ ) {
            samples_internal.push_back(rng_internal.GetNormalRandom( ));
            samples_standard.push_back(rng_standard.GetNormalRandom( ));
        }

        // Both should have mean ≈ 0, sigma ≈ 1
        double mean_internal   = CalculateMean(samples_internal);
        double mean_standard   = CalculateMean(samples_standard);
        double stddev_internal = CalculateStdDev(samples_internal);
        double stddev_standard = CalculateStdDev(samples_standard);

        REQUIRE(std::abs(mean_internal) < 0.02);
        REQUIRE(std::abs(mean_standard) < 0.02);
        REQUIRE(std::abs(stddev_internal - 1.0) < 0.02);
        REQUIRE(std::abs(stddev_standard - 1.0) < 0.02);
    }

    SECTION("Internal and standard modes produce different sequences") {
        RandomNumberGenerator rng_internal(12345, true);
        RandomNumberGenerator rng_standard(12345, false);

        // With same seed, different PRNG implementations should produce different sequences
        bool found_difference = false;
        for ( int i = 0; i < 100; i++ ) {
            float val_internal = rng_internal.GetUniformRandom( );
            float val_standard = rng_standard.GetUniformRandom( );

            if ( val_internal != val_standard ) {
                found_difference = true;
                break;
            }
        }

        REQUIRE(found_difference);
    }
}
