/**
 * @file test_tensor_statistics.cpp
 * @brief Unit tests for Tensor statistical operations (Phase 2)
 *
 * Tests verify correctness using fixed known values.
 * Tests cover:
 * - Min/Max calculations
 * - Mean (average) and Sum
 * - Variance and standard deviation
 * - SumOfSquares (mean of squared values)
 * - SetConstant
 * - Both DenseLayout and FFTWPaddedLayout
 * - 2D and 3D tensors
 */

#include "../../../core/core_headers.h"
#include "../../../core/tensor/core/tensor.h"
#include "../../../core/tensor/memory/tensor_memory_pool.h"
#include "../../../core/tensor/operations/statistics.h"
#include "../../../../include/catch2/catch.hpp"

#include <cmath>

using namespace cistem::tensor;

TEST_CASE("Tensor: GetMinMax with known values", "[Tensor][Statistics]") {
    TensorMemoryPool<float> pool;
    auto                    buffer = pool.AllocateBuffer({8, 8, 1}, false);

    Tensor<float, DenseLayout> tensor;
    tensor.AttachToBuffer(buffer.data, {8, 8, 1});

    // Fill with sequence 0, 1, 2, ..., 63
    for ( int y = 0; y < 8; y++ ) {
        for ( int x = 0; x < 8; x++ ) {
            tensor.GetValue_p(x, y, 0) = float(y * 8 + x);
        }
    }

    float min_val, max_val;
    GetMinMax(tensor, min_val, max_val);

    REQUIRE(min_val == Approx(0.0f));
    REQUIRE(max_val == Approx(63.0f));

    pool.DeallocateBuffer(buffer);
}

TEST_CASE("Tensor: Mean of constant values", "[Tensor][Statistics]") {
    TensorMemoryPool<float> pool;
    auto                    buffer = pool.AllocateBuffer({4, 4, 1}, false);

    Tensor<float, DenseLayout> tensor;
    tensor.AttachToBuffer(buffer.data, {4, 4, 1});

    // Fill with constant
    for ( int y = 0; y < 4; y++ ) {
        for ( int x = 0; x < 4; x++ ) {
            tensor.GetValue_p(x, y, 0) = 42.0f;
        }
    }

    double mean = Mean(tensor);
    REQUIRE(mean == Approx(42.0));

    pool.DeallocateBuffer(buffer);
}

TEST_CASE("Tensor: Mean of sequence", "[Tensor][Statistics]") {
    TensorMemoryPool<float> pool;
    auto                    buffer = pool.AllocateBuffer({8, 8, 1}, false);

    Tensor<float, DenseLayout> tensor;
    tensor.AttachToBuffer(buffer.data, {8, 8, 1});

    // Fill with sequence 0, 1, 2, ..., 63
    for ( int y = 0; y < 8; y++ ) {
        for ( int x = 0; x < 8; x++ ) {
            tensor.GetValue_p(x, y, 0) = float(y * 8 + x);
        }
    }

    double mean     = Mean(tensor);
    double expected = (0.0 + 63.0) / 2.0; // Average of arithmetic sequence
    REQUIRE(mean == Approx(expected));

    pool.DeallocateBuffer(buffer);
}

TEST_CASE("Tensor: Sum calculation", "[Tensor][Statistics]") {
    TensorMemoryPool<float> pool;
    auto                    buffer = pool.AllocateBuffer({4, 4, 1}, false);

    Tensor<float, DenseLayout> tensor;
    tensor.AttachToBuffer(buffer.data, {4, 4, 1});

    // Fill with ones
    for ( int y = 0; y < 4; y++ ) {
        for ( int x = 0; x < 4; x++ ) {
            tensor.GetValue_p(x, y, 0) = 1.0f;
        }
    }

    double sum = Sum(tensor);
    REQUIRE(sum == Approx(16.0)); // 4x4 = 16 elements

    pool.DeallocateBuffer(buffer);
}

TEST_CASE("Tensor: Variance of constant values", "[Tensor][Statistics]") {
    TensorMemoryPool<float> pool;
    auto                    buffer = pool.AllocateBuffer({8, 8, 1}, false);

    Tensor<float, DenseLayout> tensor;
    tensor.AttachToBuffer(buffer.data, {8, 8, 1});

    // Fill with constant (variance should be zero)
    for ( int y = 0; y < 8; y++ ) {
        for ( int x = 0; x < 8; x++ ) {
            tensor.GetValue_p(x, y, 0) = 5.0f;
        }
    }

    float variance = Variance(tensor);
    REQUIRE(variance == Approx(0.0f).margin(1e-6));

    pool.DeallocateBuffer(buffer);
}

TEST_CASE("Tensor: Variance of sequence", "[Tensor][Statistics]") {
    TensorMemoryPool<float> pool;
    auto                    buffer = pool.AllocateBuffer({8, 8, 1}, false);

    Tensor<float, DenseLayout> tensor;
    tensor.AttachToBuffer(buffer.data, {8, 8, 1});

    // Fill with sequence 0, 1, 2, ..., 63
    for ( int y = 0; y < 8; y++ ) {
        for ( int x = 0; x < 8; x++ ) {
            tensor.GetValue_p(x, y, 0) = float(y * 8 + x);
        }
    }

    float variance = Variance(tensor);
    REQUIRE(variance > 0.0f); // Should have non-zero variance

    // Verify standard deviation is sqrt of variance
    float stddev = StandardDeviation(tensor);
    REQUIRE(stddev == Approx(std::sqrt(variance)));

    pool.DeallocateBuffer(buffer);
}

TEST_CASE("Tensor: SumOfSquares (returns mean)", "[Tensor][Statistics]") {
    TensorMemoryPool<float> pool;
    auto                    buffer = pool.AllocateBuffer({4, 4, 1}, false);

    Tensor<float, DenseLayout> tensor;
    tensor.AttachToBuffer(buffer.data, {4, 4, 1});

    // Fill with ones
    for ( int y = 0; y < 4; y++ ) {
        for ( int x = 0; x < 4; x++ ) {
            tensor.GetValue_p(x, y, 0) = 1.0f;
        }
    }

    // SumOfSquares returns MEAN of sum of squares
    double mean_sum_of_squares = SumOfSquares(tensor);
    REQUIRE(mean_sum_of_squares == Approx(1.0)); // 1^2 = 1, mean = 1

    pool.DeallocateBuffer(buffer);
}

TEST_CASE("Tensor: SumOfSquares with varied values", "[Tensor][Statistics]") {
    TensorMemoryPool<float> pool;
    auto                    buffer = pool.AllocateBuffer({2, 2, 1}, false);

    Tensor<float, DenseLayout> tensor;
    tensor.AttachToBuffer(buffer.data, {2, 2, 1});

    // Fill with known values: 1, 2, 3, 4
    tensor.GetValue_p(0, 0, 0) = 1.0f;
    tensor.GetValue_p(1, 0, 0) = 2.0f;
    tensor.GetValue_p(0, 1, 0) = 3.0f;
    tensor.GetValue_p(1, 1, 0) = 4.0f;

    // Sum of squares: 1 + 4 + 9 + 16 = 30
    // Mean: 30 / 4 = 7.5
    double mean_sum_of_squares = SumOfSquares(tensor);
    REQUIRE(mean_sum_of_squares == Approx(7.5));

    pool.DeallocateBuffer(buffer);
}

TEST_CASE("Tensor: SetConstant", "[Tensor][Statistics]") {
    TensorMemoryPool<float> pool;
    auto                    buffer = pool.AllocateBuffer({8, 8, 1}, false);

    Tensor<float, DenseLayout> tensor;
    tensor.AttachToBuffer(buffer.data, {8, 8, 1});

    SetConstant(tensor, 42.0f);

    // Verify all elements set to constant
    for ( int y = 0; y < 8; y++ ) {
        for ( int x = 0; x < 8; x++ ) {
            REQUIRE(tensor.GetValue_p(x, y, 0) == Approx(42.0f));
        }
    }

    pool.DeallocateBuffer(buffer);
}

TEST_CASE("Tensor: 3D statistics", "[Tensor][Statistics][3D]") {
    TensorMemoryPool<float> pool;
    auto                    buffer = pool.AllocateBuffer({4, 4, 4}, false);

    Tensor<float, DenseLayout> tensor;
    tensor.AttachToBuffer(buffer.data, {4, 4, 4});

    // Fill with pattern
    for ( int z = 0; z < 4; z++ ) {
        for ( int y = 0; y < 4; y++ ) {
            for ( int x = 0; x < 4; x++ ) {
                tensor.GetValue_p(x, y, z) = float(x + y + z);
            }
        }
    }

    SECTION("3D min/max") {
        float min_val, max_val;
        GetMinMax(tensor, min_val, max_val);

        REQUIRE(min_val == Approx(0.0f)); // x=0, y=0, z=0
        REQUIRE(max_val == Approx(9.0f)); // x=3, y=3, z=3
    }

    SECTION("3D mean") {
        double mean = Mean(tensor);
        REQUIRE(mean > 0.0);
    }

    SECTION("3D variance") {
        float variance = Variance(tensor);
        REQUIRE(variance > 0.0f);
    }

    pool.DeallocateBuffer(buffer);
}

TEST_CASE("Tensor: Statistics with FFTWPaddedLayout", "[Tensor][Statistics][Padding]") {
    SECTION("Mean with even dimensions (padding=2)") {
        TensorMemoryPool<float> pool;
        // For 64x64, FFTW padding adds 2 elements per row (64 is even)
        size_t padded_size = 66 * 64; // (64+2) * 64
        auto   buffer      = pool.AllocateBuffer({64, 64, 1}, false, padded_size);

        Tensor<float, FFTWPaddedLayout> tensor;
        tensor.AttachToBuffer(buffer.data, {64, 64, 1});

        // Fill logical region with ones
        for ( int y = 0; y < 64; y++ ) {
            for ( int x = 0; x < 64; x++ ) {
                tensor.GetValue_p(x, y, 0) = 1.0f;
            }
        }

        double mean = Mean(tensor);
        REQUIRE(mean == Approx(1.0));

        pool.DeallocateBuffer(buffer);
    }

    SECTION("Min/Max with odd dimensions (padding=1)") {
        TensorMemoryPool<float> pool;
        // For 63x64, FFTW padding adds 1 element per row (63 is odd)
        size_t padded_size = 64 * 64; // (63+1) * 64
        auto   buffer      = pool.AllocateBuffer({63, 64, 1}, false, padded_size);

        Tensor<float, FFTWPaddedLayout> tensor;
        tensor.AttachToBuffer(buffer.data, {63, 64, 1});

        // Fill with sequence
        for ( int y = 0; y < 64; y++ ) {
            for ( int x = 0; x < 63; x++ ) {
                tensor.GetValue_p(x, y, 0) = float(y * 63 + x);
            }
        }

        float min_val, max_val;
        GetMinMax(tensor, min_val, max_val);

        REQUIRE(min_val == Approx(0.0f));
        REQUIRE(max_val == Approx(float(63 * 64 - 1)));

        pool.DeallocateBuffer(buffer);
    }
}

TEST_CASE("Tensor: Edge cases for statistics", "[Tensor][Statistics]") {
    SECTION("Single element tensor") {
        TensorMemoryPool<float> pool;
        auto                    buffer = pool.AllocateBuffer({1, 1, 1}, false);

        Tensor<float, DenseLayout> tensor;
        tensor.AttachToBuffer(buffer.data, {1, 1, 1});

        tensor.GetValue_p(0, 0, 0) = 42.0f;

        double mean = Mean(tensor);
        REQUIRE(mean == Approx(42.0));

        float min_val, max_val;
        GetMinMax(tensor, min_val, max_val);
        REQUIRE(min_val == Approx(42.0f));
        REQUIRE(max_val == Approx(42.0f));

        float variance = Variance(tensor);
        REQUIRE(variance == Approx(0.0f).margin(1e-6));

        pool.DeallocateBuffer(buffer);
    }

    SECTION("All zeros") {
        TensorMemoryPool<float> pool;
        auto                    buffer = pool.AllocateBuffer({8, 8, 1}, false);

        Tensor<float, DenseLayout> tensor;
        tensor.AttachToBuffer(buffer.data, {8, 8, 1});

        SetConstant(tensor, 0.0f);

        double mean = Mean(tensor);
        REQUIRE(mean == Approx(0.0));

        float variance = Variance(tensor);
        REQUIRE(variance == Approx(0.0f).margin(1e-6));

        float stddev = StandardDeviation(tensor);
        REQUIRE(stddev == Approx(0.0f).margin(1e-6));

        pool.DeallocateBuffer(buffer);
    }

    SECTION("Negative values") {
        TensorMemoryPool<float> pool;
        auto                    buffer = pool.AllocateBuffer({4, 4, 1}, false);

        Tensor<float, DenseLayout> tensor;
        tensor.AttachToBuffer(buffer.data, {4, 4, 1});

        // Fill with negative constant
        for ( int y = 0; y < 4; y++ ) {
            for ( int x = 0; x < 4; x++ ) {
                tensor.GetValue_p(x, y, 0) = -10.0f;
            }
        }

        double mean = Mean(tensor);
        REQUIRE(mean == Approx(-10.0));

        float min_val, max_val;
        GetMinMax(tensor, min_val, max_val);
        REQUIRE(min_val == Approx(-10.0f));
        REQUIRE(max_val == Approx(-10.0f));

        pool.DeallocateBuffer(buffer);
    }

    SECTION("Mix of positive and negative") {
        TensorMemoryPool<float> pool;
        auto                    buffer = pool.AllocateBuffer({2, 2, 1}, false);

        Tensor<float, DenseLayout> tensor;
        tensor.AttachToBuffer(buffer.data, {2, 2, 1});

        // Fill: -2, -1, 1, 2
        tensor.GetValue_p(0, 0, 0) = -2.0f;
        tensor.GetValue_p(1, 0, 0) = -1.0f;
        tensor.GetValue_p(0, 1, 0) = 1.0f;
        tensor.GetValue_p(1, 1, 0) = 2.0f;

        double mean = Mean(tensor);
        REQUIRE(mean == Approx(0.0)); // (-2 + -1 + 1 + 2) / 4 = 0

        float min_val, max_val;
        GetMinMax(tensor, min_val, max_val);
        REQUIRE(min_val == Approx(-2.0f));
        REQUIRE(max_val == Approx(2.0f));

        pool.DeallocateBuffer(buffer);
    }
}

TEST_CASE("Tensor: Normalization workflow", "[Tensor][Statistics]") {
    TensorMemoryPool<float> pool;
    auto                    buffer = pool.AllocateBuffer({8, 8, 1}, false);

    Tensor<float, DenseLayout> tensor;
    tensor.AttachToBuffer(buffer.data, {8, 8, 1});

    // Fill with varied data
    for ( int y = 0; y < 8; y++ ) {
        for ( int x = 0; x < 8; x++ ) {
            tensor.GetValue_p(x, y, 0) = float(y * 8 + x + 1);
        }
    }

    double original_mean = Mean(tensor);

    // Normalize to mean of 1.0 (would use arithmetic.h in practice)
    for ( int y = 0; y < 8; y++ ) {
        for ( int x = 0; x < 8; x++ ) {
            tensor.GetValue_p(x, y, 0) /= float(original_mean);
        }
    }

    double normalized_mean = Mean(tensor);
    REQUIRE(normalized_mean == Approx(1.0).margin(0.001));

    pool.DeallocateBuffer(buffer);
}
