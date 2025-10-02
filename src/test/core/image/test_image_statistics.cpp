/**
 * @file test_image_statistics.cpp
 * @brief Tests for Legacy Image statistical operations
 *
 * Establishes baseline behavior for Phase 2 Tensor statistical operations.
 * Tests verify:
 * - Min/Max calculations
 * - Mean (average) calculations
 * - Variance and standard deviation
 * - Sum operations
 * - Normalization operations
 *
 * Tensor implementations must match these results exactly.
 */

#include "../../../core/core_headers.h"
#include "../../../../include/catch2/catch.hpp"

TEST_CASE("Image: Min and Max", "[Image][Statistics]") {
    Image img(8, 8, 1);

    // Fill with pattern
    for ( int y = 0; y < img.logical_y_dimension; y++ ) {
        for ( int x = 0; x < img.logical_x_dimension; x++ ) {
            long addr             = img.ReturnReal1DAddressFromPhysicalCoord(x, y, 0);
            img.real_values[addr] = float(y * img.logical_x_dimension + x);
        }
    }

    float min_val, max_val;
    img.GetMinMax(min_val, max_val);

    REQUIRE(min_val == 0.0f);
    REQUIRE(max_val == Approx(float(img.logical_x_dimension * img.logical_y_dimension - 1)));
}

TEST_CASE("Image: Mean of constant values", "[Image][Statistics]") {
    Image img(4, 4, 1);

    // Fill with constant
    for ( int y = 0; y < img.logical_y_dimension; y++ ) {
        for ( int x = 0; x < img.logical_x_dimension; x++ ) {
            long addr             = img.ReturnReal1DAddressFromPhysicalCoord(x, y, 0);
            img.real_values[addr] = 42.0f;
        }
    }

    double mean = img.ReturnAverageOfRealValues( );
    REQUIRE(mean == Approx(42.0));
}

TEST_CASE("Image: Mean of sequence", "[Image][Statistics]") {
    Image img(8, 8, 1);

    // Fill with sequence 0..63
    for ( int y = 0; y < img.logical_y_dimension; y++ ) {
        for ( int x = 0; x < img.logical_x_dimension; x++ ) {
            long addr             = img.ReturnReal1DAddressFromPhysicalCoord(x, y, 0);
            img.real_values[addr] = float(y * img.logical_x_dimension + x);
        }
    }

    double mean     = img.ReturnAverageOfRealValues( );
    double expected = (0.0 + 63.0) / 2.0; // Average of 0..63
    REQUIRE(mean == Approx(expected));
}

TEST_CASE("Image: Variance calculation", "[Image][Statistics]") {
    Image img(8, 8, 1);

    // Fill with known pattern
    for ( int y = 0; y < img.logical_y_dimension; y++ ) {
        for ( int x = 0; x < img.logical_x_dimension; x++ ) {
            long addr             = img.ReturnReal1DAddressFromPhysicalCoord(x, y, 0);
            img.real_values[addr] = float(y * img.logical_x_dimension + x);
        }
    }

    float variance = img.ReturnVarianceOfRealValues( );

    // Variance should be positive for varied data
    REQUIRE(variance > 0.0);

    // Standard deviation is sqrt of variance
    float sigma = std::sqrt(variance);
    REQUIRE(sigma > 0.0);
}

TEST_CASE("Image: Sum of squares", "[Image][Statistics]") {
    Image img(4, 4, 1);

    for ( int y = 0; y < img.logical_y_dimension; y++ ) {
        for ( int x = 0; x < img.logical_x_dimension; x++ ) {
            long addr             = img.ReturnReal1DAddressFromPhysicalCoord(x, y, 0);
            img.real_values[addr] = 1.0f;
        }
    }

    // Note: ReturnSumOfSquares returns MEAN of sum of squares, not total sum
    double mean_sum_of_squares = img.ReturnSumOfSquares( );
    REQUIRE(mean_sum_of_squares == Approx(1.0)); // 4x4 pixels with value 1.0: sum=16, mean=1.0
}

TEST_CASE("Image: Set to constant", "[Image][Statistics]") {
    Image img(8, 8, 1);

    img.SetToConstant(42.0f);

    // Check all logical pixels
    for ( int y = 0; y < img.logical_y_dimension; y++ ) {
        for ( int x = 0; x < img.logical_x_dimension; x++ ) {
            long addr = img.ReturnReal1DAddressFromPhysicalCoord(x, y, 0);
            REQUIRE(img.real_values[addr] == 42.0f);
        }
    }
}

TEST_CASE("Image: Normalize to mean", "[Image][Statistics][Normalization]") {
    Image img(8, 8, 1);

    // Fill with varied data
    for ( int y = 0; y < img.logical_y_dimension; y++ ) {
        for ( int x = 0; x < img.logical_x_dimension; x++ ) {
            long addr             = img.ReturnReal1DAddressFromPhysicalCoord(x, y, 0);
            img.real_values[addr] = float(y * img.logical_x_dimension + x + 1);
        }
    }

    double original_mean = img.ReturnAverageOfRealValues( );
    img.MultiplyByConstant(1.0f / original_mean);

    // After normalization, mean should be 1.0
    double normalized_mean = img.ReturnAverageOfRealValues( );
    REQUIRE(normalized_mean == Approx(1.0).margin(0.001));
}

TEST_CASE("Image: 3D volume statistics", "[Image][Statistics][3D]") {
    Image volume(16, 16, 16);

    // Fill with pattern
    for ( int z = 0; z < volume.logical_z_dimension; z++ ) {
        for ( int y = 0; y < volume.logical_y_dimension; y++ ) {
            for ( int x = 0; x < volume.logical_x_dimension; x++ ) {
                long addr                = volume.ReturnReal1DAddressFromPhysicalCoord(x, y, z);
                volume.real_values[addr] = float(x + y + z);
            }
        }
    }

    SECTION("3D min/max") {
        float min_val, max_val;
        volume.GetMinMax(min_val, max_val);

        REQUIRE(min_val == 0.0f); // x=0, y=0, z=0
        float expected_max = float(15 + 15 + 15); // x=15, y=15, z=15
        REQUIRE(max_val == Approx(expected_max));
    }

    SECTION("3D mean") {
        double mean = volume.ReturnAverageOfRealValues( );
        REQUIRE(mean > 0.0); // Should be positive
    }

    SECTION("3D variance") {
        float variance = volume.ReturnVarianceOfRealValues( );
        REQUIRE(variance > 0.0);

        float sigma = std::sqrt(variance);
        REQUIRE(sigma > 0.0);
    }
}

TEST_CASE("Image: Statistics with padding", "[Image][Statistics][Padding]") {
    SECTION("Mean calculation ignores padding") {
        Image img(64, 64, 1); // Even -> padding of 2

        // Fill logical region
        for ( int y = 0; y < img.logical_y_dimension; y++ ) {
            for ( int x = 0; x < img.logical_x_dimension; x++ ) {
                long addr             = img.ReturnReal1DAddressFromPhysicalCoord(x, y, 0);
                img.real_values[addr] = 1.0f;
            }
        }

        double mean = img.ReturnAverageOfRealValues( );
        REQUIRE(mean == Approx(1.0));
    }

    SECTION("Min/Max ignores padding") {
        Image img(63, 64, 1); // Odd -> padding of 1

        for ( int y = 0; y < img.logical_y_dimension; y++ ) {
            for ( int x = 0; x < img.logical_x_dimension; x++ ) {
                long addr             = img.ReturnReal1DAddressFromPhysicalCoord(x, y, 0);
                img.real_values[addr] = float(y * img.logical_x_dimension + x);
            }
        }

        float min_val, max_val;
        img.GetMinMax(min_val, max_val);

        REQUIRE(min_val == 0.0f);
        REQUIRE(max_val == Approx(float(63 * 64 - 1)));
    }
}

TEST_CASE("Image: Edge cases for statistics", "[Image][Statistics]") {
    SECTION("Single pixel image") {
        Image img(1, 1, 1);
        img.real_values[0] = 42.0f;

        double mean = img.ReturnAverageOfRealValues( );
        REQUIRE(mean == Approx(42.0));

        float min_val, max_val;
        img.GetMinMax(min_val, max_val);
        REQUIRE(min_val == 42.0f);
        REQUIRE(max_val == 42.0f);
    }

    SECTION("All zeros") {
        Image img(8, 8, 1);
        img.SetToConstant(0.0f);

        double mean = img.ReturnAverageOfRealValues( );
        REQUIRE(mean == Approx(0.0));

        float variance = img.ReturnVarianceOfRealValues( );
        REQUIRE(variance == Approx(0.0));

        float sigma = std::sqrt(variance);
        REQUIRE(sigma == Approx(0.0));
    }

    SECTION("Negative values") {
        Image img(4, 4, 1);
        for ( int y = 0; y < img.logical_y_dimension; y++ ) {
            for ( int x = 0; x < img.logical_x_dimension; x++ ) {
                long addr             = img.ReturnReal1DAddressFromPhysicalCoord(x, y, 0);
                img.real_values[addr] = -10.0f;
            }
        }

        double mean = img.ReturnAverageOfRealValues( );
        REQUIRE(mean == Approx(-10.0));

        float min_val, max_val;
        img.GetMinMax(min_val, max_val);
        REQUIRE(min_val == -10.0f);
        REQUIRE(max_val == -10.0f);
    }
}
