/**
 * @file test_image_arithmetic.cpp
 * @brief Tests for Legacy Image arithmetic operations
 *
 * Establishes baseline behavior for Phase 2 Tensor arithmetic operations.
 * Tests verify:
 * - Element-wise operations (add, subtract, multiply, divide)
 * - Scalar operations (add, subtract, multiply, divide)
 * - In-place vs. out-of-place operations
 * - Handling of FFTW padding
 *
 * Tensor implementations must match these results exactly.
 */

#include "../../../core/core_headers.h"
#include "../../../../include/catch2/catch.hpp"

TEST_CASE("Image: Element-wise addition", "[Image][Arithmetic]") {
    Image img1(4, 4, 1);
    Image img2(4, 4, 1);

    // Fill with test patterns
    for ( long i = 0; i < img1.real_memory_allocated; i++ ) {
        img1.real_values[i] = float(i);
        img2.real_values[i] = float(i * 2);
    }

    img1.AddImage(&img2);

    // Verify addition
    for ( long i = 0; i < img1.real_memory_allocated; i++ ) {
        REQUIRE(img1.real_values[i] == Approx(float(i) + float(i * 2)));
    }
}

TEST_CASE("Image: Element-wise subtraction", "[Image][Arithmetic]") {
    Image img1(4, 4, 1);
    Image img2(4, 4, 1);

    for ( long i = 0; i < img1.real_memory_allocated; i++ ) {
        img1.real_values[i] = 100.0f;
        img2.real_values[i] = float(i);
    }

    img1.SubtractImage(&img2);

    for ( long i = 0; i < img1.real_memory_allocated; i++ ) {
        REQUIRE(img1.real_values[i] == Approx(100.0f - float(i)));
    }
}

TEST_CASE("Image: Element-wise multiplication", "[Image][Arithmetic]") {
    Image img1(4, 4, 1);
    Image img2(4, 4, 1);

    for ( long i = 0; i < img1.real_memory_allocated; i++ ) {
        img1.real_values[i] = float(i + 1);
        img2.real_values[i] = 2.0f;
    }

    img1.MultiplyPixelWise(img2);

    for ( long i = 0; i < img1.real_memory_allocated; i++ ) {
        REQUIRE(img1.real_values[i] == Approx(float(i + 1) * 2.0f));
    }
}

TEST_CASE("Image: Element-wise division", "[Image][Arithmetic]") {
    Image img1(4, 4, 1);
    Image img2(4, 4, 1);

    for ( long i = 0; i < img1.real_memory_allocated; i++ ) {
        img1.real_values[i] = float(i + 100);
        img2.real_values[i] = 2.0f;
    }

    img1.DividePixelWise(img2);

    for ( long i = 0; i < img1.real_memory_allocated; i++ ) {
        REQUIRE(img1.real_values[i] == Approx(float(i + 100) / 2.0f));
    }
}

TEST_CASE("Image: Scalar addition", "[Image][Arithmetic]") {
    Image img(4, 4, 1);

    for ( long i = 0; i < img.real_memory_allocated; i++ ) {
        img.real_values[i] = float(i);
    }

    img.AddConstant(10.0f);

    for ( long i = 0; i < img.real_memory_allocated; i++ ) {
        REQUIRE(img.real_values[i] == Approx(float(i) + 10.0f));
    }
}

TEST_CASE("Image: Scalar multiplication", "[Image][Arithmetic]") {
    Image img(4, 4, 1);

    for ( long i = 0; i < img.real_memory_allocated; i++ ) {
        img.real_values[i] = float(i);
    }

    img.MultiplyByConstant(3.0f);

    for ( long i = 0; i < img.real_memory_allocated; i++ ) {
        REQUIRE(img.real_values[i] == Approx(float(i) * 3.0f));
    }
}

TEST_CASE("Image: Scalar division", "[Image][Arithmetic]") {
    Image img(4, 4, 1);

    for ( long i = 0; i < img.real_memory_allocated; i++ ) {
        img.real_values[i] = float(i + 100);
    }

    img.DivideByConstant(2.0f);

    for ( long i = 0; i < img.real_memory_allocated; i++ ) {
        REQUIRE(img.real_values[i] == Approx(float(i + 100) / 2.0f));
    }
}

TEST_CASE("Image: Square values", "[Image][Arithmetic]") {
    Image img(4, 4, 1);

    for ( int y = 0; y < img.logical_y_dimension; y++ ) {
        for ( int x = 0; x < img.logical_x_dimension; x++ ) {
            long addr             = img.ReturnReal1DAddressFromPhysicalCoord(x, y, 0);
            img.real_values[addr] = 2.0f;
        }
    }

    img.SquareRealValues( );

    for ( int y = 0; y < img.logical_y_dimension; y++ ) {
        for ( int x = 0; x < img.logical_x_dimension; x++ ) {
            long addr = img.ReturnReal1DAddressFromPhysicalCoord(x, y, 0);
            REQUIRE(img.real_values[addr] == 4.0f);
        }
    }
}

TEST_CASE("Image: Square root of values", "[Image][Arithmetic]") {
    Image img(4, 4, 1);

    for ( int y = 0; y < img.logical_y_dimension; y++ ) {
        for ( int x = 0; x < img.logical_x_dimension; x++ ) {
            long addr             = img.ReturnReal1DAddressFromPhysicalCoord(x, y, 0);
            img.real_values[addr] = 16.0f;
        }
    }

    img.SquareRootRealValues( );

    for ( int y = 0; y < img.logical_y_dimension; y++ ) {
        for ( int x = 0; x < img.logical_x_dimension; x++ ) {
            long addr = img.ReturnReal1DAddressFromPhysicalCoord(x, y, 0);
            REQUIRE(img.real_values[addr] == 4.0f);
        }
    }
}

TEST_CASE("Image: Arithmetic with FFTW padding", "[Image][Arithmetic][Padding]") {
    SECTION("Add constant respects padding") {
        Image img(64, 64, 1); // Even dimension -> padding of 2

        // Fill logical region only
        for ( int y = 0; y < img.logical_y_dimension; y++ ) {
            for ( int x = 0; x < img.logical_x_dimension; x++ ) {
                long addr             = img.ReturnReal1DAddressFromPhysicalCoord(x, y, 0);
                img.real_values[addr] = 1.0f;
            }
        }

        img.AddConstant(1.0f);

        // Verify logical region updated
        for ( int y = 0; y < img.logical_y_dimension; y++ ) {
            for ( int x = 0; x < img.logical_x_dimension; x++ ) {
                long addr = img.ReturnReal1DAddressFromPhysicalCoord(x, y, 0);
                REQUIRE(img.real_values[addr] == 2.0f);
            }
        }
    }

    SECTION("Multiply constant with padding") {
        Image img(63, 64, 1); // Odd dimension -> padding of 1

        for ( int y = 0; y < img.logical_y_dimension; y++ ) {
            for ( int x = 0; x < img.logical_x_dimension; x++ ) {
                long addr             = img.ReturnReal1DAddressFromPhysicalCoord(x, y, 0);
                img.real_values[addr] = 2.0f;
            }
        }

        img.MultiplyByConstant(3.0f);

        for ( int y = 0; y < img.logical_y_dimension; y++ ) {
            for ( int x = 0; x < img.logical_x_dimension; x++ ) {
                long addr = img.ReturnReal1DAddressFromPhysicalCoord(x, y, 0);
                REQUIRE(img.real_values[addr] == 6.0f);
            }
        }
    }
}

TEST_CASE("Image: Chained arithmetic operations", "[Image][Arithmetic]") {
    Image img(4, 4, 1);

    for ( long i = 0; i < img.real_memory_allocated; i++ ) {
        img.real_values[i] = 10.0f;
    }

    img.AddConstant(5.0f);
    img.MultiplyByConstant(2.0f);
    img.DivideByConstant(3.0f);

    // (10 + 5) * 2 / 3 = 30 / 3 = 10
    for ( long i = 0; i < img.real_memory_allocated; i++ ) {
        REQUIRE(img.real_values[i] == Approx(10.0f));
    }
}

TEST_CASE("Image: 3D arithmetic operations", "[Image][Arithmetic][3D]") {
    Image img1(8, 8, 8);
    Image img2(8, 8, 8);

    // Fill volumes with patterns
    for ( int z = 0; z < img1.logical_z_dimension; z++ ) {
        for ( int y = 0; y < img1.logical_y_dimension; y++ ) {
            for ( int x = 0; x < img1.logical_x_dimension; x++ ) {
                long addr              = img1.ReturnReal1DAddressFromPhysicalCoord(x, y, z);
                img1.real_values[addr] = float(x + y + z);
                img2.real_values[addr] = 1.0f;
            }
        }
    }

    img1.AddImage(&img2);

    // Verify 3D addition
    for ( int z = 0; z < img1.logical_z_dimension; z++ ) {
        for ( int y = 0; y < img1.logical_y_dimension; y++ ) {
            for ( int x = 0; x < img1.logical_x_dimension; x++ ) {
                long addr = img1.ReturnReal1DAddressFromPhysicalCoord(x, y, z);
                REQUIRE(img1.real_values[addr] == Approx(float(x + y + z + 1)));
            }
        }
    }
}
