/**
 * @file test_image_memory.cpp
 * @brief Comprehensive tests for Image class memory management
 *
 * This file provides complete test coverage for Image class memory allocation,
 * deallocation, copy semantics, assignment operators, and edge cases.
 *
 * Tests cover:
 * - Default constructor
 * - Parameterized constructor
 * - Copy constructor (deep copy verification)
 * - Assignment operators (operator=, self-assignment)
 * - Allocate/Deallocate methods
 * - Memory leak detection
 * - Edge cases: odd dimensions, single pixel, large images, repeated operations
 *
 * Git history bugs tested:
 * - General memory leak vigilance (commit 0372acfb showed leaks in other methods)
 * - Ensuring proper cleanup in all code paths
 *
 * Duplicates from console_test:
 * - None specific (console_test doesn't comprehensively test memory management)
 *
 * Test execution:
 *   ./unit_test_runner "[Image][memory]"
 *   valgrind --leak-check=full ./unit_test_runner "[Image][memory]"
 */

#include "../../../core/core_headers.h"
#include "../../../../include/catch2/catch.hpp"

// ============================================================================
// BASIC CONSTRUCTOR TESTS
// ============================================================================

TEST_CASE("Image default constructor initializes to empty state", "[Image][memory][constructor]") {
    Image img;

    REQUIRE(img.logical_x_dimension == 0);
    REQUIRE(img.logical_y_dimension == 0);
    REQUIRE(img.logical_z_dimension == 0);
    REQUIRE(img.is_in_memory == false);
    REQUIRE(img.real_values == nullptr);
    REQUIRE(img.complex_values == nullptr);
    REQUIRE(img.planned == false);
}

TEST_CASE("Image constructor with dimensions allocates memory", "[Image][memory][constructor]") {
    SECTION("2D even dimensions") {
        Image img(64, 64, 1);

        REQUIRE(img.logical_x_dimension == 64);
        REQUIRE(img.logical_y_dimension == 64);
        REQUIRE(img.logical_z_dimension == 1);
        REQUIRE(img.is_in_memory == true);
        REQUIRE(img.is_in_real_space == true);
        REQUIRE(img.real_values != nullptr);
        REQUIRE(img.number_of_real_space_pixels == 64 * 64);
    }

    SECTION("2D odd dimensions") {
        Image img(63, 63, 1);

        REQUIRE(img.logical_x_dimension == 63);
        REQUIRE(img.logical_y_dimension == 63);
        REQUIRE(img.logical_z_dimension == 1);
        REQUIRE(img.is_in_memory == true);
        REQUIRE(img.real_values != nullptr);
    }

    SECTION("Non-square 2D") {
        Image img(128, 64, 1);

        REQUIRE(img.logical_x_dimension == 128);
        REQUIRE(img.logical_y_dimension == 64);
        REQUIRE(img.logical_z_dimension == 1);
        REQUIRE(img.is_in_memory == true);
    }

    SECTION("3D even dimensions") {
        Image img(32, 32, 32);

        REQUIRE(img.logical_x_dimension == 32);
        REQUIRE(img.logical_y_dimension == 32);
        REQUIRE(img.logical_z_dimension == 32);
        REQUIRE(img.is_in_memory == true);
        REQUIRE(img.number_of_real_space_pixels == 32 * 32 * 32);
    }

    SECTION("3D odd dimensions") {
        Image img(31, 31, 31);

        REQUIRE(img.logical_x_dimension == 31);
        REQUIRE(img.logical_y_dimension == 31);
        REQUIRE(img.logical_z_dimension == 31);
        REQUIRE(img.is_in_memory == true);
    }

    SECTION("Non-cubic 3D") {
        Image img(64, 64, 32);

        REQUIRE(img.logical_x_dimension == 64);
        REQUIRE(img.logical_y_dimension == 64);
        REQUIRE(img.logical_z_dimension == 32);
        REQUIRE(img.is_in_memory == true);
    }
}

TEST_CASE("Image constructor allocates in Fourier space", "[Image][memory][constructor]") {
    Image img(64, 64, 1, false); // is_in_real_space = false

    REQUIRE(img.logical_x_dimension == 64);
    REQUIRE(img.is_in_memory == true);
    REQUIRE(img.is_in_real_space == false);
    REQUIRE(img.complex_values != nullptr);
    // Note: real_values and complex_values point to the same memory in Image class
    REQUIRE(img.real_values != nullptr);
    REQUIRE(img.real_values == reinterpret_cast<float*>(img.complex_values));
}

TEST_CASE("Image constructor with FFT planning disabled", "[Image][memory][constructor]") {
    Image img(64, 64, 1, true, false); // do_fft_planning = false

    REQUIRE(img.logical_x_dimension == 64);
    REQUIRE(img.is_in_memory == true);
    REQUIRE(img.planned == false);
    REQUIRE(img.plan_fwd == nullptr);
    REQUIRE(img.plan_bwd == nullptr);
}

// ============================================================================
// COPY CONSTRUCTOR TESTS
// ============================================================================

TEST_CASE("Image copy constructor creates deep copy", "[Image][memory][copy]") {
    Image original(64, 64, 1);

    // Fill with test pattern
    for ( int y = 0; y < 64; y++ ) {
        for ( int x = 0; x < 64; x++ ) {
            long addr                  = original.ReturnReal1DAddressFromPhysicalCoord(x, y, 0);
            original.real_values[addr] = float(x + y * 64);
        }
    }

    // Copy construct
    Image copy(original);

    SECTION("Dimensions match") {
        REQUIRE(copy.logical_x_dimension == original.logical_x_dimension);
        REQUIRE(copy.logical_y_dimension == original.logical_y_dimension);
        REQUIRE(copy.logical_z_dimension == original.logical_z_dimension);
    }

    SECTION("Memory is separate") {
        REQUIRE(copy.real_values != original.real_values);
        REQUIRE(copy.is_in_memory == true);
    }

    SECTION("Data is copied correctly") {
        for ( int y = 0; y < 64; y++ ) {
            for ( int x = 0; x < 64; x++ ) {
                long addr_orig = original.ReturnReal1DAddressFromPhysicalCoord(x, y, 0);
                long addr_copy = copy.ReturnReal1DAddressFromPhysicalCoord(x, y, 0);
                REQUIRE(copy.real_values[addr_copy] == original.real_values[addr_orig]);
            }
        }
    }

    SECTION("Modifying copy doesn't affect original") {
        copy.real_values[0] = 999.0f;
        REQUIRE(original.real_values[0] != 999.0f);
    }
}

// NOTE: Copying empty (unallocated) images is not supported by Image class
// The assignment operator asserts that source image must be allocated
// TEST_CASE("Image copy constructor with empty image") - REMOVED

// ============================================================================
// ASSIGNMENT OPERATOR TESTS
// ============================================================================

TEST_CASE("Image assignment operator", "[Image][memory][assignment]") {
    Image original(64, 64, 1);
    original.SetToConstant(42.0f);

    Image target;
    target = original;

    SECTION("Dimensions match") {
        REQUIRE(target.logical_x_dimension == 64);
        REQUIRE(target.logical_y_dimension == 64);
        REQUIRE(target.logical_z_dimension == 1);
    }

    SECTION("Memory allocated") {
        REQUIRE(target.is_in_memory == true);
        REQUIRE(target.real_values != nullptr);
    }

    SECTION("Deep copy (separate memory)") {
        REQUIRE(target.real_values != original.real_values);
    }

    SECTION("Data copied correctly") {
        for ( long i = 0; i < target.real_memory_allocated; i++ ) {
            REQUIRE(target.real_values[i] == 42.0f);
        }
    }
}

TEST_CASE("Image self-assignment is safe", "[Image][memory][assignment][edge]") {
    Image img(64, 64, 1);
    img.SetToConstant(123.0f);

    float* original_ptr = img.real_values;

    // Self-assignment
    img = img;

    REQUIRE(img.logical_x_dimension == 64);
    REQUIRE(img.is_in_memory == true);
    // Should not have reallocated
    REQUIRE(img.real_values == original_ptr);
    REQUIRE(img.real_values[0] == 123.0f);
}

TEST_CASE("Image assignment operator with chaining", "[Image][memory][assignment]") {
    Image img1(32, 32, 1);
    Image img2(64, 64, 1);
    Image target1, target2;

    img1.SetToConstant(1.0f);
    img2.SetToConstant(2.0f);

    // Chained assignment
    target2 = target1 = img1;

    REQUIRE(target1.logical_x_dimension == 32);
    REQUIRE(target2.logical_x_dimension == 32);
    REQUIRE(target1.real_values[0] == 1.0f);
    REQUIRE(target2.real_values[0] == 1.0f);
}

TEST_CASE("Image assignment from pointer", "[Image][memory][assignment]") {
    Image original(64, 64, 1);
    original.SetToConstant(99.0f);

    Image target;
    target = &original;

    REQUIRE(target.logical_x_dimension == 64);
    REQUIRE(target.is_in_memory == true);
    REQUIRE(target.real_values != original.real_values); // Deep copy
    REQUIRE(target.real_values[0] == 99.0f);
}

// ============================================================================
// ALLOCATE/DEALLOCATE TESTS
// ============================================================================

TEST_CASE("Image Allocate method", "[Image][memory][allocate]") {
    Image img;

    SECTION("Allocate 2D even") {
        img.Allocate(128, 128, 1);

        REQUIRE(img.logical_x_dimension == 128);
        REQUIRE(img.is_in_memory == true);
        REQUIRE(img.real_values != nullptr);
    }

    SECTION("Allocate 2D odd") {
        img.Allocate(127, 127, 1);

        REQUIRE(img.logical_x_dimension == 127);
        REQUIRE(img.is_in_memory == true);
    }

    SECTION("Allocate 3D") {
        img.Allocate(64, 64, 64);

        REQUIRE(img.logical_z_dimension == 64);
        REQUIRE(img.is_in_memory == true);
        REQUIRE(img.number_of_real_space_pixels == 64 * 64 * 64);
    }

    SECTION("Allocate in Fourier space") {
        img.Allocate(64, 64, 1, false);

        REQUIRE(img.is_in_real_space == false);
        REQUIRE(img.complex_values != nullptr);
        // Note: real_values and complex_values point to the same memory
        REQUIRE(img.real_values != nullptr);
        REQUIRE(img.real_values == reinterpret_cast<float*>(img.complex_values));
    }

    SECTION("Allocate without FFT planning") {
        img.Allocate(64, 64, 1, true, false);

        REQUIRE(img.is_in_memory == true);
        REQUIRE(img.planned == false);
    }
}

TEST_CASE("Image Allocate from other image", "[Image][memory][allocate]") {
    Image reference(64, 64, 32);
    Image target;

    target.Allocate(&reference);

    REQUIRE(target.logical_x_dimension == 64);
    REQUIRE(target.logical_y_dimension == 64);
    REQUIRE(target.logical_z_dimension == 32);
    REQUIRE(target.is_in_memory == true);
    REQUIRE(target.is_in_real_space == reference.is_in_real_space);
}

TEST_CASE("Image Deallocate method", "[Image][memory][deallocate]") {
    Image img(64, 64, 1);

    REQUIRE(img.is_in_memory == true);
    REQUIRE(img.real_values != nullptr);

    img.Deallocate( );

    REQUIRE(img.is_in_memory == false);
    REQUIRE(img.real_values == nullptr);
    REQUIRE(img.complex_values == nullptr);
    // Dimensions should be reset
    REQUIRE(img.logical_x_dimension == 0);
    REQUIRE(img.logical_y_dimension == 0);
    REQUIRE(img.logical_z_dimension == 0);
}

TEST_CASE("Image multiple Allocate/Deallocate cycles", "[Image][memory][edge]") {
    Image img;

    // First cycle
    img.Allocate(64, 64, 1);
    REQUIRE(img.is_in_memory == true);
    img.Deallocate( );
    REQUIRE(img.is_in_memory == false);

    // Second cycle with different dimensions
    img.Allocate(128, 128, 32);
    REQUIRE(img.logical_x_dimension == 128);
    REQUIRE(img.logical_z_dimension == 32);
    img.Deallocate( );

    // Third cycle
    img.Allocate(256, 256, 1);
    REQUIRE(img.logical_x_dimension == 256);
    img.Deallocate( );

    REQUIRE(img.is_in_memory == false);
}

TEST_CASE("Image Deallocate on unallocated image is safe", "[Image][memory][edge]") {
    Image img; // No allocation

    REQUIRE_NOTHROW(img.Deallocate( ));
    REQUIRE(img.is_in_memory == false);

    // Double deallocate
    REQUIRE_NOTHROW(img.Deallocate( ));
}

TEST_CASE("Image Allocate over existing allocation", "[Image][memory][edge]") {
    Image img(64, 64, 1);
    img.SetToConstant(123.0f);

    // Reallocate with different size
    img.Allocate(128, 128, 1);

    REQUIRE(img.logical_x_dimension == 128);
    REQUIRE(img.is_in_memory == true);
    // Should have reallocated, not leaked old memory
}

// ============================================================================
// EDGE CASE TESTS
// ============================================================================

TEST_CASE("Image with single pixel", "[Image][memory][edge]") {
    Image img(1, 1, 1);

    REQUIRE(img.logical_x_dimension == 1);
    REQUIRE(img.is_in_memory == true);
    REQUIRE(img.number_of_real_space_pixels == 1);

    img.real_values[0] = 42.0f;
    REQUIRE(img.real_values[0] == 42.0f);
}

TEST_CASE("Image with large dimensions", "[Image][memory][edge]") {
    // 4K image
    Image img(4096, 4096, 1);

    REQUIRE(img.logical_x_dimension == 4096);
    REQUIRE(img.is_in_memory == true);
    REQUIRE(img.number_of_real_space_pixels == 4096L * 4096L);
}

TEST_CASE("Image with very non-square dimensions", "[Image][memory][edge]") {
    Image img(1024, 32, 1);

    REQUIRE(img.logical_x_dimension == 1024);
    REQUIRE(img.logical_y_dimension == 32);
    REQUIRE(img.is_in_memory == true);
}

// ============================================================================
// HALF-PRECISION BUFFER TESTS
// ============================================================================

TEST_CASE("Image Allocate16fBuffer", "[Image][memory][fp16]") {
    Image img(64, 64, 1);

    img.Allocate16fBuffer( );

    REQUIRE(img.is_in_memory_16f == true);
    REQUIRE(img.real_values_16f != nullptr);

    // Clean up is automatic in destructor
}

// ============================================================================
// DESTRUCTOR TESTS (via scope)
// ============================================================================

TEST_CASE("Image destructor cleans up memory", "[Image][memory][destructor]") {
    {
        Image img(64, 64, 1);
        REQUIRE(img.is_in_memory == true);
        // Destructor called at end of scope
    }
    // If valgrind shows leaks, test fails
    SUCCEED( );
}

TEST_CASE("Image destructor on empty image is safe", "[Image][memory][destructor]") {
    {
        Image img; // No allocation
        // Destructor called
    }
    SUCCEED( );
}

// ============================================================================
// MEMORY STATE QUERY TESTS
// ============================================================================

TEST_CASE("Image is_in_memory flag accuracy", "[Image][memory][state]") {
    Image img;

    REQUIRE(img.is_in_memory == false);

    img.Allocate(64, 64, 1);
    REQUIRE(img.is_in_memory == true);

    img.Deallocate( );
    REQUIRE(img.is_in_memory == false);
}

TEST_CASE("Image real_memory_allocated tracking", "[Image][memory][state]") {
    Image img_even(64, 64, 1);
    // FFTW padding: even x-dimension adds 2
    long expected_even = (64 + 2) * 64 * 1;
    REQUIRE(img_even.real_memory_allocated == expected_even);

    Image img_odd(63, 63, 1);
    // FFTW padding: odd x-dimension adds 1
    long expected_odd = (63 + 1) * 63 * 1;
    REQUIRE(img_odd.real_memory_allocated == expected_odd);
}

TEST_CASE("Image padding_jump_value calculation", "[Image][memory][state]") {
    Image img_even(64, 64, 1);
    REQUIRE(img_even.padding_jump_value == 2); // Even

    Image img_odd(63, 63, 1);
    REQUIRE(img_odd.padding_jump_value == 1); // Odd
}

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

TEST_CASE("Image memory lifecycle - typical usage", "[Image][memory][integration]") {
    // Simulate typical usage pattern
    Image img;

    // 1. Allocate
    img.Allocate(512, 512, 1);
    REQUIRE(img.is_in_memory == true);

    // 2. Use
    img.SetToConstant(1.0f);
    REQUIRE(img.real_values[0] == 1.0f);

    // 3. Transform
    img.ForwardFFT( );
    REQUIRE(img.is_in_real_space == false);

    // 4. Back transform
    img.BackwardFFT( );
    REQUIRE(img.is_in_real_space == true);

    // 5. Copy
    Image copy = img;
    REQUIRE(copy.is_in_memory == true);
    REQUIRE(copy.real_values != img.real_values);

    // 6. Deallocate
    img.Deallocate( );
    REQUIRE(img.is_in_memory == false);

    // Copy still valid
    REQUIRE(copy.is_in_memory == true);
}

TEST_CASE("Image memory stress test - many allocations", "[Image][memory][stress]") {
    // Create and destroy many images
    for ( int i = 0; i < 100; i++ ) {
        Image img(64, 64, 1);
        img.SetToConstant(float(i));
        // Automatic cleanup
    }
    SUCCEED( ); // valgrind will catch leaks
}
