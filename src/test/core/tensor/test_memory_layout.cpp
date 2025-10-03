/**
 * @file test_memory_layout.cpp
 * @brief Unit tests for memory layout policies and address calculation
 *
 * Tests DenseLayout, FFTWPaddedLayout, and AddressCalculator.
 */

#include "../../../core/tensor/memory/memory_layout.h"
#include "../../../../include/catch2/catch.hpp"

using namespace cistem::tensor;

TEST_CASE("DenseLayout: Pitch calculation", "[MemoryLayout]") {
    SECTION("2D layout") {
        int3   dims  = {64, 64, 1};
        size_t pitch = DenseLayout::CalculatePitch(dims);
        REQUIRE(pitch == 64);
    }

    SECTION("3D layout") {
        int3   dims  = {128, 256, 64};
        size_t pitch = DenseLayout::CalculatePitch(dims);
        REQUIRE(pitch == 128);
    }

    SECTION("Non-square layout") {
        int3   dims  = {100, 200, 50};
        size_t pitch = DenseLayout::CalculatePitch(dims);
        REQUIRE(pitch == 100);
    }
}

TEST_CASE("DenseLayout: Memory size calculation", "[MemoryLayout]") {
    SECTION("2D layout") {
        int3   dims = {64, 64, 1};
        size_t size = DenseLayout::CalculateMemorySize(dims);
        REQUIRE(size == 64 * 64);
    }

    SECTION("3D layout") {
        int3   dims = {128, 256, 64};
        size_t size = DenseLayout::CalculateMemorySize(dims);
        REQUIRE(size == 128 * 256 * 64);
    }

    SECTION("Single slice") {
        int3   dims = {512, 512, 1};
        size_t size = DenseLayout::CalculateMemorySize(dims);
        REQUIRE(size == 512 * 512);
    }
}

TEST_CASE("FFTWPaddedLayout: Padding jump value", "[MemoryLayout]") {
    SECTION("Even X dimension") {
        int3 dims    = {64, 64, 1};
        int  padding = FFTWPaddedLayout::CalculatePaddingJumpValue(dims);
        REQUIRE(padding == 2);
    }

    SECTION("Odd X dimension") {
        int3 dims    = {63, 64, 1};
        int  padding = FFTWPaddedLayout::CalculatePaddingJumpValue(dims);
        REQUIRE(padding == 1);
    }

    SECTION("Large even dimension") {
        int3 dims    = {512, 512, 1};
        int  padding = FFTWPaddedLayout::CalculatePaddingJumpValue(dims);
        REQUIRE(padding == 2);
    }

    SECTION("Large odd dimension") {
        int3 dims    = {511, 512, 1};
        int  padding = FFTWPaddedLayout::CalculatePaddingJumpValue(dims);
        REQUIRE(padding == 1);
    }
}

TEST_CASE("FFTWPaddedLayout: Pitch calculation", "[MemoryLayout]") {
    SECTION("Even X dimension") {
        int3   dims  = {64, 64, 1};
        size_t pitch = FFTWPaddedLayout::CalculatePitch(dims);
        REQUIRE(pitch == 66); // 64 + 2
    }

    SECTION("Odd X dimension") {
        int3   dims  = {63, 64, 1};
        size_t pitch = FFTWPaddedLayout::CalculatePitch(dims);
        REQUIRE(pitch == 64); // 63 + 1
    }

    SECTION("Large even dimension") {
        int3   dims  = {512, 512, 1};
        size_t pitch = FFTWPaddedLayout::CalculatePitch(dims);
        REQUIRE(pitch == 514); // 512 + 2
    }
}

TEST_CASE("FFTWPaddedLayout: Memory size calculation", "[MemoryLayout]") {
    SECTION("2D even dimensions") {
        int3   dims = {64, 64, 1};
        size_t size = FFTWPaddedLayout::CalculateMemorySize(dims);
        REQUIRE(size == 66 * 64); // (64+2) * 64
    }

    SECTION("2D odd X dimension") {
        int3   dims = {63, 64, 1};
        size_t size = FFTWPaddedLayout::CalculateMemorySize(dims);
        REQUIRE(size == 64 * 64); // (63+1) * 64
    }

    SECTION("3D volume") {
        int3   dims = {64, 64, 64};
        size_t size = FFTWPaddedLayout::CalculateMemorySize(dims);
        REQUIRE(size == 66 * 64 * 64); // (64+2) * 64 * 64
    }
}

TEST_CASE("FFTWPaddedLayout: Complex pitch calculation", "[MemoryLayout]") {
    SECTION("64x64 real to complex") {
        int3   dims          = {64, 64, 1};
        size_t complex_pitch = FFTWPaddedLayout::CalculateComplexPitch(dims);
        REQUIRE(complex_pitch == 33); // 64/2 + 1
    }

    SECTION("128x128 real to complex") {
        int3   dims          = {128, 128, 1};
        size_t complex_pitch = FFTWPaddedLayout::CalculateComplexPitch(dims);
        REQUIRE(complex_pitch == 65); // 128/2 + 1
    }

    SECTION("Odd dimension real to complex") {
        int3   dims          = {63, 64, 1};
        size_t complex_pitch = FFTWPaddedLayout::CalculateComplexPitch(dims);
        REQUIRE(complex_pitch == 32); // 63/2 + 1 = 31 + 1 = 32
    }
}
