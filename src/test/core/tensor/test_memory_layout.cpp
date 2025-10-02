/**
 * @file test_memory_layout.cpp
 * @brief Unit tests for memory layout policies and address calculation
 *
 * Tests DenseLayout, FFTWPaddedLayout, and AddressCalculator.
 */

#include "../../../core/tensor/memory/memory_layout.h"
#include "../../../core/tensor/addressing/address_calculator.h"
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

TEST_CASE("AddressCalculator: DenseLayout real space addressing", "[AddressCalculator]") {
    int3 dims = {8, 8, 1};

    SECTION("First element") {
        long addr = AddressCalculator<DenseLayout>::Real1DAddress(0, 0, 0, dims);
        REQUIRE(addr == 0);
    }

    SECTION("First row elements") {
        long addr1 = AddressCalculator<DenseLayout>::Real1DAddress(1, 0, 0, dims);
        long addr2 = AddressCalculator<DenseLayout>::Real1DAddress(2, 0, 0, dims);
        long addr3 = AddressCalculator<DenseLayout>::Real1DAddress(7, 0, 0, dims);

        REQUIRE(addr1 == 1);
        REQUIRE(addr2 == 2);
        REQUIRE(addr3 == 7);
    }

    SECTION("Second row elements") {
        long addr = AddressCalculator<DenseLayout>::Real1DAddress(0, 1, 0, dims);
        REQUIRE(addr == 8); // Start of second row
    }

    SECTION("Random element") {
        long addr = AddressCalculator<DenseLayout>::Real1DAddress(3, 5, 0, dims);
        REQUIRE(addr == 5 * 8 + 3); // y * pitch + x
    }
}

TEST_CASE("AddressCalculator: DenseLayout 3D addressing", "[AddressCalculator]") {
    int3 dims = {10, 10, 10};

    SECTION("First slice") {
        long addr = AddressCalculator<DenseLayout>::Real1DAddress(5, 5, 0, dims);
        REQUIRE(addr == 5 * 10 + 5); // y * pitch + x
    }

    SECTION("Second slice") {
        long addr = AddressCalculator<DenseLayout>::Real1DAddress(5, 5, 1, dims);
        REQUIRE(addr == 1 * 100 + 5 * 10 + 5); // z * slice_size + y * pitch + x
    }

    SECTION("Last element") {
        long addr = AddressCalculator<DenseLayout>::Real1DAddress(9, 9, 9, dims);
        REQUIRE(addr == 9 * 100 + 9 * 10 + 9);
        REQUIRE(addr == 999); // 10*10*10 - 1
    }
}

TEST_CASE("AddressCalculator: FFTWPaddedLayout real space addressing", "[AddressCalculator]") {
    int3 dims = {64, 64, 1};

    SECTION("First element") {
        long addr = AddressCalculator<FFTWPaddedLayout>::Real1DAddress(0, 0, 0, dims);
        REQUIRE(addr == 0);
    }

    SECTION("First row") {
        long addr1 = AddressCalculator<FFTWPaddedLayout>::Real1DAddress(63, 0, 0, dims);
        REQUIRE(addr1 == 63); // Last logical element in first row
    }

    SECTION("Second row starts at pitch offset") {
        long addr = AddressCalculator<FFTWPaddedLayout>::Real1DAddress(0, 1, 0, dims);
        REQUIRE(addr == 66); // Pitch = 64 + 2
    }

    SECTION("Element in second row") {
        long addr = AddressCalculator<FFTWPaddedLayout>::Real1DAddress(10, 1, 0, dims);
        REQUIRE(addr == 66 + 10); // pitch + x
    }
}

TEST_CASE("AddressCalculator: Fourier space addressing", "[AddressCalculator]") {
    int3 dims = {64, 64, 1};

    SECTION("DC component") {
        long addr = AddressCalculator<DenseLayout>::Fourier1DAddress(0, 0, 0, dims);
        REQUIRE(addr == 0);
    }

    SECTION("Hermitian symmetry - reduced X dimension") {
        // In Fourier space, complex pitch is dims.x/2 + 1 = 33
        long addr1 = AddressCalculator<DenseLayout>::Fourier1DAddress(32, 0, 0, dims);
        REQUIRE(addr1 == 32); // Last element in first row

        long addr2 = AddressCalculator<DenseLayout>::Fourier1DAddress(0, 1, 0, dims);
        REQUIRE(addr2 == 33); // First element in second row
    }

    SECTION("Element in Fourier space") {
        long addr     = AddressCalculator<DenseLayout>::Fourier1DAddress(10, 5, 0, dims);
        long expected = 5 * 33 + 10; // y * complex_pitch + x
        REQUIRE(addr == expected);
    }
}

TEST_CASE("AddressCalculator: Consistency between layouts", "[AddressCalculator]") {
    int3 dims = {64, 64, 1};

    SECTION("Dense and FFTW addressing match within logical bounds") {
        // For elements within logical dimensions, addressing should be predictable

        long dense_addr0 = AddressCalculator<DenseLayout>::Real1DAddress(10, 10, 0, dims);
        long fftw_addr0  = AddressCalculator<FFTWPaddedLayout>::Real1DAddress(10, 10, 0, dims);

        // Dense: 10 * 64 + 10 = 650
        // FFTW:  10 * 66 + 10 = 670
        REQUIRE(dense_addr0 == 650);
        REQUIRE(fftw_addr0 == 670);

        // Different because of padding, but both valid for their layouts
        REQUIRE(dense_addr0 != fftw_addr0);
    }
}
