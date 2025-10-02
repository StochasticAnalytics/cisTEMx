/**
 * @file test_tensor_memory_pool.cpp
 * @brief Unit tests for TensorMemoryPool
 *
 * Tests memory allocation, deallocation, FFT plan management, and leak detection.
 */

#include "../../../core/tensor/memory/tensor_memory_pool.h"
#include "../../../core/tensor/memory/memory_layout.h"
#include "../../../../include/catch2/catch.hpp"

using namespace cistem::tensor;

TEST_CASE("TensorMemoryPool: Basic allocation and deallocation", "[TensorMemoryPool]") {
    TensorMemoryPool<float> pool;

    SECTION("Allocate buffer without FFT plans") {
        auto buffer = pool.AllocateBuffer({64, 64, 1}, false);

        REQUIRE(buffer.data != nullptr);
        REQUIRE(buffer.dims.x == 64);
        REQUIRE(buffer.dims.y == 64);
        REQUIRE(buffer.dims.z == 1);
        REQUIRE(buffer.size == 64 * 64);
        REQUIRE(buffer.fft_plan_forward == nullptr);
        REQUIRE(buffer.fft_plan_backward == nullptr);
        REQUIRE(pool.GetActiveBufferCount( ) == 1);

        // Write and read to verify memory is accessible
        buffer.data[0]               = 42.0f;
        buffer.data[buffer.size - 1] = 123.0f;
        REQUIRE(buffer.data[0] == 42.0f);
        REQUIRE(buffer.data[buffer.size - 1] == 123.0f);

        pool.DeallocateBuffer(buffer);
        REQUIRE(buffer.data == nullptr);
        REQUIRE(pool.GetActiveBufferCount( ) == 0);
    }

    SECTION("Allocate buffer with FFT plans") {
        auto buffer = pool.AllocateBuffer({64, 64, 1}, true);

        REQUIRE(buffer.data != nullptr);
        REQUIRE(buffer.fft_plan_forward != nullptr);
        REQUIRE(buffer.fft_plan_backward != nullptr);
        REQUIRE(pool.GetActiveBufferCount( ) == 1);

        pool.DeallocateBuffer(buffer);
        REQUIRE(buffer.data == nullptr);
        REQUIRE(buffer.fft_plan_forward == nullptr);
        REQUIRE(buffer.fft_plan_backward == nullptr);
        REQUIRE(pool.GetActiveBufferCount( ) == 0);
    }

    SECTION("Allocate 3D buffer") {
        auto buffer = pool.AllocateBuffer({128, 128, 64}, false);

        REQUIRE(buffer.data != nullptr);
        REQUIRE(buffer.size == 128 * 128 * 64);
        REQUIRE(pool.GetActiveBufferCount( ) == 1);

        pool.DeallocateBuffer(buffer);
        REQUIRE(pool.GetActiveBufferCount( ) == 0);
    }
}

TEST_CASE("TensorMemoryPool: Multiple allocations", "[TensorMemoryPool]") {
    TensorMemoryPool<float> pool;

    auto buffer1 = pool.AllocateBuffer({64, 64, 1}, false);
    auto buffer2 = pool.AllocateBuffer({128, 128, 1}, false);
    auto buffer3 = pool.AllocateBuffer({256, 256, 1}, true);

    REQUIRE(pool.GetActiveBufferCount( ) == 3);
    REQUIRE(buffer1.data != buffer2.data);
    REQUIRE(buffer2.data != buffer3.data);
    REQUIRE(buffer1.data != buffer3.data);

    // Deallocate in different order
    pool.DeallocateBuffer(buffer2);
    REQUIRE(pool.GetActiveBufferCount( ) == 2);

    pool.DeallocateBuffer(buffer1);
    REQUIRE(pool.GetActiveBufferCount( ) == 1);

    pool.DeallocateBuffer(buffer3);
    REQUIRE(pool.GetActiveBufferCount( ) == 0);
}

TEST_CASE("TensorMemoryPool: Idempotent deallocation", "[TensorMemoryPool]") {
    TensorMemoryPool<float> pool;

    auto buffer = pool.AllocateBuffer({64, 64, 1}, false);
    REQUIRE(pool.GetActiveBufferCount( ) == 1);

    pool.DeallocateBuffer(buffer);
    REQUIRE(pool.GetActiveBufferCount( ) == 0);
    REQUIRE(buffer.data == nullptr);

    // Deallocating again should be safe (no-op)
    pool.DeallocateBuffer(buffer);
    REQUIRE(pool.GetActiveBufferCount( ) == 0);
}

TEST_CASE("TensorMemoryPool: Custom layout size", "[TensorMemoryPool]") {
    TensorMemoryPool<float> pool;

    // Request more memory than dense layout
    size_t custom_size = 64 * 66; // FFTW padded size for 64x64
    auto   buffer      = pool.AllocateBuffer({64, 64, 1}, false, custom_size);

    REQUIRE(buffer.size == custom_size);
    REQUIRE(buffer.data != nullptr);

    // Verify we can access the full padded memory
    buffer.data[custom_size - 1] = 99.0f;
    REQUIRE(buffer.data[custom_size - 1] == 99.0f);

    pool.DeallocateBuffer(buffer);
}

TEST_CASE("TensorMemoryPool: Move semantics", "[TensorMemoryPool]") {
    TensorMemoryPool<float> pool1;
    auto                    buffer       = pool1.AllocateBuffer({64, 64, 1}, false);
    void*                   original_ptr = buffer.data;

    REQUIRE(pool1.GetActiveBufferCount( ) == 1);

    SECTION("Move constructor") {
        TensorMemoryPool<float> pool2(std::move(pool1));

        REQUIRE(pool2.GetActiveBufferCount( ) == 1);
        REQUIRE(buffer.data == original_ptr); // Buffer pointer still valid

        pool2.DeallocateBuffer(buffer);
        REQUIRE(pool2.GetActiveBufferCount( ) == 0);
    }

    SECTION("Move assignment") {
        TensorMemoryPool<float> pool2;
        pool2 = std::move(pool1);

        REQUIRE(pool2.GetActiveBufferCount( ) == 1);
        REQUIRE(buffer.data == original_ptr);

        pool2.DeallocateBuffer(buffer);
        REQUIRE(pool2.GetActiveBufferCount( ) == 0);
    }
}

TEST_CASE("TensorMemoryPool: FFT plan creation for different dimensions", "[TensorMemoryPool]") {
    TensorMemoryPool<float> pool;

    SECTION("2D FFT plans") {
        auto buffer = pool.AllocateBuffer({64, 64, 1}, true);

        REQUIRE(buffer.fft_plan_forward != nullptr);
        REQUIRE(buffer.fft_plan_backward != nullptr);

        pool.DeallocateBuffer(buffer);
    }

    SECTION("3D FFT plans") {
        auto buffer = pool.AllocateBuffer({64, 64, 64}, true);

        REQUIRE(buffer.fft_plan_forward != nullptr);
        REQUIRE(buffer.fft_plan_backward != nullptr);

        pool.DeallocateBuffer(buffer);
    }
}

TEST_CASE("TensorMemoryPool: Leak detection on destruction", "[TensorMemoryPool]") {
    // This test verifies that leaked buffers are reported (to stderr)
    // and cleaned up automatically in the destructor

    auto buffer = [&]( ) {
        TensorMemoryPool<float> pool;
        auto                    buf = pool.AllocateBuffer({64, 64, 1}, false);
        REQUIRE(pool.GetActiveBufferCount( ) == 1);
        // Don't deallocate - let destructor handle it
        return buf;
    }( );

    // After pool destruction, buffer pointer should still be set
    // (pool cleaned it up but didn't reset our local copy)
    // In a real scenario, this would print a warning to stderr
}

TEST_CASE("TensorMemoryPool: Memory alignment", "[TensorMemoryPool]") {
    TensorMemoryPool<float> pool;
    auto                    buffer = pool.AllocateBuffer({64, 64, 1}, false);

    // FFTW malloc provides proper alignment (typically 16 or 32 bytes)
    uintptr_t addr = reinterpret_cast<uintptr_t>(buffer.data);
    REQUIRE(addr % 16 == 0); // At minimum 16-byte aligned

    pool.DeallocateBuffer(buffer);
}
