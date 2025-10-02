/**
 * @file test_tensor.cpp
 * @brief Unit tests for Tensor class
 *
 * Tests tensor views, element access, space management, and metadata.
 */

#include "../../../core/tensor/core/tensor.h"
#include "../../../core/tensor/memory/tensor_memory_pool.h"
#include "../../../core/tensor/memory/memory_layout.h"
#include "../../../../include/catch2/catch.hpp"

using namespace cistem::tensor;

TEST_CASE("Tensor: Construction and attachment", "[Tensor]") {
    TensorMemoryPool<float> pool;

    SECTION("Default construction") {
        Tensor<float, DenseLayout> tensor;

        REQUIRE_FALSE(tensor.IsAttached( ));
        REQUIRE(tensor.GetDims( ).x == 0);
        REQUIRE(tensor.GetDims( ).y == 0);
        REQUIRE(tensor.GetDims( ).z == 0);
        REQUIRE(tensor.IsInPositionSpace( ));
    }

    SECTION("Attach to buffer") {
        auto                       buffer = pool.AllocateBuffer({64, 64, 1}, false);
        Tensor<float, DenseLayout> tensor;

        tensor.AttachToBuffer(buffer.data, buffer.dims);

        REQUIRE(tensor.IsAttached( ));
        REQUIRE(tensor.GetDims( ).x == 64);
        REQUIRE(tensor.GetDims( ).y == 64);
        REQUIRE(tensor.GetDims( ).z == 1);
        REQUIRE(tensor.GetLogicalSize( ) == 64 * 64);
        REQUIRE(tensor.Data( ) == buffer.data);

        pool.DeallocateBuffer(buffer);
    }

    SECTION("Detach from buffer") {
        auto                       buffer = pool.AllocateBuffer({64, 64, 1}, false);
        Tensor<float, DenseLayout> tensor;

        tensor.AttachToBuffer(buffer.data, buffer.dims);
        REQUIRE(tensor.IsAttached( ));

        tensor.Detach( );
        REQUIRE_FALSE(tensor.IsAttached( ));
        REQUIRE(tensor.Data( ) == nullptr);

        pool.DeallocateBuffer(buffer);
    }
}

TEST_CASE("Tensor: Element access with DenseLayout", "[Tensor]") {
    TensorMemoryPool<float>    pool;
    auto                       buffer = pool.AllocateBuffer({8, 8, 1}, false);
    Tensor<float, DenseLayout> tensor;

    tensor.AttachToBuffer(buffer.data, buffer.dims);
    tensor.SetSpace(Tensor<float>::Space::Position);

    SECTION("Write and read elements") {
        // Fill with pattern
        for ( int z = 0; z < 1; z++ ) {
            for ( int y = 0; y < 8; y++ ) {
                for ( int x = 0; x < 8; x++ ) {
                    tensor(x, y, z) = float(x + y * 8 + z * 64);
                }
            }
        }

        // Verify pattern
        for ( int z = 0; z < 1; z++ ) {
            for ( int y = 0; y < 8; y++ ) {
                for ( int x = 0; x < 8; x++ ) {
                    REQUIRE(tensor(x, y, z) == float(x + y * 8 + z * 64));
                }
            }
        }
    }

    SECTION("Const element access") {
        tensor(3, 4, 0) = 42.0f;

        const auto& const_tensor = tensor;
        REQUIRE(const_tensor(3, 4, 0) == 42.0f);
    }

    pool.DeallocateBuffer(buffer);
}

TEST_CASE("Tensor: Element access with FFTWPaddedLayout", "[Tensor]") {
    TensorMemoryPool<float> pool;

    // For 64x64, FFTW padding adds 2 elements per row (64 is even)
    size_t padded_size = 66 * 64; // (64+2) * 64
    auto   buffer      = pool.AllocateBuffer({64, 64, 1}, false, padded_size);

    Tensor<float, FFTWPaddedLayout> tensor;
    tensor.AttachToBuffer(buffer.data, buffer.dims);
    tensor.SetSpace(Tensor<float, FFTWPaddedLayout>::Space::Position);

    SECTION("Pitch calculation") {
        size_t pitch = tensor.GetPitch( );
        REQUIRE(pitch == 66); // 64 + 2 for even dimension
    }

    SECTION("Physical size includes padding") {
        size_t physical_size = tensor.GetPhysicalSize( );
        REQUIRE(physical_size == 66 * 64);
    }

    SECTION("Write and read with padding") {
        // Fill first row
        for ( int x = 0; x < 64; x++ ) {
            tensor(x, 0, 0) = float(x);
        }

        // Verify first row
        for ( int x = 0; x < 64; x++ ) {
            REQUIRE(tensor(x, 0, 0) == float(x));
        }

        // Fill second row (accounting for padding in layout)
        for ( int x = 0; x < 64; x++ ) {
            tensor(x, 1, 0) = float(x + 100);
        }

        // Verify second row
        for ( int x = 0; x < 64; x++ ) {
            REQUIRE(tensor(x, 1, 0) == float(x + 100));
        }
    }

    pool.DeallocateBuffer(buffer);
}

TEST_CASE("Tensor: Space management", "[Tensor]") {
    TensorMemoryPool<float>    pool;
    auto                       buffer = pool.AllocateBuffer({64, 64, 1}, false);
    Tensor<float, DenseLayout> tensor;

    tensor.AttachToBuffer(buffer.data, buffer.dims);

    SECTION("Default space is Position") {
        REQUIRE(tensor.GetSpace( ) == Tensor<float>::Space::Position);
        REQUIRE(tensor.IsInPositionSpace( ));
        REQUIRE_FALSE(tensor.IsInMomentumSpace( ));
    }

    SECTION("Change to Momentum space") {
        tensor.SetSpace(Tensor<float>::Space::Momentum);

        REQUIRE(tensor.GetSpace( ) == Tensor<float>::Space::Momentum);
        REQUIRE_FALSE(tensor.IsInPositionSpace( ));
        REQUIRE(tensor.IsInMomentumSpace( ));
    }

    SECTION("Toggle space") {
        tensor.SetSpace(Tensor<float>::Space::Momentum);
        REQUIRE(tensor.IsInMomentumSpace( ));

        tensor.SetSpace(Tensor<float>::Space::Position);
        REQUIRE(tensor.IsInPositionSpace( ));
    }

    pool.DeallocateBuffer(buffer);
}

TEST_CASE("Tensor: Centering metadata", "[Tensor]") {
    TensorMemoryPool<float>    pool;
    auto                       buffer = pool.AllocateBuffer({64, 64, 1}, false);
    Tensor<float, DenseLayout> tensor;

    tensor.AttachToBuffer(buffer.data, buffer.dims);

    SECTION("Default centering is false") {
        REQUIRE_FALSE(tensor.IsObjectCentered( ));
        REQUIRE_FALSE(tensor.IsFFTCentered( ));
    }

    SECTION("Set object centered") {
        tensor.SetObjectCentered(true);
        REQUIRE(tensor.IsObjectCentered( ));

        tensor.SetObjectCentered(false);
        REQUIRE_FALSE(tensor.IsObjectCentered( ));
    }

    SECTION("Set FFT centered") {
        tensor.SetFFTCentered(true);
        REQUIRE(tensor.IsFFTCentered( ));

        tensor.SetFFTCentered(false);
        REQUIRE_FALSE(tensor.IsFFTCentered( ));
    }

    pool.DeallocateBuffer(buffer);
}

TEST_CASE("Tensor: Copy semantics (shallow copy)", "[Tensor]") {
    TensorMemoryPool<float>    pool;
    auto                       buffer = pool.AllocateBuffer({64, 64, 1}, false);
    Tensor<float, DenseLayout> tensor1;

    tensor1.AttachToBuffer(buffer.data, buffer.dims);
    tensor1(0, 0, 0) = 42.0f;
    tensor1.SetSpace(Tensor<float>::Space::Momentum);

    SECTION("Copy constructor creates another view") {
        Tensor<float, DenseLayout> tensor2(tensor1);

        REQUIRE(tensor2.IsAttached( ));
        REQUIRE(tensor2.Data( ) == tensor1.Data( ));
        REQUIRE(tensor2.GetDims( ).x == tensor1.GetDims( ).x);
        REQUIRE(tensor2.GetSpace( ) == tensor1.GetSpace( ));

        // Both views see same data
        REQUIRE(tensor2(0, 0, 0) == 42.0f);

        tensor1(0, 0, 0) = 99.0f;
        REQUIRE(tensor2(0, 0, 0) == 99.0f);
    }

    SECTION("Assignment operator creates another view") {
        Tensor<float, DenseLayout> tensor2;
        tensor2 = tensor1;

        REQUIRE(tensor2.Data( ) == tensor1.Data( ));
        REQUIRE(tensor2(0, 0, 0) == 42.0f);
    }

    pool.DeallocateBuffer(buffer);
}

TEST_CASE("Tensor: Multiple views of same buffer", "[Tensor]") {
    TensorMemoryPool<float> pool;
    auto                    buffer = pool.AllocateBuffer({64, 64, 1}, false);

    Tensor<float, DenseLayout> view1, view2, view3;
    view1.AttachToBuffer(buffer.data, buffer.dims);
    view2.AttachToBuffer(buffer.data, buffer.dims);
    view3.AttachToBuffer(buffer.data, buffer.dims);

    SECTION("All views see same data") {
        view1(10, 20, 0) = 123.0f;

        REQUIRE(view2(10, 20, 0) == 123.0f);
        REQUIRE(view3(10, 20, 0) == 123.0f);
    }

    SECTION("Views can have independent metadata") {
        view1.SetSpace(Tensor<float>::Space::Position);
        view2.SetSpace(Tensor<float>::Space::Momentum);
        view3.SetSpace(Tensor<float>::Space::Position);

        REQUIRE(view1.IsInPositionSpace( ));
        REQUIRE(view2.IsInMomentumSpace( ));
        REQUIRE(view3.IsInPositionSpace( ));
    }

    pool.DeallocateBuffer(buffer);
}

TEST_CASE("Tensor: 3D tensor operations", "[Tensor]") {
    TensorMemoryPool<float>    pool;
    auto                       buffer = pool.AllocateBuffer({16, 16, 16}, false);
    Tensor<float, DenseLayout> tensor;

    tensor.AttachToBuffer(buffer.data, buffer.dims);
    tensor.SetSpace(Tensor<float>::Space::Position);

    SECTION("Access 3D elements") {
        tensor(5, 7, 9) = 42.0f;
        REQUIRE(tensor(5, 7, 9) == 42.0f);

        // Verify different slices are independent
        tensor(5, 7, 0) = 1.0f;
        tensor(5, 7, 1) = 2.0f;
        tensor(5, 7, 2) = 3.0f;

        REQUIRE(tensor(5, 7, 0) == 1.0f);
        REQUIRE(tensor(5, 7, 1) == 2.0f);
        REQUIRE(tensor(5, 7, 2) == 3.0f);
    }

    SECTION("3D size calculation") {
        REQUIRE(tensor.GetLogicalSize( ) == 16 * 16 * 16);
        REQUIRE(tensor.GetPhysicalSize( ) == 16 * 16 * 16);
    }

    pool.DeallocateBuffer(buffer);
}
