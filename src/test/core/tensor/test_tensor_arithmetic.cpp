/**
 * @file test_tensor_arithmetic.cpp
 * @brief Unit tests for Tensor arithmetic operations (Phase 2)
 *
 * Tests verify correctness using fixed known values.
 * Tests cover:
 * - Element-wise operations (Add, Subtract, Multiply, Divide)
 * - Scalar operations (AddScalar, MultiplyByScalar, DivideByScalar)
 * - Unary operations (Square, SquareRoot, Abs, Negate)
 * - Both DenseLayout and FFTWPaddedLayout
 * - 2D and 3D tensors
 */

#include "../../../core/core_headers.h"
#include "../../../core/tensor/core/tensor.h"
#include "../../../core/tensor/memory/tensor_memory_pool.h"
#include "../../../core/tensor/operations/arithmetic.h"
#include "../../../../include/catch2/catch.hpp"

using namespace cistem::tensor;

TEST_CASE("Tensor: AddScalar with known values", "[Tensor][Arithmetic]") {
    TensorMemoryPool<float> pool;
    auto                    buffer = pool.AllocateBuffer({4, 4, 1}, false);

    Tensor<float, DenseLayout> tensor;
    tensor.AttachToBuffer(buffer.data, {4, 4, 1});

    // Fill with sequence 0, 1, 2, ..., 15
    for ( int y = 0; y < 4; y++ ) {
        for ( int x = 0; x < 4; x++ ) {
            tensor.GetValue_p(x, y, 0) = float(y * 4 + x);
        }
    }

    AddScalar(tensor, 10.0f);

    // Verify all values increased by 10
    for ( int y = 0; y < 4; y++ ) {
        for ( int x = 0; x < 4; x++ ) {
            float expected = float(y * 4 + x) + 10.0f;
            REQUIRE(tensor.GetValue_p(x, y, 0) == Approx(expected));
        }
    }

    pool.DeallocateBuffer(buffer);
}

TEST_CASE("Tensor: MultiplyByScalar with known values", "[Tensor][Arithmetic]") {
    TensorMemoryPool<float> pool;
    auto                    buffer = pool.AllocateBuffer({4, 4, 1}, false);

    Tensor<float, DenseLayout> tensor;
    tensor.AttachToBuffer(buffer.data, {4, 4, 1});

    // Fill with sequence
    for ( int y = 0; y < 4; y++ ) {
        for ( int x = 0; x < 4; x++ ) {
            tensor.GetValue_p(x, y, 0) = float(x + 1); // 1, 2, 3, 4
        }
    }

    MultiplyByScalar(tensor, 3.0f);

    // Verify multiplication
    for ( int y = 0; y < 4; y++ ) {
        for ( int x = 0; x < 4; x++ ) {
            float expected = float(x + 1) * 3.0f;
            REQUIRE(tensor.GetValue_p(x, y, 0) == Approx(expected));
        }
    }

    pool.DeallocateBuffer(buffer);
}

TEST_CASE("Tensor: DivideByScalar with known values", "[Tensor][Arithmetic]") {
    TensorMemoryPool<float> pool;
    auto                    buffer = pool.AllocateBuffer({4, 4, 1}, false);

    Tensor<float, DenseLayout> tensor;
    tensor.AttachToBuffer(buffer.data, {4, 4, 1});

    // Fill with even numbers
    for ( int y = 0; y < 4; y++ ) {
        for ( int x = 0; x < 4; x++ ) {
            tensor.GetValue_p(x, y, 0) = float((x + 1) * 2); // 2, 4, 6, 8
        }
    }

    DivideByScalar(tensor, 2.0f);

    // Verify division
    for ( int y = 0; y < 4; y++ ) {
        for ( int x = 0; x < 4; x++ ) {
            float expected = float(x + 1); // 1, 2, 3, 4
            REQUIRE(tensor.GetValue_p(x, y, 0) == Approx(expected));
        }
    }

    pool.DeallocateBuffer(buffer);
}

TEST_CASE("Tensor: Add element-wise", "[Tensor][Arithmetic]") {
    TensorMemoryPool<float> pool;
    auto                    buffer1 = pool.AllocateBuffer({4, 4, 1}, false);
    auto                    buffer2 = pool.AllocateBuffer({4, 4, 1}, false);

    Tensor<float, DenseLayout> tensor1, tensor2;
    tensor1.AttachToBuffer(buffer1.data, {4, 4, 1});
    tensor2.AttachToBuffer(buffer2.data, {4, 4, 1});

    // Fill tensors
    for ( int y = 0; y < 4; y++ ) {
        for ( int x = 0; x < 4; x++ ) {
            tensor1.GetValue_p(x, y, 0) = float(y * 4 + x); // 0, 1, 2, ...
            tensor2.GetValue_p(x, y, 0) = float((y * 4 + x) * 2); // 0, 2, 4, ...
        }
    }

    Add(tensor1, tensor2);

    // Verify element-wise addition
    for ( int y = 0; y < 4; y++ ) {
        for ( int x = 0; x < 4; x++ ) {
            float expected = float(y * 4 + x) + float((y * 4 + x) * 2);
            REQUIRE(tensor1.GetValue_p(x, y, 0) == Approx(expected));
        }
    }

    pool.DeallocateBuffer(buffer1);
    pool.DeallocateBuffer(buffer2);
}

TEST_CASE("Tensor: Subtract element-wise", "[Tensor][Arithmetic]") {
    TensorMemoryPool<float> pool;
    auto                    buffer1 = pool.AllocateBuffer({4, 4, 1}, false);
    auto                    buffer2 = pool.AllocateBuffer({4, 4, 1}, false);

    Tensor<float, DenseLayout> tensor1, tensor2;
    tensor1.AttachToBuffer(buffer1.data, {4, 4, 1});
    tensor2.AttachToBuffer(buffer2.data, {4, 4, 1});

    // Fill tensors
    for ( int y = 0; y < 4; y++ ) {
        for ( int x = 0; x < 4; x++ ) {
            tensor1.GetValue_p(x, y, 0) = 100.0f;
            tensor2.GetValue_p(x, y, 0) = float(y * 4 + x);
        }
    }

    Subtract(tensor1, tensor2);

    // Verify subtraction
    for ( int y = 0; y < 4; y++ ) {
        for ( int x = 0; x < 4; x++ ) {
            float expected = 100.0f - float(y * 4 + x);
            REQUIRE(tensor1.GetValue_p(x, y, 0) == Approx(expected));
        }
    }

    pool.DeallocateBuffer(buffer1);
    pool.DeallocateBuffer(buffer2);
}

TEST_CASE("Tensor: MultiplyPixelWise", "[Tensor][Arithmetic]") {
    TensorMemoryPool<float> pool;
    auto                    buffer1 = pool.AllocateBuffer({4, 4, 1}, false);
    auto                    buffer2 = pool.AllocateBuffer({4, 4, 1}, false);

    Tensor<float, DenseLayout> tensor1, tensor2;
    tensor1.AttachToBuffer(buffer1.data, {4, 4, 1});
    tensor2.AttachToBuffer(buffer2.data, {4, 4, 1});

    // Fill tensors
    for ( int y = 0; y < 4; y++ ) {
        for ( int x = 0; x < 4; x++ ) {
            tensor1.GetValue_p(x, y, 0) = float(x + 1);
            tensor2.GetValue_p(x, y, 0) = 2.0f;
        }
    }

    MultiplyPixelWise(tensor1, tensor2);

    // Verify multiplication
    for ( int y = 0; y < 4; y++ ) {
        for ( int x = 0; x < 4; x++ ) {
            float expected = float(x + 1) * 2.0f;
            REQUIRE(tensor1.GetValue_p(x, y, 0) == Approx(expected));
        }
    }

    pool.DeallocateBuffer(buffer1);
    pool.DeallocateBuffer(buffer2);
}

TEST_CASE("Tensor: DividePixelWise", "[Tensor][Arithmetic]") {
    TensorMemoryPool<float> pool;
    auto                    buffer1 = pool.AllocateBuffer({4, 4, 1}, false);
    auto                    buffer2 = pool.AllocateBuffer({4, 4, 1}, false);

    Tensor<float, DenseLayout> tensor1, tensor2;
    tensor1.AttachToBuffer(buffer1.data, {4, 4, 1});
    tensor2.AttachToBuffer(buffer2.data, {4, 4, 1});

    // Fill tensors
    for ( int y = 0; y < 4; y++ ) {
        for ( int x = 0; x < 4; x++ ) {
            tensor1.GetValue_p(x, y, 0) = float((x + 1) * 100);
            tensor2.GetValue_p(x, y, 0) = 2.0f;
        }
    }

    DividePixelWise(tensor1, tensor2);

    // Verify division
    for ( int y = 0; y < 4; y++ ) {
        for ( int x = 0; x < 4; x++ ) {
            float expected = float((x + 1) * 100) / 2.0f;
            REQUIRE(tensor1.GetValue_p(x, y, 0) == Approx(expected));
        }
    }

    pool.DeallocateBuffer(buffer1);
    pool.DeallocateBuffer(buffer2);
}

TEST_CASE("Tensor: Square values", "[Tensor][Arithmetic]") {
    TensorMemoryPool<float> pool;
    auto                    buffer = pool.AllocateBuffer({4, 4, 1}, false);

    Tensor<float, DenseLayout> tensor;
    tensor.AttachToBuffer(buffer.data, {4, 4, 1});

    // Fill with simple values
    for ( int y = 0; y < 4; y++ ) {
        for ( int x = 0; x < 4; x++ ) {
            tensor.GetValue_p(x, y, 0) = 2.0f;
        }
    }

    Square(tensor);

    // Verify squaring
    for ( int y = 0; y < 4; y++ ) {
        for ( int x = 0; x < 4; x++ ) {
            REQUIRE(tensor.GetValue_p(x, y, 0) == Approx(4.0f));
        }
    }

    pool.DeallocateBuffer(buffer);
}

TEST_CASE("Tensor: SquareRoot values", "[Tensor][Arithmetic]") {
    TensorMemoryPool<float> pool;
    auto                    buffer = pool.AllocateBuffer({4, 4, 1}, false);

    Tensor<float, DenseLayout> tensor;
    tensor.AttachToBuffer(buffer.data, {4, 4, 1});

    // Fill with perfect squares
    for ( int y = 0; y < 4; y++ ) {
        for ( int x = 0; x < 4; x++ ) {
            tensor.GetValue_p(x, y, 0) = 16.0f;
        }
    }

    SquareRoot(tensor);

    // Verify square root
    for ( int y = 0; y < 4; y++ ) {
        for ( int x = 0; x < 4; x++ ) {
            REQUIRE(tensor.GetValue_p(x, y, 0) == Approx(4.0f));
        }
    }

    pool.DeallocateBuffer(buffer);
}

TEST_CASE("Tensor: Abs values", "[Tensor][Arithmetic]") {
    TensorMemoryPool<float> pool;
    auto                    buffer = pool.AllocateBuffer({4, 4, 1}, false);

    Tensor<float, DenseLayout> tensor;
    tensor.AttachToBuffer(buffer.data, {4, 4, 1});

    // Fill with mix of positive and negative
    for ( int y = 0; y < 4; y++ ) {
        for ( int x = 0; x < 4; x++ ) {
            tensor.GetValue_p(x, y, 0) = (x % 2 == 0) ? -5.0f : 5.0f;
        }
    }

    Abs(tensor);

    // Verify all positive
    for ( int y = 0; y < 4; y++ ) {
        for ( int x = 0; x < 4; x++ ) {
            REQUIRE(tensor.GetValue_p(x, y, 0) == Approx(5.0f));
        }
    }

    pool.DeallocateBuffer(buffer);
}

TEST_CASE("Tensor: Negate values", "[Tensor][Arithmetic]") {
    TensorMemoryPool<float> pool;
    auto                    buffer = pool.AllocateBuffer({4, 4, 1}, false);

    Tensor<float, DenseLayout> tensor;
    tensor.AttachToBuffer(buffer.data, {4, 4, 1});

    // Fill with values
    for ( int y = 0; y < 4; y++ ) {
        for ( int x = 0; x < 4; x++ ) {
            tensor.GetValue_p(x, y, 0) = float(x + 1);
        }
    }

    Negate(tensor);

    // Verify negation
    for ( int y = 0; y < 4; y++ ) {
        for ( int x = 0; x < 4; x++ ) {
            REQUIRE(tensor.GetValue_p(x, y, 0) == Approx(-float(x + 1)));
        }
    }

    pool.DeallocateBuffer(buffer);
}

TEST_CASE("Tensor: Chained arithmetic operations", "[Tensor][Arithmetic]") {
    TensorMemoryPool<float> pool;
    auto                    buffer = pool.AllocateBuffer({4, 4, 1}, false);

    Tensor<float, DenseLayout> tensor;
    tensor.AttachToBuffer(buffer.data, {4, 4, 1});

    // Fill with constant
    for ( int y = 0; y < 4; y++ ) {
        for ( int x = 0; x < 4; x++ ) {
            tensor.GetValue_p(x, y, 0) = 10.0f;
        }
    }

    // Chain operations: (10 + 5) * 2 / 3 = 30 / 3 = 10
    AddScalar(tensor, 5.0f);
    MultiplyByScalar(tensor, 2.0f);
    DivideByScalar(tensor, 3.0f);

    for ( int y = 0; y < 4; y++ ) {
        for ( int x = 0; x < 4; x++ ) {
            REQUIRE(tensor.GetValue_p(x, y, 0) == Approx(10.0f));
        }
    }

    pool.DeallocateBuffer(buffer);
}

TEST_CASE("Tensor: 3D arithmetic operations", "[Tensor][Arithmetic][3D]") {
    TensorMemoryPool<float> pool;
    auto                    buffer1 = pool.AllocateBuffer({8, 8, 8}, false);
    auto                    buffer2 = pool.AllocateBuffer({8, 8, 8}, false);

    Tensor<float, DenseLayout> tensor1, tensor2;
    tensor1.AttachToBuffer(buffer1.data, {8, 8, 8});
    tensor2.AttachToBuffer(buffer2.data, {8, 8, 8});

    // Fill volumes
    for ( int z = 0; z < 8; z++ ) {
        for ( int y = 0; y < 8; y++ ) {
            for ( int x = 0; x < 8; x++ ) {
                tensor1.GetValue_p(x, y, z) = float(x + y + z);
                tensor2.GetValue_p(x, y, z) = 1.0f;
            }
        }
    }

    Add(tensor1, tensor2);

    // Verify 3D addition
    for ( int z = 0; z < 8; z++ ) {
        for ( int y = 0; y < 8; y++ ) {
            for ( int x = 0; x < 8; x++ ) {
                float expected = float(x + y + z + 1);
                REQUIRE(tensor1.GetValue_p(x, y, z) == Approx(expected));
            }
        }
    }

    pool.DeallocateBuffer(buffer1);
    pool.DeallocateBuffer(buffer2);
}

TEST_CASE("Tensor: FFTWPaddedLayout arithmetic", "[Tensor][Arithmetic][Padding]") {
    SECTION("AddScalar with even dimensions (padding=2)") {
        TensorMemoryPool<float> pool;
        // For 64x64, FFTW padding adds 2 elements per row (64 is even)
        size_t padded_size = 66 * 64; // (64+2) * 64
        auto   buffer      = pool.AllocateBuffer({64, 64, 1}, false, padded_size);

        Tensor<float, FFTWPaddedLayout> tensor;
        tensor.AttachToBuffer(buffer.data, {64, 64, 1});

        // Fill logical region
        for ( int y = 0; y < 64; y++ ) {
            for ( int x = 0; x < 64; x++ ) {
                tensor.GetValue_p(x, y, 0) = 1.0f;
            }
        }

        AddScalar(tensor, 1.0f);

        // Verify
        for ( int y = 0; y < 64; y++ ) {
            for ( int x = 0; x < 64; x++ ) {
                REQUIRE(tensor.GetValue_p(x, y, 0) == Approx(2.0f));
            }
        }

        pool.DeallocateBuffer(buffer);
    }

    SECTION("MultiplyByScalar with odd dimensions (padding=1)") {
        TensorMemoryPool<float> pool;
        // For 63x64, FFTW padding adds 1 element per row (63 is odd)
        size_t padded_size = 64 * 64; // (63+1) * 64
        auto   buffer      = pool.AllocateBuffer({63, 64, 1}, false, padded_size);

        Tensor<float, FFTWPaddedLayout> tensor;
        tensor.AttachToBuffer(buffer.data, {63, 64, 1});

        // Fill logical region
        for ( int y = 0; y < 64; y++ ) {
            for ( int x = 0; x < 63; x++ ) {
                tensor.GetValue_p(x, y, 0) = 2.0f;
            }
        }

        MultiplyByScalar(tensor, 3.0f);

        // Verify
        for ( int y = 0; y < 64; y++ ) {
            for ( int x = 0; x < 63; x++ ) {
                REQUIRE(tensor.GetValue_p(x, y, 0) == Approx(6.0f));
            }
        }

        pool.DeallocateBuffer(buffer);
    }
}
