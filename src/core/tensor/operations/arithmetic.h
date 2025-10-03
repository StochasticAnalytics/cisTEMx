#ifndef _SRC_CORE_TENSOR_OPERATIONS_ARITHMETIC_H_
#define _SRC_CORE_TENSOR_OPERATIONS_ARITHMETIC_H_

/**
 * @file arithmetic.h
 * @brief Arithmetic operations for Tensor class (Phase 2.5 - Iterator-based)
 *
 * Provides element-wise and scalar arithmetic operations for Tensors.
 * All operations are free functions that operate on Tensor views.
 *
 * Phase 2.5 improvements:
 * - Iterator-based implementation (no address recalculation)
 * - Automatic padding handling
 * - Performance matches Legacy Image
 *
 * Operations:
 * - Element-wise: Add, Subtract, Multiply, Divide
 * - Scalar: AddScalar, MultiplyByScalar, etc.
 * - Unary: Square, SquareRoot, Abs, Negate
 */

#include <cmath>
#include <algorithm>
#include "../core/tensor.h"
#include "../type_traits.h"
#include "../debug_utils.h"

namespace cistem {
namespace tensor {

/**
 * @brief Add a scalar value to all elements in-place
 *
 * @param tensor Tensor to modify
 * @param value Scalar value to add
 */
template <typename Scalar_t, typename Layout_t>
inline void AddScalar(Tensor<Scalar_t, Layout_t>& tensor, Scalar_t value) {
    TENSOR_DEBUG_ASSERT(tensor.IsAttached( ), "Tensor must be attached");

    std::for_each(tensor.begin( ), tensor.end( ), [value](Scalar_t& val) { val += value; });
}

/**
 * @brief Multiply all elements by a scalar value in-place
 *
 * @param tensor Tensor to modify
 * @param value Scalar value to multiply by
 */
template <typename Scalar_t, typename Layout_t>
inline void MultiplyByScalar(Tensor<Scalar_t, Layout_t>& tensor, Scalar_t value) {
    TENSOR_DEBUG_ASSERT(tensor.IsAttached( ), "Tensor must be attached");

    std::for_each(tensor.begin( ), tensor.end( ), [value](Scalar_t& val) { val *= value; });
}

/**
 * @brief Divide all elements by a scalar value in-place
 *
 * @param tensor Tensor to modify
 * @param value Scalar value to divide by (must be non-zero)
 */
template <typename Scalar_t, typename Layout_t>
inline void DivideByScalar(Tensor<Scalar_t, Layout_t>& tensor, Scalar_t value) {
    TENSOR_DEBUG_ASSERT(tensor.IsAttached( ), "Tensor must be attached");
    TENSOR_DEBUG_ASSERT(value != Scalar_t(0), "Division by zero");

    std::for_each(tensor.begin( ), tensor.end( ), [value](Scalar_t& val) { val /= value; });
}

/**
 * @brief Add another tensor element-wise (in-place)
 *
 * Performs: tensor1 += tensor2
 *
 * @param tensor1 Destination tensor (modified)
 * @param tensor2 Source tensor (read-only)
 */
template <typename Scalar_t, typename Layout_t>
inline void Add(Tensor<Scalar_t, Layout_t>&       tensor1,
                const Tensor<Scalar_t, Layout_t>& tensor2) {
    TENSOR_DEBUG_ASSERT(tensor1.IsAttached( ), "Tensor1 must be attached");
    TENSOR_DEBUG_ASSERT(tensor2.IsAttached( ), "Tensor2 must be attached");

    const int3 dims1 = tensor1.GetDims( );
    const int3 dims2 = tensor2.GetDims( );

    TENSOR_DEBUG_ASSERT(dims1.x == dims2.x && dims1.y == dims2.y && dims1.z == dims2.z,
                        "Tensor dimensions must match");

    // Use STL transform for optimal performance
    std::transform(tensor1.begin( ), tensor1.end( ), tensor2.begin( ), tensor1.begin( ),
                   [](Scalar_t a, Scalar_t b) { return a + b; });
}

/**
 * @brief Subtract another tensor element-wise (in-place)
 *
 * Performs: tensor1 -= tensor2
 *
 * @param tensor1 Destination tensor (modified)
 * @param tensor2 Source tensor (read-only)
 */
template <typename Scalar_t, typename Layout_t>
inline void Subtract(Tensor<Scalar_t, Layout_t>&       tensor1,
                     const Tensor<Scalar_t, Layout_t>& tensor2) {
    TENSOR_DEBUG_ASSERT(tensor1.IsAttached( ), "Tensor1 must be attached");
    TENSOR_DEBUG_ASSERT(tensor2.IsAttached( ), "Tensor2 must be attached");

    const int3 dims1 = tensor1.GetDims( );
    const int3 dims2 = tensor2.GetDims( );

    TENSOR_DEBUG_ASSERT(dims1.x == dims2.x && dims1.y == dims2.y && dims1.z == dims2.z,
                        "Tensor dimensions must match");

    std::transform(tensor1.begin( ), tensor1.end( ), tensor2.begin( ), tensor1.begin( ),
                   [](Scalar_t a, Scalar_t b) { return a - b; });
}

/**
 * @brief Multiply by another tensor element-wise (in-place)
 *
 * Performs: tensor1 *= tensor2
 *
 * @param tensor1 Destination tensor (modified)
 * @param tensor2 Source tensor (read-only)
 */
template <typename Scalar_t, typename Layout_t>
inline void MultiplyPixelWise(Tensor<Scalar_t, Layout_t>&       tensor1,
                              const Tensor<Scalar_t, Layout_t>& tensor2) {
    TENSOR_DEBUG_ASSERT(tensor1.IsAttached( ), "Tensor1 must be attached");
    TENSOR_DEBUG_ASSERT(tensor2.IsAttached( ), "Tensor2 must be attached");

    const int3 dims1 = tensor1.GetDims( );
    const int3 dims2 = tensor2.GetDims( );

    TENSOR_DEBUG_ASSERT(dims1.x == dims2.x && dims1.y == dims2.y && dims1.z == dims2.z,
                        "Tensor dimensions must match");

    std::transform(tensor1.begin( ), tensor1.end( ), tensor2.begin( ), tensor1.begin( ),
                   [](Scalar_t a, Scalar_t b) { return a * b; });
}

/**
 * @brief Divide by another tensor element-wise (in-place)
 *
 * Performs: tensor1 /= tensor2
 *
 * @param tensor1 Destination tensor (modified)
 * @param tensor2 Source tensor (read-only, must not contain zeros)
 */
template <typename Scalar_t, typename Layout_t>
inline void DividePixelWise(Tensor<Scalar_t, Layout_t>&       tensor1,
                            const Tensor<Scalar_t, Layout_t>& tensor2) {
    TENSOR_DEBUG_ASSERT(tensor1.IsAttached( ), "Tensor1 must be attached");
    TENSOR_DEBUG_ASSERT(tensor2.IsAttached( ), "Tensor2 must be attached");

    const int3 dims1 = tensor1.GetDims( );
    const int3 dims2 = tensor2.GetDims( );

    TENSOR_DEBUG_ASSERT(dims1.x == dims2.x && dims1.y == dims2.y && dims1.z == dims2.z,
                        "Tensor dimensions must match");

    std::transform(tensor1.begin( ), tensor1.end( ), tensor2.begin( ), tensor1.begin( ),
                   [](Scalar_t a, Scalar_t b) { return a / b; });
}

/**
 * @brief Square all elements in-place
 *
 * Performs: tensor(i) = tensor(i)^2
 *
 * @param tensor Tensor to modify
 */
template <typename Scalar_t, typename Layout_t>
inline void Square(Tensor<Scalar_t, Layout_t>& tensor) {
    TENSOR_DEBUG_ASSERT(tensor.IsAttached( ), "Tensor must be attached");

    std::for_each(tensor.begin( ), tensor.end( ), [](Scalar_t& val) { val = val * val; });
}

/**
 * @brief Take square root of all elements in-place
 *
 * Performs: tensor(i) = sqrt(tensor(i))
 *
 * @param tensor Tensor to modify (must contain non-negative values)
 */
template <typename Scalar_t, typename Layout_t>
inline void SquareRoot(Tensor<Scalar_t, Layout_t>& tensor) {
    TENSOR_DEBUG_ASSERT(tensor.IsAttached( ), "Tensor must be attached");

    std::for_each(tensor.begin( ), tensor.end( ), [](Scalar_t& val) { val = std::sqrt(val); });
}

/**
 * @brief Take absolute value of all elements in-place
 *
 * Performs: tensor(i) = |tensor(i)|
 *
 * @param tensor Tensor to modify
 */
template <typename Scalar_t, typename Layout_t>
inline void Abs(Tensor<Scalar_t, Layout_t>& tensor) {
    TENSOR_DEBUG_ASSERT(tensor.IsAttached( ), "Tensor must be attached");

    std::for_each(tensor.begin( ), tensor.end( ), [](Scalar_t& val) { val = std::abs(val); });
}

/**
 * @brief Negate all elements in-place
 *
 * Performs: tensor(i) = -tensor(i)
 *
 * @param tensor Tensor to modify
 */
template <typename Scalar_t, typename Layout_t>
inline void Negate(Tensor<Scalar_t, Layout_t>& tensor) {
    TENSOR_DEBUG_ASSERT(tensor.IsAttached( ), "Tensor must be attached");

    std::for_each(tensor.begin( ), tensor.end( ), [](Scalar_t& val) { val = -val; });
}

} // namespace tensor
} // namespace cistem

#endif // _SRC_CORE_TENSOR_OPERATIONS_ARITHMETIC_H_
