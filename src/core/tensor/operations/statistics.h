#ifndef _SRC_CORE_TENSOR_OPERATIONS_STATISTICS_H_
#define _SRC_CORE_TENSOR_OPERATIONS_STATISTICS_H_

/**
 * @file statistics.h
 * @brief Statistical operations for Tensor class (Phase 2.5 - Iterator-based)
 *
 * Provides statistical analysis and normalization operations for Tensors.
 * All operations are free functions that operate on Tensor views.
 *
 * Phase 2.5 improvements:
 * - Iterator-based implementation (no address recalculation)
 * - Automatic padding handling
 * - Performance matches Legacy Image
 *
 * Operations:
 * - Min, Max: Find minimum and maximum values
 * - Sum, Mean: Calculate sum and average
 * - Variance, StandardDeviation: Measure spread
 * - SumOfSquares: Calculate mean of sum of squares (matches Legacy Image behavior)
 * - SetConstant: Fill tensor with constant value
 */

#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>
#include "../core/tensor.h"
#include "../type_traits.h"
#include "../debug_utils.h"

namespace cistem {
namespace tensor {

/**
 * @brief Find minimum and maximum values in tensor
 *
 * @param tensor Tensor to analyze
 * @param min_value Output: minimum value found
 * @param max_value Output: maximum value found
 */
template <typename Scalar_t, typename Layout_t>
inline void GetMinMax(const Tensor<Scalar_t, Layout_t>& tensor,
                      Scalar_t&                         min_value,
                      Scalar_t&                         max_value) {
    TENSOR_DEBUG_ASSERT(tensor.IsAttached( ), "Tensor must be attached");

    auto [min_it, max_it] = std::minmax_element(tensor.begin( ), tensor.end( ));
    min_value             = *min_it;
    max_value             = *max_it;
}

/**
 * @brief Calculate mean (average) of all tensor elements
 *
 * @param tensor Tensor to analyze
 * @return double Mean value (uses double precision for accumulation)
 */
template <typename Scalar_t, typename Layout_t>
inline double Mean(const Tensor<Scalar_t, Layout_t>& tensor) {
    TENSOR_DEBUG_ASSERT(tensor.IsAttached( ), "Tensor must be attached");

    const int3 dims         = tensor.GetDims( );
    long       num_elements = long(dims.x) * long(dims.y) * long(dims.z);

    double sum = std::accumulate(tensor.begin( ), tensor.end( ), 0.0,
                                 [](double acc, Scalar_t val) { return acc + double(val); });

    return sum / double(num_elements);
}

/**
 * @brief Calculate sum of all tensor elements
 *
 * @param tensor Tensor to analyze
 * @return double Sum of all elements (uses double precision for accumulation)
 */
template <typename Scalar_t, typename Layout_t>
inline double Sum(const Tensor<Scalar_t, Layout_t>& tensor) {
    TENSOR_DEBUG_ASSERT(tensor.IsAttached( ), "Tensor must be attached");

    return std::accumulate(tensor.begin( ), tensor.end( ), 0.0,
                           [](double acc, Scalar_t val) { return acc + double(val); });
}

/**
 * @brief Calculate variance of all tensor elements
 *
 * Uses the formula: Var(X) = E[(X - mean)^2]
 *
 * @param tensor Tensor to analyze
 * @return Scalar_t Variance (sample variance, not population variance)
 */
template <typename Scalar_t, typename Layout_t>
inline Scalar_t Variance(const Tensor<Scalar_t, Layout_t>& tensor) {
    TENSOR_DEBUG_ASSERT(tensor.IsAttached( ), "Tensor must be attached");

    const int3 dims         = tensor.GetDims( );
    long       num_elements = long(dims.x) * long(dims.y) * long(dims.z);

    // Calculate mean first
    double mean = Mean(tensor);

    // Calculate variance using std::accumulate
    double sum_squared_deviations = std::accumulate(
            tensor.begin( ), tensor.end( ), 0.0,
            [mean](double acc, Scalar_t val) {
                double deviation = double(val) - mean;
                return acc + deviation * deviation;
            });

    return Scalar_t(sum_squared_deviations / double(num_elements));
}

/**
 * @brief Calculate standard deviation of all tensor elements
 *
 * @param tensor Tensor to analyze
 * @return Scalar_t Standard deviation (square root of variance)
 */
template <typename Scalar_t, typename Layout_t>
inline Scalar_t StandardDeviation(const Tensor<Scalar_t, Layout_t>& tensor) {
    return std::sqrt(Variance(tensor));
}

/**
 * @brief Calculate mean of sum of squares
 *
 * Note: This returns the MEAN of sum of squares, not the total sum.
 * This matches Legacy Image ReturnSumOfSquares() behavior.
 *
 * @param tensor Tensor to analyze
 * @return double Mean of squared values
 */
template <typename Scalar_t, typename Layout_t>
inline double SumOfSquares(const Tensor<Scalar_t, Layout_t>& tensor) {
    TENSOR_DEBUG_ASSERT(tensor.IsAttached( ), "Tensor must be attached");

    const int3 dims         = tensor.GetDims( );
    long       num_elements = long(dims.x) * long(dims.y) * long(dims.z);

    double sum_of_squares = std::accumulate(
            tensor.begin( ), tensor.end( ), 0.0,
            [](double acc, Scalar_t val) {
                double v = double(val);
                return acc + v * v;
            });

    return sum_of_squares / double(num_elements);
}

/**
 * @brief Set all tensor elements to a constant value
 *
 * @param tensor Tensor to modify
 * @param value Constant value to set
 */
template <typename Scalar_t, typename Layout_t>
inline void SetConstant(Tensor<Scalar_t, Layout_t>& tensor, Scalar_t value) {
    TENSOR_DEBUG_ASSERT(tensor.IsAttached( ), "Tensor must be attached");

    std::fill(tensor.begin( ), tensor.end( ), value);
}

} // namespace tensor
} // namespace cistem

#endif // _SRC_CORE_TENSOR_OPERATIONS_STATISTICS_H_
