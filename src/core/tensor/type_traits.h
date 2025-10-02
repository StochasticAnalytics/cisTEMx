#ifndef _SRC_CORE_TENSOR_TYPE_TRAITS_H_
#define _SRC_CORE_TENSOR_TYPE_TRAITS_H_

/**
 * @file type_traits.h
 * @brief Type traits for cisTEM Tensor system
 *
 * Provides compile-time type checking and constraints for Tensor scalar types.
 * Uses cistem::EnableIf (from constants.h) for SFINAE-based template constraints.
 *
 * Phase 1: Only float scalar type supported
 * Phase 5: Will add complex<float>, __half, __nv_bfloat16
 */

#include <type_traits>
#include "complex_types.h"
#include "../../constants/constants.h"

namespace cistem {
namespace tensor {

// ============================================================================
// Real Scalar Type Traits
// ============================================================================

/**
 * @brief Trait to identify real-valued scalar types
 *
 * Phase 1: float, double
 * Phase 5: Will add __half, __nv_bfloat16
 */
template <typename T>
struct is_real_scalar : std::bool_constant<
                                std::is_same_v<T, float> ||
                                std::is_same_v<T, double>
                                // Phase 5: Add __half, __nv_bfloat16
                                > {};

template <typename T>
inline constexpr bool is_real_scalar_v = is_real_scalar<T>::value;

// ============================================================================
// Complex Scalar Type Traits
// ============================================================================

/**
 * @brief Trait to identify complex scalar types
 *
 * Matches cistem::tensor::complex<T> for any T
 */
template <typename T>
struct is_complex_scalar : std::false_type {};

template <typename T>
struct is_complex_scalar<complex<T>> : std::true_type {};

template <typename T>
inline constexpr bool is_complex_scalar_v = is_complex_scalar<T>::value;

// ============================================================================
// Numeric Type Traits
// ============================================================================

/**
 * @brief Trait to identify any numeric type (real or complex)
 */
template <typename T>
struct is_numeric : std::bool_constant<
                            is_real_scalar_v<T> || is_complex_scalar_v<T>> {};

template <typename T>
inline constexpr bool is_numeric_v = is_numeric<T>::value;

// ============================================================================
// Phase Constraints
// ============================================================================

/**
 * @brief Phase 1 constraint: Only float scalar type supported
 *
 * Use this trait to constrain templates during Phase 1 implementation.
 *
 * Example usage:
 * @code
 * template<typename T, cistem::EnableIf<is_phase1_supported_v<T>, int> = 0>
 * class Tensor {
 *     // Only instantiable with float in Phase 1
 * };
 * @endcode
 */
template <typename T>
inline constexpr bool is_phase1_supported_v = std::is_same_v<T, float>;

/**
 * @brief Phase 5 constraint: All numeric types supported
 *
 * Will include float, double, complex<float>, complex<double>,
 * __half, complex<__half>, __nv_bfloat16, complex<__nv_bfloat16>
 */
template <typename T>
inline constexpr bool is_phase5_supported_v = is_numeric_v<T>;

// Phase 5: Expand to include all GPU types

// ============================================================================
// Utility Traits
// ============================================================================

/**
 * @brief Extract underlying scalar type from complex type
 *
 * For complex<T>, returns T
 * For real types, returns T itself
 */
template <typename T>
struct scalar_value_type {
    using type = T;
};

template <typename T>
struct scalar_value_type<complex<T>> {
    using type = T;
};

template <typename T>
using scalar_value_type_t = typename scalar_value_type<T>::type;

/**
 * @brief Check if type supports default arithmetic operators
 *
 * Used internally for operator overload selection
 */
template <typename T>
struct supports_default_ops : std::bool_constant<
                                      std::is_same_v<T, float> ||
                                      std::is_same_v<T, double>
                                      // Phase 5: Add GPU types as needed
                                      > {};

template <typename T>
inline constexpr bool supports_default_ops_v = supports_default_ops<T>::value;

} // namespace tensor
} // namespace cistem

#endif // _SRC_CORE_TENSOR_TYPE_TRAITS_H_
