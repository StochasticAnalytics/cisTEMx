#ifndef _SRC_CORE_TENSOR_ADDRESSING_ADDRESS_CALCULATOR_H_
#define _SRC_CORE_TENSOR_ADDRESSING_ADDRESS_CALCULATOR_H_

/**
 * @file address_calculator.h
 * @brief Address calculation for cisTEM Tensor system
 *
 * Header-only, template-based address calculation for zero-overhead indexing.
 * Templated on Layout policy to enable compile-time selection of addressing mode.
 *
 * Uses Position/Momentum nomenclature to match Tensor::Space enum and avoid
 * confusion between "real space" (position) and "real type" (float).
 *
 * PERFORMANCE CRITICAL: This code will be called millions of times.
 * All functions must be inline and suitable for compiler optimization.
 *
 * Usage:
 * @code
 * // Position space addressing
 * long addr = AddressCalculator<FFTWPaddedLayout>::PositionSpaceAddress(x, y, z, dims);
 * Scalar_t& value = data[addr];
 *
 * // Momentum space addressing (after FFT)
 * long addr_fft = AddressCalculator<FFTWPaddedLayout>::MomentumSpaceAddress(x, y, z, dims);
 * complex<float>& fft_value = reinterpret_cast<complex<float>*>(data)[addr_fft];
 * @endcode
 */

#include <cuda_runtime.h>
#include "../memory/memory_layout.h"

namespace cistem {
namespace tensor {

/**
 * @brief Calculate 1D memory addresses from 3D coordinates
 *
 * Templated on Layout policy for compile-time address calculation selection.
 * All methods are inline and constexpr where possible for maximum performance.
 *
 * Uses Position/Momentum nomenclature to match Tensor::Space enum.
 * This avoids confusion between "real space" (position) and "real type" (float).
 *
 * @tparam Layout_t Memory layout policy (DenseLayout, FFTWPaddedLayout, etc.)
 */
template <typename Layout_t>
class AddressCalculator {
  public:
    /**
     * @brief Calculate 1D address from physical 3D coordinates in position space
     *
     * Physical coordinates are the actual array indices:
     * - x: [0, dims.x)
     * - y: [0, dims.y)
     * - z: [0, dims.z)
     *
     * @tparam IndexType Return type (long by default for safety, int for GPU performance when dims are small)
     * @param x Physical X coordinate
     * @param y Physical Y coordinate
     * @param z Physical Z coordinate
     * @param dims Logical dimensions
     * @return 1D array index
     *
     * @note For GPU kernels with known small dimensions, use PositionSpaceAddress<int>() for faster 32-bit arithmetic
     */
    template <typename IndexType = long>
    static inline IndexType PositionSpaceAddress(int x, int y, int z, int3 dims) {
        size_t pitch = Layout_t::CalculatePitch(dims);
        if constexpr ( std::is_same_v<IndexType, long> ) {
            return long(pitch * dims.y) * long(z) + long(pitch) * long(y) + long(x);
        }
        else {
            return pitch * dims.y * z + pitch * y + x;
        }
    }

    /**
     * @brief Calculate 1D address from physical 3D coordinates in momentum space
     *
     * Momentum space has different dimensions due to Hermitian symmetry:
     * - X dimension: [0, dims.x/2 + 1) for real-to-complex transforms
     * - Y, Z dimensions: same as position space
     *
     * @tparam IndexType Return type (long by default for safety, int for GPU performance when dims are small)
     * @param x Physical X coordinate (momentum)
     * @param y Physical Y coordinate (momentum)
     * @param z Physical Z coordinate (momentum)
     * @param dims Logical dimensions (position space)
     * @return 1D array index (complex-valued element)
     *
     * @note For GPU kernels with known small dimensions, use MomentumSpaceAddress<int>() for faster 32-bit arithmetic
     */
    template <typename IndexType = long>
    static inline IndexType MomentumSpaceAddress(int x, int y, int z, int3 dims) {
        // Complex pitch: Hermitian symmetry means we only store half + 1 in X
        if constexpr ( std::is_same_v<IndexType, long> ) {
            long complex_pitch = dims.x / 2 + 1;
            return complex_pitch * dims.y * long(z) + complex_pitch * long(y) + long(x);
        }
        else {
            int complex_pitch = dims.x / 2 + 1;
            return complex_pitch * dims.y * z + complex_pitch * y + x;
        }
    }

    // ========================================================================
    // Utility methods
    // ========================================================================

    /**
     * @brief Calculate pitch (stride) for this layout
     *
     * @param dims Logical dimensions
     * @return Pitch in elements
     */
    static inline size_t GetPitch(int3 dims) {
        return Layout_t::CalculatePitch(dims);
    }

    /**
     * @brief Calculate total memory size for this layout
     *
     * @param dims Logical dimensions
     * @return Total elements needed (including any padding)
     */
    static inline size_t GetMemorySize(int3 dims) {
        return Layout_t::CalculateMemorySize(dims);
    }
};

// ============================================================================
// Specialized addressing for common layouts
// ============================================================================

/**
 * @brief Fast path for dense layout addressing (no padding) - Position space
 *
 * Explicitly specialized for DenseLayout to enable maximum compiler optimization.
 * Compilers can better optimize when they know there's no padding calculation.
 */
template <>
template <typename IndexType>
inline IndexType AddressCalculator<DenseLayout>::PositionSpaceAddress(int x, int y, int z, int3 dims) {
    // Dense layout: pitch == dims.x (no padding)
    if constexpr ( std::is_same_v<IndexType, long> ) {
        return long(dims.x) * long(dims.y) * long(z) + long(dims.x) * long(y) + long(x);
    }
    else {
        return dims.x * dims.y * z + dims.x * y + x;
    }
}

/**
 * @brief Fast path for FFTW padded layout addressing - Position space
 *
 * Explicitly calculates padding to enable compiler optimization.
 * Modern compilers can often eliminate the modulo check at compile time
 * when dimensions are known constants.
 */
template <>
template <typename IndexType>
inline IndexType AddressCalculator<FFTWPaddedLayout>::PositionSpaceAddress(int x, int y, int z, int3 dims) {
    // FFTW padding: pitch = dims.x + (2 if even, 1 if odd)
    if constexpr ( std::is_same_v<IndexType, long> ) {
        long pitch = long(dims.x) + ((dims.x % 2 == 0) ? 2 : 1);
        return pitch * long(dims.y) * long(z) + pitch * long(y) + long(x);
    }
    else {
        int pitch = dims.x + ((dims.x % 2 == 0) ? 2 : 1);
        return pitch * dims.y * z + pitch * y + x;
    }
}

// ============================================================================
// Helper functions for common operations
// ============================================================================

/**
 * @brief Check if coordinates are within physical bounds (position space)
 *
 * @param x X coordinate
 * @param y Y coordinate
 * @param z Z coordinate
 * @param dims Dimensions
 * @return true if coordinates are valid
 */
inline bool IsWithinPositionBounds(int x, int y, int z, int3 dims) {
    return (x >= 0 && x < dims.x &&
            y >= 0 && y < dims.y &&
            z >= 0 && z < dims.z);
}

/**
 * @brief Check if coordinates are within momentum space physical bounds
 *
 * Momentum space only stores half the X dimension due to Hermitian symmetry
 *
 * @param x X coordinate (momentum)
 * @param y Y coordinate (momentum)
 * @param z Z coordinate (momentum)
 * @param dims Logical dimensions (position space)
 * @return true if coordinates are valid in momentum space
 */
inline bool IsWithinMomentumBounds(int x, int y, int z, int3 dims) {
    return (x >= 0 && x <= dims.x / 2 &&
            y >= 0 && y < dims.y &&
            z >= 0 && z < dims.z);
}

// ============================================================================
// Legacy aliases for Image class compatibility
// ============================================================================

/**
 * @brief Legacy alias for IsWithinPositionBounds
 * @deprecated Use IsWithinPositionBounds instead
 */
inline bool IsWithinBounds(int x, int y, int z, int3 dims) {
    return IsWithinPositionBounds(x, y, z, dims);
}

/**
 * @brief Legacy alias for IsWithinMomentumBounds
 * @deprecated Use IsWithinMomentumBounds instead
 */
inline bool IsWithinFourierBounds(int x, int y, int z, int3 dims) {
    return IsWithinMomentumBounds(x, y, z, dims);
}

} // namespace tensor
} // namespace cistem

#endif // _SRC_CORE_TENSOR_ADDRESSING_ADDRESS_CALCULATOR_H_
