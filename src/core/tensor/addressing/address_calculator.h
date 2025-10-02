#ifndef _SRC_CORE_TENSOR_ADDRESSING_ADDRESS_CALCULATOR_H_
#define _SRC_CORE_TENSOR_ADDRESSING_ADDRESS_CALCULATOR_H_

/**
 * @file address_calculator.h
 * @brief Address calculation for cisTEM Tensor system
 *
 * Header-only, template-based address calculation for zero-overhead indexing.
 * Templated on Layout policy to enable compile-time selection of addressing mode.
 *
 * PERFORMANCE CRITICAL: This code will be called millions of times.
 * All functions must be inline and suitable for compiler optimization.
 *
 * Usage:
 * @code
 * long addr = AddressCalculator<FFTWPaddedLayout>::Real1DAddress(x, y, z, dims);
 * ScalarType& value = data[addr];
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
 * @tparam Layout Memory layout policy (DenseLayout, FFTWPaddedLayout, etc.)
 */
template <typename Layout>
class AddressCalculator {
  public:
    /**
     * @brief Calculate 1D address from physical 3D coordinates in real space
     *
     * Physical coordinates are the actual array indices:
     * - x: [0, dims.x)
     * - y: [0, dims.y)
     * - z: [0, dims.z)
     *
     * @param x Physical X coordinate
     * @param y Physical Y coordinate
     * @param z Physical Z coordinate
     * @param dims Logical dimensions
     * @return 1D array index
     */
    static inline long Real1DAddress(int x, int y, int z, int3 dims) {
        size_t pitch = Layout::CalculatePitch(dims);
        return long(pitch * dims.y) * long(z) + long(pitch) * long(y) + long(x);
    }

    /**
     * @brief Calculate 1D address from physical 3D coordinates in Fourier space
     *
     * Fourier space has different dimensions due to Hermitian symmetry:
     * - X dimension: [0, dims.x/2 + 1) for real-to-complex transforms
     * - Y, Z dimensions: same as real space
     *
     * @param x Physical X coordinate (Fourier)
     * @param y Physical Y coordinate (Fourier)
     * @param z Physical Z coordinate (Fourier)
     * @param dims Logical dimensions (real space)
     * @return 1D array index (complex-valued element)
     */
    static inline long Fourier1DAddress(int x, int y, int z, int3 dims) {
        // Complex pitch: Hermitian symmetry means we only store half + 1 in X
        long complex_pitch = dims.x / 2 + 1;
        return complex_pitch * dims.y * long(z) + complex_pitch * long(y) + long(x);
    }

    /**
     * @brief Calculate pitch (stride) for this layout
     *
     * @param dims Logical dimensions
     * @return Pitch in elements
     */
    static inline size_t GetPitch(int3 dims) {
        return Layout::CalculatePitch(dims);
    }

    /**
     * @brief Calculate total memory size for this layout
     *
     * @param dims Logical dimensions
     * @return Total elements needed (including any padding)
     */
    static inline size_t GetMemorySize(int3 dims) {
        return Layout::CalculateMemorySize(dims);
    }
};

// ============================================================================
// Specialized addressing for common layouts
// ============================================================================

/**
 * @brief Fast path for dense layout addressing (no padding)
 *
 * Explicitly specialized for DenseLayout to enable maximum compiler optimization.
 * Compilers can better optimize when they know there's no padding calculation.
 */
template <>
inline long AddressCalculator<DenseLayout>::Real1DAddress(int x, int y, int z, int3 dims) {
    // Dense layout: pitch == dims.x (no padding)
    return long(dims.x) * long(dims.y) * long(z) + long(dims.x) * long(y) + long(x);
}

/**
 * @brief Fast path for FFTW padded layout addressing
 *
 * Explicitly calculates padding to enable compiler optimization.
 * Modern compilers can often eliminate the modulo check at compile time
 * when dimensions are known constants.
 */
template <>
inline long AddressCalculator<FFTWPaddedLayout>::Real1DAddress(int x, int y, int z, int3 dims) {
    // FFTW padding: pitch = dims.x + (2 if even, 1 if odd)
    long pitch = long(dims.x) + ((dims.x % 2 == 0) ? 2 : 1);
    return pitch * long(dims.y) * long(z) + pitch * long(y) + long(x);
}

// ============================================================================
// Helper functions for common operations
// ============================================================================

/**
 * @brief Check if coordinates are within physical bounds (real space)
 *
 * @param x X coordinate
 * @param y Y coordinate
 * @param z Z coordinate
 * @param dims Dimensions
 * @return true if coordinates are valid
 */
inline bool IsWithinBounds(int x, int y, int z, int3 dims) {
    return (x >= 0 && x < dims.x &&
            y >= 0 && y < dims.y &&
            z >= 0 && z < dims.z);
}

/**
 * @brief Check if coordinates are within Fourier space physical bounds
 *
 * Fourier space only stores half the X dimension due to Hermitian symmetry
 *
 * @param x X coordinate (Fourier)
 * @param y Y coordinate (Fourier)
 * @param z Z coordinate (Fourier)
 * @param dims Logical dimensions (real space)
 * @return true if coordinates are valid in Fourier space
 */
inline bool IsWithinFourierBounds(int x, int y, int z, int3 dims) {
    return (x >= 0 && x <= dims.x / 2 &&
            y >= 0 && y < dims.y &&
            z >= 0 && z < dims.z);
}

} // namespace tensor
} // namespace cistem

#endif // _SRC_CORE_TENSOR_ADDRESSING_ADDRESS_CALCULATOR_H_
