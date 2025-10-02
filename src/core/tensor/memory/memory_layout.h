#ifndef _SRC_CORE_TENSOR_MEMORY_MEMORY_LAYOUT_H_
#define _SRC_CORE_TENSOR_MEMORY_MEMORY_LAYOUT_H_

/**
 * @file memory_layout.h
 * @brief Memory layout policies for cisTEM Tensor system
 *
 * Provides compile-time layout policies that control memory organization
 * and pitch calculation for different use cases:
 *
 * - DenseLayout: Contiguous memory, no padding (out-of-place FFTs)
 * - FFTWPaddedLayout: FFTW in-place R2C padding (legacy Image compatibility)
 *
 * Layout policies are template parameters to Tensor and AddressCalculator,
 * enabling zero-overhead compile-time selection of address calculation.
 */

#include <cstddef>
#include <cuda_runtime.h> // For int3, int4, etc.

namespace cistem {
namespace tensor {

// ============================================================================
// DenseLayout - Contiguous memory with no padding
// ============================================================================

/**
 * @brief Dense (contiguous) memory layout with no padding
 *
 * Use for:
 * - Out-of-place FFTs
 * - General storage where padding is not needed
 * - GPU pitched memory (when pitch == logical width)
 *
 * Memory organization:
 * - Pitch equals logical X dimension
 * - No extra padding between rows
 * - Fully contiguous in memory
 */
struct DenseLayout {
    /**
     * @brief Calculate pitch (stride) for dense layout
     *
     * @param dims Logical dimensions (x, y, z)
     * @return Pitch in elements (equals dims.x)
     */
    static inline constexpr size_t CalculatePitch(int3 dims) {
        return size_t(dims.x);
    }

    /**
     * @brief Calculate total memory required for dense layout
     *
     * @param dims Logical dimensions (x, y, z)
     * @return Total number of elements needed
     */
    static inline constexpr size_t CalculateMemorySize(int3 dims) {
        return size_t(dims.x) * size_t(dims.y) * size_t(dims.z);
    }

    static constexpr const char* Name( ) { return "DenseLayout"; }
};

// ============================================================================
// FFTWPaddedLayout - FFTW in-place R2C padding
// ============================================================================

/**
 * @brief FFTW-compatible padded layout for in-place real-to-complex FFTs
 *
 * Use for:
 * - In-place FFTW real-to-complex transforms
 * - Legacy Image class compatibility
 *
 * Memory organization:
 * - Real space: dims.x + padding_jump_value elements per row
 * - Complex space: (dims.x/2 + 1) complex elements per row
 * - Padding allows in-place transformation without extra allocation
 *
 * Padding formula (FFTW convention):
 * - Even X dimension: padding_jump_value = 2
 * - Odd X dimension:  padding_jump_value = 1
 *
 * Example for 64x64 image:
 * - Real space:    64 + 2 = 66 floats per row (dims.x + 2)
 * - Complex space: 33 complex<float> per row (dims.x/2 + 1)
 * - Both representations use same memory: 66 floats = 33 complex<float>
 */
struct FFTWPaddedLayout {
    /**
     * @brief Calculate padding jump value for FFTW layout
     *
     * @param dims Logical dimensions
     * @return Padding amount (2 for even dims.x, 1 for odd)
     */
    static inline constexpr int CalculatePaddingJumpValue(int3 dims) {
        return (dims.x % 2 == 0) ? 2 : 1;
    }

    /**
     * @brief Calculate pitch (stride) for FFTW padded layout
     *
     * @param dims Logical dimensions
     * @return Pitch in elements (dims.x + padding)
     */
    static inline constexpr size_t CalculatePitch(int3 dims) {
        return size_t(dims.x) + size_t(CalculatePaddingJumpValue(dims));
    }

    /**
     * @brief Calculate total memory required for FFTW padded layout
     *
     * Allocates extra space for in-place FFT padding
     *
     * @param dims Logical dimensions
     * @return Total number of elements needed (including padding)
     */
    static inline constexpr size_t CalculateMemorySize(int3 dims) {
        size_t pitch = CalculatePitch(dims);
        return pitch * size_t(dims.y) * size_t(dims.z);
    }

    /**
     * @brief Calculate complex array pitch for Fourier space
     *
     * After R2C transform, memory is reinterpreted as complex array
     * with different dimensions (Hermitian symmetry)
     *
     * @param dims Logical dimensions in real space
     * @return Number of complex elements per row in Fourier space
     */
    static inline constexpr size_t CalculateComplexPitch(int3 dims) {
        // Hermitian symmetry: only store half + 1 in X dimension
        return size_t(dims.x / 2 + 1);
    }

    static constexpr const char* Name( ) { return "FFTWPaddedLayout"; }
};

// ============================================================================
// Future Layout Policies (Phase 3+)
// ============================================================================

/**
 * @brief CuFFT-compatible padded layout for GPU in-place transforms
 *
 * Similar to FFTW but may have different alignment requirements
 * Will be implemented in Phase 3 when adding GPU support
 */
struct CuFFTPaddedLayout; // Forward declaration for future

/**
 * @brief Explicitly pitched layout for GPU memory
 *
 * Allows specification of arbitrary pitch (e.g., from cudaMallocPitch)
 * Will be implemented in Phase 3 when adding GPU support
 */
struct PitchedLayout; // Forward declaration for future

} // namespace tensor
} // namespace cistem

#endif // _SRC_CORE_TENSOR_MEMORY_MEMORY_LAYOUT_H_
