#ifndef _SRC_CORE_TENSOR_MEMORY_TENSOR_MEMORY_POOL_H_
#define _SRC_CORE_TENSOR_MEMORY_TENSOR_MEMORY_POOL_H_

/**
 * @file tensor_memory_pool.h
 * @brief Memory pool for cisTEM Tensor system
 *
 * TensorMemoryPool owns memory buffers and associated FFT plans.
 * Tensors are non-owning views that attach to buffers from the pool.
 *
 * Key responsibilities:
 * - Allocate/deallocate memory buffers
 * - Create and own FFT plans (plans are tied to specific memory addresses)
 * - Track all active allocations for leak detection
 * - Provide properly aligned memory for SIMD and FFT operations
 *
 * Phase 1: Only float type supported
 * Phase 5: Will support complex<float>, __half, etc.
 */

#include <vector>
#include <fftw3.h>
#include <cuda_runtime.h>
#include "../type_traits.h"
#include "../../../constants/constants.h"

namespace cistem {
namespace tensor {

/**
 * @brief Memory pool that owns buffers and FFT plans
 *
 * Usage pattern:
 * @code
 * TensorMemoryPool<float> pool;
 * auto buffer = pool.AllocateBuffer({512, 512, 1}, true);  // with FFT plans
 *
 * Tensor<float> view;
 * view.AttachToBuffer(buffer.data, buffer.dims);
 *
 * // Use tensor...
 *
 * pool.DeallocateBuffer(buffer);  // Frees memory and plans
 * @endcode
 *
 * @tparam Scalar_t Scalar data type (float in Phase 1)
 * @tparam FFTPlan_t FFT plan type (fftwf_plan in Phase 1, will support cufftHandle, etc. in Phase 5)
 */
template <typename Scalar_t,
          typename FFTPlan_t = fftwf_plan,
          typename           = cistem::EnableIf<is_phase1_supported_v<Scalar_t>>>
class TensorMemoryPool {
  public:
    /**
     * @brief Buffer handle containing data pointer, dimensions, and FFT plans
     *
     * Returned by AllocateBuffer(). User must keep this handle to deallocate.
     * FFT plans are optional and only created if requested during allocation.
     *
     * Buffer is non-copyable to prevent double-free bugs (two handles pointing
     * to same memory). It is movable to allow efficient transfer of ownership.
     *
     * Rationale: While we could use smart pointers, the non-owning handle pattern
     * matches the Tensor design (lightweight view) and has zero overhead. Making
     * it non-copyable provides safety without runtime cost.
     */
    struct Buffer {
        Scalar_t* data; ///< Pointer to allocated memory
        int3      dims; ///< Logical dimensions (x, y, z)
        size_t    size; ///< Total number of elements allocated
        FFTPlan_t fft_plan_forward; ///< Forward FFT plan (typed, e.g., fftwf_plan or cufftHandle)
        FFTPlan_t fft_plan_backward; ///< Backward FFT plan (typed, e.g., fftwf_plan or cufftHandle)

        Buffer( ) : data(nullptr), dims({0, 0, 0}), size(0),
                    fft_plan_forward(nullptr), fft_plan_backward(nullptr) {}

        // Delete copy operations to prevent double-free
        Buffer(const Buffer&)            = delete;
        Buffer& operator=(const Buffer&) = delete;

        // Allow move operations for efficient ownership transfer
        Buffer(Buffer&&)            = default;
        Buffer& operator=(Buffer&&) = default;
    };

    /**
     * @brief Construct empty memory pool
     */
    TensorMemoryPool( );

    /**
     * @brief Destructor - checks for memory leaks
     *
     * Warns and asserts (in debug builds) if any buffers are still allocated
     */
    ~TensorMemoryPool( );

    // Disable copy (buffers have unique ownership)
    TensorMemoryPool(const TensorMemoryPool&)            = delete;
    TensorMemoryPool& operator=(const TensorMemoryPool&) = delete;

    // Allow move
    TensorMemoryPool(TensorMemoryPool&& other) noexcept;
    TensorMemoryPool& operator=(TensorMemoryPool&& other) noexcept;

    /**
     * @brief Allocate a memory buffer with optional FFT plans
     *
     * Memory is allocated via fftwf_malloc() for proper SIMD alignment.
     * If create_fft_plans is true, creates FFTW plans for the given dimensions.
     *
     * @param dims Logical dimensions (x, y, z)
     * @param create_fft_plans Whether to create FFT plans for this buffer
     * @param layout_size Actual memory size (may exceed dims for padded layouts)
     * @return Buffer handle with allocated memory and optional plans
     *
     * @throws std::bad_alloc if memory allocation fails
     * @throws std::runtime_error if FFT plan creation fails
     */
    Buffer AllocateBuffer(int3 dims, bool create_fft_plans = false,
                          size_t layout_size = 0);

    /**
     * @brief Deallocate a buffer and destroy its FFT plans
     *
     * After deallocation, the buffer handle is reset to nullptr values.
     * Safe to call multiple times (idempotent).
     *
     * @param buffer Buffer handle to deallocate
     */
    void DeallocateBuffer(Buffer& buffer);

    /**
     * @brief Get number of currently allocated buffers
     *
     * Useful for leak detection and debugging
     *
     * @return Number of active buffers
     */
    size_t GetActiveBufferCount( ) const { return active_buffers_.size( ); }

    /**
     * @brief Check if any buffers are allocated
     *
     * @return true if pool has active allocations
     */
    bool HasActiveBuffers( ) const { return ! active_buffers_.empty( ); }

  private:
    /// Tracks all currently allocated data pointers for leak detection
    std::vector<Scalar_t*> active_buffers_;

    /**
     * @brief Create FFTW plans for a buffer
     *
     * Plans are created with FFTW_ESTIMATE for speed.
     * Plans are tied to the specific memory address.
     *
     * @param buffer Buffer to create plans for
     */
    void CreateFFTPlans(Buffer& buffer);

    /**
     * @brief Destroy FFTW plans for a buffer
     *
     * @param buffer Buffer whose plans to destroy
     */
    void DestroyFFTPlans(Buffer& buffer);
};

} // namespace tensor
} // namespace cistem

#endif // _SRC_CORE_TENSOR_MEMORY_TENSOR_MEMORY_POOL_H_
