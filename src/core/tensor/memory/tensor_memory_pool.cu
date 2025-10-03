/**
 * @file tensor_memory_pool.cpp
 * @brief Implementation of TensorMemoryPool
 */

#include "tensor_memory_pool.h"
#include "../debug_utils.h"
#include <algorithm>
#include <stdexcept>
#include <cstdio>
#include <mutex>

namespace cistem {
namespace tensor {

// ============================================================================
// Template instantiations for Phase 1
// ============================================================================

template class TensorMemoryPool<float, fftwf_plan>;

// ============================================================================
// Constructor / Destructor
// ============================================================================

template <typename Scalar_t, typename FFTPlan_t, typename EnableIf_t>
TensorMemoryPool<Scalar_t, FFTPlan_t, EnableIf_t>::TensorMemoryPool( ) {
    // Empty constructor - no allocations yet
}

template <typename Scalar_t, typename FFTPlan_t, typename EnableIf_t>
TensorMemoryPool<Scalar_t, FFTPlan_t, EnableIf_t>::~TensorMemoryPool( ) {
    if ( ! active_buffers_.empty( ) ) {
        std::fprintf(stderr, "WARNING: TensorMemoryPool destroyed with %zu active buffer(s)!\n",
                     active_buffers_.size( ));
        std::fprintf(stderr, "         This indicates a memory leak. Deallocate all buffers before destroying pool.\n");

        // In debug builds, this is a hard error
        TENSOR_DEBUG_ASSERT(active_buffers_.empty( ),
                            "Memory leak detected: TensorMemoryPool has %zu active buffers at destruction",
                            active_buffers_.size( ));

        // Clean up leaked buffers to avoid actual memory leaks
        // Note: We can't call DeallocateBuffer here because we only have the data pointer
        // Just free the memory and warn - this is a leak scenario anyway
        for ( Scalar_t* data_ptr : active_buffers_ ) {
            fftwf_free(data_ptr);
        }
        active_buffers_.clear( );
    }
}

// ============================================================================
// Move semantics
// ============================================================================

template <typename Scalar_t, typename FFTPlan_t, typename EnableIf_t>
TensorMemoryPool<Scalar_t, FFTPlan_t, EnableIf_t>::TensorMemoryPool(TensorMemoryPool&& other) noexcept
    : active_buffers_(std::move(other.active_buffers_)) {
    // Other pool is now empty
}

template <typename Scalar_t, typename FFTPlan_t, typename EnableIf_t>
TensorMemoryPool<Scalar_t, FFTPlan_t, EnableIf_t>& TensorMemoryPool<Scalar_t, FFTPlan_t, EnableIf_t>::operator=(TensorMemoryPool&& other) noexcept {
    if ( this != &other ) {
        // Clean up our existing buffers
        // Note: We only have data pointers, so just free the memory
        for ( Scalar_t* data_ptr : active_buffers_ ) {
            fftwf_free(data_ptr);
        }

        // Take ownership of other's buffers
        active_buffers_ = std::move(other.active_buffers_);
    }
    return *this;
}

// ============================================================================
// Buffer allocation
// ============================================================================

template <typename Scalar_t, typename FFTPlan_t, typename EnableIf_t>
typename TensorMemoryPool<Scalar_t, FFTPlan_t, EnableIf_t>::Buffer
TensorMemoryPool<Scalar_t, FFTPlan_t, EnableIf_t>::AllocateBuffer(int3 dims, bool create_fft_plans, size_t layout_size) {
    TENSOR_DEBUG_ASSERT(dims.x > 0 && dims.y > 0 && dims.z > 0,
                        "Invalid dimensions for buffer allocation: %d x %d x %d",
                        dims.x, dims.y, dims.z);

    Buffer buffer;
    buffer.dims = dims;

    // Calculate memory size
    // If layout_size is specified, use it; otherwise use dense layout
    if ( layout_size == 0 ) {
        layout_size = size_t(dims.x) * size_t(dims.y) * size_t(dims.z);
    }
    buffer.size = layout_size;

    // Allocate memory using FFTW allocator for proper alignment
    buffer.data = reinterpret_cast<Scalar_t*>(
            fftwf_malloc(sizeof(Scalar_t) * buffer.size));

    if ( buffer.data == nullptr ) {
        throw std::bad_alloc( );
    }

    // Create FFT plans if requested
    if ( create_fft_plans ) {
        try {
            CreateFFTPlans(buffer);
        } catch ( ... ) {
            // Clean up memory if plan creation fails
            fftwf_free(buffer.data);
            throw;
        }
    }

    // Track this allocation for leak detection
    active_buffers_.push_back(buffer.data);

    return buffer;
}

// ============================================================================
// Buffer deallocation
// ============================================================================

template <typename Scalar_t, typename FFTPlan_t, typename EnableIf_t>
void TensorMemoryPool<Scalar_t, FFTPlan_t, EnableIf_t>::DeallocateBuffer(Buffer& buffer) {
    if ( buffer.data == nullptr ) {
        // Already deallocated or never allocated
        return;
    }

    // Destroy FFT plans if they exist
    if ( buffer.fft_plan_forward != nullptr || buffer.fft_plan_backward != nullptr ) {
        DestroyFFTPlans(buffer);
    }

    // Free memory
    fftwf_free(buffer.data);

    // Remove from tracking
    auto it = std::find(active_buffers_.begin( ), active_buffers_.end( ), buffer.data);

    if ( it != active_buffers_.end( ) ) {
        active_buffers_.erase(it);
    }

    // Reset buffer handle to nullptr values
    buffer.data              = nullptr;
    buffer.dims              = {0, 0, 0};
    buffer.size              = 0;
    buffer.fft_plan_forward  = nullptr;
    buffer.fft_plan_backward = nullptr;
}

// ============================================================================
// FFT plan management
// ============================================================================

template <typename Scalar_t, typename FFTPlan_t, typename EnableIf_t>
void TensorMemoryPool<Scalar_t, FFTPlan_t, EnableIf_t>::CreateFFTPlans(Buffer& buffer) {
    // FFTW planning requires thread safety
    static std::mutex           fftw_mutex;
    std::lock_guard<std::mutex> lock(fftw_mutex);

    // For Phase 1: only float supported, use fftwf functions
    // Cast data pointer for FFTW (real data can alias complex data)
    float*         real_data    = reinterpret_cast<float*>(buffer.data);
    fftwf_complex* complex_data = reinterpret_cast<fftwf_complex*>(buffer.data);

    // Create plans based on dimensionality
    if ( buffer.dims.z > 1 ) {
        // 3D transforms
        buffer.fft_plan_forward = fftwf_plan_dft_r2c_3d(
                buffer.dims.z, buffer.dims.y, buffer.dims.x,
                real_data, complex_data,
                FFTW_ESTIMATE);

        buffer.fft_plan_backward = fftwf_plan_dft_c2r_3d(
                buffer.dims.z, buffer.dims.y, buffer.dims.x,
                complex_data, real_data,
                FFTW_ESTIMATE);
    }
    else {
        // 2D transforms
        buffer.fft_plan_forward = fftwf_plan_dft_r2c_2d(
                buffer.dims.y, buffer.dims.x,
                real_data, complex_data,
                FFTW_ESTIMATE);

        buffer.fft_plan_backward = fftwf_plan_dft_c2r_2d(
                buffer.dims.y, buffer.dims.x,
                complex_data, real_data,
                FFTW_ESTIMATE);
    }

    // Check that plans were created successfully
    if ( buffer.fft_plan_forward == nullptr || buffer.fft_plan_backward == nullptr ) {
        // Clean up any partial allocation
        if ( buffer.fft_plan_forward != nullptr ) {
            fftwf_destroy_plan(buffer.fft_plan_forward);
            buffer.fft_plan_forward = nullptr;
        }
        if ( buffer.fft_plan_backward != nullptr ) {
            fftwf_destroy_plan(buffer.fft_plan_backward);
            buffer.fft_plan_backward = nullptr;
        }

        throw std::runtime_error("Failed to create FFTW plans");
    }
}

template <typename Scalar_t, typename FFTPlan_t, typename EnableIf_t>
void TensorMemoryPool<Scalar_t, FFTPlan_t, EnableIf_t>::DestroyFFTPlans(Buffer& buffer) {
    // FFTW plan destruction requires thread safety
    static std::mutex           fftw_mutex;
    std::lock_guard<std::mutex> lock(fftw_mutex);

    if ( buffer.fft_plan_forward != nullptr ) {
        fftwf_destroy_plan(buffer.fft_plan_forward);
        buffer.fft_plan_forward = nullptr;
    }

    if ( buffer.fft_plan_backward != nullptr ) {
        fftwf_destroy_plan(buffer.fft_plan_backward);
        buffer.fft_plan_backward = nullptr;
    }
}

} // namespace tensor
} // namespace cistem
