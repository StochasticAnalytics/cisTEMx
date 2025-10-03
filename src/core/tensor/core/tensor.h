#ifndef _SRC_CORE_TENSOR_CORE_TENSOR_H_
#define _SRC_CORE_TENSOR_CORE_TENSOR_H_

/**
 * @file tensor.h
 * @brief Core Tensor class for cisTEM Tensor system
 *
 * Tensor is a non-owning view over memory with associated metadata.
 * It does NOT allocate or deallocate memory - that is handled by TensorMemoryPool.
 *
 * Key features:
 * - Non-owning view (like std::span)
 * - Template on scalar type and memory layout
 * - Track transform space (Position vs Momentum) independently of scalar type
 * - Fast element access via operator()
 * - Metadata for centering, addressing, etc.
 *
 * Phase 1: Only float scalar type supported
 * Phase 5: Will support complex<float>, __half, etc.
 */

#include <cuda_runtime.h>
#include "../type_traits.h"
#include "../memory/memory_layout.h"
#include "../debug_utils.h"
#include "../iterators/tensor_iterator.h"
#include "../../../constants/constants.h"

namespace cistem {
namespace tensor {

/**
 * @brief Non-owning view over tensor data with metadata
 *
 * Usage pattern:
 * @code
 * TensorMemoryPool<float> pool;
 * auto buffer = pool.AllocateBuffer({512, 512, 1}, true);
 *
 * Tensor<float, FFTWPaddedLayout> img;
 * img.AttachToBuffer(buffer.data, buffer.dims);
 * img.SetSpace(Tensor<float>::Space_t::Position);
 *
 * // Access elements
 * img(x, y, z) = 42.0f;
 * float val = img(x, y, z);
 *
 * // Detach before deallocating buffer
 * img.Detach();
 * pool.DeallocateBuffer(buffer);
 * @endcode
 *
 * @tparam Scalar_t Scalar data type (float in Phase 1)
 * @tparam Layout_t Memory layout policy (DenseLayout, FFTWPaddedLayout, etc.)
 */
template <typename Scalar_t,
          typename Layout_t = DenseLayout,
          typename          = cistem::EnableIf<is_phase1_supported_v<Scalar_t>>>
class Tensor {
  public:
    /**
     * @brief Transform space enumeration
     *
     * Position: Spatial/real domain
     * Momentum: Fourier/reciprocal domain
     *
     * Note: This is independent of scalar type!
     * - Position space can have complex<float> data
     * - Momentum space typically has complex data after R2C transform
     */
    enum class Space_t {
        Position, ///< Spatial domain (formerly "real space")
        Momentum ///< Fourier domain (formerly "Fourier/complex space")
    };

    /**
     * @brief Construct empty (detached) tensor
     */
    Tensor( );

    /**
     * @brief Destructor
     *
     * Does NOT deallocate memory (tensor is non-owning)
     */
    ~Tensor( ) = default;

    // Tensors are copyable (they're just views)
    Tensor(const Tensor&)            = default;
    Tensor& operator=(const Tensor&) = default;

    // Tensors are moveable
    Tensor(Tensor&&) noexcept            = default;
    Tensor& operator=(Tensor&&) noexcept = default;

    /**
     * @brief Attach tensor to a memory buffer
     *
     * Makes this tensor a view over the provided memory.
     * Does NOT take ownership - caller must ensure buffer lifetime exceeds tensor usage.
     *
     * @param data Pointer to memory buffer
     * @param dims Logical dimensions of the data
     */
    void AttachToBuffer(Scalar_t* data, int3 dims);

    /**
     * @brief Detach tensor from memory
     *
     * After detaching, tensor is in empty state and cannot be used for element access.
     * Safe to detach before deallocating the underlying buffer.
     */
    void Detach( );

    /**
     * @brief Check if tensor is attached to memory
     *
     * @return true if attached to valid memory
     */
    bool IsAttached( ) const { return data_ != nullptr; }

    // ========================================================================
    // Space management
    // ========================================================================

    /**
     * @brief Get current transform space
     *
     * @return Current space (Position or Momentum)
     */
    Space_t GetSpace( ) const { return space_; }

    /**
     * @brief Set transform space
     *
     * This only updates metadata - it does NOT perform a transform.
     * Use FFT operations to actually transform the data.
     *
     * @param space New space state
     */
    void SetSpace(Space_t space) { space_ = space; }

    /**
     * @brief Check if tensor is in position space
     *
     * @return true if in Position space
     */
    bool IsInPositionSpace( ) const { return space_ == Space_t::Position; }

    /**
     * @brief Check if tensor is in momentum space
     *
     * @return true if in Momentum space
     */
    bool IsInMomentumSpace( ) const { return space_ == Space_t::Momentum; }

    // ========================================================================
    // Dimensions and size queries
    // ========================================================================

    /**
     * @brief Get logical dimensions
     *
     * @return Dimensions as int3
     */
    int3 GetDims( ) const { return {dims_p_.x, dims_p_.y, dims_p_.z}; }

    /**
     * @brief Get total number of logical elements
     *
     * @return dims.x * dims.y * dims.z
     */
    long GetLogicalSize( ) const {
        return long(dims_p_.x) * long(dims_p_.y) * long(dims_p_.z);
    }

    /**
     * @brief Get position space pitch (stride) in elements
     *
     * @return Position space pitch
     */
    int GetPitch_p( ) const {
        return dims_p_.w;
    }

    /**
     * @brief Get momentum space pitch (stride) in elements
     *
     * @return Momentum space pitch (complex elements per row)
     */
    int GetPitch_m( ) const {
        return dims_m_.w;
    }

    /**
     * @brief Get memory pitch (stride) for current space
     *
     * @return Pitch in elements
     */
    size_t GetPitch( ) const {
        return (space_ == Space_t::Position) ? size_t(dims_p_.w) : size_t(dims_m_.w);
    }

    /**
     * @brief Get total physical memory size (including padding)
     *
     * @return Total elements allocated
     */
    size_t GetPhysicalSize( ) const {
        return size_t(dims_p_.w) * size_t(dims_p_.y) * size_t(dims_p_.z);
    }

    // ========================================================================
    // Element access - Position space
    // ========================================================================

    /**
     * @brief Access position space element (3D)
     *
     * @param x X coordinate
     * @param y Y coordinate
     * @param z Z coordinate
     * @return Reference to element
     */
    inline Scalar_t& GetValue_p(int x, int y, int z) {
        TENSOR_DEBUG_ASSERT(IsAttached( ), "Tensor not attached to memory");
        TENSOR_DEBUG_ASSERT(space_ == Space_t::Position, "Tensor not in Position space");
        return data_[dims_p_.w * dims_p_.y * z + dims_p_.w * y + x];
    }

    inline const Scalar_t& GetValue_p(int x, int y, int z) const {
        TENSOR_DEBUG_ASSERT(IsAttached( ), "Tensor not attached to memory");
        TENSOR_DEBUG_ASSERT(space_ == Space_t::Position, "Tensor not in Position space");
        return data_[dims_p_.w * dims_p_.y * z + dims_p_.w * y + x];
    }

    /**
     * @brief Access position space element (2D, z=0)
     *
     * @param x X coordinate
     * @param y Y coordinate
     * @return Reference to element
     */
    inline Scalar_t& GetValue_p(int x, int y) {
        TENSOR_DEBUG_ASSERT(IsAttached( ), "Tensor not attached to memory");
        TENSOR_DEBUG_ASSERT(space_ == Space_t::Position, "Tensor not in Position space");
        return data_[dims_p_.w * y + x];
    }

    inline const Scalar_t& GetValue_p(int x, int y) const {
        TENSOR_DEBUG_ASSERT(IsAttached( ), "Tensor not attached to memory");
        TENSOR_DEBUG_ASSERT(space_ == Space_t::Position, "Tensor not in Position space");
        return data_[dims_p_.w * y + x];
    }

    /**
     * @brief Access position space element (1D, y=0, z=0)
     *
     * @param x X coordinate
     * @return Reference to element
     */
    inline Scalar_t& GetValue_p(int x) {
        TENSOR_DEBUG_ASSERT(IsAttached( ), "Tensor not attached to memory");
        TENSOR_DEBUG_ASSERT(space_ == Space_t::Position, "Tensor not in Position space");
        return data_[x];
    }

    inline const Scalar_t& GetValue_p(int x) const {
        TENSOR_DEBUG_ASSERT(IsAttached( ), "Tensor not attached to memory");
        TENSOR_DEBUG_ASSERT(space_ == Space_t::Position, "Tensor not in Position space");
        return data_[x];
    }

    // ========================================================================
    // Element access - Momentum space
    // ========================================================================

    /**
     * @brief Access momentum space element (3D)
     *
     * @param x X coordinate (0 to dims.x/2)
     * @param y Y coordinate
     * @param z Z coordinate
     * @return Reference to element
     */
    inline Scalar_t& GetValue_m(int x, int y, int z) {
        TENSOR_DEBUG_ASSERT(IsAttached( ), "Tensor not attached to memory");
        TENSOR_DEBUG_ASSERT(space_ == Space_t::Momentum, "Tensor not in Momentum space");
        return data_[dims_m_.w * dims_m_.y * z + dims_m_.w * y + x];
    }

    inline const Scalar_t& GetValue_m(int x, int y, int z) const {
        TENSOR_DEBUG_ASSERT(IsAttached( ), "Tensor not attached to memory");
        TENSOR_DEBUG_ASSERT(space_ == Space_t::Momentum, "Tensor not in Momentum space");
        return data_[dims_m_.w * dims_m_.y * z + dims_m_.w * y + x];
    }

    /**
     * @brief Access momentum space element (2D, z=0)
     *
     * @param x X coordinate
     * @param y Y coordinate
     * @return Reference to element
     */
    inline Scalar_t& GetValue_m(int x, int y) {
        TENSOR_DEBUG_ASSERT(IsAttached( ), "Tensor not attached to memory");
        TENSOR_DEBUG_ASSERT(space_ == Space_t::Momentum, "Tensor not in Momentum space");
        return data_[dims_m_.w * y + x];
    }

    inline const Scalar_t& GetValue_m(int x, int y) const {
        TENSOR_DEBUG_ASSERT(IsAttached( ), "Tensor not attached to memory");
        TENSOR_DEBUG_ASSERT(space_ == Space_t::Momentum, "Tensor not in Momentum space");
        return data_[dims_m_.w * y + x];
    }

    /**
     * @brief Access momentum space element (1D, y=0, z=0)
     *
     * @param x X coordinate
     * @return Reference to element
     */
    inline Scalar_t& GetValue_m(int x) {
        TENSOR_DEBUG_ASSERT(IsAttached( ), "Tensor not attached to memory");
        TENSOR_DEBUG_ASSERT(space_ == Space_t::Momentum, "Tensor not in Momentum space");
        return data_[x];
    }

    inline const Scalar_t& GetValue_m(int x) const {
        TENSOR_DEBUG_ASSERT(IsAttached( ), "Tensor not attached to memory");
        TENSOR_DEBUG_ASSERT(space_ == Space_t::Momentum, "Tensor not in Momentum space");
        return data_[x];
    }

    // ========================================================================
    // Pointer access - Position space
    // ========================================================================

    /**
     * @brief Get pointer to position space element (3D)
     */
    inline Scalar_t* GetPointer_p(int x, int y, int z) {
        TENSOR_DEBUG_ASSERT(IsAttached( ), "Tensor not attached to memory");
        TENSOR_DEBUG_ASSERT(space_ == Space_t::Position, "Tensor not in Position space");
        return &data_[dims_p_.w * dims_p_.y * z + dims_p_.w * y + x];
    }

    inline const Scalar_t* GetPointer_p(int x, int y, int z) const {
        TENSOR_DEBUG_ASSERT(IsAttached( ), "Tensor not attached to memory");
        TENSOR_DEBUG_ASSERT(space_ == Space_t::Position, "Tensor not in Position space");
        return &data_[dims_p_.w * dims_p_.y * z + dims_p_.w * y + x];
    }

    /**
     * @brief Get pointer to position space element (2D, z=0)
     */
    inline Scalar_t* GetPointer_p(int x, int y) {
        TENSOR_DEBUG_ASSERT(IsAttached( ), "Tensor not attached to memory");
        TENSOR_DEBUG_ASSERT(space_ == Space_t::Position, "Tensor not in Position space");
        return &data_[dims_p_.w * y + x];
    }

    inline const Scalar_t* GetPointer_p(int x, int y) const {
        TENSOR_DEBUG_ASSERT(IsAttached( ), "Tensor not attached to memory");
        TENSOR_DEBUG_ASSERT(space_ == Space_t::Position, "Tensor not in Position space");
        return &data_[dims_p_.w * y + x];
    }

    /**
     * @brief Get pointer to position space element (1D, y=0, z=0)
     */
    inline Scalar_t* GetPointer_p(int x) {
        TENSOR_DEBUG_ASSERT(IsAttached( ), "Tensor not attached to memory");
        TENSOR_DEBUG_ASSERT(space_ == Space_t::Position, "Tensor not in Position space");
        return &data_[x];
    }

    inline const Scalar_t* GetPointer_p(int x) const {
        TENSOR_DEBUG_ASSERT(IsAttached( ), "Tensor not attached to memory");
        TENSOR_DEBUG_ASSERT(space_ == Space_t::Position, "Tensor not in Position space");
        return &data_[x];
    }

    // ========================================================================
    // Pointer access - Momentum space
    // ========================================================================

    /**
     * @brief Get pointer to momentum space element (3D)
     */
    inline Scalar_t* GetPointer_m(int x, int y, int z) {
        TENSOR_DEBUG_ASSERT(IsAttached( ), "Tensor not attached to memory");
        TENSOR_DEBUG_ASSERT(space_ == Space_t::Momentum, "Tensor not in Momentum space");
        return &data_[dims_m_.w * dims_m_.y * z + dims_m_.w * y + x];
    }

    inline const Scalar_t* GetPointer_m(int x, int y, int z) const {
        TENSOR_DEBUG_ASSERT(IsAttached( ), "Tensor not attached to memory");
        TENSOR_DEBUG_ASSERT(space_ == Space_t::Momentum, "Tensor not in Momentum space");
        return &data_[dims_m_.w * dims_m_.y * z + dims_m_.w * y + x];
    }

    /**
     * @brief Get pointer to momentum space element (2D, z=0)
     */
    inline Scalar_t* GetPointer_m(int x, int y) {
        TENSOR_DEBUG_ASSERT(IsAttached( ), "Tensor not attached to memory");
        TENSOR_DEBUG_ASSERT(space_ == Space_t::Momentum, "Tensor not in Momentum space");
        return &data_[dims_m_.w * y + x];
    }

    inline const Scalar_t* GetPointer_m(int x, int y) const {
        TENSOR_DEBUG_ASSERT(IsAttached( ), "Tensor not attached to memory");
        TENSOR_DEBUG_ASSERT(space_ == Space_t::Momentum, "Tensor not in Momentum space");
        return &data_[dims_m_.w * y + x];
    }

    /**
     * @brief Get pointer to momentum space element (1D, y=0, z=0)
     */
    inline Scalar_t* GetPointer_m(int x) {
        TENSOR_DEBUG_ASSERT(IsAttached( ), "Tensor not attached to memory");
        TENSOR_DEBUG_ASSERT(space_ == Space_t::Momentum, "Tensor not in Momentum space");
        return &data_[x];
    }

    inline const Scalar_t* GetPointer_m(int x) const {
        TENSOR_DEBUG_ASSERT(IsAttached( ), "Tensor not attached to memory");
        TENSOR_DEBUG_ASSERT(space_ == Space_t::Momentum, "Tensor not in Momentum space");
        return &data_[x];
    }

    /**
     * @brief Get raw data pointer (mutable)
     *
     * Use with caution - direct pointer access bypasses bounds checking
     *
     * @return Pointer to underlying data
     */
    Scalar_t* Data( ) { return data_; }

    /**
     * @brief Get raw data pointer (const)
     *
     * @return Const pointer to underlying data
     */
    const Scalar_t* Data( ) const { return data_; }

    // ========================================================================
    // Metadata (for compatibility with legacy Image)
    // ========================================================================

    /**
     * @brief Check if object is centered in box
     *
     * Whether the region of interest is near the center (vs wrapped at corners)
     * Only meaningful in position space
     *
     * @return true if centered
     */
    bool IsObjectCentered( ) const { return object_is_centered_in_box_; }

    /**
     * @brief Set object centering state
     *
     * @param centered New centering state
     */
    void SetObjectCentered(bool centered) { object_is_centered_in_box_ = centered; }

    /**
     * @brief Check if FFT is centered
     *
     * Whether DC component is at center (vs at origin)
     * Only meaningful in momentum space
     *
     * @return true if FFT centered
     */
    bool IsFFTCentered( ) const { return is_fft_centered_in_box_; }

    /**
     * @brief Set FFT centering state
     *
     * @param centered New FFT centering state
     */
    void SetFFTCentered(bool centered) { is_fft_centered_in_box_ = centered; }

    // ========================================================================
    // Iterators (Phase 2.5) - Enable range-based for and STL algorithms
    // ========================================================================

    /**
     * @brief Begin iterator (layout-aware)
     *
     * Returns appropriate iterator based on Layout_t:
     * - DenseLayout: Simple pointer-based iterator
     * - FFTWPaddedLayout: Padding-aware iterator
     *
     * Usage: for (auto& val : tensor) { val += 1.0f; }
     */
    auto begin( ) {
        TENSOR_DEBUG_ASSERT(IsAttached( ), "Tensor not attached to memory");
        if constexpr ( std::is_same_v<Layout_t, DenseLayout> ) {
            return TensorIterator_Dense<Scalar_t>(data_);
        }
        else if constexpr ( std::is_same_v<Layout_t, FFTWPaddedLayout> ) {
            int  padding_jump = (dims_p_.x % 2 == 0) ? 2 : 1;
            int3 logical_dims = make_int3(dims_p_.x, dims_p_.y, dims_p_.z);
            return TensorIterator_FFTWPadded<Scalar_t>(data_, logical_dims, dims_p_.w, padding_jump);
        }
        else {
            static_assert(std::is_same_v<Layout_t, DenseLayout> || std::is_same_v<Layout_t, FFTWPaddedLayout>,
                          "Unsupported layout type");
        }
    }

    /**
     * @brief End iterator (layout-aware)
     */
    auto end( ) {
        TENSOR_DEBUG_ASSERT(IsAttached( ), "Tensor not attached to memory");
        if constexpr ( std::is_same_v<Layout_t, DenseLayout> ) {
            long num_elements = long(dims_p_.x) * long(dims_p_.y) * long(dims_p_.z);
            return TensorIterator_Dense<Scalar_t>(data_ + num_elements);
        }
        else if constexpr ( std::is_same_v<Layout_t, FFTWPaddedLayout> ) {
            int padding_jump = (dims_p_.x % 2 == 0) ? 2 : 1;
            // End iterator points past the last element
            long final_address = long(dims_p_.w) * long(dims_p_.y) * long(dims_p_.z);
            int3 logical_dims  = make_int3(dims_p_.x, dims_p_.y, dims_p_.z);
            return TensorIterator_FFTWPadded<Scalar_t>(data_ + final_address, logical_dims, dims_p_.w, padding_jump, 0, 0, dims_p_.z);
        }
        else {
            static_assert(std::is_same_v<Layout_t, DenseLayout> || std::is_same_v<Layout_t, FFTWPaddedLayout>,
                          "Unsupported layout type");
        }
    }

    /**
     * @brief Const begin iterator
     */
    auto begin( ) const {
        TENSOR_DEBUG_ASSERT(IsAttached( ), "Tensor not attached to memory");
        if constexpr ( std::is_same_v<Layout_t, DenseLayout> ) {
            return ConstTensorIterator_Dense<Scalar_t>(data_);
        }
        else if constexpr ( std::is_same_v<Layout_t, FFTWPaddedLayout> ) {
            int  padding_jump = (dims_p_.x % 2 == 0) ? 2 : 1;
            int3 logical_dims = make_int3(dims_p_.x, dims_p_.y, dims_p_.z);
            return ConstTensorIterator_FFTWPadded<Scalar_t>(data_, logical_dims, dims_p_.w, padding_jump);
        }
        else {
            static_assert(std::is_same_v<Layout_t, DenseLayout> || std::is_same_v<Layout_t, FFTWPaddedLayout>,
                          "Unsupported layout type");
        }
    }

    /**
     * @brief Const end iterator
     */
    auto end( ) const {
        TENSOR_DEBUG_ASSERT(IsAttached( ), "Tensor not attached to memory");
        if constexpr ( std::is_same_v<Layout_t, DenseLayout> ) {
            long num_elements = long(dims_p_.x) * long(dims_p_.y) * long(dims_p_.z);
            return ConstTensorIterator_Dense<Scalar_t>(data_ + num_elements);
        }
        else if constexpr ( std::is_same_v<Layout_t, FFTWPaddedLayout> ) {
            int  padding_jump  = (dims_p_.x % 2 == 0) ? 2 : 1;
            long final_address = long(dims_p_.w) * long(dims_p_.y) * long(dims_p_.z);
            int3 logical_dims  = make_int3(dims_p_.x, dims_p_.y, dims_p_.z);
            return ConstTensorIterator_FFTWPadded<Scalar_t>(data_ + final_address, logical_dims, dims_p_.w, padding_jump, 0, 0, dims_p_.z);
        }
        else {
            static_assert(std::is_same_v<Layout_t, DenseLayout> || std::is_same_v<Layout_t, FFTWPaddedLayout>,
                          "Unsupported layout type");
        }
    }

    /**
     * @brief Const begin iterator (explicit cbegin)
     */
    auto cbegin( ) const { return begin( ); }

    /**
     * @brief Const end iterator (explicit cend)
     */
    auto cend( ) const { return end( ); }

  private:
    Scalar_t* data_; ///< Non-owning pointer to data
    int4      dims_p_; ///< Position space dimensions {x, y, z, pitch}
    int4      dims_m_; ///< Momentum space dimensions {x, y, z, complex_pitch}
    Space_t   space_; ///< Current transform space
    bool      object_is_centered_in_box_; ///< Object centering (position space)
    bool      is_fft_centered_in_box_; ///< FFT centering (momentum space)
};

// ============================================================================
// Implementation of inline methods
// ============================================================================

template <typename Scalar_t, typename Layout_t, typename EnableIf_t>
Tensor<Scalar_t, Layout_t, EnableIf_t>::Tensor( )
    : data_(nullptr),
      dims_p_({0, 0, 0, 0}),
      dims_m_({0, 0, 0, 0}),
      space_(Space_t::Position),
      object_is_centered_in_box_(false),
      is_fft_centered_in_box_(false) {
}

template <typename Scalar_t, typename Layout_t, typename EnableIf_t>
void Tensor<Scalar_t, Layout_t, EnableIf_t>::AttachToBuffer(Scalar_t* data, int3 dims) {
    TENSOR_DEBUG_ASSERT(data != nullptr, "Cannot attach to nullptr buffer");
    TENSOR_DEBUG_ASSERT(dims.x > 0 && dims.y > 0 && dims.z > 0,
                        "Invalid dimensions: %d x %d x %d", dims.x, dims.y, dims.z);

    data_ = data;

    // Pre-compute position space dimensions and pitch
    dims_p_.x = dims.x;
    dims_p_.y = dims.y;
    dims_p_.z = dims.z;
    dims_p_.w = int(Layout_t::CalculatePitch(dims));

    // Pre-compute momentum space dimensions and complex pitch
    dims_m_.x = dims.x;
    dims_m_.y = dims.y;
    dims_m_.z = dims.z;
    dims_m_.w = int(Layout_t::CalculateComplexPitch(dims));

    // Don't change space_ - let caller set it explicitly
    // Reset metadata
    object_is_centered_in_box_ = false;
    is_fft_centered_in_box_    = false;
}

template <typename Scalar_t, typename Layout_t, typename EnableIf_t>
void Tensor<Scalar_t, Layout_t, EnableIf_t>::Detach( ) {
    data_   = nullptr;
    dims_p_ = {0, 0, 0, 0};
    dims_m_ = {0, 0, 0, 0};
    // Reset metadata but keep space_ for potential debugging
}

} // namespace tensor
} // namespace cistem

#endif // _SRC_CORE_TENSOR_CORE_TENSOR_H_
