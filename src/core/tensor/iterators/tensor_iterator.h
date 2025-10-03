#ifndef _SRC_CORE_TENSOR_ITERATORS_TENSOR_ITERATOR_H_
#define _SRC_CORE_TENSOR_ITERATORS_TENSOR_ITERATOR_H_

/**
 * @file tensor_iterator.h
 * @brief Layout-aware iterators for Tensor class (Phase 2.5)
 *
 * Provides efficient iteration over Tensor data with automatic padding handling.
 * Enables:
 * - Range-based for loops: for (auto& val : tensor) { ... }
 * - STL algorithms: std::transform, std::accumulate, etc.
 * - Optimal performance: single pointer increment, no address recalculation
 *
 * Design:
 * - Template specialization for different layouts
 * - Skips padding automatically (FFTWPaddedLayout)
 * - Bidirectional iterator
 * - Const and non-const versions
 */

#include <iterator>
#include <cuda_runtime.h>
#include "../memory/memory_layout.h"

namespace cistem {
namespace tensor {

/**
 * @brief Iterator for DenseLayout tensors (no padding)
 *
 * Simple contiguous memory iteration.
 */
template <typename Scalar_t>
class TensorIterator_Dense {
  public:
    // STL iterator traits
    using iterator_category = std::random_access_iterator_tag;
    using value_type        = Scalar_t;
    using difference_type   = std::ptrdiff_t;
    using pointer           = Scalar_t*;
    using reference         = Scalar_t&;

    TensorIterator_Dense( ) : ptr_(nullptr) {}

    explicit TensorIterator_Dense(pointer ptr) : ptr_(ptr) {}

    // Dereference
    reference operator*( ) const { return *ptr_; }

    pointer operator->( ) const { return ptr_; }

    reference operator[](difference_type n) const { return ptr_[n]; }

    // Increment/Decrement
    TensorIterator_Dense& operator++( ) {
        ++ptr_;
        return *this;
    }

    TensorIterator_Dense operator++(int) {
        TensorIterator_Dense tmp = *this;
        ++ptr_;
        return tmp;
    }

    TensorIterator_Dense& operator--( ) {
        --ptr_;
        return *this;
    }

    TensorIterator_Dense operator--(int) {
        TensorIterator_Dense tmp = *this;
        --ptr_;
        return tmp;
    }

    // Arithmetic
    TensorIterator_Dense& operator+=(difference_type n) {
        ptr_ += n;
        return *this;
    }

    TensorIterator_Dense& operator-=(difference_type n) {
        ptr_ -= n;
        return *this;
    }

    TensorIterator_Dense operator+(difference_type n) const {
        return TensorIterator_Dense(ptr_ + n);
    }

    TensorIterator_Dense operator-(difference_type n) const {
        return TensorIterator_Dense(ptr_ - n);
    }

    difference_type operator-(const TensorIterator_Dense& other) const {
        return ptr_ - other.ptr_;
    }

    // Comparison
    bool operator==(const TensorIterator_Dense& other) const { return ptr_ == other.ptr_; }

    bool operator!=(const TensorIterator_Dense& other) const { return ptr_ != other.ptr_; }

    bool operator<(const TensorIterator_Dense& other) const { return ptr_ < other.ptr_; }

    bool operator<=(const TensorIterator_Dense& other) const { return ptr_ <= other.ptr_; }

    bool operator>(const TensorIterator_Dense& other) const { return ptr_ > other.ptr_; }

    bool operator>=(const TensorIterator_Dense& other) const { return ptr_ >= other.ptr_; }

  private:
    pointer ptr_;
};

/**
 * @brief Iterator for FFTWPaddedLayout tensors (automatic padding skip)
 *
 * Handles FFTW padding by:
 * - Iterating normally within each row
 * - Skipping padding_jump_value elements at end of row
 * - Advancing to next row
 */
template <typename Scalar_t>
class TensorIterator_FFTWPadded {
  public:
    // STL iterator traits
    using iterator_category = std::bidirectional_iterator_tag;
    using value_type        = Scalar_t;
    using difference_type   = std::ptrdiff_t;
    using pointer           = Scalar_t*;
    using reference         = Scalar_t&;

    TensorIterator_FFTWPadded( )
        : ptr_(nullptr), x_(0), y_(0), z_(0), dims_{0, 0, 0}, pitch_(0), padding_jump_(0) {}

    TensorIterator_FFTWPadded(pointer ptr, int3 dims, int pitch, int padding_jump, int x = 0, int y = 0, int z = 0)
        : ptr_(ptr), x_(x), y_(y), z_(z), dims_(dims), pitch_(pitch), padding_jump_(padding_jump) {}

    // Dereference
    reference operator*( ) const { return *ptr_; }

    pointer operator->( ) const { return ptr_; }

    // Increment
    TensorIterator_FFTWPadded& operator++( ) {
        ++x_;
        if ( x_ >= dims_.x ) {
            // End of row - skip padding and move to next row
            x_ = 0;
            ++y_;
            if ( y_ >= dims_.y ) {
                // End of slice - move to next slice
                y_ = 0;
                ++z_;
            }
            // Skip padding and move to start of next row
            ptr_ += padding_jump_ + 1;
        }
        else {
            // Normal increment within row
            ++ptr_;
        }
        return *this;
    }

    TensorIterator_FFTWPadded operator++(int) {
        TensorIterator_FFTWPadded tmp = *this;
        ++(*this);
        return tmp;
    }

    // Decrement
    TensorIterator_FFTWPadded& operator--( ) {
        if ( x_ == 0 ) {
            // Start of row - move to end of previous row
            x_ = dims_.x - 1;
            if ( y_ == 0 ) {
                // Start of slice - move to end of previous slice
                y_ = dims_.y - 1;
                --z_;
            }
            else {
                --y_;
            }
            // Move back over padding to end of previous row
            ptr_ -= padding_jump_ + 1;
        }
        else {
            // Normal decrement within row
            --x_;
            --ptr_;
        }
        return *this;
    }

    TensorIterator_FFTWPadded operator--(int) {
        TensorIterator_FFTWPadded tmp = *this;
        --(*this);
        return tmp;
    }

    // Comparison
    bool operator==(const TensorIterator_FFTWPadded& other) const {
        return ptr_ == other.ptr_ && x_ == other.x_ && y_ == other.y_ && z_ == other.z_;
    }

    bool operator!=(const TensorIterator_FFTWPadded& other) const {
        return ! (*this == other);
    }

    // Position queries (for debugging)
    int GetX( ) const { return x_; }

    int GetY( ) const { return y_; }

    int GetZ( ) const { return z_; }

  private:
    pointer ptr_;
    int     x_, y_, z_; // Logical position
    int3    dims_; // Tensor dimensions
    int     pitch_; // Memory pitch (logical_x + padding)
    int     padding_jump_; // Elements to skip at end of row
};

// Const iterator versions
template <typename Scalar_t>
using ConstTensorIterator_Dense = TensorIterator_Dense<const Scalar_t>;

template <typename Scalar_t>
using ConstTensorIterator_FFTWPadded = TensorIterator_FFTWPadded<const Scalar_t>;

} // namespace tensor
} // namespace cistem

#endif // _SRC_CORE_TENSOR_ITERATORS_TENSOR_ITERATOR_H_
