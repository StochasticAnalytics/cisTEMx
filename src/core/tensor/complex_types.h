#ifndef _SRC_CORE_TENSOR_COMPLEX_TYPES_H_
#define _SRC_CORE_TENSOR_COMPLEX_TYPES_H_

/**
 * @file complex_types.h
 * @brief Complex number types for cisTEM Tensor system
 *
 * Provides cistem::complex<T> templated on scalar type, with specializations
 * for CPU and GPU types. Design inspired by NVIDIA mathdx commondx/complex_types.hpp.
 *
 * Key features:
 * - Template on any scalar type (float, double, __half, __nv_bfloat16)
 * - Proper alignment for SIMD vectorization
 * - GPU compatibility via __device__ __host__ decorators
 * - Explicit conversions between precisions
 *
 * Phase 1: CPU types only (float, double)
 * Phase 5: GPU types (__half, __nv_bfloat16)
 */

#include <type_traits>
#include <cmath>

#ifdef ENABLE_GPU
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

namespace cistem {
namespace tensor {

namespace detail {

/**
 * @brief Base complex type providing common functionality
 *
 * Uses CRTP pattern for proper alignment and GPU compatibility.
 * Derived classes specialize storage and operations.
 */
template <typename T>
struct complex_base {
    using value_type = T;

    complex_base( )                   = default;
    complex_base(const complex_base&) = default;
    complex_base(complex_base&&)      = default;

#ifdef ENABLE_GPU
    __device__ __host__ constexpr complex_base(value_type re, value_type im) : x(re), y(im) {}

    __device__ __host__ constexpr value_type real( ) const { return x; }

    __device__ __host__ constexpr value_type imag( ) const { return y; }

    __device__ __host__ void real(value_type re) { x = re; }

    __device__ __host__ void imag(value_type im) { y = im; }
#else
    constexpr complex_base(value_type re, value_type im) : x(re), y(im) {}

    constexpr value_type real( ) const { return x; }

    constexpr value_type imag( ) const { return y; }

    void real(value_type re) { x = re; }

    void          imag(value_type im) { y = im; }
#endif

    // Assignment operators
    complex_base& operator=(const complex_base&) = default;
    complex_base& operator=(complex_base&&)      = default;

#ifdef ENABLE_GPU
    __device__ __host__ complex_base& operator=(value_type re) {
        x = re;
        y = value_type( ); // zero-initialized
        return *this;
    }
#else
    complex_base& operator=(value_type re) {
        x = re;
        y = value_type( );
        return *this;
    }
#endif

    // Arithmetic with scalars
#ifdef ENABLE_GPU
    __device__ __host__ complex_base& operator+=(value_type re) {
        x += re;
        return *this;
    }

    __device__ __host__ complex_base& operator-=(value_type re) {
        x -= re;
        return *this;
    }

    __device__ __host__ complex_base& operator*=(value_type re) {
        x *= re;
        y *= re;
        return *this;
    }

    __device__ __host__ complex_base& operator/=(value_type re) {
        x /= re;
        y /= re;
        return *this;
    }
#else
    complex_base& operator+=(value_type re) {
        x += re;
        return *this;
    }

    complex_base& operator-=(value_type re) {
        x -= re;
        return *this;
    }

    complex_base& operator*=(value_type re) {
        x *= re;
        y *= re;
        return *this;
    }

    complex_base& operator/=(value_type re) {
        x /= re;
        y /= re;
        return *this;
    }
#endif

    // Arithmetic with other complex
    template <typename OtherType>
#ifdef ENABLE_GPU
    __device__ __host__ complex_base& operator+=(const OtherType& other) {
#else
    complex_base& operator+=(const OtherType& other) {
#endif
        x = x + other.x;
        y = y + other.y;
        return *this;
    }

    template <typename OtherType>
#ifdef ENABLE_GPU
    __device__ __host__ complex_base& operator-=(const OtherType& other) {
#else
    complex_base& operator-=(const OtherType& other) {
#endif
        x = x - other.x;
        y = y - other.y;
        return *this;
    }

    template <typename OtherType>
#ifdef ENABLE_GPU
    __device__ __host__ complex_base& operator*=(const OtherType& other) {
#else
    complex_base& operator*=(const OtherType& other) {
#endif
        auto saved_x = x;
        x            = x * other.x - y * other.y;
        y            = saved_x * other.y + y * other.x;
        return *this;
    }

    // Data members (public for compatibility with mathdx pattern)
    value_type x, y;
};

/**
 * @brief Primary complex template with proper alignment
 */
template <typename T>
struct alignas(2 * sizeof(T)) complex : complex_base<T> {
  private:
    using base_type = complex_base<T>;

  public:
    using value_type = T;

    complex( )              = default;
    complex(const complex&) = default;
    complex(complex&&)      = default;

#ifdef ENABLE_GPU
    __device__ __host__ constexpr complex(T re, T im) : base_type(re, im) {}
#else
    constexpr complex(T re, T im) : base_type(re, im) {}
#endif

    // Explicit conversion from other complex types
    template <typename K>
#ifdef ENABLE_GPU
    __device__ __host__ explicit constexpr complex(const complex<K>& other)
#else
    explicit constexpr complex(const complex<K>& other)
#endif
        : complex(T(other.real( )), T(other.imag( ))) {
        // Type conversions are explicit to prevent unintended precision loss
    }

    // Use base class operators
    using base_type::operator+=;
    using base_type::operator-=;
    using base_type::operator*=;
    using base_type::operator/=;
    using base_type::operator=;

    complex& operator=(const complex&) = default;
    complex& operator=(complex&&)      = default;
};

// Type trait to determine which types can use default operators
template <typename T>
struct use_default_operator {
    static constexpr bool value = std::is_same_v<T, float> ||
                                  std::is_same_v<T, double>;
    // GPU types will be added in Phase 5
};

// Binary operators for types with default operators
template <typename T, typename = typename std::enable_if_t<use_default_operator<T>::value>>
#ifdef ENABLE_GPU
__device__ __host__ inline complex<T> operator*(const complex<T>& a, const complex<T>& b) {
#else
inline complex<T> operator*(const complex<T>& a, const complex<T>& b) {
#endif
    return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}

template <typename T, typename = typename std::enable_if_t<use_default_operator<T>::value>>
#ifdef ENABLE_GPU
__device__ __host__ inline complex<T> operator+(const complex<T>& a, const complex<T>& b) {
#else
inline complex<T> operator+(const complex<T>& a, const complex<T>& b) {
#endif
    return {a.x + b.x, a.y + b.y};
}

template <typename T, typename = typename std::enable_if_t<use_default_operator<T>::value>>
#ifdef ENABLE_GPU
__device__ __host__ inline complex<T> operator-(const complex<T>& a, const complex<T>& b) {
#else
inline complex<T> operator-(const complex<T>& a, const complex<T>& b) {
#endif
    return {a.x - b.x, a.y - b.y};
}

template <typename T, typename = typename std::enable_if_t<use_default_operator<T>::value>>
#ifdef ENABLE_GPU
__device__ __host__ inline complex<T> operator/(const complex<T>& a, const complex<T>& b) {
#else
inline complex<T> operator/(const complex<T>& a, const complex<T>& b) {
#endif
    T denom = b.x * b.x + b.y * b.y;
    return {(a.x * b.x + a.y * b.y) / denom,
            (a.y * b.x - a.x * b.y) / denom};
}

template <typename T, typename = typename std::enable_if_t<use_default_operator<T>::value>>
#ifdef ENABLE_GPU
__device__ __host__ inline bool operator==(const complex<T>& a, const complex<T>& b) {
#else
inline bool operator==(const complex<T>& a, const complex<T>& b) {
#endif
    return (a.x == b.x) && (a.y == b.y);
}

template <typename T, typename = typename std::enable_if_t<use_default_operator<T>::value>>
#ifdef ENABLE_GPU
__device__ __host__ inline bool operator!=(const complex<T>& a, const complex<T>& b) {
#else
inline bool operator!=(const complex<T>& a, const complex<T>& b) {
#endif
    return ! (a == b);
}

} // namespace detail

// Public cistem::tensor::complex<T> type
// Note: Use cistem::tensor::complex, NOT std::complex, to avoid conflicts
template <typename T>
using complex = detail::complex<T>;

// Common complex number functions
template <typename T>
#ifdef ENABLE_GPU
__device__ __host__ inline T abs(const complex<T>& z) {
#else
inline T abs(const complex<T>& z) {
#endif
    return std::sqrt(z.real( ) * z.real( ) + z.imag( ) * z.imag( ));
}

template <typename T>
#ifdef ENABLE_GPU
__device__ __host__ inline T norm(const complex<T>& z) {
#else
inline T norm(const complex<T>& z) {
#endif
    return z.real( ) * z.real( ) + z.imag( ) * z.imag( );
}

template <typename T>
#ifdef ENABLE_GPU
__device__ __host__ inline complex<T> conj(const complex<T>& z) {
#else
inline complex<T> conj(const complex<T>& z) {
#endif
    return complex<T>(z.real( ), -z.imag( ));
}

template <typename T>
#ifdef ENABLE_GPU
__device__ __host__ inline T arg(const complex<T>& z) {
#else
inline T arg(const complex<T>& z) {
#endif
    return std::atan2(z.imag( ), z.real( ));
}

} // namespace tensor
} // namespace cistem

#endif // _SRC_CORE_TENSOR_COMPLEX_TYPES_H_
