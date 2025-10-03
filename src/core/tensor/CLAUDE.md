# Tensor System Development Guidelines

**IMPORTANT: Read this entire file completely and carefully before working on Tensor system code. Every section contains critical information for successful implementation.**

## Overview

The Tensor system is a complete refactoring of the legacy `Image` class, designed to address fundamental architectural limitations while maintaining performance. This is a large-scale, multi-month effort requiring systematic, incremental implementation.

### Design Philosophy

**Separation of Concerns**:
- **Memory ownership**: `TensorMemoryPool` owns buffers and FFT plans
- **Data views**: `Tensor` is a non-owning view over memory with metadata
- **Operations**: Standalone functions/methods operating on Tensors

**Type System Clarity**:
- **Scalar type** (float, complex&lt;float&gt;, __half) is independent of **transform space** (Position, Momentum)
- Legacy Image conflated "real space" (position) with "real type" (float), causing confusion
- New system: Position space images can be complex-valued; Momentum space uses cistem::complex&lt;T&gt;

**Performance Requirements**:
- Must match or exceed legacy Image class performance (within 5% threshold)
- Critical paths (addressing, element access) must be inlined
- Link-Time Optimization (LTO) required for release builds
- Benchmark every phase against legacy implementation

## Current Status: Phase 1 - Foundation [COMPLETED]

**Status**: Phase 1 infrastructure complete and tested (January 2025)

**Completed Deliverables**:
1. ✅ `cistem::tensor::complex<T>` type system (foundation for future phases)
2. ✅ Type traits with Phase 1 constraints (`is_phase1_supported_v`)
3. ✅ `TensorMemoryPool<float>` with FFT plan ownership
4. ✅ `Tensor<float, DenseLayout>` and `Tensor<float, FFTWPaddedLayout>`
5. ✅ Inline address calculation (`AddressCalculator`)
6. ✅ Template-based debug utilities (`TENSOR_DEBUG_ASSERT`)
7. ✅ Comprehensive unit tests (334 assertions, 27 test cases, 100% pass)
8. ✅ Build system integration (libgpucore with `-DENABLEGPU`)

**Test Results**:
- TensorMemoryPool: 52 assertions in 8 test cases ✅
- Tensor class: 242 assertions in 8 test cases ✅
- Memory layouts: 19 assertions in 6 test cases ✅
- AddressCalculator: 21 assertions in 5 test cases ✅

**Key Features**:
- STL containers (std::vector, std::mutex) instead of wxWidgets
- Template-based assertions (no macro overhead in release builds)
- Independent of core_headers (minimal dependencies)
- CUDA int3/int4 vector types throughout
- Space enum decoupling scalar type from transform space

**Next**: Phase 2 - Core operations (arithmetic, statistics, FFT execution)

## Core Architecture

### Namespace Organization

```cpp
namespace cistem {
namespace tensor {
    // All Tensor system types live here
    template<typename T> class complex;
    template<typename T, typename Layout> class Tensor;
    template<typename T> class TensorMemoryPool;
}
}
```

### Type System

#### cistem::complex&lt;T&gt; (complex_types.h)

Inspired by NVIDIA mathdx, provides:
- Template-based complex type for any scalar T
- Specializations for GPU types (__half, nv_bfloat16) when available
- Proper alignment for vectorization
- Host/Device decorators for CUDA compatibility

**Key Design Points**:
- Use `alignas(2 * sizeof(T))` for proper alignment
- Provide `real()` and `imag()` accessors
- Implement operators (+, -, *, /) with SFINAE for appropriate types
- Separate specializations for __half, __nv_bfloat16 when `ENABLE_GPU` defined

#### Type Traits (type_traits.h)

Use `cistem::EnableIf` (from constants.h) for C++17 concept-like constraints:

```cpp
template<typename T>
struct is_real_scalar : std::bool_constant<
    std::is_same_v<T, float> ||
    std::is_same_v<T, double>
    // Add __half, __nv_bfloat16 in later phases
> {};

template<typename T>
struct is_complex_scalar { /* ... */ };

// Phase 1: Only float allowed
template<typename T>
constexpr bool is_phase1_supported_v = std::is_same_v<T, float>;
```

**Usage in Templates**:
```cpp
template<typename T, cistem::EnableIf<is_phase1_supported_v<T>, int> = 0>
class Tensor {
    // Implementation
};
```

### Memory Management

#### TensorMemoryPool&lt;T&gt; (memory/tensor_memory_pool.h/.cpp)

**Responsibilities**:
- Allocate/deallocate memory buffers
- Own FFT plans (plans are tied to specific memory addresses)
- Track allocations for leak detection
- Provide page-locked memory for GPU transfers (when `ENABLE_GPU`)

**Key Design**:
```cpp
template<typename ScalarType>
class TensorMemoryPool {
public:
    struct Buffer {
        ScalarType* data;
        size_t size;
        void* fft_plan_forward;  // Backend-specific FFT plan
        void* fft_plan_backward; // Backend-specific FFT plan
    };

    Buffer AllocateBuffer(size_t num_elements, bool create_fft_plans = false);
    void DeallocateBuffer(Buffer& buffer);

private:
    std::vector<Buffer> active_buffers_;
};
```

**Rationale**: FFT plans are associated with specific memory addresses (FFTW requirement), so whoever owns the memory must own the plans.

#### Memory Layouts (memory/memory_layout.h)

**DenseLayout**: Contiguous memory with no padding
- Use for: Out-of-place FFTs, general storage
- Pitch: `logical_x_dimension`

**FFTWPaddedLayout**: In-place FFT with padding
- Use for: In-place real-to-complex FFTs (legacy Image compatibility)
- Pitch: `logical_x_dimension + padding_jump_value`
- `padding_jump_value = (x_dim % 2 == 0) ? 2 : 1`

**Why Template Policy**:
- Compile-time selection of address calculation
- Zero runtime overhead
- Type safety (can't accidentally mix layouts)

### Tensor View

#### Tensor&lt;T, Layout&gt; (core/tensor.h)

**Non-owning view** over memory with metadata:

```cpp
template<typename ScalarType, typename Layout = DenseLayout>
class Tensor {
    static_assert(is_numeric_v<ScalarType>, "Must be numeric type");
    static_assert(is_phase1_supported_v<ScalarType>, "Phase 1: float only");

public:
    enum class Space { Position, Momentum };

    // Attach to buffer from pool (does NOT take ownership)
    void AttachToBuffer(ScalarType* data, int3 dimensions);

    // Space state (independent of scalar type!)
    Space GetSpace() const;
    void SetSpace(Space space);

    // Element access
    ScalarType& operator()(int x, int y, int z);
    const ScalarType& operator()(int x, int y, int z) const;

    // Raw access
    ScalarType* Data();
    const ScalarType* Data() const;

private:
    ScalarType* data_;  // Non-owning pointer
    int3 dims_;
    Space space_;
    // Centering, addressing metadata
};
```

**Critical**: Tensor does NOT allocate or deallocate. It's a view, like std::span.

### Addressing System

#### AddressCalculator (addressing/address_calculator.h)

**Header-only, templated on Layout** for zero-overhead addressing:

```cpp
template<typename Layout>
class AddressCalculator {
public:
    static inline long Real1DAddress(int x, int y, int z, int3 dims) {
        long pitch = Layout::CalculatePitch(dims.x);
        return (pitch * dims.y) * long(z) + pitch * long(y) + long(x);
    }

    static inline long Fourier1DAddress(int x, int y, int z, int3 physical_bounds) {
        // Similar for Fourier/complex addressing
    }
};
```

**Performance Critical**: This will be called millions of times. MUST be inlined. Verify with:
```bash
g++ -S -O3 -flto file.cpp
# Check assembly for inlining
```

#### CoordinateMapper (addressing/index_mapper.h)

Handles Physical ↔ Logical coordinate conversions, Hermitian symmetry, centering metadata.

### FFT System

#### FFTPlan (fft/fft_plan.h)

Wrapper around backend-specific plans (FFTW, MKL, cuFFT):

```cpp
template<typename Backend = FFTWBackend>
class FFTPlan {
    typename Backend::PlanType plan_forward_;
    typename Backend::PlanType plan_backward_;

public:
    template<typename T, typename Layout>
    void CreatePlans(T* data, int3 dims);

    void DestroyPlans();

    template<typename T>
    void ExecuteForward(T* data);

    template<typename T>
    void ExecuteBackward(T* data);
};
```

**Note**: Plans stored in `TensorMemoryPool::Buffer`, managed by pool.

## Performance Requirements

### Benchmarking

Every completed phase MUST include benchmarks comparing to legacy Image:

```cpp
// In unit tests or separate benchmark suite
void BenchmarkFFT_Image_512x512() {
    Image img(512, 512, 1);
    auto start = high_resolution_clock::now();
    for (int i = 0; i < 1000; i++) {
        img.ForwardFFT();
        img.BackwardFFT();
    }
    auto end = high_resolution_clock::now();
    return duration_cast<milliseconds>(end - start).count();
}

void BenchmarkFFT_Tensor_512x512() {
    TensorMemoryPool<float> pool;
    auto buffer = pool.AllocateBuffer(512 * 512, true);
    Tensor<float, FFTWPaddedLayout> tensor;
    tensor.AttachToBuffer(buffer.data, {512, 512, 1});

    auto start = high_resolution_clock::now();
    for (int i = 0; i < 1000; i++) {
        FFTForward(tensor, buffer.fft_plan_forward);
        FFTBackward(tensor, buffer.fft_plan_backward);
    }
    auto end = high_resolution_clock::now();
    return duration_cast<milliseconds>(end - start).count();
}

TEST_CASE("Tensor FFT performance") {
    auto image_time = BenchmarkFFT_Image_512x512();
    auto tensor_time = BenchmarkFFT_Tensor_512x512();

    // Must be within 5% of Image performance
    REQUIRE(tensor_time < image_time * 1.05);
}
```

### Optimization Checklist

- [ ] All address calculations inlined (verify assembly)
- [ ] No unnecessary copies (use move semantics)
- [ ] Link-Time Optimization enabled for release (`-flto`)
- [ ] Profile hot paths with `perf` or VTune
- [ ] Compare cache behavior (use `perf stat -e cache-misses`)
- [ ] Benchmark every operation against legacy Image

## Compile-Time Design Patterns

The Tensor system uses advanced C++ template metaprogramming patterns inspired by NVIDIA's mathdx library. These patterns enable zero-cost abstractions while maintaining type safety and clean architecture.

**Detailed Research**: See `.claude/cache/nvidia_mathdx_patterns.md` for comprehensive analysis of NVIDIA's compile-time patterns and how they apply to cisTEM.

### Avoiding Circular Dependencies

**The Problem**: Template-heavy code can easily create circular include dependencies:
```
complex_types.h → type_traits.h → complex_types.h  // ERROR
```

**NVIDIA's Solution**: **Inline critical traits** directly in the header that needs them.

#### When to Inline Traits

**Inline a trait if**:
- It's needed by operators/methods in the SAME header
- It's small (< 20 lines)
- The duplication cost is lower than dependency complexity

**Keep trait in type_traits.h if**:
- It's used by multiple unrelated headers
- It defines complex metaprogramming logic
- It's about relationships between multiple types

#### Example: supports_default_ops

```cpp
// complex_types.h - Inline the critical trait
namespace detail {
    template<typename T>
    struct supports_default_ops : std::bool_constant<
        std::is_same_v<T, float> ||
        std::is_same_v<T, double>
    > {};

    template<typename T>
    inline constexpr bool supports_default_ops_v = supports_default_ops<T>::value;

    // Operators use inline trait - no external dependency
    template <typename T, typename = cistem::EnableIf<supports_default_ops_v<T>>>
    cisTEM_DEVICE_HOST inline complex<T> operator*(
        const complex<T>& a, const complex<T>& b) {
        return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
    }
}
```

```cpp
// type_traits.h - Can include complex_types.h safely
#include "complex_types.h"

// Optionally re-export for convenience
template<typename T>
using supports_default_ops = detail::supports_default_ops<T>;

template<typename T>
inline constexpr bool supports_default_ops_v = detail::supports_default_ops_v<T>;

// Define OTHER traits here (not needed by complex operators)
template<typename T>
struct is_real_scalar { /* ... */ };
```

**Benefits**:
- ✅ No circular dependency
- ✅ `complex_types.h` is self-contained
- ✅ Minimal duplication (just trait definition)
- ✅ Follows proven production pattern

### Template Performance Patterns

#### GPU-Optimized Integer Types

GPUs perform significantly better with 32-bit integer arithmetic than 64-bit. Use templated return types with sensible defaults:

```cpp
template<typename Layout>
class AddressCalculator {
public:
    // Default to long (safe for large images), allow int for GPU performance
    template<typename IndexType = long>
    static inline IndexType Real1DAddress(int x, int y, int z, int3 dims) {
        size_t pitch = Layout::CalculatePitch(dims);
        if constexpr (std::is_same_v<IndexType, long>) {
            // 64-bit path (safe default)
            return long(pitch * dims.y) * long(z) + long(pitch) * long(y) + long(x);
        } else {
            // 32-bit path (GPU optimization)
            return pitch * dims.y * z + pitch * y + x;
        }
    }
};
```

**Usage**:
```cpp
// CPU code - safe default
auto addr = AddressCalculator<DenseLayout>::Real1DAddress(x, y, z, dims);

// GPU kernel - explicit int for performance
__global__ void kernel() {
    auto addr = AddressCalculator<DenseLayout>::Real1DAddress<int>(x, y, z, dims);
}
```

**Rationale**: User must explicitly opt-in to `int` (via `Real1DAddress<int>()`), so overflow risk is their responsibility. Default `long` ensures safety.

### Future Patterns (Not Yet Implemented)

#### Expression Hierarchies

For future descriptor systems (e.g., TensorDescriptor), consider NVIDIA's expression hierarchy pattern:

```cpp
// Base expression types (defines "kind" of descriptor)
struct descriptor_expression {};
struct layout_descriptor : descriptor_expression {};
struct space_descriptor : descriptor_expression {};

// Concrete descriptors
template<typename Layout>
struct LayoutPolicy : layout_descriptor {
    using type = Layout;
};

// Compile-time descriptor assembly
template<typename... Descriptors>
class TensorDescriptor {
    // Extract specific descriptor with defaults
    using layout = get_or_default_t<layout_descriptor, Descriptors..., DefaultLayout>;
    using space = get_or_default_t<space_descriptor, Descriptors..., PositionSpace>;
};
```

This enables flexible, compile-time configuration: `Tensor<ScalarType, LayoutPolicy<Dense>, PositionSpace>`.

**When to use**: Phase 3+, when we need flexible compile-time Tensor configuration.

### Key Takeaways

1. **Don't over-factor** - Small inline duplication beats complex dependencies
2. **Template for GPU performance** - int vs long matters on GPUs
3. **Default to safety** - Use `long` default, allow `int` opt-in
4. **Learn from production code** - NVIDIA's patterns are battle-tested
5. **Document decisions** - Explain WHY a trait is inlined vs separated

## Code Style

Follow cisTEM conventions from root CLAUDE.md, with Tensor-specific additions:

### Template Formatting
```cpp
// Template parameters on same line if they fit
template<typename T, typename Layout = DenseLayout>
class Tensor { /* ... */ };

// Multi-line if complex
template<typename ScalarType,
         typename Layout,
         typename = cistem::EnableIf<is_numeric_v<ScalarType>>>
class AdvancedTensor { /* ... */ };
```

### Naming Conventions
- **Classes**: PascalCase (`Tensor`, `TensorMemoryPool`)
- **Functions**: PascalCase for methods (`AttachToBuffer`), camelCase for free functions
- **Template parameters**: PascalCase (`ScalarType`, `Layout`)
- **Member variables**: snake_case with trailing underscore (`data_`, `dims_`)

### Header Guards
Use full path from project root:
```cpp
#ifndef _SRC_CORE_TENSOR_CORE_TENSOR_H_
#define _SRC_CORE_TENSOR_CORE_TENSOR_H_
// ...
#endif
```

## Testing Strategy

### Unit Tests
Each component gets comprehensive unit tests in `src/core/tensor/tests/`:

```cpp
TEST_CASE("TensorMemoryPool allocates and tracks buffers") {
    TensorMemoryPool<float> pool;
    auto buffer = pool.AllocateBuffer(1000);

    REQUIRE(buffer.data != nullptr);
    REQUIRE(buffer.size == 1000);

    // Write and read
    buffer.data[0] = 42.0f;
    REQUIRE(buffer.data[0] == 42.0f);

    pool.DeallocateBuffer(buffer);
    REQUIRE(buffer.data == nullptr);
}

TEST_CASE("Tensor addressing matches legacy Image") {
    Image legacy_img(64, 64, 1);

    TensorMemoryPool<float> pool;
    auto buffer = pool.AllocateBuffer(64 * 64);
    Tensor<float, DenseLayout> tensor;
    tensor.AttachToBuffer(buffer.data, {64, 64, 1});

    // Fill with same pattern
    for (int z = 0; z < 1; z++) {
        for (int y = 0; y < 64; y++) {
            for (int x = 0; x < 64; x++) {
                float value = float(x + y * 64);
                legacy_img.real_values[legacy_img.ReturnReal1DAddressFromPhysicalCoord(x, y, z)] = value;
                tensor(x, y, z) = value;
            }
        }
    }

    // Verify addressing matches
    for (int z = 0; z < 1; z++) {
        for (int y = 0; y < 64; y++) {
            for (int x = 0; x < 64; x++) {
                float legacy_val = legacy_img.ReturnRealPixelFromPhysicalCoord(x, y, z);
                float tensor_val = tensor(x, y, z);
                REQUIRE(legacy_val == tensor_val);
            }
        }
    }
}
```

### Memory Leak Detection
Run valgrind on all tests:
```bash
valgrind --leak-check=full --show-leak-kinds=all ./tensor_tests
# REQUIRE: "All heap blocks were freed -- no leaks are possible"
```

### Numerical Accuracy
For operations that should be bit-identical to Image class:
```cpp
REQUIRE(tensor_result == image_result);  // Exact match
```

For operations with floating-point rounding:
```cpp
REQUIRE(std::abs(tensor_result - image_result) < 1e-6f);
```

## Common Patterns

### Creating and Using a Tensor

```cpp
// Create memory pool
TensorMemoryPool<float> pool;

// Allocate buffer with FFT plans
auto buffer = pool.AllocateBuffer(512 * 512 * 512, true);

// Create non-owning Tensor view
Tensor<float, FFTWPaddedLayout> volume;
volume.AttachToBuffer(buffer.data, {512, 512, 512});
volume.SetSpace(Tensor<float>::Space::Position);

// Use tensor
for (int z = 0; z < 512; z++) {
    for (int y = 0; y < 512; y++) {
        for (int x = 0; x < 512; x++) {
            volume(x, y, z) = SomeFunction(x, y, z);
        }
    }
}

// Transform to momentum space
FFTForward(volume, buffer.fft_plan_forward);
volume.SetSpace(Tensor<float>::Space::Momentum);

// Clean up (pool owns buffer and plans)
pool.DeallocateBuffer(buffer);
```

### Multiple Views of Same Data

```cpp
auto buffer = pool.AllocateBuffer(1024 * 1024);

Tensor<float> view1, view2;
view1.AttachToBuffer(buffer.data, {1024, 1024, 1});
view2.AttachToBuffer(buffer.data, {1024, 1024, 1});

// Both views see same data
view1(0, 0, 0) = 42.0f;
REQUIRE(view2(0, 0, 0) == 42.0f);
```

## Migration from Legacy Image

### DO NOT:
- Copy-paste Image code without understanding it
- Mix Image and Tensor in same algorithm (except during testing)
- Bypass memory pool to directly allocate buffers
- Use Tensor after deallocating its buffer

### DO:
- Read corresponding Image implementation for algorithm understanding
- Benchmark new Tensor implementation against Image
- Document any algorithmic changes or optimizations
- Test numerical accuracy against Image results
- Keep temporal debugging changes marked with `// revert -`

## Future Phases (Not Yet Implemented)

**Phase 2**: Core operations (arithmetic, statistics, FFT operations)

**Phase 3**: I/O and geometric transformations

**Phase 4**: Cryo-EM specific operations (CTF, filtering, FSC)

**Phase 5**: Multi-precision support (complex&lt;float&gt;, __half, etc.)

## Questions and Clarifications

When you encounter ambiguity or need design decisions:

1. Check this CLAUDE.md first
2. Check root `/workspaces/cisTEM/CLAUDE.md` for general cisTEM patterns
3. Check `.claude/cache/tensor_design_notes.md` for detailed technical decisions
4. Consult with the user for architectural decisions
5. Document new decisions in appropriate CLAUDE.md file

## References

- **Legacy Image class**: `src/core/image.h`, `src/core/image.cpp`
- **GpuImage patterns**: `src/gpu/GpuImage.h`, `src/gpu/GpuImage.cu`
- **NVIDIA mathdx complex types**: `/sa_shared/git/FastFFT/include/nvidia-srcs/.../commondx/complex_types.hpp`
- **cistem::EnableIf**: `src/constants/constants.h:27`
- **cisTEM general guidelines**: `/workspaces/cisTEM/CLAUDE.md`
