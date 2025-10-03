# Tensor System Development Summary

**Last Updated:** 2025-10-03
**Current Phase:** Phase 3 - FFT Operations (Planning)
**Overall Status:** Foundation Complete, FFT Architecture Design In Progress

This is the living summary document for Tensor system development. It distills key information from detailed planning documents and tracks overall progress.

---

## Quick Reference

| Document | Purpose | Location |
|----------|---------|----------|
| **This file** | High-level summary, progress tracking | `src/core/tensor/DEVELOPMENT_SUMMARY.md` |
| **User Guide** | How to use Tensor system | `src/core/tensor/CLAUDE.md` |
| **Code Walkthrough** | Systematic code tour with file references | `.claude/cache/tensor_code_walkthrough.md` |
| **FFT Plan** | Phase 3 implementation plan | `src/core/tensor/PHASE3_FFT_PLAN.md` |
| **Design Notes** | Detailed technical decisions | `.claude/cache/tensor_design_notes.md` |

---

## Project Vision

**Goal:** Replace Legacy Image class with modern C++ Tensor system that:
- Separates memory ownership from data views
- Provides zero-overhead abstractions (templates + inlining)
- Supports multiple memory layouts (dense, FFTW-padded, etc.)
- Enables future GPU and mixed-precision support
- Maintains full backward compatibility during transition

**Not a Goal:** Complete rewrite of all algorithms. Focus on infrastructure that allows gradual migration.

---

## Architecture Overview

```
User Code
    ↓
Tensor<float, FFTWPaddedLayout>  ← Non-owning view with metadata
    ↓
AddressCalculator<FFTWPaddedLayout>  ← Inlined, zero overhead
    ↓
FFTWPaddedLayout::CalculatePitch()  ← Compile-time layout policy
    ↓
TensorMemoryPool<float>  ← Owns memory + FFT plans
```

**Key Design Principles:**
1. **Separation of concerns:** Memory ownership (pool) vs data views (tensor) vs operations (free functions)
2. **Zero runtime overhead:** Templates + inlining + LTO
3. **Type safety:** Layouts can't mix accidentally
4. **Proper ownership:** FFT plans tied to memory addresses (whoever owns memory owns plans)

---

## Current Status by Phase

### ✅ Phase 1: Foundation (COMPLETE)

**Deliverables:**
- ✅ Type traits system (`type_traits.h`)
- ✅ Complex type (`complex_types.h`) - GPU-compatible, aligned for SIMD
- ✅ Memory layouts (`memory_layout.h`) - DenseLayout, FFTWPaddedLayout
- ✅ Address calculator (`address_calculator.h`) - Zero-overhead indexing
- ✅ Tensor class (`tensor.h`) - Non-owning view with metadata
- ✅ Memory pool (`memory_pool.h`) - Owns buffers, tracks allocations
- ✅ Space enum (Position/Momentum) - Decouples type from transform space

**Status:** Foundation is solid, no known issues.

**Key Files:**
- `src/core/tensor/type_traits.h` - Type constraints (currently float only)
- `src/core/tensor/complex_types.h` - cistem::complex<T> with GPU support
- `src/core/tensor/memory/memory_layout.h` - Layout policies
- `src/core/tensor/memory/address_calculator.h` - Indexing logic
- `src/core/tensor/core/tensor.h` - Main Tensor class
- `src/core/tensor/memory/memory_pool.h` - Memory ownership

### ✅ Phase 2: Core Operations (COMPLETE)

**Deliverables:**
- ✅ Arithmetic operations (11 operations) - `operations/arithmetic.h`
- ✅ Statistical operations (7 operations) - `operations/statistics.h`
- ✅ Iterator infrastructure - `iterators/tensor_iterator.h`
- ✅ STL algorithm integration - All operations use `std::for_each`, `std::transform`, etc.
- ✅ Comprehensive unit tests (35 test cases, 9,169 assertions, 100% pass rate)

**Performance:**
- Single pointer increment (iterator) vs address recalculation (GetValue_p)
- STL algorithms enable compiler optimizations (vectorization, loop unrolling)
- Expected: Match or exceed Legacy Image performance (to be benchmarked)

**Key Files:**
- `src/core/tensor/operations/arithmetic.h` - AddScalar, MultiplyByScalar, Add, etc.
- `src/core/tensor/operations/statistics.h` - Mean, Variance, Sum, etc.
- `src/core/tensor/iterators/tensor_iterator.h` - Dense and FFTWPadded iterators
- `src/test/core/tensor/test_tensor_arithmetic.cpp` - Arithmetic tests
- `src/test/core/tensor/test_tensor_statistics.cpp` - Statistics tests

### 🔄 Phase 3: FFT Operations (IN PROGRESS - Planning)

**Status:** Architecture design phase - **Critical decision point on FastFFT integration**

**Current Focus:** Out-of-place memory allocation (2x buffer pattern)

**Key Insight:** FastFFT allocates 2x memory (buffer_1 + buffer_2) for ping-pong transforms. Tensor should adopt this pattern for classical FFTs (FFTW/cuFFT) regardless of FastFFT integration strategy.

**Open Questions for Developer (FastFFT Maintainer):**
1. Would you add an external workspace API to FastFFT?
2. What memory alignment does FastFFT require?
3. Type compatibility requirements for borrowed pointers?
4. Preferred integration strategy (tight vs loose coupling)?

**Detailed Plan:** See `src/core/tensor/PHASE3_FFT_PLAN.md`

**Next Immediate Task:** Implement Phase 3A.0 - Out-of-Place Memory Support
- Add `data_secondary` buffer to memory pool
- Allocate 2x in `Allocate()` method
- Expose `GetPrimaryBuffer()` / `GetSecondaryBuffer()` in Tensor API

---

## Key Design Decisions

### 1. Memory Ownership Separation (2025-01-02)

**Decision:** Tensor is non-owning view, TensorMemoryPool owns buffers.

**Rationale:**
- Legacy Image violates single responsibility (owns memory + performs operations)
- Particle class workaround duplicated methods (maintainability nightmare)
- Modern C++ best practice: views don't own resources

**Impact:** Clean architecture, easier to reason about lifetimes.

### 2. FFT Plan Ownership (2025-01-02)

**Decision:** FFT plans owned by TensorMemoryPool, stored with buffer.

**Rationale:**
- FFTW/cuFFT plans are tied to specific memory addresses
- Whoever owns the memory must own the plan
- Cannot safely move/reuse plans between buffers

**Implementation:**
```cpp
struct Buffer {
    T* data;
    void* fft_plan_forward;   // Backend-specific, stored as void*
    void* fft_plan_backward;
};
```

### 3. Position vs Momentum Space (2025-01-02)

**Decision:** Decouple scalar type from transform space.

**Old Terminology (confusing):**
- "Real space" (position) conflated with "real type" (float)
- "Complex space" ambiguous (complex-typed OR Fourier space?)

**New Terminology (clear):**
- **Position space:** Spatial domain (formerly "real space")
- **Momentum space:** Fourier/reciprocal domain (formerly "Fourier/complex space")
- **Scalar type:** Completely independent (float, complex<float>, __half, etc.)

**Example:**
```cpp
Tensor<float> img;                     // float-valued position space
Tensor<complex<float>> fourier;        // complex-valued momentum space
Tensor<complex<float>> complex_img;    // complex-valued POSITION space (valid!)
```

### 4. Out-of-Place Memory Support (2025-10-03)

**Decision:** Adopt FastFFT's 2x buffer allocation pattern.

**Rationale:**
- Classical out-of-place FFTs need separate input/output buffers
- FastFFT uses 2x allocation (buffer_1 + buffer_2) for ping-pong transforms
- Matching this pattern enables compatibility regardless of integration strategy
- Enables efficient multi-stage transforms

**Implementation:**
```cpp
struct Buffer {
    T* data;              // Primary buffer (buffer_1)
    T* data_secondary;    // Secondary buffer (buffer_2)
    size_t single_buffer_size;  // Size of EACH buffer
};
```

**Status:** Approved, ready to implement in Phase 3A.0.

### 5. Template Constraints (2025-01-02)

**Decision:** Use `cistem::EnableIf` for C++17 concept-like constraints.

**Rationale:**
- C++20 concepts not available (codebase is C++17)
- Existing `cistem::EnableIf` infrastructure in constants.h
- Provides clear compile-time errors

**Current Constraint:** Phase 1 only supports `float`, will expand in Phase 5.

---

## File Organization

```
src/core/tensor/
├── CLAUDE.md                      # User guide
├── DEVELOPMENT_SUMMARY.md         # This file
├── PHASE3_FFT_PLAN.md            # FFT implementation plan
│
├── type_traits.h                  # Type constraints
├── complex_types.h                # cistem::complex<T>
│
├── core/
│   └── tensor.h                   # Main Tensor class
│
├── memory/
│   ├── memory_layout.h            # DenseLayout, FFTWPaddedLayout
│   ├── address_calculator.h       # Indexing logic
│   └── memory_pool.h              # Buffer ownership
│
├── operations/
│   ├── arithmetic.h               # AddScalar, MultiplyByScalar, etc.
│   └── statistics.h               # Mean, Variance, Sum, etc.
│
└── iterators/
    └── tensor_iterator.h          # Dense and FFTWPadded iterators
```

---

## Build Integration

**Compilation:**
- Tensor tests only build with GPU builds (`-DENABLEGPU` flag)
- Integrated into `unit_test_runner`
- Uses Intel MKL for FFT operations

**Makefile:**
- `src/Makefile.am` includes tensor test files
- Updates: October 3, 2025

**Last Successful Build:** October 3, 2025

---

## Testing Status

**Unit Test Summary:**
- **Test Cases:** 35
- **Assertions:** 9,169
- **Pass Rate:** 100%
- **Last Run:** 2025-10-03

**Test Coverage:**
- ✅ Type traits and constraints
- ✅ Complex type operations
- ✅ Memory layouts (Dense, FFTWPadded)
- ✅ Address calculation
- ✅ Iterator functionality (Dense and FFTWPadded)
- ✅ Arithmetic operations (11 operations)
- ✅ Statistical operations (7 operations)
- ⏳ FFT operations (Phase 3)
- ⏳ I/O operations (Future)

**Known Issues:**
- Intel compiler warnings on iterator methods (false positives, harmless)
- No functional issues

---

## Performance Characteristics

### Achieved (Phase 2)

**Iterator Efficiency:**
- Single pointer increment vs address recalculation
- Zero overhead through inlining
- FFTWPadded iterator automatically skips padding (no manual tracking)

**STL Algorithm Benefits:**
- Compiler can vectorize and unroll loops
- Better optimization on modern architectures
- More expressive code

### To Be Measured (Future)

**Benchmarks planned:**
- AddScalar on 512×512 image
- Element-wise Add on two 512×512 images
- Mean calculation on 512×512×512 volume
- Variance on 512×512×512 volume

**Target:** Match or exceed Legacy Image performance (within 5%)

---

## Migration Strategy

### Phase-by-Phase Replacement

**Not doing:** Complete rewrite of all algorithms at once.

**Doing:** Gradual migration as refactoring opportunities arise.

**Pattern:**
1. Identify Image method needing work
2. Check if Tensor equivalent exists
3. If not, implement in Tensor
4. Test thoroughly (compare with Image)
5. Migrate code to use Tensor
6. Mark Image method as deprecated

**Compatibility During Transition:**
- Legacy Image continues to work
- New code can use Tensor
- No breaking changes to existing APIs
- Gradual migration over time

---

## Future Phases (Tentative)

### Phase 4: I/O Operations
- MRC file read/write
- Image format conversions
- Memory-mapped I/O for large datasets

### Phase 5: Multi-Precision Support
- `complex<float>` - Complex-valued images
- `__half` - FP16 for GPU acceleration
- `__nv_bfloat16` - Brain float for ML workflows
- Type conversion utilities

### Phase 6: Advanced Operations
- Geometric transformations (rotation, translation, etc.)
- Filtering operations
- Convolution/correlation
- Image alignment

### Phase 7: GPU Acceleration
- CUDA kernels for core operations
- cuFFT integration (or FastFFT if tight integration chosen)
- Stream management
- Multi-GPU support

### Phase 8: Optimization & Hardening
- Performance benchmarking vs Legacy Image
- SIMD optimization
- Cache optimization
- Production hardening

---

## Lessons Learned

### Architecture

1. **Separation of concerns pays off** - Memory pool handling ownership separately from Tensor views makes reasoning about lifetimes much easier.

2. **Type constraints are crucial** - Using EnableIf to restrict types at compile time prevents confusing errors deep in template instantiation.

3. **Layout policies work** - Template-based layout selection gives zero overhead while maintaining flexibility.

### Implementation

1. **int3 vs int4 confusion** - Tensor stores `int4 dims_p_` (with pitch as .w), but iterators expect `int3` logical dimensions. Solution: Extract logical dims before passing to iterator constructor.

2. **Static assert for coverage** - `if constexpr` without `else` can cause missing return warnings. Solution: Add `else { static_assert(...); }` to catch unsupported types at compile time.

3. **STL algorithms are a win** - Even simple operations benefit: `std::fill` is clearer than a loop, `std::minmax_element` is more efficient than manual min/max tracking.

4. **Plan caching is critical** - FFT plan creation is expensive. Lazy creation with caching (GpuImage pattern) is essential for performance.

### Testing

1. **Buffer allocation with padding** - FFTWPaddedLayout requires explicit padded size calculation. Tests must call `AllocateBuffer({x,y,z}, false, padded_size)` not just `AllocateBuffer({x,y,z})`.

2. **Comprehensive assertions matter** - 9,169 assertions across 35 test cases caught numerous edge cases during development.

---

## Active Decisions & Blockers

### ⏳ Pending: FastFFT Integration Strategy

**Decision Needed:** How to integrate with FastFFT?

**Options:**
1. **Tight integration:** Add external workspace API to FastFFT (zero duplication, clean ownership)
2. **Loose coupling:** Compatible layouts, separate management (simpler, some duplication)
3. **Hybrid:** Flexible usage based on use case

**Blocker:** Awaiting developer decision (FastFFT maintainer)

**Impact:** Affects Phase 3A.2 implementation plan

**Next Step:** Developer to review `PHASE3_FFT_PLAN.md` and provide feedback on integration options

---

## Recent Changes

### 2025-10-03
- **Phase 3 Planning:** Deep analysis of FastFFT memory architecture
- **Discovered:** Out-of-place memory support needed (2x buffer allocation)
- **Created:** `PHASE3_FFT_PLAN.md` with detailed FFT implementation plan
- **Created:** This summary document (`DEVELOPMENT_SUMMARY.md`)
- **Decision:** Implement out-of-place support first (Phase 3A.0) regardless of FastFFT integration strategy

### 2025-10-03 (Earlier)
- **Phase 2.5 Complete:** All operations refactored to use STL algorithms
- **Test Status:** 35 test cases, 9,169 assertions, 100% pass rate
- **Performance:** Iterator-based operations with zero overhead

### 2025-10-02
- **Phase 2 Complete:** Core operations implemented
- **Iterators:** Dense and FFTWPadded iterators working
- **Build:** Integrated into unit_test_runner

---

## Contact & References

**Primary Documents:**
- User guide: `src/core/tensor/CLAUDE.md`
- Code walkthrough: `.claude/cache/tensor_code_walkthrough.md`
- FFT plan: `src/core/tensor/PHASE3_FFT_PLAN.md`
- Design notes: `.claude/cache/tensor_design_notes.md`

**Legacy Code Reference:**
- Image class: `src/core/image.h`, `src/core/image.cpp`
- Particle class: Shows problems with Image API

**External Dependencies:**
- FastFFT: `/sa_shared/git/FastFFT/`
- Intel MKL: System library
- CUDA: Optional, for GPU support

---

**Note:** This document is updated as phases complete and decisions are made. When updating, increment the "Last Updated" date and add entries to "Recent Changes" section.
