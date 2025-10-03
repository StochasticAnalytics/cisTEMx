# Tensor Phase 3: FFT Implementation Plan

**Last Updated:** 2025-10-03
**Status:** Planning - Architecture Design Phase
**Critical Decision Point:** Out-of-place memory allocation & FastFFT integration strategy

---

## Executive Summary

Phase 3 will add FFT operations to the Tensor system. Deep analysis of FastFFT's memory architecture reveals that proper **out-of-place transform support** is foundational and must be implemented first. This document explores integration strategies with FastFFT, which the developer maintains and can potentially modify.

**Key Insight:** FastFFT allocates 2x memory (buffer_1 + buffer_2) for ping-pong transforms. Tensor should adopt this pattern for classical FFTs (FFTW/cuFFT) regardless of FastFFT integration strategy.

---

## FastFFT Memory Architecture Analysis

### 1. Internal Buffer Management

**Source:** `/sa_shared/git/FastFFT/src/fastfft/FastFFT.cu:289-305`

```cpp
void AllocateBufferMemory() {
    constexpr size_t compute_memory_scalar = 2;  // Allocate 2x needed

    cudaMallocAsync(&d_ptr.buffer_1,
                    2 * compute_memory_wanted_ * sizeof(ComputeBaseType),
                    cudaStreamPerThread);

    // buffer_2 points to the second half
    d_ptr.buffer_2 = &d_ptr.buffer_1[compute_memory_wanted_ / 2];
}
```

**Characteristics:**
- **Owns its buffers:** Allocated lazily after BOTH forward and inverse plans set
- **Size calculation:** `max(fwd_input, fwd_output, inv_input, inv_output)` - one size fits all
- **Ping-pong pattern:** buffer_1 ↔ buffer_2 for multi-stage 2D/3D transforms
- **Lifetime:** Entire `FourierTransformer` object lifetime
- **No per-call allocation:** Workspace persists, avoiding expensive alloc/free

### 2. External Pointer Borrowing Pattern

**Source:** `/sa_shared/git/FastFFT/src/fastfft/FastFFT.cu:355-402`

```cpp
void FwdImageInvFFT(PositionSpaceType* input_ptr,    // USER owns
                    OtherImageType* image_to_search,  // USER owns
                    PositionSpaceType* output_ptr,    // USER owns
                    PreOpType pre_op, IntraOpType intra_op, PostOpType post_op) {

    d_ptr.external_input = input_ptr;   // Borrow for this call
    d_ptr.external_output = output_ptr; // Borrow for this call

    Generic_Fwd_Image_Inv(image_to_search, pre_op, intra_op, post_op);
}
```

**Pattern:** User maintains ownership, FastFFT borrows pointers during execution.

### 3. No Explicit Plan Objects

**Unlike GpuImage/cuFFT:**
- GpuImage stores: `cufftHandle cuda_plan_forward`, `cuda_plan_inverse`
- FastFFT stores: **NOTHING** - plans are compile-time template instantiations

**Kernel Selection** (Lines 417-449):
```cpp
// "Plan" is implicit in kernel choice based on dimensions
switch (fwd_size_change_type) {
    case no_change: SetPrecisionAndExectutionMethod<...>(r2c_none_XY, ...);
    case decrease:  SetPrecisionAndExectutionMethod<...>(r2c_decrease_XY, ...);
    case increase:  SetPrecisionAndExectutionMethod<...>(r2c_increase_XY, ...);
}
```

### 4. Fused Operations

**Source:** `src/gpu/TemplateMatchingCore.cu:586`

```cpp
FT.FwdImageInvFFT(
    d_projection.real_values_fp16,           // Input (FP16)
    d_input_image->complex_values_fp16,      // Image to correlate with
    ccf_output,                               // Output
    noop,                                     // Pre-op functor
    conj_mul_then_scale,                      // Intra-op functor (between FFTs)
    noop                                      // Post-op functor
);
// Does: FFT(input) → complex_multiply(FFT_result, image) → IFFT → output
```

**This is fundamentally different** from Image's separate `ForwardFFT()` → `MultiplyPixelWise()` → `BackwardFFT()`

---

## Architectural Mismatch Analysis

| Aspect | Tensor Memory Pool | FastFFT | Compatible? |
|--------|-------------------|---------|-------------|
| **Ownership** | Pool owns all buffers | FastFFT owns internal buffers | ❌ Conflicting |
| **Lifecycle** | Buffers recycled per-operation | Buffers live for object lifetime | ❌ Different |
| **Size** | Exact tensor size | Max of all transforms | ❌ Different |
| **API** | Data lives in Tensor | Borrows external pointers | ✅ Can work with |
| **Plans** | Store in pool metadata | Compile-time templates | ⚠️ N/A |
| **Operations** | Separate function calls | Fused functors | ❌ Different paradigm |

**Conclusion:** Direct unification is architecturally challenging. Better approach: compatible memory layouts with clear integration boundaries.

---

## Proposed Architecture: Out-of-Place Buffering First

### Phase 3A: Add Out-of-Place Support to Tensor Memory Pool

**Adopt FastFFT's 2x allocation pattern:**

```cpp
template<typename T>
class TensorMemoryPool {
    struct Buffer {
        T* data;              // Primary buffer (buffer_1 in FastFFT terms)
        T* data_secondary;    // Secondary buffer (buffer_2 in FastFFT terms)
        size_t single_buffer_size;  // Size of EACH buffer

        // FFT plan storage (for classical FFTW/cuFFT)
        void* fft_plan_forward;
        void* fft_plan_backward;
        int3 planned_dims;
        bool plan_is_valid;
    };

    Buffer* Allocate(size_t elements) {
        // Allocate 2x for out-of-place transforms (FastFFT pattern)
        buffer.data = AllocateMemory(2 * elements * sizeof(T));
        buffer.data_secondary = &buffer.data[elements];  // Halfway through
        buffer.single_buffer_size = elements;
        return &buffer;
    }
};
```

**Benefits:**
1. Supports classical out-of-place FFTs (FFTW/cuFFT)
2. Compatible memory layout with FastFFT (both use 2x allocation)
3. Enables ping-pong operations for multi-stage transforms
4. No memory duplication if we expose buffers to FastFFT

### Phase 3B: Tensor API Extensions

```cpp
template<typename Scalar_t, typename Layout_t>
class Tensor {
public:
    // Existing
    Scalar_t* Data();

    // NEW: Access to secondary buffer for out-of-place operations
    Scalar_t* GetPrimaryBuffer();      // Same as Data()
    Scalar_t* GetSecondaryBuffer();    // Access to buffer_2
    size_t GetBufferSize();            // Size of EACH buffer
};
```

---

## FastFFT Integration Options

Since the developer maintains FastFFT, we can consider API extensions.

### Option 1: FastFFT External Workspace API (Tight Integration)

**New FastFFT API:**
```cpp
class FourierTransformer {
public:
    // NEW: Allow external workspace injection
    void SetExternalWorkspace(ComputeBaseType* buffer_1,
                             ComputeBaseType* buffer_2,
                             size_t buffer_size);

    // Existing APIs work as before
    void SetForwardFFTPlan(...);  // Doesn't allocate if external workspace set
    void SetInverseFFTPlan(...);
    void FwdImageInvFFT(...);
};
```

**Tensor Integration:**
```cpp
FastFFT::FourierTransformer<float, float, float2, 2> FT;
FT.SetExternalWorkspace(tensor.GetPrimaryBuffer(),
                        tensor.GetSecondaryBuffer(),
                        tensor.GetBufferSize());
FT.SetForwardFFTPlan(...);  // Uses external workspace, no internal allocation
```

**Pros:**
- Zero memory duplication
- Clean ownership (Tensor owns, FastFFT borrows)
- Consistent with FastFFT's external pointer borrowing pattern

**Cons:**
- Requires FastFFT API changes
- Added complexity to FastFFT initialization
- Need to handle workspace lifetime carefully

### Option 2: Compatible Layouts, Separate Management (Loose Coupling)

**Keep as-is:**
- Tensor allocates 2x for classical FFTs (FFTW/cuFFT)
- FastFFT still manages its own workspace
- Compatible layout means easy data sharing when needed

**Usage:**
```cpp
// Classical FFT path
ForwardFFT(tensor);  // Uses tensor's buffer_1 → buffer_2

// FastFFT path (when fused operations needed)
FastFFT::FourierTransformer<...> FT;
FT.SetForwardFFTPlan(...);  // FastFFT allocates its own workspace
FT.FwdImageInvFFT(tensor.GetPrimaryBuffer(), ...);  // Borrows Tensor data
```

**Pros:**
- No FastFFT changes needed
- Clear separation of concerns
- Each system works independently

**Cons:**
- Memory duplication when using FastFFT
- More complex for users (need to understand when to use which)

### Option 3: Unified Memory Layout, Flexible Usage

**Both allocate 2x, but use whichever makes sense:**
- Classical FFTW/cuFFT: Use Tensor's buffer_1 ↔ buffer_2
- Fused operations: Use FastFFT's internal workspace, borrow Tensor data as input/output
- Memory layout is compatible, so easy to convert between

**Pros:**
- Flexibility
- Compatible layouts enable easy transitions
- No forced coupling

**Cons:**
- Potential confusion about which buffers are used when
- Memory duplication in some scenarios

---

## Open Questions for Developer (FastFFT Maintainer)

### 1. External Workspace API

**Question:** Would you consider adding an external workspace API to FastFFT?

**Proposed signature:**
```cpp
void SetExternalWorkspace(ComputeBaseType* buffer_1,
                         ComputeBaseType* buffer_2,
                         size_t buffer_size);
```

**Use case:** Avoid memory duplication when Tensor already has compatible 2x allocation.

**Trade-offs:**
- Pro: Clean integration, zero duplication
- Con: API complexity, lifetime management

### 2. Memory Alignment Requirements

**Question:** Does FastFFT require specific memory alignment for its buffers?

**Context:**
- FastFFT uses `__align__(16)` for shared memory (`FastFFT.cuh:28`)
- Should Tensor buffers have specific alignment for FastFFT compatibility?
- Current Tensor alignment: default for `new T[]`

### 3. Type Compatibility

**Question:** FastFFT uses `DevicePointers<ComputeBaseType*, PositionSpaceType*, OtherImageType*>` template specialization. Should Tensor expose similar type flexibility?

**Context:**
- FastFFT supports: `<float*, float*, float2*>`, `<float*, __half*, float2*>`, `<float*, __half*, __half2*>`, etc.
- Tensor currently: `Tensor<float>`, future: `Tensor<complex<float>>`, `Tensor<__half>`
- Need to ensure type compatibility for borrowing pointers

### 4. Preferred Integration Strategy

**Question:** Which integration approach do you prefer?

**Options:**
1. **Option 1:** Add external workspace API to FastFFT (tight integration, no duplication)
2. **Option 2:** Keep separate, document compatible patterns (loose coupling, some duplication)
3. **Option 3:** Hybrid - compatible layouts, use whichever is appropriate

**User's preference will guide implementation plan.**

---

## GpuImage Pattern Analysis - Classical FFT Reference

### 1. Plan Management: Lazy Creation with Caching

**Pattern from GpuImage.cu:3804-3850:**

```cpp
void GpuImage::SetCufftPlan(cistem::fft_type::Enum plan_type,
                            void* input_buffer, void* output_buffer,
                            cudaStream_t wanted_stream) {
    // Check if plan already exists for requested type
    if (plan_type == set_plan_type && cufft_batch_size == set_batch_size) {
        // Reuse existing plan - only update stream if needed
        if (wanted_stream != set_stream_for_cufft) {
            cufftSetStream(cuda_plan_forward, wanted_stream);
            cufftSetStream(cuda_plan_inverse, wanted_stream);
        }
        return;  // Early exit - avoid expensive plan recreation
    }

    // Destroy old plan if changing parameters
    if (set_plan_type != cistem::fft_type::Enum::unset) {
        cufftDestroy(cuda_plan_inverse);
        cufftDestroy(cuda_plan_forward);
    }

    // Create new plan with requested parameters
    // ... plan creation logic ...
}
```

**Key Insights:**
- Plans are EXPENSIVE to create (especially for large dimensions)
- Cache and reuse plans whenever possible
- Only recreate when dimensions or type change
- Store plan metadata (type, dimensions, batch size) to enable caching

**Application to Tensor (CPU - FFTW/MKL):**

```cpp
// In TensorMemoryPool<T>::Buffer
struct Buffer {
    T* data;
    T* data_secondary;  // NEW: Out-of-place support
    size_t single_buffer_size;

    void* fft_plan_forward;   // fftwf_plan cast to void*
    void* fft_plan_backward;
    int3 planned_dims;        // Track what plan was created for
    bool plan_is_valid;       // Track if plan exists
};

void EnsureFFTPlan(Buffer& buffer, int3 dims) {
    if (buffer.plan_is_valid && buffer.planned_dims == dims) {
        return;  // Reuse existing plan - CRITICAL for performance
    }

    // Destroy old plan if dimensions changed
    if (buffer.plan_is_valid) {
        fftwf_destroy_plan((fftwf_plan)buffer.fft_plan_forward);
        fftwf_destroy_plan((fftwf_plan)buffer.fft_plan_backward);
    }

    // Create new plans
    buffer.fft_plan_forward = (void*)fftwf_plan_dft_r2c_3d(...);
    buffer.fft_plan_backward = (void*)fftwf_plan_dft_c2r_3d(...);
    buffer.planned_dims = dims;
    buffer.plan_is_valid = true;
}
```

### 2. State Tracking: Space Enum

**Pattern from GpuImage.h:40, GpuImage.cu multiple locations:**

```cpp
bool is_in_real_space;          // Position vs Momentum space
bool is_fft_centered_in_box;    // DC component centered
bool object_is_centred_in_box;  // Object centered in real space

// State updates in FFT methods
void ForwardFFT() {
    MyDebugAssertTrue(is_in_real_space, "Already in Fourier space");
    // Execute FFT
    is_in_real_space = false;  // Update state
}
```

**Application to Tensor:**
- **Already implemented in Phase 1!**

```cpp
enum class Space { Position, Momentum };
Space GetSpace() const;
void SetSpace(Space space);
```

### 3. Templated FFT Execution with Explicit Specialization

**Pattern from GpuImage.cu:3067-3078:**

```cpp
// Empty base template - compile-time type checking
template <class InputType, class OutputType>
void GpuImage::_ForwardFFT() { }

// Explicit specialization for float → float2
template <>
void GpuImage::_ForwardFFT<float, float2>() {
    cufftExecR2C(cuda_plan_forward,
                 (cufftReal*)position_space_ptr,
                 (cufftComplex*)momentum_space_ptr);
}
```

**Application to Tensor:**

```cpp
// src/core/tensor/operations/fft_operations.h

// Empty base template
template <typename InputType, typename OutputType>
void _ExecuteForwardFFT(void* plan, InputType* in, OutputType* out) { }

// Specialization for float → cistem::complex<float>
template <>
void _ExecuteForwardFFT<float, cistem::complex<float>>(
    void* plan, float* in, cistem::complex<float>* out) {

    fftwf_plan fftw_plan = (fftwf_plan)plan;
    fftwf_execute_dft_r2c(fftw_plan, in,
                         reinterpret_cast<fftwf_complex*>(out));
}

// Public API
template <typename Scalar_t, typename Layout_t>
void ForwardFFT(Tensor<Scalar_t, Layout_t>& tensor, bool scale = true) {
    TENSOR_DEBUG_ASSERT(tensor.GetSpace() == Tensor::Space::Position);

    // Lazy plan creation (cached in memory pool buffer)
    auto& buffer = tensor.GetMemoryPool().GetBuffer(tensor.GetBufferIndex());
    EnsureFFTPlan(buffer, tensor.GetDims());

    if (scale) {
        MultiplyByScalar(tensor, tensor.GetFourierNormalizationFactor());
    }

    _ExecuteForwardFFT<Scalar_t, cistem::complex<Scalar_t>>(
        buffer.fft_plan_forward,
        tensor.GetPrimaryBuffer(),
        tensor.GetSecondaryBuffer());  // Out-of-place to secondary

    tensor.SetSpace(Tensor::Space::Momentum);
}
```

---

## Implementation Plan (Pending FastFFT Decision)

### Phase 3A.0: Out-of-Place Memory Support (Week 1) - FOUNDATION

**Critical First Step - Independent of FastFFT Integration Strategy**

**Tasks:**
1. **Add secondary buffer to TensorMemoryPool::Buffer** (1 day)
   - `T* data_secondary` member
   - `size_t single_buffer_size` (replaces current `size`)
   - Allocate 2x in `Allocate()` method
   - Update `Deallocate()` to free full 2x allocation

2. **Add Tensor API for buffer access** (1 day)
   - `GetPrimaryBuffer()` - returns `data`
   - `GetSecondaryBuffer()` - returns `data_secondary`
   - `GetBufferSize()` - returns `single_buffer_size`

3. **Update existing operations** (1 day)
   - Verify iterator still works (should use primary buffer)
   - Update tests to validate 2x allocation
   - Ensure no memory leaks

4. **Add plan storage to Buffer** (1 day)
   - `void* fft_plan_forward/backward`
   - `int3 planned_dims`, `bool plan_is_valid`
   - Destructor cleanup

**Deliverable:** Tensor has proper out-of-place buffer support, ready for any FFT backend.

### Phase 3A.1: CPU FFT Infrastructure (Week 2) - CLASSICAL FFT

**Prerequisites:** Phase 3A.0 complete, awaiting FastFFT integration decision

**Tasks:**
1. **Implement EnsureFFTPlan() in memory pool** (2 days)
   - Lazy plan creation pattern (GpuImage.cu:3804 pattern)
   - Plan caching and reuse logic
   - Dimension change detection
   - Plan destruction when buffer deallocated

2. **Create fft_operations.h/.cpp** (2 days)
   - `ForwardFFT()`, `BackwardFFT()` template functions
   - Type-safe execution with template specialization (GpuImage.cu:3067 pattern)
   - Integration with Tensor::Space enum
   - Out-of-place execution using buffer_1 → buffer_2

3. **Add unit tests** (1 day)
   - Test plan caching behavior
   - Test dimension changes trigger recreation
   - Verify transform correctness (compare with Image)
   - Test out-of-place execution

### Phase 3A.2: GPU FFT Infrastructure (Week 3) - DEFERRED

**Status:** Awaiting FastFFT integration decision

**If Option 1 (External Workspace):**
- Implement FastFFT external workspace API
- Integrate with Tensor buffers
- No cuFFT implementation needed (use FastFFT for everything)

**If Option 2/3 (Separate/Hybrid):**
- Implement classical cuFFT support (similar to CPU path)
- Document when to use FastFFT vs cuFFT
- Add examples for both approaches

---

## Decision Log

### Decision 1: Out-of-Place Support is Foundational (2025-10-03)

**Decision:** Implement 2x buffer allocation FIRST, before any FFT backend integration.

**Rationale:**
- FastFFT uses 2x allocation - we should match for compatibility
- Classical out-of-place FFTs need it anyway
- Independent of FastFFT integration strategy
- Enables ping-pong operations for multi-stage transforms

**Status:** Approved, ready to implement

### Decision 2: FastFFT Integration Strategy (2025-10-03)

**Decision:** **PENDING - Awaiting developer input**

**Options:**
1. External workspace API (tight integration)
2. Compatible layouts, separate management (loose coupling)
3. Hybrid approach (flexible usage)

**Blocker:** Need FastFFT maintainer decision on API modifications

### Decision 3: Skip cuFFT Callbacks

**Decision:** Do NOT implement cuFFT callbacks in initial implementation.

**Rationale:**
- GpuImage notes they're deprecated in CUDA 11.4+
- FastFFT provides superior fused operation capabilities
- Don't invest in deprecated tech

**Status:** Approved

### Decision 4: Store Plans as void* (2025-10-03)

**Decision:** Store FFT plans as `void*` not typed handles.

**Rationale:**
- Keeps TensorMemoryPool independent of FFT library headers
- Matches GpuImage pattern
- Enables ABI stability
- Backend abstraction (could use FFTW or MKL interchangeably)

**Status:** Approved

---

## Success Criteria

### Phase 3A.0 Complete When:
1. ✅ TensorMemoryPool allocates 2x buffer (primary + secondary)
2. ✅ Tensor exposes GetPrimaryBuffer(), GetSecondaryBuffer()
3. ✅ Existing operations still work (using primary buffer)
4. ✅ Unit tests validate memory allocation
5. ✅ No memory leaks

### Phase 3A.1 Complete When:
1. ✅ Tensor supports `ForwardFFT()` and `BackwardFFT()` on CPU
2. ✅ Plans are cached and reused (no recreation on same dimensions)
3. ✅ Plans are destroyed and recreated when dimensions change
4. ✅ All unit tests pass (transform correctness + plan caching)
5. ✅ Out-of-place transforms work correctly

### Phase 3A.2 Complete When:
(Criteria depend on FastFFT integration decision)

---

## References

### FastFFT Source Analysis
- **Memory allocation:** `/sa_shared/git/FastFFT/src/fastfft/FastFFT.cu:289-305`
- **External pointer borrowing:** `FastFFT.cu:355-402`
- **Buffer structure:** `/sa_shared/git/FastFFT/include/FastFFT.h:46-84`
- **cisTEM usage:** `src/gpu/TemplateMatchingCore.cu:383-588`

### GpuImage Patterns (Classical FFT Reference)
- **Plan caching:** `src/gpu/GpuImage.cu:3804-3850`
- **Type-safe execution:** `src/gpu/GpuImage.cu:3067-3078`
- **State tracking:** `src/gpu/GpuImage.h:40` (is_in_real_space)

### Existing Tensor Infrastructure
- **Space enum:** `src/core/tensor/core/tensor.h` (implemented in Phase 1)
- **Memory pool:** `src/core/tensor/memory/memory_pool.h`
- **Phase 2 operations:** `src/core/tensor/operations/arithmetic.h`, `statistics.h`

---

**Next Steps:** Await FastFFT integration decision, then proceed with Phase 3A.0 implementation.
