# CUDA Unit Testing Strategies

Comprehensive guide for writing testable CUDA code and unit testing GPU functionality.

## The CUDA Testing Challenge

**Testing CUDA code is fundamentally different** from testing CPU code:

**Challenges:**
- Device code runs on GPU, not directly accessible from CPU
- Can't step through device code with standard debuggers
- Global kernels are monolithic entry points
- Memory exists in separate device address space
- Kernel launches are asynchronous
- Errors may only surface later

**Traditional approach:** Run entire kernel, check final output. Problems:
- Large, complex kernels are hard to debug
- Failures give little information about what went wrong
- Can't test components in isolation
- Difficult to verify intermediate steps

**Solution:** Design GPU code for testability from the start.

## Core Strategy: `__host__ __device__` Functions

The key insight: **Write testable device functions that also compile for CPU.**

### The Pattern

```cpp
// Testable function - works on both CPU and GPU
__host__ __device__ float complexCalculation(float a, float b) {
    return (a * a + b * b) / (a + b);
}

// Unit test on CPU
TEST_CASE("Complex calculation", "[cuda][math]") {
    REQUIRE(complexCalculation(3.0f, 4.0f) == Approx(25.0f/7.0f));
    REQUIRE(complexCalculation(1.0f, 1.0f) == Approx(1.0f));
}

// Use in device code
__global__ void processKernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = complexCalculation(input[idx], input[idx+1]);
    }
}
```

### Benefits

- **Test on CPU**: Fast, easy to debug, standard tools work
- **Run on GPU**: Same code runs in kernels
- **Confidence**: If CPU version works, GPU version likely works
- **Debugging**: CPU debugging is much simpler than GPU debugging
- **Faster iteration**: No device memory allocation/copying for unit tests

## Code Decomposition for Testability

### Bad: Monolithic Kernel

```cpp
// Hard to test - everything in one kernel
__global__ void processDataKernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size) return;

    // 500 lines of complex logic
    // - Index calculations
    // - Boundary checking
    // - Mathematical operations
    // - Result aggregation
    //
    // If this fails, where's the bug?
}
```

**Problems:**
- Can't test individual components
- Failures are cryptic
- Debugging requires full kernel execution
- Changes risk breaking everything

### Good: Decomposed Into Testable Functions

```cpp
// Testable components
__host__ __device__ int calculateGlobalIndex(int blockIdx, int blockDim,
                                             int threadIdx) {
    return blockIdx * blockDim + threadIdx;
}

__host__ __device__ bool isValidIndex(int idx, int size) {
    return idx >= 0 && idx < size;
}

__host__ __device__ float computeValue(float input) {
    return sqrtf(input * input + 1.0f);
}

__host__ __device__ int calculate2DIndex(int x, int y, int width) {
    return y * width + x;
}

// Simple kernel that composes tested components
__global__ void processDataKernel(float* input, float* output, int size) {
    int idx = calculateGlobalIndex(blockIdx.x, blockDim.x, threadIdx.x);

    if (!isValidIndex(idx, size)) return;

    output[idx] = computeValue(input[idx]);
}

// Unit tests for each component
TEST_CASE("Global index calculation", "[cuda][indexing]") {
    REQUIRE(calculateGlobalIndex(0, 256, 0) == 0);
    REQUIRE(calculateGlobalIndex(1, 256, 0) == 256);
    REQUIRE(calculateGlobalIndex(1, 256, 128) == 384);
}

TEST_CASE("Index validation", "[cuda][indexing]") {
    REQUIRE(isValidIndex(0, 100) == true);
    REQUIRE(isValidIndex(99, 100) == true);
    REQUIRE(isValidIndex(100, 100) == false);
    REQUIRE(isValidIndex(-1, 100) == false);
}

TEST_CASE("Value computation", "[cuda][math]") {
    REQUIRE(computeValue(0.0f) == Approx(1.0f));
    REQUIRE(computeValue(3.0f) == Approx(sqrtf(10.0f)));
}
```

### Benefits of Decomposition

- **Each function is testable in isolation**
- **Bugs are localized** - tests tell you exactly what broke
- **Functions are reusable** across multiple kernels
- **Code is more maintainable** and readable
- **Debugging is vastly simpler** - test functions on CPU first

## Testing Strategies

### Strategy 1: Unit Test `__host__ __device__` Functions

Test device functions directly on CPU:

```cpp
// Complex math function
__host__ __device__ float3 rotateVector(float3 v, float3 axis, float angle) {
    // Rodrigues' rotation formula
    float c = cosf(angle);
    float s = sinf(angle);
    float t = 1.0f - c;

    float3 result;
    result.x = v.x * (t*axis.x*axis.x + c) +
               v.y * (t*axis.x*axis.y - s*axis.z) +
               v.z * (t*axis.x*axis.z + s*axis.y);
    // ... y and z components

    return result;
}

TEST_CASE("Vector rotation", "[cuda][geometry]") {
    float3 v = {1.0f, 0.0f, 0.0f};
    float3 z_axis = {0.0f, 0.0f, 1.0f};
    float angle = M_PI / 2.0f;  // 90 degrees

    float3 rotated = rotateVector(v, z_axis, angle);

    REQUIRE(rotated.x == Approx(0.0f).margin(1e-6));
    REQUIRE(rotated.y == Approx(1.0f));
    REQUIRE(rotated.z == Approx(0.0f).margin(1e-6));
}
```

### Strategy 2: CPU Reference Implementation

Compare GPU results against known-good CPU implementation:

```cpp
// CPU reference implementation (simple, obviously correct)
std::vector<float> cpuFFT(const std::vector<float>& input) {
    // Simple, straightforward implementation
    // May be slow, but correctness is obvious
    std::vector<float> result(input.size());
    for (size_t k = 0; k < input.size(); ++k) {
        float real = 0, imag = 0;
        for (size_t n = 0; n < input.size(); ++n) {
            float angle = -2 * M_PI * k * n / input.size();
            real += input[n] * cos(angle);
            imag += input[n] * sin(angle);
        }
        result[k] = sqrt(real*real + imag*imag);
    }
    return result;
}

// GPU implementation (optimized, complex)
std::vector<float> gpuFFT(const std::vector<float>& input);

// Test: GPU matches CPU reference
#ifdef cisTEM_USE_CUDA
TEST_CASE("GPU FFT matches CPU reference", "[gpu][fft]") {
    if (!cuda_device_available()) {
        SKIP("No CUDA device available");
    }

    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

    auto cpu_result = cpuFFT(input);
    auto gpu_result = gpuFFT(input);

    REQUIRE(cpu_result.size() == gpu_result.size());
    for (size_t i = 0; i < cpu_result.size(); ++i) {
        REQUIRE(gpu_result[i] == Approx(cpu_result[i]).epsilon(1e-4));
    }
}
#endif
```

### Strategy 3: Property-Based Testing

Test mathematical properties that must hold:

```cpp
#ifdef cisTEM_USE_CUDA
TEST_CASE("FFT properties", "[gpu][fft][properties]") {
    if (!cuda_device_available()) {
        SKIP("No CUDA device available");
    }

    std::vector<float> input = generateTestSignal();

    // Property: Forward then inverse FFT should give original
    auto freq_domain = gpuFFT_forward(input);
    auto reconstructed = gpuFFT_inverse(freq_domain);

    REQUIRE(reconstructed.size() == input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        REQUIRE(reconstructed[i] == Approx(input[i]).epsilon(1e-5));
    }

    // Property: Parseval's theorem (energy conservation)
    float time_energy = 0, freq_energy = 0;
    for (size_t i = 0; i < input.size(); ++i) {
        time_energy += input[i] * input[i];
        freq_energy += freq_domain[i] * freq_domain[i];
    }
    REQUIRE(freq_energy == Approx(time_energy).epsilon(1e-4));
}
#endif
```

### Strategy 4: Kernel Launch Parameter Validation

Test parameter validation before actual kernel launch:

```cpp
struct KernelLaunchParams {
    dim3 grid_dim;
    dim3 block_dim;
    size_t shared_mem;

    bool validate() const {
        // Check CUDA limits
        if (grid_dim.x > 65535 || grid_dim.y > 65535 || grid_dim.z > 65535) {
            return false;
        }
        if (block_dim.x * block_dim.y * block_dim.z > 1024) {
            return false;
        }
        if (shared_mem > 48 * 1024) {  // 48KB typical limit
            return false;
        }
        return true;
    }
};

TEST_CASE("Kernel launch parameter validation", "[cuda][validation]") {
    SECTION("valid parameters") {
        KernelLaunchParams params{{256, 256, 1}, {32, 32, 1}, 1024};
        REQUIRE(params.validate());
    }

    SECTION("grid dimension exceeds limit") {
        KernelLaunchParams params{{65536, 1, 1}, {256, 1, 1}, 0};
        REQUIRE_FALSE(params.validate());
    }

    SECTION("block size too large") {
        KernelLaunchParams params{{256, 1, 1}, {33, 33, 1}, 0};
        // 33*33 = 1089 > 1024
        REQUIRE_FALSE(params.validate());
    }

    SECTION("shared memory too large") {
        KernelLaunchParams params{{256, 1, 1}, {256, 1, 1}, 64 * 1024};
        REQUIRE_FALSE(params.validate());
    }
}
```

### Strategy 5: Test Small Fragments with cuda_gtest_plugin

For very small CUDA code fragments, `cuda_gtest_plugin` can run them on both CPU and GPU:

```cpp
// Simple device function
__device__ __host__ int add(int a, int b) {
    return a + b;
}

// Test runs on both CPU and GPU
TEST(CudaTest, AddFunction) {
    EXPECT_EQ(add(2, 3), 5);
}
// Automatically generates TEST_CUDA version for device execution
```

**Limitations:**
- Only for very small, self-contained functions
- Can't test kernels directly
- Limited to simple assertions

**When to use:** Testing mathematical utility functions in isolation.

## Memory Testing Patterns

### Test Device Memory Operations

```cpp
#ifdef cisTEM_USE_CUDA
TEST_CASE("Device pointer operations", "[gpu][memory]") {
    if (!cuda_device_available()) {
        SKIP("No CUDA device available");
    }

    SECTION("allocation and deallocation") {
        DevicePointerArray<float> ptr;
        ptr.resize(1024);

        REQUIRE(ptr.size() == 1024);
        REQUIRE(ptr.data() != nullptr);

        ptr.Deallocate();
        REQUIRE(ptr.size() == 0);
    }

    SECTION("host-to-device copy") {
        std::vector<float> host_data = {1.0f, 2.0f, 3.0f, 4.0f};
        DevicePointerArray<float> device_data;
        device_data.CopyFromHost(host_data.data(), host_data.size());

        std::vector<float> result(host_data.size());
        device_data.CopyToHost(result.data());

        REQUIRE(result == host_data);
    }
}
#endif
```

### Keep Allocations Small

```cpp
// Good: Small allocation for unit test
TEST_CASE("GPU kernel correctness", "[gpu]") {
    std::vector<float> input(256);  // Small test data
    // Test with small data
}

// Bad: Huge allocation for unit test
TEST_CASE("GPU performance", "[gpu]") {
    std::vector<float> input(100'000'000);  // 400MB!
    // This is not a unit test, it's a benchmark
}
```

## Error Checking

### Always Check CUDA Errors

```cpp
// Helper macro for error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error( \
                std::string("CUDA error: ") + cudaGetErrorString(err)); \
        } \
    } while(0)

// Use in tests
TEST_CASE("CUDA operations with error checking", "[gpu]") {
    float* d_ptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, 1024 * sizeof(float)));

    // If allocation fails, test fails with clear error message

    CUDA_CHECK(cudaFree(d_ptr));
}
```

### Check Kernel Launch Errors

```cpp
// Launch kernel
myKernel<<<grid, block>>>(args);

// Check for launch errors
cudaError_t launch_err = cudaGetLastError();
if (launch_err != cudaSuccess) {
    // Handle launch configuration error
}

// Wait for completion
cudaError_t exec_err = cudaDeviceSynchronize();
if (exec_err != cudaSuccess) {
    // Handle execution error
}
```

## Design Benefits

### Before: Untestable CUDA Code

```cpp
// Monolithic, untestable
__global__ void processImageKernel(float* image, int width, int height) {
    // 300 lines of complex index calculations
    // 200 lines of mathematical operations
    // 100 lines of boundary checks
    //
    // If this fails, good luck debugging
}
```

**Problems:**
- Can't test components in isolation
- Must run full kernel to test anything
- Debugging requires device debugging tools
- Slow iteration cycle

### After: Testable CUDA Code

```cpp
// Tested on CPU
__host__ __device__ int2 calculate2DIndex(int thread_id, int width);
__host__ __device__ bool isValidPixel(int2 pos, int width, int height);
__host__ __device__ float applyFilter(float center, float* neighbors);
__host__ __device__ float normalizeValue(float val, float min, float max);

// Simple kernel composing tested functions
__global__ void processImageKernel(float* image, int width, int height) {
    int2 pos = calculate2DIndex(threadIdx.x + blockIdx.x * blockDim.x, width);

    if (!isValidPixel(pos, width, height)) return;

    float neighbors[9];
    // ... gather neighbors ...

    float result = applyFilter(image[pos.y * width + pos.x], neighbors);
    image[pos.y * width + pos.x] = normalizeValue(result, 0.0f, 1.0f);
}

// Each function has CPU unit tests
```

**Benefits:**
- Each component tested in isolation
- Bugs are localized immediately
- CPU debugging for most development
- Fast iteration cycle
- High confidence in correctness

## cisTEMx GPU Testing Patterns

From existing code in `src/test/gpu/`:

```cpp
// Pattern 1: Compilation gating
#ifdef cisTEM_USE_CUDA
TEST_CASE("GPU tests", "[gpu]") {
    // ...
}
#endif

// Pattern 2: Runtime device check
TEST_CASE("GPU functionality", "[gpu]") {
    if (!cuda_device_available()) {
        SKIP("No CUDA device available");
    }
    // Test implementation
}

// Pattern 3: DevicePointerArray testing
TEST_CASE("Device pointer operations", "[gpu][memory]") {
    DevicePointerArray<float2> testPtr;
    testPtr.resize(4);

    REQUIRE(testPtr.size() == 4);
    // Test operations

    testPtr.Deallocate();
}
```

## Best Practices Summary

### 1. Design for Testability

- Write `__host__ __device__` functions
- Decompose kernels into small, testable components
- Test mathematical logic on CPU first
- Keep functions focused and single-purpose

### 2. Test Strategy

- **Unit test**: `__host__ __device__` functions on CPU
- **Validation test**: Compare GPU vs. CPU reference
- **Property test**: Verify mathematical properties
- **Integration test**: Full kernel with small data

### 3. Always Gate GPU Tests

```cpp
#ifdef cisTEM_USE_CUDA
TEST_CASE("GPU test", "[gpu]") {
    if (!cuda_device_available()) {
        SKIP("No CUDA device available");
    }
    // Test
}
#endif
```

### 4. Check All CUDA Errors

- Wrap CUDA calls in error checking
- Check kernel launch errors
- Check execution errors with `cudaDeviceSynchronize()`

### 5. Keep Test Allocations Small

- Use small data sizes for unit tests
- < 1MB typical
- Bounded execution time (<200ms)
- Tag larger tests with `[slow]`

### 6. Provide Clear Skip Messages

```cpp
if (!cuda_device_available()) {
    SKIP("No CUDA device available");
}

if (compute_capability < 5.0) {
    SKIP("Requires compute capability >= 5.0");
}
```

## Resources

- **NVIDIA Documentation**: CUDA Best Practices Guide
- **Testing Tools**: cuda_gtest_plugin, compute-sanitizer
- **Debugging**: cuda-gdb, compute-sanitizer, Nsight
- **cisTEMx Examples**: `src/test/gpu/` directory

## Key Insight

**The fundamental insight for CUDA testing**: Don't try to test monolithic kernels. Instead, decompose into `__host__ __device__` functions that can be tested on CPU, then compose those tested functions into kernels.

This approach:
- Makes code testable without special GPU testing frameworks
- Enables fast iteration with CPU debugging
- Produces more modular, maintainable code
- Gives high confidence in correctness
- Works with existing C++ testing frameworks (Catch2, GoogleTest)
