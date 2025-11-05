/*
 * Copyright YEAR cisTEMx Development Team
 *
 * Licensed under...
 */

/**
 * @file cuda_test_template.cpp
 * @brief Template demonstrating testable CUDA code patterns
 *
 * Key strategies:
 * 1. Write __host__ __device__ functions for CPU testing
 * 2. Decompose kernels into small, testable components
 * 3. Compare GPU results against CPU reference implementations
 * 4. Test components independently before integration
 *
 * Replace COMPONENT_NAME with actual component name.
 */

#include <vector>
#include <cmath>
#include "../../include/catch2/catch.hpp"

// ====================
// Testable Device Functions
// ====================

/**
 * @brief Example: Mathematical calculation testable on both CPU and GPU
 *
 * __host__ __device__ allows this function to be:
 * - Tested on CPU (fast, easy debugging)
 * - Used in GPU kernels
 */
__host__ __device__ float complexCalculation(float a, float b) {
    return (a * a + b * b) / (a + b);
}

/**
 * @brief Example: Index calculation testable on CPU
 */
__host__ __device__ int calculateGlobalIndex(int blockIdx, int blockDim, int threadIdx) {
    return blockIdx * blockDim + threadIdx;
}

/**
 * @brief Example: Boundary check testable on CPU
 */
__host__ __device__ bool isValidIndex(int idx, int size) {
    return idx >= 0 && idx < size;
}

/**
 * @brief Example: 2D to 1D index conversion
 */
__host__ __device__ int calculate2DIndex(int x, int y, int width) {
    return y * width + x;
}

// ====================
// CPU Unit Tests for Device Functions
// ====================

TEST_CASE("Complex calculation", "[cuda][math]") {
    // Test on CPU - fast, easy to debug
    SECTION("normal values") {
        float result = complexCalculation(3.0f, 4.0f);
        REQUIRE(result == Approx(25.0f / 7.0f));
    }

    SECTION("equal values") {
        float result = complexCalculation(5.0f, 5.0f);
        REQUIRE(result == Approx(5.0f));
    }

    SECTION("small values") {
        float result = complexCalculation(0.1f, 0.1f);
        REQUIRE(result == Approx(0.1f));
    }
}

TEST_CASE("Global index calculation", "[cuda][indexing]") {
    SECTION("first thread in first block") {
        int idx = calculateGlobalIndex(0, 256, 0);
        REQUIRE(idx == 0);
    }

    SECTION("first thread in second block") {
        int idx = calculateGlobalIndex(1, 256, 0);
        REQUIRE(idx == 256);
    }

    SECTION("middle thread in middle block") {
        int idx = calculateGlobalIndex(5, 256, 128);
        REQUIRE(idx == 5 * 256 + 128);
    }
}

TEST_CASE("Index validation", "[cuda][indexing]") {
    SECTION("valid indices") {
        REQUIRE(isValidIndex(0, 100));
        REQUIRE(isValidIndex(50, 100));
        REQUIRE(isValidIndex(99, 100));
    }

    SECTION("invalid indices") {
        REQUIRE_FALSE(isValidIndex(-1, 100));
        REQUIRE_FALSE(isValidIndex(100, 100));
        REQUIRE_FALSE(isValidIndex(1000, 100));
    }
}

TEST_CASE("2D index calculation", "[cuda][indexing]") {
    SECTION("top-left corner") {
        REQUIRE(calculate2DIndex(0, 0, 10) == 0);
    }

    SECTION("first row") {
        REQUIRE(calculate2DIndex(5, 0, 10) == 5);
    }

    SECTION("second row") {
        REQUIRE(calculate2DIndex(3, 1, 10) == 13);
    }
}

// ====================
// Kernel Example (Composition of Tested Functions)
// ====================

#ifdef cisTEM_USE_CUDA

/**
 * @brief Example kernel that composes tested functions
 *
 * Since all component functions are tested on CPU, we have
 * high confidence this kernel works correctly.
 */
__global__ void processDataKernel(float* input, float* output, int size) {
    // Use tested index calculation
    int idx = calculateGlobalIndex(blockIdx.x, blockDim.x, threadIdx.x);

    // Use tested validation
    if ( ! isValidIndex(idx, size) ) {
        return;
    }

    // Use tested mathematical operation
    output[idx] = complexCalculation(input[idx], input[idx] + 1.0f);
}

#endif // cisTEM_USE_CUDA

// ====================
// CPU Reference Implementation
// ====================

/**
 * @brief CPU reference implementation for validation
 *
 * This should be simple and obviously correct, even if slow.
 */
std::vector<float> cpuReferenceImplementation(const std::vector<float>& input) {
    std::vector<float> output(input.size( ));
    for ( size_t i = 0; i < input.size( ); ++i ) {
        output[i] = complexCalculation(input[i], input[i] + 1.0f);
    }
    return output;
}

// ====================
// GPU Integration Tests
// ====================

#ifdef cisTEM_USE_CUDA

/**
 * @brief Helper to check if CUDA device is available
 */
bool cuda_device_available( ) {
    int         device_count = 0;
    cudaError_t error        = cudaGetDeviceCount(&device_count);
    return error == cudaSuccess && device_count > 0;
}

/**
 * @brief GPU wrapper function for testing
 */
std::vector<float> gpuImplementation(const std::vector<float>& input) {
    if ( ! cuda_device_available( ) ) {
        throw std::runtime_error("No CUDA device available");
    }

    // Allocate device memory
    float* d_input;
    float* d_output;
    size_t size_bytes = input.size( ) * sizeof(float);

    cudaMalloc(&d_input, size_bytes);
    cudaMalloc(&d_output, size_bytes);

    // Copy input to device
    cudaMemcpy(d_input, input.data( ), size_bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    int threads_per_block = 256;
    int num_blocks        = (input.size( ) + threads_per_block - 1) / threads_per_block;
    processDataKernel<<<num_blocks, threads_per_block>>>(d_input, d_output, input.size( ));

    // Check for launch errors
    cudaError_t launch_err = cudaGetLastError( );
    if ( launch_err != cudaSuccess ) {
        cudaFree(d_input);
        cudaFree(d_output);
        throw std::runtime_error(cudaGetErrorString(launch_err));
    }

    // Wait for completion
    cudaError_t sync_err = cudaDeviceSynchronize( );
    if ( sync_err != cudaSuccess ) {
        cudaFree(d_input);
        cudaFree(d_output);
        throw std::runtime_error(cudaGetErrorString(sync_err));
    }

    // Copy result back to host
    std::vector<float> output(input.size( ));
    cudaMemcpy(output.data( ), d_output, size_bytes, cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}

/**
 * @brief Test GPU implementation against CPU reference
 */
TEST_CASE("GPU matches CPU reference", "[gpu][validation]") {
    if ( ! cuda_device_available( ) ) {
        SKIP("No CUDA device available");
    }

    // Small test data for fast execution
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

    // CPU reference (simple, obviously correct)
    auto cpu_result = cpuReferenceImplementation(input);

    // GPU implementation (optimized, complex)
    auto gpu_result = gpuImplementation(input);

    // Compare results
    REQUIRE(cpu_result.size( ) == gpu_result.size( ));
    for ( size_t i = 0; i < cpu_result.size( ); ++i ) {
        REQUIRE(gpu_result[i] == Approx(cpu_result[i]).epsilon(1e-5));
    }
}

/**
 * @brief Test property: Processing doesn't change size
 */
TEST_CASE("GPU processing preserves size", "[gpu][properties]") {
    if ( ! cuda_device_available( ) ) {
        SKIP("No CUDA device available");
    }

    std::vector<float> input(128, 1.0f);

    auto result = gpuImplementation(input);

    REQUIRE(result.size( ) == input.size( ));
}

/**
 * @brief Test edge case: Empty input
 */
TEST_CASE("GPU handles empty input", "[gpu][edge]") {
    if ( ! cuda_device_available( ) ) {
        SKIP("No CUDA device available");
    }

    std::vector<float> input;

    auto result = gpuImplementation(input);

    REQUIRE(result.empty( ));
}

/**
 * @brief Test edge case: Single element
 */
TEST_CASE("GPU handles single element", "[gpu][edge]") {
    if ( ! cuda_device_available( ) ) {
        SKIP("No CUDA device available");
    }

    std::vector<float> input = {5.0f};

    auto cpu_result = cpuReferenceImplementation(input);
    auto gpu_result = gpuImplementation(input);

    REQUIRE(gpu_result.size( ) == 1);
    REQUIRE(gpu_result[0] == Approx(cpu_result[0]));
}

#endif // cisTEM_USE_CUDA

// ====================
// Kernel Launch Parameter Validation
// ====================

/**
 * @brief Example: Validate kernel launch parameters
 */
struct KernelLaunchParams {
    dim3   grid_dim;
    dim3   block_dim;
    size_t shared_mem;

    bool validate( ) const {
        // Check CUDA grid dimension limits
        if ( grid_dim.x > 65535 || grid_dim.y > 65535 || grid_dim.z > 65535 ) {
            return false;
        }

        // Check block size limit (typical: 1024 threads)
        if ( block_dim.x * block_dim.y * block_dim.z > 1024 ) {
            return false;
        }

        // Check shared memory limit (typical: 48KB)
        if ( shared_mem > 48 * 1024 ) {
            return false;
        }

        return true;
    }
};

TEST_CASE("Kernel launch parameter validation", "[cuda][validation]") {
    SECTION("valid parameters") {
        KernelLaunchParams params{{256, 256, 1}, {256, 1, 1}, 1024};
        REQUIRE(params.validate( ));
    }

    SECTION("grid dimension exceeds limit") {
        KernelLaunchParams params{{65536, 1, 1}, {256, 1, 1}, 0};
        REQUIRE_FALSE(params.validate( ));
    }

    SECTION("block size too large") {
        KernelLaunchParams params{{256, 1, 1}, {33, 33, 1}, 0};
        // 33 * 33 = 1089 > 1024
        REQUIRE_FALSE(params.validate( ));
    }

    SECTION("shared memory too large") {
        KernelLaunchParams params{{256, 1, 1}, {256, 1, 1}, 64 * 1024};
        REQUIRE_FALSE(params.validate( ));
    }
}

// ====================
// Summary
// ====================

/*
 * KEY TAKEAWAYS:
 *
 * 1. Write __host__ __device__ functions that can be tested on CPU
 * 2. Decompose kernels into small, testable components
 * 3. Test components independently on CPU (fast, easy debugging)
 * 4. Compose tested components into kernels (high confidence)
 * 5. Validate GPU implementation against CPU reference
 * 6. Test properties and edge cases
 * 7. Always gate GPU tests with #ifdef and runtime checks
 * 8. Keep test allocations small (<1MB) for fast execution
 *
 * This approach enables:
 * - Fast iteration with CPU debugging
 * - High confidence in correctness
 * - Modular, maintainable code
 * - Works with standard C++ testing frameworks
 */
