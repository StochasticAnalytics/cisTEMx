/*
 * gpu_core_headers.h
 *
 *  Created on: Jul 31, 2019
 *      Author: himesb
 */

#ifndef GPU_CORE_HEADERS_H_
#define GPU_CORE_HEADERS_H_

#include "../core/core_headers.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cutensor.h>

const int MAX_GPU_COUNT = 32;

// The following block is
#define gMin(a, b) (((a) < (b)) ? (a) : (b))
#define gMax(a, b) (((a) > (b)) ? (a) : (b))

// clang-format off

#ifndef ENABLE_GPU_DEBUG
#define cudaErr(err, ...) { err; }
#define nppErr(err, ...) { err; }
#define cuTensorErr(err, ...) { err; }
#define cufftErr(err, ...) { err; }
#define postcheck 
#define precheck 
#else
// The static path to the error code definitions is brittle, but better than the internet. At least you can click in VSCODE to get there.
#define nppErr(npp_stat)  {if (npp_stat != NPP_SUCCESS) { std::cerr << "NPP_CHECK_NPP - npp_stat = " << npp_stat ; wxPrintf(" at %s:(%d)\nFind error codes at /usr/local/cuda-11.7/targets/x86_64-linux/include/nppdefs.h:(170)\n\n",__FILE__,__LINE__); DEBUG_ABORT} }
#define cudaErr(error) { auto status = static_cast<cudaError_t>(error); if (status != cudaSuccess) { std::cerr << cudaGetErrorString(status) << " :-> "; MyPrintWithDetails(""); DEBUG_ABORT} }
#define cufftErr(error) { auto status = static_cast<cufftResult>(error); if (status != CUFFT_SUCCESS) { std::cerr << cistem::gpu::cufft_error_types[status] << " :-> "; MyPrintWithDetails(""); DEBUG_ABORT} }
#define cuTensorErr(error) { auto status = static_cast<cutensorStatus_t>(error); if (status != CUTENSOR_STATUS_SUCCESS) { std::cerr << cutensorGetErrorString(status) << " :-> "; MyPrintWithDetails(""); DEBUG_ABORT} }
#define postcheck { cudaErr(cudaPeekAtLastError()); cudaError_t error = cudaStreamSynchronize(cudaStreamPerThread); cudaErr(error); }
#define precheck { cudaErr(cudaGetLastError()) }
#endif

// clang-format on
template <typename T>
void print_pointer_atrributes(T ptr, const char* ptr_name = nullptr) {

    cudaPointerAttributes attr;
    cudaErr(cudaPointerGetAttributes(&attr, ptr));
    std::cerr << "\n";
    if ( ptr_name ) {
        std::cerr << "Pointer " << ptr_name << std::endl;
    }
    std::cerr << "Your pointer is for: ";
    switch ( attr.type ) {
        case 0:
            std::cerr << "Unmanaged memory" << std::endl;
            break;
        case 1:
            std::cerr << "Host memory" << std::endl;
            break;
        case 2:
            std::cerr << "Device memory" << std::endl;
            break;
        case 3:
            std::cerr << "Managed memory" << std::endl;
            break;
        default:
            std::cerr << "Unknown memory" << std::endl;
            break;
    }
    std::cerr << "\n";
    std::cerr << "with possible device address () " << attr.devicePointer << std::endl;
    std::cerr << "with possible host address () " << attr.hostPointer << std::endl;
    return;
}

// Limits for specific kernels
constexpr int ntds_x_WhitenPS = 32;
constexpr int ntds_y_WhitenPS = 32;
// Complex data type
typedef float2                            Complex;
static __device__ __host__ inline Complex ComplexAdd(Complex, Complex);
static __device__ __host__ inline void    ComplexScale(Complex*, float);
static __device__ __host__ inline Complex ComplexScale(Complex&, float&);
static __device__ __host__ inline Complex ComplexMul(Complex, Complex);
static __device__ __host__ inline Complex ComplexConjMul(Complex, Complex);
static __device__ __host__ inline Complex ComplexConjMulAndScale(Complex a, Complex b, float s);

////////////////////////////////////////////////////////////////////////////////
// Complex operations
////////////////////////////////////////////////////////////////////////////////

// Complex addition
static __device__ __host__ inline Complex ComplexAdd(Complex a, Complex b) {
    Complex c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}

// Complex scale
static __device__ __host__ inline Complex ComplexScale(Complex& a, float& s) {
    Complex c;
    c.x = s * a.x;
    c.y = s * a.y;
    return c;
}

// Complex scale
static __device__ __host__ inline void ComplexScale(Complex* a, float s) {
    a->x *= s;
    a->y *= s;
}

// Complex multiplication
static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b) {
    Complex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

// Complex multiplication
static __device__ __host__ inline Complex ComplexMulAndScale(Complex a, Complex b, float s) {
    Complex c;
    c.x = s * (a.x * b.x - a.y * b.y);
    c.y = s * (a.x * b.y + a.y * b.x);
    return c;
}

// Complex a * conj b multiplication
static __device__ __host__ inline Complex ComplexConjMul(Complex a, Complex b) {
    Complex c;
    c.x = a.x * b.x + a.y * b.y;
    c.y = a.y * b.x - a.x * b.y;
    return c;
}

// Complex a * conj b multiplication
static __device__ __host__ inline Complex ComplexConjMul(const Complex& a, Complex& b) {
    Complex c;
    c.x = a.x * b.x + a.y * b.y;
    c.y = a.y * b.x - a.x * b.y;
    return c;
}

// Complex a * conj b multiplication
static __device__ __host__ inline float RealPartOfComplexConjMul(Complex a, Complex b) {
    return a.x * b.x + a.y * b.y;
}

// Complex a * conj b multiplication
static __device__ __host__ inline float ComplexModulus(Complex a) {
    return sqrtf(float(a.x * a.x + a.y * a.y));
}

// Complex a * conj b multiplication
static __device__ __host__ inline float ComplexModulusSquared(Complex a) {
    return float(a.x * a.x + a.y * a.y);
}

// Complex a * conj b multiplication
static __device__ __host__ inline Complex ComplexConjMulAndScale(Complex a, Complex b, float s) {
    Complex c;
    c.x = s * (a.x * b.x + a.y * b.y);
    c.y = s * (a.y * b.x - a.x * b.y);
    return c;
}

// static constexpr int warpSize = 32;

#endif /* GPU_CORE_HEADERS_H_ */
