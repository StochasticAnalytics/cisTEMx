/*
Provide an interface to the cuTensor library to the cistem GpuImage class
*/

#ifndef _SRC_GPU_TENSORMANAGER_H_
#define _SRC_GPU_TENSORMANAGER_H_

#include <cutensor.h>
#include <cistem_config.h>
#include "../core/cistem_constants.h"

class GpuImage;

template <typename TypeA, typename TypeB, typename TypeC, typename TypeD, typename TypeCompute>
struct TensorTypes {
    // Use this to catch unsupported input/ compute types and throw exception.
    bool _a_type       = false;
    bool _b_type       = false;
    bool _c_type       = false;
    bool _d_type       = false;
    bool _compute_type = false;
};

// Input real, compute single-precision
template <>
struct TensorTypes<float, float, float, float, float> {
    cudaDataType_t        _a_type       = CUDA_R_32F;
    cudaDataType_t        _b_type       = CUDA_R_32F;
    cudaDataType_t        _c_type       = CUDA_R_32F;
    cudaDataType_t        _d_type       = CUDA_R_32F;
    cutensorComputeType_t _compute_type = CUTENSOR_COMPUTE_32F;
};

template <class TypeA = float, class TypeB = float, class TypeC = float, class TypeD = float, class TypeCompute = float>
class TensorManager {

  public:
    TensorManager( );
    TensorManager(const GpuImage& wanted_props);
    ~TensorManager( );

    inline void SetMacroOP(cistem::gpu::tensor_op::Enum wanted_macro_op) { macro_op = wanted_macro_op; };

    inline void SetAlphaAndBeta(TypeCompute wanted_alpha, TypeCompute wanted_beta) {
        alpha = wanted_alpha;
        beta  = wanted_beta;
    };

    inline void SetExtent(char mode, int64_t wanted_extent) {
        auto search = extent_of_each_mode.find(mode);
        if ( search != extent_of_each_mode.end( ) ) {
            extent_of_each_mode[mode] = wanted_extent;
        }
        else {
            MyAssertTrue(false, "Could not find mode in wanted_extent");
        }
    };

    inline int64_t GetExtent(char mode) {
        auto search = extent_of_each_mode.find(mode);
        if ( search != extent_of_each_mode.end( ) ) {
            return extent_of_each_mode[mode];
        }
        else {
            MyAssertTrue(false, "Could not find mode in wanted_extent");
        }
    };

    template <cistem::gpu::tensor_id::Enum TID, char Mode>
    inline void SetModes(TID tid, Mode mode) {
        modes[tid].push_back(mode);
        n_modes[tid]        = modes[tid].size( );
        is_set_modes[tid]   = true;
        is_set_n_modes[tid] = true;
        MyDebugAssert(modes[tid].size( ) <= cistem::gpu::max_tensor_manager_dimensions, "Too many modes for a given tensor ID.");
        extent_of_each_mode.try_emplace(mode, 0); // This will be checked later for proper setting, but don't overwrite if it already exists
        is_tensor_active[tid] = true;
    };

    template <cistem::gpu::tensor_id::Enum TID, char Mode, char... OtherModes>
    inline void SetModes(TID tid, Mode mode, OtherModes... other_modes) {
        modes[tid].push_back(mode);
        SetModes<TID, OtherModes...>(tid, other_modes...);
    };

    inline void SetExtentOfTensor(cistem::gpu::tensor_id tid) {
        MyDebugAssertTrue(is_set_modes[tid], "Tensor ID not set.");
        MyDebugAssertTrue(is_set_n_modes[tid], "Number of modes not set.");
        for ( auto mode : modes[tid] )
            extents_of_each_tensor[tid].push_back(extent_of_each_mode[mode]);

        MyDebugAssertTrue(extents_of_each_tensor[tid].size( ) == n_modes[tid], "Number of extents not equal to number of modes.");
        is_set_extents_of_each_tensor[tid] = true;
    }

    inline void SetNElementsInEachTensor(cistem::gpu::tensor_id tid) {
        MyDebugAssertTrue(is_set_extents_of_each_tensor[tid], "Extents of each tensor not set.");
        MyDebugAssertFalse(is_set_n_elements_in_each_tensor[tid], "Number of elements in each tensor already set.");
        for ( auto extent : extents_of_each_tensor[tid] )
            n_elements_in_each_tensor[tid] *= extent;

        is_set_n_elements_in_each_tensor = true;
    }

    inline void SetNElementsForAllActiveTensors( ) {
        for ( int tid = 0; tid < cistem::gpu::max_tensor_manager_tensors; tid++ ) {
            if ( is_tensor_active[tid] )
                SetNElementsInEachTensor(tid);
        }
    }

    inline void SetUnaryOperator(cistem::gpu::tensor_id tid, cutensorOperator_t wanted_unary_op) {
        MyDebugAssertTrue(is_tensor_active[tid], "Tensor ID not active.");
        unary_op[tid]        = wanted_unary_op;
        is_set_unary_op[tid] = true;
    };

  private:
    std::array<std::vector<int32_t>, cistem::gpu::max_tensor_manager_tensors> modes;
    std::array<bool, cistem::gpu::max_tensor_manager_tensors>                 is_set_modes;

    std::array<int32_t, cistem::gpu::max_tensor_manager_tensors> n_modes;
    std::array<bool, cistem::gpu::max_tensor_manager_tensors>    is_set_n_modes;

    std::unordered_map<int32_t, int64_t> extent_of_each_mode;

    std::array<std::vector<int64_t>, cistem::gpu::max_tensor_manager_tensors> extents_of_each_tensor;
    std::array<bool, cistem::gpu::max_tensor_manager_tensors>                 is_set_extents_of_each_tensor;

    std::array<cutensorTensorDescriptor_t, cistem::gpu::max_tensor_manager_tensors> tensor_descriptor;
    std::array<bool, cistem::gpu::max_tensor_manager_tensors>                       is_set_tensor_descriptor;

    std::array<cutensorOperator_t, cistem::gpu::max_tensor_manager_tensors> unary_operator;
    std::array<bool, cistem::gpu::max_tensor_manager_tensors>               is_set_unary_operator;

    std::array<size_t, cistem::gpu::max_tensor_manager_tensors> n_elements_in_each_tensor;
    std::array<bool, cistem::gpu::max_tensor_manager_tensors>   is_set_n_elements_in_each_tensor;

    std::array<bool, cistem::gpu::max_tensor_manager_tensors> is_tensor_allocated;
    std::array<bool, cistem::gpu::max_tensor_manager_tensors> is_tensor_active;

    TensorTypes<TypeA, TypeB, TypeC, TypeD, TypeCompute> tensor_types;

    TypeCompute alpha;
    TypeCompute beta;

    cutensorHandle_t handle;

    template <int ArraySize, class ArrayType>
    bool CheckForSetMetaData(ArrayType& is_property_set);

    uint64_t workspace_size;
    void*    workspace_ptr;

    cistem::gpu::tensor_op::Enum macro_op;
};
#endif