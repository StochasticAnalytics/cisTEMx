#include "gpu_core_headers.h"

#include "TensorManager.h"

template <class TypeA, class TypeB, class TypeC, class TypeD, class ComputeType>
TensorManager<TypeA, TypeB, TypeC, TypeD, ComputeType>::TensorManager( ) {

    cuTensorErr(cutensorInit(&handle));
    SetDefaultValues( );
}

template <class TypeA, class TypeB, class TypeC, class TypeD, class ComputeType>
TensorManager<TypeA, TypeB, TypeC, TypeD, ComputeType>::TensorManager(const GpuImage& wanted_props) {
    // Set all the properties of the tensor manager based on the reference GpuImage.
    cuTensorErr(cutensorInit(&handle));
}

template <class TypeA, class TypeB, class TypeC, class TypeD, class ComputeType>
TensorManager<TypeA, TypeB, TypeC, TypeD, ComputeType>::~TensorManager( ) {

    // TODO: should there be any cleanup of the handle?
}

template <class TypeA, class TypeB, class TypeC, class TypeD, class ComputeType>
template <class ArrayType>
bool TensorManager<TypeA, TypeB, TypeC, TypeD, ComputeType>::CheckForSetMetaData(ArrayType& is_property_set) {

    for ( int i = 0; i < cg::max_tensor_manager_tensors; i++ ) {
        if ( is_tensor_active[i] && ! is_property_set[i] ) {
            return false;
            break;
        }
    }
    return true;
}

template <class TypeA, class TypeB, class TypeC, class TypeD, class ComputeType>
void TensorManager<TypeA, TypeB, TypeC, TypeD, ComputeType>::SetDefaultValues( ) {
    alpha = ComputeType(0);
    beta  = ComputeType(0);

    workspace_size = 0;
    workspace_ptr  = nullptr;

    std::size_t i;

    for ( i = 0; i != is_tensor_allocated.size( ); ++i ) {
        is_tensor_allocated[i] = false;
    }

    for ( i = 0; i != is_tensor_active.size( ); ++i ) {
        is_tensor_active[i] = false;
    }

    for ( i = 0; i != is_set_tensor_descriptor.size( ); ++i ) {
        is_set_tensor_descriptor[i] = false;
    }

    for ( i = 0; i != is_set_unary_operator.size( ); ++i ) {
        is_set_unary_operator[i] = false;
    }

    for ( i = 0; i != is_set_n_elements_in_each_tensor.size( ); ++i ) {
        is_set_n_elements_in_each_tensor[i] = false;
    }

    for ( i = 0; i != n_elements_in_each_tensor.size( ); ++i ) {
        n_elements_in_each_tensor[i] = 1;
    }

    for ( i = 0; i != is_set_n_elements_in_each_tensor.size( ); ++i ) {
        is_set_n_elements_in_each_tensor[i] = false;
    }

    SetTensorCudaType<TypeA>(TensorID::A);
    SetTensorCudaType<TypeB>(TensorID::B);
    SetTensorCudaType<TypeC>(TensorID::C);
    SetTensorCudaType<TypeD>(TensorID::D);
}

template <class TypeA, class TypeB, class TypeC, class TypeD, class ComputeType>
template <class ThisType>
void TensorManager<TypeA, TypeB, TypeC, TypeD, ComputeType>::SetTensorCudaType(TensorID tid) {

    if ( std::is_same_v<ThisType, float> )
        tensor_cuda_types[tid] = CUDA_R_32F;
    return;

    if ( std::is_same_v<ThisType, float2> )
        tensor_cuda_types[tid] = CUDA_C_32F;
    return;

    if ( std::is_same_v<ThisType, nv_bfloat16> )
        tensor_cuda_types[tid] = CUDA_R_16BF;
    return;

    if ( std::is_same_v<ThisType, nv_bfloat162> )
        tensor_cuda_types[tid] = CUDA_C_16BF;
    return;

    if ( std::is_same_v<ThisType, __half> )
        tensor_cuda_types[tid] = CUDA_R_16F;
    return;

    if ( std::is_same_v<ThisType, __half2> )
        tensor_cuda_types[tid] = CUDA_C_16F;
    return;

    // If we got here there is a problem.

    std::cerr << "Error: TensorManager::SetTensorCudaType: Unsupported type.\n";
    exit(1);
}

template <class TypeA, class TypeB, class TypeC, class TypeD, class ComputeType>
void TensorManager<TypeA, TypeB, TypeC, TypeD, ComputeType>::SetTensorDescriptors( ) {

    MyDebugAssertTrue(CheckForSetMetaData(is_set_n_modes), "Set n_modes before setting tensor descriptors");
    MyDebugAssertTrue(CheckForSetMetaData(is_set_extents_of_each_tensor), "Set extents of each tensor before setting tensor descriptors");
    MyDebugAssertTrue(CheckForSetMetaData(is_set_unary_operator), "Set unary operator before setting tensor descriptors");

    for ( std::size_t i = 0; i != tensor_descriptor.size( ); ++i ) {
        if ( is_tensor_active[i] ) {
            // std::cerr << "Trying to set descriptor for tensor " << i << "\n";
            // std::cerr << "n_modes: " << n_modes[i] << "\n";
            // for ( auto v : extents_of_each_tensor[i] ) {
            //     std::cerr << "extent is " << v << "\n";
            // }
            // std::cerr << "Cuda data type is " << tensor_cuda_types[i] << "\n";
            // std::cerr << "unary operator is " << unary_operator[i] << "\n";
            cuTensorErr(cutensorInitTensorDescriptor(&handle,
                                                     &tensor_descriptor[i],
                                                     n_modes[i],
                                                     extents_of_each_tensor[i].data( ),
                                                     NULL /* stride assuming a packed layout including FFTW padding*/,
                                                     tensor_cuda_types[i],
                                                     unary_operator[i]));
        }
    }
}

// template <class TypeA, class TypeB, class TypeC, class TypeD, class ComputeType>
// template <char Mode>
// void TensorManager<TypeA, TypeB, TypeC, TypeD, ComputeType>::SetModes(TensorID tid) {
//     modes[tid].push_back(Mode);
//     n_modes[tid]        = modes[tid].size( );
//     is_set_modes[tid]   = true;
//     is_set_n_modes[tid] = true;
//     // MyDebugAssert(modes[tid].size( ) <= cg::max_tensor_manager_dimensions, "Too many modes for a given tensor ID.");
//     extent_of_each_mode.try_emplace(Mode, 0); // This will be checked later for proper setting, but don't overwrite if it already exists
//     is_tensor_active[tid] = true;
// };

// template <class TypeA, class TypeB, class TypeC, class TypeD, class ComputeType>
// template <char Mode, char... OtherModes>
// void TensorManager<TypeA, TypeB, TypeC, TypeD, ComputeType>::SetModes(TensorID tid) {
//     modes[tid].push_back(Mode);
//     SetModes<OtherModes...>(tid);
// };

// template <class TypeA, class TypeB, class TypeC, class TypeD, class ComputeType>
// TensorManager<TypeA, TypeB, TypeC, TypeD, ComputeType>::TensorManager( ) {
// }

// So we can do separate compilation
template class TensorManager<float, float, float, float, float>;

// template void TensorManager<float, float, float, float, float>::SetModes<char>(TensorID tid);
// template void TensorManager<float, float, float, float, float>::SetModes<char, char>(TensorID tid);
// template void TensorManager<float, float, float, float, float>::SetModes<char, char, char>(TensorID tid);
// template void TensorManager<float, float, float, float, float>::SetModes<char, char, char, char>(TensorID tid);
