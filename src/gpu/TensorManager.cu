#include "gpu_core_headers.h"

#include "TensorManager.h"

template <class TypeA, class TypeB, class TypeC, class TypeD, class ComputeType>
TensorManager<TypeA, TypeB, TypeC, TypeD, ComputeType>::TensorManager( ) {

    cuTensorErr(cutensorInit(&handle));
    SetDefaultValues<TypeA, TypeB, TypeC, TypeD, ComputeType>( );
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
template <int ArraySize, class ArrayType>
bool TensorManager<TypeA, TypeB, TypeC, TypeD, ComputeType>::CheckForSetMetaData(ArrayType& is_property_set) {

    for ( int i = 0; i < ArraySize; i++ ) {
        if ( is_tensor_active[i] && ! is_property_set[i] ) {
            return false;
            break;
        }
    }
    return true;
}

template <class TypeA, class TypeB, class TypeC, class TypeD, class ComputeType>
TensorManager<TypeA, TypeB, TypeC, TypeD, ComputeType>::SetDefaultValues( ) {
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

    for ( i = 0; i != is_tensor_managed.size( ); ++i ) {
        is_set_unary_operator[i] = false;
    }

    for ( i = 0; i != is_set_unary_operator.size( ); ++i ) {
        is_set_n_elements_in_each_tensor[i] = false;
    }

    for ( i = 0; i != n_elements_in_each_tensor.size( ); ++i ) {
        n_elements_in_each_tensor[i] = 1;
    }

    for ( i = 0; i != is_set_n_elements_in_each_tensor.size( ); ++i ) {
        is_set_n_elements_in_each_tensor[i] = false;
    }
}

template <class TypeA, class TypeB, class TypeC, class TypeD, class ComputeType>
TensorManager<TypeA, TypeB, TypeC, TypeD, ComputeType>::SetTensorDescriptors( ) {

    MyDebugAssertTrue(CheckForSetMetaData(is_set_n_modes), "Set n_modes first");
    MyDebugAssertTrue(CheckForSetMetaData(is_set_extents_of_each_tensor), "Set extents of each tensor first");
    MyDebugAssertTrue(CheckNumberOfThreads(is_set_unary_operator), "Set unary operator first");

    for ( std::size_t i = 0; i != tensor_descriptor.size( ); ++i ) {
        if ( is_tensor_active[i] ) {
            cuTensorErr(cutensorInitTensorDescriptor(&handle,
                                                     &tensor_descriptor[i],
                                                     n_modes[i],
                                                     extents_of_each_tensor[i].data( ),
                                                     NULL /* stride assuming a packed layout including FFTW padding*/,
                                                     TypeA,
                                                     unary_operator[i]));
        }
    }
}

// template <class TypeA, class TypeB, class TypeC, class TypeD, class ComputeType>
// TensorManager<TypeA, TypeB, TypeC, TypeD, ComputeType>::TensorManager( ) {
// }
