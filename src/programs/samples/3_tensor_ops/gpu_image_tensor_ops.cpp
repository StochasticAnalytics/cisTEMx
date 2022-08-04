#include <cistem_config.h>

#ifdef ENABLEGPU
#include "../../../gpu/gpu_core_headers.h"
#else
#error "GPU is not enabled"
#include "../../../core/core_headers.h"
#endif

#include "../../../gpu/GpuImage.h"

#include "../common/common.h"
#include "gpu_image_tensor_ops.h"

bool DoBasicTensorOpsTest(const wxString& hiv_images_80x80x10_filename, wxString& temp_directory) {

    bool passed;
    bool all_passed = true;

    SamplesPrintTestStartMessage("Starting basic tensor ops tests:", false);

    all_passed = all_passed && TestCudaSample(hiv_images_80x80x10_filename, temp_directory);
    all_passed = all_passed && TestTensorManagerManual(hiv_images_80x80x10_filename, temp_directory);

    SamplesBeginTest("Tensor ops overall", passed);
    SamplesPrintResult(all_passed, __LINE__);

    return all_passed;
}

bool TestCudaSample(const wxString& hiv_images_80x80x10_filename, wxString& temp_directory) {

    bool passed     = true;
    bool all_passed = true;

    SamplesBeginTest("Integration of libcutensor", passed);

    typedef float floatTypeA;
    typedef float floatTypeB;
    typedef float floatTypeC;
    typedef float floatTypeCompute;

    cudaDataType_t        typeA       = CUDA_R_32F;
    cudaDataType_t        typeC       = CUDA_R_32F;
    cutensorComputeType_t typeCompute = CUTENSOR_COMPUTE_32F;

    floatTypeCompute alpha = (floatTypeCompute)1.0f;
    floatTypeCompute beta  = (floatTypeCompute)0.f;

    /**********************
     * Computing (full) reduction : C_{} = alpha * A_{m,h,k,v} + beta * C_{m,v}
     *********************/

    std::vector<int32_t> modeA{'m', 'h', 'k'};
    std::vector<int32_t> modeC{'n'};
    int32_t              nmodeA = modeA.size( );
    int32_t              nmodeC = modeC.size( );

    std::unordered_map<int32_t, int64_t> extent;
    extent['m'] = 196;
    extent['h'] = 256;
    extent['k'] = 64;
    extent['n'] = 1;

    std::vector<int64_t> extentC;
    for ( auto mode : modeC )
        extentC.push_back(extent[mode]);
    std::vector<int64_t> extentA;
    for ( auto mode : modeA )
        extentA.push_back(extent[mode]);

    /**********************
     * Allocating data
     *********************/

    size_t elementsA = 1;
    for ( auto mode : modeA )
        elementsA *= extent[mode];
    size_t elementsC = 1;
    for ( auto mode : modeC )
        elementsC *= extent[mode];

    size_t sizeA = sizeof(floatTypeA) * elementsA;
    size_t sizeC = sizeof(floatTypeC) * elementsC;

    void *A_d, *C_d;
    cudaErr(cudaMalloc((void**)&A_d, sizeA));
    cudaErr(cudaMalloc((void**)&C_d, sizeC));

    floatTypeA* A = (floatTypeA*)malloc(sizeof(floatTypeA) * elementsA);
    floatTypeC* C = (floatTypeC*)malloc(sizeof(floatTypeC) * elementsC);

    if ( A == NULL || C == NULL ) {
        wxPrintf("Error: Host allocation of A, B, or C.\n");
        return -1;
    }

    /*******************
     * Initialize data
     *******************/

    // Record initialization values to compare to the results of the tensor ops.
    double sum_a = 0.0;
    double sum_c = 0.0;
    size_t n_sum = 0;
    double max_a = std::numeric_limits<floatTypeA>::min( );
    double min_a = std::numeric_limits<floatTypeA>::max( );
    for ( int64_t i = 0; i < elementsA; i++ ) {
        A[i] = (((float)rand( )) / RAND_MAX - 0.5) * 100;
        sum_a += A[i];
        n_sum++;
        max_a = (max_a > A[i]) ? max_a : A[i];
        min_a = (min_a < A[i]) ? min_a : A[i];
    }

    for ( int64_t i = 0; i < elementsC; i++ ) {
        C[i] = (((float)rand( )) / RAND_MAX - 0.5) * 100;
        sum_c += C[i];
    }

    cudaErr(cudaMemcpy(C_d, C, sizeC, cudaMemcpyHostToDevice));
    cudaErr(cudaMemcpy(A_d, A, sizeA, cudaMemcpyHostToDevice));
    /*************************
     * cuTENSOR
     *************************/

    cutensorHandle_t handle;
    cuTensorErr(cutensorInit(&handle));

    /**********************
     * Create Tensor Descriptors
     **********************/
    cutensorTensorDescriptor_t descA;
    cuTensorErr(cutensorInitTensorDescriptor(&handle,
                                             &descA,
                                             nmodeA,
                                             extentA.data( ),
                                             NULL /* stride */,
                                             typeA, CUTENSOR_OP_IDENTITY));

    cutensorTensorDescriptor_t descC;
    cuTensorErr(cutensorInitTensorDescriptor(&handle,
                                             &descC,
                                             nmodeC,
                                             extentC.data( ),
                                             NULL /* stride */,
                                             typeC, CUTENSOR_OP_IDENTITY));

    const cutensorOperator_t opReduce  = CUTENSOR_OP_ADD;
    const cutensorOperator_t opCompute = CUTENSOR_OP_MUL;
    const cutensorOperator_t opMax     = CUTENSOR_OP_MAX;
    const cutensorOperator_t opMin     = CUTENSOR_OP_MIN;
    /**********************
     * Querry workspace
     **********************/
    uint64_t worksize = 0;
    cuTensorErr(cutensorReductionGetWorkspaceSize(&handle,
                                                  A_d, &descA, modeA.data( ),
                                                  C_d, &descC, modeC.data( ),
                                                  C_d, &descC, modeC.data( ),
                                                  opReduce, typeCompute, &worksize));
    void* work = nullptr;
    if ( worksize > 0 ) {
        if ( cudaSuccess != cudaMalloc(&work, worksize) ) {
            work     = nullptr;
            worksize = 0;
        }
    }
    /**********************
     * Run
     **********************/
    cutensorStatus_t err;
    for ( int i = 0; i < 3; ++i ) {
        cudaErr(cudaMemcpy(C_d, C, sizeC, cudaMemcpyHostToDevice));
        cudaErr(cudaDeviceSynchronize( ));

        err = cutensorReduction(&handle,
                                (const void*)&alpha, A_d, &descA, modeA.data( ),
                                (const void*)&beta, C_d, &descC, modeC.data( ),
                                C_d, &descC, modeC.data( ),
                                opReduce, typeCompute, work, worksize, 0 /* stream */);

        if ( err != CUTENSOR_STATUS_SUCCESS ) {
            wxPrintf("ERROR: %s in line %d\n", cutensorGetErrorString(err), __LINE__);
        }
    }
    /*************************/

    cudaErr(cudaDeviceSynchronize( ));
    cudaErr(cudaMemcpy(C, C_d, sizeC, cudaMemcpyDeviceToHost));

    // Check the reduced value
    passed = passed && FloatsAreAlmostTheSame(C[0] / n_sum, sum_a / n_sum);

    worksize = 0;
    cuTensorErr(cutensorReductionGetWorkspaceSize(&handle,
                                                  A_d, &descA, modeA.data( ),
                                                  C_d, &descC, modeC.data( ),
                                                  C_d, &descC, modeC.data( ),
                                                  opMax, typeCompute, &worksize));
    work = nullptr;
    if ( worksize > 0 ) {
        if ( cudaSuccess != cudaMalloc(&work, worksize) ) {
            work     = nullptr;
            worksize = 0;
        }
    }

    cudaErr(cudaMemcpy(C_d, C, sizeC, cudaMemcpyHostToDevice));
    cudaErr(cudaDeviceSynchronize( ));

    err = cutensorReduction(&handle,
                            (const void*)&alpha, A_d, &descA, modeA.data( ),
                            (const void*)&beta, C_d, &descC, modeC.data( ),
                            C_d, &descC, modeC.data( ),
                            opMax, typeCompute, work, worksize, 0 /* stream */);

    cudaErr(cudaDeviceSynchronize( ));
    cudaErr(cudaMemcpy(C, C_d, sizeC, cudaMemcpyDeviceToHost));

    // Chec the max value found
    passed = passed && FloatsAreAlmostTheSame(C[0], max_a);

    worksize = 0;
    cuTensorErr(cutensorReductionGetWorkspaceSize(&handle,
                                                  A_d, &descA, modeA.data( ),
                                                  C_d, &descC, modeC.data( ),
                                                  C_d, &descC, modeC.data( ),
                                                  opMin, typeCompute, &worksize));
    work = nullptr;
    if ( worksize > 0 ) {
        if ( cudaSuccess != cudaMalloc(&work, worksize) ) {
            work     = nullptr;
            worksize = 0;
        }
    }

    cudaErr(cudaMemcpy(C_d, C, sizeC, cudaMemcpyHostToDevice));
    cudaErr(cudaDeviceSynchronize( ));

    err = cutensorReduction(&handle,
                            (const void*)&alpha, A_d, &descA, modeA.data( ),
                            (const void*)&beta, C_d, &descC, modeC.data( ),
                            C_d, &descC, modeC.data( ),
                            opMin, typeCompute, work, worksize, 0 /* stream */);

    cudaErr(cudaDeviceSynchronize( ));
    cudaErr(cudaMemcpy(C, C_d, sizeC, cudaMemcpyDeviceToHost));

    // Check the min value found
    passed = passed && FloatsAreAlmostTheSame(C[0], min_a);

    if ( A )
        free(A);
    if ( C )
        free(C);
    if ( A_d )
        cudaFree(A_d);
    if ( C_d )
        cudaFree(C_d);
    if ( work )
        cudaFree(work);

    all_passed = all_passed && passed;
    SamplesTestResult(passed);

    return all_passed;
}

bool TestTensorManagerManual(const wxString& hiv_images_80x80x10_filename, wxString& temp_directory) {
    bool passed     = true;
    bool all_passed = true;

    SamplesBeginTest("TensorManager manual setup", passed);

    TensorManager my_tm<float, float, float, float>;

    my_tm.SetAlphaAndBeta(1.f, 0.f);

    my_tm.SetModes(cistem::gpu::tensor_id::A, 'm', 'h', 'k');
    my_tm.SetModes(cistem::gpu::tensor_id::C, 'n');

    // int32_t nmodeA = modeA.size( );
    // int32_t nmodeC = modeC.size( );

    my_tm.SetExtent('m', 196);
    my_tm.SetExtent('h', 256);
    my_tm.SetExtent('k', 64);
    my_tm.SetExtent('n', 1);

    my_tm.SetExtentOfTensor(cistem::gpu::tensor_id::A);
    my_tm.SetExtentOfTensor(cistem::gpu::tensor_id::C);

    my_tm.SetNElementsForAllActiveTensors( );
    /**********************
     * Allocating data
     *********************/

    // I'm not quite sure how I want to deal with FFTW padding yet, but for now, we will
    //    1) assume it is there
    //    2) assum it is included in the extent size (hence the -2 in the image allocation)
    Image    image_A;
    GpuImage gpu_image_A;

    float  C   = 0;
    float* d_C = nullptr;

    image_A.Allocate(my_tm.GetExtent('m') - 2, my_tm.GetExtent('h'), my_tm.GetExtent('k'), true);
    gpu_image_A.Init(image_A);

    size_t sizeA = sizeof(floatTypeA) * elementsA;
    size_t sizeC = sizeof(floatTypeC) * elementsC;

    cudaErr(cudaMalloc((void**)&d_C, my_tm.GetNElementsForTensor(cistem::gpu::tensor_id::C)));

    /*******************
     * Initialize data
     *******************/

    // Record initialization values to compare to the results of the tensor ops.

    image_A.FillWithNoiseFromNormalDistribution(0.f, 1.f);
    EmpiricalDistribution my_dist = image_A.ReturnDistributionOfRealValues( );

    float mean_a = my_dist.GetSampleMean( );
    float min_a  = my_dist.GetSampleMin( );
    float max_a  = my_dist.GetSampleMax( );
    float n_a    = my_dist.GetSampleCount( );

    gpu_image_A.CopyHostToDevice(true);

    /**********************
     * Create Tensor Descriptors
     **********************/
    cutensorTensorDescriptor_t descA;
    cuTensorErr(cutensorInitTensorDescriptor(&handle,
                                             &descA,
                                             nmodeA,
                                             extentA.data( ),
                                             NULL /* stride */,
                                             typeA, CUTENSOR_OP_IDENTITY));

    cutensorTensorDescriptor_t descC;
    cuTensorErr(cutensorInitTensorDescriptor(&handle,
                                             &descC,
                                             nmodeC,
                                             extentC.data( ),
                                             NULL /* stride */,
                                             typeC, CUTENSOR_OP_IDENTITY));

    const cutensorOperator_t opReduce  = CUTENSOR_OP_ADD;
    const cutensorOperator_t opCompute = CUTENSOR_OP_MUL;
    const cutensorOperator_t opMax     = CUTENSOR_OP_MAX;
    const cutensorOperator_t opMin     = CUTENSOR_OP_MIN;
    /**********************
     * Querry workspace
     **********************/
    uint64_t worksize = 0;
    cuTensorErr(cutensorReductionGetWorkspaceSize(&handle,
                                                  A_d, &descA, modeA.data( ),
                                                  C_d, &descC, modeC.data( ),
                                                  C_d, &descC, modeC.data( ),
                                                  opReduce, typeCompute, &worksize));
    void* work = nullptr;
    if ( worksize > 0 ) {
        if ( cudaSuccess != cudaMalloc(&work, worksize) ) {
            work     = nullptr;
            worksize = 0;
        }
    }
    /**********************
     * Run
     **********************/
    cutensorStatus_t err;
    for ( int i = 0; i < 3; ++i ) {
        cudaErr(cudaMemcpy(C_d, C, sizeC, cudaMemcpyHostToDevice));
        cudaErr(cudaDeviceSynchronize( ));

        err = cutensorReduction(&handle,
                                (const void*)&alpha, A_d, &descA, modeA.data( ),
                                (const void*)&beta, C_d, &descC, modeC.data( ),
                                C_d, &descC, modeC.data( ),
                                opReduce, typeCompute, work, worksize, 0 /* stream */);

        if ( err != CUTENSOR_STATUS_SUCCESS ) {
            wxPrintf("ERROR: %s in line %d\n", cutensorGetErrorString(err), __LINE__);
        }
    }
    /*************************/

    cudaErr(cudaDeviceSynchronize( ));
    cudaErr(cudaMemcpy(C, C_d, sizeC, cudaMemcpyDeviceToHost));

    // Check the reduced value
    passed = passed && FloatsAreAlmostTheSame(C[0] / n_sum, sum_a / n_sum);

    worksize = 0;
    cuTensorErr(cutensorReductionGetWorkspaceSize(&handle,
                                                  A_d, &descA, modeA.data( ),
                                                  C_d, &descC, modeC.data( ),
                                                  C_d, &descC, modeC.data( ),
                                                  opMax, typeCompute, &worksize));
    work = nullptr;
    if ( worksize > 0 ) {
        if ( cudaSuccess != cudaMalloc(&work, worksize) ) {
            work     = nullptr;
            worksize = 0;
        }
    }

    cudaErr(cudaMemcpy(C_d, C, sizeC, cudaMemcpyHostToDevice));
    cudaErr(cudaDeviceSynchronize( ));

    err = cutensorReduction(&handle,
                            (const void*)&alpha, A_d, &descA, modeA.data( ),
                            (const void*)&beta, C_d, &descC, modeC.data( ),
                            C_d, &descC, modeC.data( ),
                            opMax, typeCompute, work, worksize, 0 /* stream */);

    cudaErr(cudaDeviceSynchronize( ));
    cudaErr(cudaMemcpy(C, C_d, sizeC, cudaMemcpyDeviceToHost));

    // Chec the max value found
    passed = passed && FloatsAreAlmostTheSame(C[0], max_a);

    worksize = 0;
    cuTensorErr(cutensorReductionGetWorkspaceSize(&handle,
                                                  A_d, &descA, modeA.data( ),
                                                  C_d, &descC, modeC.data( ),
                                                  C_d, &descC, modeC.data( ),
                                                  opMin, typeCompute, &worksize));
    work = nullptr;
    if ( worksize > 0 ) {
        if ( cudaSuccess != cudaMalloc(&work, worksize) ) {
            work     = nullptr;
            worksize = 0;
        }
    }

    cudaErr(cudaMemcpy(C_d, C, sizeC, cudaMemcpyHostToDevice));
    cudaErr(cudaDeviceSynchronize( ));

    err = cutensorReduction(&handle,
                            (const void*)&alpha, A_d, &descA, modeA.data( ),
                            (const void*)&beta, C_d, &descC, modeC.data( ),
                            C_d, &descC, modeC.data( ),
                            opMin, typeCompute, work, worksize, 0 /* stream */);

    cudaErr(cudaDeviceSynchronize( ));
    cudaErr(cudaMemcpy(C, C_d, sizeC, cudaMemcpyDeviceToHost));

    // Check the min value found
    passed = passed && FloatsAreAlmostTheSame(C[0], min_a);

    if ( A )
        free(A);
    if ( C )
        free(C);
    if ( A_d )
        cudaFree(A_d);
    if ( C_d )
        cudaFree(C_d);
    if ( work )
        cudaFree(work);

    all_passed = all_passed && passed;
    SamplesTestResult(passed);

    return all_passed;
}