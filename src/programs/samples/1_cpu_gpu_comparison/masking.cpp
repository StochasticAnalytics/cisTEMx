#include <cistem_config.h>

#ifdef ENABLEGPU
#include "../../../gpu/gpu_core_headers.h"
#else
#error "GPU is not enabled"
#include "../../../core/core_headers.h"
#endif

#include "../../../gpu/GpuImage.h"

#include "../common/common.h"
#include "masking.h"

#define HANDLE_ERROR(x)                                           \
    {                                                             \
        const auto err = x;                                       \
        if ( err != CUTENSOR_STATUS_SUCCESS ) {                   \
            wxPrintf("Error: %s\n", cutensorGetErrorString(err)); \
            return err;                                           \
        }                                                         \
    };

struct GPUTimer {
    GPUTimer( ) {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
        cudaEventRecord(start_, 0);
    }

    ~GPUTimer( ) {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start( ) {
        cudaEventRecord(start_, 0);
    }

    float seconds( ) {
        cudaEventRecord(stop_, 0);
        cudaEventSynchronize(stop_);
        float time;
        cudaEventElapsedTime(&time, start_, stop_);
        return time * 1e-3;
    }

  private:
    cudaEvent_t start_, stop_;
};

bool CPUvsGPUMaskingTest(const wxString& hiv_image_80x80x1_filename, wxString& temp_directory) {

    bool passed;
    bool all_passed = true;

    SamplesPrintTestStartMessage("Starting CPU vs GPU masking tests:", false);

    all_passed = all_passed && DoCosineMaskingTest(hiv_image_80x80x1_filename, temp_directory);
    all_passed = all_passed && DoTestCuTensorCompilation(hiv_image_80x80x1_filename, temp_directory);
    all_passed = all_passed && DoTestCuTensorReduction(hiv_image_80x80x1_filename, temp_directory);

    SamplesBeginTest("CPU vs GPU overall", passed);
    SamplesPrintResult(all_passed, __LINE__);
    wxPrintf("\n\n");

    return all_passed;
}

bool DoCosineMaskingTest(const wxString& hiv_image_80x80x1_filename, wxString& temp_directory) {

    bool passed     = true;
    bool all_passed = true;

    SamplesBeginTest("Cosine mask real space", passed);

    wxString tmp_img_filename = temp_directory + "/tmp1.mrc";

    MRCFile input_file(hiv_image_80x80x1_filename.ToStdString( ), false);
    MRCFile output_file(tmp_img_filename.ToStdString( ), false);

    Image    cpu_image;
    Image    gpu_host_image;
    GpuImage gpu_image;

    cpu_image.ReadSlice(&input_file, 1);
    gpu_host_image.ReadSlice(&input_file, 1);

    gpu_image.Init(gpu_host_image);
    gpu_image.CopyHostToDevice( );

    float wanted_mask_radius;
    float wanted_mask_edge;
    bool  invert;
    bool  force_mask_value;
    float wanted_mask_value;

    RandomNumberGenerator my_rand(pi_v<float>);

    int n_loops = 1;
    for ( int i = 0; i < n_loops; i++ ) {

        // Make some random parameters.
        wanted_mask_radius = 0.f; // GetUniformRandomSTD(0.0f, cpu_image.logical_x_dimension / 2.0f);
        wanted_mask_edge   = 20.f; //GetUniformRandomSTD(0.0f, 20.0f);
        wanted_mask_value  = 0.f; //GetUniformRandomSTD(0.0f, 1.0f);
        if ( my_rand.GetUniformRandomSTD(0.0f, 1.0f > 0.5f) ) {
            invert = true;
        }
        else {
            invert = false;
        }
        if ( my_rand.GetUniformRandomSTD(0.0f, 1.0f > 0.5f) ) {
            force_mask_value = true;
        }
        else {
            force_mask_value = false;
        }

        // FIXME: for intial run, fix the values.
        invert           = false;
        force_mask_value = false;

        cpu_image.CosineMask(wanted_mask_radius, wanted_mask_edge, invert, force_mask_value, wanted_mask_value);
        // gpu_image.CosineMask(wanted_mask_radius, wanted_mask_edge, invert, force_mask_value, wanted_mask_value);
    }

    all_passed = all_passed && passed;
    SamplesTestResult(passed);

    return all_passed;
}

// almost verbatim from the nvidia samples - positive control for compiling/linking cutensor.
bool DoTestCuTensorCompilation(const wxString& hiv_image_80x80x1_filename, const wxString& temp_directory) {

    cistem_timer::StopWatch timer;

    timer.start("descriptors");
    typedef float floatTypeA;
    typedef float floatTypeB;
    typedef float floatTypeC;
    typedef float floatTypeCompute;

    cudaDataType_t        typeA       = CUDA_R_32F;
    cudaDataType_t        typeC       = CUDA_R_32F;
    cutensorComputeType_t typeCompute = CUTENSOR_COMPUTE_32F;

    floatTypeCompute alpha = (floatTypeCompute)1.1f;
    floatTypeCompute beta  = (floatTypeCompute)0.f;

    /**********************
     * Computing (partial) reduction : C_{m,v} = alpha * A_{m,h,k,v} + beta * C_{m,v}
     *********************/

    std::vector<int32_t> modeA{'m', 'h', 'k', 'v'};
    std::vector<int32_t> modeC{'m', 'v'};
    int32_t              nmodeA = modeA.size( );
    int32_t              nmodeC = modeC.size( );

    std::unordered_map<int32_t, int64_t> extent;
    extent['m'] = 196;
    extent['v'] = 64;
    extent['h'] = 256;
    extent['k'] = 64;

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
    wxPrintf("Total memory: %.2f GiB\n", (sizeA + sizeC) / 1024. / 1024. / 1024);
    timer.lap("descriptors");

    timer.start("allocation");
    void *A_d, *C_d;
    cudaErr(cudaMalloc((void**)&A_d, sizeA));
    cudaErr(cudaMalloc((void**)&C_d, sizeC));

    floatTypeA* A = (floatTypeA*)malloc(sizeof(floatTypeA) * elementsA);
    floatTypeC* C = (floatTypeC*)malloc(sizeof(floatTypeC) * elementsC);

    if ( A == NULL || C == NULL ) {
        wxPrintf("Error: Host allocation of A, B, or C.\n");
        return -1;
    }

    timer.lap("allocation");
    /*******************
     * Initialize data
     *******************/

    timer.start("initialization");
    for ( int64_t i = 0; i < elementsA; i++ )
        A[i] = (((float)rand( )) / RAND_MAX - 0.5) * 100;
    for ( int64_t i = 0; i < elementsC; i++ )
        C[i] = (((float)rand( )) / RAND_MAX - 0.5) * 100;
    timer.lap("initialization");

    timer.start("copy");
    cudaErr(cudaMemcpy(C_d, C, sizeC, cudaMemcpyHostToDevice));
    cudaErr(cudaMemcpy(A_d, A, sizeA, cudaMemcpyHostToDevice));
    timer.lap("copy");
    /*************************
     * cuTENSOR
     *************************/

    timer.start("cutensor handle init");
    cutensorHandle_t handle;
    HANDLE_ERROR(cutensorInit(&handle));

    timer.lap("cutensor handle init");
    /**********************
     * Create Tensor Descriptors
     **********************/
    timer.start("cutensor descriptor init");
    cutensorTensorDescriptor_t descA;
    HANDLE_ERROR(cutensorInitTensorDescriptor(&handle,
                                              &descA,
                                              nmodeA,
                                              extentA.data( ),
                                              NULL /* stride */,
                                              typeA, CUTENSOR_OP_IDENTITY));

    cutensorTensorDescriptor_t descC;
    HANDLE_ERROR(cutensorInitTensorDescriptor(&handle,
                                              &descC,
                                              nmodeC,
                                              extentC.data( ),
                                              NULL /* stride */,
                                              typeC, CUTENSOR_OP_IDENTITY));

    const cutensorOperator_t opReduce  = CUTENSOR_OP_ADD;
    const cutensorOperator_t opCompute = CUTENSOR_OP_MUL;
    /**********************
     * Querry workspace
     **********************/
    timer.start("cutensor workspace query");
    uint64_t worksize = 0;
    HANDLE_ERROR(cutensorReductionGetWorkspace(&handle,
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
    timer.lap("cutensor workspace query");
    /**********************
     * Run
     **********************/
    timer.start("cutensor reduction");
    double           minTimeCUTENSOR = 1e100;
    cutensorStatus_t err;
    for ( int i = 0; i < 3; ++i ) {
        cudaErr(cudaMemcpy(C_d, C, sizeC, cudaMemcpyHostToDevice));
        cudaErr(cudaDeviceSynchronize( ));

        // Set up timing
        GPUTimer timer;
        timer.start( );

        err = cutensorReduction(&handle,
                                (const void*)&alpha, A_d, &descA, modeA.data( ),
                                (const void*)&beta, C_d, &descC, modeC.data( ),
                                C_d, &descC, modeC.data( ),
                                opReduce, typeCompute, work, worksize, 0 /* stream */);

        // Synchronize and measure timing
        auto time = timer.seconds( );

        if ( err != CUTENSOR_STATUS_SUCCESS ) {
            wxPrintf("ERROR: %s in line %d\n", cutensorGetErrorString(err), __LINE__);
        }
        minTimeCUTENSOR = (minTimeCUTENSOR < time) ? minTimeCUTENSOR : time;
    }
    timer.lap("cutensor reduction");
    /*************************/

    double transferedBytes = sizeC + sizeA;
    transferedBytes += ((float)beta != 0.f) ? sizeC : 0;
    transferedBytes /= 1e9;
    wxPrintf("cuTensor: %.2f GB/s\n", transferedBytes / minTimeCUTENSOR);

    timer.start("cleanup");
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
    timer.lap("cleanup");

    timer.print_times( );
    return true;
}

bool DoTestCuTensorReduction(const wxString& hiv_image_80x80x1_filename, const wxString& temp_directory) {

    cistem_timer::StopWatch timer;

    timer.start("descriptors");
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
    wxPrintf("Total memory: %.2f GiB\n", (sizeA + sizeC) / 1024. / 1024. / 1024);
    timer.lap("descriptors");

    timer.start("allocation");
    void *A_d, *C_d;
    cudaErr(cudaMalloc((void**)&A_d, sizeA));
    cudaErr(cudaMalloc((void**)&C_d, sizeC));

    floatTypeA* A = (floatTypeA*)malloc(sizeof(floatTypeA) * elementsA);
    floatTypeC* C = (floatTypeC*)malloc(sizeof(floatTypeC) * elementsC);

    if ( A == NULL || C == NULL ) {
        wxPrintf("Error: Host allocation of A, B, or C.\n");
        return -1;
    }

    timer.lap("allocation");
    /*******************
     * Initialize data
     *******************/

    double sum_a = 0.0;
    double sum_c = 0.0;
    double max_a = std::numeric_limits<floatTypeA>::min( );
    double min_a = std::numeric_limits<floatTypeA>::max( );
    timer.start("initialization");
    for ( int64_t i = 0; i < elementsA; i++ ) {
        A[i] = (((float)rand( )) / RAND_MAX - 0.5) * 100;
        sum_a += A[i];
        max_a = (max_a > A[i]) ? max_a : A[i];
        min_a = (min_a < A[i]) ? min_a : A[i];
    }

    for ( int64_t i = 0; i < elementsC; i++ ) {
        C[i] = (((float)rand( )) / RAND_MAX - 0.5) * 100;
        sum_c += C[i];
    }

    timer.lap("initialization");

    wxPrintf("Sum of A: %.2f\n", sum_a);
    wxPrintf("Sum of C: %.2f\n", sum_c);
    wxPrintf("Max of A: %.2f\n", max_a);
    wxPrintf("Min of A: %.2f\n", min_a);

    timer.start("copy");
    cudaErr(cudaMemcpy(C_d, C, sizeC, cudaMemcpyHostToDevice));
    cudaErr(cudaMemcpy(A_d, A, sizeA, cudaMemcpyHostToDevice));
    timer.lap("copy");
    /*************************
     * cuTENSOR
     *************************/

    timer.start("cutensor handle init");
    cutensorHandle_t handle;
    HANDLE_ERROR(cutensorInit(&handle));

    timer.lap("cutensor handle init");
    /**********************
     * Create Tensor Descriptors
     **********************/
    timer.start("cutensor descriptor init");
    cutensorTensorDescriptor_t descA;
    HANDLE_ERROR(cutensorInitTensorDescriptor(&handle,
                                              &descA,
                                              nmodeA,
                                              extentA.data( ),
                                              NULL /* stride */,
                                              typeA, CUTENSOR_OP_IDENTITY));

    cutensorTensorDescriptor_t descC;
    HANDLE_ERROR(cutensorInitTensorDescriptor(&handle,
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
    timer.start("cutensor workspace query");
    uint64_t worksize = 0;
    HANDLE_ERROR(cutensorReductionGetWorkspace(&handle,
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
    timer.lap("cutensor workspace query");
    /**********************
     * Run
     **********************/
    timer.start("cutensor reduction");
    double           minTimeCUTENSOR = 1e100;
    cutensorStatus_t err;
    for ( int i = 0; i < 3; ++i ) {
        cudaErr(cudaMemcpy(C_d, C, sizeC, cudaMemcpyHostToDevice));
        cudaErr(cudaDeviceSynchronize( ));

        // Set up timing
        GPUTimer timer;
        timer.start( );

        err = cutensorReduction(&handle,
                                (const void*)&alpha, A_d, &descA, modeA.data( ),
                                (const void*)&beta, C_d, &descC, modeC.data( ),
                                C_d, &descC, modeC.data( ),
                                opReduce, typeCompute, work, worksize, 0 /* stream */);

        // Synchronize and measure timing
        auto time = timer.seconds( );

        if ( err != CUTENSOR_STATUS_SUCCESS ) {
            wxPrintf("ERROR: %s in line %d\n", cutensorGetErrorString(err), __LINE__);
        }
        minTimeCUTENSOR = (minTimeCUTENSOR < time) ? minTimeCUTENSOR : time;
    }
    timer.lap("cutensor reduction");
    /*************************/

    cudaErr(cudaDeviceSynchronize( ));
    cudaErr(cudaMemcpy(C, C_d, sizeC, cudaMemcpyDeviceToHost));
    wxPrintf("Reduced value in C is %.2f\n", C[0]);

    worksize = 0;
    HANDLE_ERROR(cutensorReductionGetWorkspace(&handle,
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
    wxPrintf("Reduced max value in C is %.2f\n", C[0]);

    worksize = 0;
    HANDLE_ERROR(cutensorReductionGetWorkspace(&handle,
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
    wxPrintf("Reduced min value in C is %.2f\n", C[0]);

    timer.start("cleanup");
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
    timer.lap("cleanup");

    timer.print_times( );
    return true;
}