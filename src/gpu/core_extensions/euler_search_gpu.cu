
#include <cistem_config.h>

#include "../../gpu/gpu_core_headers.h"
#include "../../gpu/GpuImage.h"

#include "../../programs/refine3d/batched_search.h"

// #define USE_CUTENSOR_FOR_REDUCTION
// #define SYNC_TIMER

/**
 * @brief Runs a brute force search over a pre-specified range of Euler angles.
 * 
 * 1) For in-plane angles at each position on the euler sphere, an image cache of 2d rotations is created. If testing mirrors, these are included in the cache.
 *     a) Cache is generated from the _particle_ image, which is multiplied by the CTF, then padded, iFFT, and rotated in position space, FFT, real space quadrant swapped.
 *     b) if testing mirrors, that mirror is next in sequence in the cache.
 * 2) The search positions are then looped over the euler sphere, at each position doing a conjugate multiplication out of place (to re-use the projection cache and reference projection).
 * 3) The score and translation are taken from the position space peak after a backward transform.
 * 
 * 
 * @param particle 
 * @param input_3d 
 * @param projections 
 */
template <>
void EulerSearch::Run<GpuImage>(Particle& particle, Image& input_3d, GpuImage* projections, int batch_size) {

    MyDebugAssertTrue(number_of_search_positions > 0, "EulerSearch not initialized");
    MyDebugAssertTrue(particle.particle_image->is_in_memory, "Particle image not allocated");
    MyDebugAssertTrue(input_3d.is_in_memory, "3D reference map not allocated");
    //	MyDebugAssertTrue(particle.particle_image->logical_x_dimension == input_3d.logical_x_dimension && particle.particle_image->logical_y_dimension == input_3d.logical_y_dimension, "Error: Image and 3D reference incompatible");

    int i;
    int j;
    int k;
    //	int sample_rate = 0;
    int pixel_counter;
    int psi_i;
    int number_of_psi_positions;
    int max_pix_x         = max_search_x / particle.pixel_size;
    int max_pix_y         = max_search_y / particle.pixel_size;
    int padding_factor_2d = 4;
    //	float psi;
    float           best_inplane_score;
    float           best_inplane_values[3];
    float           temp_float[6];
    float           effective_bfactor = 100.0;
    bool            mirrored_match;
    Peak            found_peak;
    AnglesAndShifts angles;
    Image*          flipped_image    = new Image;
    Image*          padded_image     = new Image;
    Image*          projection_image = new Image;
    Image*          rotated_image    = new Image;
    GpuImage*       rotation_cache   = NULL;

    BatchedSearch batch;

    timer.start("Initial Allocations");
    flipped_image->Allocate(particle.particle_image->logical_x_dimension, particle.particle_image->logical_y_dimension, false);
    padded_image->Allocate(padding_factor_2d * flipped_image->logical_x_dimension, padding_factor_2d * flipped_image->logical_y_dimension, true);
    projection_image->Allocate(particle.particle_image->logical_x_dimension, particle.particle_image->logical_y_dimension, false);
    rotated_image->Allocate(particle.particle_image->logical_x_dimension, particle.particle_image->logical_y_dimension, false);

    // Setup and allocate our gpu image, but do not pin the ost memory since we'll copy from many different images in the intial implementation.
    GpuImage gpu_projection_image, gpu_correlation_map;
    // For some reason the implicit broadcasting is not working?? Just do it explicitly for the projection_image
    gpu_projection_image.Allocate(particle.particle_image->logical_x_dimension, particle.particle_image->logical_y_dimension, 1, false);
    gpu_correlation_map.Allocate(particle.particle_image->logical_x_dimension, particle.particle_image->logical_y_dimension, batch_size, false);
    timer.lap("Initial Allocations");

    for ( i = 0; i < best_parameters_to_keep + 1; ++i ) {
        ZeroFloatArray(list_of_best_parameters[i], 5);
        list_of_best_parameters[i][5] = -std::numeric_limits<float>::max( );
    }

    if ( parameter_map.psi ) {
        number_of_psi_positions = myroundint(psi_max / psi_step);
        if ( number_of_psi_positions < 1 )
            number_of_psi_positions = 1;
    }
    else {
        number_of_psi_positions = 1;
    }
    psi_i = number_of_psi_positions;
    if ( test_mirror )
        psi_i *= 2;

    batch.Init(gpu_projection_image, psi_i, batch_size, test_mirror, max_pix_x, max_pix_y);

    timer.start("Allocate rotation cache");
    rotation_cache = new GpuImage[batch.n_batches( )];
    for ( batch.index = 0; batch.index < batch.n_batches( ); batch.index++ ) {
        rotation_cache[batch.index].Allocate(particle.particle_image->logical_x_dimension, particle.particle_image->logical_y_dimension, batch.n_images_in_this_batch( ), false);
    }

    Image tmp_rot;
    Image mirrored;

    tmp_rot.Allocate(particle.particle_image->logical_x_dimension, particle.particle_image->logical_y_dimension, false);
    if ( test_mirror ) {
        mirrored.CopyFrom(&tmp_rot);
    }
    timer.lap("Allocate rotation cache");

    timer.start("Pin host pointers");
    // Pin the temporary images for fast transfer
    float* pinned_rot_ptr;
    float* pinned_mirrored_ptr;
    // FIXME for now always pin the memory - this might be a bad choice for single copy or small images, but is required for asynch xfer and is ~2x as fast after pinning

    cudaErr(cudaHostRegister(tmp_rot.real_values, sizeof(float) * batch.stride( ), cudaHostRegisterDefault));
    cudaErr(cudaHostGetDevicePointer(&pinned_rot_ptr, tmp_rot.real_values, 0));
    if ( test_mirror ) {
        cudaErr(cudaHostRegister(mirrored.real_values, sizeof(float) * batch.stride( ), cudaHostRegisterDefault));
        cudaErr(cudaHostGetDevicePointer(&pinned_mirrored_ptr, mirrored.real_values, 0));
    }
    timer.lap("Pin host pointers");

    timer.start("Copy CTF Pad iFFT");
    flipped_image->CopyFrom(particle.particle_image);
    //	flipped_image->MultiplyPixelWiseReal(*particle.ctf_image, particle.is_phase_flipped);
    flipped_image->MultiplyPixelWiseReal(*particle.ctf_image, true);
    flipped_image->ClipInto(padded_image);
    // Not clear of CorrectSinc helps here since it might actually be better
    // if the periphery of the image is a bit attenuated for the search
    //	padded_image.CorrectSinc();
    padded_image->BackwardFFT( );
    timer.lap("Copy CTF Pad iFFT");

    timer.start("Make rotation cache");
    psi_i = 0;

    for ( batch.index = 0; batch.index < batch.n_batches( ); batch.index++ ) {
        for ( int intra_batch_idx = 0; intra_batch_idx < batch.n_images_in_this_batch( ); intra_batch_idx += batch.intra_loop_inc( ) ) {
            // Reset these flags since we are re-using the same buffer
            tmp_rot.is_in_real_space         = true;
            tmp_rot.object_is_centred_in_box = true;

            if ( parameter_map.psi ) {
                angles.GenerateRotationMatrix2D(psi_i * psi_step + psi_start);
                batch.add_angle_and_mirror(psi_i * psi_step + psi_start, false);
            }
            else {
                angles.GenerateRotationMatrix2D(psi_start);
                batch.add_angle_and_mirror(psi_start, false);
            }

            padded_image->Rotate2DSample(tmp_rot, angles);
            tmp_rot.ForwardFFT( );
            tmp_rot.SwapRealSpaceQuadrants( );
            tmp_rot.complex_values[0] = 0.0f + I * 0.0f;
            // std::cerr << "batch.index and intra_batch_idx and calc: " << batch.index << " " << intra_batch_idx << " " << batch.stride( ) << std::endl;
            cudaErr(cudaMemcpy(&rotation_cache[batch.index].real_values_gpu[batch.stride( ) * intra_batch_idx], tmp_rot.real_values, sizeof(float) * batch.stride( ), cudaMemcpyHostToDevice));

            if ( test_mirror ) {
                mirrored.MirrorYFourier2D(tmp_rot);
                cudaErr(cudaMemcpy(&rotation_cache[batch.index].real_values_gpu[batch.stride( ) * (intra_batch_idx + 1)], mirrored.real_values, sizeof(float) * batch.stride( ), cudaMemcpyHostToDevice));
                batch.add_angle_and_mirror(true);
            }
            psi_i++;
        }
    }
    timer.lap("Make rotation cache");

#ifdef USE_CUTENSOR_FOR_REDUCTION
    timer.start("making a fp32 VA mask");
    Image mask;
    mask.Allocate(particle.particle_image->logical_x_dimension, particle.particle_image->logical_y_dimension, 1, true);
    mask.SetToConstant(0.0f);
    mask.object_is_centred_in_box = false;
    mask.FindPeakAtOriginFast2DMask(max_pix_x, max_pix_y);

    GpuImage gpu_mask;
    gpu_mask.Init(mask, true, true);
    gpu_mask.CopyHostToDevice(true);
    timer.lap("making a fp32 VA mask");

    timer.start("Setting up the TensorManager");
    namespace cg   = cistem::gpu;
    using TensorID = cg::tensor_id::Enum;
    using TensorOP = cg::tensor_op::Enum;

    // TensorManager<float, float, float, float, float> my_tm;
    using ComputeType = float;
    TensorManager<ComputeType, ComputeType, ComputeType, ComputeType, ComputeType> my_tm;
    my_tm.SetAlphaAndBeta(1.f, 0.f);

    my_tm.SetModes<'x', 'y'>(TensorID::A);
    my_tm.SetModes<'n'>(TensorID::B);

    // int32_t nmodeA = modeA.size( );
    // int32_t nmodeC = modeC.size( );

    my_tm.SetExtent('x', gpu_correlation_map.dims.w);
    my_tm.SetExtent('y', gpu_correlation_map.dims.y);
    // my_tm.SetExtent('z', gpu_correlation_map.dims.z);
    my_tm.SetExtent('n', 1);

    my_tm.SetExtentOfTensor(TensorID::A);
    my_tm.SetExtentOfTensor(TensorID::B);

    my_tm.SetNElementsForAllActiveTensors( );

    float        output_max   = 0.f;
    ComputeType* d_output_max = nullptr;

    cudaErr(cudaMalloc((void**)&d_output_max, sizeof(ComputeType) * my_tm.GetNElementsInTensor(TensorID::B)));

    my_tm.TensorIsAllocated(TensorID::A);
    my_tm.TensorIsAllocated(TensorID::B);

    gpu_correlation_map.ConvertToHalfPrecision(false);
    my_tm.SetTensorPointers(TensorID::A, reinterpret_cast<ComputeType*>(gpu_correlation_map.real_values));
    my_tm.SetTensorPointers(TensorID::B, d_output_max);

    my_tm.SetUnaryOperator(TensorID::A, CUTENSOR_OP_IDENTITY);
    my_tm.SetUnaryOperator(TensorID::B, CUTENSOR_OP_IDENTITY);
    my_tm.SetTensorOperation(TensorOP::reduction);
    my_tm.SetTensorDescriptors( );

    /**********************
     * Querry workspace
     **********************/
    std::cerr << "Querying workspace" << std::endl;
    my_tm.GetWorkSpaceSize( );
    test_mirror = false; // FIXME: just for profiling
    timer.lap("Setting up the TensorManager");
#endif

    for ( int iSearchPosition = 0; iSearchPosition < number_of_search_positions; iSearchPosition++ ) {
        if ( projections == NULL ) {
            MyAssertTrue(false, "No projections provided, the gpu codepath is not setup for this scenario");
            // timer.start("Make Projection");
            // angles.Init(list_of_search_parameters[i][0], list_of_search_parameters[i][1], 0.0, 0.0, 0.0);
            // input_3d.ExtractSlice(*projection_image, angles, resolution_limit);
            // projection_image->Whiten(resolution_limit);
            // projection_image->ApplyBFactor(effective_bfactor);
            // timer.lap("Make Projection");
            // timer.start("Copy Projection");
            // gpu_projection_image.CopyHostToDevice(true);
            // timer.lap("Copy Projection");
        }
        else {
            timer.start("Copy back projection");
            // Explicitly broadcast for now until I can figure out what is broken.
            // for ( int iTest = 0; iTest < batch_size; iTest++ ) {
            //     cudaErr(cudaMemcpy(&gpu_projection_image.real_values_gpu[batch.stride() * iTest], projections[iSearchPosition].real_values, sizeof(float) * batch.stride(), cudaMemcpyDeviceToDevice));
            // }
            gpu_projection_image.CopyFrom(&projections[iSearchPosition]);
            timer.lap("Copy back projection");
        }

        best_inplane_score = -std::numeric_limits<float>::max( );
        for ( batch.index = 0; batch.index < batch.n_batches( ); batch.index++ ) {

            timer.start("Cross Correlate");

            gpu_correlation_map.is_in_real_space = false;

            // dims.z of calling image (roation cache) determines what extent of the correlation map to use
            rotation_cache[batch.index].MultiplyPixelWiseComplexConjugate(gpu_projection_image, gpu_correlation_map);
#ifdef SYNC_TIMER
            timer.lap_sync("Cross Correlate");
#else
            timer.lap("Cross Correlate");
#endif

            timer.start("FFT Correlation Map");
            // std::cerr << "wanted_batch_size: " << wanted_batch_size << std::endl;
            gpu_correlation_map.BackwardFFTBatched(batch.n_images_in_this_batch( ));
#ifdef SYNC_TIMER
            timer.lap_sync("FFT Correlation Map");
#else
            timer.lap("FFT Correlation Map");
#endif

            gpu_correlation_map.is_in_real_space         = true;
            gpu_correlation_map.object_is_centred_in_box = false;

#ifdef USE_CUTENSOR_FOR_REDUCTION
            timer.start("Masking correlation map");
            gpu_correlation_map.MultiplyPixelWise(gpu_mask);
            cudaStreamSynchronize(cudaStreamPerThread);

            timer.lap("Masking correlation map");

            // my_tm.my_types._compute_type
            timer.start("Finding peak with cuTensor");
            cuTensorErr(cutensorReduction(&my_tm.handle,
                                          (const void*)&my_tm.alpha, my_tm.my_ptrs._a_ptr, &my_tm.tensor_descriptor[TensorID::A], my_tm.modes[TensorID::A].data( ),
                                          (const void*)&my_tm.beta, my_tm.my_ptrs._b_ptr, &my_tm.tensor_descriptor[TensorID::B], my_tm.modes[TensorID::B].data( ),
                                          my_tm.my_ptrs._b_ptr, &my_tm.tensor_descriptor[TensorID::B], my_tm.modes[TensorID::B].data( ),
                                          my_tm.cutensor_op, CUTENSOR_COMPUTE_16F, my_tm.workspace_ptr, my_tm.workspace_size, cudaStreamPerThread));

            cudaStreamSynchronize(cudaStreamPerThread);
            cudaErr(cudaMemcpy(&output_max, d_output_max, sizeof(ComputeType), cudaMemcpyDeviceToHost));
            timer.lap("Finding peak with cuTensor");
#endif
            timer.start("Fine Peak Search");
            found_peak = gpu_correlation_map.FindPeakAtOriginFast2D(&batch);
#ifdef SYNC_TIMER
            timer.lap_sync("Fine Peak Search");
#else
            timer.lap("Fine Peak Search");
#endif

            // Note that found_peak.physical_address_within_image is not set in this method
            if ( found_peak.value > best_inplane_score ) {
                best_inplane_score     = found_peak.value;
                best_inplane_values[0] = 360.0 - batch.GetInPlaneAngle(myroundint(found_peak.z));
                best_inplane_values[1] = found_peak.x;
                best_inplane_values[2] = found_peak.y;
                mirrored_match         = batch.GetMirroredOrNot(myroundint(found_peak.z));
            }
        }

        if ( best_inplane_score > list_of_best_parameters[best_parameters_to_keep][5] ) {
            list_of_best_parameters[best_parameters_to_keep][5] = best_inplane_score;
            if ( mirrored_match ) {
                list_of_best_parameters[best_parameters_to_keep][0] = list_of_search_parameters[i][0];
                list_of_best_parameters[best_parameters_to_keep][1] = list_of_search_parameters[i][1] + 180.0;
                list_of_best_parameters[best_parameters_to_keep][2] = best_inplane_values[0];
                list_of_best_parameters[best_parameters_to_keep][3] = (best_inplane_values[1] * cosf(deg_2_rad(best_inplane_values[0])) - best_inplane_values[2] * sinf(deg_2_rad(best_inplane_values[0]))) * particle.pixel_size;
                list_of_best_parameters[best_parameters_to_keep][4] = (-best_inplane_values[1] * sinf(deg_2_rad(best_inplane_values[0])) - best_inplane_values[2] * cosf(deg_2_rad(best_inplane_values[0]))) * particle.pixel_size;
            }
            else {
                /* Because we found the peak in the rotated reference frame, we neet to rotate the shifts back, convert from pixels to angstrom and also negate.
                [ cos + sin      [ - x , -y ]' * pixel_size
                              * 
                -sin + cos ]
                */
                list_of_best_parameters[best_parameters_to_keep][0] = list_of_search_parameters[i][0];
                list_of_best_parameters[best_parameters_to_keep][1] = list_of_search_parameters[i][1];
                list_of_best_parameters[best_parameters_to_keep][2] = best_inplane_values[0];
                list_of_best_parameters[best_parameters_to_keep][3] = (-best_inplane_values[1] * cosf(deg_2_rad(best_inplane_values[0])) - best_inplane_values[2] * sinf(deg_2_rad(best_inplane_values[0]))) * particle.pixel_size;
                list_of_best_parameters[best_parameters_to_keep][4] = (best_inplane_values[1] * sinf(deg_2_rad(best_inplane_values[0])) - best_inplane_values[2] * cosf(deg_2_rad(best_inplane_values[0]))) * particle.pixel_size;
            }
        }
        for ( j = best_parameters_to_keep; j > 1; j-- ) {
            if ( list_of_best_parameters[j][5] > list_of_best_parameters[j - 1][5] ) {
                for ( k = 0; k < 6; k++ ) {
                    temp_float[k] = list_of_best_parameters[j - 1][k];
                }
                for ( k = 0; k < 6; k++ ) {
                    list_of_best_parameters[j - 1][k] = list_of_best_parameters[j][k];
                }
                for ( k = 0; k < 6; k++ ) {
                    list_of_best_parameters[j][k] = temp_float[k];
                }
                //				wxPrintf("best_inplane_score = %i %g\n", j - 1, list_of_best_parameters[j - 1][5]);
            }
            else {
                break;
            }
        }
    }
    /*******************************/

    // float best_score = 0.0f;
    // float best_x, best_y;
    // for ( int i = 0; i < best_parameters_to_keep; i++ ) {
    //     if ( list_of_best_parameters[i][5] > best_score ) {
    //         best_score = list_of_best_parameters[i][5];
    //         best_x     = list_of_best_parameters[i][3];
    //         best_y     = list_of_best_parameters[i][4];
    //     }
    // }
    // wxPrintf("Best Score is %f, at (%f %f)\n", best_score, best_x, best_y);

    timer.start("Clean up");
    delete flipped_image;
    delete padded_image;
    delete rotated_image;
    delete projection_image;

    delete[] rotation_cache;

    cudaErr(cudaHostUnregister(pinned_rot_ptr));
    if ( test_mirror ) {
        cudaErr(cudaHostUnregister(pinned_mirrored_ptr));
    }

    timer.lap("Clean up");
}
