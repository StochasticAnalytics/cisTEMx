
#include <cistem_config.h>

#include "../../gpu/gpu_core_headers.h"
#include "../../gpu/GpuImage.h"

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
void EulerSearch::RunGPU<GpuImage>(Particle& particle, Image& input_3d, Image* projections, GpuImage* gpu_images) {

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
    int psi_m;
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
    Image*          correlation_map  = new Image;
    Image*          rotation_cache   = NULL;

    timer.start("Initial Allocations");
    flipped_image->Allocate(particle.particle_image->logical_x_dimension, particle.particle_image->logical_y_dimension, false);
    padded_image->Allocate(padding_factor_2d * flipped_image->logical_x_dimension, padding_factor_2d * flipped_image->logical_y_dimension, true);
    projection_image->Allocate(particle.particle_image->logical_x_dimension, particle.particle_image->logical_y_dimension, false);
    rotated_image->Allocate(particle.particle_image->logical_x_dimension, particle.particle_image->logical_y_dimension, false);
    correlation_map->Allocate(particle.particle_image->logical_x_dimension, particle.particle_image->logical_y_dimension, false);
    correlation_map->object_is_centred_in_box = false;
    timer.lap("Initial Allocations");
#ifndef MKL
    float* temp_k1 = new float[flipped_image->real_memory_allocated];
    float* temp_k2;
    temp_k2 = temp_k1 + 1;
    float* real_a;
    float* real_b;
    float* real_c;
    float* real_d;
    float* real_r;
    float* real_i;
    real_a = projection_image->real_values;
    real_b = projection_image->real_values + 1;
    real_r = correlation_map->real_values;
    real_i = correlation_map->real_values + 1;
#endif

    //	if (resolution_limit != 0.0)
    //	{
    //		effective_bfactor = 2.0 * 4.0 * powf(1.0 / resolution_limit, 2);
    //	}
    //	else
    //	{
    //		effective_bfactor = 0.0;
    //	}

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

    timer.start("allocate rotation cache");
    rotation_cache = new Image[psi_i];
    for ( i = 0; i < psi_i; i++ ) {
        rotation_cache[i].Allocate(particle.particle_image->logical_x_dimension, particle.particle_image->logical_y_dimension, false);
    }
    timer.lap("allocate rotation cache");

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
    psi_m = 0;
    for ( psi_i = 0; psi_i < number_of_psi_positions; psi_i++ ) {
        //		wxPrintf("rotation_cache[psi_m].logical_z_dimension = %i\n", rotation_cache[psi_m].logical_z_dimension);
        if ( parameter_map.psi ) {
            //			flipped_image->RotateFourier2DFromIndex(rotation_cache[psi_m], kernel_index[psi_i]);
            angles.GenerateRotationMatrix2D(psi_i * psi_step + psi_start);
        }
        else {
            angles.GenerateRotationMatrix2D(psi_start);
            //			flipped_image->RotateFourier2D(rotation_cache[psi_m], angles);
        }
        padded_image->Rotate2DSample(rotation_cache[psi_m], angles);
        rotation_cache[psi_m].ForwardFFT( );
        rotation_cache[psi_m].SwapRealSpaceQuadrants( );

        psi_m++;
        if ( test_mirror ) {
            rotation_cache[psi_m - 1].MirrorYFourier2D(rotation_cache[psi_m]);
            psi_m++;
        }
    }
    timer.lap("Make rotation cache");

    for ( i = 0; i < number_of_search_positions; i++ ) {
        if ( projections == NULL ) {
            //			wxPrintf("i, phi, theta = %i, %f, %f\n", i, list_of_search_parameters[i][0], list_of_search_parameters[i][1]);
            timer.start("Make Projection");
            angles.Init(list_of_search_parameters[i][0], list_of_search_parameters[i][1], 0.0, 0.0, 0.0);
            input_3d.ExtractSlice(*projection_image, angles, resolution_limit);
            projection_image->Whiten(resolution_limit);
            projection_image->ApplyBFactor(effective_bfactor);
            timer.lap("Make Projection");
        }
        else {
            timer.start("Copy back projection");
            projection_image->CopyFrom(&projections[i]);
            timer.lap("Copy back projection");
        }

#ifndef MKL
        for ( pixel_counter = 0; pixel_counter < flipped_image->real_memory_allocated; pixel_counter += 2 ) {
            temp_k1[pixel_counter] = real_a[pixel_counter] + real_b[pixel_counter];
        };
        for ( pixel_counter = 0; pixel_counter < flipped_image->real_memory_allocated; pixel_counter += 2 ) {
            temp_k2[pixel_counter] = real_b[pixel_counter] - real_a[pixel_counter];
        };
#endif

        //		projection_image->SwapRealSpaceQuadrants();
        //		rotation_cache[0].SwapRealSpaceQuadrants();
        //		projection_image->QuickAndDirtyWriteSlice("proj.mrc", 1);
        //		rotation_cache[0].QuickAndDirtyWriteSlice("part.mrc", 1);
        //		exit(0);

        timer.start("Cross Correlate");
        best_inplane_score = -std::numeric_limits<float>::max( );
        psi_m              = 0;
        for ( psi_i = 0; psi_i < number_of_psi_positions; psi_i++ ) {
#ifndef MKL
            real_c = rotation_cache[psi_m].real_values;
            real_d = rotation_cache[psi_m].real_values + 1;
#endif

#ifdef MKL
            // Use the MKL
            vmcMulByConj(flipped_image->real_memory_allocated / 2, reinterpret_cast<MKL_Complex8*>(projection_image->complex_values), reinterpret_cast<MKL_Complex8*>(rotation_cache[psi_m].complex_values), reinterpret_cast<MKL_Complex8*>(correlation_map->complex_values), VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
#else
            for ( pixel_counter = 0; pixel_counter < flipped_image->real_memory_allocated; pixel_counter += 2 ) {
                real_r[pixel_counter] = real_a[pixel_counter] * (real_c[pixel_counter] - real_d[pixel_counter]) + real_d[pixel_counter] * temp_k1[pixel_counter];
            };
            for ( pixel_counter = 0; pixel_counter < flipped_image->real_memory_allocated; pixel_counter += 2 ) {
                real_i[pixel_counter] = real_a[pixel_counter] * (real_c[pixel_counter] - real_d[pixel_counter]) + real_c[pixel_counter] * temp_k2[pixel_counter];
            };
#endif
            correlation_map->is_in_real_space  = false;
            correlation_map->complex_values[0] = 0.0;
            timer.lap("Cross Correlate");

            timer.start("FFT Correlation Map");
            correlation_map->BackwardFFT( );
            timer.lap("FFT Correlation Map");

            timer.start("Fine Peak Search");
            found_peak = correlation_map->FindPeakAtOriginFast2D(max_pix_x, max_pix_y);
            timer.lap("Fine Peak Search");
            //				sample_rate++;
            //				projection_image->SwapRealSpaceQuadrants();
            //				rotation_cache[psi_m].SwapRealSpaceQuadrants();
            //				projection_image->QuickAndDirtyWriteSlice("proj.mrc", sample_rate);
            //				rotation_cache[psi_m].QuickAndDirtyWriteSlice("part.mrc", sample_rate);
            //				projection_image->SwapRealSpaceQuadrants();
            //				rotation_cache[psi_m].SwapRealSpaceQuadrants();
            //				wxPrintf("peak  = %g  psi = %g  theta = %g  phi = %g  x = %g  y = %g\n", found_peak.value, 360.0 - (psi_i * psi_step + psi_start),
            //						list_of_search_parameters[i][1], list_of_search_parameters[i][0], found_peak.x, found_peak.y);
            if ( found_peak.value > best_inplane_score ) {
                best_inplane_score     = found_peak.value;
                best_inplane_values[0] = 360.0 - (psi_i * psi_step + psi_start);
                best_inplane_values[1] = found_peak.x;
                best_inplane_values[2] = found_peak.y;
                mirrored_match         = false;
            }

            if ( test_mirror ) {
                psi_m++;
                timer.start("Cross Correlate");
#ifndef MKL
                real_c = rotation_cache[psi_m].real_values;
                real_d = rotation_cache[psi_m].real_values + 1;
#endif

#ifdef MKL
                // Use the MKL
                vmcMulByConj(flipped_image->real_memory_allocated / 2, reinterpret_cast<MKL_Complex8*>(projection_image->complex_values), reinterpret_cast<MKL_Complex8*>(rotation_cache[psi_m].complex_values), reinterpret_cast<MKL_Complex8*>(correlation_map->complex_values), VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
#else
                for ( pixel_counter = 0; pixel_counter < flipped_image->real_memory_allocated; pixel_counter += 2 ) {
                    real_r[pixel_counter] = real_a[pixel_counter] * (real_c[pixel_counter] - real_d[pixel_counter]) + real_d[pixel_counter] * temp_k1[pixel_counter];
                };
                for ( pixel_counter = 0; pixel_counter < flipped_image->real_memory_allocated; pixel_counter += 2 ) {
                    real_i[pixel_counter] = real_a[pixel_counter] * (real_c[pixel_counter] - real_d[pixel_counter]) + real_c[pixel_counter] * temp_k2[pixel_counter];
                };
#endif
                correlation_map->is_in_real_space  = false;
                correlation_map->complex_values[0] = 0.0;
                timer.lap("Cross Correlate");

                timer.start("FFT Correlation Map");
                correlation_map->BackwardFFT( );
                timer.lap("FFT Correlation Map");

                timer.start("Fine Peak Search");
                found_peak = correlation_map->FindPeakAtOriginFast2D(max_pix_x, max_pix_y);
                timer.lap("Fine Peak Search");
                //					sample_rate++;
                //					projection_image->SwapRealSpaceQuadrants();
                //					rotation_cache[psi_m].SwapRealSpaceQuadrants();
                //					projection_image->QuickAndDirtyWriteSlice("proj.mrc", sample_rate);
                //					rotation_cache[psi_m].QuickAndDirtyWriteSlice("part.mrc", sample_rate);
                //					projection_image->SwapRealSpaceQuadrants();
                //					rotation_cache[psi_m].SwapRealSpaceQuadrants();
                //					wxPrintf("peakm = %g  psi = %g  theta = %g  phi = %g  x = %g  y = %g\n", found_peak.value, 360.0 - (psi_i * psi_step + psi_start),
                //							list_of_search_parameters[i][1] + 180.0, list_of_search_parameters[i][0], found_peak.x, found_peak.y);
                if ( found_peak.value > best_inplane_score ) {
                    best_inplane_score     = found_peak.value;
                    best_inplane_values[0] = 360.0 - (psi_i * psi_step + psi_start);
                    best_inplane_values[1] = found_peak.x;
                    best_inplane_values[2] = found_peak.y;
                    mirrored_match         = true;
                }
            }
            //			}
            psi_m++;
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
    // *******************************
    /*	if (particle.origin_micrograph < 0) particle.origin_micrograph = 0;
	particle.origin_micrograph++;
	particle.particle_image->QuickAndDirtyWriteSlice("part.mrc", particle.origin_micrograph);
	angles.Init(list_of_best_parameters[1][0], list_of_best_parameters[1][1], list_of_best_parameters[1][2], list_of_best_parameters[1][3], list_of_best_parameters[1][4]);
	wxPrintf("params, score = %i %g %g %g %g %g %g\n", particle.origin_micrograph, list_of_best_parameters[1][0], list_of_best_parameters[1][1], list_of_best_parameters[1][2], list_of_best_parameters[1][3], list_of_best_parameters[1][4], list_of_best_parameters[1][5]);
	input_3d.ExtractSlice(*projection_image, angles, resolution_limit);
	projection_image->PhaseShift(angles.ReturnShiftX() / particle.pixel_size, angles.ReturnShiftY() / particle.pixel_size);
	projection_image->SwapRealSpaceQuadrants();
	projection_image->QuickAndDirtyWriteSlice("proj.mrc", particle.origin_micrograph); */

    timer.start("Clean up");
    delete flipped_image;
    delete padded_image;
    delete projection_image;
    delete rotated_image;
    delete correlation_map;
    psi_i = number_of_psi_positions;
    if ( test_mirror )
        psi_i *= 2;
    for ( i = 0; i < psi_i; ++i ) {
        rotation_cache[i].Deallocate( );
    }
    delete[] rotation_cache;
#ifndef MKL
    delete[] temp_k1;
#endif
    timer.lap("Clean up");
}
