
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
void EulerSearch::Run<GpuImage>(Particle& particle, Image& input_3d, GpuImage* projections) {

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
    GpuImage*       rotation_cache   = NULL;

    timer.start("Initial Allocations");
    flipped_image->Allocate(particle.particle_image->logical_x_dimension, particle.particle_image->logical_y_dimension, false);
    padded_image->Allocate(padding_factor_2d * flipped_image->logical_x_dimension, padding_factor_2d * flipped_image->logical_y_dimension, true);
    projection_image->Allocate(particle.particle_image->logical_x_dimension, particle.particle_image->logical_y_dimension, false);
    rotated_image->Allocate(particle.particle_image->logical_x_dimension, particle.particle_image->logical_y_dimension, false);

    timer.lap("Initial Allocations");

    // Setup and allocate our gpu image, but do not pin the ost memory since we'll copy from many different images in the intial implementation.
    GpuImage gpu_projection_image;
    gpu_projection_image.Init(*projection_image, false, true);

    Image correlation_map;
    correlation_map.Allocate(particle.particle_image->logical_x_dimension, particle.particle_image->logical_y_dimension, false);
    GpuImage gpu_correlation_map;
    gpu_correlation_map.Init(correlation_map, true, true);

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

    rotation_cache = new GpuImage[psi_i];
    Image tmp_rot;
    Image mirrored;

    tmp_rot.Allocate(particle.particle_image->logical_x_dimension, particle.particle_image->logical_y_dimension, false);
    if ( test_mirror ) {
        mirrored.CopyFrom(&tmp_rot);
    }

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
    psi_m                        = 0;
    bool should_pin_tmp_rot      = true;
    bool should_pin_mirrored_rot = true;
    for ( psi_i = 0; psi_i < number_of_psi_positions; psi_i++ ) {
        if ( parameter_map.psi ) {
            angles.GenerateRotationMatrix2D(psi_i * psi_step + psi_start);
        }
        else {
            angles.GenerateRotationMatrix2D(psi_start);
        }
        padded_image->Rotate2DSample(tmp_rot, angles);
        tmp_rot.ForwardFFT( );
        tmp_rot.SwapRealSpaceQuadrants( );
        tmp_rot.Conj( );
        tmp_rot.complex_values[0] = 0.0f + I * 0.0f;
        rotation_cache[psi_m].Init(tmp_rot, should_pin_tmp_rot, true);
        should_pin_tmp_rot = false;
        rotation_cache[psi_m].CopyHostToDevice(true);

        psi_m++;
        if ( test_mirror ) {

            mirrored.MirrorYFourier2D(tmp_rot);
            mirrored.Conj( );
            rotation_cache[psi_m - 1].Init(mirrored, should_pin_mirrored_rot, true);
            should_pin_mirrored_rot = false;
            rotation_cache[psi_m - 1].CopyHostToDevice(true);

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
            gpu_projection_image.CopyHostToDevice(*projection_image, false)
        }
        else {
            timer.start("Copy back projection");
            gpu_projection_image.CopyFrom(&projections[i]);
            timer.lap("Copy back projection");
        }
        timer.start("Copy cpu to gpu projection");

        timer.start("Cross Correlate");
        best_inplane_score = -std::numeric_limits<float>::max( );
        psi_m              = 0;
        for ( psi_i = 0; psi_i < number_of_psi_positions; psi_i++ ) {

            // Note 1: I have always taken the conjugate of the reference, but in the CPU code it is of the rotated particle. Here the conjgate is already taken in the cache step.
            gpu_projection_image.MultiplyPixelWise(rotation_cache[psi_m], gpu_correlation_map);

            gpu_correlation_map.is_in_real_space = false;
            timer.lap("Cross Correlate");

            timer.start("FFT Correlation Map");
            gpu_correlation_map.BackwardFFT( );
            timer.lap("FFT Correlation Map");

            timer.start("Copy Correlation Map");
            gpu_correlation_map.CopyHostToDevice(true);
            timer.lap("Copy Correlation Map");

            timer.start("Fine Peak Search");
            found_peak = correlation_map.FindPeakAtOriginFast2D(max_pix_x, max_pix_y);
            timer.lap("Fine Peak Search");

            if ( found_peak.value > best_inplane_score ) {
                best_inplane_score = found_peak.value;
                // In the image method, the image is rotated int he projection cache, (not the reference), so we need the negative of the best in-plane angle.
                // I'm not sure why this is written as angle = 360 - angle vs just angle = -angle.
                best_inplane_values[0] = 360.0 - (psi_i * psi_step + psi_start);
                best_inplane_values[1] = found_peak.x;
                best_inplane_values[2] = found_peak.y;
                mirrored_match         = false;
            }

            if ( test_mirror ) {
                psi_m++;
                // Note 1: I have always taken the conjugate of the reference, but in the CPU code it is of the rotated particle. Here the conjgate is already taken in the cache step.
                gpu_projection_image.MultiplyPixelWise(rotation_cache[psi_m], gpu_correlation_map);

                gpu_correlation_map.is_in_real_space = false;
                timer.lap("Cross Correlate");

                timer.start("FFT Correlation Map");
                gpu_correlation_map.BackwardFFT( );
                timer.lap("FFT Correlation Map");

                timer.start("Copy Correlation Map");
                gpu_correlation_map.CopyHostToDevice(true);
                timer.lap("Copy Correlation Map");

                timer.start("Fine Peak Search");
                found_peak = correlation_map.FindPeakAtOriginFast2D(max_pix_x, max_pix_y);
                timer.lap("Fine Peak Search");

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
    // *******************************

    timer.start("Clean up");
    delete flipped_image;
    delete padded_image;
    delete rotated_image;
    delete projection_image;
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
