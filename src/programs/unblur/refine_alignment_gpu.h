#define CISTEM_UNBLUR_REFINE_ALIGNMENT_DISABLE_MULTITHREADING

template <typename StorageBaseType = float>
void unblur_refine_alignment(std::vector<GpuImage>& input_stack,
                             int                    number_of_images,
                             int                    max_iterations,
                             float                  unitless_bfactor,
                             bool                   mask_central_cross,
                             int                    width_of_vertical_line,
                             int                    width_of_horizontal_line,
                             float                  inner_radius_for_peak_search,
                             float                  outer_radius_for_peak_search,
                             float                  max_shift_convergence_threshold,
                             float                  pixel_size,
                             int                    number_of_frames_for_running_average,
                             int                    savitzy_golay_window_size,
                             int                    max_threads,
                             float*                 x_shifts,
                             float*                 y_shifts,
                             StopWatch&             profile_timing_refinement_method) {

    profile_timing_refinement_method.mark_entry_or_exit_point( );

#ifdef CISTEM_UNBLUR_REFINE_ALIGNMENT_DISABLE_MULTITHREADING
    max_threads = 1;
#endif

    int running_average_counter;
    int start_frame_for_average;
    int end_frame_for_average;
    int iteration_counter;

    int number_of_middle_image    = number_of_images / 2;
    int running_average_half_size = (number_of_frames_for_running_average - 1) / 2;
    if ( running_average_half_size < 1 )
        running_average_half_size = 1;

    std::vector<float> current_x_shifts(number_of_images);
    std::vector<float> current_y_shifts(number_of_images);
    current_x_shifts.assign(current_x_shifts.size( ), 0.0f);
    current_y_shifts.assign(current_y_shifts.size( ), 0.0f);

    float middle_image_x_shift;
    float middle_image_y_shift;

    float max_shift;
    float total_shift;

    if ( IsOdd(savitzy_golay_window_size) == false )
        savitzy_golay_window_size++;
    if ( savitzy_golay_window_size < 5 )
        savitzy_golay_window_size = 5;

    std::vector<GpuImage> sum_of_images_minus_current;
    std::vector<GpuImage> correlation_map;

    sum_of_images_minus_current.reserve(max_threads);
    correlation_map.reserve(max_threads);

    bool use_fp16 = false;
    if constexpr ( std::is_same<StorageBaseType, __half>::value ) {
        use_fp16 = true;
    }

    GpuImage sum_of_images(input_stack[0].dims.x, input_stack[0].dims.y, 1, false, use_fp16);
    sum_of_images.Zeros<StorageBaseType>( );

    for ( int i = 0; i < max_threads; i++ ) {
        sum_of_images_minus_current.emplace_back(input_stack[0].dims.x, input_stack[0].dims.y, 1, false, use_fp16);

        // We currently swap back to float prior to the BackwardFFT so allocate both buffers.
        correlation_map.emplace_back(input_stack[0].dims.x, input_stack[0].dims.y, 1, false, false);
        correlation_map[i].BufferInit(GpuImage::BufferType::b_16f);
    }

    BatchedSearch batch;
    int           batch_size  = 1;
    bool          test_mirror = false;
    int           max_pix_x   = myroundint(outer_radius_for_peak_search);

    batch.Init(sum_of_images, number_of_images, batch_size, test_mirror, max_pix_x, max_pix_x);

    int peak_map_size = ReturnClosestFactorizedUpper(max_pix_x, 13, true, 32);

    std::vector<GpuImage> running_average_stack;
    std::vector<Image>    peak_map;
    peak_map.reserve(max_threads);

    for ( int i = 0; i < max_threads; i++ ) {
        peak_map.emplace_back(peak_map_size, peak_map_size, 1, true);
        cudaErr(cudaHostRegister(peak_map[i].real_values, sizeof(float) * peak_map[i].real_memory_allocated, cudaHostRegisterDefault));
    }

    Curve x_shifts_curve;
    Curve y_shifts_curve;

    bool use_running_average = (number_of_frames_for_running_average > 1) ? true : false;
    profile_timing_refinement_method.start("allocate running average");
    if ( use_running_average ) {
        running_average_stack.reserve(number_of_images);
        for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
            running_average_stack.emplace_back(input_stack[image_counter].dims.x, input_stack[image_counter].dims.y, 1, false, use_fp16);
        }
    }

    profile_timing_refinement_method.lap("allocate running average");

    // prepare the initial sum
    profile_timing_refinement_method.start("prepare initial sum");

    if ( use_running_average )
        sum_of_images.AddImageStack<StorageBaseType>(running_average_stack);
    else
        sum_of_images.AddImageStack<StorageBaseType>(input_stack);

    profile_timing_refinement_method.lap("prepare initial sum");
    // perform the main alignment loop until we reach a max shift less than wanted, or max iterations
    float wanted_inner_radius_for_peak_search;
    for ( iteration_counter = 0; iteration_counter <= max_iterations; iteration_counter++ ) {
        wanted_inner_radius_for_peak_search = (iteration_counter == 0) ? inner_radius_for_peak_search : 0.0f;
        batch.SetMinSearchExtension(myroundint(wanted_inner_radius_for_peak_search));

        //	wxPrintf("Starting iteration number %li\n\n", iteration_counter);
        max_shift = -std::numeric_limits<float>::max( );

        // make the current running average if necessary

        if ( number_of_frames_for_running_average > 1 ) {
            // TODO: fixme, overload for partial AddImageStack
            MyAssertTrue(false, "This code is not tested yet.  It is not used in the current version of the program.  It is here for future use.  Please contact the author if you need this feature.");
            for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
                start_frame_for_average = image_counter - running_average_half_size;
                end_frame_for_average   = image_counter + running_average_half_size;

                if ( start_frame_for_average < 0 ) {
                    end_frame_for_average -= start_frame_for_average; // add it to the right
                    start_frame_for_average = 0;
                }

                if ( end_frame_for_average >= number_of_images ) {
                    start_frame_for_average -= (end_frame_for_average - (number_of_images - 1));
                    end_frame_for_average = number_of_images - 1;
                }

                if ( start_frame_for_average < 0 )
                    start_frame_for_average = 0;
                if ( end_frame_for_average >= number_of_images )
                    end_frame_for_average = number_of_images - 1;
                running_average_stack[image_counter].SetToConstant(0.0f);

                for ( running_average_counter = start_frame_for_average; running_average_counter <= end_frame_for_average; running_average_counter++ ) {
                    running_average_stack[image_counter].AddImage(&input_stack[running_average_counter]);
                }
            }
        }
#pragma omp parallel for default(none) num_threads(max_threads) shared(std::cerr, number_of_images, current_x_shifts, current_y_shifts,              \
                                                                       profile_timing_refinement_method, sum_of_images_minus_current, sum_of_images, \
                                                                       use_running_average, running_average_stack, input_stack, correlation_map,     \
                                                                       unitless_bfactor,                                                             \
                                                                       width_of_horizontal_line, width_of_vertical_line,                             \
                                                                       peak_map,                                                                     \
                                                                       wanted_inner_radius_for_peak_search, outer_radius_for_peak_search) schedule(static, 1)

        for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {

            float this_shift = sqrtf(powf(current_x_shifts[image_counter], 2) + powf(current_y_shifts[image_counter], 2));
            int   my_tidx    = ReturnThreadNumberOfCurrentThread( );
            Peak  my_peak;
            int   phase_multiplier = 0;

            // if ( this_shift < 2.f ) {
            //     phase_multiplier = 2;
            // }
            // else {
            //     if ( this_shift < batch.max_pixel_radius_x( ) / 2.5f ) {
            //         phase_multiplier = 1;
            //     }
            //     else {
            //         phase_multiplier = 0;
            //     }
            // }

            // phase_multiplier = std::min(phase_multiplier, iteration_counter);
            // phase_multiplier = 0;
            //phase_multiplier = (current_x_shifts[image_counter] < batch.max_pixel_radius_x( ) / 3.0f && current_y_shifts[image_counter] < batch.max_pixel_radius_y( ) / 3.0f && iteration_counter > 0) ? 1 : 0;
            // prepare the sum reference by subtracting out the current image, applying a bfactor and masking central cross
            profile_timing_refinement_method.start("prepare sum");
            sum_of_images_minus_current[my_tidx].CopyDataFrom<StorageBaseType>(sum_of_images);
            if ( use_running_average )
                sum_of_images_minus_current[my_tidx].SubtractImage<StorageBaseType>(&running_average_stack[image_counter]);
            else
                sum_of_images_minus_current[my_tidx].SubtractImage<StorageBaseType>(&input_stack[image_counter]);

            sum_of_images_minus_current[my_tidx].ApplyBFactor<StorageBaseType>(unitless_bfactor, width_of_vertical_line, width_of_horizontal_line);

            profile_timing_refinement_method.lap("prepare sum");
            // compute the cross correlation function and find the peak
            // TODO: just replace with batched backfft as in euler search gpu
            // NOTE: the output XCF is stored in the fp16 buffer of the calling image
            // Do not swap the conjugated image
            correlation_map[my_tidx].is_in_real_space = false;
            profile_timing_refinement_method.start("compute cross correlation");
            if ( use_running_average ) {
                // FIXME
                MyAssertTrue(false, "running average is not setup yet.");
                sum_of_images_minus_current[my_tidx].CalculateCrossCorrelationImageWith(&running_average_stack[image_counter]);
            }
            else
                input_stack[image_counter].MultiplyPixelWiseComplexConjugate<StorageBaseType>(sum_of_images_minus_current[my_tidx], correlation_map[my_tidx], phase_multiplier);

            correlation_map[my_tidx].SwapRealSpaceQuadrants<StorageBaseType>( );

            // Copy back to single precision buffer for backward FFT (fp16 buffer is untested in cuFFT)
            correlation_map[my_tidx].CopyFP16buffertoFP32(false);
            correlation_map[my_tidx].BackwardFFTBatched(1);

            profile_timing_refinement_method.lap("compute cross correlation");
            profile_timing_refinement_method.start("find peak");

            // std::cerr << "On GPU : " << correlation_map.is_in_memory_gpu << std::endl;
            // correlation_map.CopyFP16buffertoFP32(false);
            // cudaErr(cudaDeviceSynchronize( ));
            // correlation_map.QuickAndDirtyWriteSlice("/tmp/xcf" + std::to_string(phase_multiplier) + ".mrc", 1);
            // exit(0);

            // For testing on the GPU this is just doing a copy which is of course a bit of a waste
            // my_peak = sum_of_images_minus_current[my_tidx].FindPeakWithParabolaFit(wanted_inner_radius_for_peak_search, outer_radius_for_peak_search);
            int address_offset = (correlation_map[my_tidx].dims.y / 2 - peak_map[my_tidx].logical_y_dimension / 2) * correlation_map[my_tidx].dims.w +
                                 (correlation_map[my_tidx].dims.x / 2 - peak_map[my_tidx].logical_x_dimension / 2);

            cudaErr(cudaMemcpy2DAsync(peak_map[my_tidx].real_values, (peak_map[my_tidx].logical_x_dimension + 2) * sizeof(float),
                                      &correlation_map[my_tidx].real_values_gpu[address_offset], correlation_map[my_tidx].dims.w * sizeof(float),
                                      (peak_map[my_tidx].logical_x_dimension + 2) * sizeof(float), peak_map[my_tidx].logical_y_dimension,
                                      cudaMemcpyDeviceToHost, cudaStreamPerThread));

            cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
            // correlation_map.QuickAndDirtyWriteSlice("/tmp/xcf" + std::to_string(phase_multiplier) + ".mrc", 1);
            // peak_map[my_tidx].QuickAndDirtyWriteSlice("/tmp/peak" + std::to_string(phase_multiplier) + ".mrc", 1);
            // exit(0);
            // my_peak = correlation_map.FindPeakAtCenterFast2d(batch);
            my_peak = peak_map[my_tidx].FindPeakWithParabolaFit(wanted_inner_radius_for_peak_search, outer_radius_for_peak_search);

            profile_timing_refinement_method.lap("find peak");
            // update the shifts..

            current_x_shifts[image_counter] = my_peak.x / float(1 + phase_multiplier);
            current_y_shifts[image_counter] = my_peak.y / float(1 + phase_multiplier);
        } // end of parallel for loop

        // smooth the shifts
        profile_timing_refinement_method.start("smooth shifts");
        x_shifts_curve.ClearData( );
        y_shifts_curve.ClearData( );

        for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
            x_shifts_curve.AddPoint(image_counter, x_shifts[image_counter] + current_x_shifts[image_counter]);
            y_shifts_curve.AddPoint(image_counter, y_shifts[image_counter] + current_y_shifts[image_counter]);

#ifdef PRINT_VERBOSE
            wxPrintf("Before = %li : %f, %f\n", image_counter, x_shifts[image_counter] + current_x_shifts[image_counter], y_shifts[image_counter] + current_y_shifts[image_counter]);
#endif
        }
        // in this case, weird things can happen (+1/-1 flips), we want to really smooth it. use a polynomial.  This should only affect the first round..
        if ( iteration_counter == 0 ) {
            if ( x_shifts_curve.number_of_points > 2 ) {
                x_shifts_curve.FitPolynomialToData(4);
                y_shifts_curve.FitPolynomialToData(4);

                // copy back

                for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
                    current_x_shifts[image_counter] = x_shifts_curve.polynomial_fit[image_counter] - x_shifts[image_counter];
                    current_y_shifts[image_counter] = y_shifts_curve.polynomial_fit[image_counter] - y_shifts[image_counter];
#ifdef PRINT_VERBOSE
                    wxPrintf("After poly = %li : %f, %f\n", image_counter, x_shifts_curve.polynomial_fit[image_counter], y_shifts_curve.polynomial_fit[image_counter]);
#endif
                }
            }
        }
        else {
            // when the input movie is dodgy (very few frames), the fitting won't work
            if ( savitzy_golay_window_size < x_shifts_curve.number_of_points ) {
                x_shifts_curve.FitSavitzkyGolayToData(savitzy_golay_window_size, 1);
                y_shifts_curve.FitSavitzkyGolayToData(savitzy_golay_window_size, 1);

                // copy them back..

                for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
                    current_x_shifts[image_counter] = x_shifts_curve.savitzky_golay_fit[image_counter] - x_shifts[image_counter];
                    current_y_shifts[image_counter] = y_shifts_curve.savitzky_golay_fit[image_counter] - y_shifts[image_counter];

#ifdef PRINT_VERBOSE
                    wxPrintf("After SG = %li : %f, %f\n", image_counter, x_shifts_curve.savitzky_golay_fit[image_counter], y_shifts_curve.savitzky_golay_fit[image_counter]);
#endif
                }
            }
        }
        profile_timing_refinement_method.lap("smooth shifts");

        // subtract shift of the middle image from all images to keep things centred around it

        middle_image_x_shift = current_x_shifts[number_of_middle_image];
        middle_image_y_shift = current_y_shifts[number_of_middle_image];

        for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
            current_x_shifts[image_counter] -= middle_image_x_shift;
            current_y_shifts[image_counter] -= middle_image_y_shift;

            total_shift = sqrt(pow(current_x_shifts[image_counter], 2) + pow(current_y_shifts[image_counter], 2));
            if ( total_shift > max_shift )
                max_shift = total_shift;
        }

// TODO: I can't see any good reason to repeatedly apply these shifts rather than applying the full shift each time a sum is made.
// it MUST accumulate errors, write a kernel that does the shift and accumulates the sum at the same time.
// actually shift the images, also add the subtracted shifts to the overall shifts
#pragma omp parallel for default(shared) num_threads(max_threads)
        for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
            profile_timing_refinement_method.start("shift image");
            input_stack[image_counter].PhaseShift<StorageBaseType>(current_x_shifts[image_counter], current_y_shifts[image_counter], 0.0);

            x_shifts[image_counter] += current_x_shifts[image_counter];
            y_shifts[image_counter] += current_y_shifts[image_counter];
            profile_timing_refinement_method.lap("shift image");
        }

        // check to see if the convergence criteria have been reached and return if so

        if ( iteration_counter >= max_iterations || (iteration_counter > 0 && max_shift <= max_shift_convergence_threshold) ) {
            profile_timing_refinement_method.start("cleanup");
            wxPrintf("returning, iteration = %i, max_shift = %f\n", iteration_counter, max_shift);

            profile_timing_refinement_method.lap("cleanup");
            // No need to apply the shifts unless it is our final iteration on the full stack, perhaps better yet we apply outside this method?
            profile_timing_refinement_method.mark_entry_or_exit_point( );
            return;
        }
        else {
            wxPrintf("Not. returning, iteration = %i, max_shift = %f\n", iteration_counter, max_shift);
        }

        // going to be doing another round so we need to make the new sum..
        profile_timing_refinement_method.start("remake sum");
        sum_of_images.Zeros<StorageBaseType>( );

        sum_of_images.AddImageStack<StorageBaseType>(input_stack);
        profile_timing_refinement_method.lap("remake sum");
        // cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
        // sum_of_images.QuickAndDirtyWriteSlice("/tmp/gpu_sum_of_images.mrc", 1);
    }

    for ( int i = 0; i < max_threads; i++ ) {
        cudaErr(cudaHostUnregister(peak_map[i].real_values));
    }

    profile_timing_refinement_method.mark_entry_or_exit_point( );
}

#undef CISTEM_UNBLUR_REFINE_ALIGNMENT_DISABLE_MULTITHREADING