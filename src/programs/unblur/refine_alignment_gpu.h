
void unblur_refine_alignment(GpuImage*  input_stack,
                             int        number_of_images,
                             int        max_iterations,
                             float      unitless_bfactor,
                             bool       mask_central_cross,
                             int        width_of_vertical_line,
                             int        width_of_horizontal_line,
                             float      inner_radius_for_peak_search,
                             float      outer_radius_for_peak_search,
                             float      max_shift_convergence_threshold,
                             float      pixel_size,
                             int        number_of_frames_for_running_average,
                             int        savitzy_golay_window_size,
                             int        max_threads,
                             float*     x_shifts,
                             float*     y_shifts,
                             StopWatch& profile_timing_refinement_method) {

    profile_timing_refinement_method.mark_entry_or_exit_point( );

    long pixel_counter;
    long image_counter;
    int  running_average_counter;
    int  start_frame_for_average;
    int  end_frame_for_average;
    long iteration_counter;

    int number_of_middle_image    = number_of_images / 2;
    int running_average_half_size = (number_of_frames_for_running_average - 1) / 2;
    if ( running_average_half_size < 1 )
        running_average_half_size = 1;

    std::vector<float> current_x_shifts(number_of_images);
    std::vector<float> current_y_shifts(number_of_images);

    std::cerr << "size and max size of vector are (n_images) " << current_x_shifts.size( ) << " " << current_x_shifts.max_size( ) << " " << number_of_images << std::endl;
    int iv = 0;
    for ( auto& x : current_x_shifts ) {
        // print initialzed values for verification
        std::cerr << "current_x_shifts " << iv << " " << x << std::endl;
        iv++;
    }

    float middle_image_x_shift;
    float middle_image_y_shift;

    float max_shift;
    float total_shift;

    if ( IsOdd(savitzy_golay_window_size) == false )
        savitzy_golay_window_size++;
    if ( savitzy_golay_window_size < 5 )
        savitzy_golay_window_size = 5;

    GpuImage  sum_of_images(input_stack[0].dims.x, input_stack[0].dims.y, 1, false);
    GpuImage  sum_of_images_minus_current(input_stack[0].dims.x, input_stack[0].dims.y, 1, false);
    GpuImage* running_average_stack;

    GpuImage* stack_for_alignment; // pointer that can be switched between running average stack and image stack if necessary
    Peak      my_peak;

    Curve x_shifts_curve;
    Curve y_shifts_curve;

    sum_of_images.SetToConstant(0.f);

    profile_timing_refinement_method.start("allocate running average");
    if ( number_of_frames_for_running_average > 1 ) {
        running_average_stack = new GpuImage[number_of_images];

        for ( image_counter = 0; image_counter < number_of_images; image_counter++ ) {
            running_average_stack[image_counter].Allocate(input_stack[image_counter].dims.x, input_stack[image_counter].dims.y, 1, false);
        }

        stack_for_alignment = running_average_stack;
    }
    else {
        // FIXME: ensure that you are only assigning the pointer and there aren't any bugs in your operator definition that
        // results in a copy;
        stack_for_alignment = input_stack;
    }

    profile_timing_refinement_method.lap("allocate running average");

    // prepare the initial sum
    profile_timing_refinement_method.start("prepare initial sum");

    stack_for_alignment->AddImageStack<float>(sum_of_images);

    profile_timing_refinement_method.lap("prepare initial sum");
    // perform the main alignment loop until we reach a max shift less than wanted, or max iterations
    float wanted_inner_radius_for_peak_search;
    for ( iteration_counter = 0; iteration_counter <= max_iterations; iteration_counter++ ) {
        wanted_inner_radius_for_peak_search = (iteration_counter == 0) ? inner_radius_for_peak_search : 0.0f;

        //	wxPrintf("Starting iteration number %li\n\n", iteration_counter);
        max_shift = -std::numeric_limits<float>::max( );

        // make the current running average if necessary

        if ( number_of_frames_for_running_average > 1 ) {
            // TODO: fixme, overload for partial AddImageStack
            MyAssertTrue(false, "This code is not tested yet.  It is not used in the current version of the program.  It is here for future use.  Please contact the author if you need this feature.");
            for ( image_counter = 0; image_counter < number_of_images; image_counter++ ) {
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

        for ( image_counter = 0; image_counter < number_of_images; image_counter++ ) {
            // prepare the sum reference by subtracting out the current image, applying a bfactor and masking central cross
            profile_timing_refinement_method.start("prepare sum");
            sum_of_images_minus_current.CopyFrom(&sum_of_images);
            sum_of_images_minus_current.SubtractImage(&stack_for_alignment[image_counter]);

#ifdef ENABLEGPU
            // Specialization to merge tehse operations into one kernel
            // TODO: merge the following into a single kernel
            // sum - current, apply bfactor, mask central cross, conj, multiply, over-sample shift
            sum_of_images_minus_current.ApplyBFactor(unitless_bfactor, width_of_vertical_line, width_of_horizontal_line);

#else
            sum_of_images_minus_current.ApplyBFactor(unitless_bfactor);
            if ( mask_central_cross == true ) {
                sum_of_images_minus_current.MaskCentralCross(width_of_vertical_line, width_of_horizontal_line);
            }
#endif
            profile_timing_refinement_method.lap("prepare sum");
            // compute the cross correlation function and find the peak
            // TODO: just replace with batched backfft as in euler search gpu
            profile_timing_refinement_method.start("compute cross correlation");
            sum_of_images_minus_current.CalculateCrossCorrelationImageWith(&stack_for_alignment[image_counter]);
            profile_timing_refinement_method.lap("compute cross correlation");
            profile_timing_refinement_method.start("find peak");
#ifdef ENABLEGPU
            my_peak.x     = 0.0f;
            my_peak.y     = 0.0f;
            my_peak.value = 0.0f;
            // TODO: Since our peak area should be quite small, create a GPU method for cudaMemcpy2Dasync
            // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1ge529b926e8fb574c2666a9a1d58b0dc1
            // Ideally, we would only write out that size from the backFFT through a callback or FastFFT - optimizations for later

#else
            // For testing on the GPU this is just doing a copy which is of course a bit of a waste
            my_peak = sum_of_images_minus_current.FindPeakWithParabolaFit(wanted_inner_radius_for_peak_search, outer_radius_for_peak_search);
#endif
            profile_timing_refinement_method.lap("find peak");
            // update the shifts..

            current_x_shifts[image_counter] = my_peak.x;
            current_y_shifts[image_counter] = my_peak.y;
        }
        profile_timing_refinement_method.start("deallocate sum minus");

        // TODO: no good reason for this deallocation
        sum_of_images_minus_current.Deallocate( );
        profile_timing_refinement_method.lap("deallocate sum minus");
        // smooth the shifts
        profile_timing_refinement_method.start("smooth shifts");
        x_shifts_curve.ClearData( );
        y_shifts_curve.ClearData( );

        for ( image_counter = 0; image_counter < number_of_images; image_counter++ ) {
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

                for ( image_counter = 0; image_counter < number_of_images; image_counter++ ) {
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

                for ( image_counter = 0; image_counter < number_of_images; image_counter++ ) {
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

        for ( image_counter = 0; image_counter < number_of_images; image_counter++ ) {
            current_x_shifts[image_counter] -= middle_image_x_shift;
            current_y_shifts[image_counter] -= middle_image_y_shift;

            total_shift = sqrt(pow(current_x_shifts[image_counter], 2) + pow(current_y_shifts[image_counter], 2));
            if ( total_shift > max_shift )
                max_shift = total_shift;
        }

// TODO: I can't see any good reason to repeatedly apply these shifts rather than applying the full shift each time a sum is made.
// it MUST accumulate errors
// actually shift the images, also add the subtracted shifts to the overall shifts
#pragma omp parallel for default(shared) num_threads(max_threads) private(image_counter)
        for ( image_counter = 0; image_counter < number_of_images; image_counter++ ) {
            profile_timing_refinement_method.start("shift image");
            input_stack[image_counter].PhaseShift(current_x_shifts[image_counter], current_y_shifts[image_counter], 0.0);

            x_shifts[image_counter] += current_x_shifts[image_counter];
            y_shifts[image_counter] += current_y_shifts[image_counter];
            profile_timing_refinement_method.lap("shift image");
        }

        // check to see if the convergence criteria have been reached and return if so

        if ( iteration_counter >= max_iterations || (iteration_counter > 0 && max_shift <= max_shift_convergence_threshold) ) {
            profile_timing_refinement_method.start("cleanup");
            wxPrintf("returning, iteration = %li, max_shift = %f\n", iteration_counter, max_shift);

            if ( number_of_frames_for_running_average > 1 ) {
                delete[] running_average_stack;
            }
            profile_timing_refinement_method.lap("cleanup");
            // No need to apply the shifts unless it is our final iteration on the full stack, perhaps better yet we apply outside this method?
            profile_timing_refinement_method.mark_entry_or_exit_point( );
            return;
        }
        else {
            wxPrintf("Not. returning, iteration = %li, max_shift = %f\n", iteration_counter, max_shift);
        }

        // going to be doing another round so we need to make the new sum..
        profile_timing_refinement_method.start("remake sum");
        sum_of_images.SetToConstant(0.0);

        for ( image_counter = 0; image_counter < number_of_images; image_counter++ ) {
            sum_of_images.AddImage(&input_stack[image_counter]);
        }
        profile_timing_refinement_method.lap("remake sum");
    }

    profile_timing_refinement_method.mark_entry_or_exit_point( );
}