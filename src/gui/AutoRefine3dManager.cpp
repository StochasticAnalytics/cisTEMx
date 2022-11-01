#include "../core/gui_core_headers.h"

// Rather than re-write the blocks that add rounds of global alignment after round 1, we override the booleans they set to true,
// when cisTEM is configured to do so by --disable-multiple-global-refinements
#ifndef DISABLE_MUTLI_GLOBAL_REFINEMENTS
constexpr bool true_if_not_configured_as_disabled = true;
#else
constexpr bool true_if_not_configured_as_disabled = false;
#endif

extern MyRefinementPackageAssetPanel* refinement_package_asset_panel;
extern MyRunProfilesPanel*            run_profiles_panel;
extern MyVolumeAssetPanel*            volume_asset_panel;
extern MyRefinementResultsPanel*      refinement_results_panel;
extern MyMainFrame*                   main_frame;

wxDEFINE_EVENT(wxEVT_COMMAND_MYTHREAD_COMPLETED, wxThreadEvent);

template <class MyParentType>
AutoRefinementManager<MyParentType>::AutoRefinementManager( ) {
    input_refinement  = NULL;
    output_refinement = NULL;
}

template <class MyParentType>
void AutoRefinementManager<MyParentType>::SetParent(MyParentType* wanted_parent) {
    my_parent = wanted_parent;
}

template <class MyParentType>
void AutoRefinementManager<MyParentType>::BeginRefinementCycle( ) {
    active_refinement_package           = &refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageSelectPanel->GetSelection( ));
    current_refinement_package_asset_id = active_refinement_package->asset_id;

    active_low_resolution_limit     = my_parent->LowResolutionLimitTextCtrl->ReturnValue( );
    active_mask_radius              = my_parent->MaskRadiusTextCtrl->ReturnValue( );
    active_global_mask_radius       = my_parent->GlobalMaskRadiusTextCtrl->ReturnValue( );
    active_inner_mask_radius        = my_parent->InnerMaskRadiusTextCtrl->ReturnValue( );
    active_target_resolution        = my_parent->TargetResolutionTextCtrl->ReturnValue( );
    active_number_results_to_refine = my_parent->NumberToRefineSpinCtrl->GetValue( );
    active_search_range_x           = my_parent->SearchRangeXTextCtrl->ReturnValue( );
    active_search_range_y           = my_parent->SearchRangeYTextCtrl->ReturnValue( );
    active_should_apply_blurring    = my_parent->ApplyBlurringYesRadioButton->GetValue( );
    active_smoothing_factor         = my_parent->SmoothingFactorTextCtrl->ReturnValue( );
    active_should_mask              = my_parent->UseMaskCheckBox->GetValue( );
    active_should_auto_mask         = my_parent->AutoMaskYesRadioButton->GetValue( );

    if ( my_parent->MaskSelectPanel->ReturnSelection( ) >= 0 )
        active_mask_asset_id = volume_asset_panel->ReturnAssetID(my_parent->MaskSelectPanel->ReturnSelection( ));
    else
        active_mask_asset_id = -1;
    if ( my_parent->MaskSelectPanel->ReturnSelection( ) >= 0 )
        active_mask_filename = volume_asset_panel->ReturnAssetLongFilename(my_parent->MaskSelectPanel->ReturnSelection( ));
    else
        active_mask_filename = "";

    active_should_low_pass_filter_mask = my_parent->LowPassMaskYesRadio->GetValue( );
    active_mask_filter_resolution      = my_parent->MaskFilterResolutionText->ReturnValue( );
    active_mask_edge                   = my_parent->MaskEdgeTextCtrl->ReturnValue( );
    active_mask_weight                 = my_parent->MaskWeightTextCtrl->ReturnValue( );

    reference_3d_contains_all_particles = false;

    active_refinement_run_profile     = run_profiles_panel->run_profile_manager.run_profiles[my_parent->RefinementRunProfileComboBox->GetSelection( )];
    active_reconstruction_run_profile = run_profiles_panel->run_profile_manager.run_profiles[my_parent->ReconstructionRunProfileComboBox->GetSelection( )];

    active_auto_crop = my_parent->AutoCropYesRadioButton->GetValue( );

    int  class_counter;
    long particle_counter;

    int  number_of_classes   = active_refinement_package->number_of_classes;
    long number_of_particles = active_refinement_package->contained_particles.GetCount( );

    wxString blank_string = "";
    current_reference_filenames.Clear( );
    current_reference_filenames.Add(blank_string, number_of_classes);

    current_reference_asset_ids.Clear( );
    current_reference_asset_ids.Add(-1, number_of_classes);

    // Clear scratch..

    global_delete_autorefine3d_scratch( );

    // setup input/output refinements

    long current_input_refinement_id = active_refinement_package->refinement_ids[0];

    input_refinement                               = main_frame->current_project.database.GetRefinementByID(current_input_refinement_id);
    output_refinement                              = new Refinement;
    output_refinement->refinement_package_asset_id = input_refinement->refinement_package_asset_id;

    //wxPrintf("RP has %li particles, refinement has %li particles\n", number_of_particles, input_refinement->number_of_particles);

    // create a refinement with random angles etc..

    output_refinement->SizeAndFillWithEmpty(number_of_particles, number_of_classes);
    output_refinement->refinement_id = main_frame->current_project.database.ReturnHighestRefinementID( ) + 1;

    // Randomise the input parameters, and set the default resolution statistics..

    class_high_res_limits.Clear( );
    class_next_high_res_limits.Clear( );

    for ( class_counter = 0; class_counter < number_of_classes; class_counter++ ) {
        for ( particle_counter = 0; particle_counter < number_of_particles; particle_counter++ ) {
            if ( number_of_classes == 1 )
                input_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].occupancy = 100.0;
            else
                input_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].occupancy = 100.00 / input_refinement->number_of_classes;

            input_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].phi             = global_random_number_generator.GetUniformRandom( ) * 180.0;
            input_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].theta           = global_random_number_generator.GetUniformRandom( ) * 180.0;
            input_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].psi             = global_random_number_generator.GetUniformRandom( ) * 180.0;
            input_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].xshift          = 0;
            input_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].yshift          = 0;
            input_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].score           = 0.0;
            input_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].image_is_active = 1;
            input_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].sigma           = 1.0;
        }

        input_refinement->class_refinement_results[class_counter].class_resolution_statistics.GenerateDefaultStatistics(active_refinement_package->estimated_particle_weight_in_kda);

        class_high_res_limits.Add(my_parent->HighResolutionLimitTextCtrl->ReturnValue( ));
        class_next_high_res_limits.Add(my_parent->HighResolutionLimitTextCtrl->ReturnValue( ));
    }

    // how many particles to use..

    long  number_of_asym_units              = number_of_particles * ReturnNumberofAsymmetricUnits(active_refinement_package->symmetry);
    float estimated_required_asym_units     = 8000.0f * expf(75.0f / powf(my_parent->HighResolutionLimitTextCtrl->ReturnValue( ), 2));
    long  wanted_start_number_of_asym_units = myroundint(estimated_required_asym_units) * number_of_classes;

    // what percentage is this.

    start_percent_used = (float(wanted_start_number_of_asym_units) / float(number_of_asym_units)) * 100.0;
    if ( start_percent_used > 100.0 )
        start_percent_used = 100.0;

    current_percent_used = start_percent_used;
    max_percent_used     = current_percent_used;

    this_is_the_final_round = false;
    number_of_rounds_run    = 0;
    percent_used_per_round.Clear( );
    resolution_per_round.Clear( );

    last_round_reconstruction_resolution = FLT_MAX;

    high_res_limit_per_round.Clear( );
    high_res_limit_per_round.Add(my_parent->HighResolutionLimitTextCtrl->ReturnValue( ));

    number_of_global_alignments.Clear( );
    number_of_global_alignments.Add(0, number_of_particles);

    rounds_since_global_alignment.Clear( );
    rounds_since_global_alignment.Add(0, number_of_particles);

    resolution_of_last_global_alignment.Clear( );
    resolution_of_last_global_alignment.Add(100.0f, number_of_particles);

    // we need to set the currently selected reference filenames..

    if ( volume_asset_panel->ReturnAssetPointer(my_parent->ReferenceSelectPanel->ReturnSelection( ))->x_size != active_refinement_package->stack_box_size ||
         volume_asset_panel->ReturnAssetPointer(my_parent->ReferenceSelectPanel->ReturnSelection( ))->y_size != active_refinement_package->stack_box_size ||
         volume_asset_panel->ReturnAssetPointer(my_parent->ReferenceSelectPanel->ReturnSelection( ))->z_size != active_refinement_package->stack_box_size ||
         fabsf(float(volume_asset_panel->ReturnAssetPointer(my_parent->ReferenceSelectPanel->ReturnSelection( ))->pixel_size) - input_refinement->resolution_statistics_pixel_size) > 0.01f ) {
        my_parent->WriteErrorText(wxString::Format("Error: The reference volume (%i, %i, %i; psize: %f) has different dimensions / pixel size from the input stack (%i; psize: %f).  This will currently not work.",
                                                   volume_asset_panel->ReturnAssetPointer(my_parent->ReferenceSelectPanel->ReturnSelection( ))->x_size,
                                                   volume_asset_panel->ReturnAssetPointer(my_parent->ReferenceSelectPanel->ReturnSelection( ))->y_size,
                                                   volume_asset_panel->ReturnAssetPointer(my_parent->ReferenceSelectPanel->ReturnSelection( ))->z_size,
                                                   volume_asset_panel->ReturnAssetPointer(my_parent->ReferenceSelectPanel->ReturnSelection( ))->pixel_size,
                                                   active_refinement_package->stack_box_size,
                                                   input_refinement->resolution_statistics_pixel_size));
    }

    for ( class_counter = 0; class_counter < number_of_classes; class_counter++ ) {

        current_reference_filenames.Item(class_counter) = volume_asset_panel->ReturnAssetLongFilename(my_parent->ReferenceSelectPanel->ReturnSelection( ));
        current_reference_asset_ids.Item(class_counter) = volume_asset_panel->ReturnAssetID(my_parent->ReferenceSelectPanel->ReturnSelection( ));
    }

    // Do we need to do masking them?

    //my_parent->Freeze();

    my_parent->InputParamsPanel->Show(false);
    my_parent->StartPanel->Show(false);
    my_parent->ProgressPanel->Show(true);
    my_parent->ExpertPanel->Show(false);
    my_parent->InfoPanel->Show(false);
    my_parent->OutputTextPanel->Show(true);
    my_parent->ShowRefinementResultsPanel->Clear( );

    if ( my_parent->ShowRefinementResultsPanel->LeftRightSplitter->IsSplit( ) == true )
        my_parent->ShowRefinementResultsPanel->LeftRightSplitter->Unsplit( );
    if ( my_parent->ShowRefinementResultsPanel->TopBottomSplitter->IsSplit( ) == true )
        my_parent->ShowRefinementResultsPanel->TopBottomSplitter->Unsplit( );
    my_parent->ShowRefinementResultsPanel->Show(true);
    my_parent->Layout( );

    //my_parent->Thaw();

    if ( active_should_auto_mask == true || active_should_mask == true ) {
        DoMasking( );
    }
    else {
        SetupRefinementJob( );
        RunRefinementJob( );
    }
}

template <class MyParentType>
void AutoRefinementManager<MyParentType>::RunRefinementJob( ) {
    running_job_type                    = REFINEMENT;
    number_of_received_particle_results = 0;
    //expected_number_of_results = input_refinement->number_of_particles * input_refinement->number_of_classes;

    output_refinement->SizeAndFillWithEmpty(input_refinement->number_of_particles, input_refinement->number_of_classes);
    //wxPrintf("Output refinement has %li particles and %i classes\n", output_refinement->number_of_particles, input_refinement->number_of_classes);
    current_output_refinement_id = main_frame->current_project.database.ReturnHighestRefinementID( ) + 1;

    output_refinement->refinement_id = current_output_refinement_id;

    output_refinement->name = wxString::Format("Auto #%li - Round %i", current_output_refinement_id, number_of_rounds_run + 1);

    output_refinement->resolution_statistics_are_generated = false;
    output_refinement->datetime_of_run                     = wxDateTime::Now( );
    output_refinement->starting_refinement_id              = input_refinement->refinement_id;

    for ( int class_counter = 0; class_counter < input_refinement->number_of_classes; class_counter++ ) {
        output_refinement->class_refinement_results[class_counter].low_resolution_limit  = active_low_resolution_limit;
        output_refinement->class_refinement_results[class_counter].high_resolution_limit = class_high_res_limits[class_counter];
        output_refinement->class_refinement_results[class_counter].mask_radius           = active_mask_radius;

        if ( IsOdd(number_of_rounds_run) == true || this_is_the_final_round == true || number_of_rounds_run == 0 )
            output_refinement->class_refinement_results[class_counter].signed_cc_resolution_limit = 0.0f;
        else
            output_refinement->class_refinement_results[class_counter].signed_cc_resolution_limit = 15.0f;

        output_refinement->class_refinement_results[class_counter].global_resolution_limit         = class_high_res_limits[class_counter];
        output_refinement->class_refinement_results[class_counter].global_mask_radius              = active_global_mask_radius;
        output_refinement->class_refinement_results[class_counter].number_results_to_refine        = active_number_results_to_refine;
        output_refinement->class_refinement_results[class_counter].angular_search_step             = CalculateAngularStep(class_high_res_limits[class_counter], active_mask_radius);
        output_refinement->class_refinement_results[class_counter].search_range_x                  = active_search_range_x;
        output_refinement->class_refinement_results[class_counter].search_range_y                  = active_search_range_y;
        output_refinement->class_refinement_results[class_counter].classification_resolution_limit = 10.0;
        output_refinement->class_refinement_results[class_counter].should_focus_classify           = false;
        output_refinement->class_refinement_results[class_counter].sphere_x_coord                  = 0;
        output_refinement->class_refinement_results[class_counter].sphere_y_coord                  = 0;
        output_refinement->class_refinement_results[class_counter].sphere_z_coord                  = 0;
        output_refinement->class_refinement_results[class_counter].should_refine_ctf               = false;
        output_refinement->class_refinement_results[class_counter].defocus_search_range            = 0;
        output_refinement->class_refinement_results[class_counter].defocus_search_step             = 0;
        output_refinement->class_refinement_results[class_counter].should_auto_mask                = active_should_auto_mask;
        output_refinement->class_refinement_results[class_counter].should_refine_input_params      = true;
        output_refinement->class_refinement_results[class_counter].should_use_supplied_mask        = active_should_mask;
        output_refinement->class_refinement_results[class_counter].mask_asset_id                   = active_mask_asset_id;
        output_refinement->class_refinement_results[class_counter].mask_edge_width                 = active_mask_edge;
        output_refinement->class_refinement_results[class_counter].outside_mask_weight             = active_mask_weight;
        output_refinement->class_refinement_results[class_counter].should_low_pass_filter_mask     = active_should_low_pass_filter_mask;
        output_refinement->class_refinement_results[class_counter].filter_resolution               = active_mask_filter_resolution;
    }

    output_refinement->percent_used = current_percent_used;

    output_refinement->resolution_statistics_box_size   = input_refinement->resolution_statistics_box_size;
    output_refinement->resolution_statistics_pixel_size = input_refinement->resolution_statistics_pixel_size;

    // launch a controller

    current_job_starttime = time(NULL);

    time_of_last_update = current_job_starttime;
    my_parent->ShowRefinementResultsPanel->AngularPlotPanel->Clear( );

    my_parent->WriteBlueText(wxString::Format(wxT("Running refinement round %2i (%.2f %%)\n"), number_of_rounds_run + 1, current_percent_used));

    for ( int class_counter = 0; class_counter < input_refinement->number_of_classes; class_counter++ ) {
        my_parent->WriteBlueText(wxString::Format(wxT("Res. limit for class #%i = %.2f"), class_counter, class_high_res_limits[class_counter]));
    }

    current_job_id       = main_frame->job_controller.AddJob(my_parent, active_refinement_run_profile.manager_command, active_refinement_run_profile.gui_address);
    my_parent->my_job_id = current_job_id;

    if ( current_job_id != -1 ) {
        my_parent->SetNumberConnectedTextToZeroAndStartTracking( );
    }

    my_parent->ProgressBar->Pulse( );
}

template <class MyParentType>
void AutoRefinementManager<MyParentType>::SetupMerge3dJob( ) {
    long number_of_particles           = active_refinement_package->contained_particles.GetCount( );
    int  number_of_reconstruction_jobs = std::min(number_of_particles, active_reconstruction_run_profile.ReturnTotalJobs( ));

    int class_counter;

    my_parent->current_job_package.Reset(active_reconstruction_run_profile, "merge3d", active_refinement_package->number_of_classes);

    for ( class_counter = 0; class_counter < active_refinement_package->number_of_classes; class_counter++ ) {
        wxString output_reconstruction_1        = "/dev/null";
        wxString output_reconstruction_2        = "/dev/null";
        wxString output_reconstruction_filtered = main_frame->current_project.volume_asset_directory.GetFullPath( ) + wxString::Format("/volume_%li_%i.mrc", output_refinement->refinement_id, class_counter + 1);

        current_reference_filenames.Item(class_counter) = output_reconstruction_filtered;

        wxString output_resolution_statistics = "/dev/null";
        float    molecular_mass_kDa           = active_refinement_package->estimated_particle_weight_in_kda;
        float    inner_mask_radius            = active_inner_mask_radius;
        float    outer_mask_radius            = active_mask_radius;
        wxString dump_file_seed_1             = main_frame->ReturnAutoRefine3DScratchDirectory( ) + wxString::Format("dump_file_%li_%i_odd_.dump", current_output_refinement_id, class_counter);
        wxString dump_file_seed_2             = main_frame->ReturnAutoRefine3DScratchDirectory( ) + wxString::Format("dump_file_%li_%i_even_.dump", current_output_refinement_id, class_counter);

        bool     save_orthogonal_views_image = true;
        wxString orthogonal_views_filename   = main_frame->current_project.volume_asset_directory.GetFullPath( ) + wxString::Format("/OrthViews/volume_%li_%i.mrc", output_refinement->refinement_id, class_counter + 1);
        float    weiner_nominator            = 1.0f;

        float alignment_res = class_high_res_limits[class_counter];

        my_parent->current_job_package.AddJob("ttttfffttibtiff", output_reconstruction_1.ToUTF8( ).data( ),
                                              output_reconstruction_2.ToUTF8( ).data( ),
                                              output_reconstruction_filtered.ToUTF8( ).data( ),
                                              output_resolution_statistics.ToUTF8( ).data( ),
                                              molecular_mass_kDa, inner_mask_radius, outer_mask_radius,
                                              dump_file_seed_1.ToUTF8( ).data( ),
                                              dump_file_seed_2.ToUTF8( ).data( ),
                                              class_counter + 1,
                                              save_orthogonal_views_image,
                                              orthogonal_views_filename.ToUTF8( ).data( ),
                                              number_of_reconstruction_jobs, weiner_nominator, alignment_res);
    }
}

template <class MyParentType>
void AutoRefinementManager<MyParentType>::RunMerge3dJob( ) {
    running_job_type = MERGE;

    // start job..

    if ( output_refinement->number_of_classes > 1 )
        my_parent->WriteBlueText("Merging and Filtering Reconstructions...");
    else
        my_parent->WriteBlueText("Merging and Filtering Reconstruction...");

    current_job_id       = main_frame->job_controller.AddJob(my_parent, active_reconstruction_run_profile.manager_command, active_reconstruction_run_profile.gui_address);
    my_parent->my_job_id = current_job_id;

    if ( current_job_id != -1 ) {
        my_parent->SetNumberConnectedTextToZeroAndStartTracking( );
    }

    my_parent->ProgressBar->Pulse( );
}

template <class MyParentType>
void AutoRefinementManager<MyParentType>::SetupReconstructionJob( ) {
    wxArrayString written_parameter_files;

    // set sigmas based on resolution..

    for ( int class_counter = 0; class_counter < output_refinement->number_of_classes; class_counter++ ) {
        for ( long particle_counter = 0; particle_counter < output_refinement->number_of_particles; particle_counter++ ) {
            if ( class_high_res_limits[class_counter] > 10.0 )
                output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].sigma = 1.0;
        }
    }

    written_parameter_files = output_refinement->WritecisTEMStarFiles(main_frame->current_project.parameter_file_directory.GetFullPath( ) + "/auto_output_par", 1.0f, 0.0f, true);

    int  class_counter;
    long counter;
    int  job_counter;
    long number_of_reconstruction_jobs;
    long number_of_reconstruction_processes;

    long number_of_particles;
    long first_particle;
    long last_particle;

    // for now, number of jobs is number of processes -1 (master)..

    number_of_particles = active_refinement_package->contained_particles.GetCount( );

    number_of_reconstruction_processes = std::min(number_of_particles, active_reconstruction_run_profile.ReturnTotalJobs( ));
    number_of_reconstruction_jobs      = number_of_reconstruction_processes;

    my_parent->current_job_package.Reset(active_reconstruction_run_profile, "reconstruct3d", number_of_reconstruction_jobs * active_refinement_package->number_of_classes);

    for ( class_counter = 0; class_counter < active_refinement_package->number_of_classes; class_counter++ ) {

        for ( job_counter = 0; job_counter < number_of_reconstruction_jobs; job_counter++ ) {

            FirstLastParticleForJob(first_particle, last_particle, number_of_particles, job_counter + 1, number_of_reconstruction_jobs);

            wxString input_particle_stack           = active_refinement_package->stack_filename;
            wxString input_parameter_file           = written_parameter_files[class_counter];
            wxString output_reconstruction_1        = "/dev/null";
            wxString output_reconstruction_2        = "/dev/null";
            wxString output_reconstruction_filtered = "/dev/null";
            wxString output_resolution_statistics   = wxString::Format("/tmp/stats_%i\n", int(first_particle));
            wxString my_symmetry                    = active_refinement_package->symmetry;

            float output_pixel_size  = active_refinement_package->output_pixel_size;
            float molecular_mass_kDa = active_refinement_package->estimated_particle_weight_in_kda;
            float inner_mask_radius  = active_inner_mask_radius;
            float outer_mask_radius  = active_mask_radius;
            float resolution_limit_rec;

            if ( this_is_the_final_round == true )
                resolution_limit_rec = 0;
            else {
                float padded_high_res_limit = input_refinement->class_refinement_results[class_counter].class_resolution_statistics.ReturnResolutionNShellsAfter(class_high_res_limits[class_counter], output_refinement->resolution_statistics_box_size / 10);

                if ( resolution_per_round.GetCount( ) > 0 ) {
                    if ( fabsf(resolution_per_round[resolution_per_round.GetCount( ) - 1] - padded_high_res_limit) < resolution_per_round[resolution_per_round.GetCount( ) - 1] * 0.05 ) {
                        resolution_limit_rec = input_refinement->class_refinement_results[class_counter].class_resolution_statistics.ReturnResolutionNShellsAfter(resolution_per_round[resolution_per_round.GetCount( ) - 1], output_refinement->resolution_statistics_box_size / 10);
                    }
                    else
                        resolution_limit_rec = padded_high_res_limit;
                }
                else
                    resolution_limit_rec = padded_high_res_limit;
            }
            //wxPrintf("\n\n\n\nres limit = %.2f\n\n\n\n", resolution_limit_rec);

            if ( resolution_limit_rec > last_round_reconstruction_resolution )
                resolution_limit_rec = last_round_reconstruction_resolution;
            last_round_reconstruction_resolution = resolution_limit_rec;

            float score_weight_conversion;
            if ( class_high_res_limits[class_counter] < 8 )
                score_weight_conversion = 2;
            else
                score_weight_conversion = 0.0;

            float score_threshold;
            if ( current_percent_used * 3.0f < 100.0f )
                score_threshold = 0.333f; // we are refining 3 times more then current_percent_used, we want to use current percent used so it is always 1/3.
            else
                score_threshold = current_percent_used / 100.0; // now 3 times current_percent_used is more than 100%, we therefire refined them all, and so just take current_percent used

            if ( score_threshold > 1.0f )
                score_threshold = 1.0f;
            // OVERIDES ABOVE!
            //score_threshold = 0.0;

            bool     adjust_scores   = true; //my_parent->AdjustScoreForDefocusYesRadio->GetValue();
            bool     invert_contrast = active_refinement_package->stack_has_white_protein;
            bool     crop_images     = active_auto_crop;
            bool     dump_arrays     = true;
            wxString dump_file_1     = main_frame->ReturnAutoRefine3DScratchDirectory( ) + wxString::Format("dump_file_%li_%i_odd_%i.dump", current_output_refinement_id, class_counter, job_counter + 1);
            wxString dump_file_2     = main_frame->ReturnAutoRefine3DScratchDirectory( ) + wxString::Format("dump_file_%li_%i_even_%i.dump", current_output_refinement_id, class_counter, job_counter + 1);

            wxString input_reconstruction;
            bool     use_input_reconstruction;

            if ( active_should_apply_blurring == true ) {
                // do we have a reference..

                if ( active_refinement_package->references_for_next_refinement[class_counter] == -1 ) {
                    input_reconstruction     = "/dev/null";
                    use_input_reconstruction = false;
                }
                else {
                    input_reconstruction     = current_reference_filenames.Item(class_counter); //volume_asset_panel->ReturnAssetLongFilename(volume_asset_panel->ReturnArrayPositionFromAssetID(refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageSelectPanel->GetSelection()).references_for_next_refinement[class_counter]));
                    use_input_reconstruction = true;
                }
            }
            else {
                input_reconstruction     = "/dev/null";
                use_input_reconstruction = false;
            }

            float resolution_limit_ref = class_high_res_limits[class_counter];
            float smoothing_factor     = active_smoothing_factor;
            float padding              = 1.0f;
            bool  normalize_particles  = true;
            bool  exclude_blank_edges  = false;
            bool  split_even_odd       = false;
            bool  centre_mass          = my_parent->AutoCenterYesRadioButton->GetValue( );

            bool threshold_input_3d = true;
            int  max_threads        = 1;

            int correct_ewald_sphere = 0;

            my_parent->current_job_package.AddJob("ttttttttiiffffffffffbbbbbbbbbbttii",
                                                  input_particle_stack.ToUTF8( ).data( ),
                                                  input_parameter_file.ToUTF8( ).data( ),
                                                  input_reconstruction.ToUTF8( ).data( ),
                                                  output_reconstruction_1.ToUTF8( ).data( ),
                                                  output_reconstruction_2.ToUTF8( ).data( ),
                                                  output_reconstruction_filtered.ToUTF8( ).data( ),
                                                  output_resolution_statistics.ToUTF8( ).data( ),
                                                  my_symmetry.ToUTF8( ).data( ),
                                                  first_particle,
                                                  last_particle,
                                                  output_pixel_size,
                                                  molecular_mass_kDa,
                                                  inner_mask_radius,
                                                  outer_mask_radius,
                                                  resolution_limit_rec,
                                                  resolution_limit_ref,
                                                  score_weight_conversion,
                                                  score_threshold,
                                                  smoothing_factor,
                                                  padding,
                                                  normalize_particles,
                                                  adjust_scores,
                                                  invert_contrast,
                                                  exclude_blank_edges,
                                                  crop_images,
                                                  split_even_odd,
                                                  centre_mass,
                                                  use_input_reconstruction,
                                                  threshold_input_3d,
                                                  dump_arrays,
                                                  dump_file_1.ToUTF8( ).data( ),
                                                  dump_file_2.ToUTF8( ).data( ),
                                                  correct_ewald_sphere,
                                                  max_threads);
        }
    }
}

// for now we take the paramter

template <class MyParentType>
void AutoRefinementManager<MyParentType>::RunReconstructionJob( ) {
    running_job_type                    = RECONSTRUCTION;
    number_of_received_particle_results = 0;
    expected_number_of_results          = output_refinement->ReturnNumberOfActiveParticlesInFirstClass( ) * output_refinement->number_of_classes;

    // in the future store the reconstruction parameters..
    // empty scratch directory..

    //	if (wxDir::Exists(main_frame->current_project.scratch_directory.GetFullPath() + "/Refine3D/") == true) wxFileName::Rmdir(main_frame->current_project.scratch_directory.GetFullPath() + "/Refine3D/", wxPATH_RMDIR_RECURSIVE);
    //	if (wxDir::Exists(main_frame->current_project.scratch_directory.GetFullPath() + "/Refine3D/") == false) wxFileName::Mkdir(main_frame->current_project.scratch_directory.GetFullPath() + "/Refine3D/");

    // launch a controller

    if ( output_refinement->number_of_classes > 1 )
        my_parent->WriteBlueText("Calculating Reconstructions...");
    else
        my_parent->WriteBlueText("Calculating Reconstruction...");

    current_job_id       = main_frame->job_controller.AddJob(my_parent, active_reconstruction_run_profile.manager_command, active_reconstruction_run_profile.gui_address);
    my_parent->my_job_id = current_job_id;

    if ( current_job_id != -1 ) {
        my_parent->SetNumberConnectedTextToZeroAndStartTracking( );
    }
    my_parent->ProgressBar->Pulse( );
}

template <class MyParentType>
void AutoRefinementManager<MyParentType>::SetupRefinementJob( ) {
    int  class_counter;
    int  number_of_classes;
    long counter;
    long number_of_refinement_jobs;
    int  number_of_refinement_processes;

    long  number_of_particles;
    long  first_particle;
    long  last_particle;
    float likelihood_to_global;
    bool  do_global_for_this_particle;

    bool use_gpu;

#ifdef SHOW_CISTEM_GPU_OPTIONS
    if ( my_parent->use_gpu_checkboxAR3D->GetValue( ) == true ) {
        use_gpu = true;
    }
    else {
        use_gpu = false;
    }
#else
    use_gpu = false;
#endif

    // Just to make sure it is working
    if ( true_if_not_configured_as_disabled ) {
        wxPrintf("Allowing multiple global refinements\n");
    }
    else {
        wxPrintf("Disallowing multiple global refinements\n");
    }
    // get the last refinement for the currently selected refinement package..

    wxArrayString written_parameter_files;
    wxArrayString written_res_files;

    float lowest_alignment_res    = FLT_MAX;
    float lowest_class_resolution = FLT_MAX;

    for ( class_counter = 0; class_counter < input_refinement->number_of_classes; class_counter++ ) {
        lowest_alignment_res = std::min(class_high_res_limits[class_counter], lowest_alignment_res);
    }

    // setup whether to do global or local refinement..

    for ( long particle_counter = 0; particle_counter < input_refinement->number_of_particles; particle_counter++ ) {
        // if the measured resolution is better than 5 angstroms and the alignment resolution is better than 8 and the highest resolution previously globaled is greater than 9, and then do a global
        if ( number_of_rounds_run == 0 )
            do_global_for_this_particle = true;
        else if ( resolution_per_round[resolution_per_round.GetCount( ) - 1] < 5.0f && lowest_alignment_res <= 8.0f && resolution_of_last_global_alignment[particle_counter] > 9.0f && reference_3d_contains_all_particles == true && number_of_rounds_run > 2 )
            do_global_for_this_particle = true_if_not_configured_as_disabled;
        else {

            // should we do local or global? Randonly decide..

            float round_adjust = powf(number_of_global_alignments[particle_counter] - floor(rounds_since_global_alignment[particle_counter] / 3), 2);
            if ( round_adjust < 1 )
                round_adjust = 1;

            // if the alignment resolution has greatly changed make it more likely to global..

            float res_adjust = resolution_of_last_global_alignment[particle_counter] - lowest_alignment_res;

            float likelihood_to_global;
            if ( resolution_of_last_global_alignment[particle_counter] <= 5.0 )
                likelihood_to_global = -5; // don't bother doing another if we've already done one at this resolution
            else
                likelihood_to_global = powf(lowest_alignment_res, 2) / ((1000.0f / res_adjust) * round_adjust); // very arbritrary

            if ( fabsf(global_random_number_generator.GetUniformRandom( )) < likelihood_to_global )
                do_global_for_this_particle = true_if_not_configured_as_disabled;
            else
                do_global_for_this_particle = false;
        }

        for ( int class_counter = 0; class_counter < input_refinement->number_of_classes; class_counter++ ) {
            if ( number_of_global_alignments[particle_counter] == 0 )
                input_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].image_is_active = 0.0;
            else {
                if ( this_is_the_final_round == true )
                    input_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].image_is_active = 1.0;
                else if ( rounds_since_global_alignment[particle_counter] == 0 )
                    input_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].image_is_active = 1.0;
                else if ( do_global_for_this_particle == true ) {
                    input_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].image_is_active = 0.0;
                }
                else
                    input_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].image_is_active = 1.0;
            }
        }
    }

    // KIND OF A HACK FOR FIRST ROUND MULTIPLE CLASSES - PAY ATTENTION TO THIS!
    // It is put back to the correct number after the job is setup.

    if ( number_of_rounds_run == 0 ) {
        number_of_classes                   = input_refinement->number_of_classes;
        input_refinement->number_of_classes = 1;
    }

    written_parameter_files = input_refinement->WritecisTEMStarFiles(main_frame->current_project.parameter_file_directory.GetFullPath( ) + "/auto_input_par", 1.0f, 0.0f, true);
    written_res_files       = input_refinement->WriteResolutionStatistics(main_frame->current_project.parameter_file_directory.GetFullPath( ) + "/auto_input_stats");

    // for now, number of jobs is number of processes -1 (master)..

    number_of_particles = active_refinement_package->contained_particles.GetCount( );

    number_of_refinement_processes = std::min(number_of_particles, active_refinement_run_profile.ReturnTotalJobs( ));
    number_of_refinement_jobs      = number_of_refinement_processes;

    if ( use_gpu ) {
        my_parent->current_job_package.Reset(active_refinement_run_profile, "refine3d_gpu", number_of_refinement_jobs * input_refinement->number_of_classes);
    }
    else {
        my_parent->current_job_package.Reset(active_refinement_run_profile, "refine3d", number_of_refinement_jobs * input_refinement->number_of_classes);
    }

    for ( class_counter = 0; class_counter < input_refinement->number_of_classes; class_counter++ ) {

        for ( counter = 0; counter < number_of_refinement_jobs; counter++ ) {

            FirstLastParticleForJob(first_particle, last_particle, number_of_particles, counter + 1, number_of_refinement_jobs);

            wxString input_particle_images           = active_refinement_package->stack_filename;
            wxString input_parameter_file            = written_parameter_files.Item(class_counter);
            wxString input_reconstruction            = current_reference_filenames.Item(class_counter);
            wxString input_reconstruction_statistics = written_res_files.Item(class_counter);
            bool     use_statistics                  = true;

            wxString ouput_matching_projections = "";
            wxString ouput_shift_file           = "/dev/null";

            wxString my_symmetry = active_refinement_package->symmetry;

            float percent_used;
            if ( number_of_rounds_run == 0 )
                percent_used = 1.0f;
            else
                percent_used = (current_percent_used * 3.0) / 100.0;
            if ( percent_used > 1 )
                percent_used = 1;

                // OVERIDES ABOVE
                //if (number_of_rounds_run == 0) percent_used = 1.0f;
                //else percent_used = current_percent_used * 0.01;
                //if (percent_used > 1) percent_used = 1;
#ifdef CISTEM_DEBUG
            wxString output_parameter_file = wxString::Format("/tmp/output_par_%li_%li.star", first_particle, last_particle);
#else
            wxString output_parameter_file = "/dev/null";
#endif

            // for now we take the paramters of the first image!!!!

            float output_pixel_size     = active_refinement_package->output_pixel_size;
            float molecular_mass_kDa    = active_refinement_package->estimated_particle_weight_in_kda;
            float mask_radius           = active_mask_radius;
            float inner_mask_radius     = active_inner_mask_radius;
            float low_resolution_limit  = active_low_resolution_limit;
            float high_resolution_limit = class_high_res_limits[class_counter];

            float signed_CC_limit;
            signed_CC_limit = 0.0f;

            //if (IsOdd(number_of_rounds_run) == true || this_is_the_final_round == true || number_of_rounds_run == 0) signed_CC_limit = 0.0f;
            //else signed_CC_limit = 15.0f;

            //float	 classification_resolution_limit		= 10.0;//class_high_res_limits[class_counter]; //my_parent->ClassificationHighResLimitTextCtrl->ReturnValue();
            float classification_resolution_limit = 20.0f + (8.0f - 20.0f) * (float(number_of_rounds_run) / 9.0f);
            if ( classification_resolution_limit < lowest_alignment_res )
                classification_resolution_limit = lowest_alignment_res;

            wxPrintf("classification res limit %i = %.2f\n", class_counter, classification_resolution_limit);

            float mask_radius_search           = active_global_mask_radius;
            float high_resolution_limit_search = class_high_res_limits[class_counter];
            float angular_step                 = std::max(CalculateAngularStep(class_high_res_limits.Item(class_counter), active_mask_radius), CalculateAngularStep(8.0, active_mask_radius));
            int   best_parameters_to_keep      = active_number_results_to_refine;
            float max_search_x                 = active_search_range_x;
            float max_search_y                 = active_search_range_y;
            float mask_center_2d_x             = 0; //my_parent->SphereXTextCtrl->ReturnValue();
            float mask_center_2d_y             = 0; //my_parent->SphereYTextCtrl->ReturnValue();
            float mask_center_2d_z             = 0; //my_parent->SphereZTextCtrl->ReturnValue();
            float mask_radius_2d               = 0; //my_parent->SphereRadiusTextCtrl->ReturnValue();

            float defocus_search_range = 0; //my_parent->DefocusSearchRangeTextCtrl->ReturnValue();
            float defocus_step         = 0; //my_parent->DefocusSearchStepTextCtrl->ReturnValue();
            float padding              = 1.0;

            bool do_global_search                      = false;
            bool do_local_refinement                   = false;
            bool do_global_search_and_local_refinement = true;

            /*
			if (number_of_rounds_run == 0)
			{
				do_global_search = true;
				do_local_refinement = false;
			}
			else
			{
				do_global_search = false;
				do_local_refinement = true;

			}*/
            /*
			if (my_parent->GlobalRefinementRadio->GetValue() == true)
			{
				do_global_search = true;
				do_local_refinement = false;
			}
			else
			{
				do_global_search = false;
				do_local_refinement = true;
			}*/

            bool refine_psi                     = true; //my_parent->RefinePsiCheckBox->GetValue();
            bool refine_theta                   = true; //my_parent->RefineThetaCheckBox->GetValue();
            bool refine_phi                     = true; //my_parent->RefinePhiCheckBox->GetValue();
            bool refine_x_shift                 = true; //my_parent->RefineXShiftCheckBox->GetValue();
            bool refine_y_shift                 = true; //my_parent->RefineYShiftCheckBox->GetValue();
            bool calculate_matching_projections = false;
            bool apply_2d_masking               = false; //my_parent->SphereClassificatonYesRadio->GetValue();
            bool ctf_refinement                 = false;
            bool invert_contrast                = active_refinement_package->stack_has_white_protein;

            bool normalize_particles = true;
            bool exclude_blank_edges = false;
            bool normalize_input_3d;

            if ( active_should_apply_blurring == true )
                normalize_input_3d = false;
            else
                normalize_input_3d = true;

            bool threshold_input_3d      = true;
            bool ignore_input_parameters = false;
            bool defocus_bias            = false;

            int max_threads = 1;

            my_parent->current_job_package.AddJob("ttttbttttiiffffffffffffifffffffffbbbbbbbbbbbbbbbibibb",
                                                  input_particle_images.ToUTF8( ).data( ),
                                                  input_parameter_file.ToUTF8( ).data( ),
                                                  input_reconstruction.ToUTF8( ).data( ),
                                                  input_reconstruction_statistics.ToUTF8( ).data( ),
                                                  use_statistics,
                                                  ouput_matching_projections.ToUTF8( ).data( ),
                                                  output_parameter_file.ToUTF8( ).data( ),
                                                  ouput_shift_file.ToUTF8( ).data( ),
                                                  my_symmetry.ToUTF8( ).data( ),
                                                  first_particle,
                                                  last_particle,
                                                  percent_used,
                                                  output_pixel_size,
                                                  molecular_mass_kDa,
                                                  inner_mask_radius,
                                                  mask_radius,
                                                  low_resolution_limit,
                                                  high_resolution_limit,
                                                  signed_CC_limit,
                                                  classification_resolution_limit,
                                                  mask_radius_search,
                                                  high_resolution_limit_search,
                                                  angular_step,
                                                  best_parameters_to_keep,
                                                  max_search_x,
                                                  max_search_y,
                                                  mask_center_2d_x,
                                                  mask_center_2d_y,
                                                  mask_center_2d_z,
                                                  mask_radius_2d,
                                                  defocus_search_range,
                                                  defocus_step,
                                                  padding,
                                                  do_global_search,
                                                  do_local_refinement,
                                                  refine_psi,
                                                  refine_theta,
                                                  refine_phi,
                                                  refine_x_shift,
                                                  refine_y_shift,
                                                  calculate_matching_projections,
                                                  apply_2d_masking,
                                                  ctf_refinement,
                                                  normalize_particles,
                                                  invert_contrast,
                                                  exclude_blank_edges,
                                                  normalize_input_3d,
                                                  threshold_input_3d,
                                                  max_threads,
                                                  do_global_search_and_local_refinement,
                                                  class_counter,
                                                  ignore_input_parameters,
                                                  defocus_bias);
        }
    }

    if ( number_of_rounds_run == 0 ) {
        input_refinement->number_of_classes = number_of_classes;
    }
}

template <class MyParentType>
void AutoRefinementManager<MyParentType>::ProcessJobResult(JobResult* result_to_process) {
    if ( running_job_type == REFINEMENT ) {

        int  current_class    = int(result_to_process->result_data[0] + 0.5);
        long current_particle = long(result_to_process->result_data[1] + 0.5) - 1;

        /*		wxPrintf("Recieved a result for particle %li, x_shift = %f, y_shift = %f, psi = %f, theta = %f, phi = %f\n",	long(result_to_process->result_data[1] + 0.5),
																														result_to_process->result_data[6],
																														result_to_process->result_data[7],
																														result_to_process->result_data[3],
																														result_to_process->result_data[4],
																														result_to_process->result_data[5]);*/

        MyDebugAssertTrue(current_particle != -1 && current_class != -1, "Current Particle (%li) or Current Class(%i) = -1!", current_particle, current_class);

        //		wxPrintf("Received a refinement result for class #%i, particle %li\n", current_class + 1, current_particle + 1);
        //wxPrintf("output refinement has %i classes and %li particles\n", output_refinement->number_of_classes, output_refinement->number_of_particles);

        output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].position_in_stack                  = long(result_to_process->result_data[1] + 0.5);
        output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].image_is_active                    = int(result_to_process->result_data[2]);
        output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].psi                                = result_to_process->result_data[3];
        output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].theta                              = result_to_process->result_data[4];
        output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].phi                                = result_to_process->result_data[5];
        output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].xshift                             = result_to_process->result_data[6];
        output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].yshift                             = result_to_process->result_data[7];
        output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].defocus1                           = result_to_process->result_data[8];
        output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].defocus2                           = result_to_process->result_data[9];
        output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].defocus_angle                      = result_to_process->result_data[10];
        output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].phase_shift                        = result_to_process->result_data[11];
        output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].occupancy                          = result_to_process->result_data[12];
        output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].logp                               = result_to_process->result_data[13];
        output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].sigma                              = result_to_process->result_data[14];
        output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].score                              = result_to_process->result_data[15];
        output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].pixel_size                         = result_to_process->result_data[17];
        output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].microscope_voltage_kv              = result_to_process->result_data[18];
        output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].microscope_spherical_aberration_mm = result_to_process->result_data[19];
        output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].beam_tilt_x                        = result_to_process->result_data[20];
        output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].beam_tilt_y                        = result_to_process->result_data[21];
        output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].image_shift_x                      = result_to_process->result_data[22];
        output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].image_shift_y                      = result_to_process->result_data[23];
        output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].amplitude_contrast                 = result_to_process->result_data[24];
        output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].assigned_subset                    = result_to_process->result_data[25];

        number_of_received_particle_results++;
        //wxPrintf("received result!\n");
        long current_time = time(NULL);

        if ( number_of_received_particle_results == 1 ) {
            current_job_starttime = current_time;
            time_of_last_update   = 0;

            float percent_used_used_multiplier;
            if ( number_of_rounds_run == 0 )
                percent_used_used_multiplier = 1.0f;
            else
                percent_used_used_multiplier = (current_percent_used * 3.0f) * 0.01f;

            if ( percent_used_used_multiplier > 1.0f )
                percent_used_used_multiplier = 1.0f;

            my_parent->ShowRefinementResultsPanel->AngularPlotPanel->SetSymmetryAndNumber(active_refinement_package->symmetry, long(float(output_refinement->number_of_particles) * percent_used_used_multiplier));
            my_parent->Layout( );
        }
        else if ( current_time != time_of_last_update ) {
            int current_percentage;
            if ( number_of_rounds_run == 0 )
                current_percentage = float(number_of_received_particle_results) / float(output_refinement->number_of_particles) * 100.0; // always 1 class for first round
            else
                current_percentage = float(number_of_received_particle_results) / float(output_refinement->number_of_particles * output_refinement->number_of_classes) * 100.0;

            time_of_last_update = current_time;
            if ( current_percentage > 100 )
                current_percentage = 100;
            my_parent->ProgressBar->SetValue(current_percentage);

            long  job_time        = current_time - current_job_starttime;
            float seconds_per_job = float(job_time) / float(number_of_received_particle_results - 1);

            long seconds_remaining;
            if ( number_of_rounds_run == 0 )
                seconds_remaining = float((input_refinement->number_of_particles) - number_of_received_particle_results) * seconds_per_job;
            else
                seconds_remaining = float((input_refinement->number_of_particles * output_refinement->number_of_classes) - number_of_received_particle_results) * seconds_per_job;

            wxTimeSpan time_remaining = wxTimeSpan(0, 0, seconds_remaining);

            my_parent->TimeRemainingText->SetLabel(time_remaining.Format("Time Remaining : %Hh:%Mm:%Ss"));
        }

        // Add this result to the list of results to be plotted onto the angular plot
        if ( current_class == 0 && output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].image_is_active >= 0 ) {
            my_parent->ShowRefinementResultsPanel->AngularPlotPanel->AddRefinementResult(&output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle]);
            // Plot this new result onto the angular plot immediately if it's one of the first few results to come in. Otherwise, only plot at regular intervals.

            if ( my_parent->ShowRefinementResultsPanel->AngularPlotPanel->refinement_results_to_plot.Count( ) * my_parent->ShowRefinementResultsPanel->AngularPlotPanel->symmetry_matrices.number_of_matrices < 1500 || current_time - my_parent->time_of_last_result_update > 0 ) {

                my_parent->ShowRefinementResultsPanel->AngularPlotPanel->Refresh( );
                my_parent->time_of_last_result_update = current_time;
            }
        }
    }
    else if ( running_job_type == RECONSTRUCTION ) {
        //wxPrintf("Got reconstruction job \n");
        number_of_received_particle_results++;
        //	wxPrintf("Received a reconstruction intermmediate result\n");

        long current_time = time(NULL);

        if ( number_of_received_particle_results == 1 ) {
            time_of_last_update   = 0;
            current_job_starttime = current_time;
        }
        else if ( current_time - time_of_last_update >= 1 ) {
            time_of_last_update    = current_time;
            int current_percentage = float(number_of_received_particle_results) / float(expected_number_of_results) * 100.0;
            if ( current_percentage > 100 )
                current_percentage = 100;
            my_parent->ProgressBar->SetValue(current_percentage);
            long  job_time          = current_time - current_job_starttime;
            float seconds_per_job   = float(job_time) / float(number_of_received_particle_results - 1);
            long  seconds_remaining = float(expected_number_of_results - number_of_received_particle_results) * seconds_per_job;

            wxTimeSpan time_remaining = wxTimeSpan(0, 0, seconds_remaining);
            my_parent->TimeRemainingText->SetLabel(time_remaining.Format("Time Remaining : %Hh:%Mm:%Ss"));
        }
    }
    else if ( running_job_type == MERGE ) {
        //	wxPrintf("received merge result!\n");

        // add to the correct resolution statistics..

        int   number_of_points = result_to_process->result_data[0];
        int   class_number     = int(result_to_process->result_data[1] + 0.5);
        int   array_position   = 2;
        float current_resolution;
        float fsc;
        float part_fsc;
        float part_ssnr;
        float rec_ssnr;

        wxPrintf("class_number = %i\n", class_number);
        // add the points..

        output_refinement->class_refinement_results[class_number - 1].class_resolution_statistics.Init(output_refinement->resolution_statistics_pixel_size, output_refinement->resolution_statistics_box_size);

        output_refinement->class_refinement_results[class_number - 1].class_resolution_statistics.FSC.ClearData( );
        output_refinement->class_refinement_results[class_number - 1].class_resolution_statistics.part_FSC.ClearData( );
        output_refinement->class_refinement_results[class_number - 1].class_resolution_statistics.part_SSNR.ClearData( );
        output_refinement->class_refinement_results[class_number - 1].class_resolution_statistics.rec_SSNR.ClearData( );

        for ( int counter = 0; counter < number_of_points; counter++ ) {
            current_resolution = result_to_process->result_data[array_position];
            array_position++;
            fsc = result_to_process->result_data[array_position];
            array_position++;
            part_fsc = result_to_process->result_data[array_position];
            array_position++;
            part_ssnr = result_to_process->result_data[array_position];
            array_position++;
            rec_ssnr = result_to_process->result_data[array_position];
            array_position++;

            output_refinement->class_refinement_results[class_number - 1].class_resolution_statistics.FSC.AddPoint(current_resolution, fsc);
            output_refinement->class_refinement_results[class_number - 1].class_resolution_statistics.part_FSC.AddPoint(current_resolution, part_fsc);
            output_refinement->class_refinement_results[class_number - 1].class_resolution_statistics.part_SSNR.AddPoint(current_resolution, part_ssnr);
            output_refinement->class_refinement_results[class_number - 1].class_resolution_statistics.rec_SSNR.AddPoint(current_resolution, rec_ssnr);
        }
    }
}

template <class MyParentType>
void AutoRefinementManager<MyParentType>::ProcessAllJobsFinished( ) {

    // Update the GUI with project timings
    extern OverviewPanel* overview_panel;
    overview_panel->SetProjectInfo( );

    //
    long  position_in_stack;
    float psi;
    float theta;
    float phi;
    float xshift;
    float yshift;
    float defocus1;
    float defocus2;
    float defocus_angle;
    float phase_shift;
    float occupancy;
    float logp;
    float sigma;
    float score;
    int   image_is_active;

    if ( running_job_type == REFINEMENT ) {
        main_frame->job_controller.KillJob(my_parent->my_job_id);

        // if this is the first round of a multiple class refinement then do random occupancies.
        // This is based on just hacking input_refinement->number_of_classes = 1 in begin refinement cycle

        if ( number_of_rounds_run == 0 && output_refinement->number_of_classes > 1 ) {
            int  class_counter;
            long particle_counter;

            for ( class_counter = 0; class_counter < output_refinement->number_of_classes; class_counter++ ) {
                for ( particle_counter = 0; particle_counter < output_refinement->number_of_particles; particle_counter++ ) {
                    output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].occupancy = fabsf(global_random_number_generator.GetUniformRandom( ) * (200.0f / float(output_refinement->number_of_classes)));

                    output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].position_in_stack                  = output_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].position_in_stack;
                    output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].phi                                = output_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].phi;
                    output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].theta                              = output_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].theta;
                    output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].psi                                = output_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].psi;
                    output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].xshift                             = output_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].xshift;
                    output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].yshift                             = output_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].yshift;
                    output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].score                              = output_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].score;
                    output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].image_is_active                    = output_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].image_is_active;
                    output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].sigma                              = output_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].sigma;
                    output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].defocus1                           = output_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].defocus1;
                    output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].defocus2                           = output_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].defocus2;
                    output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].defocus_angle                      = output_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].defocus_angle;
                    output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].phase_shift                        = output_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].phase_shift;
                    output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].logp                               = output_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].logp;
                    output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].score                              = output_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].score;
                    output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].pixel_size                         = output_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].pixel_size;
                    output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].microscope_voltage_kv              = output_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].microscope_voltage_kv;
                    output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].microscope_spherical_aberration_mm = output_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].microscope_spherical_aberration_mm;
                    output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].beam_tilt_x                        = output_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].beam_tilt_x;
                    output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].beam_tilt_y                        = output_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].beam_tilt_y;
                    output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].image_shift_x                      = output_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].image_shift_x;
                    output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].image_shift_y                      = output_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].image_shift_y;
                    output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].amplitude_contrast                 = output_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].amplitude_contrast;
                    output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].assigned_subset                    = output_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].assigned_subset;
                }

                output_refinement->class_refinement_results[class_counter].average_occupancy = 100.0f / output_refinement->number_of_classes;
            }
        }
        else // calculate occupancies..
        {
            if ( output_refinement->percent_used < 99.99 )
                output_refinement->UpdateOccupancies(false);
            else
                output_refinement->UpdateOccupancies(true);
        }

        SetupReconstructionJob( );
        RunReconstructionJob( );
    }
    else if ( running_job_type == RECONSTRUCTION ) {
        main_frame->job_controller.KillJob(my_parent->my_job_id);
        //wxPrintf("Reconstruction has finished\n");
        SetupMerge3dJob( );
        RunMerge3dJob( );
    }
    else if ( running_job_type == MERGE ) {
        long  current_reconstruction_id;
        float current_resolution_limit_rec;
        float current_score_weight_conversion;

        // launch drawer thread..

        main_frame->ClearAutoRefine3DScratch( );

        my_parent->active_orth_thread_id = my_parent->next_thread_id;
        my_parent->next_thread_id++;
        OrthDrawerThread* result_thread = new OrthDrawerThread(my_parent, current_reference_filenames, wxString::Format("Iter. #%i", number_of_rounds_run + 1), 1.0f, active_mask_radius / input_refinement->resolution_statistics_pixel_size, my_parent->active_orth_thread_id);

        if ( result_thread->Run( ) != wxTHREAD_NO_ERROR ) {
            my_parent->WriteErrorText("Error: Cannot start result creation thread, results not displayed");
            delete result_thread;
        }

        int class_counter;

        main_frame->job_controller.KillJob(my_parent->my_job_id);

        VolumeAsset temp_asset;

        temp_asset.pixel_size = output_refinement->resolution_statistics_pixel_size;
        temp_asset.x_size     = output_refinement->resolution_statistics_box_size;
        temp_asset.y_size     = output_refinement->resolution_statistics_box_size;
        temp_asset.z_size     = output_refinement->resolution_statistics_box_size;

        // add the volumes etc to the database..

        output_refinement->reference_volume_ids.Clear( );
        active_refinement_package->references_for_next_refinement.Clear( );

        main_frame->current_project.database.Begin( );
        main_frame->current_project.database.BeginVolumeAssetInsert( );

        my_parent->WriteInfoText("");

        for ( class_counter = 0; class_counter < output_refinement->number_of_classes; class_counter++ ) {
            temp_asset.asset_id = volume_asset_panel->current_asset_number;

            // add the reconstruction, get a reconstruction_id

            current_reconstruction_id        = main_frame->current_project.database.ReturnHighestReconstructionID( ) + 1;
            temp_asset.reconstruction_job_id = current_reconstruction_id;

            // add the reconstruction job

            if ( this_is_the_final_round == true )
                current_resolution_limit_rec = 0;
            else
                current_resolution_limit_rec = input_refinement->class_refinement_results[class_counter].class_resolution_statistics.ReturnResolutionNShellsAfter(class_high_res_limits[class_counter], output_refinement->resolution_statistics_box_size / 10);

            if ( class_high_res_limits[class_counter] < 8 )
                current_score_weight_conversion = 2;
            else
                current_score_weight_conversion = 0.0;

            main_frame->current_project.database.AddReconstructionJob(current_reconstruction_id, active_refinement_package->asset_id, output_refinement->refinement_id, "", active_inner_mask_radius, active_mask_radius, current_resolution_limit_rec, current_score_weight_conversion, false, active_auto_crop, false, active_should_apply_blurring, active_smoothing_factor, class_counter + 1, long(temp_asset.asset_id));

            temp_asset.asset_name = wxString::Format("Auto #%li (Rnd. %i) - Class #%i", current_output_refinement_id, number_of_rounds_run + 1, class_counter + 1);
            temp_asset.filename   = main_frame->current_project.volume_asset_directory.GetFullPath( ) + wxString::Format("/volume_%li_%i.mrc", output_refinement->refinement_id, class_counter + 1);

            output_refinement->reference_volume_ids.Add(current_reference_asset_ids[class_counter]);
            current_reference_asset_ids[class_counter] = temp_asset.asset_id;

            // set the output volume
            output_refinement->class_refinement_results[class_counter].reconstructed_volume_asset_id = temp_asset.asset_id;
            output_refinement->class_refinement_results[class_counter].reconstruction_id             = current_reconstruction_id;

            active_refinement_package->references_for_next_refinement.Add(temp_asset.asset_id);
            main_frame->current_project.database.ExecuteSQL(wxString::Format("UPDATE REFINEMENT_PACKAGE_CURRENT_REFERENCES_%li SET VOLUME_ASSET_ID=%i WHERE CLASS_NUMBER=%i", current_refinement_package_asset_id, temp_asset.asset_id, class_counter + 1));

            volume_asset_panel->AddAsset(&temp_asset);
            main_frame->current_project.database.AddNextVolumeAsset(temp_asset.asset_id, temp_asset.asset_name, temp_asset.filename.GetFullPath( ), temp_asset.reconstruction_job_id, temp_asset.pixel_size, temp_asset.x_size, temp_asset.y_size, temp_asset.z_size, temp_asset.half_map_1_filename.GetFullPath( ), temp_asset.half_map_2_filename.GetFullPath( ));
        }

        main_frame->current_project.database.EndVolumeAssetInsert( );
        wxArrayFloat average_occupancies = output_refinement->UpdatePSSNR( );

        my_parent->WriteInfoText("");

        if ( output_refinement->number_of_classes > 1 ) {
            for ( class_counter = 0; class_counter < output_refinement->number_of_classes; class_counter++ ) {
                my_parent->WriteInfoText(wxString::Format(wxT("Est. Res. Class %2i = %2.2f Å (%2.2f %%)"), class_counter + 1, output_refinement->class_refinement_results[class_counter].class_resolution_statistics.ReturnEstimatedResolution( ), average_occupancies[class_counter]));
            }
        }
        else {
            my_parent->WriteInfoText(wxString::Format(wxT("Est. Res. = %2.2f Å"), output_refinement->class_refinement_results[0].class_resolution_statistics.ReturnEstimatedResolution( )));
        }

        my_parent->WriteInfoText("");

        // calculate angular distribution histograms
        ArrayofAngularDistributionHistograms all_histograms = output_refinement->ReturnAngularDistributions(active_refinement_package->symmetry);

        for ( class_counter = 1; class_counter <= output_refinement->number_of_classes; class_counter++ ) {
            main_frame->current_project.database.AddRefinementAngularDistribution(all_histograms[class_counter - 1], output_refinement->refinement_id, class_counter);
        }

        main_frame->current_project.database.AddRefinement(output_refinement);
        ShortRefinementInfo temp_info;
        temp_info = output_refinement;
        refinement_package_asset_panel->all_refinement_short_infos.Add(temp_info);

        // add this refinment to the refinement package..

        active_refinement_package->last_refinment_id = output_refinement->refinement_id;
        active_refinement_package->refinement_ids.Add(output_refinement->refinement_id);

        main_frame->current_project.database.ExecuteSQL(wxString::Format("UPDATE REFINEMENT_PACKAGE_ASSETS SET LAST_REFINEMENT_ID=%li WHERE REFINEMENT_PACKAGE_ASSET_ID=%li", output_refinement->refinement_id, current_refinement_package_asset_id));
        main_frame->current_project.database.ExecuteSQL(wxString::Format("INSERT INTO REFINEMENT_PACKAGE_REFINEMENTS_LIST_%li (REFINEMENT_NUMBER, REFINEMENT_ID) VALUES (%li, %li);", current_refinement_package_asset_id, main_frame->current_project.database.ReturnSingleLongFromSelectCommand(wxString::Format("SELECT MAX(REFINEMENT_NUMBER) FROM REFINEMENT_PACKAGE_REFINEMENTS_LIST_%li", current_refinement_package_asset_id)) + 1, output_refinement->refinement_id));

        main_frame->current_project.database.Commit( );
        main_frame->DirtyVolumes( );
        main_frame->DirtyRefinements( );
        ///refinement_package_asset_panel->is_dirty = true;

        //		my_parent->SetDefaults();
        //refinement_results_panel->is_dirty = true;

        my_parent->ShowRefinementResultsPanel->FSCResultsPanel->AddRefinement(output_refinement);

        if ( my_parent->ShowRefinementResultsPanel->TopBottomSplitter->IsSplit( ) == false ) {
            my_parent->ShowRefinementResultsPanel->TopBottomSplitter->SplitHorizontally(my_parent->ShowRefinementResultsPanel->TopPanel, my_parent->ShowRefinementResultsPanel->BottomPanel);
            my_parent->ShowRefinementResultsPanel->FSCResultsPanel->Show(true);
        }

        my_parent->Layout( );

        //wxPrintf("Calling cycle refinement\n");
        main_frame->DirtyVolumes( );
        main_frame->DirtyRefinements( );

        // are we using all particles yet?

        if ( current_percent_used > 99.9f )
            reference_3d_contains_all_particles = true;
        CycleRefinement( );
    }
}

template <class MyParentType>
void AutoRefinementManager<MyParentType>::DoMasking( ) {
    MyDebugAssertTrue(active_should_auto_mask || active_should_mask, "DoMasking called, when masking not ticked!");
    MyDebugAssertFalse(active_should_auto_mask && active_should_mask, "Masking should either be from user file, or auto, not both");

    wxArrayString masked_filenames;
    wxFileName    current_ref_filename;
    wxString      current_masked_filename;
    wxString      filename_of_mask = active_mask_filename;

    for ( int class_counter = 0; class_counter < current_reference_filenames.GetCount( ); class_counter++ ) {
        current_ref_filename    = current_reference_filenames.Item(class_counter);
        current_masked_filename = main_frame->ReturnAutoRefine3DScratchDirectory( ) + current_ref_filename.GetName( );
        current_masked_filename += "_masked.mrc";

        masked_filenames.Add(current_masked_filename);
    }

    if ( active_should_mask ) {

        my_parent->WriteInfoText("Masking reference reconstruction with selected mask");

        float wanted_cosine_edge_width   = active_mask_edge;
        float wanted_weight_outside_mask = active_mask_weight;

        float wanted_low_pass_filter_radius;

        if ( active_should_low_pass_filter_mask == true ) {
            wanted_low_pass_filter_radius = active_mask_filter_resolution;
        }
        else {
            wanted_low_pass_filter_radius = 0.0;
        }

        my_parent->active_mask_thread_id = my_parent->next_thread_id;
        my_parent->next_thread_id++;

        Multiply3DMaskerThread* mask_thread = new Multiply3DMaskerThread(my_parent, current_reference_filenames, masked_filenames, filename_of_mask, wanted_cosine_edge_width, wanted_weight_outside_mask, wanted_low_pass_filter_radius, input_refinement->resolution_statistics_pixel_size, my_parent->active_mask_thread_id);

        if ( mask_thread->Run( ) != wxTHREAD_NO_ERROR ) {
            my_parent->WriteErrorText("Error: Cannot start masking thread, masking will not be performed");
            delete mask_thread;
        }
        else {
            current_reference_filenames = masked_filenames;
            return;
        }
    }
    else {

        my_parent->WriteInfoText("Automasking reference reconstruction");

        my_parent->active_mask_thread_id = my_parent->next_thread_id;
        my_parent->next_thread_id++;

        float current_res = input_refinement->class_refinement_results[0].class_resolution_statistics.ReturnEstimatedResolution(true);
        if ( current_res > class_high_res_limits[0] )
            current_res = class_high_res_limits[0];
        wxPrintf("Estimated resolution = %.2f\n", current_res);
        AutoMaskerThread* mask_thread = new AutoMaskerThread(my_parent, current_reference_filenames, masked_filenames, input_refinement->resolution_statistics_pixel_size, active_mask_radius, my_parent->active_mask_thread_id, current_res);

        if ( mask_thread->Run( ) != wxTHREAD_NO_ERROR ) {
            my_parent->WriteErrorText("Error: Cannot start masking thread, masking will not be performed");
            delete mask_thread;
        }
        else {
            current_reference_filenames = masked_filenames;
            return; // just return, we will startup again whent he mask thread finishes.
        }
    }
}

template <class MyParentType>
void AutoRefinementManager<MyParentType>::CycleRefinement( ) {
    percent_used_per_round.Add(current_percent_used);

    int  class_counter;
    long particle_counter;
    // use highest resoluton..
    float best_res      = FLT_MAX;
    float best_p143_res = FLT_MAX;
    float worse_res     = -FLT_MAX;

    bool should_stop = false;
    bool did_resolution_improve;

    float current_0p5_resolution;
    float current_part_0p5_resolution;
    float bleed_resolution;
    float res_for_next_round;
    float resolution_min_shells_after;
    float safe_resolution;
    float current_0p5_res_minus_bleed;
    float current_part_0p5_res_minus_bleed;
    float average_0p5_res_minus_bleed;

    float change_in_occupancies;

    int number_of_bleed_shells = ceil(output_refinement->resolution_statistics_box_size / (active_mask_radius / output_refinement->resolution_statistics_pixel_size));

    // loop over all particles, to see if they were actually active, and if so whether they were global or not this number should be consistent for all classes, so only check class 1

    for ( particle_counter = 0; particle_counter < output_refinement->number_of_particles; particle_counter++ ) {
        if ( input_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].image_is_active == 0 && output_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].image_is_active == 1 ) {
            number_of_global_alignments[particle_counter]++;
            rounds_since_global_alignment[particle_counter]       = 0;
            resolution_of_last_global_alignment[particle_counter] = high_res_limit_per_round[high_res_limit_per_round.GetCount( ) - 1];
        }
        else {
            rounds_since_global_alignment[particle_counter]++;
        }
    }

    for ( class_counter = 0; class_counter < output_refinement->number_of_classes; class_counter++ ) {
        current_0p5_resolution      = output_refinement->class_refinement_results[class_counter].class_resolution_statistics.Return0p5Resolution(false);
        current_part_0p5_resolution = output_refinement->class_refinement_results[class_counter].class_resolution_statistics.Return0p5Resolution(true);

        current_0p5_res_minus_bleed      = output_refinement->class_refinement_results[class_counter].class_resolution_statistics.ReturnResolutionNShellsBefore(current_0p5_resolution, number_of_bleed_shells + 1);
        current_part_0p5_res_minus_bleed = output_refinement->class_refinement_results[class_counter].class_resolution_statistics.ReturnResolutionNShellsBefore(current_part_0p5_resolution, number_of_bleed_shells + 1);

        if ( current_0p5_res_minus_bleed == 0 )
            current_0p5_res_minus_bleed = class_high_res_limits[class_counter];
        if ( current_part_0p5_res_minus_bleed == 0 )
            current_part_0p5_res_minus_bleed = class_high_res_limits[class_counter];

        average_0p5_res_minus_bleed = (current_0p5_res_minus_bleed + current_part_0p5_res_minus_bleed) * 0.5f;

        bleed_resolution            = output_refinement->class_refinement_results[class_counter].class_resolution_statistics.ReturnResolutionNShellsAfter(class_high_res_limits[class_counter], number_of_bleed_shells);
        resolution_min_shells_after = output_refinement->class_refinement_results[class_counter].class_resolution_statistics.ReturnResolutionNShellsAfter(class_high_res_limits[class_counter], output_refinement->resolution_statistics_box_size / 15);

        if ( bleed_resolution == 0 )
            bleed_resolution = class_high_res_limits[class_counter];
        safe_resolution = bleed_resolution;

        res_for_next_round = std::max(resolution_min_shells_after, average_0p5_res_minus_bleed);
        if ( res_for_next_round > class_high_res_limits.Item(class_counter) )
            res_for_next_round = class_high_res_limits.Item(class_counter);

        //if (res_for_next_round < 4.0) res_for_next_round = 4.0;

        class_high_res_limits[class_counter] = res_for_next_round;

        //	wxPrintf("\n\n\ncurrent_0p5_resolution = %.2f\n", current_0p5_resolution);
        //	wxPrintf("current_0p5_resolution_minus_bleed = %.2f\n", current_0p5_res_minus_bleed);
        //	wxPrintf("bleed_resolution = %.2f\n", bleed_resolution);
        //	wxPrintf("resolution_min_shells_after = %.2f\n", resolution_min_shells_after);
        //	wxPrintf("safe_resolution = %.2f\n", safe_resolution);
        //	wxPrintf("res_for_next_round = %.2f\n", res_for_next_round);
        //	wxPrintf("number_bleed_shells = %i\n\n\n\n", number_of_bleed_shells);

        if ( output_refinement->class_refinement_results[class_counter].class_resolution_statistics.Return0p5Resolution( ) < best_res ) {
            best_res = output_refinement->class_refinement_results[class_counter].class_resolution_statistics.Return0p5Resolution( );
        }

        if ( output_refinement->class_refinement_results[class_counter].class_resolution_statistics.ReturnEstimatedResolution( ) < best_p143_res ) {
            best_p143_res = output_refinement->class_refinement_results[class_counter].class_resolution_statistics.ReturnEstimatedResolution( );
        }
    }

    float lowest_res = FLT_MAX;

    for ( class_counter = 0; class_counter < input_refinement->number_of_classes; class_counter++ ) {
        lowest_res = std::min(class_high_res_limits[class_counter], lowest_res);
    }

    high_res_limit_per_round.Add(lowest_res);

    if ( resolution_per_round.GetCount( ) > 0 ) {
        if ( best_p143_res > resolution_per_round[resolution_per_round.GetCount( ) - 1] - 0.1 ) // the resolution did not improve, lets add more particles to the max
        {
            max_percent_used += max_percent_used * 2.0;
            if ( max_percent_used > 100 )
                max_percent_used = 100.0;
        }

        if ( best_p143_res < 7.0f )
            max_percent_used = 100.0f; // if the resolution is good, just use all the particles..
    }

    resolution_per_round.Add(best_p143_res);
    //current_percent_used *= 2;

    float estimated_required_asym_units = 8000.0f * expf(75.0f / powf(best_p143_res, 2));
    long  wanted_number_of_asym_units   = myroundint(estimated_required_asym_units) * output_refinement->number_of_classes;
    long  number_of_asym_units          = output_refinement->number_of_particles * ReturnNumberofAsymmetricUnits(active_refinement_package->symmetry);

    // what percentage is this.

    current_percent_used = (float(wanted_number_of_asym_units) / float(number_of_asym_units)) * 100.0;
    if ( current_percent_used < start_percent_used )
        current_percent_used = start_percent_used;
    if ( current_percent_used > 100.0 )
        current_percent_used = 100.0;

    if ( current_percent_used < max_percent_used )
        current_percent_used = max_percent_used;
    else
        (max_percent_used = current_percent_used);

    number_of_rounds_run++;

    //main_frame->DirtyRefinements();

    // what is the change in occupancies

    if ( output_refinement->number_of_classes == 1 )
        change_in_occupancies = 0.0f;
    else {
        change_in_occupancies = output_refinement->ReturnChangeInAverageOccupancy(*input_refinement);
    }

    int min_rounds_to_run;

    if ( output_refinement->number_of_classes == 1 )
        min_rounds_to_run = 5;
    else
        min_rounds_to_run = 10;

    if ( resolution_per_round.GetCount( ) >= min_rounds_to_run && max_percent_used > 99.0 && change_in_occupancies < 1.0 ) {
        should_stop            = true;
        float round_resolution = resolution_per_round[resolution_per_round.GetCount( ) - 3];

        for ( int round_counter = resolution_per_round.GetCount( ) - 2; round_counter <= resolution_per_round.GetCount( ) - 1; round_counter++ ) {
            if ( resolution_per_round[round_counter] < round_resolution - 0.001 )
                should_stop = false;
        }
    }

    if ( this_is_the_final_round == true ) {
        delete input_refinement;
        input_refinement = NULL;
        //delete output_refinement;
        my_parent->WriteBlueText("Resolution is stable - Auto refine is stopping.");
        my_parent->CancelAlignmentButton->Show(false);
        my_parent->FinishButton->Show(true);
        my_parent->TimeRemainingText->SetLabel(wxString::Format("All Done! (%s)", wxTimeSpan::Milliseconds(my_parent->stopwatch.Time( )).Format(wxT("%Hh:%Mm:%Ss"))));
        my_parent->ProgressBar->SetValue(100);
        my_parent->ProgressPanel->Layout( );
    }
    else {
        if ( should_stop == true ) {
            this_is_the_final_round = true;
            current_percent_used    = 100.0;
        }

        delete input_refinement;
        input_refinement                               = output_refinement;
        output_refinement                              = new Refinement;
        output_refinement->refinement_package_asset_id = input_refinement->refinement_package_asset_id;

        if ( active_should_mask == true || active_should_auto_mask == true ) {
            DoMasking( );
        }
        else {
            SetupRefinementJob( );
            RunRefinementJob( );
        }
    }
}

template <class MyParentType>
void AutoRefinementManager<MyParentType>::OnMaskerThreadComplete( ) {
    SetupRefinementJob( );
    RunRefinementJob( );
}

// Make sure we instantiate the relevant classes
template class AutoRefinementManager<AutoRefine3DPanelSpa>;
template class AutoRefinementManager<AutoRefine3DPanelRx>;