#define cisTEM_temp_disable_gpu_noFastFFT

//#include "../core/core_headers.h"
#include "../constants/constants.h"
#include "../core/gui_core_headers.h"
#include "TemplateMatchQueueManager.h"
#include "TemplateMatchQueueLogger.h"

// File-specific debug flags
// #define cisTEM_DEBUG_ORPHANED_SEARCH_IDS  // Uncomment to enable orphaned search_id checking

extern MyImageAssetPanel*  image_asset_panel;
extern MyVolumeAssetPanel* volume_asset_panel;
extern MyRunProfilesPanel* run_profiles_panel;
extern MyMainFrame*        main_frame;

// Global pointer to results panel (defined in projectx.cpp) - used to mark results dirty after database updates,
// retrieve active job IDs from the results view, and access the list of template match job IDs
extern MatchTemplateResultsPanel* match_template_results_panel;

MatchTemplatePanel::MatchTemplatePanel(wxWindow* parent)
    : MatchTemplatePanelParent(parent) {

    // Core template matching panel state
    my_job_id              = -1; // Job controller ID for tracking and killing distributed computation jobs (-1 = idle)
    running_job            = false;
    group_combo_is_dirty   = false;
    run_profiles_are_dirty = false;
    current_pixel_size     = -1.0f; // Initialize to invalid value to force first update

    // Queue manager integration
    running_queue_id                = -1; // Database queue ID of the currently executing search (-1 when idle)
    current_custom_cli_args         = ""; // Clear custom CLI args (used by queue for custom parameters)
    block_auto_progression_of_queue = false; // Allow queue auto-advance by default

    // Create persistent queue manager instance
    queue_manager = new TemplateMatchQueueManager(this, this);
    queue_manager->Hide( ); // Hidden until dialog is shown
    queue_manager->LoadQueueFromDatabase( ); // Initial load

#ifndef SHOW_CISTEM_GPU_OPTIONS
    UseGPURadioYes->Enable(false);
    UseGPURadioNo->Enable(false);
    UseFastFFTRadioYes->Enable(false);
    UseFastFFTRadioNo->Enable(false);
#endif

#ifndef cisTEM_USING_FastFFT
    UseFastFFTRadioYes->Enable(false);
    UseFastFFTRadioNo->Enable(false);
#endif

#ifdef cisTEM_temp_disable_gpu_noFastFFT
    UseGPURadioYes->Enable(false);
    UseGPURadioNo->Enable(false);
#endif

    // We need to allow a higher precision, otherwise, the option to resample will almost always be taken
    HighResolutionLimitNumericCtrl->SetPrecision(4);
    SetInfo( );
    FillGroupComboBox( );
    FillRunProfileComboBox( );

    wxSize input_size = InputSizer->GetMinSize( );
    input_size.x += wxSystemSettings::GetMetric(wxSYS_VSCROLL_X);
    input_size.y = -1;
    ExpertPanel->SetMinSize(input_size);
    ExpertPanel->SetSize(input_size);

    result_bitmap.Create(1, 1, 24);
    time_of_last_result_update = time(NULL);

    // Set up control constraints before setting defaults
    OutofPlaneStepNumericCtrl->SetMinMaxValue(0.0f, 360.f);
    InPlaneStepNumericCtrl->SetMinMaxValue(0.0f, 360.f);
    MinPeakRadiusNumericCtrl->SetMinMaxValue(0.0f, FLT_MAX);
    DefocusSearchRangeNumericCtrl->SetMinMaxValue(0.0f, FLT_MAX);
    DefocusSearchStepNumericCtrl->SetMinMaxValue(1.0f, FLT_MAX);
    PixelSizeSearchRangeNumericCtrl->SetMinMaxValue(0.0f, FLT_MAX);
    PixelSizeSearchStepNumericCtrl->SetMinMaxValue(0.01f, FLT_MAX);
    HighResolutionLimitNumericCtrl->SetMinMaxValue(0.0f, FLT_MAX);

    // Populate combo boxes before setting defaults
    SymmetryComboBox->Clear( );
    SymmetryComboBox->Append("C1");
    SymmetryComboBox->Append("C2");
    SymmetryComboBox->Append("C3");
    SymmetryComboBox->Append("C4");
    SymmetryComboBox->Append("D2");
    SymmetryComboBox->Append("D3");
    SymmetryComboBox->Append("D4");
    SymmetryComboBox->Append("I");
    SymmetryComboBox->Append("I2");
    SymmetryComboBox->Append("O");
    SymmetryComboBox->Append("T");
    SymmetryComboBox->Append("T2");
    SymmetryComboBox->SetSelection(0);

    // Now set defaults after controls are properly initialized
    ResetDefaults( );

    GroupComboBox->AssetComboBox->Bind(wxEVT_COMMAND_COMBOBOX_SELECTED, &MatchTemplatePanel::OnGroupComboBox, this);
}

void MatchTemplatePanel::ResetAllDefaultsClick(wxCommandEvent& event) {
    ResetDefaults( );
}

/**
 * @brief Opens clicked URLs in the system's default web browser to access referenced papers
 */
void MatchTemplatePanel::OnInfoURL(wxTextUrlEvent& event) {
    const wxMouseEvent& ev = event.GetMouseEvent( );

    // filter out mouse moves, too many of them
    if ( ev.Moving( ) )
        return;

    long start = event.GetURLStart( );

    wxTextAttr my_style;

    InfoText->GetStyle(start, my_style);

    // Launch the URL

    wxLaunchDefaultBrowser(my_style.GetURL( ));
}

/**
 * @brief Resets panel to initial input state - stops running jobs, clears results, and shows input controls
 */
void MatchTemplatePanel::Reset( ) {
    ProgressBar->SetValue(0);
    TimeRemainingText->SetLabel("Time Remaining : ???h:??m:??s");
    CancelAlignmentButton->Show(true);
    FinishButton->Show(false);

    ProgressPanel->Show(false);
    StartPanel->Show(true);
    OutputTextPanel->Show(false);
    output_textctrl->Clear( );
    ResultsPanel->Show(false);
    InputPanel->Show(true);
    //graph_is_hidden = true;
    InfoPanel->Show(true);

    ResultsPanel->Clear( );

    if ( running_job == true ) {
        main_frame->job_controller.KillJob(my_job_id);
        cached_results.Clear( );

        running_job = false;

        // If this was a queue job, notify the queue manager of termination
        if ( running_queue_id > 0 && queue_manager ) {
            QM_LOG_SEARCH("Reset: Notifying queue manager of job termination for queue ID %ld", running_queue_id);
            queue_manager->OnSearchCompleted(running_queue_id, false); // false = job failed/terminated
            running_queue_id = -1;
        }

        // Reset job state - clear job ID
        my_job_id = -1;
    }

    ResetDefaults( );
    Layout( );
}

// TODO: Move all default values to a new header cistem_default_values.h at same level as constants.h
// These values should be shared between GUI (MatchTemplatePanel) and CLI (match_template.cpp)
void MatchTemplatePanel::ResetDefaults( ) {
    OutofPlaneStepNumericCtrl->ChangeValueFloat(2.5);
    InPlaneStepNumericCtrl->ChangeValueFloat(1.5);
    MinPeakRadiusNumericCtrl->ChangeValueFloat(10.0f);

    DefocusSearchYesRadio->SetValue(true);
    PixelSizeSearchNoRadio->SetValue(true);

    SymmetryComboBox->SetValue("C1");

#ifdef SHOW_CISTEM_GPU_OPTIONS
    UseGPURadioYes->SetValue(true);
#ifdef cisTEM_USING_FastFFT
    UseFastFFTRadioYes->SetValue(true);
#endif
#else
    UseGPURadioNo->SetValue(true);
    UseFastFFTRadioNo->SetValue(true);
#endif

    DefocusSearchRangeNumericCtrl->ChangeValueFloat(1200.0f);
    DefocusSearchStepNumericCtrl->ChangeValueFloat(200.0f);
    PixelSizeSearchRangeNumericCtrl->ChangeValueFloat(0.05f);
    PixelSizeSearchStepNumericCtrl->ChangeValueFloat(0.01f);

    // Set high resolution limit based on first image in selected group
    int selected_group = GroupComboBox->GetSelection( );
    MyDebugAssertTrue(selected_group >= 0, "ResetDefaults called without a valid group selection - GroupComboBox must have a selection");

    active_group.CopyFrom(&image_asset_panel->all_groups_list->groups[selected_group]);
    MyDebugAssertTrue(active_group.number_of_members > 0, "ResetDefaults called with empty group - selected group must contain at least one image");

    // Use first image asset in the group (members[0] is the array index)
    ImageAsset* current_image = image_asset_panel->ReturnAssetPointer(active_group.members[0]);
    MyDebugAssertTrue(current_image != nullptr, "Failed to get image asset pointer - asset list may be corrupted or members[0] is invalid");

    // Update tracking variable and set default resolution limit
    current_pixel_size = current_image->pixel_size;
    HighResolutionLimitNumericCtrl->ChangeValueFloat(2.0f * current_pixel_size);
}

void MatchTemplatePanel::OnGroupComboBox(wxCommandEvent& event) {
    //	ResetDefaults();
    //	AssetGroup active_group;

    // Early return if no project is open (shouldn't happen but be defensive)
    if ( ! main_frame->current_project.is_open ) {
        wxPrintf("Warning: OnGroupComboBox called but no project is open\n");
        return;
    }

    active_group.CopyFrom(&image_asset_panel->all_groups_list->groups[GroupComboBox->GetSelection( )]);

    if ( active_group.number_of_members > 0 ) {
        ImageAsset* current_image = image_asset_panel->ReturnAssetPointer(active_group.members[0]);

        // Only update high resolution limit if pixel size has changed
        if ( current_image->pixel_size != current_pixel_size ) {
            current_pixel_size = current_image->pixel_size;
            HighResolutionLimitNumericCtrl->ChangeValueFloat(2.0f * current_pixel_size);
        }
    }

    if ( GroupComboBox->GetCount( ) > 0 )
        all_images_have_defocus_values = CheckGroupHasDefocusValues( );

    if ( all_images_have_defocus_values == true && PleaseEstimateCTFStaticText->IsShown( ) == true ) {

        PleaseEstimateCTFStaticText->Show(false);
        Layout( );
    }
    else if ( all_images_have_defocus_values == false && PleaseEstimateCTFStaticText->IsShown( ) == false ) {
        PleaseEstimateCTFStaticText->Show(true);
        Layout( );
    }
}

void MatchTemplatePanel::SetInfo( ) {
    /*	#include "icons/ctffind_definitions.cpp"
	#include "icons/ctffind_diagnostic_image.cpp"
	#include "icons/ctffind_example_1dfit.cpp"

	wxLogNull *suppress_png_warnings = new wxLogNull;
	wxBitmap definitions_bmp = wxBITMAP_PNG_FROM_DATA(ctffind_definitions);
	wxBitmap diagnostic_image_bmp = wxBITMAP_PNG_FROM_DATA(ctffind_diagnostic_image);
	wxBitmap example_1dfit_bmp = wxBITMAP_PNG_FROM_DATA(ctffind_example_1dfit);
	delete suppress_png_warnings;*/

    InfoText->GetCaret( )->Hide( );

    InfoText->BeginSuppressUndo( );
    InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
    InfoText->BeginBold( );
    InfoText->BeginUnderline( );
    InfoText->BeginFontSize(14);
    InfoText->WriteText(wxT("Match Templates"));
    InfoText->EndFontSize( );
    InfoText->EndBold( );
    InfoText->EndUnderline( );
    InfoText->Newline( );
    InfoText->Newline( );
    InfoText->EndAlignment( );

    InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
    InfoText->WriteText(wxT("TODO: update this with the 2020 paper (finding things in cells) combined with things for bigger regions (Johannes' paper) or baited recon (newer paper)."));
    InfoText->Newline( );
    InfoText->Newline( );
    InfoText->EndAlignment( );

    InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
    InfoText->BeginBold( );
    InfoText->BeginUnderline( );
    InfoText->WriteText(wxT("Program Options"));
    InfoText->EndBold( );
    InfoText->EndUnderline( );
    InfoText->Newline( );
    InfoText->Newline( );
    InfoText->EndAlignment( );

    InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
    InfoText->BeginBold( );
    InfoText->WriteText(wxT("Input Group : "));
    InfoText->EndBold( );
    InfoText->WriteText(wxT("The group of image assets to look for templates in"));
    InfoText->Newline( );
    InfoText->BeginBold( );
    InfoText->WriteText(wxT("Reference Volume : "));
    InfoText->EndBold( );
    InfoText->WriteText(wxT("The volume that will used for the template search TODO: add a description of how to generate a reference, and padding/sizing considerations."));
    InfoText->Newline( );
    InfoText->BeginBold( );
    InfoText->WriteText(wxT("Run Profile : "));
    InfoText->EndBold( );
    InfoText->WriteText(wxT("TODO: reference threading vs process balance. The selected run profile will be used to run the job. The run profile describes how the job should be run (e.g. how many processors should be used, and on which different computers).  Run profiles are set in the Run Profile panel, located under settings."));
    InfoText->Newline( );
    InfoText->Newline( );
    InfoText->EndAlignment( );

    InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
    InfoText->BeginBold( );
    InfoText->BeginUnderline( );
    InfoText->WriteText(wxT("Expert Options"));
    InfoText->EndBold( );
    InfoText->EndUnderline( );
    InfoText->Newline( );
    InfoText->Newline( );
    InfoText->EndAlignment( );

    // TODO: add section on scaling based on pixel size. Note that the input is a target that may slightly differ. Note limits on sizing and power of two (ref to fast FFT bit)
    InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
    InfoText->BeginBold( );
    InfoText->WriteText(wxT("Out of Plane Angular Step : "));
    InfoText->EndBold( );
    InfoText->WriteText(wxT("The angular step that should be used for the out of plane search.  Smaller values may increase accuracy, but will significantly increase the required processing time."));
    InfoText->Newline( );
    InfoText->BeginBold( );
    InfoText->WriteText(wxT("In Plane Angular Step : "));
    InfoText->EndBold( );
    InfoText->WriteText(wxT("The angular step that should be used for the in plane search.  As with the out of plane angle, smaller values may increase accuracy, but will significantly increase the required processing time."));
    InfoText->Newline( );
    InfoText->Newline( );
    InfoText->EndAlignment( );
    InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
    InfoText->BeginBold( );
    InfoText->BeginUnderline( );
    InfoText->WriteText(wxT("References"));
    InfoText->EndBold( );
    InfoText->EndUnderline( );
    InfoText->Newline( );
    InfoText->Newline( );
    InfoText->EndAlignment( );

    InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
    InfoText->BeginBold( );
    InfoText->WriteText(wxT("Rickgauer J.P., Grigorieff N., Denk W."));
    InfoText->EndBold( );
    InfoText->WriteText(wxT(" 2017. Single-protein detection in crowded molecular environments in cryo-EM images. Elife 6, e25648.. "));
    InfoText->BeginURL("http://doi.org/10.7554/eLife.25648");
    InfoText->BeginUnderline( );
    InfoText->BeginTextColour(*wxBLUE);
    InfoText->WriteText(wxT("doi:10.7554/eLife.25648"));
    InfoText->EndURL( );
    InfoText->EndTextColour( );
    InfoText->EndUnderline( );
    InfoText->Newline( );
    InfoText->Newline( );

    InfoText->EndSuppressUndo( );
}

void MatchTemplatePanel::FillGroupComboBox( ) {
    // Called from constructor (when panel is created) or OnUpdateUI (when groups change)
    // Return early if no project is open (can happen during workflow switching)
    if ( ! main_frame->current_project.is_open ) {
        return;
    }

    GroupComboBox->FillComboBox(true);

    if ( GroupComboBox->GetCount( ) > 0 )
        all_images_have_defocus_values = CheckGroupHasDefocusValues( );

    // Show warning if images lack defocus values, hide if they have them
    bool should_show_warning = ! all_images_have_defocus_values;
    if ( PleaseEstimateCTFStaticText->IsShown( ) != should_show_warning ) {
        PleaseEstimateCTFStaticText->Show(should_show_warning);
        Layout( );
    }
}

void MatchTemplatePanel::FillRunProfileComboBox( ) {
    RunProfileComboBox->FillWithRunProfiles( );
}

bool MatchTemplatePanel::CheckGroupHasDefocusValues( ) {
    // TODO: Future optimization - instead of fetching ALL images with CTF estimates and searching,
    // build a comma-separated list of just this group's asset IDs and use a single SQL query:
    // SELECT COUNT(DISTINCT IMAGE_ASSET_ID) FROM ESTIMATED_CTF_PARAMETERS WHERE IMAGE_ASSET_ID IN (id1,id2,...)
    // This would reduce complexity from O(n*m) to O(1) database operation

    // Use modern vector instead of wxArrayLong for better performance
    std::vector<long> images_with_defocus_values;
    main_frame->current_project.database.FillVectorFromSelectCommand(
            "SELECT DISTINCT IMAGE_ASSET_ID FROM ESTIMATED_CTF_PARAMETERS",
            images_with_defocus_values);

    for ( int image_in_group_counter = 0; image_in_group_counter < image_asset_panel->ReturnGroupSize(GroupComboBox->GetSelection( )); image_in_group_counter++ ) {
        long current_image_id = image_asset_panel->all_assets_list->ReturnAssetPointer(
                                                                          image_asset_panel->ReturnGroupMember(GroupComboBox->GetSelection( ), image_in_group_counter))
                                        ->asset_id;

        // Check if this image has a CTF estimate by searching for its ID in the vector
        // std::find returns an iterator to the found element, or end() if not found
        // This replaces the manual loop search with an optimized STL algorithm
        if ( std::find(images_with_defocus_values.begin( ), images_with_defocus_values.end( ), current_image_id) == images_with_defocus_values.end( ) ) {
            return false; // This image lacks CTF estimates, so group is incomplete
        }
    }

    return true;
}

void MatchTemplatePanel::OnUpdateUI(wxUpdateUIEvent& event) {

    // are there enough members in the selected group.
    if ( main_frame->current_project.is_open == false ) {
        RunProfileComboBox->Enable(false);
        GroupComboBox->Enable(false);
        StartEstimationButton->Enable(false);
        ReferenceSelectPanel->Enable(false);
    }
    else {
        if ( running_job == false ) {
            RunProfileComboBox->Enable(true);
            GroupComboBox->Enable(true);
            ReferenceSelectPanel->Enable(true);

            if ( RunProfileComboBox->GetCount( ) > 0 ) {
                if ( image_asset_panel->ReturnGroupSize(GroupComboBox->GetSelection( )) > 0 && run_profiles_panel->run_profile_manager.ReturnTotalJobs(RunProfileComboBox->GetSelection( )) > 0 && all_images_have_defocus_values == true ) {
                    StartEstimationButton->Enable(true);
                }
                else
                    StartEstimationButton->Enable(false);
            }
            else {
                StartEstimationButton->Enable(false);
            }

            if ( DefocusSearchYesRadio->GetValue( ) == true ) {
                DefocusRangeStaticText->Enable(true);
                DefocusSearchRangeNumericCtrl->Enable(true);
                DefocusStepStaticText->Enable(true);
                DefocusSearchStepNumericCtrl->Enable(true);
            }
            else {
                DefocusRangeStaticText->Enable(false);
                DefocusSearchRangeNumericCtrl->Enable(false);
                DefocusStepStaticText->Enable(false);
                DefocusSearchStepNumericCtrl->Enable(false);
            }

            if ( PixelSizeSearchYesRadio->GetValue( ) == true ) {
                PixelSizeRangeStaticText->Enable(true);
                PixelSizeSearchRangeNumericCtrl->Enable(true);
                PixelSizeStepStaticText->Enable(true);
                PixelSizeSearchStepNumericCtrl->Enable(true);
            }
            else {
                PixelSizeRangeStaticText->Enable(false);
                PixelSizeSearchRangeNumericCtrl->Enable(false);
                PixelSizeStepStaticText->Enable(false);
                PixelSizeSearchStepNumericCtrl->Enable(false);
            }
        }
        else {
            GroupComboBox->Enable(false);
            ReferenceSelectPanel->Enable(false);
            RunProfileComboBox->Enable(false);
        }

        if ( group_combo_is_dirty == true ) {
            FillGroupComboBox( );
            group_combo_is_dirty = false;
        }

        if ( run_profiles_are_dirty == true ) {
            FillRunProfileComboBox( );
            run_profiles_are_dirty = false;
        }

        if ( volumes_are_dirty == true ) {
            ReferenceSelectPanel->FillComboBox( );
            volumes_are_dirty = false;
        }
    }
}

void MatchTemplatePanel::StartEstimationClick(wxCommandEvent& event) {
    MyDebugAssertTrue(main_frame != nullptr, "main_frame is null");
    MyDebugAssertTrue(main_frame->current_project.is_open, "Project database is not open");

    // Add to queue silently (without dialog), then execute
    TemplateMatchQueueItem new_job = CollectJobParametersFromGui( );
    AddJobToQueue(new_job, false); // Sets new_job.database_queue_id in place

    // Block auto-progression - user wants to run ONE search only
    block_auto_progression_of_queue = true;

    MyAssertTrue(new_job.database_queue_id > 0, "Queue ID is <= 0");

    // Execute the search via unified method
    // (QUEUE_POSITION will be updated to -1 in ExecuteCurrentSearch after job controller starts)
    if ( ! ExecuteSearch(&new_job) ) {
        wxMessageBox("Failed to start search", "Error", wxOK | wxICON_ERROR);
        return;
    }
}

void MatchTemplatePanel::HandleSocketTemplateMatchResultReady(wxSocketBase* connected_socket, int& image_number, float& threshold_used, ArrayOfTemplateMatchFoundPeakInfos& peak_infos, ArrayOfTemplateMatchFoundPeakInfos& peak_changes) {
    // result is available for an image..

    // Validate image_number is within bounds
    if ( image_number < 1 || image_number > cached_results.GetCount( ) ) {
        wxPrintf("ERROR: Invalid image_number %d received from socket (cached_results size: %d)\n",
                 image_number, int(cached_results.GetCount( )));
        MyDebugAssertFalse(image_number >= 1 && image_number <= cached_results.GetCount( ),
                           "Received invalid image_number from socket - data corruption?");
        return;
    }

    cached_results[image_number - 1].found_peaks.Clear( );
    cached_results[image_number - 1].found_peaks    = peak_infos;
    cached_results[image_number - 1].used_threshold = threshold_used;

    ResultsPanel->SetActiveResult(cached_results[image_number - 1]);

    // write to database..

    // CRITICAL: If this is a queue item with no search_id yet, assign one NOW (first result being written)
    if ( running_queue_id > 0 && search_id == -1 ) {
        // First result for this queue item - assign a new search_id
        // The ONLY source of truth for search_id is TEMPLATE_MATCH_LIST
        search_id = main_frame->current_project.database.ReturnHighestTemplateMatchJobID( ) + 1;

        QM_LOG_DB("First result for queue item %ld - assigning search_id %d", running_queue_id, search_id);

        // Update the queue item with the new search_id
        if ( queue_manager ) {
            // Queue manager is present - let it update the search ID
            queue_manager->UpdateSearchIdForQueueItem(running_queue_id, search_id);
        }
        else {
            // No queue manager (StartEstimationClick path) - update database directly
            main_frame->current_project.database.UpdateSearchIdInQueueTable(running_queue_id, search_id);
            QM_LOG_DB("Direct database update: Linked queue ID %ld to search ID %d", running_queue_id, search_id);
        }

// Note: We can't check for orphaned search_ids here because we're about to write
// the first result. The check would need to happen AFTER this result is written.
#ifdef cisTEM_DEBUG_ORPHANED_SEARCH_IDS
// This check would fail here - we've assigned the search_id but haven't written results yet
// See check after result is written to database instead
#endif
    }

    MyDebugAssertTrue(search_id > 0, "Attempting to write result with invalid search_id %d", search_id);

    main_frame->current_project.database.Begin( );

    // NOTE: search_id field in structure maps to SEARCH_ID column in database
    // SEARCH_NAME now stored in TEMPLATE_MATCH_QUEUE table only (normalized schema)
    cached_results[image_number - 1].search_id = search_id;

    // Get next available TEMPLATE_MATCH_ID from database immediately before writing
    // This ensures correct ID assignment even when resuming cancelled searches
    int next_template_match_id = main_frame->current_project.database.ReturnHighestTemplateMatchID( ) + 1;

    // Fix OUTPUT_FILENAME_BASE to be just path + image filename (without IDs or suffixes)
    // The helper functions will append _<type>_<template_match_id>_<search_id>.mrc
    ImageAsset* current_image                             = image_asset_panel->ReturnAssetPointer(image_asset_panel->ReturnArrayPositionFromAssetID(cached_results[image_number - 1].image_asset_id));
    cached_results[image_number - 1].output_filename_base = main_frame->current_project.template_matching_asset_directory.GetFullPath( ) +
                                                            "/" + current_image->filename.GetName( );

    // Store the template_match_id so helper functions can use it
    cached_results[image_number - 1].template_match_id = next_template_match_id;

    main_frame->current_project.database.AddTemplateMatchingResult(next_template_match_id, cached_results[image_number - 1]);

    main_frame->current_project.database.SetActiveTemplateMatchJobForGivenImageAssetID(cached_results[image_number - 1].image_asset_id, search_id);
    main_frame->current_project.database.Commit( );
    match_template_results_panel->is_dirty = true;

    // Notify queue manager about result addition for real-time n/N updates
    if ( queue_manager ) {
        queue_manager->OnResultAdded(search_id);
    }

#ifdef cisTEM_DEBUG_ORPHANED_SEARCH_IDS
    // After writing a result, verify consistency between queue and results tables
    static int results_written = 0;
    results_written++;
    // Only check periodically to avoid performance impact
    if ( results_written % 10 == 0 ) {
        int highest_queue_search_id   = main_frame->current_project.database.GetHighestSearchIdFromQueue( );
        int highest_results_search_id = main_frame->current_project.database.ReturnHighestTemplateMatchJobID( );
        MyDebugAssertTrue(highest_queue_search_id <= highest_results_search_id,
                          "Orphaned search_ids detected: Queue table has max search_id %d but TEMPLATE_MATCH_LIST has max %d",
                          highest_queue_search_id, highest_results_search_id);
    }
#endif

    // Mark job as finished in tracker (matches FindCTFPanel.cpp:1016 pattern)
    my_job_tracker.MarkJobFinished( );
    if ( my_job_tracker.ShouldUpdate( ) == true )
        UpdateProgressBar( );
}

void MatchTemplatePanel::FinishButtonClick(wxCommandEvent& event) {
    ProgressBar->SetValue(0);
    TimeRemainingText->SetLabel("Time Remaining : ???h:??m:??s");
    CancelAlignmentButton->Show(true);
    FinishButton->Show(false);

    ProgressPanel->Show(false);
    StartPanel->Show(true);
    OutputTextPanel->Show(false);
    output_textctrl->Clear( );
    ResultsPanel->Show(false);
    //graph_is_hidden = true;
    InfoPanel->Show(true);
    InputPanel->Show(true);

    ExpertPanel->Show(true);

    running_job = false;
    Layout( );
}

void MatchTemplatePanel::TerminateButtonClick(wxCommandEvent& event) {
    // kill the job, this will kill the socket to terminate downstream processes
    // - this will have to be improved when clever network failure is incorporated

    main_frame->job_controller.KillJob(my_job_id);

    WriteInfoText("Terminated Job");
    TimeRemainingText->SetLabel("Time Remaining : Terminated");
    CancelAlignmentButton->Show(false);
    FinishButton->Show(true);
    ProgressPanel->Layout( );
    cached_results.Clear( );

    // If this was a queue job, notify the queue manager of termination
    if ( running_queue_id > 0 && queue_manager ) {
        QM_LOG_SEARCH("Notifying queue manager of job termination for queue ID %ld", running_queue_id);
        queue_manager->OnSearchCompleted(running_queue_id, false); // false = job failed/terminated
        running_queue_id = -1;
    }

    // Reset job state - clear job ID
    running_job = false;
    my_job_id   = -1;
}

void MatchTemplatePanel::WriteInfoText(wxString text_to_write) {
    output_textctrl->SetDefaultStyle(wxTextAttr(*wxBLACK));
    output_textctrl->AppendText(text_to_write);

    if ( text_to_write.EndsWith("\n") == false )
        output_textctrl->AppendText("\n");
}

void MatchTemplatePanel::WriteErrorText(wxString text_to_write) {
    output_textctrl->SetDefaultStyle(wxTextAttr(*wxRED));
    output_textctrl->AppendText(text_to_write);

    if ( text_to_write.EndsWith("\n") == false )
        output_textctrl->AppendText("\n");
}

void MatchTemplatePanel::OnSocketJobResultMsg(JobResult& received_result) {
    if ( received_result.result_size > 0 ) {
        ProcessResult(&received_result);
    }
}

void MatchTemplatePanel::OnSocketJobResultQueueMsg(ArrayofJobResults& received_queue) {
    for ( int counter = 0; counter < received_queue.GetCount( ); counter++ ) {
        ProcessResult(&received_queue.Item(counter));
    }
}

void MatchTemplatePanel::SetNumberConnectedText(wxString wanted_text) {
    NumberConnectedText->SetLabel(wanted_text);
}

void MatchTemplatePanel::SetTimeRemainingText(wxString wanted_text) {
    TimeRemainingText->SetLabel(wanted_text);
}

void MatchTemplatePanel::OnSocketAllJobsFinished( ) {
    ProcessAllJobsFinished( );
}

void MatchTemplatePanel::ProcessResult(JobResult* result_to_process) // this will have to be overidden in the parent clas when i make it.
{

    long     current_time = time(NULL);
    wxString bitmap_string;
    wxString plot_string;

    number_of_received_results++;

    if ( number_of_received_results == 1 ) {
        current_job_starttime = current_time;
        time_of_last_update   = 0;
    }
    else if ( current_time != time_of_last_update ) {
        int current_percentage;
        current_percentage = myroundint(float(number_of_received_results) / float(expected_number_of_results) * 100.0f);

        time_of_last_update = current_time;
        if ( current_percentage > 100 )
            current_percentage = 100;
        ProgressBar->SetValue(current_percentage);

        long  job_time        = current_time - current_job_starttime;
        float seconds_per_job = float(job_time) / float(number_of_received_results - 1);

        long seconds_remaining;
        seconds_remaining = float(expected_number_of_results - number_of_received_results) * seconds_per_job;

        wxTimeSpan time_remaining = wxTimeSpan(0, 0, seconds_remaining);
        TimeRemainingText->SetLabel(time_remaining.Format("Time Remaining : %Hh:%Mm:%Ss"));

        // Update queue manager display periodically to show n/N progress
        if ( running_queue_id > 0 && queue_manager ) {
            queue_manager->UpdateQueueDisplay( );
        }
    }
}

void MatchTemplatePanel::ProcessAllJobsFinished( ) {

    MyDebugAssertTrue(my_job_tracker.total_number_of_finished_jobs == my_job_tracker.total_number_of_jobs, "In ProcessAllJobsFinished, but total_number_of_finished_jobs != total_number_of_jobs. Oops.");

    // Notify queue manager that job is entering finalization phase
    // This prevents auto-advance from starting a new job while we're cleaning up
    if ( running_queue_id > 0 && queue_manager ) {
        QM_LOG_STATE("Notifying queue manager that search with queue ID %ld is entering finalization", running_queue_id);
        queue_manager->OnSearchEnteringFinalization(running_queue_id);
    }
    else {
        QM_LOG_STATE("Not entering finalization in ProcessAllJosFinished");
    }

    // Update the GUI with project timings
    extern MyOverviewPanel* overview_panel;
    overview_panel->SetProjectInfo( );

    //
    WriteResultToDataBase( );
    match_template_results_panel->is_dirty = true;

    // let the FindParticles panel check whether any of the groups are now ready to be picked
    //extern MyFindParticlesPanel *findparticles_panel;
    //findparticles_panel->CheckWhetherGroupsCanBePicked();

    cached_results.Clear( );

    // Kill the job - sends socket_time_to_die to controller/master
    main_frame->job_controller.KillJob(my_job_id);

    // Reset job state to allow next job to start - MUST happen before callback
    running_job             = false;
    my_job_id               = -1;
    current_custom_cli_args = ""; // Clear custom CLI args after job completion

    // For queue jobs: trigger auto-advance immediately (matches pattern from other panels)
    // Other panels (FindCTF, AutoRefine3d) call KillJob() then immediately continue
    if ( running_queue_id > 0 && queue_manager ) {
        long completed_queue_id = running_queue_id;
        running_queue_id        = -1; // Clear before triggering callback

        QM_LOG_SEARCH("Search with queue ID %ld completed - calling OnSearchCompleted", completed_queue_id);
        queue_manager->OnSearchCompleted(completed_queue_id, true);
    }
    else if ( running_queue_id > 0 ) {
        QM_LOG_ERROR("Queue job completed but no queue_manager registered!");
        running_queue_id = -1;
    }

    WriteInfoText("All Jobs have finished.");
    ProgressBar->SetValue(100);

    // Decide whether to continue queue or show Finish button
    // Based on block_auto_progression_of_queue flag set by launch source
    if ( block_auto_progression_of_queue ) {
        // Launched from Start button - show Finish button, no auto-advance
        block_auto_progression_of_queue = false; // Reset for next execution
        TimeRemainingText->SetLabel("Time Remaining : All Done!");
        CancelAlignmentButton->Show(false);
        FinishButton->Show(true);
    }
    else {
        // Launched from Queue Manager - check if more items to process
        bool has_active_queue_items = false;
        if ( queue_manager ) {
            has_active_queue_items = queue_manager->ExecutionQueueHasActiveItems( );
        }

        if ( has_active_queue_items ) {
            // Don't show finish button yet - more queue items to process
            TimeRemainingText->SetLabel("Time Remaining : Waiting for next queue item...");
            CancelAlignmentButton->Show(false);
            FinishButton->Show(false);

            // Safety net: 30-second timeout to show finish button if queue progression fails
            long job_id_at_timeout_start = my_job_id;

            wxTimer* timeout_timer = new wxTimer( );
            timeout_timer->Bind(wxEVT_TIMER, [this, timeout_timer, job_id_at_timeout_start](wxTimerEvent&) {
                if ( my_job_id == job_id_at_timeout_start && running_job == false &&
                     FinishButton && ! FinishButton->IsShown( ) ) {
                    QM_LOG_ERROR("Queue auto-advance timeout after 30 seconds - showing Finish button as safety net");
                    TimeRemainingText->SetLabel("Time Remaining : Queue progression stalled");
                    FinishButton->Show(true);
                    ProgressPanel->Layout( );
                }
                delete timeout_timer;
            });
            timeout_timer->StartOnce(30000);
        }
        else {
            // All done - show finish button
            TimeRemainingText->SetLabel("Time Remaining : All Done!");
            CancelAlignmentButton->Show(false);
            FinishButton->Show(true);
        }
    }
    ProgressPanel->Layout( );
}

void MatchTemplatePanel::WriteResultToDataBase( ) {
    // Results are written one at a time in HandleSocketTemplateMatchResultReady as they arrive.
    // This method is no longer used but kept for reference.
    /*
	// Legacy batch write code - no longer used
	// Results are now written incrementally in HandleSocketTemplateMatchResultReady
*/
}

void MatchTemplatePanel::UpdateProgressBar( ) {
    ProgressBar->SetValue(my_job_tracker.ReturnPercentCompleted( ));
    TimeRemainingText->SetLabel(my_job_tracker.ReturnRemainingTime( ).Format("Time Remaining : %Hh:%Mm:%Ss"));
}

// Queue functionality implementation
void MatchTemplatePanel::OnAddToQueueClick(wxCommandEvent& event) {
    // Validate that no job is currently running
    if ( running_job == false ) {
        // Collect job parameters from GUI and add to queue with dialog
        TemplateMatchQueueItem new_job = CollectJobParametersFromGui( );
        AddJobToQueue(new_job, true);
    }
    else {
        wxMessageBox("A job is currently running. Please wait for it to complete before queuing new jobs.",
                     "Job Running", wxOK | wxICON_WARNING);
    }
}

void MatchTemplatePanel::OnOpenQueueClick(wxCommandEvent& event) {
    if ( running_job == false ) {
        // Reload fresh data from database
        queue_manager->LoadQueueFromDatabase( );

        // Create dialog for queue management UI
        wxDialog* queue_dialog = new wxDialog(this, wxID_ANY, "Template Match Queue Manager",
                                              wxDefaultPosition, wxSize(720, 700),
                                              wxDEFAULT_DIALOG_STYLE | wxRESIZE_BORDER);

        // Reparent queue manager to dialog
        queue_manager->Reparent(queue_dialog);
        queue_manager->Show(true);

        // Layout
        wxBoxSizer* dialog_sizer = new wxBoxSizer(wxVERTICAL);
        dialog_sizer->Add(queue_manager, 1, wxEXPAND | wxALL, 5);
        queue_dialog->SetSizer(dialog_sizer);
        queue_dialog->Layout( );
        queue_dialog->Fit( );

        // Handle window close event
        queue_dialog->Bind(wxEVT_CLOSE_WINDOW, [this, queue_dialog](wxCloseEvent& event) {
            // Check if a job is running and ensure the panel shows the correct state
            if ( running_job == true ) {
                // Ensure we're showing the progress panel and controls
                if ( ! ProgressPanel->IsShown( ) ) {
                    ProgressPanel->Show(true);
                    StartPanel->Show(false);
                    OutputTextPanel->Show(true);
                    InfoPanel->Show(false);
                    InputPanel->Show(false);
                    Layout( );
                }
                // Ensure the terminate button is visible
                if ( ! CancelAlignmentButton->IsShown( ) ) {
                    CancelAlignmentButton->Show(true);
                    FinishButton->Show(false);
                    ProgressPanel->Layout( );
                }
            }

            // Reparent back to panel and hide
            queue_manager->Reparent(this);
            queue_manager->Hide( );
            queue_dialog->Destroy( );
        });

        queue_dialog->Show(true);
    }
    else {
        wxMessageBox("A job is currently running. Please wait for it to complete before opening the queue manager.",
                     "Job Running", wxOK | wxICON_WARNING);
    }
}

void MatchTemplatePanel::PopulateGuiFromQueueItem(const TemplateMatchQueueItem& item, bool for_editing) {
    // Populate GUI controls with values from the queue item

    // Store custom CLI args for later use during execution
    current_custom_cli_args = item.custom_cli_args;

    // Set the group and reference selections
    if ( GroupComboBox && image_asset_panel && image_asset_panel->all_groups_list ) {
        // Find the group with matching database ID and set ComboBox to that index
        for ( int group_index = 0; group_index < image_asset_panel->all_groups_list->number_of_groups; group_index++ ) {
            if ( image_asset_panel->all_groups_list->groups[group_index].id == item.image_group_id ) {
                GroupComboBox->SetSelection(group_index);
                break;
            }
        }
        // Enable combo box if we're editing
        if ( for_editing && running_job == false ) {
            GroupComboBox->Enable(true);
        }
    }

    if ( ReferenceSelectPanel && item.reference_volume_asset_id >= 0 && volume_asset_panel ) {
        int array_position = volume_asset_panel->ReturnArrayPositionFromAssetID(item.reference_volume_asset_id);
        if ( array_position >= 0 ) {
            ReferenceSelectPanel->SetSelection(array_position);
        }
        // Enable reference panel if we're editing
        if ( for_editing && running_job == false ) {
            ReferenceSelectPanel->Enable(true);
        }
    }

    // Set run profile by finding matching profile id
    if ( RunProfileComboBox && item.run_profile_id >= 0 && run_profiles_panel ) {
        for ( int profile_index = 0; profile_index < run_profiles_panel->run_profile_manager.number_of_run_profiles; profile_index++ ) {
            if ( run_profiles_panel->run_profile_manager.run_profiles[profile_index].id == item.run_profile_id ) {
                RunProfileComboBox->SetSelection(profile_index);
                QM_LOG_UI("Set RunProfileComboBox to selection %d (profile id=%d) from queue item", profile_index, item.run_profile_id);
                break;
            }
        }
        // Enable combo box if we're editing
        if ( for_editing && running_job == false ) {
            RunProfileComboBox->Enable(true);
        }
    }

    // Set symmetry
    if ( SymmetryComboBox ) {
        SymmetryComboBox->SetValue(item.symmetry);
    }

    // Set resolution limits
    if ( HighResolutionLimitNumericCtrl ) {
        HighResolutionLimitNumericCtrl->SetValue(wxString::Format("%.2f", item.high_resolution_limit));
    }

    // Set angular steps
    if ( OutofPlaneStepNumericCtrl ) {
        OutofPlaneStepNumericCtrl->SetValue(wxString::Format("%.2f", item.out_of_plane_angular_step));
    }

    if ( InPlaneStepNumericCtrl ) {
        InPlaneStepNumericCtrl->SetValue(wxString::Format("%.2f", item.in_plane_angular_step));
    }

    // Set defocus search parameters
    if ( item.defocus_search_range > 0 ) {
        if ( DefocusSearchYesRadio )
            DefocusSearchYesRadio->SetValue(true);
        if ( DefocusSearchNoRadio )
            DefocusSearchNoRadio->SetValue(false);
        if ( DefocusSearchRangeNumericCtrl ) {
            DefocusSearchRangeNumericCtrl->SetValue(wxString::Format("%.2f", item.defocus_search_range));
            DefocusSearchRangeNumericCtrl->Enable(true);
        }
        if ( DefocusSearchStepNumericCtrl ) {
            DefocusSearchStepNumericCtrl->SetValue(wxString::Format("%.2f", item.defocus_step));
            DefocusSearchStepNumericCtrl->Enable(true);
        }
    }
    else {
        if ( DefocusSearchYesRadio )
            DefocusSearchYesRadio->SetValue(false);
        if ( DefocusSearchNoRadio )
            DefocusSearchNoRadio->SetValue(true);
        if ( DefocusSearchRangeNumericCtrl )
            DefocusSearchRangeNumericCtrl->Enable(false);
        if ( DefocusSearchStepNumericCtrl )
            DefocusSearchStepNumericCtrl->Enable(false);
    }

    // Set pixel size search parameters
    if ( item.pixel_size_search_range > 0 ) {
        if ( PixelSizeSearchYesRadio )
            PixelSizeSearchYesRadio->SetValue(true);
        if ( PixelSizeSearchNoRadio )
            PixelSizeSearchNoRadio->SetValue(false);
        if ( PixelSizeSearchRangeNumericCtrl ) {
            PixelSizeSearchRangeNumericCtrl->SetValue(wxString::Format("%.4f", item.pixel_size_search_range));
            PixelSizeSearchRangeNumericCtrl->Enable(true);
        }
        if ( PixelSizeSearchStepNumericCtrl ) {
            PixelSizeSearchStepNumericCtrl->SetValue(wxString::Format("%.4f", item.pixel_size_step));
            PixelSizeSearchStepNumericCtrl->Enable(true);
        }
    }
    else {
        if ( PixelSizeSearchYesRadio )
            PixelSizeSearchYesRadio->SetValue(false);
        if ( PixelSizeSearchNoRadio )
            PixelSizeSearchNoRadio->SetValue(true);
        if ( PixelSizeSearchRangeNumericCtrl )
            PixelSizeSearchRangeNumericCtrl->Enable(false);
        if ( PixelSizeSearchStepNumericCtrl )
            PixelSizeSearchStepNumericCtrl->Enable(false);
    }

    // Set peak radius
    if ( MinPeakRadiusNumericCtrl ) {
        MinPeakRadiusNumericCtrl->SetValue(wxString::Format("%.2f", item.min_peak_radius));
    }

    // Refresh the GUI
    Update( );
    Refresh( );
}

bool MatchTemplatePanel::RunQueuedTemplateMatch(TemplateMatchQueueItem& job) {
    // Check if we're ready to run
    if ( running_job == true ) {
        QM_LOG_ERROR("Cannot run queued job - another job is already running");
        return false;
    }

    // Note: running_queue_id will be set by SetupSearchBatchFromQueueItem after confirming there's work to do

    // Populate the GUI with the queued job's parameters
    PopulateGuiFromQueueItem(job);

    // Give the GUI a chance to update
    wxYield( );

    // Trigger the template matching execution
    wxCommandEvent fake_event;
    StartEstimationClick(fake_event);

    return true;
}

TemplateMatchQueueItem MatchTemplatePanel::CollectJobParametersFromGui( ) {
    TemplateMatchQueueItem new_job;

    // Clear the member variable when collecting from GUI (not from queue)
    // (queue_status, custom_cli_args, database_queue_id all initialized by constructor)
    current_custom_cli_args = "";

    // Collect actual parameters from GUI controls
    // Use database IDs (not combo box indices) for all assets
    // groups[0].id = -1 ("All Images" virtual group), groups[N].id = database GROUP_ID
    new_job.image_group_id            = image_asset_panel->all_groups_list->groups[GroupComboBox->GetSelection( )].id;
    new_job.reference_volume_asset_id = volume_asset_panel->ReturnAssetPointer(ReferenceSelectPanel->GetSelection( ))->asset_id;
    new_job.run_profile_id            = run_profiles_panel->run_profile_manager.run_profiles[RunProfileComboBox->GetSelection( )].id;

    // Get symmetry and resolution parameters
    new_job.symmetry              = SymmetryComboBox->GetValue( ).Upper( );
    new_job.high_resolution_limit = HighResolutionLimitNumericCtrl->ReturnValue( );
    new_job.low_resolution_limit  = 300.0f; // Currently hardcoded in template matching FIXME: -> constants

    // Angular search parameters
    new_job.out_of_plane_angular_step = OutofPlaneStepNumericCtrl->ReturnValue( );
    new_job.in_plane_angular_step     = InPlaneStepNumericCtrl->ReturnValue( );

    // Defocus search parameters
    if ( DefocusSearchYesRadio->GetValue( ) ) {
        new_job.defocus_search_range = DefocusSearchRangeNumericCtrl->ReturnValue( );
        new_job.defocus_step         = DefocusSearchStepNumericCtrl->ReturnValue( );
    }
    else {
        new_job.defocus_search_range = 0.0f;
        new_job.defocus_step         = 0.0f;
    }

    // Pixel size search parameters
    if ( PixelSizeSearchYesRadio->GetValue( ) ) {
        new_job.pixel_size_search_range = PixelSizeSearchRangeNumericCtrl->ReturnValue( );
        new_job.pixel_size_step         = PixelSizeSearchStepNumericCtrl->ReturnValue( );
    }
    else {
        new_job.pixel_size_search_range = 0.0f;
        new_job.pixel_size_step         = 0.0f;
    }

    // Peak detection parameters
    new_job.min_peak_radius = MinPeakRadiusNumericCtrl->ReturnValue( );

    // Get volume parameters and generate search name
    VolumeAsset* current_volume = volume_asset_panel->ReturnAssetPointer(ReferenceSelectPanel->GetSelection( ));
    if ( current_volume ) {
        new_job.search_name               = wxString::Format("Template: %s", current_volume->filename.GetName( ));
        new_job.ref_box_size_in_angstroms = current_volume->x_size * current_volume->pixel_size;
        new_job.mask_radius               = current_volume->x_size * current_volume->pixel_size / 2.0f;
    }
    else {
        // Fallback if no volume is selected (shouldn't happen in normal use)
        wxDateTime now                    = wxDateTime::Now( );
        new_job.search_name               = wxString::Format("TM_%s", now.Format("%Y%m%d_%H%M%S"));
        new_job.ref_box_size_in_angstroms = 200.0;
        new_job.mask_radius               = 80.0;
    }

    // Get GPU and FastFFT settings from GUI controls
    new_job.use_gpu      = UseGPURadioYes->GetValue( );
    new_job.use_fast_fft = UseFastFFTRadioYes->GetValue( );

    // Additional parameters (currently not in GUI, using defaults)
    new_job.refinement_threshold       = 0.0;
    new_job.xy_change_threshold        = 0.0;
    new_job.exclude_above_xy_threshold = false;

    // Debug print for run profile information
    QM_LOG_DEBUG("=== DEBUG: Run Profile Information ===");
    QM_LOG_DEBUG("Run Profile ID (stored in queue): %d", new_job.run_profile_id);
    if ( new_job.run_profile_id >= 0 && new_job.run_profile_id < RunProfileComboBox->GetCount( ) ) {
        wxString run_profile_name = RunProfileComboBox->GetString(new_job.run_profile_id);
        QM_LOG_DEBUG("Run Profile Name: %s", run_profile_name);
        QM_LOG_DEBUG("Run Profile ComboBox Count: %d", RunProfileComboBox->GetCount( ));
    }
    else {
        QM_LOG_DEBUG("No run profile selected or invalid selection");
        QM_LOG_DEBUG("Run Profile ComboBox Count: %d", RunProfileComboBox->GetCount( ));
    }
    QM_LOG_DEBUG("=== END DEBUG ===");

    return new_job;
}

long MatchTemplatePanel::AddJobToQueue(TemplateMatchQueueItem& job, bool show_dialog) {
    // Reload fresh data from database
    queue_manager->LoadQueueFromDatabase( );

    // Add to queue (single source of truth for database operations)
    long queue_id = queue_manager->AddToExecutionQueue(job);

    // Update job with database queue ID (modified in place for caller)
    job.database_queue_id = queue_id;

    if ( show_dialog ) {
        // Create dialog for queue management UI
        wxDialog* queue_dialog = new wxDialog(this, wxID_ANY, "Template Match Queue Manager",
                                              wxDefaultPosition, wxSize(720, 700),
                                              wxDEFAULT_DIALOG_STYLE | wxRESIZE_BORDER);

        // Reparent queue manager to dialog
        queue_manager->Reparent(queue_dialog);
        queue_manager->Show(true);

        // Layout
        wxBoxSizer* dialog_sizer = new wxBoxSizer(wxVERTICAL);
        dialog_sizer->Add(queue_manager, 1, wxEXPAND | wxALL, 5);
        queue_dialog->SetSizer(dialog_sizer);
        queue_dialog->Layout( );
        queue_dialog->Fit( );

        // Handle window close event
        queue_dialog->Bind(wxEVT_CLOSE_WINDOW, [this, queue_dialog](wxCloseEvent& event) {
            // Check if a job is running and ensure the panel shows the correct state
            if ( running_job == true ) {
                // Ensure we're showing the progress panel and controls
                if ( ! ProgressPanel->IsShown( ) ) {
                    ProgressPanel->Show(true);
                    StartPanel->Show(false);
                    OutputTextPanel->Show(true);
                    InfoPanel->Show(false);
                    InputPanel->Show(false);
                    Layout( );
                }
                // Ensure the terminate button is visible
                if ( ! CancelAlignmentButton->IsShown( ) ) {
                    CancelAlignmentButton->Show(true);
                    FinishButton->Show(false);
                    ProgressPanel->Layout( );
                }
            }

            // Reparent back to panel and hide
            queue_manager->Reparent(this);
            queue_manager->Hide( );
            queue_dialog->Destroy( );
        });

        queue_dialog->Show(true);
        return -1; // Dialog mode doesn't need to track specific queue ID for caller
    }

    return queue_id; // Return database ID for non-dialog mode
}

void MatchTemplatePanel::CheckForUnfinishedWork(std::vector<long>& images_to_resume, int image_group_id, long search_id, long images_total) {
    QM_LOG_METHOD_ENTRY("CheckForUnfinishedWork");
    // FIXME: debug asserts

    // Get a list of unfinished images by performing a left join between all
    // image assets or the image assets in the desired group and the results
    // stored for this job. The image assets that don't match are what we
    // want.

    // All images
    if ( image_group_id == -1 ) {
        main_frame->current_project.database.FillVectorFromSelectCommand(
                wxString::Format("select IMAGE_ASSETS.IMAGE_ASSET_ID, COMP.IMAGE_ASSET_ID as CID FROM IMAGE_ASSETS "
                                 "LEFT JOIN (SELECT IMAGE_ASSET_ID FROM TEMPLATE_MATCH_LIST WHERE SEARCH_ID = %ld ) COMP "
                                 "ON IMAGE_ASSETS.IMAGE_ASSET_ID = COMP.IMAGE_ASSET_ID "
                                 "WHERE CID IS NULL",
                                 search_id),
                images_to_resume);
    }
    // An Image group
    else {
        main_frame->current_project.database.FillVectorFromSelectCommand(
                wxString::Format("select IMAGE_ASSETS.IMAGE_ASSET_ID, COMP.IMAGE_ASSET_ID as CID FROM IMAGE_GROUP_%i AS IMAGE_ASSETS "
                                 "LEFT JOIN (SELECT IMAGE_ASSET_ID FROM TEMPLATE_MATCH_LIST WHERE SEARCH_ID = %ld ) COMP "
                                 "ON IMAGE_ASSETS.IMAGE_ASSET_ID = COMP.IMAGE_ASSET_ID "
                                 "WHERE CID IS NULL",
                                 image_group_id, search_id),
                images_to_resume);
    }
    long images_to_be_processed        = images_to_resume.size( );
    long images_successfully_processed = images_total - images_to_be_processed;
    bool no_unfinished_jobs            = (images_to_be_processed == 0);

    MyDebugAssertFalse(no_unfinished_jobs, "No unfinished jobs, but should have been caught earlier");

    QM_LOG_SEARCH("Search %ld: Found %ld unfinished images out of %ld total (previously processed %ld)",
                  search_id, images_to_be_processed, images_total, images_successfully_processed);
    QM_LOG_METHOD_EXIT("CheckForUnfinishedWork");
    return;
}

bool MatchTemplatePanel::SetupSearchBatchFromQueueItem(const TemplateMatchQueueItem& job, long& pending_queue_id) {
    QM_LOG_METHOD_ENTRY("SetupSearchBatchFromQueueItem");
    // Freeze GUI updates in queue manager to prevent interference during setup

    if ( queue_manager ) {
        queue_manager->SetGuiUpdateFrozen(true);
    }

    // First populate GUI with the job parameters
    PopulateGuiFromQueueItem(job);

    // revert - debug: Log ALL parameters being used for execution
    QM_LOG_DEBUG("=== SetupSearchBatchFromQueueItem: ALL EXECUTION PARAMETERS ===");
    QM_LOG_DEBUG("  search_name: %s", job.search_name);
    QM_LOG_DEBUG("  image_group_id: %d", job.image_group_id);
    QM_LOG_DEBUG("  reference_volume_asset_id: %d", job.reference_volume_asset_id);
    QM_LOG_DEBUG("  run_profile_id: %d", job.run_profile_id);
    QM_LOG_DEBUG("  symmetry: %s", job.symmetry);
    QM_LOG_DEBUG("  high_resolution_limit: %.2f", job.high_resolution_limit);
    QM_LOG_DEBUG("  low_resolution_limit: %.2f", job.low_resolution_limit);
    QM_LOG_DEBUG("  out_of_plane_angular_step: %.2f", job.out_of_plane_angular_step);
    QM_LOG_DEBUG("  in_plane_angular_step: %.2f", job.in_plane_angular_step);
    QM_LOG_DEBUG("  defocus_search_range: %.2f", job.defocus_search_range);
    QM_LOG_DEBUG("  defocus_step: %.2f", job.defocus_step);
    QM_LOG_DEBUG("  pixel_size_search_range: %.4f", job.pixel_size_search_range);
    QM_LOG_DEBUG("  pixel_size_step: %.4f", job.pixel_size_step);
    QM_LOG_DEBUG("  min_peak_radius: %.2f", job.min_peak_radius);
    QM_LOG_DEBUG("  use_gpu: %d", job.use_gpu);
    QM_LOG_DEBUG("  use_fast_fft: %d", job.use_fast_fft);
    QM_LOG_DEBUG("  ref_box_size_in_angstroms: %.2f", job.ref_box_size_in_angstroms);
    QM_LOG_DEBUG("  mask_radius: %.2f", job.mask_radius);
    QM_LOG_DEBUG("=== END EXECUTION PARAMETERS ===");

    // Now run the existing setup logic that was in StartEstimationClick
    // This mirrors the logic from StartEstimationClick but uses the job parameters directly

    // We could juse the image_group_id but we've already set that in the GUI so keep consistent with prior
    // results until the bugs are sorted. revert
    // active_group.CopyFrom(&image_asset_panel->all_groups_list->groups[job.image_group_id]);
    active_group.CopyFrom(&image_asset_panel->all_groups_list->groups[GroupComboBox->GetSelection( )]);
    MyDebugAssertTrue(active_group.id == job.image_group_id, "Active group ID %d doesn't match job group ID %d", active_group.id, job.image_group_id);

    // TODO: make sure we just set N from active_group.number_of_members
    // We are resuming a search if we are here and the queue already has a search_id
    bool resume = job.search_id == -1 ? false : true;

    std::vector<long> images_to_resume;

    if ( resume ) {
        // FIXME: active_group.number_of_members
        CheckForUnfinishedWork(images_to_resume, job.image_group_id, job.search_id, active_group.number_of_members);
        active_group.RemoveAll( );
        for ( auto& id : images_to_resume ) {
            active_group.AddMember(image_asset_panel->ReturnArrayPositionFromAssetID(id));
        }
    }

    QM_LOG_DEBUG("Active group has %ld members", active_group.number_of_members);
    QM_LOG_DEBUG("Resuming value bool %d", resume);
    QM_LOG_DEBUG("Images to resume %zu", images_to_resume.size( ));

    // Store the pending queue ID - will be committed to running_queue_id only after job controller starts
    pending_queue_id = job.database_queue_id;
    MyDebugAssertFalse(pending_queue_id < 0 && resume, "Pending queue ID shows no results  (%ld) but job indicates resume", pending_queue_id);

    float resolution_limit;
    float orientations_per_process;
    float current_orientation_counter;

    int job_counter;
    int number_of_rotations = 0;
    int number_of_defocus_positions;
    int number_of_pixel_size_positions;

    bool use_gpu      = job.use_gpu;
    bool use_fast_fft = job.use_fast_fft;
    int  max_threads  = 1; // Only used for the GPU code

    int image_number_for_gui;
    int number_of_jobs_per_image_in_gui;
    int number_of_jobs;

    double voltage_kV;
    double spherical_aberration_mm;
    double amplitude_contrast;
    double defocus1;
    double defocus2;
    double defocus_angle;
    double phase_shift;
    double iciness;

    input_image_filenames.Clear( );
    cached_results.Clear( );

    ResultsPanel->Clear( );

    // Package the job details..
    EulerSearch* current_image_euler_search;
    ImageAsset*  current_image;
    VolumeAsset* current_volume;

    // Convert database asset ID to array position
    int volume_array_position = volume_asset_panel->ReturnArrayPositionFromAssetID(job.reference_volume_asset_id);
    MyDebugAssertTrue(volume_array_position >= 0, "Reference volume asset ID %d not found in volume panel", job.reference_volume_asset_id);
    current_volume         = volume_asset_panel->ReturnAssetPointer(volume_array_position);
    ref_box_size_in_pixels = current_volume->x_size / current_volume->pixel_size;

    ParameterMap parameter_map;
    parameter_map.SetAllTrue( );

    float wanted_out_of_plane_angular_step = job.out_of_plane_angular_step;
    float wanted_in_plane_angular_step     = job.in_plane_angular_step;

    float defocus_search_range    = job.defocus_search_range;
    float defocus_step            = job.defocus_step;
    float pixel_size_search_range = job.pixel_size_search_range;
    float pixel_size_step         = job.pixel_size_step;

    float min_peak_radius = job.min_peak_radius;

    wxString wanted_symmetry       = job.symmetry;
    float    high_resolution_limit = job.high_resolution_limit;

    QM_LOG_DEBUG("Setting up search with symmetry %s, defocus range %.2f, defocus step %.2f",
                 wanted_symmetry, defocus_search_range, defocus_step);

    // Find run profile by database ID (not combo box index)
    int profile_array_index = -1;
    for ( int i = 0; i < run_profiles_panel->run_profile_manager.number_of_run_profiles; i++ ) {
        if ( run_profiles_panel->run_profile_manager.run_profiles[i].id == job.run_profile_id ) {
            profile_array_index = i;
            break;
        }
    }
    MyDebugAssertTrue(profile_array_index >= 0, "Run profile ID %d not found in run profiles panel", job.run_profile_id);
    RunProfile active_refinement_run_profile = run_profiles_panel->run_profile_manager.run_profiles[profile_array_index];

    int number_of_processes = active_refinement_run_profile.ReturnTotalJobs( );

    // Get first image to make decisions about how many jobs
    current_image              = image_asset_panel->ReturnAssetPointer(active_group.members[0]);
    current_image_euler_search = new EulerSearch;
    resolution_limit           = current_image->pixel_size * 2.0f; // Nyquist limit
    current_image_euler_search->InitGrid(wanted_symmetry, wanted_out_of_plane_angular_step, 0.0, 0.0, 360.0, wanted_in_plane_angular_step, 0.0, current_image->pixel_size / resolution_limit, parameter_map, 1);

    if ( wanted_symmetry.StartsWith("C") ) {
        if ( current_image_euler_search->test_mirror == true ) {
            current_image_euler_search->theta_max = 180.0f;
        }
    }

    current_image_euler_search->CalculateGridSearchPositions(false);

    // Calculate jobs per image
    if ( use_gpu ) {
        number_of_jobs_per_image_in_gui = number_of_processes;
        number_of_jobs                  = number_of_jobs_per_image_in_gui * active_group.number_of_members;
        wxPrintf("In USEGPU:\n There are %d search positions\nThere are %d jobs per image\n", current_image_euler_search->number_of_search_positions, number_of_jobs_per_image_in_gui);
        delete current_image_euler_search;
    }
    else {
        if ( active_group.number_of_members >= 5 || current_image_euler_search->number_of_search_positions < number_of_processes * 20 )
            number_of_jobs_per_image_in_gui = number_of_processes;
        else if ( current_image_euler_search->number_of_search_positions > number_of_processes * 250 )
            number_of_jobs_per_image_in_gui = number_of_processes * 10;
        else
            number_of_jobs_per_image_in_gui = number_of_processes * 5;

        number_of_jobs = number_of_jobs_per_image_in_gui * active_group.number_of_members;
        delete current_image_euler_search;
    }

    // Calculate number of rotations
    for ( float current_psi = 0.0f; current_psi <= 360.0f; current_psi += wanted_in_plane_angular_step ) {
        number_of_rotations++;
    }

    // Initialize job package with executable name and optional custom CLI args
    wxString executable_name = "match_template";
    if ( use_gpu ) {
        executable_name += "_gpu";
    }
    // Append custom CLI args if present (ensure leading space)
    if ( ! current_custom_cli_args.IsEmpty( ) ) {
        if ( ! current_custom_cli_args.StartsWith(" ") ) {
            executable_name += " ";
        }
        executable_name += current_custom_cli_args;
        wxPrintf("Appending custom CLI args to executable: %s\n", current_custom_cli_args.ToUTF8( ).data( ));
    }

    current_job_package.Reset(active_refinement_run_profile, executable_name, number_of_jobs);

    // Initialize job tracker to measure elapsed time
    // Track by number of IMAGES (one result per image), not number of sub-jobs
    my_job_tracker.StartTracking(active_group.number_of_members);

    expected_number_of_results = 0;
    number_of_received_results = 0;

    // Set up progress dialog and job preparation
    OneSecondProgressDialog* my_progress_dialog = new OneSecondProgressDialog("Preparing Job", "Preparing Job...", active_group.number_of_members, this, wxPD_REMAINING_TIME | wxPD_AUTO_HIDE | wxPD_APP_MODAL);

    TemplateMatchJobResults temp_result;
    temp_result.parent_search_id           = -1;
    temp_result.search_type_code           = cistem::job_type::template_match_full_search;
    temp_result.mask_radius                = 0.0f;
    temp_result.min_peak_radius            = min_peak_radius;
    temp_result.exclude_above_xy_threshold = false;
    temp_result.xy_change_threshold        = 0.0f;

    // Determine search_id BEFORE calculating predicted_search_id for filenames
    if ( pending_queue_id > 0 ) {
        int existing_search_id = main_frame->current_project.database.GetSearchIdForQueueItem(pending_queue_id);
        if ( existing_search_id > 0 ) {
            search_id = existing_search_id;
            QM_LOG_DB("Queue item %ld resuming with existing search_id %d", pending_queue_id, search_id);
            int result_count = main_frame->current_project.database.ReturnSingleIntFromSelectCommand(
                    wxString::Format("SELECT COUNT(*) FROM TEMPLATE_MATCH_LIST WHERE SEARCH_ID = %d", search_id));
            MyDebugAssertTrue(result_count > 0,
                              "Queue item has search_id %d but no results in TEMPLATE_MATCH_LIST", search_id);
        }
        else {
            search_id = -1;
            QM_LOG_DB("Queue item %ld starting fresh - search_id will be assigned when first result is written", pending_queue_id);
        }
    }
    else {
        search_id = main_frame->current_project.database.ReturnHighestTemplateMatchJobID( ) + 1;
        QM_LOG_DB("Non-queue job - assigning search_id %d immediately", search_id);
    }

    // Calculate predicted_search_id for filenames (now that search_id is correctly set)
    int predicted_search_id;
    if ( search_id == -1 ) {
        predicted_search_id = main_frame->current_project.database.ReturnHighestTemplateMatchJobID( ) + 1;
    }
    else {
        predicted_search_id = search_id;
    }

    // Loop over all images to set up jobs
    for ( int image_counter = 0; image_counter < active_group.number_of_members; image_counter++ ) {
        image_number_for_gui = image_counter + 1;

        current_image = image_asset_panel->ReturnAssetPointer(active_group.members[image_counter]);

        resolution_limit           = current_image->pixel_size * 2.0f;
        current_image_euler_search = new EulerSearch;
        current_image_euler_search->InitGrid(wanted_symmetry, wanted_out_of_plane_angular_step, 0.0, 0.0, 360.0, wanted_in_plane_angular_step, 0.0, current_image->pixel_size / resolution_limit, parameter_map, 1);
        if ( wanted_symmetry.StartsWith("C") ) {
            if ( current_image_euler_search->test_mirror == true ) {
                current_image_euler_search->theta_max = 180.0f;
            }
        }
        current_image_euler_search->CalculateGridSearchPositions(false);

        if ( defocus_search_range > 0 )
            number_of_defocus_positions = 2 * myround(float(defocus_search_range) / float(defocus_step)) + 1;
        else
            number_of_defocus_positions = 1;

        if ( pixel_size_search_range > 0 )
            number_of_pixel_size_positions = 2 * myround(float(pixel_size_search_range) / float(pixel_size_step)) + 1;
        else
            number_of_pixel_size_positions = 1;

        // wxPrintf("For Image %ld\nThere are %i search positions\nThere are %i jobs per image\n",
        //          active_group.members[image_counter], current_image_euler_search->number_of_search_positions, number_of_jobs_per_image_in_gui);
        // wxPrintf("Calculating %i correlation maps\n", current_image_euler_search->number_of_search_positions * number_of_rotations * number_of_defocus_positions * number_of_pixel_size_positions);

        expected_number_of_results += current_image_euler_search->number_of_search_positions * number_of_rotations * number_of_defocus_positions * number_of_pixel_size_positions;
        orientations_per_process = float(current_image_euler_search->number_of_search_positions) / float(number_of_jobs_per_image_in_gui);
        if ( orientations_per_process < 1 )
            orientations_per_process = 1;

        // Predict what the template_match_id WILL BE when this image completes
        // Each image gets unique template_match_id, but all images in batch share predicted_search_id (calculated above)
        int predicted_template_match_id = main_frame->current_project.database.ReturnHighestTemplateMatchID( ) + image_counter + 1;

        main_frame->current_project.database.GetCTFParameters(current_image->ctf_estimation_id, voltage_kV, spherical_aberration_mm, amplitude_contrast, defocus1, defocus2, defocus_angle, phase_shift, iciness);

        // Generate output filename base in correct format: <path>/<image_name>
        // The helper functions will add _<type>_<tm_id>_<search_id>.mrc
        wxString output_filename_base = main_frame->current_project.template_matching_asset_directory.GetFullPath( ) +
                                        "/" + current_image->filename.GetName( );

        // Create temp result object with predicted IDs for generating correct filenames
        TemplateMatchJobResults temp_filename_helper;
        temp_filename_helper.output_filename_base = output_filename_base;
        temp_filename_helper.template_match_id    = predicted_template_match_id;
        temp_filename_helper.search_id            = predicted_search_id;

        // Generate output filenames using helper methods
        wxString mip_output_file             = temp_filename_helper.GetMipFilename( );
        wxString best_psi_output_file        = temp_filename_helper.GetPsiFilename( );
        wxString best_theta_output_file      = temp_filename_helper.GetThetaFilename( );
        wxString best_phi_output_file        = temp_filename_helper.GetPhiFilename( );
        wxString best_defocus_output_file    = temp_filename_helper.GetDefocusFilename( );
        wxString best_pixel_size_output_file = temp_filename_helper.GetPixelSizeFilename( );
        wxString scaled_mip_output_file      = temp_filename_helper.GetScaledMipFilename( );
        wxString output_histogram_file       = temp_filename_helper.GetHistogramFilename( );
        wxString output_result_file          = temp_filename_helper.GetProjectionResultFilename( );
        wxString correlation_avg_output_file = temp_filename_helper.GetAvgFilename( );
        wxString correlation_std_output_file = temp_filename_helper.GetStdFilename( );

        current_orientation_counter = 0;

        wxString input_search_image   = current_image->filename.GetFullPath( );
        wxString input_reconstruction = current_volume->filename.GetFullPath( );
        float    pixel_size           = current_image->pixel_size;

        input_image_filenames.Add(input_search_image);

        float low_resolution_limit = 300.0f; // FIXME set this somewhere that is not buried in the code!

        temp_result.image_asset_id            = current_image->asset_id;
        temp_result.job_name                  = wxString::Format("Template: %s", current_volume->filename.GetName( ));
        temp_result.reference_volume_asset_id = current_volume->asset_id;
        // datetime_of_run will be set when result actually arrives in HandleSocketTemplateMatchResultReady
        temp_result.symmetry                        = wanted_symmetry;
        temp_result.pixel_size                      = pixel_size;
        temp_result.voltage                         = voltage_kV;
        temp_result.spherical_aberration            = spherical_aberration_mm;
        temp_result.amplitude_contrast              = amplitude_contrast;
        temp_result.defocus1                        = defocus1;
        temp_result.defocus2                        = defocus2;
        temp_result.defocus_angle                   = defocus_angle;
        temp_result.phase_shift                     = phase_shift;
        temp_result.low_res_limit                   = low_resolution_limit;
        temp_result.high_res_limit                  = high_resolution_limit;
        temp_result.out_of_plane_step               = wanted_out_of_plane_angular_step;
        temp_result.in_plane_step                   = wanted_in_plane_angular_step;
        temp_result.defocus_search_range            = defocus_search_range;
        temp_result.defocus_step                    = defocus_step;
        temp_result.pixel_size_search_range         = pixel_size_search_range;
        temp_result.pixel_size_step                 = pixel_size_step;
        temp_result.reference_box_size_in_angstroms = ref_box_size_in_pixels * pixel_size;
        // Store just the base filename in database - we have all info to reconstruct full names
        temp_result.output_filename_base = output_filename_base;
        temp_result.template_match_id    = predicted_template_match_id;
        temp_result.search_id            = predicted_search_id; // This is the search_id in database

        cached_results.Add(temp_result);

        // Create individual jobs for this image
        for ( job_counter = 0; job_counter < number_of_jobs_per_image_in_gui; job_counter++ ) {
            int   best_parameters_to_keep = 1;
            float padding                 = 1;
            bool  ctf_refinement          = false;
            float mask_radius_search      = 0.0f;

            //  wxPrintf("\n\tFor image %i, current_orientation_counter is %f\n", image_number_for_gui, current_orientation_counter);
            if ( current_orientation_counter >= current_image_euler_search->number_of_search_positions )
                current_orientation_counter = current_image_euler_search->number_of_search_positions - 1;
            int first_search_position = myroundint(current_orientation_counter);
            current_orientation_counter += orientations_per_process;
            if ( current_orientation_counter >= current_image_euler_search->number_of_search_positions || job_counter == number_of_jobs_per_image_in_gui - 1 )
                current_orientation_counter = current_image_euler_search->number_of_search_positions - 1;
            int last_search_position = myroundint(current_orientation_counter);
            current_orientation_counter++;

            wxString directory_for_results = main_frame->current_project.image_asset_directory.GetFullPath( );

            current_job_package.AddJob("ttffffffffffifffffbfftttttttttftiiiitttfbbi",
                                       input_search_image.ToUTF8( ).data( ),
                                       input_reconstruction.ToUTF8( ).data( ),
                                       pixel_size,
                                       voltage_kV,
                                       spherical_aberration_mm,
                                       amplitude_contrast,
                                       defocus1,
                                       defocus2,
                                       defocus_angle,
                                       low_resolution_limit,
                                       high_resolution_limit,
                                       wanted_out_of_plane_angular_step,
                                       best_parameters_to_keep,
                                       defocus_search_range,
                                       defocus_step,
                                       pixel_size_search_range,
                                       pixel_size_step,
                                       padding,
                                       ctf_refinement,
                                       mask_radius_search,
                                       phase_shift,
                                       mip_output_file.ToUTF8( ).data( ),
                                       best_psi_output_file.ToUTF8( ).data( ),
                                       best_theta_output_file.ToUTF8( ).data( ),
                                       best_phi_output_file.ToUTF8( ).data( ),
                                       best_defocus_output_file.ToUTF8( ).data( ),
                                       best_pixel_size_output_file.ToUTF8( ).data( ),
                                       scaled_mip_output_file.ToUTF8( ).data( ),
                                       correlation_std_output_file.ToUTF8( ).data( ),
                                       wanted_symmetry.ToUTF8( ).data( ),
                                       wanted_in_plane_angular_step,
                                       output_histogram_file.ToUTF8( ).data( ),
                                       first_search_position,
                                       last_search_position,
                                       image_number_for_gui,
                                       number_of_jobs_per_image_in_gui,
                                       correlation_avg_output_file.ToUTF8( ).data( ),
                                       directory_for_results.ToUTF8( ).data( ),
                                       output_result_file.ToUTF8( ).data( ),
                                       min_peak_radius,
                                       use_gpu,
                                       use_fast_fft,
                                       max_threads);
        }

        delete current_image_euler_search;
        my_progress_dialog->Update(image_counter + 1);
    }

    my_progress_dialog->Destroy( );

    // Unfreeze GUI updates in queue manager now that setup is complete
    if ( queue_manager ) {
        queue_manager->SetGuiUpdateFrozen(false);
    }
    QM_LOG_METHOD_EXIT("SetupSearchBatchFromQueueItem");
    return true;
}

bool MatchTemplatePanel::ExecuteCurrentSearch(long pending_queue_id) {
    // Get the run profile to use - the GUI should be populated correctly by PopulateGuiFromQueueItem
    int run_profile_to_use = RunProfileComboBox->GetSelection( );

    // Debug print to verify run profile is correctly set
    if ( running_queue_id > 0 ) {
        QM_LOG_SEARCH("Executing search with queue ID %ld using run profile selection %d",
                      running_queue_id, run_profile_to_use);
    }
    else {
        wxPrintf("Executing GUI job with run profile selection %d\n", run_profile_to_use);
    }

    // Launch the job controller
    my_job_id = main_frame->job_controller.AddJob(this,
                                                  run_profiles_panel->run_profile_manager.run_profiles[run_profile_to_use].manager_command,
                                                  run_profiles_panel->run_profile_manager.run_profiles[run_profile_to_use].gui_address);

    if ( my_job_id != -1 ) {
        // Job controller successfully started - NOW commit all state changes
        running_job      = true;
        running_queue_id = pending_queue_id;

        // Move job to available queue (QUEUE_POSITION = -1) to mark it as executing
        if ( pending_queue_id > 0 ) {
            wxString sql = wxString::Format("UPDATE TEMPLATE_MATCH_QUEUE SET QUEUE_POSITION = -1 WHERE QUEUE_ID = %ld", pending_queue_id);
            main_frame->current_project.database.ExecuteSQL(sql);
            QM_LOG_STATE("Moved queue item %ld to available queue (QUEUE_POSITION = -1) after job controller started", pending_queue_id);
        }

        SetNumberConnectedTextToZeroAndStartTracking( );

        StartPanel->Show(false);
        ProgressPanel->Show(true);
        InputPanel->Show(false);

        ExpertPanel->Show(false);
        InfoPanel->Show(false);
        OutputTextPanel->Show(true);
        ResultsPanel->Show(true);

        GroupComboBox->Enable(false);
        Layout( );

        ProgressBar->Pulse( );
        // Job is now active in job controller (AddJob sets is_active = true)
        return true;
    }

    return false;
}

bool MatchTemplatePanel::ExecuteSearch(const TemplateMatchQueueItem* queue_item) {
    // ExecuteSearch should ALWAYS be called with a queue_item (either from QueueManager or from StartEstimationClick)
    // Both code paths now go through the queue, so queue_item should never be null
    MyDebugAssertTrue(queue_item != nullptr, "ExecuteSearch called with null queue_item - both StartEstimationClick and QueueManager should provide a queue_item");

    // Validate job parameters before execution
    MyDebugAssertTrue(queue_item->database_queue_id >= 0, "Cannot execute job with invalid database_queue_id: %ld", queue_item->database_queue_id);
    MyDebugAssertTrue(queue_item->queue_status == "pending" || queue_item->queue_status == "failed" || queue_item->queue_status == "partial",
                      "Cannot execute job with status '%s', must be 'pending', 'failed', or 'partial'", queue_item->queue_status.mb_str( ).data( ));
    MyDebugAssertFalse(queue_item->search_name.IsEmpty( ), "Cannot execute search with empty search_name");
    MyDebugAssertTrue(queue_item->image_group_id >= -1, "Cannot execute job with invalid image_group_id: %d", queue_item->image_group_id);
    MyDebugAssertTrue(queue_item->reference_volume_asset_id >= 1, "Cannot execute job with invalid reference_volume_asset_id: %d (asset IDs start at 1)", queue_item->reference_volume_asset_id);
    MyDebugAssertTrue(queue_item->run_profile_id >= 1, "Cannot execute job with invalid run_profile_id: %d (profile IDs start at 1)", queue_item->run_profile_id);

    // Check if another job is already running
    if ( running_job == true ) {
        wxMessageBox("A job is already running. Please wait for it to complete.",
                     "Job Running", wxOK | wxICON_WARNING);
        return false;
    }

    // Setup job from queue item
    QM_LOG_SEARCH("Setting up job %ld from queue item...", queue_item->database_queue_id);
    long pending_queue_id = -1;
    bool setup_success    = SetupSearchBatchFromQueueItem(*queue_item, pending_queue_id);

    if ( ! setup_success ) {
        QM_LOG_ERROR("Failed to setup job %ld - no work to do or setup failed", queue_item->database_queue_id);
        return false;
    }

    // Execute the job (launches job controller and commits state fauon success)
    return ExecuteCurrentSearch(pending_queue_id);
}

void MatchTemplatePanel::HandleSocketDisconnect(wxSocketBase* connected_socket) {
    QM_LOG_METHOD_ENTRY("HandleSocketDisconnect");
    QM_LOG_STATE("    Socket disconnect detected (my_job_id=%ld, running_queue_id=%ld)", my_job_id, running_queue_id);

    // If this was a queue job, reset panel state and trigger auto-advance
    // HandleSocketDisconnect is called when master process exits, which may happen before
    // ProcessAllJobsFinished is called (especially for fast-completing or failing jobs)
    if ( running_queue_id > 0 && queue_manager ) {
        long completed_queue_id = running_queue_id;
        running_queue_id        = -1; // Clear before triggering callback

        QM_LOG_SEARCH("Master socket disconnected - triggering OnSearchCompleted for queue ID %ld", completed_queue_id);
        queue_manager->OnSearchCompleted(completed_queue_id, true);
    }
    else if ( running_queue_id > 0 ) {
        // Queue job but no callback - shouldn't happen
        QM_LOG_ERROR("Queue job completed but no queue_manager registered!");
        running_queue_id = -1;
    }

    QM_LOG_METHOD_EXIT("HandleSocketDisconnect");
}
