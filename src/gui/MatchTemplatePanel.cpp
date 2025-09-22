//#include "../core/core_headers.h"
#include "../constants/constants.h"
#include "../core/gui_core_headers.h"
#include "TemplateMatchQueueManager.h"

// extern MyMovieAssetPanel *movie_asset_panel;
extern MyImageAssetPanel*         image_asset_panel;
extern MyVolumeAssetPanel*        volume_asset_panel;
extern MyRunProfilesPanel*        run_profiles_panel;
extern MyMainFrame*               main_frame;
extern MatchTemplateResultsPanel* match_template_results_panel;

MatchTemplatePanel::MatchTemplatePanel(wxWindow* parent)
    : MatchTemplatePanelParent(parent) {
    // Set variables

    my_job_id                 = -1;
    running_job               = false;
    running_queue_job_id      = -1; // Not running from queue
    queue_completion_callback = nullptr; // No callback initially
    current_custom_cli_args   = ""; // Clear custom CLI args

    group_combo_is_dirty   = false;
    run_profiles_are_dirty = false;
    set_up_to_resume_job   = false;

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

    ResetDefaults( );
    //	EnableMovieProcessingIfAppropriate();

    result_bitmap.Create(1, 1, 24);
    time_of_last_result_update = time(NULL);

    DefocusSearchRangeNumericCtrl->SetMinMaxValue(0.0f, FLT_MAX);
    DefocusSearchStepNumericCtrl->SetMinMaxValue(1.0f, FLT_MAX);
    PixelSizeSearchRangeNumericCtrl->SetMinMaxValue(0.0f, FLT_MAX);
    PixelSizeSearchStepNumericCtrl->SetMinMaxValue(0.01f, FLT_MAX);
    HighResolutionLimitNumericCtrl->SetMinMaxValue(0.0f, FLT_MAX);

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

    GroupComboBox->AssetComboBox->Bind(wxEVT_COMMAND_COMBOBOX_SELECTED, &MatchTemplatePanel::OnGroupComboBox, this);
}

/*
void MatchTemplatePanel::EnableMovieProcessingIfAppropriate()
{
	// Check whether all members of the group have movie parents. If not, make sure we only allow image processing
	MovieRadioButton->Enable(true);
	NoMovieFramesStaticText->Enable(true);
	NoFramesToAverageSpinCtrl->Enable(true);
	for (int counter = 0; counter < image_asset_panel->ReturnGroupSize(GroupComboBox->GetSelection()); counter ++ )
	{
		if (image_asset_panel->all_assets_list->ReturnAssetPointer(image_asset_panel->ReturnGroupMember(GroupComboBox->GetSelection(),counter))->parent_id < 0)
		{
			MovieRadioButton->SetValue(false);
			MovieRadioButton->Enable(false);
			NoMovieFramesStaticText->Enable(false);
			NoFramesToAverageSpinCtrl->Enable(false);
			ImageRadioButton->SetValue(true);
		}
	}
}
*/

void MatchTemplatePanel::ResetAllDefaultsClick(wxCommandEvent& event) {
    ResetDefaults( );
}

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
    }

    ResetDefaults( );
    Layout( );
}

void MatchTemplatePanel::ResetDefaults( ) {
    OutofPlaneStepNumericCtrl->ChangeValueFloat(2.5);
    InPlaneStepNumericCtrl->ChangeValueFloat(1.5);
    MinPeakRadiusNumericCtrl->ChangeValueFloat(10.0f);

    DefocusSearchYesRadio->SetValue(true);
    PixelSizeSearchNoRadio->SetValue(true);
    set_up_to_resume_job = false;

    SymmetryComboBox->SetValue("C1");

    if ( main_frame->current_project.is_open ) {
        // deprecated - remove: Resume run functionality replaced by Queue Manager
        // ResumeRunCheckBox->SetValue(false);
        // deprecated - remove: Resume run functionality replaced by Queue Manager
        /*
        if ( match_template_results_panel->search_ids.empty( ) )
            ResumeRunCheckBox->Enable(true);
        else
            ResumeRunCheckBox->Enable(false);
        */
    }
    else {
        // deprecated - remove: Resume run functionality replaced by Queue Manager
        // ResumeRunCheckBox->Enable(false);
    }

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

    //	AssetGroup active_group;
    active_group.CopyFrom(&image_asset_panel->all_groups_list->groups[GroupComboBox->GetSelection( )]);
    if ( active_group.number_of_members > 0 ) {
        ImageAsset* current_image;
        current_image = image_asset_panel->ReturnAssetPointer(GroupComboBox->GetSelection( ));
        HighResolutionLimitNumericCtrl->ChangeValueFloat(2.0f * current_image->pixel_size);
    }
}

void MatchTemplatePanel::OnGroupComboBox(wxCommandEvent& event) {
    //	ResetDefaults();
    //	AssetGroup active_group;

    active_group.CopyFrom(&image_asset_panel->all_groups_list->groups[GroupComboBox->GetSelection( )]);

    if ( active_group.number_of_members > 0 ) {

        ImageAsset* current_image;
        current_image = image_asset_panel->ReturnAssetPointer(active_group.members[0]);
        HighResolutionLimitNumericCtrl->ChangeValueFloat(2.0f * current_image->pixel_size);
    }

    if ( GroupComboBox->GetCount( ) > 0 && main_frame->current_project.is_open == true )
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
    GroupComboBox->FillComboBox(true);

    if ( GroupComboBox->GetCount( ) > 0 && main_frame->current_project.is_open == true )
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

void MatchTemplatePanel::FillRunProfileComboBox( ) {
    RunProfileComboBox->FillWithRunProfiles( );
}

bool MatchTemplatePanel::CheckGroupHasDefocusValues( ) {
    wxArrayLong images_with_defocus_values = main_frame->current_project.database.ReturnLongArrayFromSelectCommand("SELECT DISTINCT IMAGE_ASSET_ID FROM ESTIMATED_CTF_PARAMETERS");
    long        current_image_id;
    int         images_with_defocus_counter;
    bool        image_was_found;

    for ( int image_in_group_counter = 0; image_in_group_counter < image_asset_panel->ReturnGroupSize(GroupComboBox->GetSelection( )); image_in_group_counter++ ) {
        current_image_id = image_asset_panel->all_assets_list->ReturnAssetPointer(image_asset_panel->ReturnGroupMember(GroupComboBox->GetSelection( ), image_in_group_counter))->asset_id;
        image_was_found  = false;

        for ( images_with_defocus_counter = 0; images_with_defocus_counter < images_with_defocus_values.GetCount( ); images_with_defocus_counter++ ) {
            if ( images_with_defocus_values[images_with_defocus_counter] == current_image_id ) {
                image_was_found = true;
                break;
            }
        }

        if ( image_was_found == false )
            return false;
    }

    return true;
}

void MatchTemplatePanel::OnUpdateUI(wxUpdateUIEvent& event) {

    // We want things to be greyed out if the user is re-running the job.
    if ( set_up_to_resume_job ) {
        return;
    }

    // are there enough members in the selected group.
    if ( main_frame->current_project.is_open == false ) {
        RunProfileComboBox->Enable(false);
        GroupComboBox->Enable(false);
        StartEstimationButton->Enable(false);
        ReferenceSelectPanel->Enable(false);
        // deprecated - remove: Resume run functionality replaced by Queue Manager
        // ResumeRunCheckBox->Enable(false);
    }
    else {
        if ( match_template_results_panel->template_match_job_ids.empty( ) ) {
            // deprecated - remove: Resume run functionality replaced by Queue Manager
            // ResumeRunCheckBox->Enable(true);
        }
        else {
            // deprecated - remove: Resume run functionality replaced by Queue Manager
            // ResumeRunCheckBox->Enable(false);
        }

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

/** 
 * Disables argument controls if set_up_to_resume_job is true.
 * In that case a TemplateMatchResult must be supplied to set
 * the arguments identical to the to be resumed job.
*/

void MatchTemplatePanel::SetInputsForPossibleReRun(bool set_up_to_resume_job, TemplateMatchJobResults* job_to_resume) {

    this->set_up_to_resume_job = set_up_to_resume_job;
    bool enable_value;

    // FIXME: If the project is moved, the reference volume will have anew path and we aren't allowing updates to the reference.
    // Probably the right solution is to be able to update the path to the reference when updating the project. Or alternatively
    // to have a dialog pop up when a reference is NOT found and ask if the user wants to update the reference.
    if ( set_up_to_resume_job ) {
        // We want to disable user inputs so the job run matches the intial state.
        enable_value = false;
        // deprecated - remove: Resume run functionality replaced by Queue Manager
        // ResumeRunCheckBox->Show(true);
        // deprecated - remove: Resume run functionality replaced by Queue Manager
        // ResumeRunCheckBox->SetValue(true);
        // deprecated - remove: Resume run functionality replaced by Queue Manager
        // ResumeRunCheckBox->Enable(true);
        ResetAllDefaultsButton->Enable(false);

        // We want to set the controls to the values of the job to be resumed.

        // SetSelection requires the array position in the volume asset panel,
        // which needs to be calculated from the volume asset id.
        ReferenceSelectPanel->SetSelection(volume_asset_panel->ReturnArrayPositionFromAssetID(job_to_resume->ref_volume_asset_id));
        OutofPlaneStepNumericCtrl->SetValue(wxString::Format(wxT("%f"), job_to_resume->out_of_plane_step));
        InPlaneStepNumericCtrl->SetValue(wxString::Format(wxT("%f"), job_to_resume->in_plane_step));
        MinPeakRadiusNumericCtrl->SetValue(wxString::Format(wxT("%f"), job_to_resume->min_peak_radius));
        HighResolutionLimitNumericCtrl->SetValue(wxString::Format(wxT("%f"), job_to_resume->high_res_limit));
        SymmetryComboBox->SetValue(job_to_resume->symmetry);
        // If either range or step are 0 no search will be perfomed.
        DefocusSearchYesRadio->SetValue(job_to_resume->defocus_search_range != 0.0f && job_to_resume->defocus_step != 0.0f);
        DefocusSearchNoRadio->SetValue(job_to_resume->defocus_search_range == 0.0f || job_to_resume->defocus_step == 0.0f);
        PixelSizeSearchYesRadio->SetValue(job_to_resume->pixel_size_search_range != 0.0f && job_to_resume->pixel_size_step != 0.0f);
        PixelSizeSearchNoRadio->SetValue(job_to_resume->pixel_size_search_range == 0.0f || job_to_resume->pixel_size_step == 0.0f);
        DefocusSearchRangeNumericCtrl->SetValue(wxString::Format(wxT("%f"), job_to_resume->defocus_search_range));
        DefocusSearchStepNumericCtrl->SetValue(wxString::Format(wxT("%f"), job_to_resume->defocus_step));
        PixelSizeSearchRangeNumericCtrl->SetValue(wxString::Format(wxT("%f"), job_to_resume->pixel_size_search_range));
        PixelSizeSearchStepNumericCtrl->SetValue(wxString::Format(wxT("%f"), job_to_resume->pixel_size_step));
    }
    else {
        // We want to allow the user to not re-run the job if the disable the ReRun radio button.
        // The state remembered in "was_enabled..." is meaningless here.
        enable_value = true;
        ResetAllDefaultsButton->Enable(true);
    }

    //SetAndRememberEnableState(GroupComboBox, was_enabled_GroupComboBox, enable_value);
    SetAndRememberEnableState(ReferenceSelectPanel, was_enabled_ReferenceSelectPanel, enable_value);

    SetAndRememberEnableState(OutofPlaneStepNumericCtrl, was_enabled_OutofPlaneStepNumericCtrl, enable_value);
    SetAndRememberEnableState(InPlaneStepNumericCtrl, was_enabled_InPlaneStepNumericCtrl, enable_value);
    SetAndRememberEnableState(MinPeakRadiusNumericCtrl, was_enabled_MinPeakRadiusNumericCtrl, enable_value);

    SetAndRememberEnableState(DefocusSearchYesRadio, was_enabled_DefocusSearchYesRadio, enable_value);
    SetAndRememberEnableState(DefocusSearchNoRadio, was_enabled_DefocusSearchNoRadio, enable_value);
    SetAndRememberEnableState(PixelSizeSearchYesRadio, was_enabled_PixelSizeSearchYesRadio, enable_value);
    SetAndRememberEnableState(PixelSizeSearchNoRadio, was_enabled_PixelSizeSearchNoRadio, enable_value);

    SetAndRememberEnableState(SymmetryComboBox, was_enabled_SymmetryComboBox, enable_value);
    SetAndRememberEnableState(HighResolutionLimitNumericCtrl, was_enabled_HighResolutionLimitNumericCtrl, enable_value);
    SetAndRememberEnableState(DefocusSearchRangeNumericCtrl, was_enabled_DefocusSearchRangeNumericCtrl, enable_value);
    SetAndRememberEnableState(DefocusSearchStepNumericCtrl, was_enabled_DefocusSearchStepNumericCtrl, enable_value);

    // It is okay to change the run profile for a rerun
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

void MatchTemplatePanel::StartEstimationClick(wxCommandEvent& event) {
    MyDebugAssertTrue(main_frame != nullptr, "main_frame is null");
    MyDebugAssertTrue(main_frame->current_project.is_open, "Project database is not open");

    // Add to queue silently (without dialog), then execute
    TemplateMatchQueueItem new_job  = CollectJobParametersFromGui( );
    long                   queue_id = AddJobToQueue(new_job, false); // No dialog for direct execution

    if ( queue_id > 0 ) {
        // Update the job with the database queue ID
        new_job.database_queue_id = queue_id;

        // Move job to available queue (position -1) since we're executing immediately
        // This matches what QueueManager::RunNextJob does before execution
        wxString sql = wxString::Format("UPDATE TEMPLATE_MATCH_QUEUE SET QUEUE_POSITION = -1 WHERE QUEUE_ID = %ld;", queue_id);
        main_frame->current_project.database.ExecuteSQL(sql);
        wxPrintf("StartEstimationClick: Moved job %ld to available queue (position -1)\n", queue_id);

        // Execute the job via unified method
        if ( ! ExecuteJob(&new_job) ) {
            wxMessageBox("Failed to start job", "Error", wxOK | wxICON_ERROR);
            return;
        }
    }
    else {
        wxMessageBox("Failed to add job to queue", "Error", wxOK | wxICON_ERROR);
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

    main_frame->current_project.database.Begin( );

    cached_results[image_number - 1].job_id = search_id;
    main_frame->current_project.database.AddTemplateMatchingResult(database_queue_id, cached_results[image_number - 1]);
    database_queue_id++;

    main_frame->current_project.database.SetActiveTemplateMatchJobForGivenImageAssetID(cached_results[image_number - 1].image_asset_id, search_id);
    main_frame->current_project.database.Commit( );
    match_template_results_panel->is_dirty = true;

    // Notify queue manager about result addition for real-time n/N updates
    if ( queue_completion_callback ) {
        queue_completion_callback->OnResultAdded(search_id);
    }
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

    //running_job = false;
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
    }

    // results should be ..

    // Defocus 1 (Angstroms)
    // Defocus 2 (Angstroms)
    // Astigmatism Angle (degrees)
    // Additional phase shift (e.g. from phase plate) radians
    // Score
    // Resolution (Angstroms) to which Thon rings are well fit by the CTF
    // Reolution (Angstroms) at which aliasing was detected

    /*

	if (current_time - time_of_last_result_update > 5)
	{
		// we need the filename of the image..

		wxString image_filename = image_asset_panel->ReturnAssetPointer(active_group.members[result_to_process->job_number])->filename.GetFullPath();

		ResultsPanel->Draw(my_job_package.jobs[result_to_process->job_number].arguments[3].ReturnStringArgument(), my_job_package.jobs[result_to_process->job_number].arguments[16].ReturnBoolArgument(), result_to_process->result_data[0], result_to_process->result_data[1], result_to_process->result_data[2], result_to_process->result_data[3], result_to_process->result_data[4], result_to_process->result_data[5], result_to_process->result_data[6], image_filename);
		time_of_last_result_update = time(NULL);
	}
*/

    //	my_job_tracker.MarkJobFinished();
    //	if (my_job_tracker.ShouldUpdate() == true) UpdateProgressBar();

    // store the results..
    //buffered_results[result_to_process->job_number] = result_to_process;
}

void MatchTemplatePanel::ProcessAllJobsFinished( ) {

    MyDebugAssertTrue(my_job_tracker.total_number_of_finished_jobs == my_job_tracker.total_number_of_jobs, "In ProcessAllJobsFinished, but total_number_of_finished_jobs != total_number_of_jobs. Oops.");

    // Notify queue manager that job is entering finalization phase
    // This prevents auto-advance from starting a new job while we're cleaning up
    if ( running_queue_job_id > 0 && queue_completion_callback ) {
        wxPrintf("Notifying queue manager that job %ld is entering finalization\n", running_queue_job_id);
        queue_completion_callback->OnJobEnteringFinalization(running_queue_job_id);
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

    // Kill the job (in case it isn't already dead)
    main_frame->job_controller.KillJob(my_job_id);

    // Reset job state to allow next job to start - MUST happen before callback
    running_job             = false;
    my_job_id               = -1;
    current_custom_cli_args = ""; // Clear custom CLI args after job completion

    // Update queue status if this was a queue job
    if ( running_queue_job_id > 0 ) {
        wxPrintf("Queue job %ld completed - updating status\n", running_queue_job_id);

        // Note: Queue status is now computed from completion data, no database update needed

        // Notify queue manager if callback is registered - called AFTER clearing running_job
        if ( queue_completion_callback ) {
            wxPrintf("Notifying queue manager of job completion\n");
            queue_completion_callback->OnJobCompleted(running_queue_job_id, true);
        }

        // Clear the queue job ID
        running_queue_job_id = -1;
    }

    WriteInfoText("All Jobs have finished.");
    ProgressBar->SetValue(100);

    // Check if there are more items in the execution queue waiting to run
    bool has_active_queue_items = false;
    if ( queue_completion_callback ) {
        has_active_queue_items = queue_completion_callback->ExecutionQueueHasActiveItems( );
    }

    if ( has_active_queue_items ) {
        // Don't show finish button yet - more queue items to process
        TimeRemainingText->SetLabel("Time Remaining : Waiting for next queue item...");
        CancelAlignmentButton->Show(false);
        FinishButton->Show(false);

        // Set a 20-second timeout to show finish button if queue doesn't progress
        // This prevents indefinite hanging if queue manager fails to start next job
        wxTimer* timeout_timer = new wxTimer();
        timeout_timer->Bind(wxEVT_TIMER, [this, timeout_timer](wxTimerEvent&) {
            // Check if we're still waiting (no job has started and finish button not shown)
            if (!running_job && FinishButton && !FinishButton->IsShown()) {
                wxPrintf("Queue transition timeout after 20 seconds - showing Finish button\n");
                TimeRemainingText->SetLabel("Time Remaining : Queue timeout - manual intervention needed");
                FinishButton->Show(true);
                ProgressPanel->Layout();

                // Try to trigger queue progression one more time if callback exists
                if (queue_completion_callback) {
                    wxPrintf("Attempting to trigger queue progression after timeout\n");
                    queue_completion_callback->ProgressExecutionQueue();
                }
            }
            delete timeout_timer;  // Clean up the timer
        });
        timeout_timer->StartOnce(20000);  // 20 second timeout
    }
    else {
        // All done - show finish button
        TimeRemainingText->SetLabel("Time Remaining : All Done!");
        CancelAlignmentButton->Show(false);
        FinishButton->Show(true);
    }
    ProgressPanel->Layout( );
}

void MatchTemplatePanel::WriteResultToDataBase( ) {
    // I have moved this to HandleSocketTemplateMatchResultReady so that things are done one result at at time.
    /*
	// find the current highest template match numbers in the database, then increment by one

	int database_queue_id = main_frame->current_project.database.ReturnHighestTemplateMatchID() + 1;
	int search_id =  main_frame->current_project.database.ReturnHighestTemplateMatchJobID() + 1;
	main_frame->current_project.database.Begin();


	for (int counter = 0; counter < cached_results.GetCount(); counter++)
	{
		cached_results[counter].job_id = search_id;
		main_frame->current_project.database.AddTemplateMatchingResult(database_queue_id, cached_results[counter]);
		database_queue_id++;
	}

	for (int counter = 0; counter < cached_results.GetCount(); counter++)
	{
		main_frame->current_project.database.SetActiveTemplateMatchJobForGivenImageAssetID(cached_results[counter].image_asset_id, search_id);
	}

	main_frame->current_project.database.Commit();

	match_template_results_panel->is_dirty = true;
*/
}

void MatchTemplatePanel::UpdateProgressBar( ) {
    ProgressBar->SetValue(my_job_tracker.ReturnPercentCompleted( ));
    TimeRemainingText->SetLabel(my_job_tracker.ReturnRemainingTime( ).Format("Time Remaining : %Hh:%Mm:%Ss"));
}

// deprecated - remove: Resume run functionality replaced by Queue Manager
/*
void MatchTemplatePanel::ResumeRunCheckBoxOnCheckBox(wxCommandEvent& event) {
    if ( event.IsChecked( ) ) {
        CheckForUnfinishedWork(true, true);
    }
    else {
        CheckForUnfinishedWork(false, true);
    }
}
*/

/** 
 * This may be called when the user clicks the resume run checkbox.
 * OR
 * When the header in the results panel is changed.
*/

wxArrayLong MatchTemplatePanel::CheckForUnfinishedWork(bool is_checked, bool is_from_check_box) {
    wxArrayLong unfinished_match_template_ids;
    if ( is_checked ) {
        // active group might have been overriden when resuming a run
        active_group.CopyFrom(&image_asset_panel->all_groups_list->groups[GroupComboBox->GetSelection( )]);

        int active_job_id                 = match_template_results_panel->ResultDataView->ReturnActiveJobID( );
        int images_total                  = active_group.number_of_members;
        int images_to_be_processed        = 0;
        int images_successfully_processed = 0;

        // Get a list of unfinished images by performing a left join between all
        // image assets or the image assets in the desired group and the results
        // stored for this job. The image assets that don't match are what we
        // want.

        // All images
        if ( active_group.id == -1 ) {
            // Find images that don't have results for this template match job yet
            // Uses LEFT JOIN to find IMAGE_ASSETS without corresponding TEMPLATE_MATCH_LIST entries
            unfinished_match_template_ids = main_frame->current_project.database.ReturnLongArrayFromSelectCommand(
                    wxString::Format("SELECT IMAGE_ASSETS.IMAGE_ASSET_ID, "
                                     "COMPLETED.IMAGE_ASSET_ID AS COMPLETED_IMAGE_ID "
                                     "FROM IMAGE_ASSETS "
                                     "LEFT JOIN (SELECT IMAGE_ASSET_ID FROM TEMPLATE_MATCH_LIST WHERE SEARCH_ID = %i) AS COMPLETED "
                                     "ON IMAGE_ASSETS.IMAGE_ASSET_ID = COMPLETED.IMAGE_ASSET_ID "
                                     "WHERE COMPLETED_IMAGE_ID IS NULL",
                                     active_job_id));
        }
        // An Image group
        else {
            // Find images in this group that don't have results for this template match job yet
            // Uses LEFT JOIN to find IMAGE_GROUP entries without corresponding TEMPLATE_MATCH_LIST entries
            unfinished_match_template_ids = main_frame->current_project.database.ReturnLongArrayFromSelectCommand(
                    wxString::Format("SELECT IMAGE_ASSETS.IMAGE_ASSET_ID, "
                                     "COMPLETED.IMAGE_ASSET_ID AS COMPLETED_IMAGE_ID "
                                     "FROM IMAGE_GROUP_%i AS IMAGE_ASSETS "
                                     "LEFT JOIN (SELECT IMAGE_ASSET_ID FROM TEMPLATE_MATCH_LIST WHERE SEARCH_ID = %i) AS COMPLETED "
                                     "ON IMAGE_ASSETS.IMAGE_ASSET_ID = COMPLETED.IMAGE_ASSET_ID "
                                     "WHERE COMPLETED_IMAGE_ID IS NULL",
                                     active_group.id, active_job_id));
        }
        images_to_be_processed        = unfinished_match_template_ids.GetCount( );
        images_successfully_processed = images_total - images_to_be_processed;
        no_unfinished_jobs            = (images_to_be_processed == 0);

        if ( no_unfinished_jobs ) {
            wxPrintf("No unfinished jobs.\n");
            // Only create the dialog if triggered by the checkbox, not by the header change.
            if ( is_from_check_box ) {
                wxMessageDialog* check_dialog = new wxMessageDialog(this,
                                                                    wxString::Format("There is no unfinished work for job %d.\n\nYou may select another job by clicking the header in the TM results panel.",
                                                                                     active_job_id, "Please Confirm", wxOK));
                check_dialog->ShowModal( );
            }
            // deprecated - remove: Resume run functionality replaced by Queue Manager
            // ResumeRunCheckBox->SetValue(false);
            SetInputsForPossibleReRun(false);
        }
        else {
            wxPrintf("Checking for unfinished work for job %d\n", active_job_id);
            // Only create the dialog if triggered by the checkbox, not by the header change.
            if ( is_from_check_box ) {
                wxMessageDialog* check_dialog = new wxMessageDialog(this,
                                                                    wxString::Format("Resuming work for job %d, which has %d/%d images completed.\n",
                                                                                     active_job_id, images_successfully_processed, images_total, "Please Confirm", wxOK));
                check_dialog->ShowModal( );
            }

            int job_id_to_resume = match_template_results_panel->ResultDataView->ReturnActiveJobID( );
            // This returns just one of the finished jobs, so we can get the arguments from it.
            long                    database_queue_id     = main_frame->current_project.database.GetTemplateMatchIdForGivenJobId(job_id_to_resume);
            TemplateMatchJobResults job_results_to_resume = main_frame->current_project.database.GetTemplateMatchingResultByID(database_queue_id);
            SetInputsForPossibleReRun(true, &job_results_to_resume);
        }
    }
    else {
        wxPrintf("ELSE\n");
        SetInputsForPossibleReRun(false);
    }
    return unfinished_match_template_ids;
}

// Queue functionality implementation
void MatchTemplatePanel::OnAddToQueueClick(wxCommandEvent& event) {
    // Validate that no job is currently running
    if ( ! running_job ) {
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
    // Open queue manager dialog without adding new items
    wxDialog* queue_dialog = new wxDialog(this, wxID_ANY, "Template Match Queue Manager",
                                          wxDefaultPosition, wxSize(900, 700),
                                          wxDEFAULT_DIALOG_STYLE | wxRESIZE_BORDER);

    // Create queue manager for this dialog
    TemplateMatchQueueManager* queue_manager = new TemplateMatchQueueManager(queue_dialog, this);

    // Load existing queue from database
    queue_manager->LoadQueueFromDatabase( );

    // Layout
    wxBoxSizer* dialog_sizer = new wxBoxSizer(wxVERTICAL);
    dialog_sizer->Add(queue_manager, 1, wxEXPAND | wxALL, 5);

    // Add Close button
    wxButton* close_button = new wxButton(queue_dialog, wxID_OK, "Close");
    dialog_sizer->Add(close_button, 0, wxALIGN_CENTER | wxALL, 5);

    queue_dialog->SetSizer(dialog_sizer);

    // Force layout update for proper visibility of both tables
    queue_dialog->Layout( );
    queue_dialog->Fit( ); // Auto-size dialog to fit content

    // Connect close button to destroy the dialog
    close_button->Bind(wxEVT_BUTTON, [queue_dialog](wxCommandEvent&) {
        queue_dialog->Destroy( );
    });

    // Use Show() instead of ShowModal() to keep parent controls enabled
    // This allows editing of parameters in the MatchTemplatePanel while the queue dialog is open
    queue_dialog->Show(true);
}

void MatchTemplatePanel::OnHeaderClickAddToQueue( ) {
    // Get the active job ID from the results panel
    int active_job_id = match_template_results_panel->ResultDataView->ReturnActiveJobID( );

    if ( active_job_id <= 0 ) {
        wxMessageBox("No active template match job selected. Please click on a result row first.",
                     "No Active Job", wxOK | wxICON_WARNING);
        return;
    }

    // Ask user if they want to add this job to the Queue Manager
    wxMessageDialog dialog(this,
                           wxString::Format("Add Template Match Job %d to Queue Manager?\n\n"
                                            "The job will be added to the Available Jobs table where you can "
                                            "configure parameters and move it to the execution queue.",
                                            active_job_id),
                           "Add to Queue Manager", wxYES_NO | wxICON_QUESTION);

    if ( dialog.ShowModal( ) == wxID_YES ) {
        // Check if this job is already in the queue
        // TODO: Implement queue check logic

        // Collect current GUI parameters for the job
        TemplateMatchQueueItem gui_params = CollectJobParametersFromGui( );

        // Create queue item from the active job
        TemplateMatchQueueItem queue_item;
        queue_item.database_queue_id = active_job_id;
        queue_item.search_name       = gui_params.search_name; // Use the template name from GUI
        queue_item.queue_status      = "pending";
        queue_item.queue_order       = -1; // Add to available jobs table

        // Use existing resume logic to determine if job is complete
        wxArrayLong unfinished_ids = CheckForUnfinishedWork(true, false);
        if ( unfinished_ids.IsEmpty( ) ) {
            queue_item.queue_status = "complete";
        }

        // Copy relevant parameters from GUI to queue item
        queue_item.image_group_id             = gui_params.image_group_id;
        queue_item.reference_volume_asset_id  = gui_params.reference_volume_asset_id;
        queue_item.run_profile_id             = gui_params.run_profile_id;
        queue_item.use_gpu                    = gui_params.use_gpu;
        queue_item.use_fast_fft               = gui_params.use_fast_fft;
        queue_item.symmetry                   = gui_params.symmetry;
        queue_item.pixel_size                 = gui_params.pixel_size;
        queue_item.voltage                    = gui_params.voltage;
        queue_item.spherical_aberration       = gui_params.spherical_aberration;
        queue_item.amplitude_contrast         = gui_params.amplitude_contrast;
        queue_item.defocus1                   = gui_params.defocus1;
        queue_item.defocus2                   = gui_params.defocus2;
        queue_item.defocus_angle              = gui_params.defocus_angle;
        queue_item.phase_shift                = gui_params.phase_shift;
        queue_item.low_resolution_limit       = gui_params.low_resolution_limit;
        queue_item.high_resolution_limit      = gui_params.high_resolution_limit;
        queue_item.out_of_plane_angular_step  = gui_params.out_of_plane_angular_step;
        queue_item.in_plane_angular_step      = gui_params.in_plane_angular_step;
        queue_item.defocus_search_range       = gui_params.defocus_search_range;
        queue_item.defocus_step               = gui_params.defocus_step;
        queue_item.pixel_size_search_range    = gui_params.pixel_size_search_range;
        queue_item.pixel_size_step            = gui_params.pixel_size_step;
        queue_item.refinement_threshold       = gui_params.refinement_threshold;
        queue_item.ref_box_size_in_angstroms  = gui_params.ref_box_size_in_angstroms;
        queue_item.mask_radius                = gui_params.mask_radius;
        queue_item.min_peak_radius            = gui_params.min_peak_radius;
        queue_item.xy_change_threshold        = gui_params.xy_change_threshold;
        queue_item.exclude_above_xy_threshold = gui_params.exclude_above_xy_threshold;

        // Add to queue and show queue manager
        AddJobToQueue(queue_item, true);

        wxPrintf("Added Template Match Job %d to Queue Manager\n", active_job_id);
    }
}

void MatchTemplatePanel::PopulateGuiFromQueueItem(const TemplateMatchQueueItem& item, bool for_editing) {
    // Populate GUI controls with values from the queue item

    // Store custom CLI args for later use during execution
    current_custom_cli_args = item.custom_cli_args;

    // Set the group and reference selections
    if ( GroupComboBox && item.image_group_id >= 0 ) {
        GroupComboBox->SetSelection(item.image_group_id);
        // Enable combo box if we're editing
        if ( for_editing && ! running_job ) {
            GroupComboBox->Enable(true);
        }
    }

    if ( ReferenceSelectPanel && item.reference_volume_asset_id >= 0 ) {
        ReferenceSelectPanel->SetSelection(item.reference_volume_asset_id);
        // Enable reference panel if we're editing
        if ( for_editing && ! running_job ) {
            ReferenceSelectPanel->Enable(true);
        }
    }

    // Set run profile
    if ( RunProfileComboBox && item.run_profile_id >= 0 ) {
        RunProfileComboBox->SetSelection(item.run_profile_id);
        wxPrintf("Set RunProfileComboBox to selection %d from queue item\n", item.run_profile_id);
        // Enable combo box if we're editing
        if ( for_editing && ! running_job ) {
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
    if ( running_job ) {
        wxPrintf("Cannot run queued job - another job is already running\n");
        return false;
    }

    // Store the queue job ID so we can update its status when complete
    running_queue_job_id = job.database_queue_id;

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

    // Set initial queue status and custom args
    new_job.queue_status    = "pending";
    new_job.custom_cli_args = "";
    // Also clear the member variable when collecting from GUI (not from queue)
    current_custom_cli_args = "";

    // Collect actual parameters from GUI controls
    new_job.database_queue_id         = -1; // Will be assigned when stored to database
    new_job.image_group_id            = GroupComboBox->GetSelection( );
    new_job.reference_volume_asset_id = ReferenceSelectPanel->GetSelection( );
    new_job.run_profile_id            = RunProfileComboBox->GetSelection( );

    // Get symmetry and resolution parameters
    new_job.symmetry              = SymmetryComboBox->GetValue( ).Upper( );
    new_job.high_resolution_limit = HighResolutionLimitNumericCtrl->ReturnValue( );
    new_job.low_resolution_limit  = 300.0f; // Currently hardcoded in template matching

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

    // Get CTF parameters from the first image in the selected group
    AssetGroup temp_group;
    temp_group.CopyFrom(&image_asset_panel->all_groups_list->groups[new_job.image_group_id]);
    if ( temp_group.number_of_members > 0 ) {
        ImageAsset* first_image = image_asset_panel->ReturnAssetPointer(temp_group.members[0]);
        new_job.pixel_size      = first_image->pixel_size;

        // Get CTF parameters from database
        double voltage_kV, spherical_aberration_mm, amplitude_contrast;
        double defocus1, defocus2, defocus_angle, phase_shift, iciness;
        main_frame->current_project.database.GetCTFParameters(first_image->ctf_estimation_id,
                                                              voltage_kV, spherical_aberration_mm, amplitude_contrast,
                                                              defocus1, defocus2, defocus_angle, phase_shift, iciness);

        new_job.voltage              = voltage_kV;
        new_job.spherical_aberration = spherical_aberration_mm;
        new_job.amplitude_contrast   = amplitude_contrast;
        new_job.defocus1             = defocus1;
        new_job.defocus2             = defocus2;
        new_job.defocus_angle        = defocus_angle;
        new_job.phase_shift          = phase_shift;
    }
    else {
        // Fallback values if no images in group
        new_job.pixel_size           = 1.0;
        new_job.voltage              = 300.0;
        new_job.spherical_aberration = 2.7;
        new_job.amplitude_contrast   = 0.07;
        new_job.defocus1             = 10000.0;
        new_job.defocus2             = 10000.0;
        new_job.defocus_angle        = 0.0;
        new_job.phase_shift          = 0.0;
    }

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
    wxPrintf("\n=== DEBUG: Run Profile Information ===\n");
    wxPrintf("Run Profile ID (stored in queue): %d\n", new_job.run_profile_id);
    if ( new_job.run_profile_id >= 0 && new_job.run_profile_id < RunProfileComboBox->GetCount( ) ) {
        wxString run_profile_name = RunProfileComboBox->GetString(new_job.run_profile_id);
        wxPrintf("Run Profile Name: %s\n", run_profile_name);
        wxPrintf("Run Profile ComboBox Count: %d\n", RunProfileComboBox->GetCount( ));
    }
    else {
        wxPrintf("No run profile selected or invalid selection\n");
        wxPrintf("Run Profile ComboBox Count: %d\n", RunProfileComboBox->GetCount( ));
    }
    wxPrintf("=== END DEBUG ===\n\n");

    return new_job;
}

long MatchTemplatePanel::AddJobToQueue(const TemplateMatchQueueItem& job, bool show_dialog) {
    if ( show_dialog ) {
        // Show the queue manager with the new job
        wxDialog* queue_dialog = new wxDialog(this, wxID_ANY, "Template Match Queue Manager",
                                              wxDefaultPosition, wxSize(900, 700),
                                              wxDEFAULT_DIALOG_STYLE | wxRESIZE_BORDER);

        // Create queue manager for this dialog
        TemplateMatchQueueManager* queue_manager = new TemplateMatchQueueManager(queue_dialog, this);

        // Load existing queue from database first
        queue_manager->LoadQueueFromDatabase( );

        // Add the new job to the queue
        queue_manager->AddToExecutionQueue(job);

        // Job added successfully - no popup needed

        // Layout
        wxBoxSizer* dialog_sizer = new wxBoxSizer(wxVERTICAL);
        dialog_sizer->Add(queue_manager, 1, wxEXPAND | wxALL, 5);

        wxButton* close_button = new wxButton(queue_dialog, wxID_OK, "Close");
        dialog_sizer->Add(close_button, 0, wxALIGN_CENTER | wxALL, 5);

        queue_dialog->SetSizer(dialog_sizer);

        // Force layout update for proper visibility of both tables
        queue_dialog->Layout( );
        queue_dialog->Fit( ); // Auto-size dialog to fit content

        // Connect close button to destroy the dialog
        close_button->Bind(wxEVT_BUTTON, [queue_dialog](wxCommandEvent&) {
            queue_dialog->Destroy( );
        });

        // Use Show() instead of ShowModal() to keep parent controls enabled
        // This allows editing of parameters in the MatchTemplatePanel while the queue dialog is open
        queue_dialog->Show(true);

        // Return -1 for dialog mode since we don't track the specific queue ID
        return -1;
    }
    else {
        // Add to queue without dialog - use database directly
        if ( main_frame && main_frame->current_project.is_open ) {
            long queue_id = main_frame->current_project.database.AddToTemplateMatchQueue(
                    job.search_name, job.image_group_id, job.reference_volume_asset_id, job.run_profile_id,
                    job.use_gpu, job.use_fast_fft, job.symmetry,
                    job.pixel_size, job.voltage, job.spherical_aberration, job.amplitude_contrast,
                    job.defocus1, job.defocus2, job.defocus_angle, job.phase_shift,
                    job.low_resolution_limit, job.high_resolution_limit,
                    job.out_of_plane_angular_step, job.in_plane_angular_step,
                    job.defocus_search_range, job.defocus_step,
                    job.pixel_size_search_range, job.pixel_size_step,
                    job.refinement_threshold, job.ref_box_size_in_angstroms,
                    job.mask_radius, job.min_peak_radius,
                    job.xy_change_threshold, job.exclude_above_xy_threshold,
                    job.custom_cli_args);

            if ( queue_id > 0 ) {
                // Create a copy with the assigned queue ID
                TemplateMatchQueueItem job_with_id = job;
                job_with_id.database_queue_id      = queue_id;
                job_with_id.queue_status           = "pending";

                // Note: QueueManager instances are dialog-scoped, so no persistent queue needed here
                return queue_id;
            }
            else {
                MyAssertTrue(false, "Failed to add search to database queue - AddToTemplateMatchQueue returned %ld", queue_id);
                return -1; // This should not happen for new searches
            }
        }
        else {
            MyAssertTrue(false, "Cannot add search to queue - database not open or main_frame invalid");
            return -1;
        }
    }
}

bool MatchTemplatePanel::SetupJobFromQueueItem(const TemplateMatchQueueItem& job) {
    // Freeze GUI updates in queue manager to prevent interference during setup
    if ( queue_completion_callback ) {
        queue_completion_callback->SetGuiUpdateFrozen(true);
    }

    // First populate GUI with the job parameters
    PopulateGuiFromQueueItem(job);

    // Debug prints to check job parameters
    wxPrintf("\n=== DEBUG: SetupJobFromQueueItem Parameters ===\n");
    wxPrintf("Search Name: %s\n", job.search_name);
    wxPrintf("Image Group ID: %d\n", job.image_group_id);
    wxPrintf("Reference Volume Asset ID: %d\n", job.reference_volume_asset_id);
    wxPrintf("Run Profile ID: %d\n", job.run_profile_id);
    wxPrintf("Use GPU: %s\n", job.use_gpu ? "true" : "false");
    wxPrintf("Use Fast FFT: %s\n", job.use_fast_fft ? "true" : "false");
    wxPrintf("Symmetry: %s\n", job.symmetry);
    wxPrintf("Pixel Size: %.4f\n", job.pixel_size);
    wxPrintf("Voltage: %.2f\n", job.voltage);
    wxPrintf("Spherical Aberration: %.2f\n", job.spherical_aberration);
    wxPrintf("Amplitude Contrast: %.3f\n", job.amplitude_contrast);
    wxPrintf("High Resolution Limit: %.2f\n", job.high_resolution_limit);
    wxPrintf("Low Resolution Limit: %.2f\n", job.low_resolution_limit);
    wxPrintf("Out of Plane Angular Step: %.2f\n", job.out_of_plane_angular_step);
    wxPrintf("In Plane Angular Step: %.2f\n", job.in_plane_angular_step);
    wxPrintf("Defocus Search Range: %.2f\n", job.defocus_search_range);
    wxPrintf("Defocus Step: %.2f\n", job.defocus_step);
    wxPrintf("Pixel Size Search Range: %.4f\n", job.pixel_size_search_range);
    wxPrintf("Pixel Size Step: %.4f\n", job.pixel_size_step);
    wxPrintf("Min Peak Radius: %.2f\n", job.min_peak_radius);
    wxPrintf("Reference Box Size (Angstroms): %.2f\n", job.ref_box_size_in_angstroms);
    wxPrintf("Mask Radius: %.2f\n", job.mask_radius);
    wxPrintf("=== END DEBUG ===\n\n");

    // Now run the existing setup logic that was in StartEstimationClick
    // This mirrors the logic from StartEstimationClick but uses the job parameters directly

    active_group.CopyFrom(&image_asset_panel->all_groups_list->groups[job.image_group_id]);

    // Check if this is a resume job (for now, assume not resuming from queue)
    bool resume = false;

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

    current_volume         = volume_asset_panel->ReturnAssetPointer(job.reference_volume_asset_id);
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

    wxPrintf("\n\nWanted symmetry %s, Defocus Range %3.3f, Defocus Step %3.3f\n", wanted_symmetry, defocus_search_range, defocus_step);

    RunProfile active_refinement_run_profile = run_profiles_panel->run_profile_manager.run_profiles[RunProfileComboBox->GetSelection( )];

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
    if (!current_custom_cli_args.IsEmpty()) {
        if (!current_custom_cli_args.StartsWith(" ")) {
            executable_name += " ";
        }
        executable_name += current_custom_cli_args;
        wxPrintf("Appending custom CLI args to executable: %s\n", current_custom_cli_args);
    }

    current_job_package.Reset(active_refinement_run_profile, executable_name, number_of_jobs);

    expected_number_of_results = 0;
    number_of_received_results = 0;

    // Set up progress dialog and job preparation
    OneSecondProgressDialog* my_progress_dialog = new OneSecondProgressDialog("Preparing Job", "Preparing Job...", active_group.number_of_members, this, wxPD_REMAINING_TIME | wxPD_AUTO_HIDE | wxPD_APP_MODAL);

    TemplateMatchJobResults temp_result;
    temp_result.input_job_id               = -1;
    temp_result.job_type                   = cistem::job_type::template_match_full_search;
    temp_result.mask_radius                = 0.0f;
    temp_result.min_peak_radius            = min_peak_radius;
    temp_result.exclude_above_xy_threshold = false;
    temp_result.xy_change_threshold        = 0.0f;

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

        wxPrintf("For Image %li\nThere are %i search positions\nThere are %i jobs per image\n", active_group.members[image_counter], current_image_euler_search->number_of_search_positions, number_of_jobs_per_image_in_gui);
        wxPrintf("Calculating %i correlation maps\n", current_image_euler_search->number_of_search_positions * number_of_rotations * number_of_defocus_positions * number_of_pixel_size_positions);

        expected_number_of_results += current_image_euler_search->number_of_search_positions * number_of_rotations * number_of_defocus_positions * number_of_pixel_size_positions;
        orientations_per_process = float(current_image_euler_search->number_of_search_positions) / float(number_of_jobs_per_image_in_gui);
        if ( orientations_per_process < 1 )
            orientations_per_process = 1;

        int number_of_previous_template_matches = main_frame->current_project.database.ReturnNumberOfPreviousTemplateMatchesByAssetID(current_image->asset_id);
        main_frame->current_project.database.GetCTFParameters(current_image->ctf_estimation_id, voltage_kV, spherical_aberration_mm, amplitude_contrast, defocus1, defocus2, defocus_angle, phase_shift, iciness);

        // Generate output filenames
        wxString mip_output_file = main_frame->current_project.template_matching_asset_directory.GetFullPath( );
        mip_output_file += wxString::Format("/%s_mip_%i_%i.mrc", current_image->filename.GetName( ), current_image->asset_id, number_of_previous_template_matches);

        wxString best_psi_output_file = main_frame->current_project.template_matching_asset_directory.GetFullPath( );
        best_psi_output_file += wxString::Format("/%s_psi_%i_%i.mrc", current_image->filename.GetName( ), current_image->asset_id, number_of_previous_template_matches);

        wxString best_theta_output_file = main_frame->current_project.template_matching_asset_directory.GetFullPath( );
        best_theta_output_file += wxString::Format("/%s_theta_%i_%i.mrc", current_image->filename.GetName( ), current_image->asset_id, number_of_previous_template_matches);

        wxString best_phi_output_file = main_frame->current_project.template_matching_asset_directory.GetFullPath( );
        best_phi_output_file += wxString::Format("/%s_phi_%i_%i.mrc", current_image->filename.GetName( ), current_image->asset_id, number_of_previous_template_matches);

        wxString best_defocus_output_file = main_frame->current_project.template_matching_asset_directory.GetFullPath( );
        best_defocus_output_file += wxString::Format("/%s_defocus_%i_%i.mrc", current_image->filename.GetName( ), current_image->asset_id, number_of_previous_template_matches);

        wxString best_pixel_size_output_file = main_frame->current_project.template_matching_asset_directory.GetFullPath( );
        best_pixel_size_output_file += wxString::Format("/%s_pixel_size_%i_%i.mrc", current_image->filename.GetName( ), current_image->asset_id, number_of_previous_template_matches);

        wxString scaled_mip_output_file = main_frame->current_project.template_matching_asset_directory.GetFullPath( );
        scaled_mip_output_file += wxString::Format("/%s_scaled_mip_%i_%i.mrc", current_image->filename.GetName( ), current_image->asset_id, number_of_previous_template_matches);

        wxString output_histogram_file = main_frame->current_project.template_matching_asset_directory.GetFullPath( );
        output_histogram_file += wxString::Format("/%s_histogram_%i_%i.txt", current_image->filename.GetName( ), current_image->asset_id, number_of_previous_template_matches);

        wxString output_result_file = main_frame->current_project.template_matching_asset_directory.GetFullPath( );
        output_result_file += wxString::Format("/%s_plotted_result_%i_%i.mrc", current_image->filename.GetName( ), current_image->asset_id, number_of_previous_template_matches);

        wxString correlation_avg_output_file = main_frame->current_project.template_matching_asset_directory.GetFullPath( );
        correlation_avg_output_file += wxString::Format("/%s_avg_%i_%i.mrc", current_image->filename.GetName( ), current_image->asset_id, number_of_previous_template_matches);

        wxString correlation_std_output_file = main_frame->current_project.template_matching_asset_directory.GetFullPath( );
        correlation_std_output_file += wxString::Format("/%s_std_%i_%i.mrc", current_image->filename.GetName( ), current_image->asset_id, number_of_previous_template_matches);

        current_orientation_counter = 0;

        wxString input_search_image   = current_image->filename.GetFullPath( );
        wxString input_reconstruction = current_volume->filename.GetFullPath( );
        float    pixel_size           = current_image->pixel_size;

        input_image_filenames.Add(input_search_image);

        float low_resolution_limit = 300.0f; // FIXME set this somewhere that is not buried in the code!

        temp_result.image_asset_id                  = current_image->asset_id;
        temp_result.job_name                        = wxString::Format("Template: %s", current_volume->filename.GetName( ));
        temp_result.ref_volume_asset_id             = current_volume->asset_id;
        wxDateTime now                              = wxDateTime::Now( );
        temp_result.datetime_of_run                 = (long int)now.GetAsDOS( );
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
        temp_result.mip_filename                    = mip_output_file;
        temp_result.scaled_mip_filename             = scaled_mip_output_file;
        temp_result.psi_filename                    = best_psi_output_file;
        temp_result.theta_filename                  = best_theta_output_file;
        temp_result.phi_filename                    = best_phi_output_file;
        temp_result.defocus_filename                = best_defocus_output_file;
        temp_result.pixel_size_filename             = best_pixel_size_output_file;
        temp_result.histogram_filename              = output_histogram_file;
        temp_result.projection_result_filename      = output_result_file;
        temp_result.avg_filename                    = correlation_avg_output_file;
        temp_result.std_filename                    = correlation_std_output_file;

        cached_results.Add(temp_result);

        // Create individual jobs for this image
        for ( job_counter = 0; job_counter < number_of_jobs_per_image_in_gui; job_counter++ ) {
            int   best_parameters_to_keep = 1;
            float padding                 = 1;
            bool  ctf_refinement          = false;
            float mask_radius_search      = 0.0f;

            wxPrintf("\n\tFor image %i, current_orientation_counter is %f\n", image_number_for_gui, current_orientation_counter);
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

    // Get ID's from database for writing results as they come in
    database_queue_id = main_frame->current_project.database.ReturnHighestTemplateMatchID( ) + 1;
    search_id         = main_frame->current_project.database.ReturnHighestTemplateMatchJobID( ) + 1;

    // Update the queue item with the actual SEARCH_ID for n/N tracking
    if ( running_queue_job_id > 0 ) {
        if ( queue_completion_callback ) {
            // Queue manager is present - let it update the search ID
            queue_completion_callback->UpdateSearchIdForQueueItem(running_queue_job_id, search_id);
        }
        else {
            // No queue manager (StartEstimationClick path) - update database directly
            // This is critical for status computation since status is derived from completion data
            main_frame->current_project.database.UpdateSearchIdInQueueTable(running_queue_job_id, search_id);
            wxPrintf("Direct database update: Linked queue ID %ld to search ID %d\n", running_queue_job_id, search_id);
        }
    }

    // Unfreeze GUI updates in queue manager now that setup is complete
    if ( queue_completion_callback ) {
        queue_completion_callback->SetGuiUpdateFrozen(false);
    }

    return true;
}

bool MatchTemplatePanel::ExecuteCurrentJob( ) {
    // Get the run profile to use - the GUI should be populated correctly by PopulateGuiFromQueueItem
    int run_profile_to_use = RunProfileComboBox->GetSelection( );

    // Debug print to verify run profile is correctly set
    if ( running_queue_job_id > 0 ) {
        wxPrintf("Executing queue job %ld with run profile selection %d\n",
                 running_queue_job_id, run_profile_to_use);
    }
    else {
        wxPrintf("Executing GUI job with run profile selection %d\n", run_profile_to_use);
    }

    // Launch the job controller
    my_job_id = main_frame->job_controller.AddJob(this,
                                                  run_profiles_panel->run_profile_manager.run_profiles[run_profile_to_use].manager_command,
                                                  run_profiles_panel->run_profile_manager.run_profiles[run_profile_to_use].gui_address);

    if ( my_job_id != -1 ) {
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
        running_job = true;
        return true;
    }

    return false;
}

bool MatchTemplatePanel::ExecuteJob(const TemplateMatchQueueItem* queue_item) {
    // If queue_item is provided, we're being called from the queue and need to setup the job parameters.
    // If queue_item is null, we're being called from StartEstimationClick and need to setup from GUI.
    if ( queue_item ) {
        // Validate job parameters before execution
        MyDebugAssertTrue(queue_item->database_queue_id >= 0, "Cannot execute job with invalid database_queue_id: %ld", queue_item->database_queue_id);
        MyDebugAssertTrue(queue_item->queue_status == "pending" || queue_item->queue_status == "failed" || queue_item->queue_status == "partial",
                          "Cannot execute job with status '%s', must be 'pending', 'failed', or 'partial'", queue_item->queue_status.mb_str( ).data( ));
        MyDebugAssertFalse(queue_item->search_name.IsEmpty( ), "Cannot execute search with empty search_name");
        MyDebugAssertTrue(queue_item->image_group_id >= 0, "Cannot execute job with invalid image_group_id: %d", queue_item->image_group_id);
        MyDebugAssertTrue(queue_item->reference_volume_asset_id >= 0, "Cannot execute job with invalid reference_volume_asset_id: %d", queue_item->reference_volume_asset_id);

        // Check if another job is already running
        if ( running_job ) {
            wxMessageBox("A job is already running. Please wait for it to complete.",
                         "Job Running", wxOK | wxICON_WARNING);
            return false;
        }

        // Store the queue job ID so we can update its status when complete
        running_queue_job_id = queue_item->database_queue_id;

        // Setup job from queue item
        wxPrintf("Setting up job %ld from queue item...\n", queue_item->database_queue_id);
        bool setup_success = SetupJobFromQueueItem(*queue_item);

        if ( ! setup_success ) {
            wxPrintf("Failed to setup job %ld\n", queue_item->database_queue_id);
            running_queue_job_id = -1;
            return false;
        }
    }
    else {
        // GUI job - need to setup from current GUI state first
        TemplateMatchQueueItem gui_job = CollectJobParametersFromGui( );
        if ( ! SetupJobFromQueueItem(gui_job) ) {
            wxPrintf("Failed to setup GUI job\n");
            return false;
        }
    }

    // Execute the job (current ExecuteCurrentJob logic)
    return ExecuteCurrentJob( );
}

void MatchTemplatePanel::SetQueueCompletionCallback(TemplateMatchQueueManager* queue_manager) {
    queue_completion_callback = queue_manager;
    wxPrintf("Queue completion callback set for queue manager %p\n", queue_manager);
}

void MatchTemplatePanel::ClearQueueCompletionCallback( ) {
    wxPrintf("Queue completion callback cleared\n");
    queue_completion_callback = nullptr;
}
