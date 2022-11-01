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

AutoRefine3DPanelSpa::AutoRefine3DPanelSpa(wxWindow* parent)
    : AutoRefine3DPanelParent(parent) {

    buffered_results = NULL;

    // Fill combo box..

    //FillGroupComboBox();

    my_job_id   = -1;
    running_job = false;

    //	group_combo_is_dirty = false;
    //	run_profiles_are_dirty = false;

    SetInfo( );
    //	FillGroupComboBox();
    //	FillRunProfileComboBox();

    wxSize input_size = InputSizer->GetMinSize( );
    input_size.x += wxSystemSettings::GetMetric(wxSYS_VSCROLL_X);
    input_size.y = -1;
    ExpertPanel->SetMinSize(input_size);
    ExpertPanel->SetSize(input_size);

#ifndef SHOW_CISTEM_GPU_OPTIONS
    use_gpu_checkboxAR3D->Show(false);
#endif

    // set values //

    /*
	AmplitudeContrastNumericCtrl->SetMinMaxValue(0.0f, 1.0f);
	MinResNumericCtrl->SetMinMaxValue(0.0f, 50.0f);
	MaxResNumericCtrl->SetMinMaxValue(0.0f, 50.0f);
	DefocusStepNumericCtrl->SetMinMaxValue(1.0f, FLT_MAX);
	ToleratedAstigmatismNumericCtrl->SetMinMaxValue(0.0f, FLT_MAX);
	MinPhaseShiftNumericCtrl->SetMinMaxValue(-3.15, 3.15);
	MaxPhaseShiftNumericCtrl->SetMinMaxValue(-3.15, 3.15);
	PhaseShiftStepNumericCtrl->SetMinMaxValue(0.001, 3.15);

	result_bitmap.Create(1,1, 24);
	time_of_last_result_update = time(NULL);*/

    refinement_package_combo_is_dirty = false;
    run_profiles_are_dirty            = false;
    //	input_params_combo_is_dirty = false;
    selected_refinement_package = -1;

    RefinementPackageSelectPanel->AssetComboBox->Bind(wxEVT_COMMAND_COMBOBOX_SELECTED, &AutoRefine3DPanelSpa::OnRefinementPackageComboBox, this);
    Bind(wxEVT_AUTOMASKERTHREAD_COMPLETED, &AutoRefine3DPanelSpa::OnMaskerThreadComplete, this);
    Bind(wxEVT_MULTIPLY3DMASKTHREAD_COMPLETED, &AutoRefine3DPanelSpa::OnMaskerThreadComplete, this);
    Bind(RETURN_PROCESSED_IMAGE_EVT, &AutoRefine3DPanelSpa::OnOrthThreadComplete, this);

    my_refinement_manager.SetParent(this);

    FillRefinementPackagesComboBox( );

    long time_of_last_result_update;

    auto_mask_value = (main_frame->GetCurrentWorkflow( ) == cistem::workflow::single_particle) ? true : false;

    active_orth_thread_id = -1;
    active_mask_thread_id = -1;
    next_thread_id        = 1;
}

void AutoRefine3DPanelSpa::Reset( ) {
    ProgressBar->SetValue(0);
    TimeRemainingText->SetLabel("Time Remaining : ???h:??m:??s");
    CancelAlignmentButton->Show(true);
    FinishButton->Show(false);

    InputParamsPanel->Show(true);
    ProgressPanel->Show(false);
    StartPanel->Show(true);
    OutputTextPanel->Show(false);
    output_textctrl->Clear( );
    ShowRefinementResultsPanel->Show(false);
    ShowRefinementResultsPanel->Clear( );
    InfoPanel->Show(true);

    UseMaskCheckBox->SetValue(false);

    ExpertToggleButton->SetValue(false);
    ExpertPanel->Show(false);

    RefinementPackageSelectPanel->Clear( );
    ReferenceSelectPanel->Clear( );
    RefinementRunProfileComboBox->Clear( );
    ReconstructionRunProfileComboBox->Clear( );

    if ( running_job == true ) {
        main_frame->job_controller.KillJob(my_job_id);

        active_mask_thread_id = -1;
        active_orth_thread_id = -1;

        running_job = false;
    }

    Layout( );

    if ( my_refinement_manager.output_refinement != NULL ) {
        delete my_refinement_manager.output_refinement;
        my_refinement_manager.output_refinement = NULL;
    }

    SetDefaults( );
    global_delete_autorefine3d_scratch( );
}

void AutoRefine3DPanelSpa::SetInfo( ) {

    std::cerr << "AutoRefine3DPanelSpa::SetInfo" << std::endl;
    wxLogNull* suppress_png_warnings = new wxLogNull;
#include "icons/niko_picture1.cpp"
    wxBitmap niko_picture1_bmp = wxBITMAP_PNG_FROM_DATA(niko_picture1);

#include "icons/niko_picture2.cpp"
    wxBitmap niko_picture2_bmp = wxBITMAP_PNG_FROM_DATA(niko_picture2);
    delete suppress_png_warnings;

    InfoText->GetCaret( )->Hide( );

    InfoText->BeginSuppressUndo( );
    InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
    InfoText->BeginBold( );
    InfoText->BeginUnderline( );
    InfoText->BeginFontSize(14);
    InfoText->WriteText(wxT("3D Auto Refinement"));
    InfoText->EndFontSize( );
    InfoText->EndBold( );
    InfoText->EndUnderline( );
    InfoText->Newline( );
    InfoText->Newline( );
    InfoText->EndAlignment( );

    InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
    if ( main_frame->GetCurrentWorkflow( ) == cistem::workflow::single_particle ) {
        InfoText->WriteText(wxT("This panel allows users to refine a 3D reconstruction to high resolution using Frealign (Grigorieff, 2016) without the need to set many of the parameters that are required for manual refinement (see Manual Refine panel). In the simplest case, all that is required is the specification of a refinement package (set up under Assets), a starting reference (for example, a reconstruction obtained from the ab-initio procedure) and an initial resolution limit used in the refinement. The resolution should start low, for at 30 Å, to remove potential bias in the starting reference. However, for particles that are close to spherical, such as apoferritin, a higher resolution should be specified, between 8 and 12 Å (see Expert Options).  If the starting reference contains errors, the refinement may finish before converging to the correct answer.  This can often be solved by running another auto-refinement beginning with the result of the previous refinement.  In some cases multiple rounds may be needed to reach full convergence."));
    }
    else {
        InfoText->WriteText(wxT("Use a priori knowldege to get somewhere fast."));
    }
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
    InfoText->WriteText(wxT("Starting Reference : "));
    InfoText->EndBold( );
    InfoText->WriteText(wxT("The initial 3D reconstruction used to align particles against. This should be of reasonable quality to ensure successful refinement."));
    InfoText->Newline( );
    InfoText->BeginBold( );
    InfoText->WriteText(wxT("Initial Res. Limit (Å) : "));
    InfoText->EndBold( );
    InfoText->WriteText(wxT("The starting resolution limit used to align particles against the starting reference. In most cases, this should specify a relatively low resolution to remove potential bias in the starting reference."));
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

    InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
    InfoText->BeginUnderline( );
    InfoText->WriteText(wxT("General Refinement"));
    InfoText->EndUnderline( );
    InfoText->Newline( );
    InfoText->Newline( );
    InfoText->EndAlignment( );

    InfoText->BeginBold( );
    InfoText->WriteText(wxT("Low/High-Resolution Limit (Å) : "));
    InfoText->EndBold( );
    InfoText->WriteText(wxT("The data used for refinement is usually bandpass-limited to exclude spurious low-resolution features in the particle background (set by the low-resolution limit) and high-resolution noise (set by the high-resolution limit). It is good practice to set the low-resolution limit to 2.5x the approximate particle mask radius. The high-resolution limit should remain significantly below the resolution of the reference used for refinement to enable unbiased resolution estimation using the Fourier Shell Correlation curve."));
    InfoText->Newline( );

    InfoText->BeginBold( );
    InfoText->WriteText(wxT("Inner/Outer Mask Radius (Å) : "));
    InfoText->EndBold( );
    InfoText->WriteText(wxT("Radii describing a spherical mask with an inner and outer radius that will be applied to the final reconstruction and to the half reconstructions to calculate Fourier Shell Correlation curve. The inner radius is normally set to 0.0 but can assume non-zero values to remove density inside a particle if it represents largely disordered features, such as the genomic RNA or DNA of a virus."));
    InfoText->Newline( );
    InfoText->Newline( );

    InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
    InfoText->BeginUnderline( );
    InfoText->WriteText(wxT("Global Search"));
    InfoText->EndUnderline( );
    InfoText->Newline( );
    InfoText->Newline( );
    InfoText->EndAlignment( );

    InfoText->BeginBold( );
    InfoText->WriteText(wxT("Global Mask Radius (Å) : "));
    InfoText->EndBold( );
    InfoText->WriteText(wxT("The radius describing the area within the boxed-out particles that contains the particles. This radius is usually larger than the particle radius to account for particles that are not perfectly centered. The best value will depend on the way the particles were picked."));
    InfoText->Newline( );

    InfoText->BeginBold( );
    InfoText->WriteText(wxT("Number of Results to Refine : "));
    InfoText->EndBold( );
    InfoText->WriteText(wxT("For a global search, an angular grid search is performed and the alignment parameters for the N best matching projections are then refined further in a local refinement. Only the set of parameters yielding the best score (correlation coefficient) is kept. Increasing N will increase the chances of finding the correct particle orientations but will slow down the search. A value of 20 is recommended."));
    InfoText->Newline( );

    InfoText->BeginBold( );
    InfoText->WriteText(wxT("Search Range in X/Y (Å) : "));
    InfoText->EndBold( );
    InfoText->WriteText(wxT("The global search can be limited in the X and Y directions (measured from the box center) to ensure that only particles close to the box center are found. This is useful when the particle density is high and particles end up close to each other. In this case, it is usually still possible to align all particles in a cluster of particles (assuming they do not significantly overlap). The values provided here for the search range should be set to exclude the possibility that the same particle is selected twice and counted as two different particles."));
    InfoText->Newline( );

    InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
    InfoText->BeginUnderline( );
    InfoText->WriteText(wxT("Reconstruction"));
    InfoText->EndUnderline( );
    InfoText->Newline( );
    InfoText->Newline( );
    InfoText->EndAlignment( );

    InfoText->BeginBold( );
    InfoText->WriteText(wxT("Autocrop Images? "));
    InfoText->EndBold( );
    InfoText->WriteText(wxT("Should the particle images be cropped to a minimum size determined by the mask radius to accelerate 3D reconstruction? This is usually not recommended as it increases interpolation artifacts."));
    InfoText->Newline( );

    InfoText->BeginBold( );
    InfoText->WriteText(wxT("Apply Likelihood Blurring? "));
    InfoText->EndBold( );
    InfoText->WriteText(wxT("Should the reconstructions be blurred by inserting each particle image at multiple orientations, weighted by a likelihood function? Enable this option if the ab-initio procedure appears to suffer from over-fitting and the appearance of spurious high-resolution features."));
    InfoText->Newline( );

    InfoText->BeginBold( );
    InfoText->WriteText(wxT("Smoothing Factor : "));
    InfoText->EndBold( );
    InfoText->WriteText(wxT("A factor that reduces the range of likelihoods used for blurring. A smaller number leads to more blurring. The user should try values between 0.1 and 1."));
    InfoText->Newline( );

    InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
    InfoText->BeginUnderline( );
    InfoText->WriteText(wxT("Masking"));
    InfoText->EndUnderline( );
    InfoText->Newline( );
    InfoText->Newline( );
    InfoText->EndAlignment( );

    InfoText->BeginBold( );
    InfoText->WriteText(wxT("Use Auto-Masking? "));
    InfoText->EndBold( );
    InfoText->WriteText(wxT("Should the 3D reconstructions be masked? Masking can suppress spurious density features that could be amplified during the iterative refinement. Masking should only be disabled if it appears to interfere with the reconstruction process."));
    InfoText->Newline( );
    InfoText->Newline( );

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
    InfoText->WriteText(wxT("Grigorieff, N.,"));
    InfoText->EndBold( );
    InfoText->WriteText(wxT(" 2016. Frealign: An exploratory tool for single-particle cryo-EM. Methods Enzymol. 579, 191-226. "));
    InfoText->BeginURL("http://dx.doi.org/10.1016/bs.mie.2016.04.013");
    InfoText->BeginUnderline( );
    InfoText->BeginTextColour(*wxBLUE);
    InfoText->WriteText(wxT("doi:10.1016/bs.mie.2016.04.013"));
    InfoText->EndURL( );
    InfoText->EndTextColour( );
    InfoText->EndUnderline( );
    InfoText->EndAlignment( );
    InfoText->Newline( );
    InfoText->Newline( );

    InfoText->Layout( );
}

void AutoRefine3DPanelSpa::OnInfoURL(wxTextUrlEvent& event) {
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

void AutoRefine3DPanelSpa::ResetAllDefaultsClick(wxCommandEvent& event) {
    // TODO : should probably check that the user hasn't changed the defaults yet in the future
    SetDefaults( );
}

void AutoRefine3DPanelSpa::SetDefaults( ) {
    if ( RefinementPackageSelectPanel->GetCount( ) > 0 ) {
        float calculated_high_resolution_cutoff;
        float local_mask_radius;
        float global_mask_radius;
        float search_range;

        ExpertPanel->Freeze( );

        calculated_high_resolution_cutoff = 20.0;

        local_mask_radius  = refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageSelectPanel->GetSelection( )).estimated_particle_size_in_angstroms * 0.65;
        global_mask_radius = refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageSelectPanel->GetSelection( )).estimated_particle_size_in_angstroms * 0.8;

        search_range = refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageSelectPanel->GetSelection( )).estimated_particle_size_in_angstroms * 0.15;

        // Set the values..

        float low_res_limit = refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageSelectPanel->GetSelection( )).estimated_particle_size_in_angstroms * 1.5;
        if ( low_res_limit > 300.00 )
            low_res_limit = 300.00;

        LowResolutionLimitTextCtrl->SetValue(wxString::Format("%.2f", low_res_limit));
        HighResolutionLimitTextCtrl->SetValue(wxString::Format("%.2f", calculated_high_resolution_cutoff));
        MaskRadiusTextCtrl->SetValue(wxString::Format("%.2f", local_mask_radius));

        GlobalMaskRadiusTextCtrl->SetValue(wxString::Format("%.2f", global_mask_radius));
        NumberToRefineSpinCtrl->SetValue(20);
        SearchRangeXTextCtrl->SetValue(wxString::Format("%.2f", search_range));
        SearchRangeYTextCtrl->SetValue(wxString::Format("%.2f", search_range));

        InnerMaskRadiusTextCtrl->SetValue("0.00");
        TargetResolutionTextCtrl->SetValue("0.00");

        AutoCropYesRadioButton->SetValue(false);
        AutoCropNoRadioButton->SetValue(true);

        ApplyBlurringNoRadioButton->SetValue(true);
        ApplyBlurringYesRadioButton->SetValue(false);
        SmoothingFactorTextCtrl->SetValue("1.00");

        AutoCenterYesRadioButton->SetValue(true);
        AutoCenterNoRadioButton->SetValue(false);

        AutoMaskYesRadioButton->SetValue(true);
        AutoMaskNoRadioButton->SetValue(false);

        MaskEdgeTextCtrl->ChangeValueFloat(10.00);
        MaskWeightTextCtrl->ChangeValueFloat(0.00);
        LowPassMaskYesRadio->SetValue(false);
        LowPassMaskNoRadio->SetValue(true);
        MaskFilterResolutionText->ChangeValueFloat(20.00);

#ifdef SHOW_CISTEM_GPU_OPTIONS
        use_gpu_checkboxAR3D->SetValue(true);
#else
        use_gpu_checkboxAR3D->SetValue(false); // Already disabled, but also set to un-ticked for visual consistency.
#endif

        ExpertPanel->Thaw( );
    }
}

void AutoRefine3DPanelSpa::OnUpdateUI(wxUpdateUIEvent& event) {

    // using namespace cistem::workflow;
    // cistem::workflow::Enum my_workflow = main_frame->GetCurrentWorkflow( );

    // are there enough members in the selected group.
    if ( main_frame->current_project.is_open ) {
        // (my_workflow == single_particle) ? std::cout << "OnUpdateUI<cistem::workflow::spa>\n" : std::cout << "OnUpdateUI<cistem::workflow::pharma>\n";

        if ( running_job ) {
            ReferenceSelectPanel->Enable(false);
            RefinementPackageSelectPanel->Enable(false);
            ExpertToggleButton->Enable(false);
            InitialResLimitStaticText->Enable(false);
            HighResolutionLimitTextCtrl->Enable(false);
            UseMaskCheckBox->Enable(false);
            MaskSelectPanel->Enable(false);

            if ( ExpertPanel->IsShown( ) == true ) {
                ExpertToggleButton->SetValue(false);
                ExpertPanel->Show(false);
                Layout( );
            }
        }
        else {
            if ( ReferenceSelectPanel->GetCount( ) > 0 )
                ReferenceSelectPanel->Enable(true);
            else
                ReferenceSelectPanel->Enable(false);

#ifdef SHOW_CISTEM_GPU_OPTIONS
            use_gpu_checkboxAR3D->Enable(true);
#endif
            RefinementRunProfileComboBox->Enable(true);
            ReconstructionRunProfileComboBox->Enable(true);
            InitialResLimitStaticText->Enable(true);
            HighResolutionLimitTextCtrl->Enable(true);
            UseMaskCheckBox->Enable(true);
            ExpertToggleButton->Enable(true);

            if ( RefinementPackageSelectPanel->GetCount( ) > 0 ) {
                RefinementPackageSelectPanel->Enable(true);
                //				InputParametersComboBox->Enable(true);

                if ( UseMaskCheckBox->GetValue( ) == true ) {
                    MaskSelectPanel->Enable(true);
                }
                else {
                    MaskSelectPanel->Enable(false);
                    //MaskSelectPanel->AssetComboBox->ChangeValue("");
                }

                if ( PleaseCreateRefinementPackageText->IsShown( ) ) {
                    PleaseCreateRefinementPackageText->Show(false);
                    Layout( );
                }
            }
            else {
                UseMaskCheckBox->Enable(false);
                MaskSelectPanel->Enable(false);
                //MaskSelectPanel->AssetComboBox->ChangeValue("");
                RefinementPackageSelectPanel->ChangeValue("");
                RefinementPackageSelectPanel->Enable(false);
                use_gpu_checkboxAR3D->Enable(false); // Doesn't matter if SHOW_CISTEM_GPU_OPTIONS

                //				InputParametersComboBox->ChangeValue("");
                //			InputParametersComboBox->Enable(false);

                if ( PleaseCreateRefinementPackageText->IsShown( ) == false ) {
                    PleaseCreateRefinementPackageText->Show(true);
                    Layout( );
                }
            }

            if ( ExpertToggleButton->GetValue( ) == true ) {

                if ( ApplyBlurringYesRadioButton->GetValue( ) == true ) {
                    SmoothingFactorTextCtrl->Enable(true);
                    SmoothingFactorStaticText->Enable(true);
                }
                else {
                    SmoothingFactorTextCtrl->Enable(false);
                    SmoothingFactorStaticText->Enable(false);
                }
                if ( UseMaskCheckBox->GetValue( ) == false ) {
                    MaskEdgeStaticText->Enable(false);
                    MaskEdgeTextCtrl->Enable(false);
                    MaskWeightStaticText->Enable(false);
                    MaskWeightTextCtrl->Enable(false);
                    LowPassYesNoStaticText->Enable(false);
                    LowPassMaskYesRadio->Enable(false);
                    LowPassMaskNoRadio->Enable(false);
                    FilterResolutionStaticText->Enable(false);
                    MaskFilterResolutionText->Enable(false);

                    AutoCenterYesRadioButton->Enable(true);
                    AutoCenterNoRadioButton->Enable(true);
                    AutoCenterStaticText->Enable(true);

                    AutoMaskStaticText->Enable(true);
                    AutoMaskYesRadioButton->Enable(true);
                    AutoMaskNoRadioButton->Enable(true);

                    if ( AutoMaskYesRadioButton->GetValue( ) != auto_mask_value ) {
                        if ( auto_mask_value == true )
                            AutoMaskYesRadioButton->SetValue(true);
                        else
                            AutoMaskNoRadioButton->SetValue(true);
                    }
                }
                else {

                    AutoCenterYesRadioButton->Enable(false);
                    AutoCenterNoRadioButton->Enable(false);
                    AutoCenterStaticText->Enable(false);

                    AutoMaskStaticText->Enable(false);
                    AutoMaskYesRadioButton->Enable(false);
                    AutoMaskNoRadioButton->Enable(false);

                    if ( AutoMaskYesRadioButton->GetValue( ) != false ) {
                        AutoMaskNoRadioButton->SetValue(true);
                    }

                    MaskEdgeStaticText->Enable(true);
                    MaskEdgeTextCtrl->Enable(true);
                    MaskWeightStaticText->Enable(true);
                    MaskWeightTextCtrl->Enable(true);
                    LowPassYesNoStaticText->Enable(true);
                    LowPassMaskYesRadio->Enable(true);
                    LowPassMaskNoRadio->Enable(true);

                    if ( LowPassMaskYesRadio->GetValue( ) == true ) {
                        FilterResolutionStaticText->Enable(true);
                        MaskFilterResolutionText->Enable(true);
                    }
                    else {
                        FilterResolutionStaticText->Enable(false);
                        MaskFilterResolutionText->Enable(false);
                    }
                }
            }

            bool estimation_button_status = false;

            if ( RefinementPackageSelectPanel->GetCount( ) > 0 && ReconstructionRunProfileComboBox->GetCount( ) > 0 ) {
                if ( run_profiles_panel->run_profile_manager.ReturnTotalJobs(RefinementRunProfileComboBox->GetSelection( )) > 0 && run_profiles_panel->run_profile_manager.ReturnTotalJobs(ReconstructionRunProfileComboBox->GetSelection( )) > 0 ) {
                    if ( RefinementPackageSelectPanel->GetSelection( ) != wxNOT_FOUND && ReferenceSelectPanel->GetSelection( ) != wxNOT_FOUND ) {
                        if ( UseMaskCheckBox->GetValue( ) == false || MaskSelectPanel->AssetComboBox->GetSelection( ) != wxNOT_FOUND )
                            estimation_button_status = true;
                    }
                }
            }

            StartRefinementButton->Enable(estimation_button_status);

            if ( refinement_package_combo_is_dirty == true ) {
                FillRefinementPackagesComboBox( );
                refinement_package_combo_is_dirty = false;
            }

            if ( run_profiles_are_dirty == true ) {
                FillRunProfileComboBoxes( );
                run_profiles_are_dirty = false;
            }

            if ( volumes_are_dirty == true ) {
                ReferenceSelectPanel->FillComboBox( );
                MaskSelectPanel->FillComboBox( );
                volumes_are_dirty = false;
            }
        }
    }
    else {
        ReferenceSelectPanel->Enable(false);
        RefinementPackageSelectPanel->Enable(false);
        //		InputParametersComboBox->Enable(false);
        RefinementRunProfileComboBox->Enable(false);
        ReconstructionRunProfileComboBox->Enable(false);
        ExpertToggleButton->Enable(false);
        StartRefinementButton->Enable(false);
        //		LocalRefinementRadio->Enable(false);
        //		GlobalRefinementRadio->Enable(false);
        //		NumberRoundsSpinCtrl->Enable(false);
        UseMaskCheckBox->Enable(false);
        MaskSelectPanel->Enable(false);

        if ( ExpertPanel->IsShown( ) == true ) {
            ExpertToggleButton->SetValue(false);
            ExpertPanel->Show(false);
            Layout( );
        }

        if ( RefinementPackageSelectPanel->GetCount( ) > 0 ) {
            RefinementPackageSelectPanel->Clear( );
            RefinementPackageSelectPanel->ChangeValue("");
        }

        if ( ReferenceSelectPanel->GetCount( ) > 0 ) {
            ReferenceSelectPanel->Clear( );
            ReferenceSelectPanel->ChangeValue("");
        }
        /*
		if (InputParametersComboBox->GetCount() > 0)
		{
			InputParametersComboBox->Clear();
			InputParametersComboBox->ChangeValue("");
		}*/

        if ( ReconstructionRunProfileComboBox->GetCount( ) > 0 ) {
            ReconstructionRunProfileComboBox->Clear( );
            //ReconstructionRunProfileComboBox->ChangeValue("");
        }

        if ( RefinementRunProfileComboBox->GetCount( ) > 0 ) {
            RefinementRunProfileComboBox->Clear( );
            //RefinementRunProfileComboBox->ChangeValue("");
        }

        if ( PleaseCreateRefinementPackageText->IsShown( ) ) {
            PleaseCreateRefinementPackageText->Show(false);
            Layout( );
        }
    }
}

void AutoRefine3DPanelSpa::OnAutoMaskButton(wxCommandEvent& event) {
    auto_mask_value = AutoMaskYesRadioButton->GetValue( );
}

void AutoRefine3DPanelSpa::OnUseMaskCheckBox(wxCommandEvent& event) {
    if ( UseMaskCheckBox->GetValue( ) == true ) {
        AutoCenterYesRadioButton->SetValue(false); // should we even disable auto centering?
        AutoCenterNoRadioButton->SetValue(true);
        AutoMaskYesRadioButton->SetValue(false);
        AutoMaskNoRadioButton->SetValue(true);
        auto_mask_value = false;
        MaskSelectPanel->FillComboBox( );
    }
    else {
        AutoMaskYesRadioButton->SetValue(true);
        AutoMaskNoRadioButton->SetValue(false);
        auto_mask_value = true;
        AutoCenterYesRadioButton->SetValue(true);
        AutoCenterNoRadioButton->SetValue(false);
    }
}

void AutoRefine3DPanelSpa::OnExpertOptionsToggle(wxCommandEvent& event) {

    if ( ExpertToggleButton->GetValue( ) == true ) {
        ExpertPanel->Show(true);
        Layout( );
    }
    else {
        ExpertPanel->Show(false);
        Layout( );
    }
}

void AutoRefine3DPanelSpa::FillRefinementPackagesComboBox( ) {
    if ( RefinementPackageSelectPanel->FillComboBox( ) == false )
        NewRefinementPackageSelected( );
}

void AutoRefine3DPanelSpa::NewRefinementPackageSelected( ) {
    selected_refinement_package = RefinementPackageSelectPanel->GetSelection( );
    SetDefaults( );

    if ( RefinementPackageSelectPanel->GetCount( ) > 0 && ReferenceSelectPanel->GetCount( ) > 0 )
        ReferenceSelectPanel->SetSelection(volume_asset_panel->ReturnArrayPositionFromAssetID(refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageSelectPanel->GetSelection( )).references_for_next_refinement[0]));
    //wxPrintf("New Refinement Package Selection\n");
}

void AutoRefine3DPanelSpa::OnRefinementPackageComboBox(wxCommandEvent& event) {

    NewRefinementPackageSelected( );
}

void AutoRefine3DPanelSpa::OnInputParametersComboBox(wxCommandEvent& event) {
    //SetDefaults();
}

void AutoRefine3DPanelSpa::TerminateButtonClick(wxCommandEvent& event) {
    main_frame->job_controller.KillJob(my_job_id);

    active_mask_thread_id = -1;
    active_orth_thread_id = -1;

    WriteBlueText("Terminated Job");
    TimeRemainingText->SetLabel("Time Remaining : Terminated");
    CancelAlignmentButton->Show(false);
    FinishButton->Show(true);
    ProgressPanel->Layout( );
    /*
	if (buffered_results != NULL)
	{
		delete [] buffered_results;
		buffered_results = NULL;
	}*/
}

void AutoRefine3DPanelSpa::FinishButtonClick(wxCommandEvent& event) {
    ProgressBar->SetValue(0);
    TimeRemainingText->SetLabel("Time Remaining : ???h:??m:??s");
    CancelAlignmentButton->Show(true);
    FinishButton->Show(false);

    InputParamsPanel->Show(true);
    ProgressPanel->Show(false);
    StartPanel->Show(true);
    OutputTextPanel->Show(false);
    output_textctrl->Clear( );
    ShowRefinementResultsPanel->Show(false);
    ShowRefinementResultsPanel->Clear( );
    InfoPanel->Show(true);

    if ( my_refinement_manager.output_refinement != NULL ) {
        delete my_refinement_manager.output_refinement;
        my_refinement_manager.output_refinement = NULL;
    }

    if ( ExpertToggleButton->GetValue( ) == true )
        ExpertPanel->Show(true);
    else
        ExpertPanel->Show(false);
    running_job = false;
    Layout( );

    //CTFResultsPanel->CTF2DResultsPanel->should_show = false;
    //CTFResultsPanel->CTF2DResultsPanel->Refresh();
}

void AutoRefine3DPanelSpa::StartRefinementClick(wxCommandEvent& event) {
    stopwatch.Start( );
    my_refinement_manager.BeginRefinementCycle( );
}

void AutoRefine3DPanelSpa::WriteInfoText(wxString text_to_write) {
    output_textctrl->SetDefaultStyle(wxTextAttr(*wxBLACK));
    output_textctrl->AppendText(text_to_write);

    if ( text_to_write.EndsWith("\n") == false )
        output_textctrl->AppendText("\n");
}

void AutoRefine3DPanelSpa::WriteBlueText(wxString text_to_write) {
    output_textctrl->SetDefaultStyle(wxTextAttr(*wxBLUE));
    output_textctrl->AppendText(text_to_write);

    if ( text_to_write.EndsWith("\n") == false )
        output_textctrl->AppendText("\n");
}

void AutoRefine3DPanelSpa::WriteErrorText(wxString text_to_write) {
    output_textctrl->SetDefaultStyle(wxTextAttr(*wxRED));
    output_textctrl->AppendText(text_to_write);

    if ( text_to_write.EndsWith("\n") == false )
        output_textctrl->AppendText("\n");
}

void AutoRefine3DPanelSpa::FillRunProfileComboBoxes( ) {
    ReconstructionRunProfileComboBox->FillWithRunProfiles( );
    RefinementRunProfileComboBox->FillWithRunProfiles( );
}

void AutoRefine3DPanelSpa::OnSocketJobResultMsg(JobResult& received_result) {
    my_refinement_manager.ProcessJobResult(&received_result);
}

void AutoRefine3DPanelSpa::OnSocketJobResultQueueMsg(ArrayofJobResults& received_queue) {
    for ( int counter = 0; counter < received_queue.GetCount( ); counter++ ) {
        my_refinement_manager.ProcessJobResult(&received_queue.Item(counter));
    }
}

void AutoRefine3DPanelSpa::SetNumberConnectedText(wxString wanted_text) {
    NumberConnectedText->SetLabel(wanted_text);
}

void AutoRefine3DPanelSpa::SetTimeRemainingText(wxString wanted_text) {
    TimeRemainingText->SetLabel(wanted_text);
}

void AutoRefine3DPanelSpa::OnSocketAllJobsFinished( ) {
    my_refinement_manager.ProcessAllJobsFinished( );
}

void AutoRefine3DPanelSpa::OnMaskerThreadComplete(wxThreadEvent& my_event) {
    if ( my_event.GetInt( ) == active_mask_thread_id )
        my_refinement_manager.OnMaskerThreadComplete( );
}

void AutoRefine3DPanelSpa::OnOrthThreadComplete(ReturnProcessedImageEvent& my_event) {

    Image* new_image = my_event.GetImage( );

    if ( my_event.GetInt( ) == active_orth_thread_id ) {
        if ( new_image != NULL ) {
            ShowRefinementResultsPanel->ShowOrthDisplayPanel->OpenImage(new_image, my_event.GetString( ), true);

            if ( ShowRefinementResultsPanel->LeftRightSplitter->IsSplit( ) == false ) {
                ShowRefinementResultsPanel->LeftRightSplitter->SplitVertically(ShowRefinementResultsPanel->LeftPanel, ShowRefinementResultsPanel->RightPanel, 600);
                Layout( );
            }
        }
    }
    else {
        delete new_image;
    }
}