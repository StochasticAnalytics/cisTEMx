#include "../core/gui_core_headers.h"
#include "TemplateMatchQueueItemEditor.h"
#include "MatchTemplatePanel.h"
#include "TemplateMatchQueueManager.h"

TemplateMatchQueueItemEditor::TemplateMatchQueueItemEditor(wxWindow*                  parent,
                                                           MatchTemplatePanel*        tm_panel,
                                                           TemplateMatchQueueManager* qm)
    : TemplateMatchingQueueManagerInputParent(parent),
      match_template_panel_ptr(tm_panel),
      queue_manager_ptr(qm),
      controls_helper(nullptr) {

    // Build control bundle struct
    TemplateMatchControls controls;
    controls.group_combo             = GroupComboBox;
    controls.reference_panel         = ReferenceSelectPanel;
    controls.run_profile_combo       = RunProfileComboBox;
    controls.symmetry_combo          = SymmetryComboBox;
    controls.high_res_limit          = HighResolutionLimitNumericCtrl;
    controls.out_of_plane_step       = OutofPlaneStepNumericCtrl;
    controls.in_plane_step           = InPlaneStepNumericCtrl;
    controls.defocus_search_yes      = DefocusSearchYesRadio;
    controls.defocus_search_no       = DefocusSearchNoRadio;
    controls.defocus_search_range    = DefocusSearchRangeNumericCtrl;
    controls.defocus_search_step     = DefocusSearchStepNumericCtrl;
    controls.pixel_size_search_yes   = PixelSizeSearchYesRadio;
    controls.pixel_size_search_no    = PixelSizeSearchNoRadio;
    controls.pixel_size_search_range = PixelSizeSearchRangeNumericCtrl;
    controls.pixel_size_search_step  = PixelSizeSearchStepNumericCtrl;
    controls.min_peak_radius         = MinPeakRadiusNumericCtrl;
    controls.use_gpu_yes             = UseGPURadioYes;
    controls.use_gpu_no              = UseGPURadioNo;
    controls.use_fast_fft_yes        = UseFastFFTRadioYes;
    controls.use_fast_fft_no         = UseFastFFTRadioNo;
    controls.custom_cli_args_text    = custom_cli_args_text;

    // Create helper with control references
    controls_helper = new TemplateMatchControlsHelper(controls);

    // Explicitly show InputPanel and ExpertPanel (may be hidden by default in some layouts)
    InputPanel->Show(true);
    ExpertPanel->Show(true);

    // Force layout update to ensure child panels are properly sized and positioned
    Layout( );

    // Panel starts hidden (the outer TemplateMatchQueueItemEditor panel)
    Show(false);

    QM_LOG_UI("TemplateMatchQueueItemEditor created");
}

TemplateMatchQueueItemEditor::~TemplateMatchQueueItemEditor( ) {
    delete controls_helper;
    controls_helper = nullptr;
}

// Event handler implementations
void TemplateMatchQueueItemEditor::OnUpdateQueueItemClick(wxCommandEvent& event) {
    QM_LOG_METHOD_ENTRY("OnUpdateQueueItemClick");

    // Delegate to queue manager for save logic
    if ( queue_manager_ptr ) {
        queue_manager_ptr->OnEditorSaveClick( );
    }
    else {
        wxMessageBox("Internal error: queue_manager_ptr is null", "Error", wxOK | wxICON_ERROR);
    }
}

void TemplateMatchQueueItemEditor::OnCancelUpdateQueueItemClick(wxCommandEvent& event) {
    QM_LOG_METHOD_ENTRY("OnCancelUpdateQueueItemClick");

    // Delegate to queue manager to switch back to queue view
    if ( queue_manager_ptr ) {
        queue_manager_ptr->OnEditorCancelClick( );
    }
    else {
        wxMessageBox("Internal error: queue_manager_ptr is null", "Error", wxOK | wxICON_ERROR);
    }
}

void TemplateMatchQueueItemEditor::ResetAllDefaultsClick(wxCommandEvent& event) {
    // Not implemented for editor - defaults are for new searches, not editing existing ones
    // Just skip the event
    event.Skip( );
}
