#ifndef __src_gui_AutoRefine3DPanelSpa_h__
#define __src_gui_AutoRefine3DPanelSpa_h__

#include "ProjectX_gui.h"
#include "AutoRefine3dManager.h"

// We need the forward declartion so we can reference the class as a tempalte parameter
class AutoRefine3dPanelSpa;

class AutoRefine3DPanelSpa : public AutoRefine3DPanelParent {

    friend class AutoRefinementManager<AutoRefine3DPanelSpa>;

  protected:
    // Handlers for Refine3DPanel events.

    void OnUpdateUI(wxUpdateUIEvent& event);
    void OnExpertOptionsToggle(wxCommandEvent& event);
    void OnInfoURL(wxTextUrlEvent& event);
    void TerminateButtonClick(wxCommandEvent& event);
    void FinishButtonClick(wxCommandEvent& event);
    void StartRefinementClick(wxCommandEvent& event);
    void ResetAllDefaultsClick(wxCommandEvent& event);

    void OnUseMaskCheckBox(wxCommandEvent& event);
    void OnAutoMaskButton(wxCommandEvent& event);

    // overridden socket methods..

    void OnSocketJobResultMsg(JobResult& received_result);
    void OnSocketJobResultQueueMsg(ArrayofJobResults& received_queue);
    void SetNumberConnectedText(wxString wanted_text);
    void SetTimeRemainingText(wxString wanted_text);
    void OnSocketAllJobsFinished( );

    int length_of_process_number;

    AutoRefinementManager<AutoRefine3DPanelSpa> my_refinement_manager;

    int active_orth_thread_id;
    int active_mask_thread_id;
    int next_thread_id;

  public:
    wxStopWatch stopwatch;

    long time_of_last_result_update;

    bool refinement_package_combo_is_dirty;
    bool run_profiles_are_dirty;
    bool volumes_are_dirty;

    JobResult* buffered_results;
    long       selected_refinement_package;

    //int length_of_process_number;

    JobTracker my_job_tracker;

    bool auto_mask_value; // this is needed to keep track of the automask, as the radiobutton will be overidden to no when masking is selected

    bool running_job;

    AutoRefine3DPanelSpa(wxWindow* parent);

    void Reset( );
    void SetDefaults( );
    void SetInfo( );

    void WriteInfoText(wxString text_to_write);
    void WriteErrorText(wxString text_to_write);
    void WriteBlueText(wxString text_to_write);

    void FillRefinementPackagesComboBox( );
    void FillRunProfileComboBoxes( );

    void NewRefinementPackageSelected( );

    void OnRefinementPackageComboBox(wxCommandEvent& event);
    void OnInputParametersComboBox(wxCommandEvent& event);

    void OnMaskerThreadComplete(wxThreadEvent& my_event);
    void OnOrthThreadComplete(ReturnProcessedImageEvent& my_event);
};

#endif
