#ifndef __GenericRunnerPanel__
#define __GenericRunnerPanel__

class GenericRunnerPanel : public JobPanel {

  public:
    GenericRunnerPanel(wxWindow*      parent,
                       wxWindowID     id    = wxID_ANY,
                       const wxPoint& pos   = wxDefaultPosition,
                       const wxSize&  size  = wxSize(869, 566),
                       long           style = wxTAB_TRAVERSAL);
    ~GenericRunnerPanel( );

    // UI elements
    wxComboBox* RunProfileComboBox;
    wxButton*   StartButton;
    wxTextCtrl* output_textctrl;

    // State
    bool run_profiles_are_dirty;
    int  expected_result_count;
    int  received_result_count;
    int  passed_result_count;

    // Event handlers
    void OnStartButtonClick(wxCommandEvent& event);
    void OnUpdateUI(wxUpdateUIEvent& event);

    // Overridden socket methods (from JobPanel / SocketCommunicator)
    void WriteInfoText(wxString text_to_write);
    void WriteErrorText(wxString text_to_write);
    void SetNumberConnectedText(wxString wanted_text);
    void SetTimeRemainingText(wxString wanted_text);
    void OnSocketAllJobsFinished( );

    // Result handling — receives batched JobResults relayed by the master.
    // AddJobToResultQueue → master_job_queue → OnSocketJobResultQueueMsg (queue variant)
    void OnSocketJobResultQueueMsg(ArrayofJobResults& received_queue);

    // Disconnect handling — re-enables StartButton on failure
    void HandleSocketDisconnect(wxSocketBase* connected_socket) override;

    void FillRunProfileComboBox( );
};

#endif
