#ifndef __MatchTemplatePanel__
#define __MatchTemplatePanel__

// Forward declaration
class TemplateMatchQueueItem;

class MatchTemplatePanel : public MatchTemplatePanelParent {

    // The results and actions panels need to talk to each other.
    friend class MatchTemplateResultsPanel;
    friend class TemplateMatchQueueManager; // Allow QueueManager to access protected panels
    long my_job_id;

    JobTracker my_job_tracker;

    // Check if a job is currently running by querying job controller directly
    // Replaces the old running_job boolean to avoid redundant state tracking
    inline bool IsJobRunning( ) const {
        return my_job_id >= 0 && my_job_id < MAX_GUI_JOBS &&
               main_frame && main_frame->job_controller.job_list[my_job_id].is_active;
    }

    Image    result_image;
    wxBitmap result_bitmap;

    wxArrayString input_image_filenames;

    float ref_box_size_in_pixels;

    AssetGroup active_group;
    bool       all_images_have_defocus_values;
    float      current_pixel_size; // Track pixel size to avoid unnecessary resolution limit updates

    ArrayOfTemplateMatchJobResults cached_results;

    void CheckForUnfinishedWork(std::vector<long>& images_to_resume, int image_group_id, long search_id, long images_total);

  public:
    MatchTemplatePanel(wxWindow* parent);

    bool group_combo_is_dirty;
    bool run_profiles_are_dirty;
    bool volumes_are_dirty;
    bool no_unfinished_jobs = true; // Just for testing, will be set locally by DB functions

    long time_of_last_result_update;

    long expected_number_of_results;
    long number_of_received_results;
    long current_job_starttime;
    long time_of_last_update;
    long queue_wait_start_time; // Time when we started waiting for next queue item

    // Search tracking
    int search_id; // TEMPLATE_MATCH_LIST.SEARCH_ID - created when first result written

    // Queue tracking
    long                             running_queue_id; // Database queue ID of the currently executing search (-1 when idle)
    class TemplateMatchQueueManager* queue_completion_callback; // For live queue updates
    wxString                         current_custom_cli_args; // Custom CLI args for current job
    static wxDialog*                 active_queue_dialog; // Track the currently open queue manager dialog

    // methods
    void WriteResultToDataBase( );
    void OnUpdateUI(wxUpdateUIEvent& event);
    void FillGroupComboBox( );
    void FillRunProfileComboBox( );
    /**
     * @brief Initiates immediate template matching search execution bypassing queue dialog
     *
     * Collects GUI parameters, creates temporary QueueManager, adds search to execution queue,
     * and begins processing immediately and approximates the legacy workflow with one search at a time.
     * Used for "Start Estimation" button workflow.
     */
    void StartEstimationClick(wxCommandEvent& event);
    void FinishButtonClick(wxCommandEvent& event);
    void TerminateButtonClick(wxCommandEvent& event);
    void ResetAllDefaultsClick(wxCommandEvent& event);

    void OnSocketJobResultMsg(JobResult& received_result);
    void OnSocketJobResultQueueMsg(ArrayofJobResults& received_queue);
    void SetNumberConnectedText(wxString wanted_text);
    void SetTimeRemainingText(wxString wanted_text);
    void OnSocketAllJobsFinished( );
    void HandleSocketTemplateMatchResultReady(wxSocketBase* connected_socket, int& image_number, float& threshold_used, ArrayOfTemplateMatchFoundPeakInfos& peak_infos, ArrayOfTemplateMatchFoundPeakInfos& peak_changes, long& elapsed_time_seconds);
    void HandleSocketDisconnect(wxSocketBase* connected_socket) override; // Detect master process exit for queue auto-advance
    bool CheckGroupHasDefocusValues( );

    //void Refresh();
    void SetInfo( );
    void OnInfoURL(wxTextUrlEvent& event);
    void OnGroupComboBox(wxCommandEvent& event);

    void WriteInfoText(wxString text_to_write);
    void WriteErrorText(wxString text_to_write);

    void ProcessResult(JobResult* result_to_process);
    void ProcessAllJobsFinished( );
    void UpdateProgressBar( );

    void Reset( );
    void ResetDefaults( );

    // Queue functionality
    /**
     * @brief Adds current GUI parameters as search to queue and shows queue management dialog
     *
     * Validates no search is currently running, collects GUI parameters into TemplateMatchQueueItem,
     * and opens queue manager dialog for user interaction. Enables batch processing workflows.
     */
    void OnAddToQueueClick(wxCommandEvent& event);

    /**
     * @brief Opens queue management dialog to view and manage existing searches
     *
     * Creates dialog-scoped QueueManager, loads existing searches from database, and provides
     * UI for queue manipulation, priority assignment, and batch execution control.
     */
    void OnOpenQueueClick(wxCommandEvent& event);
    void PopulateGuiFromQueueItem(const TemplateMatchQueueItem& item, bool for_editing = false);
    bool RunQueuedTemplateMatch(TemplateMatchQueueItem& job);

    // Shared job execution methods
    TemplateMatchQueueItem CollectJobParametersFromGui( );
    /**
     * @brief Core method to add search to execution queue with optional dialog display
     *
     * Creates dialog-scoped QueueManager, loads existing queue from database, adds the provided
     * search, and optionally shows queue management dialog. Used by both immediate execution
     * (show_dialog=false) and interactive queueing (show_dialog=true) workflows.
     *
     * @param job Search parameters collected from GUI via CollectJobParametersFromGui()
     * @param show_dialog If true, opens queue manager dialog; if false, adds silently for immediate execution
     * @return Database queue ID when show_dialog=false, -1 when show_dialog=true (dialog mode doesn't track ID)
     */
    long AddJobToQueue(const TemplateMatchQueueItem& job, bool show_dialog = true);
    bool SetupSearchFromQueueItem(const TemplateMatchQueueItem& job, long& pending_queue_id);
    bool ExecuteCurrentSearch(long pending_queue_id);
    bool ExecuteSearch(const TemplateMatchQueueItem* queue_item);

    // Queue completion callback support
    void SetQueueCompletionCallback(class TemplateMatchQueueManager* queue_manager);
    void ClearQueueCompletionCallback( );
};

#endif
