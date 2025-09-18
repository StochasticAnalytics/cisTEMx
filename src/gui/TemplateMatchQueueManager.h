#ifndef _SRC_GUI_TEMPLATEMATCHQUEUEMANAGER_H_
#define _SRC_GUI_TEMPLATEMATCHQUEUEMANAGER_H_

#include <wx/wx.h>
#include <wx/dataview.h>
#include <deque>

class TemplateMatchQueueItem {
public:
    long template_match_id;
    wxString job_name;
    wxString queue_status;  // "pending", "running", "complete", "failed"
    int queue_order;  // 0 = running, 1+ = pending queue position
    wxString custom_cli_args;

    // Store all the parameters needed to run the job
    // These will be populated when adding to queue
    int image_group_id;
    int reference_volume_asset_id;
    int run_profile_id;
    bool use_gpu;
    bool use_fast_fft;
    wxString symmetry;
    float pixel_size;
    float voltage;
    float spherical_aberration;
    float amplitude_contrast;
    float defocus1;
    float defocus2;
    float defocus_angle;
    float phase_shift;
    float low_resolution_limit;
    float high_resolution_limit;
    float out_of_plane_angular_step;
    float in_plane_angular_step;
    float defocus_search_range;
    float defocus_step;
    float pixel_size_search_range;
    float pixel_size_step;
    float refinement_threshold;
    float ref_box_size_in_angstroms;
    float mask_radius;
    float min_peak_radius;
    float xy_change_threshold;
    bool exclude_above_xy_threshold;

    TemplateMatchQueueItem() {
        template_match_id = -1;
        queue_status = "pending";
        queue_order = -1;  // Will be assigned when added to queue
        image_group_id = -1;
        reference_volume_asset_id = -1;
        run_profile_id = -1;
        use_gpu = false;
        use_fast_fft = false;

        // Initialize other parameters to safe defaults
        pixel_size = 0.0f;
        voltage = 0.0f;
        spherical_aberration = 0.0f;
        amplitude_contrast = 0.0f;
        defocus1 = 0.0f;
        defocus2 = 0.0f;
        defocus_angle = 0.0f;
        phase_shift = 0.0f;
        low_resolution_limit = 0.0f;
        high_resolution_limit = 0.0f;
        out_of_plane_angular_step = 0.0f;
        in_plane_angular_step = 0.0f;
        defocus_search_range = 0.0f;
        defocus_step = 0.0f;
        pixel_size_search_range = 0.0f;
        pixel_size_step = 0.0f;
        refinement_threshold = 0.0f;
        ref_box_size_in_angstroms = 0.0f;
        mask_radius = 0.0f;
        min_peak_radius = 0.0f;
        xy_change_threshold = 0.0f;
        exclude_above_xy_threshold = false;
    }

    // Validation method to check if this queue item has all required parameters for job execution
    bool AreJobParametersValid() const {
        MyDebugAssertTrue(template_match_id >= 0, "template_match_id must be >= 0, got %ld", template_match_id);
        MyDebugAssertTrue(image_group_id >= 0, "image_group_id must be >= 0, got %d", image_group_id);
        MyDebugAssertTrue(reference_volume_asset_id >= 0, "reference_volume_asset_id must be >= 0, got %d", reference_volume_asset_id);
        MyDebugAssertTrue(run_profile_id >= 0, "run_profile_id must be >= 0, got %d", run_profile_id);
        MyDebugAssertTrue(pixel_size > 0.0f, "pixel_size must be > 0.0, got %f", pixel_size);
        MyDebugAssertTrue(voltage > 0.0f, "voltage must be > 0.0, got %f", voltage);
        MyDebugAssertTrue(spherical_aberration >= 0.0f, "spherical_aberration must be >= 0.0, got %f", spherical_aberration);
        MyDebugAssertTrue(amplitude_contrast >= 0.0f && amplitude_contrast <= 1.0f, "amplitude_contrast must be 0.0-1.0, got %f", amplitude_contrast);
        MyDebugAssertTrue(queue_status == "pending" || queue_status == "running" || queue_status == "complete" || queue_status == "failed",
                         "Invalid queue_status: %s", queue_status.mb_str().data());
        MyDebugAssertTrue(!job_name.IsEmpty(), "job_name cannot be empty");
        MyDebugAssertFalse(symmetry.IsEmpty(), "symmetry cannot be empty");
        return true;
    }
};

class TemplateMatchQueueManager : public wxPanel {
private:
    wxDataViewListCtrl* queue_list_ctrl;
    wxButton* run_selected_button;
    wxButton* clear_queue_button;
    wxTextCtrl* position_input;
    wxButton* set_position_button;
    wxButton* remove_selected_button;

    // Queue instance variables
    std::deque<TemplateMatchQueueItem> execution_queue;
    long currently_running_id;

    // Selection-based execution tracking
    std::deque<int> selected_jobs_for_execution;  // Queue IDs selected for execution
    bool execution_in_progress;  // True while processing selection queue
    bool needs_database_load;  // True if we haven't loaded from DB yet

    // Pointer to match template panel for job execution and database access
    MatchTemplatePanel* match_template_panel_ptr;

    wxColour GetStatusColor(const wxString& status);
    void UpdateQueueDisplay();
    int GetSelectedRow();
    void DeselectJobInUI(int template_match_id);

public:
    TemplateMatchQueueManager(wxWindow* parent, MatchTemplatePanel* match_template_panel = nullptr);
    ~TemplateMatchQueueManager();

    // Queue management methods
    void AddToQueue(const TemplateMatchQueueItem& item);
    void RemoveFromQueue(int index);
    void ClearQueue();

    // Execution methods
    void RunSelectedJob();
    void RunNextJob();
    void RunNextSelectedJob();
    bool ExecuteJob(TemplateMatchQueueItem& job_to_run);
    void UpdateJobStatus(long template_match_id, const wxString& new_status);
    void ContinueQueueExecution();
    TemplateMatchQueueItem* GetNextPendingJob();

    // Job completion callback
    void OnJobCompleted(long template_match_id, bool success);

    // Selection management
    void PopulateSelectionQueueFromUI();
    void RemoveJobFromSelectionQueue(int template_match_id);
    bool HasJobsInSelectionQueue() const;
    bool HasPendingJobs();
    bool IsJobRunning() const;

    // Validation methods
    void ValidateQueueConsistency() const;

    // Inline job status helper methods
    inline bool IsJobRunning(int queue_index) const {
        return queue_index >= 0 && queue_index < execution_queue.size() &&
               execution_queue[queue_index].queue_status == "running";
    }

    inline bool IsJobPending(int queue_index) const {
        return queue_index >= 0 && queue_index < execution_queue.size() &&
               execution_queue[queue_index].queue_status == "pending";
    }

    inline bool IsJobComplete(int queue_index) const {
        return queue_index >= 0 && queue_index < execution_queue.size() &&
               execution_queue[queue_index].queue_status == "complete";
    }

    inline bool IsJobFailed(int queue_index) const {
        return queue_index >= 0 && queue_index < execution_queue.size() &&
               execution_queue[queue_index].queue_status == "failed";
    }

    // Event handlers
    void OnRunSelectedClick(wxCommandEvent& event);
    void OnClearQueueClick(wxCommandEvent& event);
    void OnSetPositionClick(wxCommandEvent& event);
    void OnRemoveSelectedClick(wxCommandEvent& event);
    void OnSelectionChanged(wxDataViewEvent& event);
    void OnItemValueChanged(wxDataViewEvent& event);

    // Load/Save queue from database
    void LoadQueueFromDatabase();
    void SaveQueueToDatabase();

    DECLARE_EVENT_TABLE()
};

#endif // _SRC_GUI_TEMPLATEMATCHQUEUEMANAGER_H_