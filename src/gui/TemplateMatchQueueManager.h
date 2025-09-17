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
    wxString custom_cli_args;

    // Store all the parameters needed to run the job
    // These will be populated when adding to queue
    int image_group_id;
    int reference_volume_asset_id;
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
        image_group_id = -1;
        reference_volume_asset_id = -1;
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
    wxButton* run_all_button;
    wxButton* clear_queue_button;
    wxButton* move_up_button;
    wxButton* move_down_button;
    wxButton* remove_selected_button;

    // Use static queue to persist across dialog instances
    static std::deque<TemplateMatchQueueItem> execution_queue;
    static long currently_running_id;
    bool needs_database_load;  // True if we haven't loaded from DB yet

    // Pointer to main frame for database access
    MyMainFrame* main_frame_ptr;

    wxColour GetStatusColor(const wxString& status);
    void UpdateQueueDisplay();
    int GetSelectedRow();

public:
    TemplateMatchQueueManager(wxWindow* parent, MyMainFrame* main_frame = nullptr);

    // Queue management methods
    void AddToQueue(const TemplateMatchQueueItem& item);
    void RemoveFromQueue(int index);
    void ClearQueue();
    void MoveItemUp(int index);
    void MoveItemDown(int index);

    // Execution methods
    void RunSelectedJob();
    void RunAllJobs();
    void RunNextJob();
    bool ExecuteJob(TemplateMatchQueueItem& job_to_run);
    void UpdateJobStatus(long template_match_id, const wxString& new_status);
    static void UpdateJobStatusStatic(long template_match_id, const wxString& new_status);
    TemplateMatchQueueItem* GetNextPendingJob();
    bool HasPendingJobs();
    bool IsJobRunning();
    static bool IsJobRunningStatic();

    // Validation methods
    void ValidateQueueConsistency() const;

    // Event handlers
    void OnRunSelectedClick(wxCommandEvent& event);
    void OnRunAllClick(wxCommandEvent& event);
    void OnClearQueueClick(wxCommandEvent& event);
    void OnMoveUpClick(wxCommandEvent& event);
    void OnMoveDownClick(wxCommandEvent& event);
    void OnRemoveSelectedClick(wxCommandEvent& event);
    void OnSelectionChanged(wxDataViewEvent& event);
    void OnItemValueChanged(wxDataViewEvent& event);

    // Load/Save queue from database
    void LoadQueueFromDatabase();
    void SaveQueueToDatabase();

    DECLARE_EVENT_TABLE()
};

#endif // _SRC_GUI_TEMPLATEMATCHQUEUEMANAGER_H_