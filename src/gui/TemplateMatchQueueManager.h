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
    int image_asset_id;
    int reference_volume_asset_id;
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
        image_asset_id = -1;
        reference_volume_asset_id = -1;
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
    long currently_running_id;

    wxColour GetStatusColor(const wxString& status);
    void UpdateQueueDisplay();
    int GetSelectedRow();

public:
    TemplateMatchQueueManager(wxWindow* parent);

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
    TemplateMatchQueueItem* GetNextPendingJob();
    bool HasPendingJobs();
    bool IsJobRunning();

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