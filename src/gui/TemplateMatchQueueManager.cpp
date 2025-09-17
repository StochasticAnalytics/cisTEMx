#include "../core/gui_core_headers.h"
#include "TemplateMatchQueueManager.h"
#include "MatchTemplatePanel.h"

// Define static members
std::deque<TemplateMatchQueueItem> TemplateMatchQueueManager::execution_queue;
long TemplateMatchQueueManager::currently_running_id = -1;
TemplateMatchQueueManager* TemplateMatchQueueManager::active_instance = nullptr;

BEGIN_EVENT_TABLE(TemplateMatchQueueManager, wxPanel)
    EVT_BUTTON(wxID_ANY, TemplateMatchQueueManager::OnRunSelectedClick)
    EVT_DATAVIEW_SELECTION_CHANGED(wxID_ANY, TemplateMatchQueueManager::OnSelectionChanged)
    EVT_DATAVIEW_ITEM_VALUE_CHANGED(wxID_ANY, TemplateMatchQueueManager::OnItemValueChanged)
END_EVENT_TABLE()

TemplateMatchQueueManager::TemplateMatchQueueManager(wxWindow* parent, MatchTemplatePanel* match_template_panel)
    : wxPanel(parent, wxID_ANY), match_template_panel_ptr(match_template_panel) {

    // currently_running_id is static, initialized once
    needs_database_load = true;  // Need to load from database on first access

    // Create the main sizer
    wxBoxSizer* main_sizer = new wxBoxSizer(wxVERTICAL);

    // Create the list control
    queue_list_ctrl = new wxDataViewListCtrl(this, wxID_ANY,
                                            wxDefaultPosition, wxSize(500, 200),
                                            wxDV_SINGLE | wxDV_ROW_LINES);

    // Add columns
    queue_list_ctrl->AppendTextColumn("ID", wxDATAVIEW_CELL_INERT, 60);
    queue_list_ctrl->AppendTextColumn("Job Name", wxDATAVIEW_CELL_INERT, 200);
    queue_list_ctrl->AppendTextColumn("Status", wxDATAVIEW_CELL_INERT, 100);
    queue_list_ctrl->AppendTextColumn("CLI Args", wxDATAVIEW_CELL_EDITABLE, 140);  // Make editable

    // Create button panel
    wxPanel* button_panel = new wxPanel(this, wxID_ANY);
    wxBoxSizer* button_sizer = new wxBoxSizer(wxHORIZONTAL);

    move_up_button = new wxButton(button_panel, wxID_ANY, "Move Up");
    move_down_button = new wxButton(button_panel, wxID_ANY, "Move Down");
    run_selected_button = new wxButton(button_panel, wxID_ANY, "Run Selected");
    run_all_button = new wxButton(button_panel, wxID_ANY, "Run All");
    remove_selected_button = new wxButton(button_panel, wxID_ANY, "Remove");
    clear_queue_button = new wxButton(button_panel, wxID_ANY, "Clear Queue");

    button_sizer->Add(move_up_button, 0, wxALL, 5);
    button_sizer->Add(move_down_button, 0, wxALL, 5);
    button_sizer->AddSpacer(20);
    button_sizer->Add(run_selected_button, 0, wxALL, 5);
    button_sizer->Add(run_all_button, 0, wxALL, 5);
    button_sizer->AddSpacer(20);
    button_sizer->Add(remove_selected_button, 0, wxALL, 5);
    button_sizer->Add(clear_queue_button, 0, wxALL, 5);

    button_panel->SetSizer(button_sizer);

    // Add to main sizer
    main_sizer->Add(queue_list_ctrl, 1, wxEXPAND | wxALL, 5);
    main_sizer->Add(button_panel, 0, wxEXPAND | wxALL, 5);

    SetSizer(main_sizer);

    // Connect events
    move_up_button->Bind(wxEVT_BUTTON, &TemplateMatchQueueManager::OnMoveUpClick, this);
    move_down_button->Bind(wxEVT_BUTTON, &TemplateMatchQueueManager::OnMoveDownClick, this);
    run_selected_button->Bind(wxEVT_BUTTON, &TemplateMatchQueueManager::OnRunSelectedClick, this);
    run_all_button->Bind(wxEVT_BUTTON, &TemplateMatchQueueManager::OnRunAllClick, this);
    remove_selected_button->Bind(wxEVT_BUTTON, &TemplateMatchQueueManager::OnRemoveSelectedClick, this);
    clear_queue_button->Bind(wxEVT_BUTTON, &TemplateMatchQueueManager::OnClearQueueClick, this);

    // Don't load from database in constructor - it may be called during workflow switch
    // when main_frame is in an inconsistent state

    // Set this as the active instance for static method access
    active_instance = this;

    // Display any existing queue items
    UpdateQueueDisplay();
}

TemplateMatchQueueManager::~TemplateMatchQueueManager() {
    // Clear the active instance pointer if this is the active instance
    if (active_instance == this) {
        active_instance = nullptr;
    }
}

void TemplateMatchQueueManager::AddToQueue(const TemplateMatchQueueItem& item) {
    // Validate the queue item before adding
    MyDebugAssertTrue(item.image_group_id >= 0, "Cannot add item with invalid image_group_id: %d", item.image_group_id);
    MyDebugAssertTrue(item.reference_volume_asset_id >= 0, "Cannot add item with invalid reference_volume_asset_id: %d", item.reference_volume_asset_id);
    MyDebugAssertFalse(item.job_name.IsEmpty(), "Cannot add item with empty job_name");
    MyDebugAssertTrue(item.queue_status == "pending" || item.queue_status == "running" || item.queue_status == "complete" || item.queue_status == "failed",
                     "Cannot add item with invalid queue_status: %s", item.queue_status.mb_str().data());

    if (match_template_panel_ptr && main_frame && main_frame->current_project.is_open) {
        // Add item to database and get the new queue ID
        long new_queue_id = main_frame->current_project.database.AddToTemplateMatchQueue(
            item.job_name, item.image_group_id, item.reference_volume_asset_id, item.run_profile_id,
            item.use_gpu, item.use_fast_fft, item.symmetry,
            item.pixel_size, item.voltage, item.spherical_aberration, item.amplitude_contrast,
            item.defocus1, item.defocus2, item.defocus_angle, item.phase_shift,
            item.low_resolution_limit, item.high_resolution_limit,
            item.out_of_plane_angular_step, item.in_plane_angular_step,
            item.defocus_search_range, item.defocus_step,
            item.pixel_size_search_range, item.pixel_size_step,
            item.refinement_threshold, item.ref_box_size_in_angstroms,
            item.mask_radius, item.min_peak_radius,
            item.xy_change_threshold, item.exclude_above_xy_threshold,
            item.custom_cli_args);

        // Create a copy with the new database ID and add to in-memory queue
        TemplateMatchQueueItem new_item = item;
        new_item.template_match_id = new_queue_id;
        execution_queue.push_back(new_item);

        UpdateQueueDisplay();

        // Highlight the newly added item (last item in queue)
        if (queue_list_ctrl && execution_queue.size() > 0) {
            int new_row = execution_queue.size() - 1;
            queue_list_ctrl->SelectRow(new_row);
        }
    }
}

void TemplateMatchQueueManager::RemoveFromQueue(int index) {
    MyDebugAssertTrue(index >= 0, "RemoveFromQueue called with negative index: %d", index);
    MyDebugAssertTrue(index < execution_queue.size(), "RemoveFromQueue called with index %d >= queue size %zu", index, execution_queue.size());
    MyDebugAssertFalse(execution_queue.empty(), "Cannot remove from empty queue");

    if (index >= 0 && index < execution_queue.size()) {
        // Don't allow removal of currently running jobs
        MyDebugAssertTrue(execution_queue[index].queue_status != "running",
                         "Cannot remove currently running job (ID: %ld)", execution_queue[index].template_match_id);

        // Remove from database first
        if (match_template_panel_ptr && main_frame && main_frame->current_project.is_open) {
            main_frame->current_project.database.RemoveFromQueue(execution_queue[index].template_match_id);
        }

        execution_queue.erase(execution_queue.begin() + index);
        UpdateQueueDisplay();
    }
}

void TemplateMatchQueueManager::ClearQueue() {
    // Clear all items and remove from database
    if (match_template_panel_ptr && main_frame && main_frame->current_project.is_open) {
        // Remove all items from database
        for (const auto& item : execution_queue) {
            main_frame->current_project.database.RemoveFromQueue(item.template_match_id);
        }
    }

    execution_queue.clear();
    UpdateQueueDisplay();
}

void TemplateMatchQueueManager::MoveItemUp(int index) {
    MyDebugAssertTrue(index > 0, "MoveItemUp called with index %d, must be > 0", index);
    MyDebugAssertTrue(index < execution_queue.size(), "MoveItemUp called with index %d >= queue size %zu", index, execution_queue.size());
    MyDebugAssertFalse(execution_queue.empty(), "Cannot move items in empty queue");

    if (index > 0 && index < execution_queue.size()) {
        // Don't allow moving running jobs
        MyDebugAssertTrue(execution_queue[index].queue_status != "running" && execution_queue[index - 1].queue_status != "running",
                         "Cannot move running jobs (current: %s, target: %s)",
                         execution_queue[index].queue_status.mb_str().data(),
                         execution_queue[index - 1].queue_status.mb_str().data());

        std::swap(execution_queue[index], execution_queue[index - 1]);
        UpdateQueueDisplay();
        SaveQueueToDatabase();
    }
}

void TemplateMatchQueueManager::MoveItemDown(int index) {
    MyDebugAssertTrue(index >= 0, "MoveItemDown called with negative index: %d", index);
    MyDebugAssertTrue(index < execution_queue.size() - 1, "MoveItemDown called with index %d, must be < %zu", index, execution_queue.size() - 1);
    MyDebugAssertFalse(execution_queue.empty(), "Cannot move items in empty queue");

    if (index >= 0 && index < execution_queue.size() - 1) {
        // Don't allow moving running jobs
        MyDebugAssertTrue(execution_queue[index].queue_status != "running" && execution_queue[index + 1].queue_status != "running",
                         "Cannot move running jobs (current: %s, target: %s)",
                         execution_queue[index].queue_status.mb_str().data(),
                         execution_queue[index + 1].queue_status.mb_str().data());

        std::swap(execution_queue[index], execution_queue[index + 1]);
        UpdateQueueDisplay();
        SaveQueueToDatabase();
    }
}

wxColour TemplateMatchQueueManager::GetStatusColor(const wxString& status) {
    if (status == "running") {
        return wxColour(0, 128, 0);  // Green
    } else if (status == "complete") {
        return wxColour(0, 0, 255);  // Blue
    } else if (status == "failed") {
        return wxColour(255, 0, 0);  // Red
    } else {  // pending
        return wxColour(255, 0, 0);  // Red
    }
}

void TemplateMatchQueueManager::UpdateQueueDisplay() {
    MyDebugAssertTrue(queue_list_ctrl != nullptr, "queue_list_ctrl is null in UpdateQueueDisplay");

    queue_list_ctrl->DeleteAllItems();

    for (size_t i = 0; i < execution_queue.size(); ++i) {
        MyDebugAssertTrue(i < execution_queue.size(), "Queue index %zu out of bounds (size: %zu)", i, execution_queue.size());
        wxVector<wxVariant> data;
        data.push_back(wxString::Format("%ld", execution_queue[i].template_match_id));
        data.push_back(execution_queue[i].job_name);

        // Add status with a colored indicator prefix
        wxString status_display;
        if (execution_queue[i].queue_status == "running") {
            status_display = "● " + execution_queue[i].queue_status;  // Green circle
        } else if (execution_queue[i].queue_status == "complete") {
            status_display = "✓ " + execution_queue[i].queue_status;  // Check mark
        } else if (execution_queue[i].queue_status == "failed") {
            status_display = "✗ " + execution_queue[i].queue_status;  // X mark
        } else {  // pending
            status_display = "○ " + execution_queue[i].queue_status;  // Empty circle
        }
        data.push_back(status_display);

        data.push_back(execution_queue[i].custom_cli_args);

        queue_list_ctrl->AppendItem(data);
    }
}

int TemplateMatchQueueManager::GetSelectedRow() {
    wxDataViewItem item = queue_list_ctrl->GetSelection();
    if (item.IsOk()) {
        return queue_list_ctrl->ItemToRow(item);
    }
    return -1;
}

void TemplateMatchQueueManager::RunSelectedJob() {
    int selected = GetSelectedRow();
    if (selected >= 0 && selected < execution_queue.size()) {
        if (execution_queue[selected].queue_status == "pending") {
            ExecuteJob(execution_queue[selected]);
        }
    }
}

void TemplateMatchQueueManager::RunAllJobs() {
    MyPrintWithDetails("=== RUN ALL JOBS INITIATED ===");

    // Print comprehensive queue state information
    MyDebugPrint("Queue state summary:");
    MyDebugPrint("  Total jobs in queue: %zu", execution_queue.size());
    MyDebugPrint("  Currently running job ID: %ld", currently_running_id);
    MyDebugPrint("  Is job running: %s", IsJobRunning() ? "YES" : "NO");

    // Count jobs by status
    int pending_count = 0;
    int running_count = 0;
    int complete_count = 0;
    int failed_count = 0;

    MyDebugPrint("Individual job details:");
    for (size_t i = 0; i < execution_queue.size(); ++i) {
        const auto& job = execution_queue[i];
        MyDebugPrint("  [%zu] ID:%ld Name:'%s' Status:'%s' ImageGroup:%d RefVol:%d",
                     i, job.template_match_id, job.job_name.mb_str().data(),
                     job.queue_status.mb_str().data(), job.image_group_id, job.reference_volume_asset_id);

        if (job.queue_status == "pending") pending_count++;
        else if (job.queue_status == "running") running_count++;
        else if (job.queue_status == "complete") complete_count++;
        else if (job.queue_status == "failed") failed_count++;
    }

    MyDebugPrint("Status counts: pending=%d, running=%d, complete=%d, failed=%d",
                 pending_count, running_count, complete_count, failed_count);

    // Validate preconditions for batch execution
    if (execution_queue.empty()) {
        MyPrintWithDetails("RunAllJobs called but queue is empty - nothing to do");
        return;
    }

    if (pending_count == 0) {
        MyPrintWithDetails("RunAllJobs called but no pending jobs found - nothing to do");
        return;
    }

    if (IsJobRunning()) {
        MyPrintWithDetails("RunAllJobs called but job %ld is already running - deferring", currently_running_id);
        return;
    }

    MyDebugPrint("=== PROCEEDING TO START BATCH EXECUTION ===");

    // Start running the first pending job
    // When it completes, RunNextJob will be called to continue
    RunNextJob();
}

void TemplateMatchQueueManager::RunNextJob() {
    MyPrintWithDetails("=== RUN NEXT JOB CALLED ===");

    // Check if a job is already running
    if (IsJobRunning()) {
        MyDebugPrint("Job %ld is already running - skipping RunNextJob", currently_running_id);
        return;
    }

    MyDebugPrint("No job currently running - searching for next pending job");

    // Find and run the next pending job
    TemplateMatchQueueItem* next_job = GetNextPendingJob();
    if (next_job != nullptr) {
        MyDebugPrint("Found next pending job:");
        MyDebugPrint("  ID: %ld", next_job->template_match_id);
        MyDebugPrint("  Name: '%s'", next_job->job_name.mb_str().data());
        MyDebugPrint("  Status: '%s'", next_job->queue_status.mb_str().data());
        MyDebugPrint("  Image Group: %d", next_job->image_group_id);
        MyDebugPrint("  Reference Volume: %d", next_job->reference_volume_asset_id);
        MyDebugPrint("=== EXECUTING JOB %ld ===", next_job->template_match_id);

        ExecuteJob(*next_job);
    } else {
        MyPrintWithDetails("No pending jobs found - batch execution complete or no jobs to run");
    }
}

bool TemplateMatchQueueManager::ExecuteJob(TemplateMatchQueueItem& job_to_run) {
    // Validate job parameters before execution
    MyDebugAssertTrue(job_to_run.template_match_id >= 0, "Cannot execute job with invalid template_match_id: %ld", job_to_run.template_match_id);
    MyDebugAssertTrue(job_to_run.queue_status == "pending", "Cannot execute job with status '%s', must be 'pending'", job_to_run.queue_status.mb_str().data());
    MyDebugAssertFalse(job_to_run.job_name.IsEmpty(), "Cannot execute job with empty job_name");
    MyDebugAssertTrue(job_to_run.image_group_id >= 0, "Cannot execute job with invalid image_group_id: %d", job_to_run.image_group_id);
    MyDebugAssertTrue(job_to_run.reference_volume_asset_id >= 0, "Cannot execute job with invalid reference_volume_asset_id: %d", job_to_run.reference_volume_asset_id);

    // Check if another job is already running
    MyDebugAssertFalse(IsJobRunning(), "Attempted to execute job %ld while job %ld is already running", job_to_run.template_match_id, currently_running_id);
    if (IsJobRunning()) {
        wxMessageBox("A job is already running. Please wait for it to complete.",
                    "Job Running", wxOK | wxICON_WARNING);
        return false;
    }

    // Update status to running
    UpdateJobStatus(job_to_run.template_match_id, "running");
    currently_running_id = job_to_run.template_match_id;

    // Verify state change was successful
    MyDebugAssertTrue(currently_running_id == job_to_run.template_match_id, "Failed to set currently_running_id correctly");
    MyDebugAssertTrue(IsJobRunning(), "Job should be marked as running after status update");

    // Use the stored MatchTemplatePanel pointer to execute the job
    MyDebugAssertTrue(match_template_panel_ptr != nullptr, "match_template_panel_ptr is null - cannot execute jobs");

    if (match_template_panel_ptr) {
        // Store the queue job ID so we can update its status when complete
        match_template_panel_ptr->running_queue_job_id = job_to_run.template_match_id;

        // Use the same 2-step process as StartEstimationClick:
        // 1. Setup job from queue item
        // 2. Execute current job

        wxPrintf("Setting up job %ld from queue item...\n", job_to_run.template_match_id);
        bool setup_success = match_template_panel_ptr->SetupJobFromQueueItem(job_to_run);

        if (setup_success) {
            wxPrintf("Executing job %ld...\n", job_to_run.template_match_id);
            bool execution_success = match_template_panel_ptr->ExecuteCurrentJob();

            if (execution_success) {
                wxPrintf("Job %ld started successfully\n", job_to_run.template_match_id);
                // Job status will be updated to "complete" when the job finishes via ProcessAllJobsFinished
                return true;
            } else {
                wxPrintf("Failed to start job %ld\n", job_to_run.template_match_id);
                UpdateJobStatus(job_to_run.template_match_id, "failed");
                currently_running_id = -1;
                match_template_panel_ptr->running_queue_job_id = -1;
                return false;
            }
        } else {
            wxPrintf("Failed to setup job %ld\n", job_to_run.template_match_id);
            UpdateJobStatus(job_to_run.template_match_id, "failed");
            currently_running_id = -1;
            match_template_panel_ptr->running_queue_job_id = -1;
            return false;
        }
    } else {
        // Critical failure - template panel not available
        MyAssertTrue(false, "Critical error: match_template_panel not available for job execution");
        wxPrintf("Error: match_template_panel not available\n");
        UpdateJobStatus(job_to_run.template_match_id, "failed");
        currently_running_id = -1;
    }

    return true;
}

bool TemplateMatchQueueManager::IsJobRunning() {
    return currently_running_id != -1;
}

bool TemplateMatchQueueManager::IsJobRunningStatic() {
    return currently_running_id != -1;
}

void TemplateMatchQueueManager::UpdateJobStatus(long template_match_id, const wxString& new_status) {
    MyDebugAssertTrue(template_match_id >= 0, "Invalid template_match_id in UpdateJobStatus: %ld", template_match_id);
    MyDebugAssertTrue(new_status == "pending" || new_status == "running" || new_status == "complete" || new_status == "failed",
                     "Invalid new_status in UpdateJobStatus: %s", new_status.mb_str().data());

    bool found_job = false;
    for (auto& item : execution_queue) {
        if (item.template_match_id == template_match_id) {
            // Validate status transitions
            MyDebugAssertTrue(item.queue_status != new_status, "Attempted to set status to same value: %s", new_status.mb_str().data());

            // Validate allowed transitions
            if (item.queue_status == "running" && (new_status == "complete" || new_status == "failed")) {
                // Valid: running -> complete/failed
                MyDebugAssertTrue(currently_running_id == template_match_id, "Status change from running but template_match_id %ld != currently_running_id %ld", template_match_id, currently_running_id);
            } else if (item.queue_status == "pending" && new_status == "running") {
                // Valid: pending -> running
                MyDebugAssertFalse(IsJobRunning(), "Cannot start job %ld when job %ld is already running", template_match_id, currently_running_id);
            } else {
                MyDebugAssertTrue(false, "Invalid status transition: %s -> %s for job %ld",
                                 item.queue_status.mb_str().data(), new_status.mb_str().data(), template_match_id);
            }

            item.queue_status = new_status;
            found_job = true;
            break;
        }
    }

    MyDebugAssertTrue(found_job, "Job with template_match_id %ld not found in queue for status update", template_match_id);

    UpdateQueueDisplay();
    SaveQueueToDatabase();
}

void TemplateMatchQueueManager::SetCurrentlyRunningIdStatic(long template_match_id) {
    MyDebugAssertTrue(template_match_id >= 0, "Invalid template_match_id in SetCurrentlyRunningIdStatic: %ld", template_match_id);

    wxPrintf("SetCurrentlyRunningIdStatic: setting currently_running_id from %ld to %ld\n",
             currently_running_id, template_match_id);

    currently_running_id = template_match_id;
}

void TemplateMatchQueueManager::UpdateJobStatusStatic(long template_match_id, const wxString& new_status) {
    MyDebugAssertTrue(template_match_id >= 0, "Invalid template_match_id in UpdateJobStatusStatic: %ld", template_match_id);
    MyDebugAssertTrue(new_status == "pending" || new_status == "running" || new_status == "complete" || new_status == "failed",
                     "Invalid new_status in UpdateJobStatusStatic: %s", new_status.mb_str().data());

    // This method is typically called from job completion callbacks
    // Most common case is running -> complete/failed
    if (new_status == "complete" || new_status == "failed") {
        MyDebugAssertTrue(currently_running_id == template_match_id,
                         "Job completion for %ld but currently_running_id is %ld", template_match_id, currently_running_id);
    }

    bool found_job = false;
    for (auto& item : execution_queue) {
        if (item.template_match_id == template_match_id) {
            // Basic transition validation
            MyDebugAssertTrue(item.queue_status != new_status, "Static update: Attempted to set status to same value: %s", new_status.mb_str().data());

            item.queue_status = new_status;
            found_job = true;
            // Can't call UpdateQueueDisplay() here as we don't have a UI instance
            // The next time a queue manager is opened, it will show the updated status
            break;
        }
    }

    MyDebugAssertTrue(found_job, "Static update: Job with template_match_id %ld not found in queue", template_match_id);

    // Clear the currently running ID if this job was running and is now complete/failed
    if (currently_running_id == template_match_id &&
        (new_status == "complete" || new_status == "failed")) {
        currently_running_id = -1;
        MyDebugAssertTrue(currently_running_id == -1, "currently_running_id should be cleared after job completion");
    }

    // TODO: Check if there are any running queue manager instances and trigger next job
    // For now, this will be handled when the queue manager is opened again
}

TemplateMatchQueueItem* TemplateMatchQueueManager::GetNextPendingJob() {
    MyDebugAssertFalse(IsJobRunning(), "GetNextPendingJob called while job %ld is running", currently_running_id);

    for (auto& item : execution_queue) {
        if (item.queue_status == "pending") {
            // Validate the job we're about to return
            MyDebugAssertTrue(item.template_match_id >= 0, "Found pending job with invalid template_match_id: %ld", item.template_match_id);
            MyDebugAssertFalse(item.job_name.IsEmpty(), "Found pending job with empty job_name");
            return &item;
        }
    }
    return nullptr;
}

bool TemplateMatchQueueManager::HasPendingJobs() {
    for (const auto& item : execution_queue) {
        if (item.queue_status == "pending") {
            return true;
        }
    }
    return false;
}

void TemplateMatchQueueManager::OnRunSelectedClick(wxCommandEvent& event) {
    RunSelectedJob();
}

void TemplateMatchQueueManager::OnRunAllClick(wxCommandEvent& event) {
    RunAllJobs();
}

void TemplateMatchQueueManager::OnClearQueueClick(wxCommandEvent& event) {
    wxMessageDialog dialog(this, "Clear all pending jobs from the queue?",
                          "Confirm Clear", wxYES_NO | wxICON_QUESTION);
    if (dialog.ShowModal() == wxID_YES) {
        ClearQueue();
    }
}

void TemplateMatchQueueManager::OnMoveUpClick(wxCommandEvent& event) {
    int selected = GetSelectedRow();
    if (selected > 0) {
        MoveItemUp(selected);
        queue_list_ctrl->SelectRow(selected - 1);
    }
}

void TemplateMatchQueueManager::OnMoveDownClick(wxCommandEvent& event) {
    int selected = GetSelectedRow();
    if (selected >= 0 && selected < execution_queue.size() - 1) {
        MoveItemDown(selected);
        queue_list_ctrl->SelectRow(selected + 1);
    }
}

void TemplateMatchQueueManager::OnRemoveSelectedClick(wxCommandEvent& event) {
    int selected = GetSelectedRow();
    if (selected >= 0) {
        wxString job_name = execution_queue[selected].job_name;
        wxMessageDialog dialog(this,
                              wxString::Format("Remove job '%s' from the queue?", job_name),
                              "Confirm Remove", wxYES_NO | wxICON_QUESTION);
        if (dialog.ShowModal() == wxID_YES) {
            RemoveFromQueue(selected);
        }
    }
}

void TemplateMatchQueueManager::OnSelectionChanged(wxDataViewEvent& event) {
    int selected = GetSelectedRow();
    bool has_selection = (selected >= 0);

    // Validate GUI components are available
    MyDebugAssertTrue(move_up_button != nullptr, "move_up_button is null in OnSelectionChanged");
    MyDebugAssertTrue(move_down_button != nullptr, "move_down_button is null in OnSelectionChanged");
    MyDebugAssertTrue(remove_selected_button != nullptr, "remove_selected_button is null in OnSelectionChanged");
    MyDebugAssertTrue(run_selected_button != nullptr, "run_selected_button is null in OnSelectionChanged");

    // Validate selection bounds
    if (has_selection) {
        MyDebugAssertTrue(selected < execution_queue.size(), "Selected row %d >= queue size %zu", selected, execution_queue.size());
    }

    move_up_button->Enable(has_selection && selected > 0);
    move_down_button->Enable(has_selection && selected < execution_queue.size() - 1);
    remove_selected_button->Enable(has_selection);
    run_selected_button->Enable(has_selection &&
                               execution_queue[selected].queue_status == "pending");

    // Populate the GUI with the selected item's parameters
    if (has_selection && selected < execution_queue.size()) {
        // Import the MatchTemplatePanel header to get access to the panel
        extern MatchTemplatePanel* match_template_panel;
        MyDebugAssertTrue(match_template_panel != nullptr, "match_template_panel is null when trying to populate GUI");

        if (match_template_panel) {
            match_template_panel->PopulateGuiFromQueueItem(execution_queue[selected]);
        }
    }
}

void TemplateMatchQueueManager::OnItemValueChanged(wxDataViewEvent& event) {
    // Get the row and column that was edited
    wxDataViewItem item = event.GetItem();
    MyDebugAssertTrue(item.IsOk(), "Invalid wxDataViewItem in OnItemValueChanged");

    if (!item.IsOk()) {
        return;
    }

    MyDebugAssertTrue(queue_list_ctrl != nullptr, "queue_list_ctrl is null in OnItemValueChanged");

    int row = queue_list_ctrl->ItemToRow(item);
    int col = event.GetColumn();

    MyDebugAssertTrue(row >= 0, "Invalid row %d in OnItemValueChanged", row);
    MyDebugAssertTrue(row < execution_queue.size(), "Row %d >= queue size %zu in OnItemValueChanged", row, execution_queue.size());
    MyDebugAssertTrue(col >= 0, "Invalid column %d in OnItemValueChanged", col);

    // Check if this is the CLI Args column (column 3)
    if (col == 3 && row >= 0 && row < execution_queue.size()) {
        // Don't allow editing running jobs
        MyDebugAssertTrue(execution_queue[row].queue_status != "running",
                         "Cannot edit CLI args for running job (ID: %ld)", execution_queue[row].template_match_id);

        // Get the new value
        wxVariant value;
        queue_list_ctrl->GetValue(value, row, col);
        wxString new_cli_args = value.GetString();

        // TODO: Validate CLI flags here before accepting changes
        // Should check for valid flags, syntax, and dangerous options

        // Update the queue item
        execution_queue[row].custom_cli_args = new_cli_args;

        // Save to database if needed
        SaveQueueToDatabase();
    }
}

void TemplateMatchQueueManager::LoadQueueFromDatabase() {
    MyDebugPrint("LoadQueueFromDatabase called. match_template_panel_ptr=%p", match_template_panel_ptr);
    if (match_template_panel_ptr && main_frame && main_frame->current_project.is_open) {
        MyDebugPrint("Loading queue from database...");
        execution_queue.clear();

        // Get all queue IDs from database in order
        wxArrayLong queue_ids = main_frame->current_project.database.GetQueuedTemplateMatchIDs();
        MyDebugPrint("Found %zu queue items in database", queue_ids.GetCount());

        // Load each queue item from database
        for (size_t i = 0; i < queue_ids.GetCount(); i++) {
            TemplateMatchQueueItem temp_item;
            temp_item.template_match_id = queue_ids[i];

            // Load item details from database
            bool success = main_frame->current_project.database.GetQueueItemByID(
                queue_ids[i],
                temp_item.job_name,
                temp_item.queue_status,
                temp_item.custom_cli_args,
                temp_item.image_group_id,
                temp_item.reference_volume_asset_id,
                temp_item.run_profile_id,
                temp_item.use_gpu,
                temp_item.use_fast_fft,
                temp_item.symmetry,
                temp_item.pixel_size,
                temp_item.voltage,
                temp_item.spherical_aberration,
                temp_item.amplitude_contrast,
                temp_item.defocus1,
                temp_item.defocus2,
                temp_item.defocus_angle,
                temp_item.phase_shift,
                temp_item.low_resolution_limit,
                temp_item.high_resolution_limit,
                temp_item.out_of_plane_angular_step,
                temp_item.in_plane_angular_step,
                temp_item.defocus_search_range,
                temp_item.defocus_step,
                temp_item.pixel_size_search_range,
                temp_item.pixel_size_step,
                temp_item.refinement_threshold,
                temp_item.ref_box_size_in_angstroms,
                temp_item.mask_radius,
                temp_item.min_peak_radius,
                temp_item.xy_change_threshold,
                temp_item.exclude_above_xy_threshold);

            if (success) {
                execution_queue.push_back(temp_item);
            }
        }

        // Mark that we've loaded from database
        needs_database_load = false;

        // Update display
        UpdateQueueDisplay();
    }
}

void TemplateMatchQueueManager::SaveQueueToDatabase() {
    if (match_template_panel_ptr && main_frame && main_frame->current_project.is_open) {
        // Update status for all items in queue
        for (const auto& item : execution_queue) {
            main_frame->current_project.database.UpdateQueueStatus(item.template_match_id, item.queue_status);
        }
    }
}

void TemplateMatchQueueManager::ValidateQueueConsistency() const {
    int running_jobs_count = 0;
    long found_running_id = -1;

    for (size_t i = 0; i < execution_queue.size(); ++i) {
        const auto& item = execution_queue[i];

        // Validate basic item consistency
        MyDebugAssertTrue(item.template_match_id >= 0, "Queue item %zu has invalid template_match_id: %ld", i, item.template_match_id);
        MyDebugAssertFalse(item.job_name.IsEmpty(), "Queue item %zu (ID: %ld) has empty job_name", i, item.template_match_id);
        MyDebugAssertTrue(item.queue_status == "pending" || item.queue_status == "running" ||
                         item.queue_status == "complete" || item.queue_status == "failed",
                         "Queue item %zu (ID: %ld) has invalid status: %s", i, item.template_match_id, item.queue_status.mb_str().data());

        // Track running jobs
        if (item.queue_status == "running") {
            running_jobs_count++;
            found_running_id = item.template_match_id;
        }

        // Check for duplicate IDs
        for (size_t j = i + 1; j < execution_queue.size(); ++j) {
            MyDebugAssertTrue(execution_queue[j].template_match_id != item.template_match_id,
                             "Duplicate template_match_id %ld found at indices %zu and %zu", item.template_match_id, i, j);
        }
    }

    // Validate running state consistency
    MyDebugAssertTrue(running_jobs_count <= 1, "Multiple running jobs found (%d), should be at most 1", running_jobs_count);

    if (running_jobs_count == 1) {
        MyDebugAssertTrue(currently_running_id == found_running_id,
                         "currently_running_id (%ld) doesn't match running job ID (%ld)", currently_running_id, found_running_id);
        MyDebugAssertTrue(IsJobRunningStatic(), "IsJobRunning() returns false but running job exists");
    } else {
        MyDebugAssertTrue(currently_running_id == -1, "currently_running_id is %ld but no running jobs found", currently_running_id);
        MyDebugAssertFalse(IsJobRunningStatic(), "IsJobRunning() returns true but no running jobs found");
    }
}

void TemplateMatchQueueManager::ContinueQueueExecution() {
    // Static method to continue queue execution after a job completes
    // This is called from ProcessAllJobsFinished to continue with the next job

    if (IsJobRunningStatic()) {
        // A job is already running, don't start another
        MyDebugPrint("Job is still running, not continuing queue execution");
        return;
    }

    // Use the active instance to continue execution
    if (active_instance != nullptr) {
        MyDebugPrint("Using active queue manager instance to continue execution");
        active_instance->RunNextJob();
    } else {
        MyDebugPrint("No active queue manager instance - cannot continue execution");
        MyDebugPrint("This may happen if the queue dialog was closed");
    }
}