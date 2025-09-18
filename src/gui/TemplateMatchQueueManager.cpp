#include "../core/gui_core_headers.h"
#include "TemplateMatchQueueManager.h"
#include "MatchTemplatePanel.h"

// No more static members needed with unified architecture

BEGIN_EVENT_TABLE(TemplateMatchQueueManager, wxPanel)
    EVT_BUTTON(wxID_ANY, TemplateMatchQueueManager::OnRunSelectedClick)
    EVT_DATAVIEW_SELECTION_CHANGED(wxID_ANY, TemplateMatchQueueManager::OnSelectionChanged)
    EVT_DATAVIEW_ITEM_VALUE_CHANGED(wxID_ANY, TemplateMatchQueueManager::OnItemValueChanged)
END_EVENT_TABLE()

TemplateMatchQueueManager::TemplateMatchQueueManager(wxWindow* parent, MatchTemplatePanel* match_template_panel)
    : wxPanel(parent, wxID_ANY), match_template_panel_ptr(match_template_panel), currently_running_id(-1) {

    // Initialize state variables
    needs_database_load = true;  // Need to load from database on first access
    execution_in_progress = false;  // No jobs running initially

    // Create the main sizer
    wxBoxSizer* main_sizer = new wxBoxSizer(wxVERTICAL);

    // Create the list control
    queue_list_ctrl = new wxDataViewListCtrl(this, wxID_ANY,
                                            wxDefaultPosition, wxSize(500, 200),
                                            wxDV_MULTIPLE | wxDV_ROW_LINES);

    // Add columns
    queue_list_ctrl->AppendTextColumn("Queue Order", wxDATAVIEW_CELL_INERT, 80);
    queue_list_ctrl->AppendTextColumn("ID", wxDATAVIEW_CELL_INERT, 60);
    queue_list_ctrl->AppendTextColumn("Job Name", wxDATAVIEW_CELL_INERT, 200);
    queue_list_ctrl->AppendTextColumn("Status", wxDATAVIEW_CELL_INERT, 100);
    queue_list_ctrl->AppendTextColumn("CLI Args", wxDATAVIEW_CELL_EDITABLE, 140);  // Make editable

    // Create button panel
    wxPanel* button_panel = new wxPanel(this, wxID_ANY);
    wxBoxSizer* button_sizer = new wxBoxSizer(wxHORIZONTAL);

    // Create position setting controls
    wxStaticText* position_label = new wxStaticText(button_panel, wxID_ANY, "Queue Position:");
    position_input = new wxTextCtrl(button_panel, wxID_ANY, "", wxDefaultPosition, wxSize(60, -1));
    set_position_button = new wxButton(button_panel, wxID_ANY, "Set Position");

    run_selected_button = new wxButton(button_panel, wxID_ANY, "Run Selected");
    remove_selected_button = new wxButton(button_panel, wxID_ANY, "Remove");
    clear_queue_button = new wxButton(button_panel, wxID_ANY, "Clear Queue");

    button_sizer->Add(position_label, 0, wxALL | wxALIGN_CENTER_VERTICAL, 5);
    button_sizer->Add(position_input, 0, wxALL, 5);
    button_sizer->Add(set_position_button, 0, wxALL, 5);
    button_sizer->AddSpacer(20);
    button_sizer->Add(run_selected_button, 0, wxALL, 5);
    button_sizer->AddSpacer(20);
    button_sizer->Add(remove_selected_button, 0, wxALL, 5);
    button_sizer->Add(clear_queue_button, 0, wxALL, 5);

    button_panel->SetSizer(button_sizer);

    // Add to main sizer
    main_sizer->Add(queue_list_ctrl, 1, wxEXPAND | wxALL, 5);
    main_sizer->Add(button_panel, 0, wxEXPAND | wxALL, 5);

    SetSizer(main_sizer);

    // Connect events
    set_position_button->Bind(wxEVT_BUTTON, &TemplateMatchQueueManager::OnSetPositionClick, this);
    run_selected_button->Bind(wxEVT_BUTTON, &TemplateMatchQueueManager::OnRunSelectedClick, this);
    remove_selected_button->Bind(wxEVT_BUTTON, &TemplateMatchQueueManager::OnRemoveSelectedClick, this);
    clear_queue_button->Bind(wxEVT_BUTTON, &TemplateMatchQueueManager::OnClearQueueClick, this);

    // Don't load from database in constructor - it may be called during workflow switch
    // when main_frame is in an inconsistent state

    // No longer need active instance tracking with unified architecture

    // Display any existing queue items
    UpdateQueueDisplay();
}

TemplateMatchQueueManager::~TemplateMatchQueueManager() {
    // Clear completion callback if we set one
    if (match_template_panel_ptr) {
        match_template_panel_ptr->ClearQueueCompletionCallback();
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

        // Create a copy with the new database ID and assign queue order
        TemplateMatchQueueItem new_item = item;
        new_item.template_match_id = new_queue_id;

        // Assign next available queue order (find highest current order + 1)
        int max_order = 0;
        for (const auto& existing_item : execution_queue) {
            if (existing_item.queue_order > max_order) {
                max_order = existing_item.queue_order;
            }
        }
        new_item.queue_order = max_order + 1;

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

void TemplateMatchQueueManager::SetJobPosition(int job_index, int new_position) {
    // Validate inputs
    MyDebugAssertTrue(job_index >= 0 && job_index < execution_queue.size(),
                     "Invalid job_index: %d (queue size: %zu)", job_index, execution_queue.size());
    MyDebugAssertTrue(new_position >= 1 && new_position <= execution_queue.size(),
                     "Invalid new_position: %d (valid range: 1-%zu)", new_position, execution_queue.size());

    int current_position = execution_queue[job_index].queue_order;
    if (current_position == new_position) {
        return; // No change needed
    }

    // Update queue orders for all affected jobs
    if (current_position < new_position) {
        // Moving job to a later position - decrement jobs between current and new position
        for (auto& item : execution_queue) {
            if (item.queue_order > current_position && item.queue_order <= new_position) {
                item.queue_order--;
            }
        }
    } else {
        // Moving job to an earlier position - increment jobs between new and current position
        for (auto& item : execution_queue) {
            if (item.queue_order >= new_position && item.queue_order < current_position) {
                item.queue_order++;
            }
        }
    }

    // Set the moved job to its new position
    execution_queue[job_index].queue_order = new_position;

    wxPrintf("SetJobPosition: Moved job %ld from position %d to %d\n",
             execution_queue[job_index].template_match_id, current_position, new_position);
}

void TemplateMatchQueueManager::ProgressQueue() {
    // Find the job with queue_order = 1 (next to run)
    TemplateMatchQueueItem* next_job = nullptr;
    for (auto& item : execution_queue) {
        if (item.queue_order == 1 && item.queue_status == "pending") {
            next_job = &item;
            break;
        }
    }

    if (next_job) {
        // Set the next job to running (queue_order = 0)
        next_job->queue_order = 0;
        next_job->queue_status = "running";
        currently_running_id = next_job->template_match_id;

        // Decrement all other jobs' queue orders
        for (auto& item : execution_queue) {
            if (item.queue_order > 0) {
                item.queue_order--;
            }
        }

        wxPrintf("ProgressQueue: Started job %ld, decremented all other positions\n", next_job->template_match_id);

        // Execute the next job
        ExecuteJob(*next_job);
    } else {
        wxPrintf("ProgressQueue: No pending jobs found with queue_order = 1\n");
    }
}

// Old MoveItem methods removed - replaced with queue order system

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

    // Preserve current selections before rebuilding
    wxDataViewItemArray selected_items;
    std::vector<int> selected_template_ids;
    int selection_count = queue_list_ctrl->GetSelections(selected_items);

    for (int i = 0; i < selection_count; i++) {
        int row = queue_list_ctrl->ItemToRow(selected_items[i]);
        if (row != wxNOT_FOUND && row < int(execution_queue.size())) {
            selected_template_ids.push_back(execution_queue[row].template_match_id);
        }
    }

    queue_list_ctrl->DeleteAllItems();

    for (size_t i = 0; i < execution_queue.size(); ++i) {
        MyDebugAssertTrue(i < execution_queue.size(), "Queue index %zu out of bounds (size: %zu)", i, execution_queue.size());
        wxVector<wxVariant> data;
        data.push_back(wxString::Format("%d", execution_queue[i].queue_order));
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

    // Restore selections after rebuilding
    for (int template_id : selected_template_ids) {
        for (size_t i = 0; i < execution_queue.size(); ++i) {
            if (execution_queue[i].template_match_id == template_id) {
                wxDataViewItem item = queue_list_ctrl->RowToItem(int(i));
                if (item.IsOk()) {
                    queue_list_ctrl->Select(item);
                    wxPrintf("Restored selection for job %d (row %zu)\n", template_id, i);
                }
                break;
            }
        }
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
    // Get currently selected UI items
    wxDataViewItemArray selected_items;
    int selection_count = queue_list_ctrl->GetSelections(selected_items);

    if (selection_count == 0) {
        wxMessageBox("No jobs selected for execution", "No Selection", wxOK | wxICON_WARNING);
        return;
    }

    if (IsJobRunning()) {
        wxMessageBox("A job is already running. Please wait for it to complete.", "Job Running", wxOK | wxICON_WARNING);
        return;
    }

    // Collect selected job indices and reorder them to start from position 1
    std::vector<int> selected_indices;
    for (int i = 0; i < selection_count; i++) {
        int row = queue_list_ctrl->ItemToRow(selected_items[i]);
        if (row != wxNOT_FOUND && row < int(execution_queue.size())) {
            if (execution_queue[row].queue_status == "pending") {
                selected_indices.push_back(row);
            }
        }
    }

    if (selected_indices.empty()) {
        wxMessageBox("No pending jobs selected", "No Pending Jobs", wxOK | wxICON_WARNING);
        return;
    }

    // Reorder selected jobs to positions 1, 2, 3, etc.
    for (int i = 0; i < selected_indices.size(); i++) {
        SetJobPosition(selected_indices[i], i + 1);
    }

    // Start the first job (position 1 → 0)
    ProgressQueue();

    wxPrintf("Started execution of %zu selected jobs\n", selected_indices.size());
}

// RunAllJobs method removed - use Run Selected with multi-selection instead

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
    // Basic validation moved to TMP's ExecuteJob method

    // Check if another job is already running
    MyDebugAssertFalse(IsJobRunning(), "Attempted to execute job %ld while job %ld is already running", job_to_run.template_match_id, currently_running_id);
    if (IsJobRunning()) {
        wxMessageBox("A job is already running. Please wait for it to complete.",
                    "Job Running", wxOK | wxICON_WARNING);
        return false;
    }

    // Use the stored MatchTemplatePanel pointer to execute the job
    MyDebugAssertTrue(match_template_panel_ptr != nullptr, "match_template_panel_ptr is null - cannot execute jobs");

    if (match_template_panel_ptr) {
        // Register this queue manager for completion callbacks
        match_template_panel_ptr->SetQueueCompletionCallback(this);

        // Use TMP's unified ExecuteJob method
        wxPrintf("Executing job %ld via unified method...\n", job_to_run.template_match_id);
        bool execution_success = match_template_panel_ptr->ExecuteJob(&job_to_run);

        if (execution_success) {
            // Job executed successfully - now mark it as running and track it
            UpdateJobStatus(job_to_run.template_match_id, "running");
            currently_running_id = job_to_run.template_match_id;
            wxPrintf("Job %ld started successfully and marked as running\n", job_to_run.template_match_id);
            // Job status will be updated to "complete" when the job finishes via ProcessAllJobsFinished
            return true;
        } else {
            wxPrintf("Failed to start job %ld\n", job_to_run.template_match_id);
            UpdateJobStatus(job_to_run.template_match_id, "failed");
            return false;
        }
    } else {
        // Critical failure - template panel not available
        MyAssertTrue(false, "Critical error: match_template_panel not available for job execution");
        wxPrintf("Error: match_template_panel not available\n");
        UpdateJobStatus(job_to_run.template_match_id, "failed");
        currently_running_id = -1;
    }

    return false;
}

bool TemplateMatchQueueManager::IsJobRunning() const {
    return currently_running_id != -1;
}

// IsJobRunningStatic removed - no longer needed with unified architecture

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

// Static methods removed - no longer needed with unified architecture

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

// OnRunAllClick removed - use Run Selected with multi-selection instead

void TemplateMatchQueueManager::OnClearQueueClick(wxCommandEvent& event) {
    wxMessageDialog dialog(this, "Clear all pending jobs from the queue?",
                          "Confirm Clear", wxYES_NO | wxICON_QUESTION);
    if (dialog.ShowModal() == wxID_YES) {
        ClearQueue();
    }
}

void TemplateMatchQueueManager::OnSetPositionClick(wxCommandEvent& event) {
    int selected = GetSelectedRow();
    if (selected < 0) {
        wxMessageBox("Please select a job to reposition.", "No Selection", wxOK | wxICON_WARNING);
        return;
    }

    // Get the desired position from input
    wxString position_text = position_input->GetValue();
    long desired_position;
    if (!position_text.ToLong(&desired_position)) {
        wxMessageBox("Please enter a valid integer position.", "Invalid Input", wxOK | wxICON_ERROR);
        return;
    }

    // Validate position range
    int queue_size = execution_queue.size();
    if (desired_position < 1 || desired_position > queue_size) {
        wxMessageBox(wxString::Format("Position must be between 1 and %d", queue_size),
                     "Invalid Range", wxOK | wxICON_ERROR);
        return;
    }

    // Don't allow reordering running jobs
    if (execution_queue[selected].queue_status == "running") {
        wxMessageBox("Cannot reorder running jobs.", "Job Running", wxOK | wxICON_WARNING);
        return;
    }

    // Get current job and its current position
    int current_position = execution_queue[selected].queue_order;
    if (current_position == desired_position) {
        // No change needed
        return;
    }

    // Reorder jobs in the queue
    SetJobPosition(selected, desired_position);

    // Update display and save to database
    UpdateQueueDisplay();
    SaveQueueToDatabase();

    wxPrintf("Moved job from position %d to %d\n", current_position, desired_position);
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
    // Validate GUI components are available
    MyDebugAssertTrue(position_input != nullptr, "position_input is null in OnSelectionChanged");
    MyDebugAssertTrue(set_position_button != nullptr, "set_position_button is null in OnSelectionChanged");
    MyDebugAssertTrue(remove_selected_button != nullptr, "remove_selected_button is null in OnSelectionChanged");
    MyDebugAssertTrue(run_selected_button != nullptr, "run_selected_button is null in OnSelectionChanged");

    // Get current selection state, but don't interfere with execution queue
    bool has_selection = false;
    std::vector<int> current_ui_selection;  // Temporary selection for button enabling

    if (!execution_in_progress) {
        // Not executing - safe to update selection queue from UI
        PopulateSelectionQueueFromUI();
        has_selection = !selected_jobs_for_execution.empty();

        // Copy selection queue for button logic
        current_ui_selection.assign(selected_jobs_for_execution.begin(), selected_jobs_for_execution.end());
    } else {
        // Execution in progress - get current UI selection without modifying execution queue
        wxDataViewItemArray selected_items;
        int selection_count = queue_list_ctrl->GetSelections(selected_items);
        has_selection = (selection_count > 0);

        // Convert to template_match_ids for button logic
        for (int i = 0; i < selection_count; i++) {
            int row = queue_list_ctrl->ItemToRow(selected_items[i]);
            if (row != wxNOT_FOUND && row < int(execution_queue.size())) {
                current_ui_selection.push_back(execution_queue[row].template_match_id);
            }
        }
    }

    // Check selection status for button enabling
    bool any_running = false;
    bool all_pending = true;
    int first_selected_index = -1;
    int last_selected_index = -1;

    if (has_selection) {
        // Find the queue indices for the currently selected template_match_ids
        for (int template_match_id : current_ui_selection) {
            for (int i = 0; i < execution_queue.size(); i++) {
                if (execution_queue[i].template_match_id == template_match_id) {
                    if (first_selected_index == -1) first_selected_index = i;
                    last_selected_index = i;

                    if (IsJobRunning(i)) any_running = true;
                    if (!IsJobPending(i)) all_pending = false;
                    break;
                }
            }
        }
    }

    // Enable controls based on selection and job status
    // Position setting only works with single selection and non-running jobs
    bool single_selection = current_ui_selection.size() == 1;
    position_input->Enable(single_selection && !any_running);
    set_position_button->Enable(single_selection && !any_running);
    remove_selected_button->Enable(has_selection && !any_running);
    run_selected_button->Enable(has_selection && all_pending);

    // Update position input with current job's queue order if single selection
    if (single_selection && first_selected_index >= 0 && first_selected_index < execution_queue.size()) {
        position_input->SetValue(wxString::Format("%d", execution_queue[first_selected_index].queue_order));
    } else {
        position_input->Clear();
    }

    // Populate the GUI with the first selected item's parameters
    if (has_selection && first_selected_index >= 0 && first_selected_index < execution_queue.size()) {
        MyDebugAssertTrue(match_template_panel_ptr != nullptr, "match_template_panel_ptr is null when trying to populate GUI");

        if (match_template_panel_ptr) {
            match_template_panel_ptr->PopulateGuiFromQueueItem(execution_queue[first_selected_index]);
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

    // Check if this is the CLI Args column (column 4 after adding Queue Order)
    if (col == 4 && row >= 0 && row < execution_queue.size()) {
        // Don't allow editing running jobs
        MyDebugAssertTrue(!IsJobRunning(row),
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
        MyDebugAssertTrue(IsJobRunning(), "IsJobRunning() returns false but running job exists");
    } else {
        MyDebugAssertTrue(currently_running_id == -1, "currently_running_id is %ld but no running jobs found", currently_running_id);
        MyDebugAssertFalse(IsJobRunning(), "IsJobRunning() returns true but no running jobs found");
    }
}

void TemplateMatchQueueManager::ContinueQueueExecution() {
    // Instance method to continue queue execution after a job completes
    // This is called from ProcessAllJobsFinished to continue with the next job

    if (IsJobRunning()) {
        // A job is already running, don't start another
        MyDebugPrint("Job is still running, not continuing queue execution");
        return;
    }

    // Continue execution using this instance
    MyDebugPrint("Continuing queue execution with next job");
    RunNextJob();
}

void TemplateMatchQueueManager::OnJobCompleted(long template_match_id, bool success) {
    wxPrintf("Queue manager received job completion notification for job %ld (success: %s)\n",
             template_match_id, success ? "true" : "false");

    // Update job status in our queue
    const wxString& status = success ? "complete" : "failed";
    UpdateJobStatus(template_match_id, status);

    // Clear currently running ID since job is done
    currently_running_id = -1;

    // Update the display to show new status
    UpdateQueueDisplay();

    // Progress to next job in queue order system
    ProgressQueue();
}

void TemplateMatchQueueManager::PopulateSelectionQueueFromUI() {
    // Clear existing selection queue
    selected_jobs_for_execution.clear();

    // Get all selected items from the UI
    wxDataViewItemArray selected_items;
    int selection_count = queue_list_ctrl->GetSelections(selected_items);

    wxPrintf("Populating selection queue from %d selected UI items\n", selection_count);

    // Add template_match_id for each selected item to execution queue
    for (int i = 0; i < selection_count; i++) {
        int row = queue_list_ctrl->ItemToRow(selected_items[i]);
        if (row != wxNOT_FOUND && row < int(execution_queue.size())) {
            int template_match_id = execution_queue[row].template_match_id;
            selected_jobs_for_execution.push_back(template_match_id);
            wxPrintf("  Added job %d to selection queue\n", template_match_id);
        }
    }

    wxPrintf("Selection queue populated with %zu jobs\n", selected_jobs_for_execution.size());
}

void TemplateMatchQueueManager::RemoveJobFromSelectionQueue(int template_match_id) {
    auto it = std::find(selected_jobs_for_execution.begin(), selected_jobs_for_execution.end(), template_match_id);
    if (it != selected_jobs_for_execution.end()) {
        selected_jobs_for_execution.erase(it);
        wxPrintf("Removed job %d from selection queue (%zu remaining)\n", template_match_id, selected_jobs_for_execution.size());
    }
}

bool TemplateMatchQueueManager::HasJobsInSelectionQueue() const {
    return !selected_jobs_for_execution.empty();
}

void TemplateMatchQueueManager::RunNextSelectedJob() {
    if (selected_jobs_for_execution.empty()) {
        wxPrintf("No jobs in selection queue - execution complete\n");
        return;
    }

    if (IsJobRunning()) {
        wxPrintf("Job already running - cannot start next selected job\n");
        return;
    }

    // Get next job ID from selection queue
    int next_job_id = selected_jobs_for_execution.front();

    // Find the job in the main queue
    for (auto& job : execution_queue) {
        if (job.template_match_id == next_job_id) {
            if (job.queue_status == "pending") {
                wxPrintf("Starting next selected job %d\n", next_job_id);

                // Remove from selection queue (job is starting)
                selected_jobs_for_execution.pop_front();

                // Deselect the job in the UI since it's now running
                DeselectJobInUI(next_job_id);

                // Execute the job
                ExecuteJob(job);
                return;
            } else {
                wxPrintf("Skipping job %d - status is '%s', not 'pending'\n",
                        next_job_id, job.queue_status.mb_str().data());
                // Remove from selection queue and try next
                selected_jobs_for_execution.pop_front();
                RunNextSelectedJob();  // Recursive call to try next job
                return;
            }
        }
    }

    // Job not found - remove from selection queue and try next
    wxPrintf("Job %d not found in queue - removing from selection\n", next_job_id);
    selected_jobs_for_execution.pop_front();
    RunNextSelectedJob();  // Recursive call to try next job
}

void TemplateMatchQueueManager::DeselectJobInUI(int template_match_id) {
    // Find the row corresponding to this job ID
    for (size_t i = 0; i < execution_queue.size(); ++i) {
        if (execution_queue[i].template_match_id == template_match_id) {
            wxDataViewItem item = queue_list_ctrl->RowToItem(int(i));
            if (item.IsOk()) {
                queue_list_ctrl->Unselect(item);
                wxPrintf("Deselected job %d from UI (row %zu)\n", template_match_id, i);
            }
            return;
        }
    }
    wxPrintf("Could not find job %d to deselect in UI\n", template_match_id);
}