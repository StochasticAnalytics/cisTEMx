#include "../core/gui_core_headers.h"
#include "TemplateMatchQueueManager.h"
#include "MatchTemplatePanel.h"

// Define static member
std::deque<TemplateMatchQueueItem> TemplateMatchQueueManager::execution_queue;

BEGIN_EVENT_TABLE(TemplateMatchQueueManager, wxPanel)
    EVT_BUTTON(wxID_ANY, TemplateMatchQueueManager::OnRunSelectedClick)
    EVT_DATAVIEW_SELECTION_CHANGED(wxID_ANY, TemplateMatchQueueManager::OnSelectionChanged)
    EVT_DATAVIEW_ITEM_VALUE_CHANGED(wxID_ANY, TemplateMatchQueueManager::OnItemValueChanged)
END_EVENT_TABLE()

TemplateMatchQueueManager::TemplateMatchQueueManager(wxWindow* parent)
    : wxPanel(parent, wxID_ANY) {

    currently_running_id = -1;
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

    // Display any existing queue items
    UpdateQueueDisplay();
}

void TemplateMatchQueueManager::AddToQueue(const TemplateMatchQueueItem& item) {
    execution_queue.push_back(item);
    UpdateQueueDisplay();
    SaveQueueToDatabase();
}

void TemplateMatchQueueManager::RemoveFromQueue(int index) {
    if (index >= 0 && index < execution_queue.size()) {
        execution_queue.erase(execution_queue.begin() + index);
        UpdateQueueDisplay();
        SaveQueueToDatabase();
    }
}

void TemplateMatchQueueManager::ClearQueue() {
    // Only clear pending items, keep running, completed, and failed
    auto it = execution_queue.begin();
    while (it != execution_queue.end()) {
        if (it->queue_status == "pending") {
            // Also remove from database by setting status to 'cancelled'
            extern MyMainFrame* main_frame;
            if (main_frame && main_frame->current_project.is_open) {
                wxString update_query = wxString::Format(
                    "UPDATE TEMPLATE_MATCH_LIST SET QUEUE_STATUS = 'cancelled' "
                    "WHERE TEMPLATE_MATCH_ID = %ld", it->template_match_id);
                main_frame->current_project.database.ExecuteSQL(update_query.ToUTF8().data());
            }
            it = execution_queue.erase(it);
        } else {
            ++it;
        }
    }
    UpdateQueueDisplay();
}

void TemplateMatchQueueManager::MoveItemUp(int index) {
    if (index > 0 && index < execution_queue.size()) {
        std::swap(execution_queue[index], execution_queue[index - 1]);
        UpdateQueueDisplay();
        SaveQueueToDatabase();
    }
}

void TemplateMatchQueueManager::MoveItemDown(int index) {
    if (index >= 0 && index < execution_queue.size() - 1) {
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
    queue_list_ctrl->DeleteAllItems();

    for (size_t i = 0; i < execution_queue.size(); ++i) {
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
    // Start running the first pending job
    // When it completes, RunNextJob will be called to continue
    RunNextJob();
}

void TemplateMatchQueueManager::RunNextJob() {
    // Check if a job is already running
    if (IsJobRunning()) {
        return;
    }

    // Find and run the next pending job
    TemplateMatchQueueItem* next_job = GetNextPendingJob();
    if (next_job != nullptr) {
        ExecuteJob(*next_job);
    }
}

bool TemplateMatchQueueManager::ExecuteJob(TemplateMatchQueueItem& job_to_run) {
    // Check if another job is already running
    if (IsJobRunning()) {
        wxMessageBox("A job is already running. Please wait for it to complete.",
                    "Job Running", wxOK | wxICON_WARNING);
        return false;
    }

    // Update status to running
    UpdateJobStatus(job_to_run.template_match_id, "running");
    currently_running_id = job_to_run.template_match_id;

    // Get the MatchTemplatePanel to execute the job
    extern MatchTemplatePanel* match_template_panel;
    if (match_template_panel) {
        // Execute the job through the panel
        bool success = match_template_panel->RunQueuedTemplateMatch(job_to_run);

        if (!success) {
            // Job failed to start
            UpdateJobStatus(job_to_run.template_match_id, "failed");
            currently_running_id = -1;

            wxMessageBox(wxString::Format("Failed to start job: %s", job_to_run.job_name),
                        "Job Failed", wxOK | wxICON_ERROR);

            // Try to run the next job if we're in batch mode
            RunNextJob();
        }
        // Note: Status will be updated to "complete" when the job finishes
        // via the panel's job completion callback
    } else {
        wxPrintf("Error: match_template_panel not available\n");
        UpdateJobStatus(job_to_run.template_match_id, "failed");
        currently_running_id = -1;
    }

    return true;
}

bool TemplateMatchQueueManager::IsJobRunning() {
    return currently_running_id != -1;
}

void TemplateMatchQueueManager::UpdateJobStatus(long template_match_id, const wxString& new_status) {
    for (auto& item : execution_queue) {
        if (item.template_match_id == template_match_id) {
            item.queue_status = new_status;
            UpdateQueueDisplay();
            SaveQueueToDatabase();
            break;
        }
    }
}

TemplateMatchQueueItem* TemplateMatchQueueManager::GetNextPendingJob() {
    for (auto& item : execution_queue) {
        if (item.queue_status == "pending") {
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
        RemoveFromQueue(selected);
    }
}

void TemplateMatchQueueManager::OnSelectionChanged(wxDataViewEvent& event) {
    int selected = GetSelectedRow();
    bool has_selection = (selected >= 0);

    move_up_button->Enable(has_selection && selected > 0);
    move_down_button->Enable(has_selection && selected < execution_queue.size() - 1);
    remove_selected_button->Enable(has_selection);
    run_selected_button->Enable(has_selection &&
                               execution_queue[selected].queue_status == "pending");

    // Populate the GUI with the selected item's parameters
    if (has_selection && selected < execution_queue.size()) {
        // Import the MatchTemplatePanel header to get access to the panel
        extern MatchTemplatePanel* match_template_panel;
        if (match_template_panel) {
            match_template_panel->PopulateGuiFromQueueItem(execution_queue[selected]);
        }
    }
}

void TemplateMatchQueueManager::OnItemValueChanged(wxDataViewEvent& event) {
    // Get the row and column that was edited
    wxDataViewItem item = event.GetItem();
    if (!item.IsOk()) {
        return;
    }

    int row = queue_list_ctrl->ItemToRow(item);
    int col = event.GetColumn();

    // Check if this is the CLI Args column (column 3)
    if (col == 3 && row >= 0 && row < execution_queue.size()) {
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
    // TODO: Implement database loading once TEMPLATE_MATCH_LIST table is created
    // For now, use in-memory storage only
    needs_database_load = false;
    return;

    // Load all jobs that are not complete (pending, running, failed)
    extern MyMainFrame* main_frame;
    if (false && main_frame && main_frame->current_project.is_open) {
        execution_queue.clear();

        // Query for all non-complete jobs
        wxString query = "SELECT TEMPLATE_MATCH_ID, JOB_NAME, QUEUE_STATUS, CUSTOM_CLI_ARGS, "
                        "IMAGE_ASSET_ID, REFERENCE_VOLUME_ASSET_ID, USED_SYMMETRY, "
                        "USED_PIXEL_SIZE, USED_VOLTAGE, USED_SPHERICAL_ABERRATION, "
                        "USED_AMPLITUDE_CONTRAST, USED_DEFOCUS1, USED_DEFOCUS2, "
                        "USED_DEFOCUS_ANGLE, USED_PHASE_SHIFT, LOW_RESOLUTION_LIMIT, "
                        "HIGH_RESOLUTION_LIMIT, OUT_OF_PLANE_ANGULAR_STEP, "
                        "IN_PLANE_ANGULAR_STEP, DEFOCUS_SEARCH_RANGE, DEFOCUS_STEP, "
                        "PIXEL_SIZE_SEARCH_RANGE, PIXEL_SIZE_STEP, REFINEMENT_THRESHOLD, "
                        "REF_BOX_SIZE_IN_ANGSTROMS, MASK_RADIUS, MIN_PEAK_RADIUS, "
                        "XY_CHANGE_THRESHOLD, EXCLUDE_ABOVE_XY_THRESHOLD "
                        "FROM TEMPLATE_MATCH_LIST WHERE QUEUE_STATUS != 'complete' "
                        "ORDER BY TEMPLATE_MATCH_ID";

        // Execute query and populate execution_queue
        bool more_data;
        TemplateMatchQueueItem temp_item;

        more_data = main_frame->current_project.database.BeginBatchSelect(query.ToUTF8().data());

        while (more_data == true) {
            more_data = main_frame->current_project.database.GetFromBatchSelect(
                "ltttiitrrrrrrrrrrrrrrrrrrrib",
                &temp_item.template_match_id,
                &temp_item.job_name,
                &temp_item.queue_status,
                &temp_item.custom_cli_args,
                &temp_item.image_asset_id,
                &temp_item.reference_volume_asset_id,
                &temp_item.symmetry,
                &temp_item.pixel_size,
                &temp_item.voltage,
                &temp_item.spherical_aberration,
                &temp_item.amplitude_contrast,
                &temp_item.defocus1,
                &temp_item.defocus2,
                &temp_item.defocus_angle,
                &temp_item.phase_shift,
                &temp_item.low_resolution_limit,
                &temp_item.high_resolution_limit,
                &temp_item.out_of_plane_angular_step,
                &temp_item.in_plane_angular_step,
                &temp_item.defocus_search_range,
                &temp_item.defocus_step,
                &temp_item.pixel_size_search_range,
                &temp_item.pixel_size_step,
                &temp_item.refinement_threshold,
                &temp_item.ref_box_size_in_angstroms,
                &temp_item.mask_radius,
                &temp_item.min_peak_radius,
                &temp_item.xy_change_threshold,
                &temp_item.exclude_above_xy_threshold);

            if (more_data == false && temp_item.template_match_id != -1) {
                // Add the last item if it was successfully read
                execution_queue.push_back(temp_item);
            } else if (more_data == true) {
                // Add the item for all other successful reads
                execution_queue.push_back(temp_item);
            }
        }

        main_frame->current_project.database.EndBatchSelect();

        // Mark that we've loaded from database
        needs_database_load = false;

        // Update display
        UpdateQueueDisplay();
    }
}

void TemplateMatchQueueManager::SaveQueueToDatabase() {
    // TODO: Implement database saving once TEMPLATE_MATCH_LIST table is created
    // For now, use in-memory storage only
    return;

    // Update QUEUE_STATUS and CUSTOM_CLI_ARGS in database for all items in queue
    extern MyMainFrame* main_frame;
    if (false && main_frame && main_frame->current_project.is_open) {
        main_frame->current_project.database.Begin();

        for (const auto& item : execution_queue) {
            wxString update_query = wxString::Format(
                "UPDATE TEMPLATE_MATCH_LIST SET QUEUE_STATUS = '%s', CUSTOM_CLI_ARGS = '%s' "
                "WHERE TEMPLATE_MATCH_ID = %ld",
                item.queue_status, item.custom_cli_args, item.template_match_id);

            // Execute update query
            main_frame->current_project.database.ExecuteSQL(update_query.ToUTF8().data());
        }

        main_frame->current_project.database.Commit();

        // After saving, we need to reload from DB next time
        needs_database_load = true;
    }
}