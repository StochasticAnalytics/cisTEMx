#include "../core/gui_core_headers.h"
#include "TemplateMatchQueueManager.h"

BEGIN_EVENT_TABLE(TemplateMatchQueueManager, wxPanel)
    EVT_BUTTON(wxID_ANY, TemplateMatchQueueManager::OnRunSelectedClick)
    EVT_DATAVIEW_SELECTION_CHANGED(wxID_ANY, TemplateMatchQueueManager::OnSelectionChanged)
END_EVENT_TABLE()

TemplateMatchQueueManager::TemplateMatchQueueManager(wxWindow* parent)
    : wxPanel(parent, wxID_ANY) {

    currently_running_id = -1;

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
    queue_list_ctrl->AppendTextColumn("CLI Args", wxDATAVIEW_CELL_INERT, 140);

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

    // Load any existing queue items from database
    LoadQueueFromDatabase();
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
    // Only clear pending items, keep running and completed
    auto it = execution_queue.begin();
    while (it != execution_queue.end()) {
        if (it->queue_status == "pending") {
            it = execution_queue.erase(it);
        } else {
            ++it;
        }
    }
    UpdateQueueDisplay();
    SaveQueueToDatabase();
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
            // Mark as running and execute
            UpdateJobStatus(execution_queue[selected].template_match_id, "running");
            // TODO: Trigger actual job execution
        }
    }
}

void TemplateMatchQueueManager::RunAllJobs() {
    // Find first pending job and run it
    for (auto& item : execution_queue) {
        if (item.queue_status == "pending") {
            UpdateJobStatus(item.template_match_id, "running");
            // TODO: Trigger actual job execution
            break;
        }
    }
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
}

void TemplateMatchQueueManager::LoadQueueFromDatabase() {
    // Load incomplete jobs from TEMPLATE_MATCH_LIST where QUEUE_STATUS != 'complete'
    extern MyMainFrame* main_frame;
    if (main_frame && main_frame->current_project.is_open) {
        // Query for incomplete jobs
        wxString query = "SELECT TEMPLATE_MATCH_ID, JOB_NAME, QUEUE_STATUS, CUSTOM_CLI_ARGS, "
                        "IMAGE_ASSET_ID, REFERENCE_VOLUME_ASSET_ID "
                        "FROM TEMPLATE_MATCH_LIST WHERE QUEUE_STATUS IN ('pending', 'running')";

        // TODO: Execute query and populate execution_queue
        // main_frame->current_project.database can be used to execute queries
        // For now, just update display
        UpdateQueueDisplay();
    }
}

void TemplateMatchQueueManager::SaveQueueToDatabase() {
    // Update QUEUE_STATUS in database for all items in queue
    extern MyMainFrame* main_frame;
    if (main_frame && main_frame->current_project.is_open) {
        for (const auto& item : execution_queue) {
            wxString update_query = wxString::Format(
                "UPDATE TEMPLATE_MATCH_LIST SET QUEUE_STATUS = '%s', CUSTOM_CLI_ARGS = '%s' "
                "WHERE TEMPLATE_MATCH_ID = %ld",
                item.queue_status, item.custom_cli_args, item.template_match_id);

            // TODO: Execute update query using main_frame->current_project.database
        }
    }
}