#include "../core/gui_core_headers.h"
#include "TemplateMatchQueueManager.h"
#include "MatchTemplatePanel.h"
#include <algorithm>

// No more static members needed with unified architecture

BEGIN_EVENT_TABLE(TemplateMatchQueueManager, wxPanel)
EVT_BUTTON(wxID_ANY, TemplateMatchQueueManager::OnRunSelectedClick)
EVT_LIST_ITEM_SELECTED(wxID_ANY, TemplateMatchQueueManager::OnSelectionChanged)
EVT_LIST_ITEM_DESELECTED(wxID_ANY, TemplateMatchQueueManager::OnSelectionChanged)
EVT_LIST_BEGIN_DRAG(wxID_ANY, TemplateMatchQueueManager::OnBeginDrag)
END_EVENT_TABLE( )

TemplateMatchQueueManager::TemplateMatchQueueManager(wxWindow* parent, MatchTemplatePanel* match_template_panel)
    : wxPanel(parent, wxID_ANY), match_template_panel_ptr(match_template_panel), currently_running_id(-1), last_populated_queue_id(-1) {

    // Initialize state variables
    auto_progress_queue = true;  // Auto-progress to next job when one completes
    hide_completed_jobs = true;  // Hide completed searches by default
    gui_update_frozen   = false; // GUI updates allowed initially
    job_is_finalizing   = false; // No job in finalization initially
    drag_in_progress    = false;
    updating_display    = false; // Not updating display initially
    dragged_row         = -1;
    dragged_job_id      = -1;
    mouse_down          = false;

    // Create the main sizer
    wxBoxSizer* main_sizer = new wxBoxSizer(wxVERTICAL);

    // Create execution queue section (top)
    wxStaticText* execution_queue_label = new wxStaticText(this, wxID_ANY, "Execution Queue (searches will run in order - drag to reorder):");
    execution_queue_ctrl                = new wxListCtrl(this, wxID_ANY,
                                                         wxDefaultPosition, wxSize(700, 200),
                                                         wxLC_REPORT);

    // Add columns to execution queue (same as available queue, no queue order)
    execution_queue_ctrl->AppendColumn("Queue ID", wxLIST_FORMAT_LEFT, 70);
    execution_queue_ctrl->AppendColumn("Search ID", wxLIST_FORMAT_LEFT, 70);
    execution_queue_ctrl->AppendColumn("Search Name", wxLIST_FORMAT_LEFT, 180);
    execution_queue_ctrl->AppendColumn("Status", wxLIST_FORMAT_LEFT, 90);
    execution_queue_ctrl->AppendColumn("Progress", wxLIST_FORMAT_LEFT, 70);
    execution_queue_ctrl->AppendColumn("CLI Args", wxLIST_FORMAT_LEFT, 120);

    // wxListCtrl doesn't use EnableDragSource/EnableDropTarget - we'll implement manual drag and drop

    // Legacy compatibility - point to execution queue
    queue_list_ctrl = execution_queue_ctrl;

    // Create combined controls panel with Run Queue on left and Panel Display on right
    wxPanel*    controls_panel = new wxPanel(this, wxID_ANY);
    wxBoxSizer* controls_sizer = new wxBoxSizer(wxHORIZONTAL);

    // Run Queue button on the left
    run_selected_button = new wxButton(controls_panel, wxID_ANY, "Run Queue");

    // Panel display toggle on the right
    wxStaticText* display_label = new wxStaticText(controls_panel, wxID_ANY, "Panel Display:");
    panel_display_toggle = new wxToggleButton(controls_panel, wxID_ANY, "Show Input Panel",
                                              wxDefaultPosition, wxSize(150, -1));
    panel_display_toggle->SetToolTip("Toggle between Input and Progress panels");

    controls_sizer->Add(run_selected_button, 0, wxALL, 5);
    controls_sizer->AddStretchSpacer();
    controls_sizer->Add(display_label, 0, wxALIGN_CENTER_VERTICAL | wxRIGHT, 5);
    controls_sizer->Add(panel_display_toggle, 0, wxALIGN_CENTER_VERTICAL | wxRIGHT, 5);
    controls_panel->SetSizer(controls_sizer);

    // Create available searches section with hide completed checkbox
    wxPanel*    available_header_panel = new wxPanel(this, wxID_ANY);
    wxBoxSizer* available_header_sizer = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* available_jobs_label = new wxStaticText(available_header_panel, wxID_ANY, "Available Searches (not queued for execution):");
    hide_completed_checkbox = new wxCheckBox(available_header_panel, wxID_ANY, "Hide completed jobs");
    hide_completed_checkbox->SetValue(true);  // Checked by default

    available_header_sizer->Add(available_jobs_label, 0, wxALIGN_CENTER_VERTICAL | wxRIGHT, 10);
    available_header_sizer->Add(hide_completed_checkbox, 0, wxALIGN_CENTER_VERTICAL);
    available_header_sizer->AddStretchSpacer();
    available_header_panel->SetSizer(available_header_sizer);
    available_jobs_ctrl                = new wxListCtrl(this, wxID_ANY,
                                                        wxDefaultPosition, wxSize(700, 150),
                                                        wxLC_REPORT);

    // Set minimum size to ensure visibility
    available_jobs_ctrl->SetMinSize(wxSize(700, 150));
    wxPrintf("Created available_jobs_ctrl: %p with min size 700x150\n", available_jobs_ctrl); // Debug output

    // Add columns to available searches (same structure as execution queue)
    available_jobs_ctrl->AppendColumn("Queue ID", wxLIST_FORMAT_LEFT, 70);
    available_jobs_ctrl->AppendColumn("Search ID", wxLIST_FORMAT_LEFT, 70);
    available_jobs_ctrl->AppendColumn("Search Name", wxLIST_FORMAT_LEFT, 180);
    available_jobs_ctrl->AppendColumn("Status", wxLIST_FORMAT_LEFT, 90);
    available_jobs_ctrl->AppendColumn("Progress", wxLIST_FORMAT_LEFT, 70);
    available_jobs_ctrl->AppendColumn("CLI Args", wxLIST_FORMAT_LEFT, 120);

    // Create CLI args section with Update Selected button
    wxPanel*    cli_args_panel = new wxPanel(this, wxID_ANY);
    wxBoxSizer* cli_args_sizer = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* cli_args_label = new wxStaticText(cli_args_panel, wxID_ANY, "Custom CLI Arguments:");
    custom_cli_args_text = new wxTextCtrl(cli_args_panel, wxID_ANY, wxEmptyString,
                                          wxDefaultPosition, wxSize(400, -1));
    update_selected_button = new wxButton(cli_args_panel, wxID_ANY, "Update Selected");
    update_selected_button->Enable(false); // Disabled until pending item selected

    cli_args_sizer->Add(cli_args_label, 0, wxALIGN_CENTER_VERTICAL | wxALL, 5);
    cli_args_sizer->Add(custom_cli_args_text, 1, wxEXPAND | wxALL, 5);
    cli_args_sizer->Add(update_selected_button, 0, wxALL, 5);
    cli_args_panel->SetSizer(cli_args_sizer);

    // Create bottom controls panel with left and right button groups
    wxPanel*    bottom_controls = new wxPanel(this, wxID_ANY);
    wxBoxSizer* bottom_sizer    = new wxBoxSizer(wxHORIZONTAL);

    // Left side - stacked Add/Remove from Queue buttons
    wxBoxSizer* left_button_sizer = new wxBoxSizer(wxVERTICAL);
    add_to_queue_button      = new wxButton(bottom_controls, wxID_ANY, "Add to Execution Queue");
    remove_from_queue_button = new wxButton(bottom_controls, wxID_ANY, "Remove from Execution Queue");
    left_button_sizer->Add(add_to_queue_button, 0, wxEXPAND | wxBOTTOM, 5);
    left_button_sizer->Add(remove_from_queue_button, 0, wxEXPAND);

    // Right side - stacked Remove Selected/Clear All buttons
    wxBoxSizer* right_button_sizer = new wxBoxSizer(wxVERTICAL);
    remove_selected_button = new wxButton(bottom_controls, wxID_ANY, "Remove Selected");
    clear_queue_button     = new wxButton(bottom_controls, wxID_ANY, "Clear All");
    right_button_sizer->Add(remove_selected_button, 0, wxEXPAND | wxBOTTOM, 5);
    right_button_sizer->Add(clear_queue_button, 0, wxEXPAND);

    bottom_sizer->Add(left_button_sizer, 0, wxALL, 5);
    bottom_sizer->AddStretchSpacer();
    bottom_sizer->Add(right_button_sizer, 0, wxALL, 5);
    bottom_controls->SetSizer(bottom_sizer);


    // Add all sections to main sizer with adjusted proportions for better visibility
    main_sizer->Add(execution_queue_label, 0, wxEXPAND | wxALL, 5);
    main_sizer->Add(execution_queue_ctrl, 2, wxEXPAND | wxALL, 5); // Slightly larger proportion
    main_sizer->Add(controls_panel, 0, wxEXPAND | wxALL, 5);    // Combined Run Queue and Panel display
    main_sizer->Add(available_header_panel, 0, wxEXPAND | wxALL, 5);
    main_sizer->Add(available_jobs_ctrl, 1, wxEXPAND | wxALL, 5); // Smaller but still expandable
    main_sizer->Add(cli_args_panel, 0, wxEXPAND | wxALL, 5);  // Custom CLI args field with Update button
    main_sizer->Add(bottom_controls, 0, wxEXPAND | wxALL, 5);

    SetSizer(main_sizer);

    // Force layout update - important for dialog visibility
    Layout( );
    wxPrintf("TemplateMatchQueueManager: Layout() called\n");

    // Connect events
    update_selected_button->Bind(wxEVT_BUTTON, &TemplateMatchQueueManager::OnUpdateSelectedClick, this);
    run_selected_button->Bind(wxEVT_BUTTON, &TemplateMatchQueueManager::OnRunSelectedClick, this);
    add_to_queue_button->Bind(wxEVT_BUTTON, &TemplateMatchQueueManager::OnAddToQueueClick, this);
    remove_from_queue_button->Bind(wxEVT_BUTTON, &TemplateMatchQueueManager::OnRemoveFromQueueClick, this);
    remove_selected_button->Bind(wxEVT_BUTTON, &TemplateMatchQueueManager::OnRemoveSelectedClick, this);
    clear_queue_button->Bind(wxEVT_BUTTON, &TemplateMatchQueueManager::OnClearQueueClick, this);
    hide_completed_checkbox->Bind(wxEVT_CHECKBOX, &TemplateMatchQueueManager::OnHideCompletedToggle, this);
    panel_display_toggle->Bind(wxEVT_TOGGLEBUTTON, &TemplateMatchQueueManager::OnPanelDisplayToggle, this);

    // Bind selection events for available searches table
    available_jobs_ctrl->Bind(wxEVT_LIST_ITEM_SELECTED, &TemplateMatchQueueManager::OnAvailableJobsSelectionChanged, this);
    available_jobs_ctrl->Bind(wxEVT_LIST_ITEM_DESELECTED, &TemplateMatchQueueManager::OnAvailableJobsSelectionChanged, this);

    // Bind mouse events for manual drag and drop implementation
    execution_queue_ctrl->Bind(wxEVT_LEFT_DOWN, &TemplateMatchQueueManager::OnMouseLeftDown, this);
    execution_queue_ctrl->Bind(wxEVT_MOTION, &TemplateMatchQueueManager::OnMouseMotion, this);
    execution_queue_ctrl->Bind(wxEVT_LEFT_UP, &TemplateMatchQueueManager::OnMouseLeftUp, this);

    // Don't load from database in constructor - it may be called during workflow switch
    // when main_frame is in an inconsistent state

    // No longer need active instance tracking with unified architecture

    // Display any existing queue items
    UpdateQueueDisplay( );
}

TemplateMatchQueueManager::~TemplateMatchQueueManager( ) {
    // Manual drag and drop doesn't need cleanup

    // Clear completion callback if we set one
    if ( match_template_panel_ptr ) {
        match_template_panel_ptr->ClearQueueCompletionCallback( );
    }
}

bool TemplateMatchQueueManager::ValidateQueueItem(const TemplateMatchQueueItem& item, wxString& error_message) {
    // Validate basic parameters
    if (item.image_group_id < 0) {
        error_message = wxString::Format("Invalid image group ID: %d", item.image_group_id);
        return false;
    }
    if (item.reference_volume_asset_id < 0) {
        error_message = wxString::Format("Invalid reference volume ID: %d", item.reference_volume_asset_id);
        return false;
    }
    if (item.search_name.IsEmpty()) {
        error_message = "Search name cannot be empty";
        return false;
    }
    if (item.queue_status != "pending" && item.queue_status != "running" &&
        item.queue_status != "complete" && item.queue_status != "failed" &&
        item.queue_status != "partial") {
        error_message = wxString::Format("Invalid queue status: %s", item.queue_status);
        return false;
    }

    error_message = "";
    return true;
}

void TemplateMatchQueueManager::AddToExecutionQueue(const TemplateMatchQueueItem& item) {
    // Validate the queue item before adding
    wxString error_message;
    if (!ValidateQueueItem(item, error_message)) {
        wxMessageBox(error_message, "Invalid Queue Item", wxOK | wxICON_ERROR);
        return;
    }

    MyDebugAssertTrue(match_template_panel_ptr != nullptr, "AddToExecutionQueue called with null match_template_panel_ptr");
    MyDebugAssertTrue(main_frame != nullptr, "AddToExecutionQueue called with null main_frame");
    MyDebugAssertTrue(main_frame->current_project.is_open, "AddToExecutionQueue called with no project open");

    // Add item to database and get the new queue ID
    long new_database_queue_id = main_frame->current_project.database.AddToTemplateMatchQueue(
            item.search_name, item.image_group_id, item.reference_volume_asset_id, item.run_profile_id,
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
    new_item.database_queue_id      = new_database_queue_id;
    new_item.search_id               = -1;  // Search ID is only assigned when first result is written

    // Assign next available queue order (find highest current order + 1)
    int max_order = -1; // Start at -1 so first item gets queue_order=0
    for ( const auto& existing_item : execution_queue ) {
        if ( existing_item.queue_order > max_order ) {
            max_order = existing_item.queue_order;
        }
    }
    new_item.queue_order = max_order + 1;

    execution_queue.push_back(new_item);

    UpdateQueueDisplay( );

    // Highlight the newly added item (last item in queue)
    if ( queue_list_ctrl && execution_queue.size( ) > 0 ) {
        int new_row = execution_queue.size( ) - 1;
        queue_list_ctrl->SetItemState(new_row, wxLIST_STATE_SELECTED, wxLIST_STATE_SELECTED);
    }
}

// revert - DEBUG helper function to print queue state
void TemplateMatchQueueManager::PrintQueueState( ) {
    wxPrintf("  Execution Queue (%zu items):\n", execution_queue.size( ));
    for ( const auto& item : execution_queue ) {
        if ( item.queue_order >= 0 ) {
            wxPrintf("    [%d] ID=%ld, Status=%s, Name=%s\n",
                     item.queue_order, item.database_queue_id,
                     item.queue_status.mb_str( ).data( ), item.search_name.mb_str( ).data( ));
        }
    }
    wxPrintf("  Available items in execution_queue (queue_order < 0): %d\n",
             int(std::count_if(execution_queue.begin( ), execution_queue.end( ),
                               [](const auto& item) { return item.queue_order < 0; })));
    wxPrintf("  Available Queue (%zu items)\n", available_queue.size( ));
    wxPrintf("  Currently running job ID: %ld\n\n", currently_running_id);
}

bool TemplateMatchQueueManager::IsDatabaseAvailable(MyMainFrame* frame_ptr) const {
    return (frame_ptr != nullptr && frame_ptr->current_project.is_open);
}

void TemplateMatchQueueManager::RemoveFromExecutionQueue(int index) {
    MyDebugAssertTrue(index >= 0, "RemoveFromQueue called with negative index: %d", index);
    MyDebugAssertTrue(index < execution_queue.size( ), "RemoveFromQueue called with index %d >= queue size %zu", index, execution_queue.size( ));
    MyDebugAssertFalse(execution_queue.empty( ), "Cannot remove from empty queue");

    if ( index >= 0 && index < execution_queue.size( ) ) {
        // Don't allow removal of currently running searches
        MyDebugAssertTrue(execution_queue[index].queue_status != "running",
                          "Cannot remove currently running job (ID: %ld)", execution_queue[index].database_queue_id);

        // Remove from database first
        MyDebugAssertTrue(match_template_panel_ptr != nullptr, "RemoveFromExecutionQueue: match_template_panel_ptr is null");
        MyDebugAssertTrue(main_frame != nullptr, "RemoveFromExecutionQueue: main_frame is null");
        MyDebugAssertTrue(main_frame->current_project.is_open, "RemoveFromExecutionQueue: no project open");
        main_frame->current_project.database.RemoveFromQueue(execution_queue[index].database_queue_id);

        execution_queue.erase(execution_queue.begin( ) + index);
        UpdateQueueDisplay( );
    }
}

void TemplateMatchQueueManager::ClearExecutionQueue( ) {
    // Clear all items and remove from database
    MyDebugAssertTrue(match_template_panel_ptr != nullptr, "ClearExecutionQueue: match_template_panel_ptr is null");
    MyDebugAssertTrue(main_frame != nullptr, "ClearExecutionQueue: main_frame is null");
    MyDebugAssertTrue(main_frame->current_project.is_open, "ClearExecutionQueue: no project open");

    // Remove all items from database
    for ( const auto& item : execution_queue ) {
        main_frame->current_project.database.RemoveFromQueue(item.database_queue_id);
    }

    execution_queue.clear( );
    UpdateQueueDisplay( );
}

void TemplateMatchQueueManager::SetExecutionQueuePosition(int job_index, int new_position) {
    // Validate inputs
    MyDebugAssertTrue(job_index >= 0 && job_index < execution_queue.size( ),
                      "Invalid job_index: %d (queue size: %zu)", job_index, execution_queue.size( ));
    MyDebugAssertTrue(new_position >= 1 && new_position <= execution_queue.size( ),
                      "Invalid new_position: %d (valid range: 1-%zu)", new_position, execution_queue.size( ));

    int current_position = execution_queue[job_index].queue_order;
    if ( current_position == new_position ) {
        return; // No change needed
    }

    // Update queue orders for all affected jobs
    if ( current_position < new_position ) {
        // Moving job to a later position - decrement jobs between current and new position
        for ( auto& item : execution_queue ) {
            if ( item.queue_order > current_position && item.queue_order <= new_position ) {
                item.queue_order--;
            }
        }
    }
    else {
        // Moving job to an earlier position - increment jobs between new and current position
        for ( auto& item : execution_queue ) {
            if ( item.queue_order >= new_position && item.queue_order < current_position ) {
                item.queue_order++;
            }
        }
    }

    // Set the moved job to its new position
    execution_queue[job_index].queue_order = new_position;

    wxPrintf("SetJobPosition: Moved job %ld from position %d to %d\n",
             execution_queue[job_index].database_queue_id, current_position, new_position);
}

void TemplateMatchQueueManager::ProgressExecutionQueue( ) {
    if constexpr ( skip_search_execution_for_queue_debugging ) {
        wxPrintf("\n=== PROGRESS EXECUTION QUEUE CALLED ===\n");
        wxPrintf("Queue state at start of ProgressExecutionQueue:\n");
        PrintQueueState( );
    }

    // New logic: find the job at priority 0 and run it - allow any non-complete jobs
    TemplateMatchQueueItem* next_job = nullptr;
    for ( auto& item : execution_queue ) {
        if ( item.queue_order == 0 && item.queue_status != "complete" ) {
            next_job = &item;
            if constexpr ( skip_search_execution_for_queue_debugging ) {
                wxPrintf("Found next job to run: ID=%ld at queue_order=0\n", item.database_queue_id);
            }
            break;
        }
    }

    if ( next_job ) {
        // Don't change status yet - let MatchTemplatePanel validate first
        // Move to priority -1 (out of execution queue) but keep original status
        int original_queue_order = next_job->queue_order;
        next_job->queue_order    = -1;

        // Shift all other execution queue jobs up by one priority (1->0, 2->1, 3->2, etc.)
        for ( auto& item : execution_queue ) {
            if ( item.queue_order > 0 ) {
                item.queue_order--;
            }
        }

        wxPrintf("ProgressQueue: Starting job %ld (moved to priority -1, status: %s)\n",
                 next_job->database_queue_id, next_job->queue_status);

        // Execute the job - status will be changed to "running" by MatchTemplatePanel after validation
        if ( ExecuteJob(*next_job) ) {
            wxPrintf("ProgressQueue: Job %ld execution started successfully\n", next_job->database_queue_id);
        }
        else {
            // If execution failed, mark as failed and ensure it stays in available queue
            next_job->queue_status = "failed";
            // next_job->queue_order is already -1

            // Critical: Ensure currently_running_id is cleared when job fails to start
            // ExecuteJob should not have set it if it returned false, but clear it to be safe
            currently_running_id = -1;

            wxPrintf("ProgressQueue: Job %ld execution failed, trying next job\n", next_job->database_queue_id);

            // Update display and save failed job status
            SaveQueueToDatabase( );
            UpdateQueueDisplay( );

            // Recursively try the next job in the queue
            if ( auto_progress_queue ) {
                wxPrintf("ProgressQueue: Attempting next job after failure\n");
                ProgressExecutionQueue( );
            }
            return; // Early return to avoid duplicate save/update below
        }

        // Update display and save to database
        SaveQueueToDatabase( );
        UpdateQueueDisplay( );
    }
    else {
        wxPrintf("ProgressQueue: No pending jobs found at priority 0\n");
    }
}

// Old MoveItem methods removed - replaced with queue order system

wxColour TemplateMatchQueueManager::GetStatusColor(const wxString& status) {
    if ( status == "running" ) {
        return wxColour(0, 200, 0); // Bright green (lighter than complete)
    }
    else if ( status == "complete" ) {
        return wxColour(0, 100, 0); // Dark green
    }
    else if ( status == "failed" ) {
        return wxColour(139, 69, 19); // Dark brick red
    }
    else { // pending
        return wxColour(128, 128, 128); // Gray for pending
    }
}

void TemplateMatchQueueManager::SetStatusDisplay(wxListCtrl* list_ctrl, long item_index, const wxString& status) {
    // Generate status display text with appropriate prefix
    wxString status_display;
    if ( status == "running" ) {
        status_display = "● " + status; // Filled circle for running
    }
    else if ( status == "complete" ) {
        status_display = "✓ " + status; // Check mark for complete
    }
    else if ( status == "failed" ) {
        status_display = "✗ " + status; // X mark for failed
    }
    else { // pending
        status_display = "○ " + status; // Empty circle for pending
    }

    // Set the status text
    int status_column = (list_ctrl == queue_list_ctrl) ? 3 : 2; // execution queue has queue order column
    list_ctrl->SetItem(item_index, status_column, status_display);

    // Get color and font styling
    wxColour text_color = GetStatusColor(status);
    wxFont   item_font  = list_ctrl->GetFont( ); // Get default font

    // Apply font styling based on status
    if ( status == "running" ) {
        item_font.MakeBold( );
        item_font.MakeItalic( );
    }
    else if ( status == "complete" ) {
        item_font.MakeBold( );
    }
    // failed and pending use default font

    // Apply color and font to the entire row
    list_ctrl->SetItemTextColour(item_index, text_color);
    list_ctrl->SetItemFont(item_index, item_font);
}

void TemplateMatchQueueManager::PopulateListControl(wxListCtrl*                                 ctrl,
                                                    const std::vector<TemplateMatchQueueItem*>& items,
                                                    bool                                        is_execution_queue) {
    MyDebugAssertTrue(ctrl != nullptr, "PopulateListControl: ctrl is null");

    ctrl->DeleteAllItems( );

    for ( size_t idx = 0; idx < items.size( ); ++idx ) {
        const auto* item = items[idx];
        long        row;

        // Both execution and available queues now have the same column structure
        // Queue ID is first column - always show it
        row = ctrl->InsertItem(idx, wxString::Format("%ld", item->database_queue_id));

        // Search ID is second column - show if valid (> 0), otherwise show empty string
        if (item->search_id > 0) {
            ctrl->SetItem(row, 1, wxString::Format("%ld", item->search_id));
        } else {
            // No valid search ID yet (job hasn't started or search_id is -1 or 0) - show empty string
            ctrl->SetItem(row, 1, "");
        }

        // CRITICAL: Store the database_queue_id as item data so we can retrieve it later
        // This allows us to identify which queue item a row represents regardless of sorting/filtering
        ctrl->SetItemData(row, item->database_queue_id);

        ctrl->SetItem(row, 2, item->search_name);
        // Status is column 3
        SetStatusDisplay(ctrl, row, item->queue_status);
        // Progress is column 4
        SearchCompletionInfo completion = GetSearchCompletionInfo(item->database_queue_id);
        ctrl->SetItem(row, 4, completion.GetCompletionString( ));
        // Custom args is column 5
        ctrl->SetItem(row, 5, item->custom_cli_args);
    }
}

void TemplateMatchQueueManager::UpdateQueueDisplay( ) {
    MyDebugAssertTrue(queue_list_ctrl != nullptr, "queue_list_ctrl is null in UpdateQueueDisplay");

    // Prevent drag operations during display update to avoid GTK crashes
    updating_display = true;

    // Preserve current selections before rebuilding
    std::vector<long> selected_template_ids;
    long              item = queue_list_ctrl->GetNextItem(-1, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
    while ( item != -1 ) {
        if ( item < int(execution_queue.size( )) ) {
            for ( const auto& job : execution_queue ) {
                if ( job.queue_order == item ) {
                    selected_template_ids.push_back(job.database_queue_id);
                    break;
                }
            }
        }
        item = queue_list_ctrl->GetNextItem(item, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
    }

    // Build sorted list of execution queue items
    std::vector<TemplateMatchQueueItem*> sorted_items;
    for ( auto& job : execution_queue ) {
        if ( job.queue_order >= 0 ) {
            sorted_items.push_back(&job);
        }
    }

    // Sort by queue_order
    std::sort(sorted_items.begin( ), sorted_items.end( ),
              [](const TemplateMatchQueueItem* a, const TemplateMatchQueueItem* b) {
                  return a->queue_order < b->queue_order;
              });

    // Use consolidated method to populate the list
    PopulateListControl(queue_list_ctrl, sorted_items, true);

    // Restore selections after rebuilding
    for ( long template_id : selected_template_ids ) {
        for ( const auto& job : execution_queue ) {
            if ( job.database_queue_id == template_id && job.queue_order >= 0 ) {
                int row = job.queue_order;
                if ( row >= 0 && row < queue_list_ctrl->GetItemCount( ) ) {
                    queue_list_ctrl->SetItemState(row, wxLIST_STATE_SELECTED, wxLIST_STATE_SELECTED);
                }
                break;
            }
        }
    }

    // Update the available jobs table with items not in execution queue
    UpdateAvailableJobsDisplay( );

    // Re-enable drag operations after display update is complete
    updating_display = false;
}

void TemplateMatchQueueManager::UpdateAvailableJobsDisplay( ) {
    wxPrintf("UpdateAvailableJobsDisplay called\n");
    MyDebugAssertTrue(available_jobs_ctrl != nullptr, "available_jobs_ctrl is null in UpdateAvailableJobsDisplay");

    // Build list of available items
    std::vector<TemplateMatchQueueItem*> available_items;

    // Add jobs from execution_queue with queue_order < 0
    for ( auto& job : execution_queue ) {
        if ( job.queue_order < 0 ) {
            // Skip completed jobs if hide_completed_jobs is enabled
            if ( ! hide_completed_jobs || job.queue_status != "complete" ) {
                available_items.push_back(&job);
            }
        }
    }

    // Add jobs from available_queue that aren't in execution_queue
    for ( auto& job : available_queue ) {
        // Skip completed jobs if hide_completed_jobs is enabled
        if ( hide_completed_jobs && job.queue_status == "complete" ) {
            continue;
        }

        // Check if this job is already in execution_queue to avoid duplicates
        bool found_in_execution = false;
        for ( const auto& exec_job : execution_queue ) {
            if ( exec_job.database_queue_id == job.database_queue_id ) {
                found_in_execution = true;
                break;
            }
        }

        if ( ! found_in_execution ) {
            available_items.push_back(&job);
        }
    }

    // Use consolidated method to populate the list
    PopulateListControl(available_jobs_ctrl, available_items, false);

    wxPrintf("UpdateAvailableJobsDisplay: Added %zu available jobs\n", available_items.size( ));
}

int TemplateMatchQueueManager::GetSelectedRow( ) {
    long selected_item = queue_list_ctrl->GetNextItem(-1, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
    if ( selected_item != -1 ) {
        return selected_item;
    }
    return -1;
}

// RunAllJobs method removed - use Run Selected with multi-selection instead
// RunSelectedJob method removed - redundant with RunNextJob

void TemplateMatchQueueManager::RunNextJob( ) {
    MyPrintWithDetails("=== RUN NEXT JOB CALLED ===");

    // Check if a job is already running or finalizing
    if ( IsJobRunning( ) ) {
        if ( job_is_finalizing ) {
            wxPrintf("Job is still finalizing - deferring RunNextJob\n");
            // Schedule another attempt after finalization completes
            wxTimer* retry_timer = new wxTimer();
            retry_timer->Bind(wxEVT_TIMER, [this, retry_timer](wxTimerEvent&) {
                wxPrintf("Retrying RunNextJob after finalization delay\n");
                if ( ! IsJobRunning( ) ) {
                    RunNextJob();
                }
                delete retry_timer;
            });
            retry_timer->StartOnce(1000);  // Retry in 1 second
        } else {
            MyDebugPrint("Job %ld is already running - skipping RunNextJob", currently_running_id);
        }
        return;
    }

    MyDebugPrint("No job currently running - searching for next pending job");

    // Find the job at priority 0 (top of execution queue) - allow any non-complete jobs
    TemplateMatchQueueItem* next_job = nullptr;
    for ( auto& item : execution_queue ) {
        if ( item.queue_order == 0 && item.queue_status != "complete" ) {
            next_job = &item;
            break;
        }
    }

    if ( ! next_job ) {
        wxMessageBox("No pending jobs in execution queue at priority 0.", "No Jobs to Run", wxOK | wxICON_WARNING);
        return;
    }

    MyDebugPrint("Found next pending job:");
    MyDebugPrint("  ID: %ld", next_job->database_queue_id);
    MyDebugPrint("  Name: '%s'", next_job->search_name.mb_str( ).data( ));
    MyDebugPrint("  Status: '%s'", next_job->queue_status.mb_str( ).data( ));
    MyDebugPrint("  Image Group: %d", next_job->image_group_id);
    MyDebugPrint("  Reference Volume: %d", next_job->reference_volume_asset_id);

    // Enable auto-progression for queue mode - each job completion will trigger the next
    SetAutoProgressQueue(true);

    // Move to priority -1 (out of execution queue) but keep original status
    next_job->queue_order = -1; // Move running job out of execution queue for crash safety

    // Shift all other execution queue jobs up by one priority (1->0, 2->1, 3->2, etc.)
    for ( auto& item : execution_queue ) {
        if ( item.queue_order > 0 ) {
            item.queue_order--;
        }
    }

    MyDebugPrint("=== EXECUTING JOB %ld ===", next_job->database_queue_id);

    // Execute the job - status will be changed to "running" by MatchTemplatePanel after validation
    if ( ExecuteJob(*next_job) ) {
        wxPrintf("RunNextJob: Started job %ld (moved to priority -1, status: %s)\n",
                 next_job->database_queue_id, next_job->queue_status);
    }
    else {
        // If execution failed, mark as failed and restore to available queue
        next_job->queue_status = "failed";
        // queue_order already set to -1, which is correct for available queue
        wxPrintf("RunNextJob: Job %ld execution failed\n", next_job->database_queue_id);
    }

    // Save queue state and update display
    SaveQueueToDatabase( );
    UpdateQueueDisplay( );
}

bool TemplateMatchQueueManager::ExecuteJob(TemplateMatchQueueItem& job_to_run) {
    // Basic validation moved to TMP's ExecuteJob method

    // Check if another job is already running
    MyDebugAssertFalse(IsJobRunning( ), "Attempted to execute job %ld while job %ld is already running", job_to_run.database_queue_id, currently_running_id);
    if ( IsJobRunning( ) ) {
        wxMessageBox("A job is already running. Please wait for it to complete.",
                     "Job Running", wxOK | wxICON_WARNING);
        return false;
    }

    // Use the stored MatchTemplatePanel pointer to execute the job
    MyDebugAssertTrue(match_template_panel_ptr != nullptr, "match_template_panel_ptr is null - cannot execute jobs");

    if ( match_template_panel_ptr ) {
        // Register this queue manager for completion callbacks
        match_template_panel_ptr->SetQueueCompletionCallback(this);

        if constexpr ( skip_search_execution_for_queue_debugging ) {
            // Debug mode: Mark job as running immediately, then simulate execution time
            wxPrintf("\n=== DEBUG MODE: Starting job in debug mode ===\n");
            wxPrintf("Job %ld will be marked as running\n", job_to_run.database_queue_id);
            wxPrintf("Current queue state BEFORE marking as running:\n");
            PrintQueueState( );

            // Mark job as running (same as production code path)
            UpdateJobStatus(job_to_run.database_queue_id, "running");
            currently_running_id = job_to_run.database_queue_id;
            wxPrintf("Job %ld is now marked as RUNNING (currently_running_id set)\n", job_to_run.database_queue_id);

            // Update button state when job starts
            UpdateButtonState();

            // Update displays to show running status
            UpdateQueueDisplay( );
            UpdateAvailableJobsDisplay( );

            wxPrintf("Queue state AFTER marking as running:\n");
            PrintQueueState( );

            // Non-blocking wait for 5 seconds to simulate job execution
            // Process events during this time so GUI remains responsive
            wxPrintf("\n>>> Starting 5 second wait to simulate job execution...\n");
            wxPrintf(">>> You can now interact with the queue (remove jobs, etc.)\n");

            wxStopWatch timer;
            timer.Start( );
            while ( timer.Time( ) < 5000 ) {
                // Process pending events to keep GUI responsive
                wxYieldIfNeeded( );
                wxMilliSleep(50); // Small sleep to avoid consuming 100% CPU
            }

            wxPrintf(">>> Execution simulation complete for job %ld\n\n", job_to_run.database_queue_id);

            // Now simulate job completion
            wxPrintf("=== DEBUG: Simulating job completion ===\n");
            OnJobCompleted(job_to_run.database_queue_id, true); // Simulate successful completion
            wxPrintf("=== Job completion simulation done ===\n\n");

            return true;
        }
        else {
            // Normal execution path
            bool execution_success = match_template_panel_ptr->ExecuteJob(&job_to_run);

            if ( execution_success ) {
                // Job executed successfully - now mark it as running and track it
                UpdateJobStatus(job_to_run.database_queue_id, "running");
                currently_running_id = job_to_run.database_queue_id;
                wxPrintf("Job %ld started successfully and marked as running\n", job_to_run.database_queue_id);

                // Update button state when job starts
                UpdateButtonState();

                // Job status will be updated to "complete" when the job finishes via ProcessAllJobsFinished
                return true;
            }
            else {
                wxPrintf("Failed to start job %ld\n", job_to_run.database_queue_id);
                UpdateJobStatus(job_to_run.database_queue_id, "failed");
                return false;
            }
        }
    }
    else {
        // Critical failure - template panel not available
        MyAssertTrue(false, "Critical error: match_template_panel not available for job execution");
        wxPrintf("Error: match_template_panel not available\n");
        UpdateJobStatus(job_to_run.database_queue_id, "failed");
        currently_running_id = -1;
    }

    return false;
}

bool TemplateMatchQueueManager::IsJobRunning( ) const {
    // A job is considered "running" if it's actively processing OR finalizing
    bool is_running = currently_running_id != -1 || job_is_finalizing;
    if ( is_running ) {
        if ( job_is_finalizing ) {
            wxPrintf("IsJobRunning: Job is in finalization phase\n");
        } else {
            wxPrintf("IsJobRunning: Job %ld is currently running\n", currently_running_id);
        }
    }
    return is_running;
}

// IsJobRunningStatic removed - no longer needed with unified architecture

void TemplateMatchQueueManager::UpdateJobStatus(long database_queue_id, const wxString& new_status) {
    MyDebugAssertTrue(database_queue_id >= 0, "Invalid database_queue_id in UpdateJobStatus: %ld", database_queue_id);
    MyDebugAssertTrue(new_status == "pending" || new_status == "running" || new_status == "complete" || new_status == "failed",
                      "Invalid new_status in UpdateJobStatus: %s", new_status.mb_str( ).data( ));

    bool found_job = false;
    for ( auto& item : execution_queue ) {
        if ( item.database_queue_id == database_queue_id ) {
            // Validate status transitions
            MyDebugAssertTrue(item.queue_status != new_status, "Attempted to set status to same value: %s", new_status.mb_str( ).data( ));

            // Validate allowed transitions
            if ( item.queue_status == "running" && (new_status == "complete" || new_status == "failed") ) {
                // Valid: running -> complete/failed
                // Allow if currently_running_id matches OR if job is finalizing (currently_running_id may be -1)
                MyDebugAssertTrue(currently_running_id == database_queue_id || job_is_finalizing,
                                  "Status change from running but database_queue_id %ld != currently_running_id %ld and not finalizing",
                                  database_queue_id, currently_running_id);
            }
            else if ( item.queue_status == "partial" && (new_status == "complete" || new_status == "failed") ) {
                // Valid: partial -> complete/failed (when partial job finishes or fails)
                wxPrintf("Partial job %ld transitioning to %s\n", database_queue_id, new_status);
            }
            else if ( item.queue_status == "pending" && new_status == "running" ) {
                // Valid: pending -> running
                MyDebugAssertFalse(IsJobRunning( ), "Cannot start job %ld when job %ld is already running", database_queue_id, currently_running_id);
            }
            else if ( item.queue_status == "pending" && new_status == "failed" ) {
                // Valid: pending -> failed (when job fails to start)
                wxPrintf("Job %ld failed to start, marking as failed\n", database_queue_id);
            }
            else if ( item.queue_status == "failed" && new_status == "running" ) {
                // Valid: failed -> running (when resuming a failed job)
                MyDebugAssertFalse(IsJobRunning( ), "Cannot resume failed job %ld when job %ld is already running", database_queue_id, currently_running_id);
                wxPrintf("Resuming failed job %ld, marking as running\n", database_queue_id);
            }
            else if ( item.queue_status == "partial" && new_status == "running" ) {
                // Valid: partial -> running (when resuming a partially complete job)
                MyDebugAssertFalse(IsJobRunning( ), "Cannot resume partial job %ld when job %ld is already running", database_queue_id, currently_running_id);
                wxPrintf("Resuming partial job %ld, marking as running\n", database_queue_id);
            }
            else {
                MyDebugAssertTrue(false, "Invalid status transition: %s -> %s for job %ld",
                                  item.queue_status.mb_str( ).data( ), new_status.mb_str( ).data( ), database_queue_id);
            }

            item.queue_status = new_status;
            found_job         = true;
            break;
        }
    }

    MyDebugAssertTrue(found_job, "Job with database_queue_id %ld not found in queue for status update", database_queue_id);

    UpdateQueueDisplay( );
    SaveQueueToDatabase( );
}

// Static methods removed - no longer needed with unified architecture

// GetNextPendingJob method removed - functionality moved into RunNextJob

bool TemplateMatchQueueManager::HasPendingJobs( ) {
    for ( const auto& item : execution_queue ) {
        if ( item.queue_status == "pending" ) {
            return true;
        }
    }
    return false;
}

void TemplateMatchQueueManager::OnRunSelectedClick(wxCommandEvent& event) {
    RunNextJob( );
}

// OnRunAllClick removed - use Run Selected with multi-selection instead

void TemplateMatchQueueManager::OnClearQueueClick(wxCommandEvent& event) {
    wxMessageDialog dialog(this, "Clear all pending jobs from the queue?",
                           "Confirm Clear", wxYES_NO | wxICON_QUESTION);
    if ( dialog.ShowModal( ) == wxID_YES ) {
        ClearExecutionQueue( );
    }
}

void TemplateMatchQueueManager::OnHideCompletedToggle(wxCommandEvent& event) {
    hide_completed_jobs = event.IsChecked( );
    wxPrintf("Hide completed jobs toggled to: %s\n", hide_completed_jobs ? "true" : "false");

    // Refresh the available jobs display with the new filter
    UpdateAvailableJobsDisplay( );
}

void TemplateMatchQueueManager::OnAvailableJobsSelectionChanged(wxListEvent& event) {
    wxPrintf("OnAvailableJobsSelectionChanged called\n");

    // Check if any items are selected in available jobs table
    bool                    has_available_selection = false;
    long                    first_selected_row      = available_jobs_ctrl->GetNextItem(-1, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
    TemplateMatchQueueItem* first_selected_item     = nullptr;

    if ( first_selected_row != -1 ) {
        has_available_selection = true;

        // Get the database_queue_id from the item data
        long database_queue_id = available_jobs_ctrl->GetItemData(first_selected_row);

        // Find the queue item with this database_queue_id
        // First check execution_queue
        for ( auto& item : execution_queue ) {
            if ( item.database_queue_id == database_queue_id ) {
                first_selected_item = &item;
                break;
            }
        }

        // If not found in execution_queue, check available_queue
        if ( ! first_selected_item ) {
            for ( auto& item : available_queue ) {
                if ( item.database_queue_id == database_queue_id ) {
                    first_selected_item = &item;
                    break;
                }
            }
        }
    }

    // Enable Add to Queue button based on available jobs selection
    add_to_queue_button->Enable(has_available_selection);

    // Populate the GUI with the first selected item's parameters
    // Note: We don't check gui_update_frozen here because we want to allow editing of pending items
    // The freeze is only meant to prevent interference during job execution
    if ( first_selected_item && match_template_panel_ptr ) {
        // Only populate GUI if the item is pending (editable) or we're not currently running a job
        bool is_editable = (first_selected_item->queue_status == "pending" ||
                           first_selected_item->queue_status == "failed" ||
                           first_selected_item->queue_status == "partial");

        if (is_editable || !gui_update_frozen) {
            wxPrintf("Populating GUI from available queue item: %s (status: %s)\n",
                    first_selected_item->search_name, first_selected_item->queue_status);
            match_template_panel_ptr->PopulateGuiFromQueueItem(*first_selected_item, true);

            // Also populate the custom CLI args field
            if (custom_cli_args_text) {
                custom_cli_args_text->SetValue(first_selected_item->custom_cli_args);
            }

            // Track which item was populated and enable/disable update button
            last_populated_queue_id = first_selected_item->database_queue_id;
            UpdateButtonState();
        }
    }

    wxPrintf("Available jobs selection changed - Add to Queue button %s\n",
             has_available_selection ? "enabled" : "disabled");
}

void TemplateMatchQueueManager::UpdateButtonState() {
    // Enable update button only if:
    // 1. We have a populated item ID
    // 2. The item exists and has status "pending"

    if (last_populated_queue_id <= 0) {
        update_selected_button->Enable(false);
        return;
    }

    // Find the item in either queue
    TemplateMatchQueueItem* item = nullptr;
    for (auto& queue_item : execution_queue) {
        if (queue_item.database_queue_id == last_populated_queue_id) {
            item = &queue_item;
            break;
        }
    }
    if (!item) {
        for (auto& queue_item : available_queue) {
            if (queue_item.database_queue_id == last_populated_queue_id) {
                item = &queue_item;
                break;
            }
        }
    }

    // Enable button only if item found and status is pending
    bool should_enable = (item != nullptr && item->queue_status == "pending");
    update_selected_button->Enable(should_enable);

    if (should_enable) {
        wxPrintf("Update button enabled for pending item %ld\n", last_populated_queue_id);
    } else if (item) {
        wxPrintf("Update button disabled - item %ld has status '%s'\n",
                 last_populated_queue_id, item->queue_status);
    } else {
        wxPrintf("Update button disabled - item %ld not found\n", last_populated_queue_id);
    }
}

void TemplateMatchQueueManager::OnUpdateSelectedClick(wxCommandEvent& event) {
    // Find the original queue item
    TemplateMatchQueueItem* original_item = nullptr;
    for (auto& queue_item : execution_queue) {
        if (queue_item.database_queue_id == last_populated_queue_id) {
            original_item = &queue_item;
            break;
        }
    }
    if (!original_item) {
        for (auto& queue_item : available_queue) {
            if (queue_item.database_queue_id == last_populated_queue_id) {
                original_item = &queue_item;
                break;
            }
        }
    }

    if (!original_item) {
        wxMessageBox("Could not find the original queue item", "Error", wxOK | wxICON_ERROR);
        return;
    }

    // Verify it's still pending
    if (original_item->queue_status != "pending") {
        wxMessageBox("Can only update pending items", "Error", wxOK | wxICON_ERROR);
        return;
    }

    // Collect current parameters from GUI
    TemplateMatchQueueItem updated_item = match_template_panel_ptr->CollectJobParametersFromGui();

    // Get custom CLI args from the text control
    if (custom_cli_args_text) {
        updated_item.custom_cli_args = custom_cli_args_text->GetValue();
    }

    // Preserve database ID and queue position
    updated_item.database_queue_id = original_item->database_queue_id;
    updated_item.queue_order = original_item->queue_order;
    updated_item.search_id = original_item->search_id;
    updated_item.queue_status = original_item->queue_status;

    // Validate the updated item
    wxString validation_error;
    if (!ValidateQueueItem(updated_item, validation_error)) {
        wxMessageBox(validation_error, "Invalid Parameters", wxOK | wxICON_ERROR);
        return;
    }

    // Build comparison dialog
    wxString changes;
    bool has_changes = false;

    // Compare key fields and list changes
    if (original_item->search_name != updated_item.search_name) {
        changes += wxString::Format("Search Name: %s -> %s\n",
                                    original_item->search_name, updated_item.search_name);
        has_changes = true;
    }
    if (original_item->image_group_id != updated_item.image_group_id) {
        changes += wxString::Format("Image Group ID: %d -> %d\n",
                                    original_item->image_group_id, updated_item.image_group_id);
        has_changes = true;
    }
    if (original_item->reference_volume_asset_id != updated_item.reference_volume_asset_id) {
        changes += wxString::Format("Reference Volume ID: %d -> %d\n",
                                    original_item->reference_volume_asset_id, updated_item.reference_volume_asset_id);
        has_changes = true;
    }
    if (original_item->run_profile_id != updated_item.run_profile_id) {
        changes += wxString::Format("Run Profile ID: %d -> %d\n",
                                    original_item->run_profile_id, updated_item.run_profile_id);
        has_changes = true;
    }
    if (original_item->high_resolution_limit != updated_item.high_resolution_limit) {
        changes += wxString::Format("High Resolution: %.2f -> %.2f\n",
                                    original_item->high_resolution_limit, updated_item.high_resolution_limit);
        has_changes = true;
    }
    if (original_item->out_of_plane_angular_step != updated_item.out_of_plane_angular_step) {
        changes += wxString::Format("Out-of-Plane Step: %.2f -> %.2f\n",
                                    original_item->out_of_plane_angular_step, updated_item.out_of_plane_angular_step);
        has_changes = true;
    }
    if (original_item->in_plane_angular_step != updated_item.in_plane_angular_step) {
        changes += wxString::Format("In-Plane Step: %.2f -> %.2f\n",
                                    original_item->in_plane_angular_step, updated_item.in_plane_angular_step);
        has_changes = true;
    }
    if (original_item->custom_cli_args != updated_item.custom_cli_args) {
        changes += wxString::Format("Custom CLI Args: '%s' -> '%s'\n",
                                    original_item->custom_cli_args, updated_item.custom_cli_args);
        has_changes = true;
    }
    // Add more comparisons as needed...

    if (!has_changes) {
        wxMessageBox("No changes detected", "Update Queue Item", wxOK | wxICON_INFORMATION);
        return;
    }

    // Show confirmation dialog
    wxString message = wxString::Format("Confirm the following changes to queue item %ld:\n\n%s",
                                        original_item->database_queue_id, changes);
    wxMessageDialog dialog(this, message, "Confirm Update", wxYES_NO | wxICON_QUESTION);

    if (dialog.ShowModal() != wxID_YES) {
        return;
    }

    // Update the item in memory
    *original_item = updated_item;

    // Update in database
    if (!UpdateQueueItemInDatabase(updated_item)) {
        // Error message already shown in UpdateQueueItemInDatabase
        return;
    }

    // Refresh displays
    UpdateQueueDisplay();
    UpdateAvailableJobsDisplay();

    wxMessageBox("Queue item updated successfully", "Success", wxOK | wxICON_INFORMATION);
}


void TemplateMatchQueueManager::OnRemoveSelectedClick(wxCommandEvent& event) {
    int selected = GetSelectedRow( );
    if ( selected >= 0 ) {
        wxString        search_name = execution_queue[selected].search_name;
        wxMessageDialog dialog(this,
                               wxString::Format("Remove search '%s' from the queue?", search_name),
                               "Confirm Remove", wxYES_NO | wxICON_QUESTION);
        if ( dialog.ShowModal( ) == wxID_YES ) {
            RemoveFromExecutionQueue(selected);
        }
    }
}

void TemplateMatchQueueManager::OnSelectionChanged(wxListEvent& event) {
    // Clean up any stale drag state - selection changes often happen after cancelled drags
    CleanupDragState( );

    // Validate GUI components are available
    MyDebugAssertTrue(remove_selected_button != nullptr, "remove_selected_button is null in OnSelectionChanged");
    MyDebugAssertTrue(run_selected_button != nullptr, "run_selected_button is null in OnSelectionChanged");

    // Check if any items are selected in the execution queue
    bool has_selection = (queue_list_ctrl->GetSelectedItemCount( ) > 0);

    // Find first selected item for GUI population and check for running jobs
    int  first_selected_index = -1;
    bool any_running          = false;

    if ( has_selection ) {
        // Check all selected items for running jobs
        long selected_row = queue_list_ctrl->GetNextItem(-1, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
        while ( selected_row != -1 ) {
            // Get the database_queue_id from the item data
            long database_queue_id = queue_list_ctrl->GetItemData(selected_row);

            // Find the job with this database_queue_id
            for ( size_t i = 0; i < execution_queue.size( ); ++i ) {
                if ( execution_queue[i].database_queue_id == database_queue_id ) {
                    if ( first_selected_index == -1 ) {
                        first_selected_index = i; // Save first for GUI population
                    }
                    if ( execution_queue[i].queue_status == "running" ) {
                        any_running = true;
                    }
                    break;
                }
            }
            selected_row = queue_list_ctrl->GetNextItem(selected_row, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
        }
    }

    // Check if there's a job at priority 0 ready to run (not complete)
    bool has_job_at_priority_0 = false;
    for ( const auto& job : execution_queue ) {
        if ( job.queue_order == 0 && job.queue_status != "complete" ) {
            has_job_at_priority_0 = true;
            break;
        }
    }

    // Enable controls based on selection and job status
    remove_selected_button->Enable(has_selection && ! any_running);
    run_selected_button->Enable(has_job_at_priority_0 && ! IsJobRunning( ));
    remove_from_queue_button->Enable(has_selection && ! any_running);

    // Populate the GUI with the first selected item's parameters
    // Note: We don't check gui_update_frozen here for pending items because we want to allow editing
    // The freeze is only meant to prevent interference during job execution
    if ( has_selection && first_selected_index >= 0 && match_template_panel_ptr ) {
        const auto& selected_item = execution_queue[first_selected_index];

        // Only populate GUI if the item is pending (editable) or we're not currently running a job
        bool is_editable = (selected_item.queue_status == "pending" ||
                           selected_item.queue_status == "failed" ||
                           selected_item.queue_status == "partial");

        if (is_editable || !gui_update_frozen) {
            wxPrintf("Populating GUI from execution queue item %d (status: %s)\n",
                    first_selected_index, selected_item.queue_status.mb_str().data());
            match_template_panel_ptr->PopulateGuiFromQueueItem(selected_item, true);

            // Also populate the custom CLI args field
            if (custom_cli_args_text) {
                custom_cli_args_text->SetValue(selected_item.custom_cli_args);
            }

            // Track which item was populated and enable/disable update button
            last_populated_queue_id = selected_item.database_queue_id;
            UpdateButtonState();
        }
    }

    event.Skip( );
}

void TemplateMatchQueueManager::OnAddToQueueClick(wxCommandEvent& event) {
    // Get selected jobs from available jobs table
    std::vector<long> selected_database_ids;
    long              selected_row = available_jobs_ctrl->GetNextItem(-1, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
    while ( selected_row != -1 ) {
        // Get the database_queue_id from the item data
        long database_queue_id = available_jobs_ctrl->GetItemData(selected_row);
        selected_database_ids.push_back(database_queue_id);
        selected_row = available_jobs_ctrl->GetNextItem(selected_row, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
    }

    if ( selected_database_ids.empty( ) ) {
        wxMessageBox("Please select jobs from the Available Jobs table to add to the execution queue.",
                     "No Selection", wxOK | wxICON_WARNING);
        return;
    }

    // Count jobs in execution queue to determine next position
    int next_queue_order = 0;
    for ( const auto& job : execution_queue ) {
        if ( job.queue_order >= 0 ) { // Count jobs in execution queue
            next_queue_order++;
        }
    }

    // Move selected available jobs to execution queue
    std::vector<long>     selected_job_ids;
    std::vector<wxString> blocked_jobs; // Track jobs that can't be moved

    // Helper lambda to process a job for moving to execution queue
    auto process_job_for_queue = [&](TemplateMatchQueueItem* job, bool needs_move_from_available, size_t available_index) {
        // Check if this job is selected (by database_queue_id)
        for ( long selected_id : selected_database_ids ) {
            if ( job->database_queue_id == selected_id ) {
                wxPrintf("DEBUG: Processing job %ld with status '%s' for adding to queue\n",
                         job->database_queue_id, job->queue_status);
                // Allow any non-complete jobs to be moved to execution queue
                if ( job->queue_status != "complete" ) {
                    wxPrintf("DEBUG: Job %ld allowed - status '%s' is not complete\n",
                             job->database_queue_id, job->queue_status);
                    job->queue_order = next_queue_order++;
                    selected_job_ids.push_back(job->database_queue_id);

                    if ( needs_move_from_available ) {
                        // Move from available_queue to execution_queue
                        execution_queue.push_back(*job);
                        available_queue.erase(available_queue.begin( ) + available_index);
                        return true; // Indicate we removed an item
                    }
                }
                else {
                    wxPrintf("DEBUG: Job %ld BLOCKED - status '%s' equals complete\n",
                             job->database_queue_id, job->queue_status);
                    blocked_jobs.push_back(wxString::Format("Job %ld (%s)",
                                                            job->database_queue_id,
                                                            job->queue_status));
                }
                break;
            }
        }
        return false;
    };

    // First, check jobs from execution_queue with queue_order < 0
    for ( size_t i = 0; i < execution_queue.size( ); ++i ) {
        if ( execution_queue[i].queue_order < 0 ) { // This is an available job
            // Skip completed jobs if hide_completed_jobs is enabled (must match display logic)
            if ( hide_completed_jobs && execution_queue[i].queue_status == "complete" ) {
                continue;
            }
            process_job_for_queue(&execution_queue[i], false, 0);
        }
    }

    // Then, check jobs from available_queue (not in execution_queue)
    for ( size_t i = 0; i < available_queue.size( ); ++i ) {
        // Skip if already in execution_queue
        bool found_in_execution = false;
        for ( const auto& exec_job : execution_queue ) {
            if ( exec_job.database_queue_id == available_queue[i].database_queue_id ) {
                found_in_execution = true;
                break;
            }
        }

        if ( ! found_in_execution ) {
            // Skip completed jobs if hide_completed_jobs is enabled (must match display logic)
            if ( hide_completed_jobs && available_queue[i].queue_status == "complete" ) {
                continue;
            }
            if ( process_job_for_queue(&available_queue[i], true, i) ) {
                i--; // Adjust index after erase
            }
        }
    }

    // Show warning if some jobs were blocked
    if ( ! blocked_jobs.empty( ) ) {
        wxString message = "The following jobs cannot be added to execution queue:\n\n";
        for ( const auto& job : blocked_jobs ) {
            message += "• " + job + "\n";
        }
        message += "\nOnly incomplete jobs can be queued for execution.";
        wxMessageBox(message, "Some Jobs Not Added", wxOK | wxICON_INFORMATION);
    }

    if ( ! selected_job_ids.empty( ) ) {
        SaveQueueToDatabase( );
        UpdateQueueDisplay( );
    }
}

void TemplateMatchQueueManager::OnRemoveFromQueueClick(wxCommandEvent& event) {
    // Move selected jobs from execution queue to available jobs (queue_order = -1)
    std::vector<long> selected_database_ids;
    long              selected_row = execution_queue_ctrl->GetNextItem(-1, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
    while ( selected_row != -1 ) {
        // Get the database_queue_id from the item data
        long database_queue_id = execution_queue_ctrl->GetItemData(selected_row);
        selected_database_ids.push_back(database_queue_id);
        selected_row = execution_queue_ctrl->GetNextItem(selected_row, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
    }

    if ( selected_database_ids.empty( ) ) {
        wxMessageBox("Please select jobs to remove from the execution queue.",
                     "No Selection", wxOK | wxICON_WARNING);
        return;
    }

    wxMessageDialog dialog(this,
                           wxString::Format("Remove %zu selected job(s) from execution queue?\n\n"
                                            "They will be moved to Available Jobs.",
                                            selected_database_ids.size( )),
                           "Confirm Remove from Queue", wxYES_NO | wxICON_QUESTION);

    if ( dialog.ShowModal( ) == wxID_YES ) {
        if constexpr ( skip_search_execution_for_queue_debugging ) {
            wxPrintf("\n=== REMOVING JOBS FROM EXECUTION QUEUE ===\n");
            wxPrintf("Queue state BEFORE removal:\n");
            PrintQueueState( );
        }

        // Get database_queue_ids for selected items (already collected above)
        std::vector<long> selected_ids = selected_database_ids;

        if constexpr ( skip_search_execution_for_queue_debugging ) {
            for ( long id : selected_ids ) {
                for ( const auto& job : execution_queue ) {
                    if ( job.database_queue_id == id ) {
                        wxPrintf("Selected job: ID=%ld, queue_order=%d\n",
                                 job.database_queue_id, job.queue_order);
                        break;
                    }
                }
            }
        }

        // Move jobs to available table (queue_order = -1)
        for ( long job_id : selected_ids ) {
            for ( auto& job : execution_queue ) {
                if ( job.database_queue_id == job_id ) {
                    if constexpr ( skip_search_execution_for_queue_debugging ) {
                        wxPrintf("Moving job %ld from queue_order %d to -1 (available)\n",
                                 job_id, job.queue_order);
                    }
                    job.queue_order = -1; // Move to available jobs
                    break;
                }
            }
        }

        // Renumber remaining execution queue jobs
        int new_position = 0;
        for ( auto& job : execution_queue ) {
            if ( job.queue_order >= 0 ) { // Still in execution queue
                job.queue_order = new_position++;
            }
        }

        if constexpr ( skip_search_execution_for_queue_debugging ) {
            wxPrintf("Queue state AFTER removal and renumbering:\n");
            PrintQueueState( );
        }

        UpdateQueueDisplay( );
        SaveQueueToDatabase( );
        wxPrintf("Removed %zu jobs from execution queue\n", selected_ids.size( ));

        if constexpr ( skip_search_execution_for_queue_debugging ) {
            wxPrintf("=== REMOVAL COMPLETE ===\n\n");
        }
    }
}

void TemplateMatchQueueManager::OnBeginDrag(wxListEvent& event) {
    // Note: This event handler exists for compatibility but manual drag-and-drop
    // is now handled through mouse events (OnMouseLeftDown, OnMouseMotion, OnMouseLeftUp)
    wxPrintf("OnBeginDrag called - using manual mouse drag implementation\n");
}

void TemplateMatchQueueManager::CleanupDragState( ) {
    if ( drag_in_progress ) {
        drag_in_progress = false;
        mouse_down       = false;
        dragged_row      = -1;
        dragged_job_id   = -1;
    }
}

void TemplateMatchQueueManager::LoadQueueFromDatabase( ) {
    MyDebugPrint("LoadQueueFromDatabase called. match_template_panel_ptr=%p", match_template_panel_ptr);
    MyDebugAssertTrue(match_template_panel_ptr != nullptr, "LoadQueueFromDatabase: match_template_panel_ptr is null");
    MyDebugAssertTrue(main_frame != nullptr, "LoadQueueFromDatabase: main_frame is null");
    MyDebugAssertTrue(main_frame->current_project.is_open, "LoadQueueFromDatabase: no project open");

    MyDebugPrint("Loading queue from database...");
    execution_queue.clear( );

    // Get all queue IDs from database in order
    std::vector<long> queue_ids;
    main_frame->current_project.database.GetQueuedTemplateMatchIDs(queue_ids);
    MyDebugPrint("Found %zu queue items in database", queue_ids.size( ));

    // Load each queue item from database and separate into execution vs available
    std::vector<TemplateMatchQueueItem> execution_items;
    std::vector<TemplateMatchQueueItem> available_items;

    for ( size_t i = 0; i < queue_ids.size( ); i++ ) {
        TemplateMatchQueueItem temp_item;
        temp_item.database_queue_id = queue_ids[i];

        // Load item details from database
        bool success = main_frame->current_project.database.GetQueueItemByID(
                queue_ids[i],
                temp_item.search_name,
                temp_item.search_id, // This will be loaded from SEARCH_ID
                temp_item.queue_order, // This will be loaded from QUEUE_POSITION
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

        if ( success ) {
            // Compute queue status from completion data
            std::pair<int, int> completion_counts;
            if ( temp_item.search_id > 0 ) {
                // Use template match job ID for completion tracking
                completion_counts = main_frame->current_project.database.GetSearchCompletionCounts(temp_item.search_id, temp_item.image_group_id);
            }
            else {
                // No results yet, so completion is 0/N where N is image count in group
                // Use GetSearchCompletionCounts with dummy search ID (-1) to get total count
                std::pair<int, int> group_counts = main_frame->current_project.database.GetSearchCompletionCounts(-1, temp_item.image_group_id);
                completion_counts                = std::make_pair(0, group_counts.second);
            }
            temp_item.queue_status = ComputeStatusFromProgress(completion_counts.first, completion_counts.second, currently_running_id, temp_item.database_queue_id);

            // Force completed/failed/partial jobs to available queue regardless of stored queue_order
            if ( temp_item.queue_status == "complete" || temp_item.queue_status == "failed" || temp_item.queue_status == "partial" ) {
                temp_item.queue_order = -1; // Move completed/failed/partial jobs to available queue
                available_items.push_back(temp_item);
            }
            else if ( temp_item.queue_order == -1 ) {
                // Available job
                available_items.push_back(temp_item);
            }
            else if ( temp_item.queue_order > 0 ) {
                // Execution queue job - convert from 1-based to 0-based for sorting
                temp_item.queue_order--; // Convert 1,2,3... to 0,1,2...
                execution_items.push_back(temp_item);
            }
            else {
                // Invalid queue_order (0) - treat as available job
                temp_item.queue_order = -1;
                available_items.push_back(temp_item);
            }
        }
        else {
            MyAssertTrue(false, "Failed to load queue item from database - queue_id=%ld, GetQueueItemByID returned false",
                         queue_ids[i]);
        }
    }

    // Sort execution items by their stored queue_order for proper sequencing
    std::sort(execution_items.begin( ), execution_items.end( ),
              [](const TemplateMatchQueueItem& a, const TemplateMatchQueueItem& b) {
                  return a.queue_order < b.queue_order;
              });

    // Reassign consecutive 0-based queue positions to execution items
    for ( size_t i = 0; i < execution_items.size( ); i++ ) {
        execution_items[i].queue_order = int(i);
    }

    // Crash recovery: Reset any orphaned "running" jobs to "failed"
    // On project load, no jobs should be running since there's no detach mechanism
    bool found_orphaned_jobs = false;
    for ( auto& item : execution_items ) {
        if ( item.queue_status == "running" ) {
            wxPrintf("CRASH RECOVERY: Found orphaned running job %ld, marking as failed\n", item.database_queue_id);
            item.queue_status = "failed";
            item.queue_order  = -1; // Move to available queue
            available_items.push_back(item);
            found_orphaned_jobs = true;

            // Note: Queue status is now computed from completion data, no database update needed
            main_frame->current_project.database.UpdateQueuePosition(item.database_queue_id, -1);
        }
    }
    for ( auto& item : available_items ) {
        if ( item.queue_status == "running" ) {
            wxPrintf("CRASH RECOVERY: Found orphaned running job %ld, marking as failed\n", item.database_queue_id);
            item.queue_status   = "failed";
            found_orphaned_jobs = true;

            // Note: Queue status is now computed from completion data, no database update needed
        }
    }

    // Remove orphaned jobs from execution_items since they're now in available_items
    if ( found_orphaned_jobs ) {
        execution_items.erase(
                std::remove_if(execution_items.begin( ), execution_items.end( ),
                               [](const TemplateMatchQueueItem& item) { return item.queue_status == "failed" && item.queue_order == -1; }),
                execution_items.end( ));
    }

    // Combine execution and available items into the main queue
    execution_queue.clear( );
    for ( const auto& item : execution_items ) {
        execution_queue.push_back(item);
    }
    for ( const auto& item : available_items ) {
        execution_queue.push_back(item);
    }

    // Save crash recovery changes back to database
    if ( found_orphaned_jobs ) {
        wxPrintf("CRASH RECOVERY: Saving corrected job statuses to database\n");
        SaveQueueToDatabase( );
    }

    // Auto-populate available searches that have no queue entries (orphaned searches)
    PopulateAvailableSearchesNotInQueueFromDatabase( );

    // Update display
    UpdateQueueDisplay( );
}

void TemplateMatchQueueManager::SaveQueueToDatabase( ) {
    MyDebugAssertTrue(match_template_panel_ptr != nullptr, "SaveQueueToDatabase: match_template_panel_ptr is null");
    MyDebugAssertTrue(main_frame != nullptr, "SaveQueueToDatabase: main_frame is null");
    MyDebugAssertTrue(main_frame->current_project.is_open, "SaveQueueToDatabase: no project open");

    // Update queue position for all items in queue (status is now computed from completion data)
    for ( const auto& item : execution_queue ) {

        // Convert 0-based queue positions to 1-based for database storage
        // Keep -1 as-is for available jobs
        int db_queue_order = (item.queue_order >= 0) ? (item.queue_order + 1) : item.queue_order;
        main_frame->current_project.database.UpdateQueuePosition(item.database_queue_id, db_queue_order);
    }
}

bool TemplateMatchQueueManager::UpdateQueueItemInDatabase(const TemplateMatchQueueItem& item) {
    MyDebugAssertTrue(main_frame != nullptr, "UpdateQueueItemInDatabase: main_frame is null");
    MyDebugAssertTrue(main_frame->current_project.is_open, "UpdateQueueItemInDatabase: no project open");
    MyDebugAssertTrue(item.database_queue_id > 0, "UpdateQueueItemInDatabase: invalid database_queue_id");

    // First verify the item exists in the database
    wxString check_sql = wxString::Format("SELECT COUNT(*) FROM TEMPLATE_MATCH_QUEUE WHERE QUEUE_ID = %ld;",
                                          item.database_queue_id);
    int count = main_frame->current_project.database.ReturnSingleIntFromSelectCommand(check_sql);

    if (count == 0) {
        wxString error_msg = wxString::Format("Queue item %ld not found in database", item.database_queue_id);
        wxMessageBox(error_msg, "Database Error", wxOK | wxICON_ERROR);
        MyDebugAssertTrue(count > 0, "Queue item %ld not found in database", item.database_queue_id);
        return false;
    }

    // Build UPDATE statement
    // Note: wxString::Format/Printf has a hard limit of 30 arguments due to WX_DEFINE_VARARG_FUNC macro
    // implementation. Since we need to update 32 fields, we split into two Format calls.
    // The proper solution would be to use prepared statements like in database.cpp, but that would
    // require adding a new method to the Database class.

    // Part 1: Update first 15 fields
    wxString update_sql_part1 = wxString::Format(
        "UPDATE TEMPLATE_MATCH_QUEUE SET "
        "JOB_NAME = '%s', "
        "IMAGE_GROUP_ID = %d, "
        "REFERENCE_VOLUME_ASSET_ID = %d, "
        "RUN_PROFILE_ID = %d, "
        "USE_GPU = %d, "
        "USE_FAST_FFT = %d, "
        "SYMMETRY = '%s', "
        "PIXEL_SIZE = %f, "
        "VOLTAGE = %f, "
        "SPHERICAL_ABERRATION = %f, "
        "AMPLITUDE_CONTRAST = %f, "
        "DEFOCUS1 = %f, "
        "DEFOCUS2 = %f, "
        "DEFOCUS_ANGLE = %f, "
        "PHASE_SHIFT = %f",
        item.search_name.ToUTF8().data(),
        item.image_group_id,
        item.reference_volume_asset_id,
        item.run_profile_id,
        item.use_gpu ? 1 : 0,
        item.use_fast_fft ? 1 : 0,
        item.symmetry.ToUTF8().data(),
        item.pixel_size,
        item.voltage,
        item.spherical_aberration,
        item.amplitude_contrast,
        item.defocus1,
        item.defocus2,
        item.defocus_angle,
        item.phase_shift);

    // Part 2: Update remaining 16 fields plus WHERE clause
    wxString update_sql_part2 = wxString::Format(
        ", LOW_RESOLUTION_LIMIT = %f"
        ", HIGH_RESOLUTION_LIMIT = %f"
        ", OUT_OF_PLANE_ANGULAR_STEP = %f"
        ", IN_PLANE_ANGULAR_STEP = %f"
        ", DEFOCUS_SEARCH_RANGE = %f"
        ", DEFOCUS_STEP = %f"
        ", PIXEL_SIZE_SEARCH_RANGE = %f"
        ", PIXEL_SIZE_STEP = %f"
        ", REFINEMENT_THRESHOLD = %f"
        ", REF_BOX_SIZE_IN_ANGSTROMS = %f"
        ", MASK_RADIUS = %f"
        ", MIN_PEAK_RADIUS = %f"
        ", XY_CHANGE_THRESHOLD = %f"
        ", EXCLUDE_ABOVE_XY_THRESHOLD = %d"
        ", CUSTOM_CLI_ARGS = '%s'"
        " WHERE QUEUE_ID = %ld;",
        item.low_resolution_limit,
        item.high_resolution_limit,
        item.out_of_plane_angular_step,
        item.in_plane_angular_step,
        item.defocus_search_range,
        item.defocus_step,
        item.pixel_size_search_range,
        item.pixel_size_step,
        item.refinement_threshold,
        item.ref_box_size_in_angstroms,
        item.mask_radius,
        item.min_peak_radius,
        item.xy_change_threshold,
        item.exclude_above_xy_threshold ? 1 : 0,
        item.custom_cli_args.ToUTF8().data(),
        item.database_queue_id);

    // Combine the two parts
    wxString update_sql = update_sql_part1 + update_sql_part2;

    // Execute the update within a transaction
    main_frame->current_project.database.Begin();

    // ExecuteSQL returns SQLITE_OK (0) on success, non-zero on error
    int sql_result = main_frame->current_project.database.ExecuteSQL(update_sql);

    if (sql_result != SQLITE_OK) {
        main_frame->current_project.database.Commit();  // Commit to end transaction

        wxMessageBox("Failed to update queue item in database", "Database Error", wxOK | wxICON_ERROR);
        return false;
    }

    // Commit the transaction
    main_frame->current_project.database.Commit();
    wxPrintf("Successfully updated queue item %ld in database\n", item.database_queue_id);
    return true;
}

// Manual drag and drop implementation for wxListCtrl
void TemplateMatchQueueManager::OnMouseLeftDown(wxMouseEvent& event) {

    // Prevent drag operations during display updates
    if ( updating_display ) {
        event.Skip( );
        return;
    }

    mouse_down     = true;
    drag_start_pos = event.GetPosition( );

    // Find which item was clicked
    int  flags;
    long hit_item = queue_list_ctrl->HitTest(drag_start_pos, flags);
    if ( hit_item != wxNOT_FOUND ) {
        dragged_row = hit_item;

        // Find the corresponding job
        for ( const auto& job : execution_queue ) {
            if ( job.queue_order == hit_item ) {
                dragged_job_id = job.database_queue_id;
                break;
            }
        }
    }

    event.Skip( );
}

void TemplateMatchQueueManager::OnMouseMotion(wxMouseEvent& event) {
    if ( ! mouse_down || updating_display ) {
        event.Skip( );
        return;
    }

    // Check if we've moved far enough to start a drag
    wxPoint current_pos = event.GetPosition( );
    int     dx          = current_pos.x - drag_start_pos.x;
    int     dy          = current_pos.y - drag_start_pos.y;

    if ( abs(dx) > 5 || abs(dy) > 5 ) { // Start drag after moving 5 pixels
        if ( ! drag_in_progress && dragged_row != -1 ) {
            drag_in_progress = true;
            queue_list_ctrl->SetCursor(wxCursor(wxCURSOR_HAND));
        }
    }

    event.Skip( );
}

void TemplateMatchQueueManager::OnMouseLeftUp(wxMouseEvent& event) {

    if ( drag_in_progress ) {
        // Find drop target
        wxPoint drop_pos = event.GetPosition( );
        int     flags;
        long    drop_item = queue_list_ctrl->HitTest(drop_pos, flags);

        if ( drop_item != wxNOT_FOUND && drop_item != dragged_row ) {
            // Perform the reorder
            ReorderQueueItems(dragged_row, drop_item);
        }
    }

    // Reset drag state
    mouse_down       = false;
    drag_in_progress = false;
    dragged_row      = -1;
    dragged_job_id   = -1;
    queue_list_ctrl->SetCursor(wxCursor(wxCURSOR_ARROW));

    event.Skip( );
}

void TemplateMatchQueueManager::ReorderQueueItems(int old_position, int new_position) {

    // Find the job being moved
    TemplateMatchQueueItem* moved_job = nullptr;
    for ( auto& job : execution_queue ) {
        if ( job.queue_order == old_position ) {
            moved_job = &job;
            break;
        }
    }

    if ( ! moved_job ) {
        wxPrintf("ERROR: Could not find job at position %d\n", old_position);
        return;
    }

    // Adjust queue orders
    if ( old_position < new_position ) {
        // Moving down: shift items up
        for ( auto& job : execution_queue ) {
            if ( job.queue_order > old_position && job.queue_order <= new_position ) {
                job.queue_order--;
            }
        }
    }
    else {
        // Moving up: shift items down
        for ( auto& job : execution_queue ) {
            if ( job.queue_order >= new_position && job.queue_order < old_position ) {
                job.queue_order++;
            }
        }
    }

    // Set new position for moved job
    moved_job->queue_order = new_position;

    // Update display
    UpdateQueueDisplay( );
}

void TemplateMatchQueueManager::ValidateQueueConsistency( ) const {
    int  running_jobs_count = 0;
    long found_running_id   = -1;

    for ( size_t i = 0; i < execution_queue.size( ); ++i ) {
        const auto& item = execution_queue[i];

        // Validate basic item consistency
        MyDebugAssertTrue(item.database_queue_id >= 0, "Queue item %zu has invalid database_queue_id: %ld", i, item.database_queue_id);
        MyDebugAssertFalse(item.search_name.IsEmpty( ), "Queue item %zu (ID: %ld) has empty search_name", i, item.database_queue_id);
        MyDebugAssertTrue(item.queue_status == "pending" || item.queue_status == "running" ||
                                  item.queue_status == "complete" || item.queue_status == "failed",
                          "Queue item %zu (ID: %ld) has invalid status: %s", i, item.database_queue_id, item.queue_status.mb_str( ).data( ));

        // Track running jobs
        if ( item.queue_status == "running" ) {
            running_jobs_count++;
            found_running_id = item.database_queue_id;
        }

        // Check for duplicate IDs
        for ( size_t j = i + 1; j < execution_queue.size( ); ++j ) {
            MyDebugAssertTrue(execution_queue[j].database_queue_id != item.database_queue_id,
                              "Duplicate database_queue_id %ld found at indices %zu and %zu", item.database_queue_id, i, j);
        }
    }

    // Validate running state consistency
    MyDebugAssertTrue(running_jobs_count <= 1, "Multiple running jobs found (%d), should be at most 1", running_jobs_count);

    if ( running_jobs_count == 1 ) {
        MyDebugAssertTrue(currently_running_id == found_running_id,
                          "currently_running_id (%ld) doesn't match running job ID (%ld)", currently_running_id, found_running_id);
        MyDebugAssertTrue(IsJobRunning( ), "IsJobRunning() returns false but running job exists");
    }
    else {
        MyDebugAssertTrue(currently_running_id == -1, "currently_running_id is %ld but no running jobs found", currently_running_id);
        MyDebugAssertFalse(IsJobRunning( ), "IsJobRunning() returns true but no running jobs found");
    }
}

void TemplateMatchQueueManager::ContinueQueueExecution( ) {
    // Instance method to continue queue execution after a job completes
    // This is called from ProcessAllJobsFinished to continue with the next job

    wxPrintf("ContinueQueueExecution: currently_running_id=%ld, finalizing=%s\n",
             currently_running_id, job_is_finalizing ? "true" : "false");

    if ( IsJobRunning( ) ) {
        // A job is already running or finalizing, don't start another
        if ( job_is_finalizing ) {
            wxPrintf("ContinueQueueExecution: Job is finalizing, not continuing queue execution\n");
        } else {
            wxPrintf("ContinueQueueExecution: Job %ld is still running, not continuing queue execution\n", currently_running_id);
        }
        return;
    }

    // Continue execution using this instance
    wxPrintf("ContinueQueueExecution: No job running, continuing with next job\n");
    RunNextJob( );
}

bool TemplateMatchQueueManager::ExecutionQueueHasActiveItems() const {
    // Check if there are any items in the execution queue (priority >= 0)
    // These are items waiting to run (pending/failed/partial)
    // Running items are at priority -1 in the available queue
    for (const auto& item : execution_queue) {
        if (item.queue_order >= 0) {
            return true;
        }
    }
    return false;
}

void TemplateMatchQueueManager::OnJobEnteringFinalization(long database_queue_id) {
    // Mark that a job is entering finalization phase
    // This prevents auto-advance from starting a new job prematurely
    wxPrintf("Job %ld entering finalization phase\n", database_queue_id);

    // Only set the finalization flag, don't clear currently_running_id yet
    // because UpdateJobStatus still needs it for validation
    if ( currently_running_id == database_queue_id ) {
        job_is_finalizing = true;
    }
}

void TemplateMatchQueueManager::OnJobCompleted(long database_queue_id, bool success) {
    // Ensure database is available for status updates
    MyDebugAssertTrue(IsDatabaseAvailable(main_frame), "OnJobCompleted: Database not available");

    if constexpr ( skip_search_execution_for_queue_debugging ) {
        wxPrintf("\n=== OnJobCompleted called ===\n");
        wxPrintf("Job ID: %ld, Success: %s\n", database_queue_id, success ? "true" : "false");
        wxPrintf("Currently running ID was: %ld\n", currently_running_id);
        wxPrintf("Queue state at job completion:\n");
        PrintQueueState( );
    }

    wxPrintf("Queue manager received job completion notification for job %ld (success: %s)\n",
             database_queue_id, success ? "true" : "false");

    // Update job status in our queue
    const wxString& status = success ? "complete" : "failed";
    UpdateJobStatus(database_queue_id, status);

    // Update the completed job (which should already be at priority -1 from when it started running)
    bool job_found = false;
    for ( auto& job : execution_queue ) {
        if ( job.database_queue_id == database_queue_id ) {
            wxPrintf("Found completed job %ld at priority %d, updating status to %s\n",
                     database_queue_id, job.queue_order, status);
            job.queue_status = status;
            // Job should already be at queue_order = -1 (available queue) from when it started running
            if ( job.queue_order != -1 ) {
                wxPrintf("WARNING: Completed job %ld was not at priority -1 (was %d), correcting\n",
                         database_queue_id, job.queue_order);
                job.queue_order = -1;
            }
            job_found = true;
            break;
        }
    }

    if ( ! job_found ) {
        wxPrintf("WARNING: Could not find completed job %ld in execution queue\n", database_queue_id);
    }

    // No need to renumber - jobs were already shifted when the completed job started running
    // The execution queue should already have consecutive priorities: 0, 1, 2, 3...

    // Clear currently running ID and finalization flag since job is completely done
    if ( currently_running_id == database_queue_id ) {
        currently_running_id = -1;
    }
    job_is_finalizing = false;  // Job has finished finalizing

    // Update button state since no job is running
    UpdateButtonState();

    // Update both displays to show new status and position changes
    UpdateQueueDisplay( );
    UpdateAvailableJobsDisplay( );

    // Save changes to database - using auto-commit mode (each UPDATE commits immediately)
    SaveQueueToDatabase( );

    if constexpr ( skip_search_execution_for_queue_debugging ) {
        wxPrintf("\nQueue state after job completion processing:\n");
        PrintQueueState( );
    }

    // Only auto-progress if enabled (default false - user controls progression)
    if ( auto_progress_queue ) {
        if constexpr ( skip_search_execution_for_queue_debugging ) {
            wxPrintf("\n=== Auto-progressing to next job in queue ===\n");
        }

        // Add a small delay before auto-advancing to ensure all finalization completes
        // This prevents race conditions where the next job might start before the previous
        // job's results are fully written to the database or UI is updated
        wxTimer* delay_timer = new wxTimer();
        delay_timer->Bind(wxEVT_TIMER, [this, delay_timer](wxTimerEvent&) {
            wxPrintf("Auto-advance timer fired - progressing queue\n");
            ProgressExecutionQueue( );
            delete delay_timer;  // Clean up the timer
        });
        delay_timer->StartOnce(500);  // 500ms delay before auto-advancing
        wxPrintf("Scheduled auto-advance in 500ms\n");
    }
    else {
        if constexpr ( skip_search_execution_for_queue_debugging ) {
            wxPrintf("\nJob completed - auto-progression disabled (user controls queue progression)\n");
        }
    }

    if constexpr ( skip_search_execution_for_queue_debugging ) {
        wxPrintf("=== OnJobCompleted finished ===\n\n");
    }
}

// Selection-based methods removed - using priority-based execution only

// RunNextSelectedJob method removed - using priority-based execution only

void TemplateMatchQueueManager::DeselectJobInUI(long database_queue_id) {
    // Find the row corresponding to this job ID
    for ( size_t i = 0; i < execution_queue.size( ); ++i ) {
        if ( execution_queue[i].database_queue_id == database_queue_id ) {
            queue_list_ctrl->SetItemState(int(i), 0, wxLIST_STATE_SELECTED);
            wxPrintf("Deselected job %ld from UI (row %zu)\n", database_queue_id, i);
            return;
        }
    }
    wxPrintf("Could not find job %ld to deselect in UI\n", database_queue_id);
}

SearchCompletionInfo TemplateMatchQueueManager::GetSearchCompletionInfo(long queue_id) {
    SearchCompletionInfo info;
    info.search_id = -1;  // Will be set when we find the actual search_id

    // Note: We receive a QUEUE_ID and need to find the corresponding SEARCH_ID
    // QUEUE_ID: From TEMPLATE_MATCH_QUEUE table (unique for each queue entry)
    // SEARCH_ID: From TEMPLATE_MATCH_LIST table (links to actual results)

    // Find the queue item and extract its search_id and image_group_id
    long search_id        = -1;
    int  image_group_id   = -1;
    bool found_queue_item = false;

    // Search both queues for the item
    for ( const auto& item : execution_queue ) {
        if ( item.database_queue_id == queue_id ) {
            search_id        = item.search_id;
            image_group_id   = item.image_group_id;
            found_queue_item = true;
            break;
        }
    }

    if ( ! found_queue_item ) {
        for ( const auto& item : available_queue ) {
            if ( item.database_queue_id == queue_id ) {
                search_id        = item.search_id;
                image_group_id   = item.image_group_id;
                found_queue_item = true;
                break;
            }
        }
    }

    if ( ! found_queue_item ) {
        MyAssertTrue(false, "GetSearchCompletionInfo could not find queue_id %ld in either execution_queue (%zu items) or available_queue (%zu items)",
                     queue_id, execution_queue.size( ), available_queue.size( ));
    }

    // Store the search_id in the info struct
    info.search_id = search_id;

    // If still not found, get image group from database by querying the TEMPLATE_MATCH_QUEUE table
    if ( image_group_id == -1 && search_id > 0 && main_frame && main_frame->current_project.is_open ) {
        // Get the image group from the TEMPLATE_MATCH_QUEUE table
        wxString sql = wxString::Format(
                "SELECT IMAGE_GROUP_ID FROM TEMPLATE_MATCH_QUEUE "
                "WHERE SEARCH_ID = %ld LIMIT 1",
                search_id);

        image_group_id = main_frame->current_project.database.ReturnSingleIntFromSelectCommand(sql);
        if ( image_group_id <= 0 ) {
            // This should not happen as all searches must have a valid image_group_id
            MyAssertTrue(false, "Search ID %ld has no valid IMAGE_GROUP_ID in TEMPLATE_MATCH_QUEUE", search_id);
            image_group_id = -1;
        }
    }

    // Get completion counts from database
    if ( image_group_id > 0 && main_frame && main_frame->current_project.is_open ) {
        if ( search_id > 0 ) {
            // SEARCH_ID exists - get actual completion counts
            auto counts          = main_frame->current_project.database.GetSearchCompletionCounts(search_id, image_group_id);
            info.completed_count = counts.first;
            info.total_count     = counts.second;
        }
        else {
            // No SEARCH_ID yet (no results written) - show 0/N where N is the image group size
            wxString sql         = wxString::Format("SELECT COUNT(*) FROM IMAGE_GROUP_%d", image_group_id);
            info.total_count     = main_frame->current_project.database.ReturnSingleIntFromSelectCommand(sql);
            info.completed_count = 0;
        }

        // For already completed jobs with 0 completed count, set completed = total
        // This handles the case where the search was completed but GetSearchCompletionCounts returns 0
        if ( info.total_count > 0 && info.completed_count == 0 ) {
            // Check if this search should be marked as complete based on queue status
            bool search_is_complete = false;
            for ( const auto& item : execution_queue ) {
                if ( item.database_queue_id == queue_id && item.queue_status == "complete" ) {
                    search_is_complete = true;
                    break;
                }
            }
            if ( ! search_is_complete ) {
                for ( const auto& item : available_queue ) {
                    if ( item.database_queue_id == queue_id && item.queue_status == "complete" ) {
                        search_is_complete = true;
                        break;
                    }
                }
            }

            if ( search_is_complete ) {
                info.completed_count = info.total_count;
            }
        }
    }
    else {
        // Can't determine anything without image_group_id
        info.completed_count = 0;
        info.total_count     = 0;
    }

    // Debug output for n/N tracking
    wxPrintf("GetSearchCompletionInfo: Queue ID %ld, SEARCH_ID %ld, ImageGroup %d -> %d/%d\n",
             queue_id, search_id, image_group_id,
             info.completed_count, info.total_count);

    return info;
}

void TemplateMatchQueueManager::RefreshSearchCompletionInfo( ) {
    // Update execution queue display
    UpdateQueueDisplay( );

    // Update available jobs display
    UpdateAvailableJobsDisplay( );
}

void TemplateMatchQueueManager::OnResultAdded(long search_id) {
    wxPrintf("OnResultAdded called for job %ld - refreshing n/N display\n", search_id);

    // Refresh the completion info for this job and update displays
    RefreshSearchCompletionInfo( );
}

void TemplateMatchQueueManager::UpdateSearchIdForQueueItem(long queue_database_queue_id, long database_search_id) {
    // Update the queue item with the actual SEARCH_ID for correct n/N tracking
    wxPrintf("UpdateSearchIdForQueueItem: Mapping queue ID %ld to database SEARCH_ID %ld\n",
             queue_database_queue_id, database_search_id);

    // Check execution queue first
    for ( auto& item : execution_queue ) {
        if ( item.database_queue_id == queue_database_queue_id ) {
            wxPrintf("Found queue item %ld in execution queue, setting search_id = %ld\n",
                     queue_database_queue_id, database_search_id);
            item.search_id = database_search_id;

            // Update database with the template match job ID
            if ( main_frame && main_frame->current_project.is_open ) {
                main_frame->current_project.database.UpdateSearchIdInQueueTable(queue_database_queue_id, database_search_id);
            }
            return;
        }
    }

    // Check available queue
    for ( auto& item : available_queue ) {
        if ( item.database_queue_id == queue_database_queue_id ) {
            wxPrintf("Found queue item %ld in available queue, setting search_id = %ld\n",
                     queue_database_queue_id, database_search_id);
            item.search_id = database_search_id;

            // Update database with the template match job ID
            if ( main_frame && main_frame->current_project.is_open ) {
                main_frame->current_project.database.UpdateSearchIdInQueueTable(queue_database_queue_id, database_search_id);
            }
            return;
        }
    }

    wxPrintf("Warning: Could not find queue item %ld to update with database job ID %ld\n",
             queue_database_queue_id, database_search_id);
}

/**
 * @brief Populates the available queue with orphaned template match searches
 *
 * This method finds all template matching searches that have been run (have results in TEMPLATE_MATCH_LIST)
 * but don't have a corresponding entry in TEMPLATE_MATCH_QUEUE. These "orphaned" searches appear in the
 * "Available Searches" section of the queue manager UI and typically occur when:
 * - Projects are migrated from older versions of cisTEM that didn't have the queue system
 * - Users explicitly removed the queue entry after the search started
 *
 * The method:
 * 1. Retrieves all unique SEARCH_IDs from TEMPLATE_MATCH_LIST table along with their IMAGE_GROUP_IDs
 * 2. Checks if each SEARCH_ID already exists in either the execution or available queue
 * 3. Creates minimal queue items for searches not already tracked, with status derived from completion counts
 *
 * @note Queue items created here have database_queue_id = -1 since they don't exist in TEMPLATE_MATCH_QUEUE table
 * @note All new searches go through the queue - searches only appear here if explicitly removed by the user or from migration
 */
void TemplateMatchQueueManager::PopulateAvailableSearchesNotInQueueFromDatabase( ) {
    MyDebugAssertTrue(main_frame != nullptr, "PopulateAvailableSearchesNotInQueueFromDatabase called with null main_frame");
    MyDebugAssertTrue(main_frame->current_project.is_open, "PopulateAvailableSearchesNotInQueueFromDatabase called with no project open");

    // Get all template match search IDs and their image groups from database
    std::vector<std::pair<long, int>> search_id_and_group_pairs;
    main_frame->current_project.database.GetAllTemplateMatchSearchIds(search_id_and_group_pairs);

    for ( const auto& [search_id, image_group_id] : search_id_and_group_pairs ) {

        // Check if this search is already in our queues
        bool found_in_execution = false;
        bool found_in_available = false;

        for ( const auto& item : execution_queue ) {
            if ( item.search_id == search_id ) {
                found_in_execution = true;
                break;
            }
        }

        if ( ! found_in_execution ) {
            for ( const auto& item : available_queue ) {
                if ( item.search_id == search_id ) {
                    found_in_available = true;
                    break;
                }
            }
        }

        // If not found in either queue, add to available queue
        if ( ! found_in_execution && ! found_in_available ) {
            // Skip orphaned searches for now - they shouldn't be in the queue manager
            // These are searches that have results but no queue entry, likely from
            // legacy projects or manual database edits
            wxPrintf("Skipping orphaned search %ld - has results but no queue entry\n", search_id);
            continue;
        }
    }

    // Refresh the display
    UpdateAvailableJobsDisplay( );
}

void TemplateMatchQueueManager::OnPanelDisplayToggle(wxCommandEvent& event) {
    if (!match_template_panel_ptr) return;

    bool show_progress = event.IsChecked();

    // Update button label to show current state
    if (show_progress) {
        panel_display_toggle->SetLabel("Show Progress Panel");
    } else {
        panel_display_toggle->SetLabel("Show Input Panel");
    }

    // Toggle the panels in MatchTemplatePanel
    if (show_progress) {
        // Show progress/running panels
        match_template_panel_ptr->StartPanel->Show(false);
        match_template_panel_ptr->ProgressPanel->Show(true);
        match_template_panel_ptr->InputPanel->Show(false);
        match_template_panel_ptr->ExpertPanel->Show(false);
        match_template_panel_ptr->InfoPanel->Show(false);
        match_template_panel_ptr->OutputTextPanel->Show(true);
        match_template_panel_ptr->ResultsPanel->Show(true);
        // Keep cancel button visible if job is running
        if (match_template_panel_ptr->running_job) {
            match_template_panel_ptr->CancelAlignmentButton->Show(true);
            match_template_panel_ptr->FinishButton->Show(false);
        }
    } else {
        // Show input panels
        match_template_panel_ptr->ProgressPanel->Show(false);
        match_template_panel_ptr->StartPanel->Show(true);
        match_template_panel_ptr->OutputTextPanel->Show(false);
        match_template_panel_ptr->ResultsPanel->Show(false);
        match_template_panel_ptr->InfoPanel->Show(true);
        match_template_panel_ptr->InputPanel->Show(true);
        match_template_panel_ptr->ExpertPanel->Show(true);
    }

    // Force layout refresh
    match_template_panel_ptr->Layout();
    match_template_panel_ptr->Update();
}