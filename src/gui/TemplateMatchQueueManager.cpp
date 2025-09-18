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
    : wxPanel(parent, wxID_ANY), match_template_panel_ptr(match_template_panel), currently_running_id(-1) {

    // Initialize state variables
    needs_database_load   = true; // Need to load from database on first access
    execution_in_progress = false; // No jobs running initially
    drag_in_progress      = false;
    updating_display      = false;  // Not updating display initially
    dragged_row           = -1;
    dragged_job_id        = -1;
    mouse_down           = false;

    // Create the main sizer
    wxBoxSizer* main_sizer = new wxBoxSizer(wxVERTICAL);

    // Create execution queue section (top)
    wxStaticText* execution_queue_label = new wxStaticText(this, wxID_ANY, "Execution Queue (jobs will run in order - drag to reorder):");
    execution_queue_ctrl                = new wxListCtrl(this, wxID_ANY,
                                                         wxDefaultPosition, wxSize(700, 200),
                                                         wxLC_REPORT | wxLC_SINGLE_SEL);

    // Add columns to execution queue (with queue order)
    execution_queue_ctrl->AppendColumn("Queue Order", wxLIST_FORMAT_LEFT, 80);
    execution_queue_ctrl->AppendColumn("ID", wxLIST_FORMAT_LEFT, 60);
    execution_queue_ctrl->AppendColumn("Job Name", wxLIST_FORMAT_LEFT, 200);
    execution_queue_ctrl->AppendColumn("Status", wxLIST_FORMAT_LEFT, 100);
    execution_queue_ctrl->AppendColumn("CLI Args", wxLIST_FORMAT_LEFT, 140);

    // wxListCtrl doesn't use EnableDragSource/EnableDropTarget - we'll implement manual drag and drop

    // Legacy compatibility - point to execution queue
    queue_list_ctrl = execution_queue_ctrl;

    // Create execution queue controls
    wxPanel*    execution_controls = new wxPanel(this, wxID_ANY);
    wxBoxSizer* execution_sizer    = new wxBoxSizer(wxHORIZONTAL);

    // Drag-and-drop controls
    assign_priority_button = new wxButton(execution_controls, wxID_ANY, "Assign Priority");
    assign_priority_button->Enable(false); // Disabled until items are dragged

    // Execution controls
    run_selected_button = new wxButton(execution_controls, wxID_ANY, "Run Selected");
    cancel_run_button   = new wxButton(execution_controls, wxID_ANY, "Cancel Active Run");
    cancel_run_button->Enable(false); // Disabled until job is running

    execution_sizer->Add(assign_priority_button, 0, wxALL, 5);
    execution_sizer->AddSpacer(20);
    execution_sizer->Add(run_selected_button, 0, wxALL, 5);
    execution_sizer->Add(cancel_run_button, 0, wxALL, 5);

    execution_controls->SetSizer(execution_sizer);

    // Create movement controls (between tables)
    wxPanel*    movement_controls = new wxPanel(this, wxID_ANY);
    wxBoxSizer* movement_sizer    = new wxBoxSizer(wxHORIZONTAL);

    add_to_queue_button      = new wxButton(movement_controls, wxID_ANY, "Add to Execution Queue ↑");
    remove_from_queue_button = new wxButton(movement_controls, wxID_ANY, "Remove from Execution Queue ↓");

    movement_sizer->Add(add_to_queue_button, 0, wxALL, 5);
    movement_sizer->Add(remove_from_queue_button, 0, wxALL, 5);

    movement_controls->SetSizer(movement_sizer);

    // Create available jobs section (bottom)
    wxStaticText* available_jobs_label = new wxStaticText(this, wxID_ANY, "Available Jobs (not queued for execution):");
    available_jobs_ctrl                = new wxListCtrl(this, wxID_ANY,
                                                        wxDefaultPosition, wxSize(700, 200),
                                                        wxLC_REPORT | wxLC_SINGLE_SEL);

    // Add columns to available jobs (no queue order column)
    available_jobs_ctrl->AppendColumn("ID", wxLIST_FORMAT_LEFT, 60);
    available_jobs_ctrl->AppendColumn("Job Name", wxLIST_FORMAT_LEFT, 200);
    available_jobs_ctrl->AppendColumn("Status", wxLIST_FORMAT_LEFT, 100);
    available_jobs_ctrl->AppendColumn("CLI Args", wxLIST_FORMAT_LEFT, 140);

    // Create general controls
    wxPanel*    general_controls = new wxPanel(this, wxID_ANY);
    wxBoxSizer* general_sizer    = new wxBoxSizer(wxHORIZONTAL);

    remove_selected_button = new wxButton(general_controls, wxID_ANY, "Remove Selected");
    clear_queue_button     = new wxButton(general_controls, wxID_ANY, "Clear All");

    general_sizer->Add(remove_selected_button, 0, wxALL, 5);
    general_sizer->Add(clear_queue_button, 0, wxALL, 5);

    general_controls->SetSizer(general_sizer);

    // Add all sections to main sizer (50/50 split with controls between)
    main_sizer->Add(execution_queue_label, 0, wxEXPAND | wxALL, 5);
    main_sizer->Add(execution_queue_ctrl, 1, wxEXPAND | wxALL, 5);
    main_sizer->Add(execution_controls, 0, wxEXPAND | wxALL, 5);
    main_sizer->Add(movement_controls, 0, wxEXPAND | wxALL, 5);
    main_sizer->Add(available_jobs_label, 0, wxEXPAND | wxALL, 5);
    main_sizer->Add(available_jobs_ctrl, 1, wxEXPAND | wxALL, 5);
    main_sizer->Add(general_controls, 0, wxEXPAND | wxALL, 5);

    SetSizer(main_sizer);

    // Connect events
    assign_priority_button->Bind(wxEVT_BUTTON, &TemplateMatchQueueManager::OnAssignPriorityClick, this);
    cancel_run_button->Bind(wxEVT_BUTTON, &TemplateMatchQueueManager::OnCancelRunClick, this);
    run_selected_button->Bind(wxEVT_BUTTON, &TemplateMatchQueueManager::OnRunSelectedClick, this);
    add_to_queue_button->Bind(wxEVT_BUTTON, &TemplateMatchQueueManager::OnAddToQueueClick, this);
    remove_from_queue_button->Bind(wxEVT_BUTTON, &TemplateMatchQueueManager::OnRemoveFromQueueClick, this);
    remove_selected_button->Bind(wxEVT_BUTTON, &TemplateMatchQueueManager::OnRemoveSelectedClick, this);
    clear_queue_button->Bind(wxEVT_BUTTON, &TemplateMatchQueueManager::OnClearQueueClick, this);

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

void TemplateMatchQueueManager::AddToQueue(const TemplateMatchQueueItem& item) {
    // Validate the queue item before adding
    MyDebugAssertTrue(item.image_group_id >= 0, "Cannot add item with invalid image_group_id: %d", item.image_group_id);
    MyDebugAssertTrue(item.reference_volume_asset_id >= 0, "Cannot add item with invalid reference_volume_asset_id: %d", item.reference_volume_asset_id);
    MyDebugAssertFalse(item.job_name.IsEmpty( ), "Cannot add item with empty job_name");
    MyDebugAssertTrue(item.queue_status == "pending" || item.queue_status == "running" || item.queue_status == "complete" || item.queue_status == "failed",
                      "Cannot add item with invalid queue_status: %s", item.queue_status.mb_str( ).data( ));

    if ( match_template_panel_ptr && main_frame && main_frame->current_project.is_open ) {
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
        new_item.template_match_id      = new_queue_id;

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
}

void TemplateMatchQueueManager::RemoveFromQueue(int index) {
    MyDebugAssertTrue(index >= 0, "RemoveFromQueue called with negative index: %d", index);
    MyDebugAssertTrue(index < execution_queue.size( ), "RemoveFromQueue called with index %d >= queue size %zu", index, execution_queue.size( ));
    MyDebugAssertFalse(execution_queue.empty( ), "Cannot remove from empty queue");

    if ( index >= 0 && index < execution_queue.size( ) ) {
        // Don't allow removal of currently running jobs
        MyDebugAssertTrue(execution_queue[index].queue_status != "running",
                          "Cannot remove currently running job (ID: %ld)", execution_queue[index].template_match_id);

        // Remove from database first
        if ( match_template_panel_ptr && main_frame && main_frame->current_project.is_open ) {
            main_frame->current_project.database.RemoveFromQueue(execution_queue[index].template_match_id);
        }

        execution_queue.erase(execution_queue.begin( ) + index);
        UpdateQueueDisplay( );
    }
}

void TemplateMatchQueueManager::ClearQueue( ) {
    // Clear all items and remove from database
    if ( match_template_panel_ptr && main_frame && main_frame->current_project.is_open ) {
        // Remove all items from database
        for ( const auto& item : execution_queue ) {
            main_frame->current_project.database.RemoveFromQueue(item.template_match_id);
        }
    }

    execution_queue.clear( );
    UpdateQueueDisplay( );
}

void TemplateMatchQueueManager::SetJobPosition(int job_index, int new_position) {
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
             execution_queue[job_index].template_match_id, current_position, new_position);
}

void TemplateMatchQueueManager::ProgressQueue( ) {
    // Find the job with queue_order = 1 (next to run)
    TemplateMatchQueueItem* next_job = nullptr;
    for ( auto& item : execution_queue ) {
        if ( item.queue_order == 1 && item.queue_status == "pending" ) {
            next_job = &item;
            break;
        }
    }

    if ( next_job ) {
        // Set the next job to running (queue_order = 0)
        next_job->queue_order  = 0;
        next_job->queue_status = "running";
        currently_running_id   = next_job->template_match_id;

        // Decrement all other jobs' queue orders
        for ( auto& item : execution_queue ) {
            if ( item.queue_order > 0 ) {
                item.queue_order--;
            }
        }

        wxPrintf("ProgressQueue: Started job %ld, decremented all other positions\n", next_job->template_match_id);

        // Execute the next job
        ExecuteJob(*next_job);
    }
    else {
        wxPrintf("ProgressQueue: No pending jobs found with queue_order = 1\n");
    }
}

// Old MoveItem methods removed - replaced with queue order system

wxColour TemplateMatchQueueManager::GetStatusColor(const wxString& status) {
    if ( status == "running" ) {
        return wxColour(0, 128, 0); // Green
    }
    else if ( status == "complete" ) {
        return wxColour(0, 0, 255); // Blue
    }
    else if ( status == "failed" ) {
        return wxColour(255, 0, 0); // Red
    }
    else { // pending
        return wxColour(255, 0, 0); // Red
    }
}

void TemplateMatchQueueManager::UpdateQueueDisplay( ) {
    wxPrintf("UpdateQueueDisplay called - execution_queue.size()=%zu\n", execution_queue.size()); // revert - debug output for troubleshooting drag-and-drop
    MyDebugAssertTrue(queue_list_ctrl != nullptr, "queue_list_ctrl is null in UpdateQueueDisplay");

    // revert - Prevent drag operations during display update to avoid GTK crashes
    updating_display = true;

    // Preserve current selections before rebuilding
    std::vector<long> selected_template_ids;
    wxPrintf("About to get selected items from wxListCtrl\n"); // revert - debug output for troubleshooting drag-and-drop

    // Get currently selected items in wxListCtrl
    long item = queue_list_ctrl->GetNextItem(-1, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
    while ( item != -1 ) {
        // For wxListCtrl, the item index directly corresponds to the row in our sorted display
        if ( item < int(execution_queue.size()) ) {
            // Find the job with this queue_order (since we sort by queue_order)
            for ( const auto& job : execution_queue ) {
                if ( job.queue_order == item ) {
                    selected_template_ids.push_back(job.template_match_id);
                    wxPrintf("UpdateQueueDisplay: Preserved selection for job %ld\n", job.template_match_id); // revert - extreme debug
                    break;
                }
            }
        }
        item = queue_list_ctrl->GetNextItem(item, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
    }

    queue_list_ctrl->DeleteAllItems( );

    // Sort execution_queue by queue_order so row position matches priority
    // Create a sorted index vector to avoid modifying the original queue
    std::vector<size_t> sorted_indices;
    for ( size_t i = 0; i < execution_queue.size( ); ++i ) {
        if ( execution_queue[i].queue_order >= 0 ) { // Only show items in execution queue
            sorted_indices.push_back(i);
        }
    }

    // Sort indices by queue_order
    std::sort(sorted_indices.begin( ), sorted_indices.end( ),
              [this](size_t a, size_t b) {
                  return execution_queue[a].queue_order < execution_queue[b].queue_order;
              });

    // Display sorted items
    for ( size_t idx = 0; idx < sorted_indices.size( ); ++idx ) {
        size_t i = sorted_indices[idx];
        MyDebugAssertTrue(i < execution_queue.size( ), "Queue index %zu out of bounds (size: %zu)", i, execution_queue.size( ));

        // For wxListCtrl, add item with first column text then set other columns
        long item_index = queue_list_ctrl->InsertItem(idx, wxString::Format("%d", execution_queue[i].queue_order));

        // Set the remaining columns
        queue_list_ctrl->SetItem(item_index, 1, wxString::Format("%ld", execution_queue[i].template_match_id));
        queue_list_ctrl->SetItem(item_index, 2, execution_queue[i].job_name);

        // Add status with a colored indicator prefix
        wxString status_display;
        if ( execution_queue[i].queue_status == "running" ) {
            status_display = "● " + execution_queue[i].queue_status; // Green circle
        }
        else if ( execution_queue[i].queue_status == "complete" ) {
            status_display = "✓ " + execution_queue[i].queue_status; // Check mark
        }
        else if ( execution_queue[i].queue_status == "failed" ) {
            status_display = "✗ " + execution_queue[i].queue_status; // X mark
        }
        else { // pending
            status_display = "○ " + execution_queue[i].queue_status; // Empty circle
        }
        queue_list_ctrl->SetItem(item_index, 3, status_display);
        queue_list_ctrl->SetItem(item_index, 4, execution_queue[i].custom_cli_args);
    }

    // Restore selections after rebuilding
    for ( long template_id : selected_template_ids ) {
        for ( size_t i = 0; i < execution_queue.size( ); ++i ) {
            if ( execution_queue[i].template_match_id == template_id ) {
                // revert - debug prints for drag and drop troubleshooting
                wxPrintf("RestoreSelection: Found job %ld at execution_queue index %zu, queue_order=%d\n",
                         template_id, i, execution_queue[i].queue_order);

                // The row should be based on queue_order, not array index!
                int row = execution_queue[i].queue_order;
                wxPrintf("RestoreSelection: Using row %d (queue_order) instead of %zu (array index)\n", row, i);

                // revert - Add safety checks to prevent array bounds errors
                if (row < 0 || row >= queue_list_ctrl->GetItemCount()) {
                    wxPrintf("ERROR: Row %d is out of bounds (list item count: %ld)\n", row, queue_list_ctrl->GetItemCount());
                    break;
                }

                // For wxListCtrl, we can directly select by row index
                wxPrintf("About to call SetItemState() for job %ld (row %d)\n", template_id, row); // revert - debug
                queue_list_ctrl->SetItemState(row, wxLIST_STATE_SELECTED, wxLIST_STATE_SELECTED);
                wxPrintf("SetItemState() completed for job %ld (row %d)\n", template_id, row); // revert - debug
                break;
            }
        }
    }

    // revert - Re-enable drag operations after display update is complete
    updating_display = false;
}

int TemplateMatchQueueManager::GetSelectedRow( ) {
    long selected_item = queue_list_ctrl->GetNextItem(-1, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
    if ( selected_item != -1 ) {
        return selected_item;
    }
    return -1;
}

void TemplateMatchQueueManager::RunSelectedJob( ) {
    // Get currently selected UI items
    std::vector<long> selected_rows;
    long selected_item = queue_list_ctrl->GetNextItem(-1, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
    while ( selected_item != -1 ) {
        selected_rows.push_back(selected_item);
        selected_item = queue_list_ctrl->GetNextItem(selected_item, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
    }

    if ( selected_rows.empty() ) {
        wxMessageBox("No jobs selected for execution", "No Selection", wxOK | wxICON_WARNING);
        return;
    }

    if ( IsJobRunning( ) ) {
        wxMessageBox("A job is already running. Please wait for it to complete.", "Job Running", wxOK | wxICON_WARNING);
        return;
    }

    // Collect selected job indices and reorder them to start from position 1
    std::vector<int> selected_indices;
    for ( long row : selected_rows ) {
        if ( row < int(execution_queue.size( )) ) {
            if ( execution_queue[row].queue_status == "pending" ) {
                selected_indices.push_back(row);
            }
        }
    }

    if ( selected_indices.empty( ) ) {
        wxMessageBox("No pending jobs selected", "No Pending Jobs", wxOK | wxICON_WARNING);
        return;
    }

    // Reorder selected jobs to positions 1, 2, 3, etc.
    for ( int i = 0; i < selected_indices.size( ); i++ ) {
        SetJobPosition(selected_indices[i], i + 1);
    }

    // Start the first job (position 1 → 0)
    ProgressQueue( );

    wxPrintf("Started execution of %zu selected jobs\n", selected_indices.size( ));
}

// RunAllJobs method removed - use Run Selected with multi-selection instead

void TemplateMatchQueueManager::RunNextJob( ) {
    MyPrintWithDetails("=== RUN NEXT JOB CALLED ===");

    // Check if a job is already running
    if ( IsJobRunning( ) ) {
        MyDebugPrint("Job %ld is already running - skipping RunNextJob", currently_running_id);
        return;
    }

    MyDebugPrint("No job currently running - searching for next pending job");

    // Find and run the next pending job
    TemplateMatchQueueItem* next_job = GetNextPendingJob( );
    if ( next_job != nullptr ) {
        MyDebugPrint("Found next pending job:");
        MyDebugPrint("  ID: %ld", next_job->template_match_id);
        MyDebugPrint("  Name: '%s'", next_job->job_name.mb_str( ).data( ));
        MyDebugPrint("  Status: '%s'", next_job->queue_status.mb_str( ).data( ));
        MyDebugPrint("  Image Group: %d", next_job->image_group_id);
        MyDebugPrint("  Reference Volume: %d", next_job->reference_volume_asset_id);
        MyDebugPrint("=== EXECUTING JOB %ld ===", next_job->template_match_id);

        ExecuteJob(*next_job);
    }
    else {
        MyPrintWithDetails("No pending jobs found - batch execution complete or no jobs to run");
    }
}

bool TemplateMatchQueueManager::ExecuteJob(TemplateMatchQueueItem& job_to_run) {
    // Basic validation moved to TMP's ExecuteJob method

    // Check if another job is already running
    MyDebugAssertFalse(IsJobRunning( ), "Attempted to execute job %ld while job %ld is already running", job_to_run.template_match_id, currently_running_id);
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

        // Use TMP's unified ExecuteJob method
        wxPrintf("Executing job %ld via unified method...\n", job_to_run.template_match_id);
        bool execution_success = match_template_panel_ptr->ExecuteJob(&job_to_run);

        if ( execution_success ) {
            // Job executed successfully - now mark it as running and track it
            UpdateJobStatus(job_to_run.template_match_id, "running");
            currently_running_id = job_to_run.template_match_id;
            wxPrintf("Job %ld started successfully and marked as running\n", job_to_run.template_match_id);
            // Job status will be updated to "complete" when the job finishes via ProcessAllJobsFinished
            return true;
        }
        else {
            wxPrintf("Failed to start job %ld\n", job_to_run.template_match_id);
            UpdateJobStatus(job_to_run.template_match_id, "failed");
            return false;
        }
    }
    else {
        // Critical failure - template panel not available
        MyAssertTrue(false, "Critical error: match_template_panel not available for job execution");
        wxPrintf("Error: match_template_panel not available\n");
        UpdateJobStatus(job_to_run.template_match_id, "failed");
        currently_running_id = -1;
    }

    return false;
}

bool TemplateMatchQueueManager::IsJobRunning( ) const {
    return currently_running_id != -1;
}

// IsJobRunningStatic removed - no longer needed with unified architecture

void TemplateMatchQueueManager::UpdateJobStatus(long template_match_id, const wxString& new_status) {
    MyDebugAssertTrue(template_match_id >= 0, "Invalid template_match_id in UpdateJobStatus: %ld", template_match_id);
    MyDebugAssertTrue(new_status == "pending" || new_status == "running" || new_status == "complete" || new_status == "failed",
                      "Invalid new_status in UpdateJobStatus: %s", new_status.mb_str( ).data( ));

    bool found_job = false;
    for ( auto& item : execution_queue ) {
        if ( item.template_match_id == template_match_id ) {
            // Validate status transitions
            MyDebugAssertTrue(item.queue_status != new_status, "Attempted to set status to same value: %s", new_status.mb_str( ).data( ));

            // Validate allowed transitions
            if ( item.queue_status == "running" && (new_status == "complete" || new_status == "failed") ) {
                // Valid: running -> complete/failed
                MyDebugAssertTrue(currently_running_id == template_match_id, "Status change from running but template_match_id %ld != currently_running_id %ld", template_match_id, currently_running_id);
            }
            else if ( item.queue_status == "pending" && new_status == "running" ) {
                // Valid: pending -> running
                MyDebugAssertFalse(IsJobRunning( ), "Cannot start job %ld when job %ld is already running", template_match_id, currently_running_id);
            }
            else {
                MyDebugAssertTrue(false, "Invalid status transition: %s -> %s for job %ld",
                                  item.queue_status.mb_str( ).data( ), new_status.mb_str( ).data( ), template_match_id);
            }

            item.queue_status = new_status;
            found_job         = true;
            break;
        }
    }

    MyDebugAssertTrue(found_job, "Job with template_match_id %ld not found in queue for status update", template_match_id);

    UpdateQueueDisplay( );
    SaveQueueToDatabase( );
}

// Static methods removed - no longer needed with unified architecture

TemplateMatchQueueItem* TemplateMatchQueueManager::GetNextPendingJob( ) {
    MyDebugAssertFalse(IsJobRunning( ), "GetNextPendingJob called while job %ld is running", currently_running_id);

    for ( auto& item : execution_queue ) {
        if ( item.queue_status == "pending" ) {
            // Validate the job we're about to return
            MyDebugAssertTrue(item.template_match_id >= 0, "Found pending job with invalid template_match_id: %ld", item.template_match_id);
            MyDebugAssertFalse(item.job_name.IsEmpty( ), "Found pending job with empty job_name");
            return &item;
        }
    }
    return nullptr;
}

bool TemplateMatchQueueManager::HasPendingJobs( ) {
    for ( const auto& item : execution_queue ) {
        if ( item.queue_status == "pending" ) {
            return true;
        }
    }
    return false;
}

void TemplateMatchQueueManager::OnRunSelectedClick(wxCommandEvent& event) {
    RunSelectedJob( );
}

// OnRunAllClick removed - use Run Selected with multi-selection instead

void TemplateMatchQueueManager::OnClearQueueClick(wxCommandEvent& event) {
    wxMessageDialog dialog(this, "Clear all pending jobs from the queue?",
                           "Confirm Clear", wxYES_NO | wxICON_QUESTION);
    if ( dialog.ShowModal( ) == wxID_YES ) {
        ClearQueue( );
    }
}

void TemplateMatchQueueManager::OnAssignPriorityClick(wxCommandEvent& event) {
    // Apply drag-and-drop changes and unfreeze UI
    assign_priority_button->Enable(false);

    // Enable all other buttons
    run_selected_button->Enable(true);
    add_to_queue_button->Enable(true);
    remove_from_queue_button->Enable(true);
    remove_selected_button->Enable(true);
    clear_queue_button->Enable(true);

    // Update database with new priorities
    SaveQueueToDatabase( );

    // Refresh display
    UpdateQueueDisplay( );

    wxPrintf("Priority assignment completed - UI unfrozen\n");
}

void TemplateMatchQueueManager::OnCancelRunClick(wxCommandEvent& event) {
    // Find and cancel the currently running job
    wxMessageDialog dialog(this, "Cancel the currently running job?",
                           "Confirm Cancel", wxYES_NO | wxICON_QUESTION);
    if ( dialog.ShowModal( ) == wxID_YES ) {
        // Find running job and reset its status
        for ( auto& job : execution_queue ) {
            if ( job.queue_status == "running" ) {
                job.queue_status     = "pending";
                job.queue_order      = -1; // Move to available jobs table
                currently_running_id = -1;

                // Update cancel button state
                cancel_run_button->Enable(false);

                UpdateQueueDisplay( );
                SaveQueueToDatabase( );
                wxPrintf("Job cancelled and moved to available jobs\n");
                break;
            }
        }
    }
}

void TemplateMatchQueueManager::OnRemoveSelectedClick(wxCommandEvent& event) {
    int selected = GetSelectedRow( );
    if ( selected >= 0 ) {
        wxString        job_name = execution_queue[selected].job_name;
        wxMessageDialog dialog(this,
                               wxString::Format("Remove job '%s' from the queue?", job_name),
                               "Confirm Remove", wxYES_NO | wxICON_QUESTION);
        if ( dialog.ShowModal( ) == wxID_YES ) {
            RemoveFromQueue(selected);
        }
    }
}

void TemplateMatchQueueManager::OnSelectionChanged(wxListEvent& event) {
    wxPrintf("OnSelectionChanged: ENTRY\n"); // revert - extreme debug

    // Clean up any stale drag state - selection changes often happen after cancelled drags
    CleanupDragState();
    wxPrintf("OnSelectionChanged: CleanupDragState completed\n"); // revert - extreme debug

    // Validate GUI components are available
    MyDebugAssertTrue(remove_selected_button != nullptr, "remove_selected_button is null in OnSelectionChanged");
    MyDebugAssertTrue(run_selected_button != nullptr, "run_selected_button is null in OnSelectionChanged");
    wxPrintf("OnSelectionChanged: GUI validation completed\n"); // revert - extreme debug

    // Get current selection state, but don't interfere with execution queue
    bool             has_selection = false;
    std::vector<int> current_ui_selection; // Temporary selection for button enabling

    wxPrintf("OnSelectionChanged: execution_in_progress = %s\n", execution_in_progress ? "true" : "false"); // revert - extreme debug

    if ( ! execution_in_progress ) {
        wxPrintf("OnSelectionChanged: Not executing - calling PopulateSelectionQueueFromUI\n"); // revert - extreme debug
        // Not executing - safe to update selection queue from UI
        PopulateSelectionQueueFromUI( );
        wxPrintf("OnSelectionChanged: PopulateSelectionQueueFromUI completed\n"); // revert - extreme debug

        has_selection = ! selected_jobs_for_execution.empty( );
        wxPrintf("OnSelectionChanged: has_selection = %s\n", has_selection ? "true" : "false"); // revert - extreme debug

        // Copy selection queue for button logic
        current_ui_selection.assign(selected_jobs_for_execution.begin( ), selected_jobs_for_execution.end( ));
        wxPrintf("OnSelectionChanged: Copied selection queue for button logic\n"); // revert - extreme debug
    }
    else {
        wxPrintf("OnSelectionChanged: Execution in progress - getting UI selection\n"); // revert - extreme debug
        // Execution in progress - get current UI selection without modifying execution queue

        // For wxListCtrl, iterate through selected items
        long item = queue_list_ctrl->GetNextItem(-1, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
        int selection_count = 0;
        while ( item != -1 ) {
            selection_count++;
            wxPrintf("OnSelectionChanged: Processing selected item at row %ld\n", item); // revert - extreme debug

            // Find job by queue_order (not array index) to fix drag-and-drop selection
            wxPrintf("OnSelectionChanged: Looking for job at row %ld\n", item); // revert - debug
            bool found = false;
            for ( const auto& job : execution_queue ) {
                if ( job.queue_order == item ) {
                    current_ui_selection.push_back(job.template_match_id);
                    wxPrintf("OnSelectionChanged: Found job %ld at queue_order %ld\n", job.template_match_id, item); // revert - debug
                    found = true;
                    break;
                }
            }
            if (!found) {
                wxPrintf("OnSelectionChanged: No job found with queue_order %ld\n", item); // revert - debug
            }

            item = queue_list_ctrl->GetNextItem(item, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
        }

        has_selection = (selection_count > 0);
        wxPrintf("OnSelectionChanged: Got %d selected items (execution path)\n", selection_count); // revert - extreme debug
        wxPrintf("OnSelectionChanged: Finished processing execution in progress path\n"); // revert - extreme debug
    }

    wxPrintf("OnSelectionChanged: Checking selection status for button enabling\n"); // revert - extreme debug

    // Check selection status for button enabling
    bool any_running          = false;
    bool all_pending          = true;
    int  first_selected_index = -1;
    int  last_selected_index  = -1;

    wxPrintf("OnSelectionChanged: has_selection = %s\n", has_selection ? "true" : "false"); // revert - extreme debug

    if ( has_selection ) {
        wxPrintf("OnSelectionChanged: Processing %zu items in current_ui_selection\n", current_ui_selection.size()); // revert - extreme debug
        // Find the queue indices for the currently selected template_match_ids
        for ( int template_match_id : current_ui_selection ) {
            wxPrintf("OnSelectionChanged: Looking for template_match_id %d in execution_queue\n", template_match_id); // revert - extreme debug
            for ( int i = 0; i < execution_queue.size( ); i++ ) {
                wxPrintf("OnSelectionChanged: Checking execution_queue[%d] with template_match_id %ld\n", i, execution_queue[i].template_match_id); // revert - extreme debug
                if ( execution_queue[i].template_match_id == template_match_id ) {
                    wxPrintf("OnSelectionChanged: Found match at index %d\n", i); // revert - extreme debug
                    if ( first_selected_index == -1 )
                        first_selected_index = i;
                    last_selected_index = i;

                    wxPrintf("OnSelectionChanged: Calling IsJobRunning(%d)\n", i); // revert - extreme debug
                    if ( IsJobRunning(i) )
                        any_running = true;
                    wxPrintf("OnSelectionChanged: Calling IsJobPending(%d)\n", i); // revert - extreme debug
                    if ( ! IsJobPending(i) )
                        all_pending = false;
                    break;
                }
            }
        }
        wxPrintf("OnSelectionChanged: Finished processing current_ui_selection\n"); // revert - extreme debug
    }

    wxPrintf("OnSelectionChanged: Enabling controls\n"); // revert - extreme debug

    // Enable controls based on selection and job status
    remove_selected_button->Enable(has_selection && ! any_running);
    run_selected_button->Enable(has_selection && all_pending);

    // Enable table movement buttons based on selection
    add_to_queue_button->Enable(false); // TODO: Enable when available jobs table is implemented
    remove_from_queue_button->Enable(has_selection && ! any_running);

    wxPrintf("OnSelectionChanged: Controls enabled, checking GUI population\n"); // revert - extreme debug

    // Populate the GUI with the first selected item's parameters
    if ( has_selection && first_selected_index >= 0 && first_selected_index < execution_queue.size( ) ) {
        wxPrintf("OnSelectionChanged: Populating GUI with first_selected_index = %d\n", first_selected_index); // revert - extreme debug
        MyDebugAssertTrue(match_template_panel_ptr != nullptr, "match_template_panel_ptr is null when trying to populate GUI");

        if ( match_template_panel_ptr ) {
            wxPrintf("OnSelectionChanged: About to call PopulateGuiFromQueueItem\n"); // revert - extreme debug
            match_template_panel_ptr->PopulateGuiFromQueueItem(execution_queue[first_selected_index]);
            wxPrintf("OnSelectionChanged: PopulateGuiFromQueueItem completed\n"); // revert - extreme debug
        }
    } else {
        wxPrintf("OnSelectionChanged: Skipping GUI population (has_selection=%s, first_selected_index=%d, queue_size=%zu)\n",
                 has_selection ? "true" : "false", first_selected_index, execution_queue.size()); // revert - extreme debug
    }

    wxPrintf("OnSelectionChanged: EXIT\n"); // revert - extreme debug
}


void TemplateMatchQueueManager::OnAddToQueueClick(wxCommandEvent& event) {
    // TODO: Move selected jobs from available jobs table to execution queue
    // For now, show placeholder message
    wxMessageBox("Move selected jobs from Available Jobs to Execution Queue\n\n"
                 "This will assign them queue positions and make them available to run.",
                 "Add to Execution Queue", wxOK | wxICON_INFORMATION);
}

void TemplateMatchQueueManager::OnRemoveFromQueueClick(wxCommandEvent& event) {
    // Move selected jobs from execution queue to available jobs (queue_order = -1)
    std::vector<long> selected_rows;
    long selected_item = execution_queue_ctrl->GetNextItem(-1, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
    while ( selected_item != -1 ) {
        selected_rows.push_back(selected_item);
        selected_item = execution_queue_ctrl->GetNextItem(selected_item, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
    }

    if ( selected_rows.empty() ) {
        wxMessageBox("Please select jobs to remove from the execution queue.",
                     "No Selection", wxOK | wxICON_WARNING);
        return;
    }

    wxMessageDialog dialog(this,
                           wxString::Format("Remove %zu selected job(s) from execution queue?\n\n"
                                            "They will be moved to Available Jobs.",
                                            selected_rows.size()),
                           "Confirm Remove from Queue", wxYES_NO | wxICON_QUESTION);

    if ( dialog.ShowModal( ) == wxID_YES ) {
        // Get template_match_ids for selected items
        std::vector<long> selected_ids;
        for ( long row : selected_rows ) {
            if ( row < int(execution_queue.size( )) ) {
                // Find the job with this queue position (since table is sorted by queue_order)
                for ( auto& job : execution_queue ) {
                    if ( job.queue_order == row ) { // Row position = queue_order due to sorting
                        selected_ids.push_back(job.template_match_id);
                        break;
                    }
                }
            }
        }

        // Move jobs to available table (queue_order = -1)
        for ( long job_id : selected_ids ) {
            for ( auto& job : execution_queue ) {
                if ( job.template_match_id == job_id ) {
                    job.queue_order = -1; // Move to available jobs
                    wxPrintf("Moved job %ld to available jobs\n", job_id);
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

        UpdateQueueDisplay( );
        SaveQueueToDatabase( );
        wxPrintf("Removed %zu jobs from execution queue\n", selected_ids.size( ));
    }
}

void TemplateMatchQueueManager::OnBeginDrag(wxListEvent& event) {
    // Note: This event handler exists for compatibility but manual drag-and-drop
    // is now handled through mouse events (OnMouseLeftDown, OnMouseMotion, OnMouseLeftUp)
    wxPrintf("OnBeginDrag called - using manual mouse drag implementation\n");
}



void TemplateMatchQueueManager::CleanupDragState() {
    if (drag_in_progress) {
        wxPrintf("CleanupDragState: Resetting drag state\n"); // revert - debug output for troubleshooting drag-and-drop
        drag_in_progress = false;
        mouse_down = false;
        dragged_row = -1;
        dragged_job_id = -1;
    }
}

void TemplateMatchQueueManager::LoadQueueFromDatabase( ) {
    MyDebugPrint("LoadQueueFromDatabase called. match_template_panel_ptr=%p", match_template_panel_ptr);
    if ( match_template_panel_ptr && main_frame && main_frame->current_project.is_open ) {
        MyDebugPrint("Loading queue from database...");
        execution_queue.clear( );

        // Get all queue IDs from database in order
        wxArrayLong queue_ids = main_frame->current_project.database.GetQueuedTemplateMatchIDs( );
        MyDebugPrint("Found %zu queue items in database", queue_ids.GetCount( ));

        // Load each queue item from database
        for ( size_t i = 0; i < queue_ids.GetCount( ); i++ ) {
            TemplateMatchQueueItem temp_item;
            temp_item.template_match_id = queue_ids[i];

            // Load item details from database
            bool success = main_frame->current_project.database.GetQueueItemByID(
                    queue_ids[i],
                    temp_item.job_name,
                    temp_item.queue_status,
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
                // Convert legacy 1-based queue_order to 0-based indexing
                if ( temp_item.queue_order > 0 ) {
                    temp_item.queue_order--; // Convert 1,2,3... to 0,1,2...
                    wxPrintf("Converted queue_order from %d to %d for job %ld\n", // revert - debug output for troubleshooting drag-and-drop
                             temp_item.queue_order + 1, temp_item.queue_order, temp_item.template_match_id);
                }
                execution_queue.push_back(temp_item);
            }
        }

        // Mark that we've loaded from database
        needs_database_load = false;

        // Update display
        UpdateQueueDisplay( );
    }
}

void TemplateMatchQueueManager::SaveQueueToDatabase( ) {
    if ( match_template_panel_ptr && main_frame && main_frame->current_project.is_open ) {
        // Update status and queue position for all items in queue
        for ( const auto& item : execution_queue ) {
            main_frame->current_project.database.UpdateQueueStatus(item.template_match_id, item.queue_status);
            // Also update queue position - we'll need to add this method to database
            main_frame->current_project.database.UpdateQueuePosition(item.template_match_id, item.queue_order);
        }
    }
}

// Manual drag and drop implementation for wxListCtrl
void TemplateMatchQueueManager::OnMouseLeftDown(wxMouseEvent& event) {
    wxPrintf("OnMouseLeftDown called\n"); // revert - debug

    // revert - Prevent drag operations during display updates
    if (updating_display) {
        wxPrintf("OnMouseLeftDown: Vetoing - display is updating\n"); // revert - debug
        event.Skip();
        return;
    }

    mouse_down = true;
    drag_start_pos = event.GetPosition();

    // Find which item was clicked
    int flags;
    long hit_item = queue_list_ctrl->HitTest(drag_start_pos, flags);
    if (hit_item != wxNOT_FOUND) {
        dragged_row = hit_item;
        wxPrintf("OnMouseLeftDown: Hit item %ld\n", hit_item); // revert - debug

        // Find the corresponding job
        for (const auto& job : execution_queue) {
            if (job.queue_order == hit_item) {
                dragged_job_id = job.template_match_id;
                wxPrintf("OnMouseLeftDown: Found job %ld at row %ld\n", dragged_job_id, hit_item); // revert - debug
                break;
            }
        }
    }

    event.Skip();
}

void TemplateMatchQueueManager::OnMouseMotion(wxMouseEvent& event) {
    if (!mouse_down || updating_display) {
        event.Skip();
        return;
    }

    // Check if we've moved far enough to start a drag
    wxPoint current_pos = event.GetPosition();
    int dx = current_pos.x - drag_start_pos.x;
    int dy = current_pos.y - drag_start_pos.y;

    if (abs(dx) > 5 || abs(dy) > 5) { // Start drag after moving 5 pixels
        if (!drag_in_progress && dragged_row != -1) {
            wxPrintf("Starting drag for job %ld from row %d\n", dragged_job_id, dragged_row); // revert - debug
            drag_in_progress = true;
            queue_list_ctrl->SetCursor(wxCursor(wxCURSOR_HAND));
        }
    }

    event.Skip();
}

void TemplateMatchQueueManager::OnMouseLeftUp(wxMouseEvent& event) {
    wxPrintf("OnMouseLeftUp called\n"); // revert - debug

    if (drag_in_progress) {
        // Find drop target
        wxPoint drop_pos = event.GetPosition();
        int flags;
        long drop_item = queue_list_ctrl->HitTest(drop_pos, flags);

        wxPrintf("Drop at position (%d,%d), hit item %ld\n", drop_pos.x, drop_pos.y, drop_item); // revert - debug

        if (drop_item != wxNOT_FOUND && drop_item != dragged_row) {
            // Perform the reorder
            ReorderQueueItems(dragged_row, drop_item);
        }
    }

    // Reset drag state
    mouse_down = false;
    drag_in_progress = false;
    dragged_row = -1;
    dragged_job_id = -1;
    queue_list_ctrl->SetCursor(wxCursor(wxCURSOR_ARROW));

    event.Skip();
}

void TemplateMatchQueueManager::ReorderQueueItems(int old_position, int new_position) {
    wxPrintf("Reordering: old_position=%d, new_position=%d\n", old_position, new_position); // revert - debug

    // Find the job being moved
    TemplateMatchQueueItem* moved_job = nullptr;
    for (auto& job : execution_queue) {
        if (job.queue_order == old_position) {
            moved_job = &job;
            break;
        }
    }

    if (!moved_job) {
        wxPrintf("ERROR: Could not find job at position %d\n", old_position);
        return;
    }

    // Adjust queue orders
    if (old_position < new_position) {
        // Moving down: shift items up
        for (auto& job : execution_queue) {
            if (job.queue_order > old_position && job.queue_order <= new_position) {
                job.queue_order--;
            }
        }
    } else {
        // Moving up: shift items down
        for (auto& job : execution_queue) {
            if (job.queue_order >= new_position && job.queue_order < old_position) {
                job.queue_order++;
            }
        }
    }

    // Set new position for moved job
    moved_job->queue_order = new_position;

    wxPrintf("Moved job %ld from position %d to %d\n", moved_job->template_match_id, old_position, new_position); // revert - debug

    // Update display
    UpdateQueueDisplay();
}

void TemplateMatchQueueManager::ValidateQueueConsistency( ) const {
    int  running_jobs_count = 0;
    long found_running_id   = -1;

    for ( size_t i = 0; i < execution_queue.size( ); ++i ) {
        const auto& item = execution_queue[i];

        // Validate basic item consistency
        MyDebugAssertTrue(item.template_match_id >= 0, "Queue item %zu has invalid template_match_id: %ld", i, item.template_match_id);
        MyDebugAssertFalse(item.job_name.IsEmpty( ), "Queue item %zu (ID: %ld) has empty job_name", i, item.template_match_id);
        MyDebugAssertTrue(item.queue_status == "pending" || item.queue_status == "running" ||
                                  item.queue_status == "complete" || item.queue_status == "failed",
                          "Queue item %zu (ID: %ld) has invalid status: %s", i, item.template_match_id, item.queue_status.mb_str( ).data( ));

        // Track running jobs
        if ( item.queue_status == "running" ) {
            running_jobs_count++;
            found_running_id = item.template_match_id;
        }

        // Check for duplicate IDs
        for ( size_t j = i + 1; j < execution_queue.size( ); ++j ) {
            MyDebugAssertTrue(execution_queue[j].template_match_id != item.template_match_id,
                              "Duplicate template_match_id %ld found at indices %zu and %zu", item.template_match_id, i, j);
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

    if ( IsJobRunning( ) ) {
        // A job is already running, don't start another
        MyDebugPrint("Job is still running, not continuing queue execution");
        return;
    }

    // Continue execution using this instance
    MyDebugPrint("Continuing queue execution with next job");
    RunNextJob( );
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
    UpdateQueueDisplay( );

    // Progress to next job in queue order system
    ProgressQueue( );
}

void TemplateMatchQueueManager::PopulateSelectionQueueFromUI( ) {
    wxPrintf("PopulateSelectionQueueFromUI: ENTRY\n"); // revert - extreme debug

    // Clear existing selection queue
    selected_jobs_for_execution.clear( );
    wxPrintf("PopulateSelectionQueueFromUI: Cleared selection queue\n"); // revert - extreme debug

    // Get all selected items from wxListCtrl
    long item = queue_list_ctrl->GetNextItem(-1, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
    int selection_count = 0;

    while ( item != -1 ) {
        selection_count++;
        wxPrintf("PopulateSelectionQueueFromUI: Processing selected item at row %ld\n", item); // revert - extreme debug

        // Find job by queue_order (not array index) to fix drag-and-drop selection
        bool found = false;
        for ( const auto& job : execution_queue ) {
            if ( job.queue_order == item ) {
                selected_jobs_for_execution.push_back(job.template_match_id);
                wxPrintf("  Added job %ld to selection queue\n", job.template_match_id);
                found = true;
                break;
            }
        }
        if (!found) {
            wxPrintf("  WARNING: No job found with queue_order %ld\n", item);
        }

        item = queue_list_ctrl->GetNextItem(item, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
    }

    wxPrintf("Populating selection queue from %d selected UI items\n", selection_count);
    wxPrintf("Selection queue populated with %zu jobs\n", selected_jobs_for_execution.size( ));
    wxPrintf("PopulateSelectionQueueFromUI: EXIT\n"); // revert - extreme debug
}

void TemplateMatchQueueManager::RemoveJobFromSelectionQueue(long template_match_id) {
    auto it = std::find(selected_jobs_for_execution.begin( ), selected_jobs_for_execution.end( ), template_match_id);
    if ( it != selected_jobs_for_execution.end( ) ) {
        selected_jobs_for_execution.erase(it);
        wxPrintf("Removed job %ld from selection queue (%zu remaining)\n", template_match_id, selected_jobs_for_execution.size( ));
    }
}

bool TemplateMatchQueueManager::HasJobsInSelectionQueue( ) const {
    return ! selected_jobs_for_execution.empty( );
}

void TemplateMatchQueueManager::RunNextSelectedJob( ) {
    if ( selected_jobs_for_execution.empty( ) ) {
        wxPrintf("No jobs in selection queue - execution complete\n");
        return;
    }

    if ( IsJobRunning( ) ) {
        wxPrintf("Job already running - cannot start next selected job\n");
        return;
    }

    // Get next job ID from selection queue
    long next_job_id = selected_jobs_for_execution.front( );

    // Find the job in the main queue
    for ( auto& job : execution_queue ) {
        if ( job.template_match_id == next_job_id ) {
            if ( job.queue_status == "pending" ) {
                wxPrintf("Starting next selected job %ld\n", next_job_id);

                // Remove from selection queue (job is starting)
                selected_jobs_for_execution.pop_front( );

                // Deselect the job in the UI since it's now running
                DeselectJobInUI(next_job_id);

                // Execute the job
                ExecuteJob(job);
                return;
            }
            else {
                wxPrintf("Skipping job %ld - status is '%s', not 'pending'\n",
                         next_job_id, job.queue_status.mb_str( ).data( ));
                // Remove from selection queue and try next
                selected_jobs_for_execution.pop_front( );
                RunNextSelectedJob( ); // Recursive call to try next job
                return;
            }
        }
    }

    // Job not found - remove from selection queue and try next
    wxPrintf("Job %ld not found in queue - removing from selection\n", next_job_id);
    selected_jobs_for_execution.pop_front( );
    RunNextSelectedJob( ); // Recursive call to try next job
}

void TemplateMatchQueueManager::DeselectJobInUI(long template_match_id) {
    // Find the row corresponding to this job ID
    for ( size_t i = 0; i < execution_queue.size( ); ++i ) {
        if ( execution_queue[i].template_match_id == template_match_id ) {
            queue_list_ctrl->SetItemState(int(i), 0, wxLIST_STATE_SELECTED);
            wxPrintf("Deselected job %ld from UI (row %zu)\n", template_match_id, i);
            return;
        }
    }
    wxPrintf("Could not find job %ld to deselect in UI\n", template_match_id);
}