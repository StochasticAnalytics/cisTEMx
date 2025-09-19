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
    auto_progress_queue   = false; // Don't auto-progress by default (user controls progression)
    hide_completed_jobs   = false; // Show all jobs by default
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
    execution_queue_ctrl->AppendColumn("Progress", wxLIST_FORMAT_LEFT, 80);
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
    run_selected_button = new wxButton(execution_controls, wxID_ANY, "Run Queue");
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

    add_to_queue_button      = new wxButton(movement_controls, wxID_ANY, "Add to Execution Queue");
    remove_from_queue_button = new wxButton(movement_controls, wxID_ANY, "Remove from Execution Queue");

    movement_sizer->Add(add_to_queue_button, 0, wxALL, 5);
    movement_sizer->Add(remove_from_queue_button, 0, wxALL, 5);

    movement_controls->SetSizer(movement_sizer);

    // Create available jobs section (bottom)
    wxStaticText* available_jobs_label = new wxStaticText(this, wxID_ANY, "Available Jobs (not queued for execution):");
    available_jobs_ctrl                = new wxListCtrl(this, wxID_ANY,
                                                        wxDefaultPosition, wxSize(700, 150),
                                                        wxLC_REPORT | wxLC_SINGLE_SEL);

    // Set minimum size to ensure visibility
    available_jobs_ctrl->SetMinSize(wxSize(700, 150));
    wxPrintf("Created available_jobs_ctrl: %p with min size 700x150\n", available_jobs_ctrl); // Debug output

    // Add columns to available jobs (no queue order column)
    available_jobs_ctrl->AppendColumn("ID", wxLIST_FORMAT_LEFT, 60);
    available_jobs_ctrl->AppendColumn("Job Name", wxLIST_FORMAT_LEFT, 200);
    available_jobs_ctrl->AppendColumn("Status", wxLIST_FORMAT_LEFT, 100);
    available_jobs_ctrl->AppendColumn("Progress", wxLIST_FORMAT_LEFT, 80);
    available_jobs_ctrl->AppendColumn("CLI Args", wxLIST_FORMAT_LEFT, 140);

    // Create general controls
    wxPanel*    general_controls = new wxPanel(this, wxID_ANY);
    wxBoxSizer* general_sizer    = new wxBoxSizer(wxHORIZONTAL);

    remove_selected_button = new wxButton(general_controls, wxID_ANY, "Remove Selected");
    clear_queue_button     = new wxButton(general_controls, wxID_ANY, "Clear All");
    hide_completed_checkbox = new wxCheckBox(general_controls, wxID_ANY, "Hide completed jobs");

    general_sizer->Add(remove_selected_button, 0, wxALL, 5);
    general_sizer->Add(clear_queue_button, 0, wxALL, 5);
    general_sizer->AddSpacer(20);
    general_sizer->Add(hide_completed_checkbox, 0, wxALL | wxALIGN_CENTER_VERTICAL, 5);

    general_controls->SetSizer(general_sizer);

    // Add all sections to main sizer with adjusted proportions for better visibility
    main_sizer->Add(execution_queue_label, 0, wxEXPAND | wxALL, 5);
    main_sizer->Add(execution_queue_ctrl, 2, wxEXPAND | wxALL, 5); // Slightly larger proportion
    main_sizer->Add(execution_controls, 0, wxEXPAND | wxALL, 5);
    main_sizer->Add(movement_controls, 0, wxEXPAND | wxALL, 5);
    main_sizer->Add(available_jobs_label, 0, wxEXPAND | wxALL, 5);
    main_sizer->Add(available_jobs_ctrl, 1, wxEXPAND | wxALL, 5); // Smaller but still expandable
    main_sizer->Add(general_controls, 0, wxEXPAND | wxALL, 5);

    SetSizer(main_sizer);

    // Force layout update - important for dialog visibility
    Layout();
    wxPrintf("TemplateMatchQueueManager: Layout() called\n");

    // Connect events
    assign_priority_button->Bind(wxEVT_BUTTON, &TemplateMatchQueueManager::OnAssignPriorityClick, this);
    cancel_run_button->Bind(wxEVT_BUTTON, &TemplateMatchQueueManager::OnCancelRunClick, this);
    run_selected_button->Bind(wxEVT_BUTTON, &TemplateMatchQueueManager::OnRunSelectedClick, this);
    add_to_queue_button->Bind(wxEVT_BUTTON, &TemplateMatchQueueManager::OnAddToQueueClick, this);
    remove_from_queue_button->Bind(wxEVT_BUTTON, &TemplateMatchQueueManager::OnRemoveFromQueueClick, this);
    remove_selected_button->Bind(wxEVT_BUTTON, &TemplateMatchQueueManager::OnRemoveSelectedClick, this);
    clear_queue_button->Bind(wxEVT_BUTTON, &TemplateMatchQueueManager::OnClearQueueClick, this);
    hide_completed_checkbox->Bind(wxEVT_CHECKBOX, &TemplateMatchQueueManager::OnHideCompletedToggle, this);

    // Bind selection events for available jobs table
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
    // New logic: find the job at priority 0 and run it - allow pending or failed jobs
    TemplateMatchQueueItem* next_job = nullptr;
    for ( auto& item : execution_queue ) {
        if ( item.queue_order == 0 && (item.queue_status == "pending" || item.queue_status == "failed") ) {
            next_job = &item;
            break;
        }
    }

    if ( next_job ) {
        // Don't change status yet - let MatchTemplatePanel validate first
        // Move to priority -1 (out of execution queue) but keep original status
        int original_queue_order = next_job->queue_order;
        next_job->queue_order = -1;

        // Shift all other execution queue jobs up by one priority (1→0, 2→1, 3→2, etc.)
        for ( auto& item : execution_queue ) {
            if ( item.queue_order > 0 ) {
                item.queue_order--;
            }
        }

        wxPrintf("ProgressQueue: Starting job %ld (moved to priority -1, status: %s)\n",
                 next_job->template_match_id, next_job->queue_status);

        // Execute the job - status will be changed to "running" by MatchTemplatePanel after validation
        if (ExecuteJob(*next_job)) {
            wxPrintf("ProgressQueue: Job %ld execution started successfully\n", next_job->template_match_id);
        } else {
            // If execution failed, mark as failed and ensure it stays in available queue
            next_job->queue_status = "failed";
            // next_job->queue_order is already -1
            wxPrintf("ProgressQueue: Job %ld execution failed\n", next_job->template_match_id);
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
    MyDebugAssertTrue(queue_list_ctrl != nullptr, "queue_list_ctrl is null in UpdateQueueDisplay");

    // Prevent drag operations during display update to avoid GTK crashes
    updating_display = true;

    // Preserve current selections before rebuilding
    std::vector<long> selected_template_ids;

    // Get currently selected items in wxListCtrl
    long item = queue_list_ctrl->GetNextItem(-1, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
    while ( item != -1 ) {
        // For wxListCtrl, the item index directly corresponds to the row in our sorted display
        if ( item < int(execution_queue.size()) ) {
            // Find the job with this queue_order (since we sort by queue_order)
            for ( const auto& job : execution_queue ) {
                if ( job.queue_order == item ) {
                    selected_template_ids.push_back(job.template_match_id);
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
        wxPrintf("UpdateQueueDisplay: Job %ld has queue_order=%d\n",
                 execution_queue[i].template_match_id, execution_queue[i].queue_order);
        if ( execution_queue[i].queue_order >= 0 ) { // Only show items in execution queue
            sorted_indices.push_back(i);
            wxPrintf("UpdateQueueDisplay: Adding job %ld to execution display\n", execution_queue[i].template_match_id);
        } else {
            wxPrintf("UpdateQueueDisplay: Skipping job %ld (available queue)\n", execution_queue[i].template_match_id);
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
        wxColour text_color;
        if ( execution_queue[i].queue_status == "running" ) {
            status_display = "● " + execution_queue[i].queue_status; // Red circle for running
            text_color = wxColour(255, 0, 0); // Red
        }
        else if ( execution_queue[i].queue_status == "complete" ) {
            status_display = "✓ " + execution_queue[i].queue_status; // Check mark for complete
            text_color = wxColour(0, 128, 0); // Green
        }
        else if ( execution_queue[i].queue_status == "failed" ) {
            status_display = "✗ " + execution_queue[i].queue_status; // X mark for failed
            text_color = wxColour(255, 0, 0); // Red
        }
        else { // pending
            status_display = "○ " + execution_queue[i].queue_status; // Empty circle for pending
            text_color = wxColour(0, 0, 255); // Blue
        }
        queue_list_ctrl->SetItem(item_index, 3, status_display);

        // Set text color for the entire row
        queue_list_ctrl->SetItemTextColour(item_index, text_color);

        // Get completion info for Progress column
        JobCompletionInfo completion = GetJobCompletionInfo(execution_queue[i].template_match_id);
        queue_list_ctrl->SetItem(item_index, 4, completion.GetCompletionString());

        queue_list_ctrl->SetItem(item_index, 5, execution_queue[i].custom_cli_args);
    }

    // Restore selections after rebuilding
    for ( long template_id : selected_template_ids ) {
        for ( size_t i = 0; i < execution_queue.size( ); ++i ) {
            if ( execution_queue[i].template_match_id == template_id ) {
                wxPrintf("RestoreSelection: Found job %ld at execution_queue index %zu, queue_order=%d\n",
                         template_id, i, execution_queue[i].queue_order);

                // The row should be based on queue_order, not array index!
                int row = execution_queue[i].queue_order;
                wxPrintf("RestoreSelection: Using row %d (queue_order) instead of %zu (array index)\n", row, i);

                // Add safety checks to prevent array bounds errors
                if (row < 0 || row >= queue_list_ctrl->GetItemCount()) {
                    wxPrintf("ERROR: Row %d is out of bounds (list item count: %ld)\n", row, queue_list_ctrl->GetItemCount());
                    break;
                }

                // For wxListCtrl, we can directly select by row index
                queue_list_ctrl->SetItemState(row, wxLIST_STATE_SELECTED, wxLIST_STATE_SELECTED);
                break;
            }
        }
    }

    // Update the available jobs table with items not in execution queue
    UpdateAvailableJobsDisplay();

    // Re-enable drag operations after display update is complete
    updating_display = false;
}

void TemplateMatchQueueManager::UpdateAvailableJobsDisplay() {
    wxPrintf("UpdateAvailableJobsDisplay called\n");
    MyDebugAssertTrue(available_jobs_ctrl != nullptr, "available_jobs_ctrl is null in UpdateAvailableJobsDisplay");

    // Clear existing items
    available_jobs_ctrl->DeleteAllItems();

    // Add jobs with queue_order < 0 (not in execution queue) from both execution_queue and available_queue
    int display_row = 0;

    // First add jobs from execution_queue with queue_order < 0
    for (size_t i = 0; i < execution_queue.size(); ++i) {
        if (execution_queue[i].queue_order < 0) {
            // Skip completed jobs if hide_completed_jobs is enabled
            if (hide_completed_jobs && execution_queue[i].queue_status == "complete") {
                continue;
            }

            // For wxListCtrl, add item with first column text then set other columns
            long item_index = available_jobs_ctrl->InsertItem(display_row,
                                                            wxString::Format("%ld", execution_queue[i].template_match_id));

            // Set the remaining columns (no queue order column for available jobs)
            available_jobs_ctrl->SetItem(item_index, 1, execution_queue[i].job_name);

            // Add status with a colored indicator prefix
            wxString status_display;
            wxColour text_color;
            if (execution_queue[i].queue_status == "running") {
                status_display = "● " + execution_queue[i].queue_status; // Red circle for running
                text_color = wxColour(255, 0, 0); // Red
            }
            else if (execution_queue[i].queue_status == "complete") {
                status_display = "✓ " + execution_queue[i].queue_status; // Check mark for complete
                text_color = wxColour(0, 128, 0); // Green
            }
            else if (execution_queue[i].queue_status == "failed") {
                status_display = "✗ " + execution_queue[i].queue_status; // X mark for failed
                text_color = wxColour(255, 0, 0); // Red
            }
            else { // pending
                status_display = "○ " + execution_queue[i].queue_status; // Empty circle for pending
                text_color = wxColour(0, 0, 255); // Blue
            }
            available_jobs_ctrl->SetItem(item_index, 2, status_display);

            // Set text color for the status column
            available_jobs_ctrl->SetItemTextColour(item_index, text_color);

            // Get completion info for Progress column
            JobCompletionInfo completion = GetJobCompletionInfo(execution_queue[i].template_match_id);
            available_jobs_ctrl->SetItem(item_index, 3, completion.GetCompletionString());

            available_jobs_ctrl->SetItem(item_index, 4, execution_queue[i].custom_cli_args);

            wxPrintf("Added available job %ld: %s with status %s\n", execution_queue[i].template_match_id,
                    execution_queue[i].job_name.mb_str().data(), execution_queue[i].queue_status.mb_str().data());

            display_row++;
        }
    }

    // Then add jobs from available_queue (these are newly discovered jobs not yet in execution_queue)
    for (size_t i = 0; i < available_queue.size(); ++i) {
        // Skip completed jobs if hide_completed_jobs is enabled
        if (hide_completed_jobs && available_queue[i].queue_status == "complete") {
            continue;
        }

        // Check if this job is already in execution_queue to avoid duplicates
        bool found_in_execution = false;
        for (const auto& exec_job : execution_queue) {
            if (exec_job.template_match_id == available_queue[i].template_match_id) {
                found_in_execution = true;
                break;
            }
        }

        if (!found_in_execution) {
            // For wxListCtrl, add item with first column text then set other columns
            long item_index = available_jobs_ctrl->InsertItem(display_row,
                                                            wxString::Format("%ld", available_queue[i].template_match_id));

            // Set the remaining columns (no queue order column for available jobs)
            available_jobs_ctrl->SetItem(item_index, 1, available_queue[i].job_name);

            // Add status with a colored indicator prefix
            wxString status_display;
            wxColour text_color;
            if (available_queue[i].queue_status == "running") {
                status_display = "● " + available_queue[i].queue_status; // Red circle for running
                text_color = wxColour(255, 0, 0); // Red
            }
            else if (available_queue[i].queue_status == "complete") {
                status_display = "✓ " + available_queue[i].queue_status; // Check mark for complete
                text_color = wxColour(0, 128, 0); // Green
            }
            else if (available_queue[i].queue_status == "failed") {
                status_display = "✗ " + available_queue[i].queue_status; // X mark for failed
                text_color = wxColour(255, 0, 0); // Red
            }
            else { // pending
                status_display = "○ " + available_queue[i].queue_status; // Empty circle for pending
                text_color = wxColour(0, 0, 255); // Blue
            }
            available_jobs_ctrl->SetItem(item_index, 2, status_display);

            // Set text color for the status column
            available_jobs_ctrl->SetItemTextColour(item_index, text_color);

            // Get completion info for Progress column
            JobCompletionInfo completion = GetJobCompletionInfo(available_queue[i].template_match_id);
            available_jobs_ctrl->SetItem(item_index, 3, completion.GetCompletionString());

            available_jobs_ctrl->SetItem(item_index, 4, available_queue[i].custom_cli_args);

            wxPrintf("Added available job %ld: %s with status %s (from available_queue)\n", available_queue[i].template_match_id,
                    available_queue[i].job_name.mb_str().data(), available_queue[i].queue_status.mb_str().data());

            display_row++;
        }
    }

}

int TemplateMatchQueueManager::GetSelectedRow( ) {
    long selected_item = queue_list_ctrl->GetNextItem(-1, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
    if ( selected_item != -1 ) {
        return selected_item;
    }
    return -1;
}

void TemplateMatchQueueManager::RunSelectedJob( ) {
    // New "Run Queue" logic - always run the job at priority 0 (top of queue)

    if ( IsJobRunning( ) ) {
        wxMessageBox("A job is already running. Please wait for it to complete.", "Job Running", wxOK | wxICON_WARNING);
        return;
    }

    // Find the job at priority 0 (top of execution queue) - allow pending or failed jobs
    TemplateMatchQueueItem* top_job = nullptr;
    for ( auto& item : execution_queue ) {
        if ( item.queue_order == 0 && (item.queue_status == "pending" || item.queue_status == "failed") ) {
            top_job = &item;
            break;
        }
    }

    if ( !top_job ) {
        wxMessageBox("No pending jobs in execution queue at priority 0.", "No Jobs to Run", wxOK | wxICON_WARNING);
        return;
    }

    // Enable auto-progression for queue mode - each job completion will trigger the next
    SetAutoProgressQueue(true);

    // Don't change status yet - let MatchTemplatePanel validate first
    // Move to priority -1 (out of execution queue) but keep original status
    top_job->queue_order = -1;  // Move running job out of execution queue for crash safety

    // Shift all other execution queue jobs up by one priority (1→0, 2→1, 3→2, etc.)
    for ( auto& item : execution_queue ) {
        if ( item.queue_order > 0 ) {
            item.queue_order--;
        }
    }

    // Execute the job - status will be changed to "running" by MatchTemplatePanel after validation
    if (ExecuteJob(*top_job)) {
        wxPrintf("Run Queue: Started job %ld (moved to priority -1, status: %s)\n",
                 top_job->template_match_id, top_job->queue_status);
    } else {
        // If execution failed, mark as failed and restore to available queue
        top_job->queue_status = "failed";
        // queue_order already set to -1, which is correct for available queue
        wxPrintf("Run Queue: Job %ld execution failed\n", top_job->template_match_id);
    }

    // Save queue state and update display
    SaveQueueToDatabase();
    UpdateQueueDisplay();
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
            else if ( item.queue_status == "pending" && new_status == "failed" ) {
                // Valid: pending -> failed (when job fails to start)
                wxPrintf("Job %ld failed to start, marking as failed\n", template_match_id);
            }
            else if ( item.queue_status == "failed" && new_status == "running" ) {
                // Valid: failed -> running (when resuming a failed job)
                MyDebugAssertFalse(IsJobRunning( ), "Cannot resume failed job %ld when job %ld is already running", template_match_id, currently_running_id);
                wxPrintf("Resuming failed job %ld, marking as running\n", template_match_id);
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

void TemplateMatchQueueManager::OnHideCompletedToggle(wxCommandEvent& event) {
    hide_completed_jobs = event.IsChecked();
    wxPrintf("Hide completed jobs toggled to: %s\n", hide_completed_jobs ? "true" : "false");

    // Refresh the available jobs display with the new filter
    UpdateAvailableJobsDisplay();
}

void TemplateMatchQueueManager::OnAvailableJobsSelectionChanged(wxListEvent& event) {
    wxPrintf("OnAvailableJobsSelectionChanged called\n");

    // Check if any items are selected in available jobs table
    bool has_available_selection = false;
    long selected_item = available_jobs_ctrl->GetNextItem(-1, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
    if (selected_item != -1) {
        has_available_selection = true;
    }

    // Enable Add to Queue button based on available jobs selection
    add_to_queue_button->Enable(has_available_selection);

    wxPrintf("Available jobs selection changed - Add to Queue button %s\n",
             has_available_selection ? "enabled" : "disabled");
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

    // Clean up any stale drag state - selection changes often happen after cancelled drags
    CleanupDragState();

    // Validate GUI components are available
    MyDebugAssertTrue(remove_selected_button != nullptr, "remove_selected_button is null in OnSelectionChanged");
    MyDebugAssertTrue(run_selected_button != nullptr, "run_selected_button is null in OnSelectionChanged");

    // Get current selection state, but don't interfere with execution queue
    bool             has_selection = false;
    std::vector<int> current_ui_selection; // Temporary selection for button enabling


    if ( ! execution_in_progress ) {
        // Not executing - safe to update selection queue from UI
        PopulateSelectionQueueFromUI( );

        has_selection = ! selected_jobs_for_execution.empty( );

        // Copy selection queue for button logic
        current_ui_selection.assign(selected_jobs_for_execution.begin( ), selected_jobs_for_execution.end( ));
    }
    else {
        // Execution in progress - get current UI selection without modifying execution queue

        // For wxListCtrl, iterate through selected items
        long item = queue_list_ctrl->GetNextItem(-1, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
        int selection_count = 0;
        while ( item != -1 ) {
            selection_count++;

            // Find job by queue_order (not array index) to fix drag-and-drop selection
            bool found = false;
            for ( const auto& job : execution_queue ) {
                if ( job.queue_order == item ) {
                    current_ui_selection.push_back(job.template_match_id);
                    found = true;
                    break;
                }
            }
            if (!found) {
            }

            item = queue_list_ctrl->GetNextItem(item, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
        }

        has_selection = (selection_count > 0);
    }


    // Check selection status for button enabling
    bool any_running          = false;
    bool all_pending          = true;
    int  first_selected_index = -1;
    int  last_selected_index  = -1;


    if ( has_selection ) {
        // Find the queue indices for the currently selected template_match_ids
        for ( int template_match_id : current_ui_selection ) {
            for ( int i = 0; i < execution_queue.size( ); i++ ) {
                if ( execution_queue[i].template_match_id == template_match_id ) {
                    if ( first_selected_index == -1 )
                        first_selected_index = i;
                    last_selected_index = i;

                    if ( IsJobRunning(i) )
                        any_running = true;
                    if ( ! IsJobPending(i) )
                        all_pending = false;
                    break;
                }
            }
        }
    }


    // Check if there's a job at priority 0 ready to run (pending or failed)
    bool has_job_at_priority_0 = false;
    for (const auto& job : execution_queue) {
        if (job.queue_order == 0 && (job.queue_status == "pending" || job.queue_status == "failed")) {
            has_job_at_priority_0 = true;
            break;
        }
    }

    // Enable controls based on selection and job status
    remove_selected_button->Enable(has_selection && ! any_running);
    run_selected_button->Enable(has_job_at_priority_0 && !any_running);

    // Enable table movement buttons based on selection
    // Note: add_to_queue_button is controlled by OnAvailableJobsSelectionChanged
    remove_from_queue_button->Enable(has_selection && ! any_running);


    // Populate the GUI with the first selected item's parameters
    if ( has_selection && first_selected_index >= 0 && first_selected_index < execution_queue.size( ) ) {
        MyDebugAssertTrue(match_template_panel_ptr != nullptr, "match_template_panel_ptr is null when trying to populate GUI");

        if ( match_template_panel_ptr ) {
            match_template_panel_ptr->PopulateGuiFromQueueItem(execution_queue[first_selected_index]);
        }
    } else {
        wxPrintf("OnSelectionChanged: Skipping GUI population (has_selection=%s, first_selected_index=%d, queue_size=%zu)\n",
                 has_selection ? "true" : "false", first_selected_index, execution_queue.size());
    }

}


void TemplateMatchQueueManager::OnAddToQueueClick(wxCommandEvent& event) {
    // Get selected jobs from available jobs table
    std::vector<long> selected_rows;
    long selected_item = available_jobs_ctrl->GetNextItem(-1, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
    while (selected_item != -1) {
        selected_rows.push_back(selected_item);
        selected_item = available_jobs_ctrl->GetNextItem(selected_item, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
    }

    if (selected_rows.empty()) {
        wxMessageBox("Please select jobs from the Available Jobs table to add to the execution queue.",
                     "No Selection", wxOK | wxICON_WARNING);
        return;
    }

    // Count jobs in execution queue to determine next position
    int next_queue_order = 0;
    for (const auto& job : execution_queue) {
        if (job.queue_order >= 0) { // Count jobs in execution queue
            next_queue_order++;
        }
    }

    // Move selected available jobs to execution queue (only pending jobs)
    std::vector<long> selected_job_ids;
    std::vector<wxString> blocked_jobs; // Track jobs that can't be moved
    int available_row = 0;
    for (size_t i = 0; i < execution_queue.size(); ++i) {
        if (execution_queue[i].queue_order < 0) { // This is an available job
            // Check if this available job row is selected
            for (long selected_row : selected_rows) {
                if (available_row == selected_row) {
                    // Allow pending and failed jobs to be moved to execution queue
                    if (execution_queue[i].queue_status == "pending" || execution_queue[i].queue_status == "failed") {
                        execution_queue[i].queue_order = next_queue_order++;
                        selected_job_ids.push_back(execution_queue[i].template_match_id);
                    } else {
                        blocked_jobs.push_back(wxString::Format("Job %ld (%s)",
                                              execution_queue[i].template_match_id,
                                              execution_queue[i].queue_status));
                    }
                    break;
                }
            }
            available_row++;
        }
    }

    // Show warning if some jobs were blocked
    if (!blocked_jobs.empty()) {
        wxString message = "The following jobs cannot be added to execution queue because they are not pending:\n\n";
        for (const auto& job : blocked_jobs) {
            message += "• " + job + "\n";
        }
        message += "\nOnly jobs with 'pending' status can be executed.";
        wxMessageBox(message, "Some Jobs Not Added", wxOK | wxICON_INFORMATION);
    }

    if (!selected_job_ids.empty()) {
        SaveQueueToDatabase();
        UpdateQueueDisplay();
    }
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

        // Load each queue item from database and separate into execution vs available
        std::vector<TemplateMatchQueueItem> execution_items;
        std::vector<TemplateMatchQueueItem> available_items;

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
                // Force completed/failed jobs to available queue regardless of stored queue_order
                if ( temp_item.queue_status == "complete" || temp_item.queue_status == "failed" ) {
                    temp_item.queue_order = -1; // Move completed/failed jobs to available queue
                    available_items.push_back(temp_item);
                } else if ( temp_item.queue_order == -1 ) {
                    // Available job
                    available_items.push_back(temp_item);
                } else if ( temp_item.queue_order > 0 ) {
                    // Execution queue job - convert from 1-based to 0-based for sorting
                    temp_item.queue_order--; // Convert 1,2,3... to 0,1,2...
                    execution_items.push_back(temp_item);
                } else {
                    // Invalid queue_order (0) - treat as available job
                    temp_item.queue_order = -1;
                    available_items.push_back(temp_item);
                }
            }
        }

        // Sort execution items by their stored queue_order for proper sequencing
        std::sort(execution_items.begin(), execution_items.end(),
                  [](const TemplateMatchQueueItem& a, const TemplateMatchQueueItem& b) {
                      return a.queue_order < b.queue_order;
                  });

        // Reassign consecutive 0-based queue positions to execution items
        for ( size_t i = 0; i < execution_items.size(); i++ ) {
            execution_items[i].queue_order = int(i);
        }

        // Crash recovery: Reset any orphaned "running" jobs to "failed"
        // On project load, no jobs should be running since there's no detach mechanism
        bool found_orphaned_jobs = false;
        for ( auto& item : execution_items ) {
            if ( item.queue_status == "running" ) {
                wxPrintf("CRASH RECOVERY: Found orphaned running job %ld, marking as failed\n", item.template_match_id);
                item.queue_status = "failed";
                item.queue_order = -1; // Move to available queue
                available_items.push_back(item);
                found_orphaned_jobs = true;

                // Update database immediately for this job
                if ( main_frame && main_frame->current_project.is_open ) {
                    main_frame->current_project.database.UpdateQueueStatus(item.template_match_id, "failed");
                    main_frame->current_project.database.UpdateQueuePosition(item.template_match_id, -1);
                }
            }
        }
        for ( auto& item : available_items ) {
            if ( item.queue_status == "running" ) {
                wxPrintf("CRASH RECOVERY: Found orphaned running job %ld, marking as failed\n", item.template_match_id);
                item.queue_status = "failed";
                found_orphaned_jobs = true;

                // Update database immediately for this job
                if ( main_frame && main_frame->current_project.is_open ) {
                    main_frame->current_project.database.UpdateQueueStatus(item.template_match_id, "failed");
                }
            }
        }

        // Remove orphaned jobs from execution_items since they're now in available_items
        if ( found_orphaned_jobs ) {
            execution_items.erase(
                std::remove_if(execution_items.begin(), execution_items.end(),
                    [](const TemplateMatchQueueItem& item) { return item.queue_status == "failed" && item.queue_order == -1; }),
                execution_items.end());
        }

        // Combine execution and available items into the main queue
        execution_queue.clear();
        for ( const auto& item : execution_items ) {
            execution_queue.push_back(item);
        }
        for ( const auto& item : available_items ) {
            execution_queue.push_back(item);
        }

        // Mark that we've loaded from database
        needs_database_load = false;

        // Save crash recovery changes back to database
        if ( found_orphaned_jobs ) {
            wxPrintf("CRASH RECOVERY: Saving corrected job statuses to database\n");
            SaveQueueToDatabase();
        }

        // Auto-populate available jobs from all template match jobs in database
        PopulateAvailableJobsFromDatabase();

        // Update display
        UpdateQueueDisplay( );
    }
}

void TemplateMatchQueueManager::SaveQueueToDatabase( ) {
    if ( match_template_panel_ptr && main_frame && main_frame->current_project.is_open ) {
        // Update status and queue position for all items in queue
        for ( const auto& item : execution_queue ) {
            main_frame->current_project.database.UpdateQueueStatus(item.template_match_id, item.queue_status);

            // Convert 0-based queue positions to 1-based for database storage
            // Keep -1 as-is for available jobs
            int db_queue_order = (item.queue_order >= 0) ? (item.queue_order + 1) : item.queue_order;
            main_frame->current_project.database.UpdateQueuePosition(item.template_match_id, db_queue_order);
        }
    }
}

// Manual drag and drop implementation for wxListCtrl
void TemplateMatchQueueManager::OnMouseLeftDown(wxMouseEvent& event) {

    // Prevent drag operations during display updates
    if (updating_display) {
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

        // Find the corresponding job
        for (const auto& job : execution_queue) {
            if (job.queue_order == hit_item) {
                dragged_job_id = job.template_match_id;
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
            drag_in_progress = true;
            queue_list_ctrl->SetCursor(wxCursor(wxCURSOR_HAND));
        }
    }

    event.Skip();
}

void TemplateMatchQueueManager::OnMouseLeftUp(wxMouseEvent& event) {

    if (drag_in_progress) {
        // Find drop target
        wxPoint drop_pos = event.GetPosition();
        int flags;
        long drop_item = queue_list_ctrl->HitTest(drop_pos, flags);


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

    // Update the completed job (which should already be at priority -1 from when it started running)
    bool job_found = false;
    for (auto& job : execution_queue) {
        if (job.template_match_id == template_match_id) {
            wxPrintf("Found completed job %ld at priority %d, updating status to %s\n",
                     template_match_id, job.queue_order, status);
            job.queue_status = status;
            // Job should already be at queue_order = -1 (available queue) from when it started running
            if (job.queue_order != -1) {
                wxPrintf("WARNING: Completed job %ld was not at priority -1 (was %d), correcting\n",
                         template_match_id, job.queue_order);
                job.queue_order = -1;
            }
            job_found = true;
            break;
        }
    }

    if (!job_found) {
        wxPrintf("WARNING: Could not find completed job %ld in execution queue\n", template_match_id);
    }

    // No need to renumber - jobs were already shifted when the completed job started running
    // The execution queue should already have consecutive priorities: 0, 1, 2, 3...

    // Clear currently running ID since job is done
    currently_running_id = -1;

    // Update both displays to show new status and position changes
    UpdateQueueDisplay();
    UpdateAvailableJobsDisplay();

    // Save changes to database
    SaveQueueToDatabase();

    // Only auto-progress if enabled (default false - user controls progression)
    if (auto_progress_queue) {
        wxPrintf("Auto-progressing to next job in queue\n");
        ProgressQueue( );
    } else {
        wxPrintf("Job completed - auto-progression disabled (user controls queue progression)\n");
    }
}

void TemplateMatchQueueManager::PopulateSelectionQueueFromUI( ) {

    // Clear existing selection queue
    selected_jobs_for_execution.clear( );

    // Get all selected items from wxListCtrl
    long item = queue_list_ctrl->GetNextItem(-1, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
    int selection_count = 0;

    while ( item != -1 ) {
        selection_count++;

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

JobCompletionInfo TemplateMatchQueueManager::GetJobCompletionInfo(long queue_template_match_id) {
    JobCompletionInfo info;
    info.template_match_job_id = queue_template_match_id;

    // Find the actual database TEMPLATE_MATCH_JOB_ID for this queue item
    long database_template_match_job_id = -1;

    // Find the database TEMPLATE_MATCH_JOB_ID and image group for this queue item
    int image_group_id = -1;

    // First check execution queue
    for (const auto& item : execution_queue) {
        if (item.template_match_id == queue_template_match_id) {
            database_template_match_job_id = item.template_match_job_id;
            image_group_id = item.image_group_id;
            break;
        }
    }

    // If not found, check available queue
    if (database_template_match_job_id == -1) {
        for (const auto& item : available_queue) {
            if (item.template_match_id == queue_template_match_id) {
                database_template_match_job_id = item.template_match_job_id;
                image_group_id = item.image_group_id;
                break;
            }
        }
    }

    // If still not found, get image group from database by looking at the first image in this job
    if (image_group_id == -1 && database_template_match_job_id > 0 && main_frame && main_frame->current_project.is_open) {
        // Get the first image asset ID from this template match job
        wxString sql = wxString::Format(
            "SELECT IA.PARENT_IMAGE_GROUP_ID FROM IMAGE_ASSETS IA "
            "INNER JOIN TEMPLATE_MATCH_LIST TML ON IA.IMAGE_ASSET_ID = TML.IMAGE_ASSET_ID "
            "WHERE TML.TEMPLATE_MATCH_JOB_ID = %ld LIMIT 1", database_template_match_job_id);

        image_group_id = main_frame->current_project.database.ReturnSingleIntFromSelectCommand(sql);
        if (image_group_id <= 0) {
            image_group_id = -1; // Fall back to all images if no results
        }
    }

    // Get completion counts from database using the actual TEMPLATE_MATCH_JOB_ID
    if (database_template_match_job_id > 0) {
        auto counts = main_frame->current_project.database.GetJobCompletionCounts(database_template_match_job_id, image_group_id);
        info.completed_count = counts.first;
        info.total_count = counts.second;
    } else {
        // No database job ID yet - no results written
        info.completed_count = 0;
        info.total_count = 0;
    }

    // Debug output for n/N tracking
    wxPrintf("GetJobCompletionInfo: Queue ID %ld, Database Job ID %ld, ImageGroup %d -> %d/%d\n",
             queue_template_match_id, database_template_match_job_id, image_group_id,
             info.completed_count, info.total_count);

    return info;
}

void TemplateMatchQueueManager::RefreshJobCompletionInfo() {
    // Update execution queue display
    UpdateQueueDisplay();

    // Update available jobs display
    UpdateAvailableJobsDisplay();
}

void TemplateMatchQueueManager::OnResultAdded(long template_match_job_id) {
    wxPrintf("OnResultAdded called for job %ld - refreshing n/N display\n", template_match_job_id);

    // Refresh the completion info for this job and update displays
    RefreshJobCompletionInfo();
}

void TemplateMatchQueueManager::UpdateJobDatabaseId(long queue_template_match_id, long database_template_match_job_id) {
    // Update the queue item with the actual TEMPLATE_MATCH_JOB_ID for correct n/N tracking
    wxPrintf("UpdateJobDatabaseId: Mapping queue ID %ld to database TEMPLATE_MATCH_JOB_ID %ld\n",
             queue_template_match_id, database_template_match_job_id);

    // Check execution queue first
    for (auto& item : execution_queue) {
        if (item.template_match_id == queue_template_match_id) {
            wxPrintf("Found queue item %ld in execution queue, setting template_match_job_id = %ld\n",
                     queue_template_match_id, database_template_match_job_id);
            item.template_match_job_id = database_template_match_job_id;
            return;
        }
    }

    // Check available queue
    for (auto& item : available_queue) {
        if (item.template_match_id == queue_template_match_id) {
            wxPrintf("Found queue item %ld in available queue, setting template_match_job_id = %ld\n",
                     queue_template_match_id, database_template_match_job_id);
            item.template_match_job_id = database_template_match_job_id;
            return;
        }
    }

    wxPrintf("Warning: Could not find queue item %ld to update with database job ID %ld\n",
             queue_template_match_id, database_template_match_job_id);
}

void TemplateMatchQueueManager::DiscoverDatabaseJobIds() {
    if (!main_frame || !main_frame->current_project.is_open) {
        wxPrintf("DiscoverDatabaseJobIds: No project open, skipping discovery\n");
        return;
    }

    wxPrintf("DiscoverDatabaseJobIds: Starting discovery for completed jobs...\n");

    // Get all available TEMPLATE_MATCH_JOB_IDs from the database
    wxArrayLong database_job_ids = main_frame->current_project.database.GetAllTemplateMatchJobIds();

    wxPrintf("Found %zu database job IDs: ", database_job_ids.GetCount());
    for (size_t i = 0; i < database_job_ids.GetCount(); i++) {
        wxPrintf("%ld ", database_job_ids[i]);
    }
    wxPrintf("\n");

    // For each queue item that needs mapping, try to find its database job ID
    auto tryMapQueueItem = [&](TemplateMatchQueueItem& item) {
        if (item.template_match_job_id > 0) {
            // Already has a database job ID
            return;
        }

        if (item.queue_status != "complete" && item.queue_status != "running") {
            // Only map completed/running jobs (jobs with results in database)
            return;
        }

        wxPrintf("Attempting to discover database job ID for queue item %ld (status: %s, image_group: %d)\n",
                 item.template_match_id, item.queue_status, item.image_group_id);

        // Try each database job ID to see if it contains results for this queue item's image group
        for (size_t i = 0; i < database_job_ids.GetCount(); i++) {
            long database_job_id = database_job_ids[i];

            // Check if this database job ID has results for images in this queue item's image group
            auto counts = main_frame->current_project.database.GetJobCompletionCounts(database_job_id, item.image_group_id);

            if (counts.first > 0 || counts.second > 0) {
                // Found results for this image group in this database job
                wxPrintf("DISCOVERED: Queue item %ld maps to database job ID %ld (%d/%d results)\n",
                         item.template_match_id, database_job_id, counts.first, counts.second);
                item.template_match_job_id = database_job_id;
                return;
            }
        }

        wxPrintf("Could not discover database job ID for queue item %ld\n", item.template_match_id);
    };

    // Apply discovery to both queues
    for (auto& item : execution_queue) {
        tryMapQueueItem(item);
    }

    for (auto& item : available_queue) {
        tryMapQueueItem(item);
    }

    wxPrintf("DiscoverDatabaseJobIds: Discovery complete\n");
}

void TemplateMatchQueueManager::PopulateAvailableJobsFromDatabase() {
    if (!main_frame || !main_frame->current_project.is_open) {
        return;
    }

    // Get all template match job IDs from database
    wxArrayLong all_job_ids = main_frame->current_project.database.GetAllTemplateMatchJobIds();

    for (size_t i = 0; i < all_job_ids.GetCount(); ++i) {
        long job_id = all_job_ids[i];

        // Check if this job is already in our queues
        bool found_in_execution = false;
        bool found_in_available = false;

        for (const auto& item : execution_queue) {
            if (item.template_match_id == job_id) {
                found_in_execution = true;
                break;
            }
        }

        if (!found_in_execution) {
            for (const auto& item : available_queue) {
                if (item.template_match_id == job_id) {
                    found_in_available = true;
                    break;
                }
            }
        }

        // If not found in either queue, add to available queue
        if (!found_in_execution && !found_in_available) {
            // Create a minimal queue item - populate image group from database
            TemplateMatchQueueItem new_item;
            new_item.template_match_id = job_id;
            new_item.job_name = wxString::Format("Job %ld", job_id);
            new_item.queue_status = "unknown"; // Will be determined by n/N analysis
            new_item.queue_order = -1; // Not in execution queue

            // We'll set a placeholder image group - GetJobCompletionInfo will handle it
            new_item.image_group_id = -1;

            // Get completion info with proper image group
            auto counts = main_frame->current_project.database.GetJobCompletionCounts(job_id, new_item.image_group_id);
            if (counts.second > 0) {
                if (counts.first == 0) {
                    new_item.queue_status = "pending";
                } else if (counts.first == counts.second) {
                    new_item.queue_status = "complete";
                } else {
                    new_item.queue_status = "running";
                }
            }

            available_queue.push_back(new_item);

            // Debug: Get completion info to verify n/N calculation
            JobCompletionInfo completion = GetJobCompletionInfo(job_id);
            wxPrintf("Added available job %ld: %s with status %s (%s)\n",
                     job_id, new_item.job_name, new_item.queue_status, completion.GetCompletionString());
        }
    }

    // Refresh the display
    UpdateAvailableJobsDisplay();
}