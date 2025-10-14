#include "../core/gui_core_headers.h"
#include "TemplateMatchQueueManager.h"
#include "MatchTemplatePanel.h"
#include "TemplateMatchQueueItemEditor.h"
#include <algorithm>

#ifdef cisTEM_QM_LOGGING
// Initialize static members for QueueManagerLogger
std::ofstream QueueManagerLogger::log_file;
wxDateTime    QueueManagerLogger::start_time;
bool          QueueManagerLogger::is_enabled = false;
wxString      QueueManagerLogger::log_file_path;
#endif

BEGIN_EVENT_TABLE(TemplateMatchQueueManager, wxPanel)
EVT_BUTTON(wxID_ANY, TemplateMatchQueueManager::OnRunSelectedClick)
EVT_LIST_ITEM_SELECTED(wxID_ANY, TemplateMatchQueueManager::OnSelectionChanged)
EVT_LIST_ITEM_DESELECTED(wxID_ANY, TemplateMatchQueueManager::OnSelectionChanged)
EVT_LIST_BEGIN_DRAG(wxID_ANY, TemplateMatchQueueManager::OnBeginDrag)
EVT_LIST_COL_CLICK(wxID_ANY, TemplateMatchQueueManager::OnAvailableQueueColumnClick)
#ifdef cisTEM_QM_LOGGING
EVT_TOGGLEBUTTON(wxID_ANY, TemplateMatchQueueManager::OnLoggingToggle)
#endif
END_EVENT_TABLE( )

TemplateMatchQueueManager::TemplateMatchQueueManager(wxWindow* parent, MatchTemplatePanel* match_template_panel)
    : wxPanel(parent, wxID_ANY), match_template_panel_ptr(match_template_panel), currently_running_id(-1), last_populated_queue_id(-1) {

    // Initialize state variables
    auto_progress_queue            = true; // Auto-progress to next search when one completes
    hide_completed_searches        = true; // Hide completed searches by default
    gui_update_frozen              = false; // GUI updates allowed initially
    search_is_finalizing           = false; // No search in finalization initially
    available_queue_sort_column    = 0; // Default sort by Queue ID
    available_queue_sort_ascending = true; // Default ascending sort
    drag_in_progress               = false;
    updating_display               = false; // Not updating display initially
    dragged_row                    = -1;
    dragged_search_id              = -1;
    mouse_down                     = false;

    // Create the main sizer
    wxBoxSizer* main_sizer = new wxBoxSizer(wxVERTICAL);

    // Create execution queue section (top)
    execution_queue_label       = new wxStaticText(this, wxID_ANY, "Execution Queue (searches will run in order - drag to reorder):");
    wxFont execution_label_font = execution_queue_label->GetFont( );
    execution_label_font.MakeBold( );
    execution_queue_label->SetFont(execution_label_font);

    execution_queue_ctrl = new wxListCtrl(this, wxID_ANY,
                                          wxDefaultPosition, wxSize(500, 200),
                                          wxLC_REPORT);

    // Execution queue is reordered by drag-and-drop, not column sorting
    // Calculate column widths based on header text
    auto calc_width = [&](const wxString& header) -> int {
        int text_width = execution_queue_ctrl->GetTextExtent(header).GetWidth( );
        return text_width + 20; // Add padding for margins
    };

    // Add columns to execution queue (no sort indicators - uses drag-and-drop for reordering)
    execution_queue_ctrl->AppendColumn("Queue ID", wxLIST_FORMAT_LEFT, calc_width("Queue ID"));
    execution_queue_ctrl->AppendColumn("Search ID", wxLIST_FORMAT_LEFT, calc_width("Search ID"));
    execution_queue_ctrl->AppendColumn("Search Name", wxLIST_FORMAT_LEFT, 120);
    execution_queue_ctrl->AppendColumn("Status", wxLIST_FORMAT_LEFT, calc_width("Status"));
    execution_queue_ctrl->AppendColumn("Progress", wxLIST_FORMAT_LEFT, calc_width("Progress"));
    execution_queue_ctrl->AppendColumn("CLI Args", wxLIST_FORMAT_LEFT, 80);

    // wxListCtrl doesn't use EnableDragSource/EnableDropTarget - we'll implement manual drag and drop

    // Legacy compatibility - point to execution queue
    queue_list_ctrl = execution_queue_ctrl;

    // Create controls panel with Run Queue button and right-side controls
    controls_panel             = new wxPanel(this, wxID_ANY);
    wxBoxSizer* controls_sizer = new wxBoxSizer(wxHORIZONTAL);

    // Run Queue button (left side)
    run_selected_button = new wxButton(controls_panel, wxID_ANY, "Run Queue");
    controls_sizer->Add(run_selected_button, 0, wxALL, 5);
    controls_sizer->AddStretchSpacer( );

    // Right-side button stack (Update Selected above Enable Logging)
    wxBoxSizer* right_buttons_sizer = new wxBoxSizer(wxVERTICAL);

    // Update Selected button - right-aligned at top
    update_selected_button = new wxButton(controls_panel, wxID_ANY, "Update Selected", wxDefaultPosition, wxSize(130, -1));
    update_selected_button->Enable(false); // Disabled until pending item selected
    right_buttons_sizer->Add(update_selected_button, 0, wxALIGN_RIGHT | wxBOTTOM, 5);

#ifdef cisTEM_QM_LOGGING
    // Logging row: [Log file: <text>] [Enable Logging button] - logging button right-aligned with Update Selected
    wxBoxSizer* logging_row_sizer = new wxBoxSizer(wxHORIZONTAL);
    logging_label                 = new wxStaticText(controls_panel, wxID_ANY, "Log file:");
    log_file_text                 = new wxStaticText(controls_panel, wxID_ANY, "", wxDefaultPosition, wxSize(150, -1));
    logging_toggle                = new wxToggleButton(controls_panel, wxID_ANY, "Enable Logging",
                                                       wxDefaultPosition, wxSize(130, -1));
    logging_toggle->SetValue(false); // Off by default

    logging_row_sizer->Add(logging_label, 0, wxALIGN_CENTER_VERTICAL | wxRIGHT, 5);
    logging_row_sizer->Add(log_file_text, 0, wxALIGN_CENTER_VERTICAL | wxRIGHT, 5);
    logging_row_sizer->Add(logging_toggle, 0, wxALIGN_CENTER_VERTICAL, 0);
    right_buttons_sizer->Add(logging_row_sizer, 0, wxALIGN_RIGHT, 0);
#endif

    controls_sizer->Add(right_buttons_sizer, 0, wxALIGN_CENTER_VERTICAL | wxALL, 5);
    controls_panel->SetSizer(controls_sizer);

    // Create available searches section header (hide completed moved to CLI args section)
    available_searches_label    = new wxStaticText(this, wxID_ANY, "Available Searches (not queued for execution):");
    wxFont available_label_font = available_searches_label->GetFont( );
    available_label_font.MakeBold( );
    available_searches_label->SetFont(available_label_font);
    available_searches_ctrl = new wxListCtrl(this, wxID_ANY,
                                             wxDefaultPosition, wxSize(500, 150),
                                             wxLC_REPORT);

    // Set minimum size to ensure visibility
    available_searches_ctrl->SetMinSize(wxSize(500, 150));
    QM_LOG_UI("Created available_searches_ctrl: %p with min size 700x150", available_searches_ctrl);

    // Unicode sortable indicator (U+21C5) - only for sortable columns
    wxString sort_indicator_avail = wxString::FromUTF8(" ⇅");

    // Calculate column widths based on header text
    auto calc_width_avail = [&](const wxString& header, bool sortable) -> int {
        wxString header_text = sortable ? header + sort_indicator_avail : header;
        int      text_width  = available_searches_ctrl->GetTextExtent(header_text).GetWidth( );
        return text_width + 20; // Add padding for margins
    };

    // Add columns to available searches (Queue ID, Search ID, Status, and timing columns are sortable)
    available_searches_ctrl->AppendColumn(wxString::FromUTF8("Queue ID") + sort_indicator_avail, wxLIST_FORMAT_LEFT, calc_width_avail("Queue ID", true));
    available_searches_ctrl->AppendColumn(wxString::FromUTF8("Search ID") + sort_indicator_avail, wxLIST_FORMAT_LEFT, calc_width_avail("Search ID", true));
    available_searches_ctrl->AppendColumn("Search Name", wxLIST_FORMAT_LEFT, 120);
    available_searches_ctrl->AppendColumn(wxString::FromUTF8("Status") + sort_indicator_avail, wxLIST_FORMAT_LEFT, calc_width_avail("Status", true));
    available_searches_ctrl->AppendColumn("Progress", wxLIST_FORMAT_LEFT, calc_width_avail("Progress", false));
    available_searches_ctrl->AppendColumn(wxString::FromUTF8("Total Time") + sort_indicator_avail, wxLIST_FORMAT_LEFT, calc_width_avail("Total Time", true));
    available_searches_ctrl->AppendColumn(wxString::FromUTF8("Avg Time/Job") + sort_indicator_avail, wxLIST_FORMAT_LEFT, calc_width_avail("Avg Time/Job", true));
    available_searches_ctrl->AppendColumn(wxString::FromUTF8("Latest") + sort_indicator_avail, wxLIST_FORMAT_LEFT, calc_width_avail("Latest", true));
    available_searches_ctrl->AppendColumn("CLI Args", wxLIST_FORMAT_LEFT, 80);

    // Create bottom controls panel with left and right button groups and hide completed checkbox
    bottom_controls          = new wxPanel(this, wxID_ANY);
    wxBoxSizer* bottom_sizer = new wxBoxSizer(wxHORIZONTAL);

    // Hide completed checkbox (at start of bottom controls)
    hide_completed_checkbox = new wxCheckBox(bottom_controls, wxID_ANY, "Hide completed searches");
    hide_completed_checkbox->SetValue(true); // Checked by default

    // Left side - stacked Add/Remove from Queue buttons
    wxBoxSizer* left_button_sizer = new wxBoxSizer(wxVERTICAL);
    add_to_queue_button           = new wxButton(bottom_controls, wxID_ANY, "Add to Execution Queue");
    remove_from_queue_button      = new wxButton(bottom_controls, wxID_ANY, "Remove from Execution Queue");
    left_button_sizer->Add(add_to_queue_button, 0, wxEXPAND | wxBOTTOM, 5);
    left_button_sizer->Add(remove_from_queue_button, 0, wxEXPAND);

    // Right side - stacked Remove Selected/Clear All buttons
    wxBoxSizer* right_button_sizer = new wxBoxSizer(wxVERTICAL);
    remove_selected_button         = new wxButton(bottom_controls, wxID_ANY, "Remove Selected");
    clear_queue_button             = new wxButton(bottom_controls, wxID_ANY, "Clear All");
    right_button_sizer->Add(remove_selected_button, 0, wxEXPAND | wxBOTTOM, 5);
    right_button_sizer->Add(clear_queue_button, 0, wxEXPAND);

    bottom_sizer->Add(hide_completed_checkbox, 0, wxALIGN_CENTER_VERTICAL | wxALL, 5);
    bottom_sizer->AddStretchSpacer( );
    bottom_sizer->Add(left_button_sizer, 0, wxALL, 5);
    bottom_sizer->AddStretchSpacer( );
    bottom_sizer->Add(right_button_sizer, 0, wxALL, 5);
    bottom_controls->SetSizer(bottom_sizer);

    // Add all sections to main sizer with adjusted proportions for better visibility
    main_sizer->Add(execution_queue_label, 0, wxEXPAND | wxALL, 5);
    main_sizer->Add(execution_queue_ctrl, 2, wxEXPAND | wxALL, 5); // Slightly larger proportion
    main_sizer->Add(controls_panel, 0, wxEXPAND | wxALL, 5); // Run Queue + Panel Display + Logging (monitoring controls)
    main_sizer->Add(available_searches_label, 0, wxEXPAND | wxALL, 5);
    main_sizer->Add(available_searches_ctrl, 1, wxEXPAND | wxALL, 5); // Smaller but still expandable
    main_sizer->Add(bottom_controls, 0, wxEXPAND | wxALL, 5);

    SetSizer(main_sizer);

    // Force layout update - important for dialog visibility
    Layout( );
    QM_LOG_UI("Layout() called");

    // Connect events
    update_selected_button->Bind(wxEVT_BUTTON, &TemplateMatchQueueManager::OnUpdateSelectedClick, this);
    run_selected_button->Bind(wxEVT_BUTTON, &TemplateMatchQueueManager::OnRunSelectedClick, this);
    add_to_queue_button->Bind(wxEVT_BUTTON, &TemplateMatchQueueManager::OnAddToQueueClick, this);
    remove_from_queue_button->Bind(wxEVT_BUTTON, &TemplateMatchQueueManager::OnRemoveFromQueueClick, this);
    remove_selected_button->Bind(wxEVT_BUTTON, &TemplateMatchQueueManager::OnRemoveSelectedClick, this);
    clear_queue_button->Bind(wxEVT_BUTTON, &TemplateMatchQueueManager::OnClearQueueClick, this);
    hide_completed_checkbox->Bind(wxEVT_CHECKBOX, &TemplateMatchQueueManager::OnHideCompletedToggle, this);

    // Bind selection events for available searches table
    available_searches_ctrl->Bind(wxEVT_LIST_ITEM_SELECTED, &TemplateMatchQueueManager::OnAvailableSearchesSelectionChanged, this);
    available_searches_ctrl->Bind(wxEVT_LIST_ITEM_DESELECTED, &TemplateMatchQueueManager::OnAvailableSearchesSelectionChanged, this);

    // Bind mouse events for manual drag and drop implementation
    execution_queue_ctrl->Bind(wxEVT_LEFT_DOWN, &TemplateMatchQueueManager::OnMouseLeftDown, this);
    execution_queue_ctrl->Bind(wxEVT_MOTION, &TemplateMatchQueueManager::OnMouseMotion, this);
    execution_queue_ctrl->Bind(wxEVT_LEFT_UP, &TemplateMatchQueueManager::OnMouseLeftUp, this);

#ifdef cisTEM_QM_LOGGING
    // Bind logging event
    logging_toggle->Bind(wxEVT_TOGGLEBUTTON, &TemplateMatchQueueManager::OnLoggingToggle, this);
#endif

    // Don't load from database in constructor - it may be called during workflow switch
    // when main_frame is in an inconsistent state

    // Create editor panel (hidden initially)
    editor_panel = new TemplateMatchQueueItemEditor(this, match_template_panel_ptr, this);
    editor_panel->Show(false);
    editing_queue_id = -1;

    // Add editor panel to main sizer (will be shown/hidden as needed)
    main_sizer->Add(editor_panel, 1, wxEXPAND | wxALL, 5);

    // Display any existing queue items
    UpdateQueueDisplay( );
}

TemplateMatchQueueManager::~TemplateMatchQueueManager( ) {
    // Manual drag and drop doesn't need cleanup
    // No callback cleanup needed - panel has persistent queue_manager member
}

bool TemplateMatchQueueManager::ValidateQueueItem(const TemplateMatchQueueItem& item, wxString& error_message) {
    // Validate basic parameters
    if ( item.image_group_id < -1 ) {
        error_message = wxString::Format("Invalid image group ID: %d", item.image_group_id);
        return false;
    }
    if ( item.reference_volume_asset_id < 0 ) {
        error_message = wxString::Format("Invalid reference volume ID: %d", item.reference_volume_asset_id);
        return false;
    }
    if ( item.search_name.IsEmpty( ) ) {
        error_message = "Search name cannot be empty";
        return false;
    }
    if ( item.queue_status != "pending" && item.queue_status != "running" &&
         item.queue_status != "complete" && item.queue_status != "failed" &&
         item.queue_status != "partial" ) {
        error_message = wxString::Format("Invalid queue status: %s", item.queue_status);
        return false;
    }

    error_message = "";
    return true;
}

long TemplateMatchQueueManager::AddToExecutionQueue(const TemplateMatchQueueItem& item) {
    // Validate the queue item before adding
    wxString error_message;
    if ( ! ValidateQueueItem(item, error_message) ) {
        wxMessageBox(error_message, "Invalid Queue Item", wxOK | wxICON_ERROR);
        return -1;
    }

    MyDebugAssertTrue(match_template_panel_ptr != nullptr, "AddToExecutionQueue called with null match_template_panel_ptr");
    MyDebugAssertTrue(main_frame != nullptr, "AddToExecutionQueue called with null main_frame");
    MyDebugAssertTrue(main_frame->current_project.is_open, "AddToExecutionQueue called with no project open");

    // Add item to database and get the new queue ID
    long new_database_queue_id = main_frame->current_project.database.AddToTemplateMatchQueue(
            item.search_name, item.image_group_id, item.reference_volume_asset_id, item.run_profile_id,
            item.use_gpu, item.use_fast_fft, item.symmetry,
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
    new_item.search_id              = -1; // Search ID is only assigned when first result is written

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

    // Highlight the newly added item (last item in the visible list)
    if ( queue_list_ctrl && queue_list_ctrl->GetItemCount( ) > 0 ) {
        int new_row = queue_list_ctrl->GetItemCount( ) - 1;
        queue_list_ctrl->SetItemState(new_row, wxLIST_STATE_SELECTED, wxLIST_STATE_SELECTED);
    }

    return new_database_queue_id;
}

void TemplateMatchQueueManager::PrintQueueState( ) {
    QM_LOG_DEBUG("  Execution Queue (%zu items):", execution_queue.size( ));
    for ( const auto& item : execution_queue ) {
        if ( item.queue_order >= 0 ) {
            QM_LOG_DEBUG("    [%d] ID=%ld, Status=%s, Name=%s",
                         item.queue_order, item.database_queue_id,
                         item.queue_status.mb_str( ).data( ), item.search_name.mb_str( ).data( ));
        }
    }
    QM_LOG_DEBUG("  Available items in execution_queue (queue_order < 0): %d",
                 int(std::count_if(execution_queue.begin( ), execution_queue.end( ),
                                   [](const auto& item) { return item.queue_order < 0; })));
    QM_LOG_DEBUG("  Available Queue (%zu items)", available_queue.size( ));
    QM_LOG_DEBUG("  Currently running search ID: %ld", currently_running_id);
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
                          "Cannot remove currently running search (ID: %ld)", execution_queue[index].database_queue_id);

        MyDebugAssertTrue(match_template_panel_ptr != nullptr, "RemoveFromExecutionQueue: match_template_panel_ptr is null");
        MyDebugAssertTrue(main_frame != nullptr, "RemoveFromExecutionQueue: main_frame is null");
        MyDebugAssertTrue(main_frame->current_project.is_open, "RemoveFromExecutionQueue: no project open");

        // CRITICAL RULE: Only delete if search_id == -1 (no results exist), otherwise move to available queue
        if ( execution_queue[index].search_id == -1 ) {
            // Safe to delete - no results exist
            main_frame->current_project.database.RemoveFromQueue(execution_queue[index].database_queue_id);
        }
        else {
            // Has search_id - results exist, move to available queue instead
            execution_queue[index].queue_order = -1;
            UpdateQueueItemInDatabase(execution_queue[index]);
            available_queue.push_back(execution_queue[index]);
        }

        execution_queue.erase(execution_queue.begin( ) + index);
        UpdateQueueDisplay( );
    }
}

void TemplateMatchQueueManager::ClearExecutionQueue( ) {
    // CRITICAL RULE: Only delete items with search_id == -1 (no results exist)
    // Items with search_id set must NEVER be deleted - only moved to available queue
    MyDebugAssertTrue(match_template_panel_ptr != nullptr, "ClearExecutionQueue: match_template_panel_ptr is null");
    MyDebugAssertTrue(main_frame != nullptr, "ClearExecutionQueue: main_frame is null");
    MyDebugAssertTrue(main_frame->current_project.is_open, "ClearExecutionQueue: no project open");

    // Process each item - delete only if search_id == -1, otherwise move to available
    for ( auto& item : execution_queue ) {
        // Only delete if search_id == -1 (no results table entry exists)
        if ( item.search_id == -1 ) {
            // Safe to delete - no results exist
            main_frame->current_project.database.RemoveFromQueue(item.database_queue_id);
        }
        else {
            // Has search_id - results exist, NEVER delete, move to available queue instead
            item.queue_order = -1;
            UpdateQueueItemInDatabase(item);
            available_queue.push_back(item);
        }
    }

    execution_queue.clear( );
    UpdateQueueDisplay( );
}

void TemplateMatchQueueManager::SetExecutionQueuePosition(int search_index, int new_position) {
    // Validate inputs
    MyDebugAssertTrue(search_index >= 0 && search_index < execution_queue.size( ),
                      "Invalid search_index: %d (queue size: %zu)", search_index, execution_queue.size( ));
    MyDebugAssertTrue(new_position >= 1 && new_position <= execution_queue.size( ),
                      "Invalid new_position: %d (valid range: 1-%zu)", new_position, execution_queue.size( ));

    int current_position = execution_queue[search_index].queue_order;
    if ( current_position == new_position ) {
        return; // No change needed
    }

    // Update queue orders for all affected jobs
    if ( current_position < new_position ) {
        // Moving search to a later position - decrement searches between current and new position
        for ( auto& item : execution_queue ) {
            if ( item.queue_order > current_position && item.queue_order <= new_position ) {
                item.queue_order--;
            }
        }
    }
    else {
        // Moving search to an earlier position - increment searches between new and current position
        for ( auto& item : execution_queue ) {
            if ( item.queue_order >= new_position && item.queue_order < current_position ) {
                item.queue_order++;
            }
        }
    }

    // Set the moved search to its new position
    execution_queue[search_index].queue_order = new_position;

    QM_LOG_STATE("SetSearchPosition: Moved search %ld from position %d to %d",
                 execution_queue[search_index].database_queue_id, current_position, new_position);
}

void TemplateMatchQueueManager::ProgressExecutionQueue( ) {
    if constexpr ( skip_search_execution_for_queue_debugging ) {
        QM_LOG_DEBUG("=== PROGRESS EXECUTION QUEUE CALLED ===");
        QM_LOG_DEBUG("Queue state at start of ProgressExecutionQueue:");
        PrintQueueState( );
    }

    // New logic: find the search at priority 0 and run it - allow any non-complete searches
    TemplateMatchQueueItem* next_search = nullptr;
    for ( auto& item : execution_queue ) {
        if ( item.queue_order == 0 && item.queue_status != "complete" ) {
            next_search = &item;
            if constexpr ( skip_search_execution_for_queue_debugging ) {
                QM_LOG_DEBUG("Found next search to run: ID=%ld at queue_order=0", item.database_queue_id);
            }
            break;
        }
    }

    if ( next_search ) {
        // Don't change status yet - let MatchTemplatePanel validate first
        // Move to priority -1 (out of execution queue) but keep original status
        int original_queue_order = next_search->queue_order;
        next_search->queue_order = -1;

        // Shift all other execution queue jobs up by one priority (1->0, 2->1, 3->2, etc.)
        for ( auto& item : execution_queue ) {
            if ( item.queue_order > 0 ) {
                item.queue_order--;
            }
        }

        QM_LOG_SEARCH("ProgressQueue: Starting search %ld (moved to priority -1, status: %s)",
                      next_search->database_queue_id, next_search->queue_status);

        // Execute the search - status will be changed to "running" by MatchTemplatePanel after validation
        if ( ExecuteSearch(*next_search) ) {
            QM_LOG_SEARCH("ProgressQueue: Search %ld execution started successfully", next_search->database_queue_id);
        }
        else {
            // If execution failed, mark as failed and ensure it stays in available queue
            next_search->queue_status = "failed";
            // next_search->queue_order is already -1

            // Critical: Ensure currently_running_id is cleared when search fails to start
            // ExecuteSearch should not have set it if it returned false, but clear it to be safe
            currently_running_id = -1;

            QM_LOG_ERROR("ProgressQueue: Search %ld execution failed, trying next search", next_search->database_queue_id);

            // Update display and save failed search status
            SaveQueueToDatabase( );
            UpdateQueueDisplay( );

            // Recursively try the next search in the queue
            if ( auto_progress_queue ) {
                QM_LOG_SEARCH("ProgressQueue: Attempting next search after failure");
                ProgressExecutionQueue( );
            }
            return; // Early return to avoid duplicate save/update below
        }

        // Update display and save to database
        SaveQueueToDatabase( );
        UpdateQueueDisplay( );
    }
    else {
        QM_LOG_SEARCH("ProgressQueue: No pending searches found at priority 0");
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

    // Set the status text (column 3 for both execution and available queues)
    list_ctrl->SetItem(item_index, 3, status_display);

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
        if ( item->search_id > 0 ) {
            ctrl->SetItem(row, 1, wxString::Format("%ld", item->search_id));
        }
        else {
            // No valid search ID yet (search hasn't started or search_id is -1 or 0) - show empty string
            ctrl->SetItem(row, 1, "");
        }

        // CRITICAL: Store the database_queue_id as item data so we can retrieve it later
        // This allows us to identify which queue item a row represents regardless of sorting/filtering
        ctrl->SetItemData(row, item->database_queue_id);

        ctrl->SetItem(row, 2, item->search_name);
        // Status is column 3
        if ( item->queue_status == "complete" || item->queue_status == "running" ) {
            QM_LOG_UI("PopulateListControl: Setting status for queue_id %ld to '%s' in row %zu",
                      item->database_queue_id, item->queue_status.mb_str( ).data( ), idx);
        }
        SetStatusDisplay(ctrl, row, item->queue_status);
        // Progress is column 4
        SearchCompletionInfo completion = GetSearchCompletionInfo(item->database_queue_id);
        ctrl->SetItem(row, 4, completion.GetCompletionString( ));

        // Timing columns (5-7) are only for available searches with completed results
        if ( ! is_execution_queue && item->search_id > 0 ) {
            SearchTimingInfo timing = GetSearchTimingInfo(item->search_id);

            // Only display if we have actual completed jobs with timing data
            if ( timing.completed_count > 0 ) {
                // Total Time is column 5
                ctrl->SetItem(row, 5, FormatElapsedTime(timing.total_elapsed_seconds));
                // Avg Time/Job is column 6
                ctrl->SetItem(row, 6, FormatElapsedTime(timing.avg_elapsed_seconds));
                // Latest completion datetime is column 7
                ctrl->SetItem(row, 7, timing.latest_datetime);
            }
            else {
                // No completed jobs yet - leave empty
                ctrl->SetItem(row, 5, "");
                ctrl->SetItem(row, 6, "");
                ctrl->SetItem(row, 7, "");
            }
        }
        else if ( ! is_execution_queue ) {
            // No search_id or is execution queue - leave empty
            ctrl->SetItem(row, 5, "");
            ctrl->SetItem(row, 6, "");
            ctrl->SetItem(row, 7, "");
        }

        // Custom args - column 5 for execution queue, column 8 for available searches
        int cli_args_column = is_execution_queue ? 5 : 8;
        ctrl->SetItem(row, cli_args_column, item->custom_cli_args);
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
    UpdateAvailableSearchesDisplay( );

    // Re-enable drag operations after display update is complete
    updating_display = false;
}

void TemplateMatchQueueManager::UpdateAvailableSearchesDisplay( ) {
    QM_LOG_METHOD_ENTRY("UpdateAvailableSearchesDisplay");
    MyDebugAssertTrue(available_searches_ctrl != nullptr, "available_searches_ctrl is null in UpdateAvailableSearchesDisplay");

    // Build list of available items
    std::vector<TemplateMatchQueueItem*> available_items;

    // Add jobs from execution_queue with queue_order < 0
    for ( auto& job : execution_queue ) {
        if ( job.queue_order < 0 ) {
            // Skip completed searches if hide_completed_searches is enabled
            if ( ! hide_completed_searches || job.queue_status != "complete" ) {
                available_items.push_back(&job);
                QM_LOG_UI("UpdateAvailableSearchesDisplay: Added queue_id %ld (status: %s) to available list",
                          job.database_queue_id, job.queue_status.mb_str( ).data( ));
            }
        }
    }

    // Add jobs from available_queue that aren't in execution_queue
    for ( auto& job : available_queue ) {
        // Skip completed searches if hide_completed_searches is enabled
        if ( hide_completed_searches && job.queue_status == "complete" ) {
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

    // Sort items based on current column and direction
    SortAvailableItems(available_items);

    // Use consolidated method to populate the list
    PopulateListControl(available_searches_ctrl, available_items, false);

    QM_LOG_UI("UpdateAvailableSearchesDisplay: Added %zu available searches", available_items.size( ));
    QM_LOG_METHOD_EXIT("UpdateAvailableSearchesDisplay");
}

int TemplateMatchQueueManager::GetSelectedRow( ) {
    long selected_item = queue_list_ctrl->GetNextItem(-1, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
    if ( selected_item != -1 ) {
        return selected_item;
    }
    return -1;
}

// RunAllJobs method removed - use Run Selected with multi-selection instead
// RunSelectedJob method removed - redundant with RunNextSearch

void TemplateMatchQueueManager::RunNextSearch( ) {
    MyPrintWithDetails("=== RUN NEXT SEARCH CALLED ===");

    // Check if a search is already running or finalizing
    if ( IsSearchRunning( ) ) {
        if ( search_is_finalizing ) {
            QM_LOG_SEARCH("Search is still finalizing - deferring RunNextSearch");
            // Schedule another attempt after finalization completes
            wxTimer* retry_timer = new wxTimer( );
            retry_timer->Bind(wxEVT_TIMER, [this, retry_timer](wxTimerEvent&) {
                QM_LOG_SEARCH("Retrying RunNextSearch after finalization delay");
                if ( ! IsSearchRunning( ) ) {
                    RunNextSearch( );
                }
                delete retry_timer;
            });
            retry_timer->StartOnce(1000); // Retry in 1 second
        }
        else {
            MyDebugPrint("Search %ld is already running - skipping RunNextSearch", currently_running_id);
        }
        return;
    }

    MyDebugPrint("No search currently running - searching for next pending search");

    // Find the search at priority 0 (top of execution queue) - allow any non-complete searches
    TemplateMatchQueueItem* next_search = nullptr;
    for ( auto& item : execution_queue ) {
        if ( item.queue_order == 0 && item.queue_status != "complete" ) {
            next_search = &item;
            break;
        }
    }

    if ( ! next_search ) {
        wxMessageBox("No pending searches in execution queue at priority 0.", "No Searches to Run", wxOK | wxICON_WARNING);
        return;
    }

    MyDebugPrint("Found next pending search:");
    MyDebugPrint("  ID: %ld", next_search->database_queue_id);
    MyDebugPrint("  Name: '%s'", next_search->search_name.mb_str( ).data( ));
    MyDebugPrint("  Status: '%s'", next_search->queue_status.mb_str( ).data( ));
    MyDebugPrint("  Image Group: %d", next_search->image_group_id);
    MyDebugPrint("  Reference Volume: %d", next_search->reference_volume_asset_id);

    // Enable auto-progression for queue mode - each search completion will trigger the next
    SetAutoProgressQueue(true);

    // Move to priority -1 (out of execution queue) but keep original status
    next_search->queue_order = -1; // Move running search out of execution queue for crash safety

    // Shift all other execution queue jobs up by one priority (1->0, 2->1, 3->2, etc.)
    for ( auto& item : execution_queue ) {
        if ( item.queue_order > 0 ) {
            item.queue_order--;
        }
    }

    MyDebugPrint("=== EXECUTING SEARCH %ld ===", next_search->database_queue_id);

    // Execute the job - status will be changed to "running" by MatchTemplatePanel after validation
    if ( ExecuteSearch(*next_search) ) {
        QM_LOG_SEARCH("RunNextSearch: Started search %ld (moved to priority -1, status: %s)",
                      next_search->database_queue_id, next_search->queue_status);
    }
    else {
        // If execution failed, mark as failed and restore to available queue
        next_search->queue_status = "failed";
        // queue_order already set to -1, which is correct for available queue
        QM_LOG_ERROR("RunNextSearch: Search %ld execution failed", next_search->database_queue_id);
    }

    // Save queue state and update display
    SaveQueueToDatabase( );
    UpdateQueueDisplay( );
}

bool TemplateMatchQueueManager::ExecuteSearch(TemplateMatchQueueItem& search_to_run) {
    // Basic validation moved to TMP's ExecuteSearch method

    // Check if another search is already running
    MyDebugAssertFalse(IsSearchRunning( ), "Attempted to execute search %ld while search %ld is already running", search_to_run.database_queue_id, currently_running_id);
    if ( IsSearchRunning( ) ) {
        wxMessageBox("A search is already running. Please wait for it to complete.",
                     "Search Running", wxOK | wxICON_WARNING);
        return false;
    }

    // Use the stored MatchTemplatePanel pointer to execute the search
    MyDebugAssertTrue(match_template_panel_ptr != nullptr, "match_template_panel_ptr is null - cannot execute searches");

    if ( match_template_panel_ptr ) {
        // No callback registration needed - panel has persistent queue_manager member

        if constexpr ( skip_search_execution_for_queue_debugging ) {
            // Debug mode: Mark job as running immediately, then simulate execution time
            QM_LOG_DEBUG("=== DEBUG MODE: Starting search in debug mode ===");
            QM_LOG_DEBUG("Search %ld will be marked as running", search_to_run.database_queue_id);
            QM_LOG_DEBUG("Current queue state BEFORE marking as running:");
            PrintQueueState( );

            // Mark search as running (same as production code path)
            UpdateSearchStatus(search_to_run.database_queue_id, "running");
            currently_running_id = search_to_run.database_queue_id;
            QM_LOG_STATE("Search %ld is now marked as RUNNING (currently_running_id set)", search_to_run.database_queue_id);

            // Update button state when search starts
            UpdateButtonState( );

            // Update displays to show running status
            UpdateQueueDisplay( );
            UpdateAvailableSearchesDisplay( );

            QM_LOG_DEBUG("Queue state AFTER marking as running:");
            PrintQueueState( );

            // Non-blocking wait for 5 seconds to simulate search execution
            // Process events during this time so GUI remains responsive
            QM_LOG_DEBUG(">>> Starting 5 second wait to simulate search execution...");
            QM_LOG_DEBUG(">>> You can now interact with the queue (remove searches, etc.)");

            wxStopWatch timer;
            timer.Start( );
            while ( timer.Time( ) < 5000 ) {
                // Process pending events to keep GUI responsive
                wxYieldIfNeeded( );
                wxMilliSleep(50); // Small sleep to avoid consuming 100% CPU
            }

            QM_LOG_DEBUG(">>> Execution simulation complete for search %ld", search_to_run.database_queue_id);

            // Now simulate search completion
            QM_LOG_DEBUG("=== DEBUG: Simulating search completion ===");
            OnSearchCompleted(search_to_run.database_queue_id, true); // Simulate successful completion
            QM_LOG_DEBUG("=== Search completion simulation done ===");

            return true;
        }
        else {
            // Normal execution path
            bool execution_success = match_template_panel_ptr->ExecuteSearch(&search_to_run);

            if ( execution_success ) {
                // Search executed successfully - now mark it as running and track it
                UpdateSearchStatus(search_to_run.database_queue_id, "running");
                currently_running_id = search_to_run.database_queue_id;
                QM_LOG_SEARCH("Search %ld started successfully and marked as running", search_to_run.database_queue_id);

                // Update button state when search starts
                UpdateButtonState( );

                // Search status will be updated to "complete" when the search finishes via ProcessAllJobsFinished
                return true;
            }
            else {
                QM_LOG_ERROR("Failed to start search %ld", search_to_run.database_queue_id);
                UpdateSearchStatus(search_to_run.database_queue_id, "failed");
                return false;
            }
        }
    }
    else {
        // Critical failure - template panel not available
        MyAssertTrue(false, "Critical error: match_template_panel not available for search execution");
        QM_LOG_ERROR("Error: match_template_panel not available");
        UpdateSearchStatus(search_to_run.database_queue_id, "failed");
        currently_running_id = -1;
    }

    return false;
}

bool TemplateMatchQueueManager::IsSearchRunning( ) const {
    // A search is considered "running" if it's actively processing OR finalizing
    bool is_running = currently_running_id != -1 || search_is_finalizing;
    if ( is_running ) {
        if ( search_is_finalizing ) {
            QM_LOG_STATE("IsSearchRunning: Search is in finalization phase");
        }
        else {
            QM_LOG_STATE("IsSearchRunning: Search %ld is currently running", currently_running_id);
        }
    }
    return is_running;
}

// IsJobRunningStatic removed - no longer needed with unified architecture

void TemplateMatchQueueManager::UpdateSearchStatus(long database_queue_id, const wxString& new_status) {
    MyDebugAssertTrue(database_queue_id >= 0, "Invalid database_queue_id in UpdateSearchStatus: %ld", database_queue_id);
    MyDebugAssertTrue(new_status == "pending" || new_status == "running" || new_status == "complete" || new_status == "failed",
                      "Invalid new_status in UpdateSearchStatus: %s", new_status.mb_str( ).data( ));

    bool found_job = false;
    for ( auto& item : execution_queue ) {
        if ( item.database_queue_id == database_queue_id ) {
            // Validate status transitions
            MyDebugAssertTrue(item.queue_status != new_status, "Attempted to set status to same value: %s", new_status.mb_str( ).data( ));

            // Validate allowed transitions
            if ( item.queue_status == "running" && (new_status == "complete" || new_status == "failed") ) {
                // Valid: running -> complete/failed
                // Allow if currently_running_id matches OR if search is finalizing (currently_running_id may be -1)
                MyDebugAssertTrue(currently_running_id == database_queue_id || search_is_finalizing,
                                  "Status change from running but database_queue_id %ld != currently_running_id %ld and not finalizing",
                                  database_queue_id, currently_running_id);
            }
            else if ( item.queue_status == "partial" && (new_status == "complete" || new_status == "failed") ) {
                // Valid: partial -> complete/failed (when partial search finishes or fails)
                QM_LOG_STATE("Partial search %ld transitioning to %s", database_queue_id, new_status);
            }
            else if ( item.queue_status == "pending" && new_status == "running" ) {
                // Valid: pending -> running
                MyDebugAssertFalse(IsSearchRunning( ), "Cannot start search %ld when search %ld is already running", database_queue_id, currently_running_id);
            }
            else if ( item.queue_status == "pending" && new_status == "failed" ) {
                // Valid: pending -> failed (when search fails to start)
                QM_LOG_STATE("Search %ld failed to start, marking as failed", database_queue_id);
            }
            else if ( item.queue_status == "failed" && new_status == "running" ) {
                // Valid: failed -> running (when resuming a failed search)
                MyDebugAssertFalse(IsSearchRunning( ), "Cannot resume failed search %ld when search %ld is already running", database_queue_id, currently_running_id);
                QM_LOG_STATE("Resuming failed search %ld, marking as running", database_queue_id);
            }
            else if ( item.queue_status == "partial" && new_status == "running" ) {
                // Valid: partial -> running (when resuming a partially complete search)
                MyDebugAssertFalse(IsSearchRunning( ), "Cannot resume partial search %ld when search %ld is already running", database_queue_id, currently_running_id);
                QM_LOG_STATE("Resuming partial search %ld, marking as running", database_queue_id);
            }
            else {
                MyDebugAssertTrue(false, "Invalid status transition: %s -> %s for search %ld",
                                  item.queue_status.mb_str( ).data( ), new_status.mb_str( ).data( ), database_queue_id);
            }

            item.queue_status = new_status;
            found_job         = true;
            break;
        }
    }

    MyDebugAssertTrue(found_job, "Search with database_queue_id %ld not found in queue for status update", database_queue_id);

    UpdateQueueDisplay( );
    SaveQueueToDatabase( );
}

// Static methods removed - no longer needed with unified architecture

// GetNextPendingJob method removed - functionality moved into RunNextSearch

bool TemplateMatchQueueManager::HasPendingSearches( ) {
    for ( const auto& item : execution_queue ) {
        if ( item.queue_status == "pending" ) {
            return true;
        }
    }
    return false;
}

void TemplateMatchQueueManager::OnRunSelectedClick(wxCommandEvent& event) {
    QM_LOG_METHOD_ENTRY("OnRunSelectedClick");
    RunNextSearch( );
}

// OnRunAllClick removed - use Run Selected with multi-selection instead

void TemplateMatchQueueManager::OnClearQueueClick(wxCommandEvent& event) {
    QM_LOG_METHOD_ENTRY("OnClearQueueClick");

    wxMessageDialog dialog(this, "Clear all pending searches from the queue?",
                           "Confirm Clear", wxYES_NO | wxICON_QUESTION);
    int             result = dialog.ShowModal( );
    QM_LOG_UI("Clear queue confirmation dialog result: %s", (result == wxID_YES) ? "YES" : "NO");

    if ( result == wxID_YES ) {
        ClearExecutionQueue( );
    }
}

void TemplateMatchQueueManager::OnHideCompletedToggle(wxCommandEvent& event) {
    hide_completed_searches = event.IsChecked( );
    QM_LOG_UI("Hide completed searches toggled to: %s", hide_completed_searches ? "true" : "false");

    // Refresh the available jobs display with the new filter
    UpdateAvailableSearchesDisplay( );
}

void TemplateMatchQueueManager::UpdateSortIndicators(int previous_column) {
    // Column headers with sort indicator for sortable columns (0, 1, 3, 5, 6, 7)
    // Non-sortable columns (2, 4, 8) don't get the indicator
    wxString       sortable_indicator = wxString::FromUTF8(" ⇅");
    const wxString column_headers[9]  = {
             wxString::FromUTF8("Queue ID") + sortable_indicator, // 0 - sortable
             wxString::FromUTF8("Search ID") + sortable_indicator, // 1 - sortable
             "Search Name", // 2 - not sortable
             wxString::FromUTF8("Status") + sortable_indicator, // 3 - sortable
             "Progress", // 4 - not sortable
             wxString::FromUTF8("Total Time") + sortable_indicator, // 5 - sortable
             wxString::FromUTF8("Avg Time/Job") + sortable_indicator, // 6 - sortable
             wxString::FromUTF8("Latest") + sortable_indicator, // 7 - sortable
             "CLI Args" // 8 - not sortable
    };

    // Reset previous column to base header (if different from current and was sortable)
    if ( previous_column != available_queue_sort_column && previous_column >= 0 && previous_column < 9 ) {
        // Only update if it was a sortable column (0, 1, 3, 5, 6, 7)
        if ( previous_column == 0 || previous_column == 1 || previous_column == 3 ||
             previous_column == 5 || previous_column == 6 || previous_column == 7 ) {
            wxListItem col_info;
            col_info.SetMask(wxLIST_MASK_TEXT);
            col_info.SetText(column_headers[previous_column]);
            available_searches_ctrl->SetColumn(previous_column, col_info);
        }
    }

    // Add direction indicator to current column (⇅ plus ^ or v)
    // Only sortable columns (0, 1, 3, 5, 6, 7) should reach here due to OnAvailableQueueColumnClick validation
    if ( available_queue_sort_column >= 0 && available_queue_sort_column < 9 ) {
        wxString header = column_headers[available_queue_sort_column];
        header += available_queue_sort_ascending ? " ^" : " v";

        wxListItem col_info;
        col_info.SetMask(wxLIST_MASK_TEXT);
        col_info.SetText(header);
        available_searches_ctrl->SetColumn(available_queue_sort_column, col_info);
    }
}

void TemplateMatchQueueManager::OnAvailableSearchesSelectionChanged(wxListEvent& event) {
    QM_LOG_METHOD_ENTRY("OnAvailableSearchesSelectionChanged");

    // Check if any items are selected in available jobs table
    bool                    has_available_selection = false;
    long                    first_selected_row      = available_searches_ctrl->GetNextItem(-1, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
    TemplateMatchQueueItem* first_selected_item     = nullptr;

    // Clear execution queue selections to prevent multi-select confusion
    if ( first_selected_row != -1 && queue_list_ctrl ) {
        for ( int i = 0; i < queue_list_ctrl->GetItemCount( ); i++ ) {
            queue_list_ctrl->SetItemState(i, 0, wxLIST_STATE_SELECTED);
        }
    }

    if ( first_selected_row != -1 ) {
        has_available_selection = true;

        // Get the database_queue_id from the item data
        long database_queue_id = available_searches_ctrl->GetItemData(first_selected_row);

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

    // Enable buttons based on available jobs selection
    add_to_queue_button->Enable(has_available_selection);
    remove_selected_button->Enable(has_available_selection); // Allow removing items from available queue

    QM_LOG_UI("OnAvailableSearchesSelectionChanged: remove_selected_button->Enable(%s)",
              has_available_selection ? "true" : "false");

    // Populate the GUI with the first selected item's parameters
    // Note: We don't check gui_update_frozen here because we want to allow editing of pending items
    // The freeze is only meant to prevent interference during search execution
    if ( first_selected_item && match_template_panel_ptr ) {
        // Only populate GUI if the item is pending (editable) or we're not currently running a search
        bool is_editable = (first_selected_item->queue_status == "pending" ||
                            first_selected_item->queue_status == "failed" ||
                            first_selected_item->queue_status == "partial");

        if ( is_editable || ! gui_update_frozen ) {
            QM_LOG_UI("Populating GUI from available queue item: %s (status: %s)",
                      first_selected_item->search_name, first_selected_item->queue_status);
            match_template_panel_ptr->PopulateGuiFromQueueItem(*first_selected_item, true);

            // Track which item was populated and enable/disable update button
            last_populated_queue_id = first_selected_item->database_queue_id;
            UpdateButtonState( );
        }
    }

    QM_LOG_UI("Available jobs selection changed - Add to Queue button %s",
              has_available_selection ? "enabled" : "disabled");
    QM_LOG_METHOD_EXIT("OnAvailableSearchesSelectionChanged");
}

void TemplateMatchQueueManager::OnAvailableQueueColumnClick(wxListEvent& event) {
    // Only handle clicks on available_searches_ctrl, ignore execution queue clicks
    if ( event.GetEventObject( ) != available_searches_ctrl ) {
        event.Skip( );
        return;
    }

    int clicked_column = event.GetColumn( );

    // Only columns 0 (Queue ID), 1 (Search ID), 3 (Status), 5 (Total Time), 6 (Avg Time/Job), 7 (Latest) are sortable
    // Ignore clicks on 2 (Search Name), 4 (Progress), 8 (CLI Args)
    if ( clicked_column == 2 || clicked_column == 4 || clicked_column == 8 ) {
        QM_LOG_UI("Ignoring click on non-sortable column %d", clicked_column);
        return;
    }

    int previous_column = available_queue_sort_column;

    // Toggle sort direction if clicking same column, otherwise default to ascending
    if ( clicked_column == available_queue_sort_column ) {
        available_queue_sort_ascending = ! available_queue_sort_ascending;
    }
    else {
        available_queue_sort_column    = clicked_column;
        available_queue_sort_ascending = true;
    }

    QM_LOG_UI("Available queue column %d clicked - sorting %s",
              clicked_column, available_queue_sort_ascending ? "ascending" : "descending");

    // Update column headers with sort indicators
    UpdateSortIndicators(previous_column);

    // Refresh display with new sort order
    UpdateAvailableSearchesDisplay( );
}

void TemplateMatchQueueManager::SortAvailableItems(std::vector<TemplateMatchQueueItem*>& items) {
    // Sort based on current column and direction
    std::sort(items.begin( ), items.end( ),
              [this](const TemplateMatchQueueItem* a, const TemplateMatchQueueItem* b) -> bool {
                  bool ascending = available_queue_sort_ascending;
                  int  result    = 0;

                  switch ( available_queue_sort_column ) {
                      case 0: // Queue ID
                          result = (a->database_queue_id < b->database_queue_id) ? -1 : (a->database_queue_id > b->database_queue_id) ? 1
                                                                                                                                      : 0;
                          break;

                      case 1: // Search ID (empty always sorts to end regardless of direction)
                      {
                          bool a_is_empty = (a->search_id <= 0);
                          bool b_is_empty = (b->search_id <= 0);

                          if ( a_is_empty && b_is_empty ) {
                              // Both empty - treat as equal
                              result = 0;
                          }
                          else if ( a_is_empty || b_is_empty ) {
                              // One is empty - ensure empty always goes to end
                              // When ascending: empty gets result=1 (after), becomes false (stays after)
                              // When descending: empty gets result=-1 (before inverted), becomes false (stays after)
                              if ( a_is_empty ) {
                                  result = ascending ? 1 : -1;
                              }
                              else {
                                  result = ascending ? -1 : 1;
                              }
                          }
                          else {
                              // Both valid - normal comparison
                              result = (a->search_id < b->search_id) ? -1 : (a->search_id > b->search_id) ? 1
                                                                                                          : 0;
                          }
                          break;
                      }

                      case 2: // Search Name (case-insensitive)
                          result = a->search_name.CmpNoCase(b->search_name);
                          break;

                      case 3: // Status (running > pending > failed > complete when ascending)
                      {
                          auto get_status_priority = [](const wxString& status) -> int {
                              if ( status == "running" )
                                  return 3;
                              if ( status == "pending" )
                                  return 2;
                              if ( status == "failed" )
                                  return 1;
                              if ( status == "complete" )
                                  return 0;
                              return -1; // Unknown status sorts last
                          };

                          int a_priority = get_status_priority(a->queue_status);
                          int b_priority = get_status_priority(b->queue_status);
                          // Invert comparison for status so ascending shows running first
                          result = (a_priority > b_priority) ? -1 : (a_priority < b_priority) ? 1
                                                                                              : 0;
                          break;
                      }

                      case 4: // Progress (parse "X/Y" format and compare as percentage)
                      {
                          SearchCompletionInfo a_completion = GetSearchCompletionInfo(a->database_queue_id);
                          SearchCompletionInfo b_completion = GetSearchCompletionInfo(b->database_queue_id);

                          float a_percent = a_completion.GetCompletionPercentage( );
                          float b_percent = b_completion.GetCompletionPercentage( );

                          result = (a_percent < b_percent) ? -1 : (a_percent > b_percent) ? 1
                                                                                          : 0;
                          break;
                      }

                      case 5: // Total Time (numeric comparison, empty values sort to end)
                      {
                          SearchTimingInfo a_timing = GetSearchTimingInfo(a->search_id);
                          SearchTimingInfo b_timing = GetSearchTimingInfo(b->search_id);

                          bool a_has_timing = (a_timing.completed_count > 0);
                          bool b_has_timing = (b_timing.completed_count > 0);

                          if ( ! a_has_timing && ! b_has_timing ) {
                              result = 0; // Both empty
                          }
                          else if ( ! a_has_timing || ! b_has_timing ) {
                              // Empty always sorts to end
                              result = a_has_timing ? (ascending ? -1 : 1) : (ascending ? 1 : -1);
                          }
                          else {
                              // Both have timing - compare values
                              result = (a_timing.total_elapsed_seconds < b_timing.total_elapsed_seconds) ? -1 : (a_timing.total_elapsed_seconds > b_timing.total_elapsed_seconds) ? 1
                                                                                                                                                                                  : 0;
                          }
                          break;
                      }

                      case 6: // Avg Time/Job (numeric comparison, empty values sort to end)
                      {
                          SearchTimingInfo a_timing = GetSearchTimingInfo(a->search_id);
                          SearchTimingInfo b_timing = GetSearchTimingInfo(b->search_id);

                          bool a_has_timing = (a_timing.completed_count > 0);
                          bool b_has_timing = (b_timing.completed_count > 0);

                          if ( ! a_has_timing && ! b_has_timing ) {
                              result = 0; // Both empty
                          }
                          else if ( ! a_has_timing || ! b_has_timing ) {
                              // Empty always sorts to end
                              result = a_has_timing ? (ascending ? -1 : 1) : (ascending ? 1 : -1);
                          }
                          else {
                              // Both have timing - compare values
                              result = (a_timing.avg_elapsed_seconds < b_timing.avg_elapsed_seconds) ? -1 : (a_timing.avg_elapsed_seconds > b_timing.avg_elapsed_seconds) ? 1
                                                                                                                                                                          : 0;
                          }
                          break;
                      }

                      case 7: // Latest completion datetime (string comparison, empty values sort to end)
                      {
                          SearchTimingInfo a_timing = GetSearchTimingInfo(a->search_id);
                          SearchTimingInfo b_timing = GetSearchTimingInfo(b->search_id);

                          bool a_has_datetime = ! a_timing.latest_datetime.IsEmpty( );
                          bool b_has_datetime = ! b_timing.latest_datetime.IsEmpty( );

                          if ( ! a_has_datetime && ! b_has_datetime ) {
                              result = 0; // Both empty
                          }
                          else if ( ! a_has_datetime || ! b_has_datetime ) {
                              // Empty always sorts to end
                              result = a_has_datetime ? (ascending ? -1 : 1) : (ascending ? 1 : -1);
                          }
                          else {
                              // Both have datetime - lexicographic comparison (YYYY-MM-DD HH:MM format sorts correctly)
                              result = a_timing.latest_datetime.Cmp(b_timing.latest_datetime);
                          }
                          break;
                      }

                      case 8: // CLI Args (string comparison)
                          result = a->custom_cli_args.Cmp(b->custom_cli_args);
                          break;

                      default:
                          result = 0; // Unknown column, maintain current order
                          break;
                  }

                  // Apply sort direction
                  if ( ascending ) {
                      return result < 0;
                  }
                  else {
                      return result > 0;
                  }
              });
}

void TemplateMatchQueueManager::UpdateButtonState( ) {
    // Enable update button only if:
    // 1. We have a populated item ID
    // 2. The item exists and has status "pending"

    if ( last_populated_queue_id <= 0 ) {
        update_selected_button->Enable(false);
        return;
    }

    // Find the item in either queue
    TemplateMatchQueueItem* item = nullptr;
    for ( auto& queue_item : execution_queue ) {
        if ( queue_item.database_queue_id == last_populated_queue_id ) {
            item = &queue_item;
            break;
        }
    }
    if ( ! item ) {
        for ( auto& queue_item : available_queue ) {
            if ( queue_item.database_queue_id == last_populated_queue_id ) {
                item = &queue_item;
                break;
            }
        }
    }

    // Enable button only if item found and status is pending
    bool should_enable = (item != nullptr && item->queue_status == "pending");
    update_selected_button->Enable(should_enable);

    if ( should_enable ) {
        QM_LOG_UI("Update button enabled for pending item %ld", last_populated_queue_id);
    }
    else if ( item ) {
        QM_LOG_UI("Update button disabled - item %ld has status '%s'",
                  last_populated_queue_id, item->queue_status);
    }
    else {
        QM_LOG_UI("Update button disabled - item %ld not found", last_populated_queue_id);
    }
}

void TemplateMatchQueueManager::OnUpdateSelectedClick(wxCommandEvent& event) {
    QM_LOG_METHOD_ENTRY("OnUpdateSelectedClick");

    // Enforce single selection from either queue
    long                    selected_count    = 0;
    long                    selected_queue_id = -1;
    TemplateMatchQueueItem* item_to_edit      = nullptr;

    // Check execution queue for selections
    long item = queue_list_ctrl->GetNextItem(-1, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
    while ( item != -1 ) {
        selected_count++;
        selected_queue_id = queue_list_ctrl->GetItemData(item);
        item              = queue_list_ctrl->GetNextItem(item, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
    }

    // Check available queue for selections
    item = available_searches_ctrl->GetNextItem(-1, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
    while ( item != -1 ) {
        selected_count++;
        selected_queue_id = available_searches_ctrl->GetItemData(item);
        item              = available_searches_ctrl->GetNextItem(item, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
    }

    // Validate selection count
    if ( selected_count == 0 ) {
        wxMessageBox("Please select a queue item to edit", "No Selection",
                     wxOK | wxICON_INFORMATION);
        return;
    }
    if ( selected_count > 1 ) {
        wxMessageBox("Please select only one queue item to edit", "Multiple Selection",
                     wxOK | wxICON_INFORMATION);
        return;
    }

    // Find the selected item
    for ( auto& qi : execution_queue ) {
        if ( qi.database_queue_id == selected_queue_id ) {
            item_to_edit = &qi;
            break;
        }
    }
    if ( ! item_to_edit ) {
        for ( auto& qi : available_queue ) {
            if ( qi.database_queue_id == selected_queue_id ) {
                item_to_edit = &qi;
                break;
            }
        }
    }

    if ( ! item_to_edit ) {
        wxMessageBox("Could not find selected queue item", "Error", wxOK | wxICON_ERROR);
        return;
    }

    // Check if item is editable (no results written yet)
    if ( item_to_edit->search_id > 0 ) {
        wxMessageBox("Cannot edit queue items that have started execution (have results)",
                     "Cannot Edit", wxOK | wxICON_WARNING);
        return;
    }

    // Store editing state and switch to editor view
    editing_queue_id = item_to_edit->database_queue_id;

    // CRITICAL: Fill combo boxes BEFORE populating from item
    // Otherwise combo boxes are empty and SetSelection causes segfault
    editor_panel->FillComboBoxes( );

    editor_panel->PopulateFromQueueItem(*item_to_edit);
    SwitchToEditorView( );
}

void TemplateMatchQueueManager::OnRemoveSelectedClick(wxCommandEvent& event) {
    QM_LOG_METHOD_ENTRY("OnRemoveSelectedClick");

    // Check which queue has selections - execution queue or available queue
    bool has_execution_selection = (queue_list_ctrl->GetSelectedItemCount( ) > 0);
    bool has_available_selection = (available_searches_ctrl && available_searches_ctrl->GetSelectedItemCount( ) > 0);

    QM_LOG_UI("Remove Selected clicked: execution_selection=%s, available_selection=%s",
              has_execution_selection ? "true" : "false",
              has_available_selection ? "true" : "false");

    if ( has_execution_selection ) {
        QM_LOG_UI("Processing EXQ deletion request");

        // Get all selected items from execution queue using GetItemData()
        std::vector<long> selected_database_ids;
        long              selected_row = queue_list_ctrl->GetNextItem(-1, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);

        while ( selected_row != -1 ) {
            // CRITICAL: Use GetItemData() to get database_queue_id, NOT the row number!
            // Row numbers don't match execution_queue indices because of available items
            long database_queue_id = queue_list_ctrl->GetItemData(selected_row);
            selected_database_ids.push_back(database_queue_id);
            QM_LOG_UI("Found selected EXQ item: row=%ld, queue_id=%ld", selected_row, database_queue_id);
            selected_row = queue_list_ctrl->GetNextItem(selected_row, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
        }

        QM_LOG_UI("Total EXQ items selected: %zu", selected_database_ids.size( ));

        if ( selected_database_ids.empty( ) ) {
            QM_LOG_UI("No items selected - returning");
            return; // Nothing selected
        }

        // Build confirmation message - use queue_id since we're removing queue items, not searches
        wxString message;
        if ( selected_database_ids.size( ) == 1 ) {
            message = wxString::Format("Remove queue item %ld from execution queue?", selected_database_ids[0]);
        }
        else {
            message = wxString::Format("Remove %d selected queue items from execution queue?", int(selected_database_ids.size( )));
        }

        QM_LOG_UI("Showing confirmation dialog: %s", message.mb_str( ).data( ));
        wxMessageDialog dialog(this, message, "Confirm Remove", wxYES_NO | wxICON_QUESTION);
        int             result = dialog.ShowModal( );
        QM_LOG_UI("Dialog result: %s", (result == wxID_YES) ? "YES" : "NO");

        if ( result == wxID_YES ) {
            // Find execution_queue indices for each database_queue_id and remove
            // Must work backwards to maintain index validity during deletion
            for ( auto it = selected_database_ids.rbegin( ); it != selected_database_ids.rend( ); ++it ) {
                long database_queue_id = *it;
                QM_LOG_UI("Processing queue_id %ld for removal from execution queue", database_queue_id);

                // Find this item in execution_queue by database_queue_id
                int found_index = -1;
                for ( size_t i = 0; i < execution_queue.size( ); i++ ) {
                    if ( execution_queue[i].database_queue_id == database_queue_id &&
                         execution_queue[i].queue_order >= 0 ) { // Only remove from execution queue
                        found_index = int(i);
                        QM_LOG_UI("Found queue_id %ld at execution_queue[%d]", database_queue_id, found_index);
                        break;
                    }
                }

                if ( found_index >= 0 ) {
                    RemoveFromExecutionQueue(found_index);
                    QM_LOG_UI("Removed queue_id %ld from execution queue", database_queue_id);
                }
                else {
                    QM_LOG_UI("WARNING: queue_id %ld not found in execution queue!", database_queue_id);
                }
            }
            QM_LOG_UI("EXQ removal complete");
        }
    }
    else if ( has_available_selection ) {
        QM_LOG_UI("Processing AVQ deletion request");

        // Get all selected items from available queue
        std::vector<long> selected_database_ids;
        long              selected_row = available_searches_ctrl->GetNextItem(-1, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);

        while ( selected_row != -1 ) {
            long database_queue_id = available_searches_ctrl->GetItemData(selected_row);
            selected_database_ids.push_back(database_queue_id);
            QM_LOG_UI("Found selected AVQ item: row=%ld, queue_id=%ld", selected_row, database_queue_id);
            selected_row = available_searches_ctrl->GetNextItem(selected_row, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
        }

        QM_LOG_UI("Total AVQ items selected: %zu", selected_database_ids.size( ));

        if ( selected_database_ids.empty( ) ) {
            QM_LOG_UI("No items selected - returning");
            return; // Nothing selected
        }

        // Build confirmation message - use queue_id since only items without results can be deleted
        wxString message;
        if ( selected_database_ids.size( ) == 1 ) {
            message = wxString::Format("Permanently delete queue item %ld?", selected_database_ids[0]);
        }
        else {
            message = wxString::Format("Permanently delete %d selected queue items?", int(selected_database_ids.size( )));
        }

        QM_LOG_UI("Showing confirmation dialog: %s", message.mb_str( ).data( ));
        wxMessageDialog dialog(this, message, "Confirm Delete", wxYES_NO | wxICON_QUESTION);
        int             result = dialog.ShowModal( );
        QM_LOG_UI("Dialog result: %s", (result == wxID_YES) ? "YES" : "NO");

        if ( result == wxID_YES ) {
            // CRITICAL RULE: Only delete items with search_id == -1 (no results exist)
            // Use main_frame member variable, not GetParent() cast
            if ( main_frame && main_frame->current_project.is_open ) {
                int deletion_count  = 0;
                int protected_count = 0;

                for ( long database_queue_id : selected_database_ids ) {
                    QM_LOG_UI("Processing queue_id %ld for deletion", database_queue_id);

                    // Find the item - check execution_queue (for items with queue_order < 0) and available_queue
                    TemplateMatchQueueItem* found_item = nullptr;

                    // First check execution_queue for items with queue_order < 0 (available items)
                    for ( auto& item : execution_queue ) {
                        if ( item.database_queue_id == database_queue_id && item.queue_order < 0 ) {
                            found_item = &item;
                            QM_LOG_UI("Found in execution_queue with queue_order < 0: search_id=%ld", item.search_id);
                            break;
                        }
                    }

                    // If not found, check available_queue
                    if ( ! found_item ) {
                        for ( auto& item : available_queue ) {
                            if ( item.database_queue_id == database_queue_id ) {
                                found_item = &item;
                                QM_LOG_UI("Found in available_queue: search_id=%ld", item.search_id);
                                break;
                            }
                        }
                    }

                    if ( found_item ) {
                        // Only delete if search_id == -1 (no results table entry exists)
                        if ( found_item->search_id == -1 ) {
                            // Safe to delete - no results exist
                            QM_LOG_UI("Deleting queue_id %ld from database (search_id=-1)", database_queue_id);
                            main_frame->current_project.database.RemoveFromQueue(database_queue_id);

                            // Remove from whichever deque it was in
                            execution_queue.erase(
                                    std::remove_if(execution_queue.begin( ), execution_queue.end( ),
                                                   [database_queue_id](const TemplateMatchQueueItem& item) {
                                                       return item.database_queue_id == database_queue_id;
                                                   }),
                                    execution_queue.end( ));
                            available_queue.erase(
                                    std::remove_if(available_queue.begin( ), available_queue.end( ),
                                                   [database_queue_id](const TemplateMatchQueueItem& item) {
                                                       return item.database_queue_id == database_queue_id;
                                                   }),
                                    available_queue.end( ));

                            deletion_count++;
                        }
                        else {
                            // Has search_id - results exist, NEVER delete
                            QM_LOG_UI("Protected queue_id %ld (search_id=%ld != -1)", database_queue_id, found_item->search_id);
                            protected_count++;
                        }
                    }
                    else {
                        QM_LOG_UI("WARNING: queue_id %ld not found in execution_queue or available_queue!", database_queue_id);
                    }
                }

                QM_LOG_UI("Deletion complete: deleted=%d, protected=%d", deletion_count, protected_count);

                // Inform user if some items were protected
                if ( protected_count > 0 ) {
                    wxString warning = wxString::Format(
                            "%d queue item(s) with results (search_id set) were protected from deletion.\n"
                            "Only queue items without results (search_id = -1) can be permanently deleted.\n"
                            "%d queue item(s) were deleted.",
                            protected_count, deletion_count);
                    wxMessageBox(warning, "Deletion Protection", wxOK | wxICON_INFORMATION);
                }
            }
            UpdateQueueDisplay( );
        }
    }
}

void TemplateMatchQueueManager::OnSelectionChanged(wxListEvent& event) {
    // Only handle events from the execution queue control
    if ( event.GetEventObject( ) != execution_queue_ctrl ) {
        event.Skip( );
        return;
    }

    // Clean up any stale drag state - selection changes often happen after cancelled drags
    CleanupDragState( );

    // Validate GUI components are available
    MyDebugAssertTrue(remove_selected_button != nullptr, "remove_selected_button is null in OnSelectionChanged");
    MyDebugAssertTrue(run_selected_button != nullptr, "run_selected_button is null in OnSelectionChanged");

    // Check if any items are selected in the execution queue
    bool has_selection = (queue_list_ctrl->GetSelectedItemCount( ) > 0);

    // Clear available queue selections to prevent multi-select confusion
    if ( has_selection && available_searches_ctrl ) {
        for ( int i = 0; i < available_searches_ctrl->GetItemCount( ); i++ ) {
            available_searches_ctrl->SetItemState(i, 0, wxLIST_STATE_SELECTED);
        }
    }

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

    // Check if there's a search at priority 0 ready to run (not complete)
    bool has_job_at_priority_0 = false;
    for ( const auto& search : execution_queue ) {
        if ( search.queue_order == 0 && search.queue_status != "complete" ) {
            has_job_at_priority_0 = true;
            break;
        }
    }

    // Check if there are available queue selections
    bool has_available_selection = (available_searches_ctrl && available_searches_ctrl->GetSelectedItemCount( ) > 0);

    // Enable controls based on selection and search status
    // Remove button should work for both execution and available queue selections
    bool enable_remove = (has_selection && ! any_running) || has_available_selection;
    remove_selected_button->Enable(enable_remove);
    run_selected_button->Enable(has_job_at_priority_0 && ! IsSearchRunning( ));
    remove_from_queue_button->Enable(has_selection && ! any_running);

    QM_LOG_UI("OnSelectionChanged: remove_selected_button->Enable(%s) [exec_sel=%s, avail_sel=%s, any_running=%s]",
              enable_remove ? "true" : "false",
              has_selection ? "true" : "false",
              has_available_selection ? "true" : "false",
              any_running ? "true" : "false");

    // Populate the GUI with the first selected item's parameters
    // Note: We don't check gui_update_frozen here for pending items because we want to allow editing
    // The freeze is only meant to prevent interference during search execution
    if ( has_selection && first_selected_index >= 0 && match_template_panel_ptr ) {
        const auto& selected_item = execution_queue[first_selected_index];

        // Only populate GUI if the item is pending (editable) or we're not currently running a search
        bool is_editable = (selected_item.queue_status == "pending" ||
                            selected_item.queue_status == "failed" ||
                            selected_item.queue_status == "partial");

        if ( is_editable || ! gui_update_frozen ) {
            QM_LOG_UI("Populating GUI from execution queue item %d (status: %s)",
                      first_selected_index, selected_item.queue_status.mb_str( ).data( ));
            match_template_panel_ptr->PopulateGuiFromQueueItem(selected_item, true);

            // Track which item was populated and enable/disable update button
            last_populated_queue_id = selected_item.database_queue_id;
            UpdateButtonState( );
        }
    }

    event.Skip( );
}

void TemplateMatchQueueManager::OnAddToQueueClick(wxCommandEvent& event) {
    QM_LOG_METHOD_ENTRY("OnAddToQueueClick");

    // Get selected jobs from available jobs table
    std::vector<long> selected_database_ids;
    long              selected_row = available_searches_ctrl->GetNextItem(-1, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
    while ( selected_row != -1 ) {
        // Get the database_queue_id from the item data
        long database_queue_id = available_searches_ctrl->GetItemData(selected_row);
        selected_database_ids.push_back(database_queue_id);
        selected_row = available_searches_ctrl->GetNextItem(selected_row, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
    }

    if ( selected_database_ids.empty( ) ) {
        wxMessageBox("Please select searches from the Available Searches table to add to the execution queue.",
                     "No Selection", wxOK | wxICON_WARNING);
        return;
    }

    // Count jobs in execution queue to determine next position
    int next_queue_order = 0;
    for ( const auto& search : execution_queue ) {
        if ( search.queue_order >= 0 ) { // Count searches in execution queue
            next_queue_order++;
        }
    }

    // Move selected available jobs to execution queue
    std::vector<long>     selected_job_ids;
    std::vector<wxString> blocked_jobs; // Track jobs that can't be moved

    // Helper lambda to process a search for moving to execution queue
    auto process_search_for_queue = [&](TemplateMatchQueueItem* search, bool needs_move_from_available, size_t available_index) {
        // Check if this search is selected (by database_queue_id)
        for ( long selected_id : selected_database_ids ) {
            if ( search->database_queue_id == selected_id ) {
                QM_LOG_DEBUG("Processing search %ld with status '%s' for adding to queue",
                             search->database_queue_id, search->queue_status);
                // Allow pending, failed, or partial searches to be moved to execution queue
                // Do NOT allow running or complete searches
                if ( search->queue_status == "pending" ||
                     search->queue_status == "failed" ||
                     search->queue_status == "partial" ) {
                    QM_LOG_DEBUG("Search %ld allowed - status '%s' can be queued",
                                 search->database_queue_id, search->queue_status);
                    search->queue_order = next_queue_order++;
                    selected_job_ids.push_back(search->database_queue_id);

                    if ( needs_move_from_available ) {
                        // Move from available_queue to execution_queue
                        execution_queue.push_back(*search);
                        available_queue.erase(available_queue.begin( ) + available_index);
                        return true; // Indicate we removed an item
                    }
                }
                else {
                    QM_LOG_DEBUG("Search %ld BLOCKED - status '%s' cannot be queued",
                                 search->database_queue_id, search->queue_status);
                    blocked_jobs.push_back(wxString::Format("Search %ld (%s)",
                                                            search->database_queue_id,
                                                            search->queue_status));
                }
                break;
            }
        }
        return false;
    };

    // First, check jobs from execution_queue with queue_order < 0
    for ( size_t i = 0; i < execution_queue.size( ); ++i ) {
        if ( execution_queue[i].queue_order < 0 ) { // This is an available search
            // Skip completed searches if hide_completed_searches is enabled (must match display logic)
            if ( hide_completed_searches && execution_queue[i].queue_status == "complete" ) {
                continue;
            }
            process_search_for_queue(&execution_queue[i], false, 0);
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
            // Skip completed searches if hide_completed_searches is enabled (must match display logic)
            if ( hide_completed_searches && available_queue[i].queue_status == "complete" ) {
                continue;
            }
            if ( process_search_for_queue(&available_queue[i], true, i) ) {
                i--; // Adjust index after erase
            }
        }
    }

    // Show warning if some jobs were blocked
    if ( ! blocked_jobs.empty( ) ) {
        wxString message = "The following searches cannot be added to execution queue:\n\n";
        for ( const auto& job : blocked_jobs ) {
            message += "• " + job + "\n";
        }
        message += "\nOnly pending, failed, or partial searches can be queued.\n";
        message += "Running and complete searches cannot be re-queued.";
        wxMessageBox(message, "Some Searches Not Added", wxOK | wxICON_INFORMATION);
    }

    if ( ! selected_job_ids.empty( ) ) {
        SaveQueueToDatabase( );
        UpdateQueueDisplay( );
    }
}

void TemplateMatchQueueManager::OnRemoveFromQueueClick(wxCommandEvent& event) {
    QM_LOG_METHOD_ENTRY("OnRemoveFromQueueClick");

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
        wxMessageBox("Please select searches to remove from the execution queue.",
                     "No Selection", wxOK | wxICON_WARNING);
        return;
    }

    wxMessageDialog dialog(this,
                           wxString::Format("Remove %d selected search(es) from execution queue?\n\n"
                                            "They will be moved to Available Searches.",
                                            int(selected_database_ids.size( ))),
                           "Confirm Remove from Queue", wxYES_NO | wxICON_QUESTION);

    if ( dialog.ShowModal( ) == wxID_YES ) {
        if constexpr ( skip_search_execution_for_queue_debugging ) {
            QM_LOG_DEBUG("=== REMOVING JOBS FROM EXECUTION QUEUE ===");
            QM_LOG_DEBUG("Queue state BEFORE removal:");
            PrintQueueState( );
        }

        // Get database_queue_ids for selected items (already collected above)
        std::vector<long> selected_ids = selected_database_ids;

        if constexpr ( skip_search_execution_for_queue_debugging ) {
            for ( long id : selected_ids ) {
                for ( const auto& job : execution_queue ) {
                    if ( job.database_queue_id == id ) {
                        QM_LOG_DEBUG("Selected job: ID=%ld, queue_order=%d",
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
                        QM_LOG_DEBUG("Moving job %ld from queue_order %d to -1 (available)",
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
            QM_LOG_DEBUG("Queue state AFTER removal and renumbering:");
            PrintQueueState( );
        }

        UpdateQueueDisplay( );
        SaveQueueToDatabase( );
        QM_LOG_STATE("Removed %zu jobs from execution queue", selected_ids.size( ));

        if constexpr ( skip_search_execution_for_queue_debugging ) {
            QM_LOG_DEBUG("=== REMOVAL COMPLETE ===");
        }
    }
}

void TemplateMatchQueueManager::OnBeginDrag(wxListEvent& event) {
    // Note: This event handler exists for compatibility but manual drag-and-drop
    // is now handled through mouse events (OnMouseLeftDown, OnMouseMotion, OnMouseLeftUp)
    QM_LOG_UI("OnBeginDrag called - using manual mouse drag implementation");
}

void TemplateMatchQueueManager::CleanupDragState( ) {
    if ( drag_in_progress ) {
        drag_in_progress  = false;
        mouse_down        = false;
        dragged_row       = -1;
        dragged_search_id = -1;
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
            QM_LOG_ERROR("CRASH RECOVERY: Found orphaned running job %ld, marking as failed", item.database_queue_id);
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
            QM_LOG_ERROR("CRASH RECOVERY: Found orphaned running job %ld, marking as failed", item.database_queue_id);
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
        QM_LOG_STATE("CRASH RECOVERY: Saving corrected job statuses to database");
        SaveQueueToDatabase( );
    }

    // Auto-populate available searches that have no queue entries (orphaned searches)
    PopulateAvailableSearchesNotInQueueFromDatabase( );

    // Update display
    UpdateQueueDisplay( );

    // Set initial sort indicator (Queue ID ascending by default)
    UpdateSortIndicators(-1);
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
    int      count     = main_frame->current_project.database.ReturnSingleIntFromSelectCommand(check_sql);

    if ( count == 0 ) {
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

    // Update all fields (no longer need to split into parts after removing 8 CTF fields)
    wxString update_sql_part1 = wxString::Format(
            "UPDATE TEMPLATE_MATCH_QUEUE SET "
            "SEARCH_NAME = '%s', "
            "IMAGE_GROUP_ID = %d, "
            "REFERENCE_VOLUME_ASSET_ID = %d, "
            "RUN_PROFILE_ID = %d, "
            "USE_GPU = %d, "
            "USE_FAST_FFT = %d, "
            "SYMMETRY = '%s'",
            item.search_name.ToUTF8( ).data( ),
            item.image_group_id,
            item.reference_volume_asset_id,
            item.run_profile_id,
            item.use_gpu ? 1 : 0,
            item.use_fast_fft ? 1 : 0,
            item.symmetry.ToUTF8( ).data( ));

    // Part 2: Update remaining fields plus WHERE clause
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
            item.custom_cli_args.ToUTF8( ).data( ),
            item.database_queue_id);

    // Combine the two parts
    wxString update_sql = update_sql_part1 + update_sql_part2;

    // Execute the update within a transaction
    main_frame->current_project.database.Begin( );

    // ExecuteSQL returns SQLITE_OK (0) on success, non-zero on error
    int sql_result = main_frame->current_project.database.ExecuteSQL(update_sql);

    if ( sql_result != SQLITE_OK ) {
        main_frame->current_project.database.Commit( ); // Commit to end transaction

        wxMessageBox("Failed to update queue item in database", "Database Error", wxOK | wxICON_ERROR);
        return false;
    }

    // Commit the transaction
    main_frame->current_project.database.Commit( );
    QM_LOG_DB("Successfully updated queue item %ld in database", item.database_queue_id);
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
                dragged_search_id = job.database_queue_id;
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
    mouse_down        = false;
    drag_in_progress  = false;
    dragged_row       = -1;
    dragged_search_id = -1;
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
        QM_LOG_ERROR("Could not find job at position %d", old_position);
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
        MyDebugAssertTrue(IsSearchRunning( ), "IsSearchRunning() returns false but running search exists");
    }
    else {
        MyDebugAssertTrue(currently_running_id == -1, "currently_running_id is %ld but no running jobs found", currently_running_id);
        MyDebugAssertFalse(IsSearchRunning( ), "IsSearchRunning() returns true but no running searches found");
    }
}

void TemplateMatchQueueManager::ContinueQueueExecution( ) {
    // Instance method to continue queue execution after a job completes
    // This is called from ProcessAllJobsFinished to continue with the next job

    QM_LOG_SEARCH("ContinueQueueExecution: currently_running_id=%ld, finalizing=%s",
                  currently_running_id, search_is_finalizing ? "true" : "false");

    if ( IsSearchRunning( ) ) {
        // A job is already running or finalizing, don't start another
        if ( search_is_finalizing ) {
            QM_LOG_SEARCH("ContinueQueueExecution: Job is finalizing, not continuing queue execution");
        }
        else {
            QM_LOG_SEARCH("ContinueQueueExecution: Job %ld is still running, not continuing queue execution", currently_running_id);
        }
        return;
    }

    // Continue execution using this instance
    QM_LOG_SEARCH("ContinueQueueExecution: No job running, continuing with next job");
    RunNextSearch( );
}

bool TemplateMatchQueueManager::ExecutionQueueHasActiveItems( ) const {
    // Check if there are any items in the execution queue (priority >= 0)
    // These are items waiting to run (pending/failed/partial)
    // Running items are at priority -1 in the available queue
    for ( const auto& item : execution_queue ) {
        if ( item.queue_order >= 0 ) {
            return true;
        }
    }
    return false;
}

void TemplateMatchQueueManager::OnSearchEnteringFinalization(long database_queue_id) {
    // Mark that a job is entering finalization phase
    // This prevents auto-advance from starting a new job prematurely
    QM_LOG_STATE("Job %ld entering finalization phase", database_queue_id);

    // Only set the finalization flag, don't clear currently_running_id yet
    // because UpdateSearchStatus still needs it for validation
    if ( currently_running_id == database_queue_id ) {
        search_is_finalizing = true;
    }
}

void TemplateMatchQueueManager::OnSearchCompleted(long database_queue_id, bool success) {
    // Ensure database is available for status updates
    MyDebugAssertTrue(IsDatabaseAvailable(main_frame), "OnSearchCompleted: Database not available");

    if constexpr ( skip_search_execution_for_queue_debugging ) {
        QM_LOG_DEBUG("=== OnSearchCompleted called ===");
        QM_LOG_DEBUG("Job ID: %ld, Success: %s", database_queue_id, success ? "true" : "false");
        QM_LOG_DEBUG("Currently running ID was: %ld", currently_running_id);
        QM_LOG_DEBUG("Queue state at job completion:");
        PrintQueueState( );
    }

    QM_LOG_SEARCH("Queue manager received job completion notification for job %ld (success: %s)",
                  database_queue_id, success ? "true" : "false");

    // Update job status in our queue
    const wxString& status = success ? "complete" : "failed";

    QM_LOG_STATE("OnSearchCompleted: About to update status for queue_id %ld to '%s'",
                 database_queue_id, status.mb_str( ).data( ));

    UpdateSearchStatus(database_queue_id, status);

    // Verify the job was moved to available queue (priority -1) when it started running
    bool job_found = false;
    for ( auto& job : execution_queue ) {
        if ( job.database_queue_id == database_queue_id ) {
            QM_LOG_DEBUG("Found completed job %ld at priority %d with status %s",
                         database_queue_id, job.queue_order, job.queue_status.mb_str( ).data( ));
            // Job should already be at queue_order = -1 (available queue) from when it started running
            // and status should already be updated by UpdateSearchStatus above
            if ( job.queue_order != -1 ) {
                QM_LOG_ERROR("WARNING: Completed job %ld was not at priority -1 (was %d), correcting",
                             database_queue_id, job.queue_order);
                job.queue_order = -1;
                // Need to update display since we changed queue_order
                UpdateQueueDisplay( );
            }
            job_found = true;
            break;
        }
    }

    if ( ! job_found ) {
        QM_LOG_ERROR("WARNING: Could not find completed job %ld in execution queue", database_queue_id);
    }

    // No need to renumber - jobs were already shifted when the completed job started running
    // The execution queue should already have consecutive priorities: 0, 1, 2, 3...

    // Clear currently running ID and finalization flag since job is completely done
    if ( currently_running_id == database_queue_id ) {
        currently_running_id = -1;
    }
    search_is_finalizing = false; // Search has finished finalizing

    // Update button state since no job is running
    UpdateButtonState( );

    // Update display to show new status (UpdateQueueDisplay already calls UpdateAvailableSearchesDisplay)
    QM_LOG_UI("OnSearchCompleted: Calling UpdateQueueDisplay to refresh status display");
    UpdateQueueDisplay( );

    // Save changes to database - using auto-commit mode (each UPDATE commits immediately)
    SaveQueueToDatabase( );

    if constexpr ( skip_search_execution_for_queue_debugging ) {
        QM_LOG_DEBUG("Queue state after job completion processing:");
        PrintQueueState( );
    }

    // Only auto-progress if enabled (default false - user controls progression)
    if ( auto_progress_queue ) {
        if constexpr ( skip_search_execution_for_queue_debugging ) {
            QM_LOG_DEBUG("=== Auto-progressing to next job in queue ===");
        }

        // Add a small delay before auto-advancing to ensure all finalization completes
        // This prevents race conditions where the next job might start before the previous
        // job's results are fully written to the database or UI is updated
        wxTimer* delay_timer = new wxTimer( );
        delay_timer->Bind(wxEVT_TIMER, [this, delay_timer](wxTimerEvent&) {
            QM_LOG_SEARCH("Auto-advance timer fired - progressing queue");
            // Force another display update before starting next search to ensure complete status is visible
            UpdateQueueDisplay( );
            ProgressExecutionQueue( );
            delete delay_timer; // Clean up the timer
        });
        delay_timer->StartOnce(500); // 500ms delay before auto-advancing
        QM_LOG_SEARCH("Scheduled auto-advance in 500ms");
    }
    else {
        if constexpr ( skip_search_execution_for_queue_debugging ) {
            QM_LOG_DEBUG("Job completed - auto-progression disabled (user controls queue progression)");
        }
    }

    if constexpr ( skip_search_execution_for_queue_debugging ) {
        QM_LOG_DEBUG("=== OnSearchCompleted finished ===");
    }
}

// Selection-based methods removed - using priority-based execution only

// RunNextSelectedJob method removed - using priority-based execution only

void TemplateMatchQueueManager::DeselectSearchInUI(long database_queue_id) {
    // Find the row corresponding to this job ID
    for ( size_t i = 0; i < execution_queue.size( ); ++i ) {
        if ( execution_queue[i].database_queue_id == database_queue_id ) {
            queue_list_ctrl->SetItemState(int(i), 0, wxLIST_STATE_SELECTED);
            QM_LOG_UI("Deselected job %ld from UI (row %zu)", database_queue_id, i);
            return;
        }
    }
    QM_LOG_UI("Could not find job %ld to deselect in UI", database_queue_id);
}

SearchCompletionInfo TemplateMatchQueueManager::GetSearchCompletionInfo(long queue_id) {
    SearchCompletionInfo info;
    info.search_id = -1; // Will be set when we find the actual search_id

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
    QM_LOG_DB_VALUES("GetSearchCompletionInfo: Queue ID %ld, SEARCH_ID %ld, ImageGroup %d -> %d/%d",
                     queue_id, search_id, image_group_id,
                     info.completed_count, info.total_count);

    return info;
}

wxString TemplateMatchQueueManager::FormatElapsedTime(double seconds) {
    // Handle invalid/zero values
    if ( seconds <= 0.0 ) {
        return "";
    }

    // Convert to days, hours, minutes, seconds
    int days    = int(seconds / 86400.0);
    int hours   = int((seconds - days * 86400.0) / 3600.0);
    int minutes = int((seconds - days * 86400.0 - hours * 3600.0) / 60.0);
    int secs    = int(seconds - days * 86400.0 - hours * 3600.0 - minutes * 60.0);

    // Format with days if >= 1 day
    if ( days > 0 ) {
        return wxString::Format("%d:%02d:%02d:%02d", days, hours, minutes, secs);
    }
    // Format without days if < 1 day
    return wxString::Format("%02d:%02d:%02d", hours, minutes, secs);
}

SearchTimingInfo TemplateMatchQueueManager::GetSearchTimingInfo(long search_id) {
    SearchTimingInfo info;

    // Return empty info if search_id is invalid or database not available
    if ( search_id <= 0 || ! main_frame || ! main_frame->current_project.is_open ) {
        return info;
    }

    // Query TEMPLATE_MATCH_LIST for timing data
    // Use COALESCE to handle NULL aggregates (when no rows match)
    wxString sql = wxString::Format(
            "SELECT "
            "COALESCE(SUM(ELAPSED_TIME_SECONDS), 0.0), "
            "COALESCE(AVG(ELAPSED_TIME_SECONDS), 0.0), "
            "MAX(DATETIME_OF_RUN), "
            "COUNT(*) "
            "FROM TEMPLATE_MATCH_LIST "
            "WHERE SEARCH_ID = %ld",
            search_id);

    bool more_data = main_frame->current_project.database.BeginBatchSelect(sql.ToUTF8( ).data( ));

    if ( more_data ) {
        long datetime_timestamp = 0;
        main_frame->current_project.database.GetFromBatchSelect("rrll",
                                                                &info.total_elapsed_seconds,
                                                                &info.avg_elapsed_seconds,
                                                                &datetime_timestamp,
                                                                &info.completed_count);
        main_frame->current_project.database.EndBatchSelect( );

        // Convert datetime timestamp to formatted string if available
        if ( datetime_timestamp > 0 ) {
            wxDateTime dt;
            dt.Set(time_t(datetime_timestamp));
            info.latest_datetime = dt.Format("%Y-%m-%d %H:%M");
        }

        // If no completed jobs, reset all timing fields
        if ( info.completed_count == 0 ) {
            info.total_elapsed_seconds = 0.0;
            info.avg_elapsed_seconds   = 0.0;
            info.latest_datetime       = "";
        }

        QM_LOG_DB_VALUES("GetSearchTimingInfo: SEARCH_ID %ld -> total=%.2fs, avg=%.2fs, latest=%s, count=%d",
                         search_id, info.total_elapsed_seconds, info.avg_elapsed_seconds,
                         info.latest_datetime.mb_str( ).data( ), info.completed_count);
    }

    return info;
}

void TemplateMatchQueueManager::RefreshSearchCompletionInfo( ) {
    // Update execution queue display
    UpdateQueueDisplay( );

    // Update available jobs display
    UpdateAvailableSearchesDisplay( );
}

void TemplateMatchQueueManager::OnResultAdded(long search_id) {
    QM_LOG_SEARCH("OnResultAdded called for job %ld - refreshing n/N display", search_id);

    // Refresh the completion info for this job and update displays
    RefreshSearchCompletionInfo( );
}

void TemplateMatchQueueManager::UpdateSearchIdForQueueItem(long queue_database_queue_id, long database_search_id) {
    // Update the queue item with the actual SEARCH_ID for correct n/N tracking
    QM_LOG_DB("UpdateSearchIdForQueueItem: Mapping queue ID %ld to database SEARCH_ID %ld",
              queue_database_queue_id, database_search_id);

    // Check execution queue first
    for ( auto& item : execution_queue ) {
        if ( item.database_queue_id == queue_database_queue_id ) {
            QM_LOG_DB("Found queue item %ld in execution queue, setting search_id = %ld",
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
            QM_LOG_DB("Found queue item %ld in available queue, setting search_id = %ld",
                      queue_database_queue_id, database_search_id);
            item.search_id = database_search_id;

            // Update database with the template match job ID
            if ( main_frame && main_frame->current_project.is_open ) {
                main_frame->current_project.database.UpdateSearchIdInQueueTable(queue_database_queue_id, database_search_id);
            }
            return;
        }
    }

    QM_LOG_ERROR("Warning: Could not find queue item %ld to update with database job ID %ld",
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
            QM_LOG_DB("Skipping orphaned search %ld - has results but no queue entry", search_id);
            continue;
        }
    }

    // Refresh the display
    UpdateAvailableSearchesDisplay( );
}

void TemplateMatchQueueManager::SwitchToEditorView( ) {
    QM_LOG_UI("Switching to editor view for queue ID %ld", editing_queue_id);

    // Hide queue labels
    execution_queue_label->Show(false);
    available_searches_label->Show(false);
#ifdef cisTEM_QM_LOGGING
    logging_label->Show(false);
#endif

    // Hide queue controls
    execution_queue_ctrl->Show(false);
    available_searches_ctrl->Show(false);

    // Hide container panels
    controls_panel->Show(false);
    bottom_controls->Show(false);

    // Hide individual queue buttons (for extra safety, though panels should hide them)
    run_selected_button->Show(false);
    update_selected_button->Show(false);
    add_to_queue_button->Show(false);
    remove_from_queue_button->Show(false);
    remove_selected_button->Show(false);
    clear_queue_button->Show(false);
    hide_completed_checkbox->Show(false);

    // Show editor panel
    editor_panel->Show(true);

    // Explicitly show InputPanel and ExpertPanel (child panels that may be hidden)
    editor_panel->ShowEditorPanels( );

    // Force layout update on editor panel to ensure all child controls are visible
    editor_panel->Layout( );

    // Force layout update on parent dialog
    Layout( );
}

void TemplateMatchQueueManager::SwitchToQueueView( ) {
    QM_LOG_UI("Switching back to queue view");

    editing_queue_id = -1;

    // Hide editor panel
    editor_panel->Show(false);

    // Show queue labels
    execution_queue_label->Show(true);
    available_searches_label->Show(true);
#ifdef cisTEM_QM_LOGGING
    logging_label->Show(true);
#endif

    // Show queue controls
    execution_queue_ctrl->Show(true);
    available_searches_ctrl->Show(true);

    // Show container panels
    controls_panel->Show(true);
    bottom_controls->Show(true);

    // Show individual queue buttons (for extra safety, though panels should show them)
    run_selected_button->Show(true);
    update_selected_button->Show(true);
    add_to_queue_button->Show(true);
    remove_from_queue_button->Show(true);
    remove_selected_button->Show(true);
    clear_queue_button->Show(true);
    hide_completed_checkbox->Show(true);

    // Refresh display with any changes
    UpdateQueueDisplay( );

    Layout( );
}

void TemplateMatchQueueManager::OnEditorSaveClick( ) {
    QM_LOG_METHOD_ENTRY("OnEditorSaveClick");

    MyDebugAssertTrue(editing_queue_id > 0, "OnEditorSaveClick called with invalid editing_queue_id");

    // Validate inputs
    wxString error_message;
    if ( ! editor_panel->ValidateInputs(error_message) ) {
        wxMessageBox(error_message, "Validation Error", wxOK | wxICON_ERROR);
        return; // Stay in edit mode
    }

    // Extract edited values
    TemplateMatchQueueItem updated_item;
    if ( ! editor_panel->ExtractToQueueItem(updated_item) ) {
        wxMessageBox("Failed to extract values from editor", "Error", wxOK | wxICON_ERROR);
        return;
    }

    // Find original item
    TemplateMatchQueueItem* original_item = nullptr;
    for ( auto& qi : execution_queue ) {
        if ( qi.database_queue_id == editing_queue_id ) {
            original_item = &qi;
            break;
        }
    }
    if ( ! original_item ) {
        for ( auto& qi : available_queue ) {
            if ( qi.database_queue_id == editing_queue_id ) {
                original_item = &qi;
                break;
            }
        }
    }

    if ( ! original_item ) {
        wxMessageBox("Could not find original queue item", "Error", wxOK | wxICON_ERROR);
        SwitchToQueueView( );
        return;
    }

    // Preserve metadata fields
    updated_item.database_queue_id = original_item->database_queue_id;
    updated_item.search_id         = original_item->search_id;
    updated_item.queue_status      = original_item->queue_status;
    updated_item.queue_order       = original_item->queue_order;

    // Update in memory
    *original_item = updated_item;

    // Update database
    if ( ! UpdateQueueItemInDatabase(updated_item) ) {
        wxMessageBox("Failed to update queue item in database", "Database Error",
                     wxOK | wxICON_ERROR);
        return;
    }

    QM_LOG_DB("Successfully updated queue item %ld", editing_queue_id);

    // Switch back to queue view
    SwitchToQueueView( );
}

void TemplateMatchQueueManager::OnEditorCancelClick( ) {
    QM_LOG_METHOD_ENTRY("OnEditorCancelClick");

    // Just switch back - discard any changes
    SwitchToQueueView( );
}

#ifdef cisTEM_QM_LOGGING
void TemplateMatchQueueManager::OnLoggingToggle(wxCommandEvent& event) {
    bool enable_logging = logging_toggle->GetValue( );

    if ( enable_logging ) {
        // Enable logging
        QueueManagerLogManager::EnableLogging(true);

        // Display the log file path
        wxString log_path = QueueManagerLogManager::GetLogFilePath( );
        log_file_text->SetLabel(wxString::Format("Log: %s", log_path));

        // Update button label
        logging_toggle->SetLabel("Disable Logging");

        // Log that we've started
        QM_LOG_STATE("Queue Manager logging enabled");
    }
    else {
        // Log that we're stopping
        QM_LOG_STATE("Queue Manager logging disabled");

        // Clear log file path display
        log_file_text->SetLabel("");

        // Update button label
        logging_toggle->SetLabel("Enable Logging");

        // Disable logging
        QueueManagerLogManager::DisableLogging( );
    }
}
#endif
