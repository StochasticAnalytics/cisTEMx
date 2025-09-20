#ifndef _SRC_GUI_TEMPLATEMATCHQUEUEMANAGER_H_
#define _SRC_GUI_TEMPLATEMATCHQUEUEMANAGER_H_

#include <wx/wx.h>
#include <wx/dataview.h>
#include <deque>

// Forward declarations
class MyMainFrame;

/**
 * @brief Tracks completion progress for template matching searches to display n/N progress counters
 *
 * This struct encapsulates the completion state of a template matching search by storing
 * the number of completed vs total expected results, enabling real-time progress display
 * in the queue manager UI. Links to TEMPLATE_MATCH_LIST table for result counting.
 */
struct JobCompletionInfo {
    long template_match_job_id;  ///< Database ID linking to TEMPLATE_MATCH_LIST table entries
    int completed_count;         ///< Number of results currently saved to database
    int total_count;            ///< Expected total number of results for this search

    /**
     * @brief Formats completion progress as "n/N" string for UI display
     * @return Formatted string showing completed vs total count
     */
    wxString GetCompletionString() const {
        return wxString::Format("%d/%d", completed_count, total_count);
    }

    /**
     * @brief Derives search status from completion progress for consistent state tracking
     * @return Status string: "pending" (0/N), "running" (n/N where 0<n<N), or "complete" (N/N)
     */
    wxString GetStatusFromCompletion() const {
        if (completed_count == 0) return "pending";
        if (completed_count == total_count) return "complete";
        return "running";
    }

    /**
     * @brief Calculates completion percentage for progress bars and sorting
     * @return Percentage complete (0.0-100.0), or 0.0 if total_count is invalid
     */
    double GetCompletionPercentage() const {
        return total_count > 0 ? (double)completed_count / total_count * 100.0 : 0.0;
    }
};

/**
 * @brief Encapsulates all parameters needed to execute a template matching search
 *
 * Contains both queue management metadata (database_queue_id, queue_status, queue_order)
 * and complete search parameters (template, images, search space, resource allocation).
 * Serves as the data transfer object between GUI panels, QueueManager, and database.
 *
 * @note Database ID Mapping:
 *   - database_queue_id: Links to TEMPLATE_MATCH_QUEUE.QUEUE_ID (created when search is queued)
 *   - template_match_job_id: Links to TEMPLATE_MATCH_LIST.TEMPLATE_MATCH_JOB_ID (created when results are saved)
 *   The mapping is established by UpdateJobDatabaseId() when the MatchTemplatePanel begins processing.
 */
class TemplateMatchQueueItem {
public:
    long database_queue_id;         ///< Persistent database queue identifier (TEMPLATE_MATCH_QUEUE.QUEUE_ID)
    long template_match_job_id;     ///< Results table identifier (TEMPLATE_MATCH_LIST.TEMPLATE_MATCH_JOB_ID, -1 if no results yet)
    wxString search_name;           ///< User-friendly name for this template matching search (maps to JOB_NAME in database)
    wxString queue_status;          ///< Computed status: "pending", "running", "partial", "complete" (not stored in DB)
    int queue_order;                ///< Priority order: 0=running, 1+=pending queue position, -1=available queue
    wxString custom_cli_args;       ///< Additional command-line arguments for this search

    // Store all the parameters needed to run the job
    // These will be populated when adding to queue
    int image_group_id;
    int reference_volume_asset_id;
    int run_profile_id;
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
        database_queue_id = -1;
        template_match_job_id = -1;  // No database job ID until results are written
        queue_status = "pending";
        queue_order = -1;  // Will be assigned when added to queue
        image_group_id = -1;
        reference_volume_asset_id = -1;
        run_profile_id = -1;
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

    /**
     * @brief Validates that all required parameters are set for search execution
     *
     * Performs comprehensive validation of queue metadata and search parameters using
     * debug assertions. Ensures the queue item is ready for execution by MatchTemplatePanel.
     *
     * @return Always returns true (assertions will abort on invalid state)
     */
    bool AreJobParametersValid() const {
        MyDebugAssertTrue(database_queue_id >= 0, "database_queue_id must be >= 0, got %ld", database_queue_id);
        MyDebugAssertTrue(image_group_id >= 0, "image_group_id must be >= 0, got %d", image_group_id);
        MyDebugAssertTrue(reference_volume_asset_id >= 0, "reference_volume_asset_id must be >= 0, got %d", reference_volume_asset_id);
        MyDebugAssertTrue(run_profile_id >= 0, "run_profile_id must be >= 0, got %d", run_profile_id);
        MyDebugAssertTrue(pixel_size > 0.0f, "pixel_size must be > 0.0, got %f", pixel_size);
        MyDebugAssertTrue(voltage > 0.0f, "voltage must be > 0.0, got %f", voltage);
        MyDebugAssertTrue(spherical_aberration >= 0.0f, "spherical_aberration must be >= 0.0, got %f", spherical_aberration);
        MyDebugAssertTrue(amplitude_contrast >= 0.0f && amplitude_contrast <= 1.0f, "amplitude_contrast must be 0.0-1.0, got %f", amplitude_contrast);
        MyDebugAssertTrue(queue_status == "pending" || queue_status == "running" || queue_status == "complete" || queue_status == "failed",
                         "Invalid queue_status: %s", queue_status.mb_str().data());
        MyDebugAssertTrue(!search_name.IsEmpty(), "search_name cannot be empty");
        MyDebugAssertFalse(symmetry.IsEmpty(), "symmetry cannot be empty");
        return true;
    }
};

/**
 * @brief Manages execution queue and available searches for template matching workflows
 *
 * Provides dual-table interface separating searches ready for execution (execution_queue)
 * from searches available for queueing (available_queue). Each search is defined by one
 * template, one or more images, search space parameters, and resource allocation. The
 * MatchTemplatePanel subdivides searches into distributed computing jobs - the QM operates
 * at the search level only.
 *
 * **Execution Model:**
 * - Priority-based execution: searches run in queue_order (0=highest priority)
 * - Multi-select support: batch add/remove operations between queues
 * - "Run Queue" always executes highest priority search (queue_order=0)
 *
 * **Architecture & Call Stack:**
 * - QM instances are always dialog-scoped (temporary UI), not persistent panel members
 * - Database serves as persistent storage; QM loads/saves queue state between dialogs
 * - MatchTemplatePanel orchestrates search creation and execution
 *
 * **Usage Patterns:**
 * 1. Direct execution: StartEstimationClick → AddJobToQueue(false) → temporary QM → execute
 * 2. Queue management: OnAddToQueueClick → AddJobToQueue(true) → dialog QM → user interaction
 * 3. Queue viewing: OnOpenQueueClick → dialog QM → load existing searches from database
 */
class TemplateMatchQueueManager : public wxPanel {
private:
    // Debug flag for queue behavior testing - set to true to skip actual job execution
    static constexpr bool skip_search_execution_for_queue_debugging = false;

    // UI Controls - Execution queue table (top) shows searches with queue_order >= 0
    wxListCtrl* execution_queue_ctrl;

    // UI Controls - Available searches table (bottom) shows searches with queue_order < 0
    wxListCtrl* available_jobs_ctrl;

    // UI Controls - Legacy support pointer for compatibility with existing code
    wxListCtrl* queue_list_ctrl;

    // UI Controls - Execution queue management buttons
    wxButton* run_selected_button;      ///< Execute highest priority search
    wxButton* cancel_run_button;        ///< Cancel currently running search
    wxButton* assign_priority_button;   ///< Open priority assignment dialog

    // UI Controls - Queue movement buttons
    wxButton* add_to_queue_button;      ///< Move searches from available to execution queue
    wxButton* remove_from_queue_button; ///< Move searches from execution to available queue

    // UI Controls - General queue management
    wxButton* remove_selected_button;   ///< Delete selected searches entirely
    wxButton* clear_queue_button;       ///< Clear entire execution queue
    wxCheckBox* hide_completed_checkbox;///< Toggle visibility of completed searches

    // Data Collections - In-memory queue storage
    std::deque<TemplateMatchQueueItem> execution_queue; ///< Searches ready for execution (queue_order >= 0)
    std::deque<TemplateMatchQueueItem> available_queue; ///< Searches available for queueing (queue_order < 0)
    long currently_running_id;                          ///< Database ID of search currently executing

    // State Tracking - Execution and display control
    bool auto_progress_queue;   ///< True if queue should auto-advance after search completion
    bool hide_completed_jobs;   ///< True if completed searches should be hidden from available queue
    bool gui_update_frozen;     ///< True while SetupJobFromQueueItem is executing to prevent GUI interference

    // Panel Integration - Reference for job execution and database access
    MatchTemplatePanel* match_template_panel_ptr; ///< Panel for delegating search execution

    // Drag-and-Drop State - Manual implementation for wxListCtrl priority reordering
    bool drag_in_progress;    ///< True during active drag operation
    bool updating_display;    ///< Prevent drag operations during display updates
    int dragged_row;          ///< Row index being dragged
    long dragged_job_id;      ///< Database ID of search being dragged
    wxPoint drag_start_pos;   ///< Mouse position where drag operation started
    bool mouse_down;          ///< Track if mouse button is currently pressed

    // Private Helper Methods
    wxColour GetStatusColor(const wxString& status);       ///< Returns color for search status display
    void SetStatusDisplay(wxListCtrl* list_ctrl, long item_index, const wxString& status); ///< Sets status text, color, and font formatting
    void UpdateQueueDisplay();                             ///< Refreshes both execution and available queue displays
    void UpdateExecutionQueueDisplay();                    ///< Refreshes execution queue table with current data
    void UpdateAvailableJobsDisplay();                     ///< Refreshes available queue table with current data
    void PopulateListControl(wxListCtrl* ctrl,            ///< Shared method to populate list controls
                            const std::vector<TemplateMatchQueueItem*>& items,
                            bool is_execution_queue);
    int GetSelectedRow();                                   ///< Gets currently selected row index
    void DeselectJobInUI(long database_queue_id);          ///< Removes UI selection for specified search

public:
    TemplateMatchQueueManager(wxWindow* parent, MatchTemplatePanel* match_template_panel = nullptr);
    ~TemplateMatchQueueManager();

    // Execution queue management methods
    /**
     * @brief Adds search to execution queue with database persistence and priority assignment
     *
     * Persists search to database via AddToTemplateMatchQueue, assigns next available queue_order
     * for priority sequencing, and adds to in-memory execution_queue. Core method called by
     * AddJobToQueue workflow for both immediate execution and interactive queueing.
     */
    void AddToExecutionQueue(const TemplateMatchQueueItem& item);
    /**
     * @brief Removes search from execution queue by index
     * @param index Position in execution_queue to remove
     */
    void RemoveFromExecutionQueue(int index);

    /**
     * @brief Removes all searches from execution queue
     */
    void ClearExecutionQueue();

    /**
     * @brief Reorders search prioritpley by changing queue position
     * @param job_index Current position in execution queue
     * @param new_position Desired position for priority ordering
     */
    void SetExecutionQueuePosition(int job_index, int new_position);

    /**
     * @brief Advances queue after search completion by promoting next search and decrementing others
     */
    void ProgressExecutionQueue();

    // Execution methods
    /**
     * @brief Executes the highest priority search from the execution queue
     *
     * Finds search with queue_order=0, enables auto-progression, manages queue
     * reordering during execution, and delegates to ExecuteJob(). Primary method
     * for initiating queue-based execution from UI or auto-progression.
     */
    void RunNextJob();

    /**
     * @brief Core execution method that delegates search to MatchTemplatePanel
     *
     * Validates search parameters, updates status to "running", and calls panel's
     * execution methods. Returns true if delegation succeeds, false otherwise.
     */
    bool ExecuteJob(TemplateMatchQueueItem& job_to_run);

    /**
     * @brief Updates search status in both memory and database
     *
     * Synchronizes queue status changes across in-memory queues and persistent storage.
     * Used for state transitions during search lifecycle.
     */
    void UpdateJobStatus(long database_queue_id, const wxString& new_status);

    /**
     * @brief Continues queue execution after current search completes
     *
     * Called by completion callbacks to advance to next pending search.
     * Enables automatic queue progression when auto_progress_queue is enabled.
     */
    void ContinueQueueExecution();

    /**
     * @brief Callback for search completion to update queue state and trigger progression
     * @param database_queue_id Database ID of the completed search
     * @param success True if search completed successfully, false if failed
     */
    void OnJobCompleted(long database_queue_id, bool success);

    /**
     * @brief Checks if any searches are waiting for execution
     * @return True if execution queue contains pending searches
     */
    bool HasPendingJobs();

    /**
     * @brief Checks if a search is currently executing
     * @return True if execution_in_progress flag is set
     */
    bool IsJobRunning() const;

    /**
     * @brief Controls automatic queue progression after search completion
     * @param enable If true, queue will automatically advance to next search
     */
    void SetAutoProgressQueue(bool enable) { auto_progress_queue = enable; }

    /**
     * @brief Freezes GUI parameter updates during job setup to prevent interference
     * @param frozen If true, GUI population from queue selections is disabled
     */
    void SetGuiUpdateFrozen(bool frozen) { gui_update_frozen = frozen; }

    /**
     * @brief Retrieves completion progress for a specific search
     * @param template_match_job_id Database ID from TEMPLATE_MATCH_LIST table
     * @return JobCompletionInfo with current progress counts
     */
    JobCompletionInfo GetJobCompletionInfo(long template_match_job_id);

    /**
     * @brief Updates all progress displays with latest database completion counts
     */
    void RefreshJobCompletionInfo();

    /**
     * @brief Loads completed searches from database into available queue
     */
    void PopulateAvailableJobsFromDatabase();

    /**
     * @brief Updates progress display when a new result is added
     * @param template_match_job_id Database ID to update progress for
     */
    void OnResultAdded(long template_match_job_id);

    /**
     * @brief Links queue item to actual database job ID after search execution begins
     * @param queue_database_queue_id Queue database ID to update
     * @param database_template_match_job_id Actual TEMPLATE_MATCH_JOB_ID from results table
     */
    void UpdateJobDatabaseId(long queue_database_queue_id, long database_template_match_job_id);

    /**
     * @brief Discovers and populates missing database job IDs for completed searches
     */

    /**
     * @brief Validates queue state consistency for debugging
     */
    void ValidateQueueConsistency() const;

    /**
     * @brief Checks if search at queue index is currently running
     * @param queue_index Position in execution_queue to check
     * @return True if search status is "running"
     */
    inline bool IsJobRunning(int queue_index) const {
        return queue_index >= 0 && queue_index < execution_queue.size() &&
               execution_queue[queue_index].queue_status == "running";
    }

    /**
     * @brief Checks if search at queue index is pending execution
     * @param queue_index Position in execution_queue to check
     * @return True if search status is "pending"
     */
    inline bool IsJobPending(int queue_index) const {
        return queue_index >= 0 && queue_index < execution_queue.size() &&
               execution_queue[queue_index].queue_status == "pending";
    }

    /**
     * @brief Checks if search at queue index has completed successfully
     * @param queue_index Position in execution_queue to check
     * @return True if search status is "complete"
     */
    inline bool IsJobComplete(int queue_index) const {
        return queue_index >= 0 && queue_index < execution_queue.size() &&
               execution_queue[queue_index].queue_status == "complete";
    }

    /**
     * @brief Checks if search at queue index has failed
     * @param queue_index Position in execution_queue to check
     * @return True if search status is "failed"
     */
    inline bool IsJobFailed(int queue_index) const {
        return queue_index >= 0 && queue_index < execution_queue.size() &&
               execution_queue[queue_index].queue_status == "failed";
    }

    // wxWidgets event handlers
    void OnRunSelectedClick(wxCommandEvent& event);           ///< Executes highest priority search
    void OnClearQueueClick(wxCommandEvent& event);            ///< Removes all searches from execution queue
    void OnRemoveSelectedClick(wxCommandEvent& event);        ///< Removes selected searches from current table
    void OnSelectionChanged(wxListEvent& event);              ///< Updates UI state based on execution queue selection
    void OnAvailableJobsSelectionChanged(wxListEvent& event); ///< Updates UI state based on available queue selection
    void OnHideCompletedToggle(wxCommandEvent& event);        ///< Toggles display of completed searches
    void OnAssignPriorityClick(wxCommandEvent& event);        ///< Opens priority assignment dialog
    void OnCancelRunClick(wxCommandEvent& event);             ///< Cancels currently running search
    void OnAddToQueueClick(wxCommandEvent& event);            ///< Moves selected searches from available to execution queue
    void OnRemoveFromQueueClick(wxCommandEvent& event);       ///< Moves selected searches from execution to available queue

    // Manual drag-and-drop implementation for priority reordering
    void OnBeginDrag(wxListEvent& event);                     ///< Initiates drag operation for priority reordering
    void OnMouseLeftDown(wxMouseEvent& event);                ///< Tracks mouse down for drag start
    void OnMouseMotion(wxMouseEvent& event);                  ///< Handles drag motion for visual feedback
    void OnMouseLeftUp(wxMouseEvent& event);                  ///< Completes drag operation and reorders queue

    /**
     * @brief Reorders execution queue items by changing their positions
     * @param old_position Current position of item being moved
     * @param new_position Target position for the item
     */
    void ReorderQueueItems(int old_position, int new_position);

    /**
     * @brief Resets drag operation state variables to clean state
     */
    void CleanupDragState();

    /**
     * @brief Loads queue state from database into in-memory collections
     */
    void LoadQueueFromDatabase();

    /**
     * @brief Persists current queue state to database
     */
    void SaveQueueToDatabase();

    /**
     * @brief Debug helper to print current queue state to console
     */
    void PrintQueueState();

    /**
     * @brief Helper to check if database is available
     * @param frame_ptr The main_frame pointer to check
     * @return true if database is available, false otherwise
     */
    bool IsDatabaseAvailable(MyMainFrame* frame_ptr) const;

    /**
     * @brief Compute queue status from completion progress
     * @param completed Number of completed jobs
     * @param total Total number of jobs
     * @param currently_running_id ID of currently running job (-1 if none)
     * @param item_id ID of this queue item
     * @return Status string: "pending", "running", "partial", or "complete"
     */
    inline wxString ComputeStatusFromProgress(int completed, int total, long currently_running_id, long item_id) const {
        if (currently_running_id == item_id) {
            return "running";
        }
        if (completed == 0) {
            return "pending";
        }
        if (completed < total) {
            return "partial";  // Previously called "failed" but this is more accurate
        }
        return "complete";
    }

    DECLARE_EVENT_TABLE()
};

#endif // _SRC_GUI_TEMPLATEMATCHQUEUEMANAGER_H_