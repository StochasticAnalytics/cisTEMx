#ifndef _SRC_GUI_TEMPLATEMATCHQUEUEITEMEDITOR_H_
#define _SRC_GUI_TEMPLATEMATCHQUEUEITEMEDITOR_H_

#include "ProjectX_gui_matchtemplate.h"
#include "TemplateMatchControlsHelper.h"

class MatchTemplatePanel;
class TemplateMatchQueueManager;

/**
 * @brief Editor panel for modifying queued template match search parameters
 *
 * Derives from wxFormBuilder-generated TemplateMatchingQueueManagerInputParent.
 * Uses show/hide pattern to display within QueueManager, not as modal dialog.
 *
 * The editor panel allows users to modify queue item parameters before execution,
 * following the same UI pattern as MatchTemplatePanel's InputPanel/ProgressPanel switching.
 *
 * Implementation uses TemplateMatchControlsHelper to eliminate code duplication
 * with MatchTemplatePanel and provide clean testable control manipulation logic.
 */
class TemplateMatchQueueItemEditor : public TemplateMatchingQueueManagerInputParent {
  private:
    MatchTemplatePanel*          match_template_panel_ptr; ///< Access to asset panels (image groups, volumes)
    TemplateMatchQueueManager*   queue_manager_ptr; ///< Parent queue manager for callbacks
    TemplateMatchControlsHelper* controls_helper; ///< Helper for control manipulation

  public:
    /**
     * @brief Constructs editor panel with references to parent components
     * @param parent Parent window (typically TemplateMatchQueueManager)
     * @param tm_panel MatchTemplatePanel for accessing asset lists
     * @param qm TemplateMatchQueueManager for save/cancel callbacks
     */
    TemplateMatchQueueItemEditor(wxWindow*                  parent,
                                 MatchTemplatePanel*        tm_panel,
                                 TemplateMatchQueueManager* qm);

    /**
     * @brief Destructor - cleans up controls helper
     */
    ~TemplateMatchQueueItemEditor( );

    /**
     * @brief Fills all combo boxes with current asset data
     *
     * Must be called before PopulateFromQueueItem to ensure combo boxes
     * are populated and ready for selection.
     *
     * Delegates to controls_helper->FillComboBoxes().
     */
    void FillComboBoxes( ) {
        if ( controls_helper )
            controls_helper->FillComboBoxes( );
    }

    /**
     * @brief Populates all GUI controls from queue item data
     * @param item Queue item to display for editing
     *
     * Sets combo box selections, numeric controls, radio buttons, etc.
     * from the queue item's stored parameters. Handles enabling/disabling
     * dependent controls based on radio button states.
     *
     * Delegates to controls_helper->PopulateFromQueueItem().
     */
    void PopulateFromQueueItem(const TemplateMatchQueueItem& item) {
        if ( controls_helper )
            controls_helper->PopulateFromQueueItem(item);
    }

    /**
     * @brief Extracts edited values from GUI controls into queue item
     * @param item Queue item to populate with edited values
     * @return True if extraction successful, false if validation fails
     *
     * Reads all control values and validates them before populating the item.
     * Preserves metadata fields (database_queue_id, search_id, queue_status, queue_order).
     *
     * Delegates to controls_helper->ExtractToQueueItem().
     */
    bool ExtractToQueueItem(TemplateMatchQueueItem& item) {
        if ( controls_helper )
            return controls_helper->ExtractToQueueItem(item);
        return false;
    }

    /**
     * @brief Validates all input controls for consistency and valid ranges
     * @param error_message Populated with detailed error description on failure
     * @return True if all inputs valid, false otherwise
     *
     * Checks for required selections, valid numeric ranges, non-empty strings, etc.
     *
     * Delegates to controls_helper->ValidateInputs().
     */
    bool ValidateInputs(wxString& error_message) {
        if ( controls_helper )
            return controls_helper->ValidateInputs(error_message);
        error_message = "Internal error: controls_helper is null";
        return false;
    }

    /**
     * @brief Shows the InputPanel and ExpertPanel child controls
     *
     * Must be called when switching to editor view to ensure all controls are visible.
     * The parent panel being hidden/shown doesn't automatically show these child panels.
     */
    void ShowEditorPanels( ) {
        InputPanel->Show(true);
        ExpertPanel->Show(true);
        Layout( );
    }

    // Event handlers (override virtual methods from TemplateMatchingQueueManagerInputParent)

    /**
     * @brief Handles Update button click - validates and saves changes
     *
     * Delegates to queue_manager_ptr->OnEditorSaveClick() for actual save logic.
     */
    void OnUpdateQueueItemClick(wxCommandEvent& event) override;

    /**
     * @brief Handles Cancel button click - discards changes
     *
     * Delegates to queue_manager_ptr->OnEditorCancelClick() to switch back to queue view.
     */
    void OnCancelUpdateQueueItemClick(wxCommandEvent& event) override;

    /**
     * @brief Handles Reset All Defaults button - not implemented for editor
     *
     * Overrides parent method to prevent confusion (defaults are for new searches, not editing).
     */
    void ResetAllDefaultsClick(wxCommandEvent& event) override;
};

#endif // _SRC_GUI_TEMPLATEMATCHQUEUEITEMEDITOR_H_
