#ifndef _SRC_GUI_TEMPLATEMATCHCONTROLSHELPER_H_
#define _SRC_GUI_TEMPLATEMATCHCONTROLSHELPER_H_

#include "ProjectX_gui_matchtemplate.h"

/**
 * @brief Bundle of control references for template matching UI
 *
 * Avoids passing 20+ parameters to constructor. All members default to nullptr.
 */
struct TemplateMatchControls {
    // Combo boxes and selection panels
    ImageGroupPickerComboPanel*  group_combo;
    VolumeAssetPickerComboPanel* reference_panel;
    MemoryComboBox*              run_profile_combo;
    wxComboBox*                  symmetry_combo;

    // Numeric controls - resolution and angular steps
    NumericTextCtrl* high_res_limit;
    NumericTextCtrl* out_of_plane_step;
    NumericTextCtrl* in_plane_step;

    // Defocus search controls
    wxRadioButton*   defocus_search_yes;
    wxRadioButton*   defocus_search_no;
    NumericTextCtrl* defocus_search_range;
    NumericTextCtrl* defocus_search_step;

    // Pixel size search controls
    wxRadioButton*   pixel_size_search_yes;
    wxRadioButton*   pixel_size_search_no;
    NumericTextCtrl* pixel_size_search_range;
    NumericTextCtrl* pixel_size_search_step;

    // Peak detection
    NumericTextCtrl* min_peak_radius;

    // GPU and FFT options
    wxRadioButton* use_gpu_yes;
    wxRadioButton* use_gpu_no;
    wxRadioButton* use_fast_fft_yes;
    wxRadioButton* use_fast_fft_no;

    // Default constructor - initialize all pointers to nullptr
    TemplateMatchControls( );
};

/**
 * @brief Helper class for manipulating template matching GUI controls
 *
 * Eliminates code duplication between MatchTemplatePanel and TemplateMatchQueueItemEditor.
 * Provides shared logic for filling combo boxes, validating inputs, and extracting/populating
 * queue item data structures.
 *
 * Uses non-static methods with control references passed via struct to avoid excessive
 * constructor parameters while maintaining clean testability.
 */
class TemplateMatchControlsHelper {
  private:
    TemplateMatchControls controls;

  public:
    /**
     * @brief Constructs helper with references to UI controls
     * @param ctrl_refs Struct containing pointers to all relevant controls
     */
    explicit TemplateMatchControlsHelper(const TemplateMatchControls& ctrl_refs);

    /**
     * @brief Fills all combo boxes with current asset data
     *
     * Populates:
     * - Image groups from image_asset_panel
     * - Reference volumes from volume_asset_panel
     * - Run profiles from main_frame->current_project
     * - Standard symmetry options (C1, C2, D2, I, O, T, etc.)
     *
     * Must be called before PopulateFromQueueItem to ensure selections work correctly.
     */
    void FillComboBoxes( );

    /**
     * @brief Populates all GUI controls from queue item data
     * @param item Queue item to display for editing
     *
     * Sets combo box selections, numeric controls, radio buttons from stored parameters.
     * Handles enabling/disabling dependent controls (defocus/pixel size search ranges).
     */
    void PopulateFromQueueItem(const TemplateMatchQueueItem& item);

    /**
     * @brief Extracts edited values from GUI controls into queue item
     * @param item Queue item to populate with edited values
     * @return True if extraction successful, false if validation fails
     *
     * Validates inputs first, then reads all control values into the item.
     * Preserves metadata fields (database_queue_id, search_id, queue_status, queue_order).
     */
    bool ExtractToQueueItem(TemplateMatchQueueItem& item);

    /**
     * @brief Validates all input controls for consistency and valid ranges
     * @param error_message Populated with detailed error description on failure
     * @return True if all inputs valid, false otherwise
     *
     * Checks:
     * - Required selections (group, reference, run profile, symmetry)
     * - Positive numeric ranges (resolution, angular steps, peak radius)
     * - Conditional validation (defocus/pixel size search parameters when enabled)
     */
    bool ValidateInputs(wxString& error_message);
};

#endif // _SRC_GUI_TEMPLATEMATCHCONTROLSHELPER_H_
