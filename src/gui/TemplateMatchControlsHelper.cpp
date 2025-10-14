#include "../core/gui_core_headers.h"
#include "TemplateMatchControlsHelper.h"
#include "TemplateMatchQueueManager.h"

// Extern globals for asset panels (declared in MatchTemplatePanel.cpp)
extern MyImageAssetPanel*  image_asset_panel;
extern MyVolumeAssetPanel* volume_asset_panel;
extern MyRunProfilesPanel* run_profiles_panel;
extern MyMainFrame*        main_frame;

// TemplateMatchControls struct implementation
TemplateMatchControls::TemplateMatchControls( )
    : group_combo(nullptr),
      reference_panel(nullptr),
      run_profile_combo(nullptr),
      symmetry_combo(nullptr),
      high_res_limit(nullptr),
      out_of_plane_step(nullptr),
      in_plane_step(nullptr),
      defocus_search_yes(nullptr),
      defocus_search_no(nullptr),
      defocus_search_range(nullptr),
      defocus_search_step(nullptr),
      pixel_size_search_yes(nullptr),
      pixel_size_search_no(nullptr),
      pixel_size_search_range(nullptr),
      pixel_size_search_step(nullptr),
      min_peak_radius(nullptr),
      use_gpu_yes(nullptr),
      use_gpu_no(nullptr),
      use_fast_fft_yes(nullptr),
      use_fast_fft_no(nullptr),
      custom_cli_args_text(nullptr) {
}

// TemplateMatchControlsHelper implementation
TemplateMatchControlsHelper::TemplateMatchControlsHelper(const TemplateMatchControls& ctrl_refs)
    : controls(ctrl_refs) {
}

void TemplateMatchControlsHelper::FillComboBoxes( ) {
    // Fill GroupComboBox with image groups
    if ( controls.group_combo ) {
        controls.group_combo->FillComboBox(true);
    }

    // Fill ReferenceSelectPanel with volumes
    if ( controls.reference_panel ) {
        controls.reference_panel->FillComboBox( );
    }

    // Fill RunProfileComboBox using built-in method
    if ( controls.run_profile_combo ) {
        controls.run_profile_combo->FillWithRunProfiles( );
    }

    // Fill SymmetryComboBox with standard symmetries
    if ( controls.symmetry_combo ) {
        controls.symmetry_combo->Clear( );
        controls.symmetry_combo->Append("C1");
        controls.symmetry_combo->Append("C2");
        controls.symmetry_combo->Append("C3");
        controls.symmetry_combo->Append("C4");
        controls.symmetry_combo->Append("C5");
        controls.symmetry_combo->Append("C6");
        controls.symmetry_combo->Append("D2");
        controls.symmetry_combo->Append("D3");
        controls.symmetry_combo->Append("D4");
        controls.symmetry_combo->Append("I");
        controls.symmetry_combo->Append("O");
        controls.symmetry_combo->Append("T");
    }
}

void TemplateMatchControlsHelper::PopulateFromQueueItem(const TemplateMatchQueueItem& item) {
    // Set the group selection by finding matching database ID
    if ( controls.group_combo && image_asset_panel && image_asset_panel->all_groups_list ) {
        for ( int group_index = 0; group_index < image_asset_panel->all_groups_list->number_of_groups; group_index++ ) {
            if ( image_asset_panel->all_groups_list->groups[group_index].id == item.image_group_id ) {
                controls.group_combo->SetSelection(group_index);
                break;
            }
        }
    }

    // Set reference volume selection by finding matching asset_id
    if ( controls.reference_panel && item.reference_volume_asset_id >= 0 && volume_asset_panel ) {
        int array_position = volume_asset_panel->ReturnArrayPositionFromAssetID(item.reference_volume_asset_id);
        // revert - debug population
        QM_LOG_UI("PopulateFromQueueItem: asset_id=%d, array_position=%d",
                  item.reference_volume_asset_id, array_position);
        if ( array_position >= 0 ) {
            controls.reference_panel->SetSelection(array_position);
            QM_LOG_UI("PopulateFromQueueItem: SetSelection(%d) complete", array_position);
        }
        else {
            // Asset ID not found - this indicates corrupted queue item
            // Set to first available volume as safe default
            QM_LOG_UI("WARNING: Reference volume asset_id=%d not found! Setting to first volume as default.",
                      item.reference_volume_asset_id);
            if ( volume_asset_panel->all_assets_list->number_of_assets > 0 ) {
                controls.reference_panel->SetSelection(0);
                QM_LOG_UI("PopulateFromQueueItem: Set to default selection 0 (asset_id=%d)",
                          volume_asset_panel->ReturnAssetPointer(0)->asset_id);
            }
        }
    }

    // Set run profile by finding matching profile id
    if ( controls.run_profile_combo && item.run_profile_id >= 0 && run_profiles_panel ) {
        for ( int profile_index = 0; profile_index < run_profiles_panel->run_profile_manager.number_of_run_profiles; profile_index++ ) {
            if ( run_profiles_panel->run_profile_manager.run_profiles[profile_index].id == item.run_profile_id ) {
                controls.run_profile_combo->SetSelection(profile_index);
                break;
            }
        }
    }

    // Set symmetry
    if ( controls.symmetry_combo ) {
        controls.symmetry_combo->SetValue(item.symmetry);
    }

    // Set resolution limit
    if ( controls.high_res_limit ) {
        controls.high_res_limit->SetValue(wxString::Format("%.2f", item.high_resolution_limit));
    }

    // Set angular steps
    if ( controls.out_of_plane_step ) {
        controls.out_of_plane_step->SetValue(wxString::Format("%.2f", item.out_of_plane_angular_step));
    }

    if ( controls.in_plane_step ) {
        controls.in_plane_step->SetValue(wxString::Format("%.2f", item.in_plane_angular_step));
    }

    // Set defocus search parameters
    if ( item.defocus_search_range > 0 ) {
        if ( controls.defocus_search_yes )
            controls.defocus_search_yes->SetValue(true);
        if ( controls.defocus_search_no )
            controls.defocus_search_no->SetValue(false);
        if ( controls.defocus_search_range ) {
            controls.defocus_search_range->SetValue(wxString::Format("%.2f", item.defocus_search_range));
            controls.defocus_search_range->Enable(true);
        }
        if ( controls.defocus_search_step ) {
            controls.defocus_search_step->SetValue(wxString::Format("%.2f", item.defocus_step));
            controls.defocus_search_step->Enable(true);
        }
    }
    else {
        if ( controls.defocus_search_yes )
            controls.defocus_search_yes->SetValue(false);
        if ( controls.defocus_search_no )
            controls.defocus_search_no->SetValue(true);
        if ( controls.defocus_search_range )
            controls.defocus_search_range->Enable(false);
        if ( controls.defocus_search_step )
            controls.defocus_search_step->Enable(false);
    }

    // Set pixel size search parameters
    if ( item.pixel_size_search_range > 0 ) {
        if ( controls.pixel_size_search_yes )
            controls.pixel_size_search_yes->SetValue(true);
        if ( controls.pixel_size_search_no )
            controls.pixel_size_search_no->SetValue(false);
        if ( controls.pixel_size_search_range ) {
            controls.pixel_size_search_range->SetValue(wxString::Format("%.4f", item.pixel_size_search_range));
            controls.pixel_size_search_range->Enable(true);
        }
        if ( controls.pixel_size_search_step ) {
            controls.pixel_size_search_step->SetValue(wxString::Format("%.4f", item.pixel_size_step));
            controls.pixel_size_search_step->Enable(true);
        }
    }
    else {
        if ( controls.pixel_size_search_yes )
            controls.pixel_size_search_yes->SetValue(false);
        if ( controls.pixel_size_search_no )
            controls.pixel_size_search_no->SetValue(true);
        if ( controls.pixel_size_search_range )
            controls.pixel_size_search_range->Enable(false);
        if ( controls.pixel_size_search_step )
            controls.pixel_size_search_step->Enable(false);
    }

    // Set peak radius
    if ( controls.min_peak_radius ) {
        controls.min_peak_radius->SetValue(wxString::Format("%.2f", item.min_peak_radius));
    }

    // Set GPU and FastFFT options
    if ( controls.use_gpu_yes && controls.use_gpu_no ) {
        controls.use_gpu_yes->SetValue(item.use_gpu);
        controls.use_gpu_no->SetValue(! item.use_gpu);
    }

    if ( controls.use_fast_fft_yes && controls.use_fast_fft_no ) {
        controls.use_fast_fft_yes->SetValue(item.use_fast_fft);
        controls.use_fast_fft_no->SetValue(! item.use_fast_fft);
    }

    // Set custom CLI arguments
    if ( controls.custom_cli_args_text ) {
        controls.custom_cli_args_text->SetValue(item.custom_cli_args);
    }
}

bool TemplateMatchControlsHelper::ExtractToQueueItem(TemplateMatchQueueItem& item) {
    // First validate
    wxString error_message;
    if ( ! ValidateInputs(error_message) ) {
        return false;
    }

    // Extract image group ID
    if ( controls.group_combo && image_asset_panel && image_asset_panel->all_groups_list ) {
        int selected_index  = controls.group_combo->GetSelection( );
        item.image_group_id = image_asset_panel->all_groups_list->groups[selected_index].id;
    }

    // Extract reference volume asset ID
    if ( controls.reference_panel && volume_asset_panel ) {
        int selected_index             = controls.reference_panel->GetSelection( );
        item.reference_volume_asset_id = volume_asset_panel->ReturnAssetPointer(selected_index)->asset_id;
        // revert - debug extraction
        QM_LOG_UI("ExtractToQueueItem: reference_panel GetSelection()=%d, asset_id=%d",
                  selected_index, item.reference_volume_asset_id);
    }

    // Extract run profile ID
    if ( controls.run_profile_combo && run_profiles_panel ) {
        int selected_index  = controls.run_profile_combo->GetSelection( );
        item.run_profile_id = run_profiles_panel->run_profile_manager.run_profiles[selected_index].id;
    }

    // Extract symmetry
    if ( controls.symmetry_combo ) {
        item.symmetry = controls.symmetry_combo->GetValue( ).Upper( );
    }

    // Extract resolution limits
    if ( controls.high_res_limit ) {
        item.high_resolution_limit = controls.high_res_limit->ReturnValue( );
    }
    item.low_resolution_limit = 300.0f; // Hardcoded in template matching

    // Extract angular steps
    if ( controls.out_of_plane_step ) {
        item.out_of_plane_angular_step = controls.out_of_plane_step->ReturnValue( );
    }

    if ( controls.in_plane_step ) {
        item.in_plane_angular_step = controls.in_plane_step->ReturnValue( );
    }

    // Extract defocus search parameters
    if ( controls.defocus_search_yes && controls.defocus_search_yes->GetValue( ) ) {
        if ( controls.defocus_search_range ) {
            item.defocus_search_range = controls.defocus_search_range->ReturnValue( );
        }
        if ( controls.defocus_search_step ) {
            item.defocus_step = controls.defocus_search_step->ReturnValue( );
        }
    }
    else {
        item.defocus_search_range = 0.0f;
        item.defocus_step         = 0.0f;
    }

    // Extract pixel size search parameters
    if ( controls.pixel_size_search_yes && controls.pixel_size_search_yes->GetValue( ) ) {
        if ( controls.pixel_size_search_range ) {
            item.pixel_size_search_range = controls.pixel_size_search_range->ReturnValue( );
        }
        if ( controls.pixel_size_search_step ) {
            item.pixel_size_step = controls.pixel_size_search_step->ReturnValue( );
        }
    }
    else {
        item.pixel_size_search_range = 0.0f;
        item.pixel_size_step         = 0.0f;
    }

    // Extract peak detection parameters
    if ( controls.min_peak_radius ) {
        item.min_peak_radius = controls.min_peak_radius->ReturnValue( );
    }

    // Extract GPU and FastFFT settings
    if ( controls.use_gpu_yes ) {
        item.use_gpu = controls.use_gpu_yes->GetValue( );
    }

    if ( controls.use_fast_fft_yes ) {
        item.use_fast_fft = controls.use_fast_fft_yes->GetValue( );
    }

    // Extract custom CLI arguments
    if ( controls.custom_cli_args_text ) {
        item.custom_cli_args = controls.custom_cli_args_text->GetValue( );
    }

    // Get volume parameters for search name and mask radius
    if ( controls.reference_panel && volume_asset_panel ) {
        VolumeAsset* current_volume = volume_asset_panel->ReturnAssetPointer(controls.reference_panel->GetSelection( ));
        if ( current_volume ) {
            item.search_name               = wxString::Format("Template: %s", current_volume->filename.GetName( ));
            item.ref_box_size_in_angstroms = current_volume->x_size * current_volume->pixel_size;
            item.mask_radius               = current_volume->x_size * current_volume->pixel_size / 2.0f;
        }
    }

    // Additional parameters (currently not in GUI, using defaults)
    item.refinement_threshold       = 0.0;
    item.xy_change_threshold        = 0.0;
    item.exclude_above_xy_threshold = false;

    // revert - debug: Log ALL extracted parameters
    QM_LOG_UI("=== ExtractToQueueItem: ALL PARAMETERS ===");
    QM_LOG_UI("  image_group_id: %d", item.image_group_id);
    QM_LOG_UI("  reference_volume_asset_id: %d", item.reference_volume_asset_id);
    QM_LOG_UI("  run_profile_id: %d", item.run_profile_id);
    QM_LOG_UI("  symmetry: %s", item.symmetry);
    QM_LOG_UI("  high_resolution_limit: %.2f", item.high_resolution_limit);
    QM_LOG_UI("  low_resolution_limit: %.2f", item.low_resolution_limit);
    QM_LOG_UI("  out_of_plane_angular_step: %.2f", item.out_of_plane_angular_step);
    QM_LOG_UI("  in_plane_angular_step: %.2f", item.in_plane_angular_step);
    QM_LOG_UI("  defocus_search_range: %.2f", item.defocus_search_range);
    QM_LOG_UI("  defocus_step: %.2f", item.defocus_step);
    QM_LOG_UI("  pixel_size_search_range: %.4f", item.pixel_size_search_range);
    QM_LOG_UI("  pixel_size_step: %.4f", item.pixel_size_step);
    QM_LOG_UI("  min_peak_radius: %.2f", item.min_peak_radius);
    QM_LOG_UI("  use_gpu: %d", item.use_gpu);
    QM_LOG_UI("  use_fast_fft: %d", item.use_fast_fft);
    QM_LOG_UI("  ref_box_size_in_angstroms: %.2f", item.ref_box_size_in_angstroms);
    QM_LOG_UI("  mask_radius: %.2f", item.mask_radius);
    QM_LOG_UI("  search_name: %s", item.search_name);
    QM_LOG_UI("=== END PARAMETERS ===");

    return true;
}

bool TemplateMatchControlsHelper::ValidateInputs(wxString& error_message) {
    // Check image group selection
    if ( ! controls.group_combo || controls.group_combo->GetSelection( ) < 0 ) {
        error_message = "Please select an image group";
        return false;
    }

    // Check reference volume selection
    if ( ! controls.reference_panel || controls.reference_panel->GetSelection( ) < 0 ) {
        error_message = "Please select a reference volume";
        return false;
    }

    // Check run profile selection
    if ( ! controls.run_profile_combo || controls.run_profile_combo->GetSelection( ) < 0 ) {
        error_message = "Please select a run profile";
        return false;
    }

    // Check symmetry
    if ( ! controls.symmetry_combo || controls.symmetry_combo->GetValue( ).IsEmpty( ) ) {
        error_message = "Please enter a symmetry value";
        return false;
    }

    // Validate numeric controls have positive values
    if ( controls.high_res_limit && controls.high_res_limit->ReturnValue( ) <= 0 ) {
        error_message = "High resolution limit must be positive";
        return false;
    }

    if ( controls.out_of_plane_step && controls.out_of_plane_step->ReturnValue( ) <= 0 ) {
        error_message = "Out-of-plane angular step must be positive";
        return false;
    }

    if ( controls.in_plane_step && controls.in_plane_step->ReturnValue( ) <= 0 ) {
        error_message = "In-plane angular step must be positive";
        return false;
    }

    if ( controls.min_peak_radius && controls.min_peak_radius->ReturnValue( ) < 0 ) {
        error_message = "Minimum peak radius cannot be negative";
        return false;
    }

    // Validate defocus search parameters if enabled
    if ( controls.defocus_search_yes && controls.defocus_search_yes->GetValue( ) ) {
        if ( controls.defocus_search_range && controls.defocus_search_range->ReturnValue( ) <= 0 ) {
            error_message = "Defocus search range must be positive when defocus search is enabled";
            return false;
        }
        if ( controls.defocus_search_step && controls.defocus_search_step->ReturnValue( ) <= 0 ) {
            error_message = "Defocus search step must be positive when defocus search is enabled";
            return false;
        }
    }

    // Validate pixel size search parameters if enabled
    if ( controls.pixel_size_search_yes && controls.pixel_size_search_yes->GetValue( ) ) {
        if ( controls.pixel_size_search_range && controls.pixel_size_search_range->ReturnValue( ) <= 0 ) {
            error_message = "Pixel size search range must be positive when pixel size search is enabled";
            return false;
        }
        if ( controls.pixel_size_search_step && controls.pixel_size_search_step->ReturnValue( ) <= 0 ) {
            error_message = "Pixel size search step must be positive when pixel size search is enabled";
            return false;
        }
    }

    error_message = "";
    return true;
}
