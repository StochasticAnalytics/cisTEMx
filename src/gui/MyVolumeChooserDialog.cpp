/*
 * Original Copyright (c) 2017, Howard Hughes Medical Institute
 * Licensed under Janelia Research Campus Software License 1.2
 * See license_details/LICENSE-JANELIA.txt
 *
 * Modifications Copyright (c) 2025, Stochastic Analytics, LLC
 * Modifications licensed under MPL 2.0 for academic use; 
 * commercial license required for commercial use.
 * See LICENSE.md for details.
 */

#include "../core/gui_core_headers.h"

extern MyVolumeAssetPanel* volume_asset_panel;

MyVolumeChooserDialog::MyVolumeChooserDialog(wxWindow* parent)
    : VolumeChooserDialog(parent) {
    selected_volume_id   = -1;
    selected_volume_name = "Generate from params.";
    ComboBox->FillComboBox(true);
}

void MyVolumeChooserDialog::OnCancelClick(wxCommandEvent& event) {
    Destroy( );
}

void MyVolumeChooserDialog::OnRenameClick(wxCommandEvent& event) {
    if ( ComboBox->GetSelection( ) == 0 ) {
        selected_volume_id   = -1;
        selected_volume_name = "Generate from params.";
    }
    else {
        selected_volume_id   = volume_asset_panel->ReturnAssetID(ComboBox->GetSelection( ) - 1);
        selected_volume_name = volume_asset_panel->ReturnAssetName(ComboBox->GetSelection( ) - 1);
    }

    EndModal(wxID_OK);
}
