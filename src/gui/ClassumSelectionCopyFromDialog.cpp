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

extern MyRefinementPackageAssetPanel* refinement_package_asset_panel;

ClassumSelectionCopyFromDialog::ClassumSelectionCopyFromDialog(wxWindow* parent)
    : ClassumSelectionCopyFromDialogParent(parent) {
}

void ClassumSelectionCopyFromDialog::OnOKButtonClick(wxCommandEvent& event) {
    selected_selection_array_position = original_array_positions.Item(SelectionListCtrl->GetNextItem(-1, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED));

    if ( selected_selection_array_position == -1 )
        EndModal(wxID_CANCEL);
    else
        EndModal(wxID_OK);
}

void ClassumSelectionCopyFromDialog::OnCancelButtonClick(wxCommandEvent& event) {
    EndModal(wxID_CANCEL);
}

void ClassumSelectionCopyFromDialog::FillWithSelections(int number_of_classes) {
    int counter;
    int old_width;
    int current_width;
    int list_position = 0;

    original_array_positions.Clear( );
    Freeze( );

    SelectionListCtrl->ClearAll( );
    SelectionListCtrl->InsertColumn(0, wxT("Selection"), wxLIST_FORMAT_CENTRE, wxLIST_AUTOSIZE_USEHEADER);
    SelectionListCtrl->InsertColumn(1, wxT("Creation Date"), wxLIST_FORMAT_CENTRE, wxLIST_AUTOSIZE_USEHEADER);
    SelectionListCtrl->InsertColumn(2, wxT("Number Selected"), wxLIST_FORMAT_CENTRE, wxLIST_AUTOSIZE_USEHEADER);

    for ( counter = 0; counter < refinement_package_asset_panel->all_classification_selections.GetCount( ); counter++ ) {
        if ( refinement_package_asset_panel->all_classification_selections.Item(counter).number_of_classes == number_of_classes ) {
            original_array_positions.Add(counter);
            SelectionListCtrl->InsertItem(list_position, refinement_package_asset_panel->all_classification_selections.Item(counter).name);
            SelectionListCtrl->SetItem(list_position, 1, refinement_package_asset_panel->all_classification_selections.Item(counter).creation_date.FormatISOCombined(' '));
            SelectionListCtrl->SetItem(list_position, 2, wxString::Format("%i", refinement_package_asset_panel->all_classification_selections.Item(counter).number_of_selections));
            list_position++;
        }

        if ( original_array_positions.GetCount( ) > 0 ) {
            SelectionListCtrl->SetItemState(0, wxLIST_STATE_SELECTED, wxLIST_STATE_SELECTED);
            OkButton->Enable(true);
        }
        else
            OkButton->Enable(false);
    }

    for ( counter = 0; counter < SelectionListCtrl->GetColumnCount( ); counter++ ) {
        old_width = SelectionListCtrl->GetColumnWidth(counter);
        SelectionListCtrl->SetColumnWidth(counter, wxLIST_AUTOSIZE);
        current_width = SelectionListCtrl->GetColumnWidth(counter);

        if ( old_width > current_width )
            SelectionListCtrl->SetColumnWidth(counter, wxLIST_AUTOSIZE_USEHEADER);
    }

    Thaw( );
}
