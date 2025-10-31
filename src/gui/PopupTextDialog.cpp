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

PopupTextDialog::PopupTextDialog(wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style)
    : PopupTextDialogParent(parent, id, title, pos, size, style) {
    ClipBoardButton->SetBitmap(wxArtProvider::GetBitmap(wxART_COPY));
}

void PopupTextDialog::OnCopyToClipboardClick(wxCommandEvent& event) {
    if ( wxTheClipboard->Open( ) ) {
        wxTheClipboard->SetData(new wxTextDataObject(OutputTextCtrl->GetValue( )));
        wxTheClipboard->Close( );
    }
}

void PopupTextDialog::OnSaveButtonClick(wxCommandEvent& event) {
    ProperOverwriteCheckSaveDialog* saveFileDialog;
    saveFileDialog = new ProperOverwriteCheckSaveDialog(this, _("Save txt file"), "TXT files (*.txt)|*.txt", ".txt");
    if ( saveFileDialog->ShowModal( ) == wxID_CANCEL ) {
        saveFileDialog->Destroy( );
        return;
    }

    // save the file then..

    OutputTextCtrl->SaveFile(saveFileDialog->ReturnProperPath( ));
    saveFileDialog->Destroy( );
}

void PopupTextDialog::OnCloseButtonClick(wxCommandEvent& event) {
    Destroy( );
}
