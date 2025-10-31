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

#ifndef __PopupTextDialog__
#define __PopupTextDialog__

class PopupTextDialog : public PopupTextDialogParent {

  public:
    PopupTextDialog(wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxDEFAULT_DIALOG_STYLE);
    void OnCopyToClipboardClick(wxCommandEvent& event);
    void OnCloseButtonClick(wxCommandEvent& event);
    void OnSaveButtonClick(wxCommandEvent& event);
};

#endif
