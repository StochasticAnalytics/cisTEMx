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

MySettingsPanel::MySettingsPanel(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
    : SettingsPanel(parent, id, pos, size, style) {
    // Bind OnListBookPageChanged from
    Bind(wxEVT_LISTBOOK_PAGE_CHANGED, wxBookCtrlEventHandler(MySettingsPanel::OnSettingsBookPageChanged), this);
}

// TODO: destructor

void MySettingsPanel::OnSettingsBookPageChanged(wxBookCtrlEvent& event) {
    extern MyRunProfilesPanel* run_profiles_panel;

#ifdef __WXOSX__
    // Necessary for MacOS to refresh the panels
    if ( event.GetSelection( ) == 0 ) {
        run_profiles_panel->Layout( );
        run_profiles_panel->Refresh( );
    }
#endif
}
