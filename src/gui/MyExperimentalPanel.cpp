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

MyExperimentalPanel::MyExperimentalPanel(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
    : ExperimentalPanel(parent, id, pos, size, style) {
    // Bind OnListBookPageChanged from
    Bind(wxEVT_LISTBOOK_PAGE_CHANGED, wxBookCtrlEventHandler(MyExperimentalPanel::OnExperimentalBookPageChanged), this);
}

// TODO: destructor

void MyExperimentalPanel::OnExperimentalBookPageChanged(wxBookCtrlEvent& event) {
    extern RefineTemplateDevPanel* refine_template_dev_panel;

#ifdef __WXOSX__
    // Necessary for MacOS to refresh the panels
    if ( event.GetSelection( ) == 0 ) {
        match_template_panel->Layout( );
        match_template_panel->Refresh( );
    }
    else if ( event.GetSelection( ) == 1 ) {
        match_template_results_panel->Layout( );
        match_template_results_panel->Refresh( );
    }
    else if ( event.GetSelection( ) == 2 ) {
        refine_template_panel->Layout( );
        refine_template_panel->Refresh( );
    }
#endif
}
