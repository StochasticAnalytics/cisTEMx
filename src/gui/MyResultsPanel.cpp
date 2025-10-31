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

MyResultsPanel::MyResultsPanel(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
    : ResultsPanel(parent, id, pos, size, style) {
    // Bind OnListBookPageChanged from
    Bind(wxEVT_LISTBOOK_PAGE_CHANGED, wxBookCtrlEventHandler(MyResultsPanel::OnResultsBookPageChanged), this);
}

// TODO: destructor

void MyResultsPanel::OnResultsBookPageChanged(wxBookCtrlEvent& event) {
    extern MyMovieAlignResultsPanel* movie_results_panel;
    extern MyFindCTFResultsPanel*    ctf_results_panel;
    extern MyPickingResultsPanel*    picking_results_panel;
    extern Refine2DResultsPanel*     refine2d_results_panel;
    extern MyRefinementResultsPanel* refinement_results_panel;

    // We we were editing the particle picking results, and we move away from Results, we may need to do some database stuff
    if ( event.GetOldSelection( ) == 2 )
        picking_results_panel->UpdateResultsFromBitmapPanel( );

#ifdef __WXOSX__
    // Necessary for MacOS to refresh the panels
    if ( event.GetSelection( ) == 0 ) {
        movie_results_panel->Layout( );
        movie_results_panel->Refresh( );
    }
    else if ( event.GetSelection( ) == 1 ) {
        ctf_results_panel->Layout( );
        ctf_results_panel->Refresh( );
    }
    else if ( event.GetSelection( ) == 2 ) {
        picking_results_panel->Layout( );
        picking_results_panel->Refresh( );
    }
    else if ( event.GetSelection( ) == 3 ) {
        refine2d_results_panel->Layout( );
        refine2d_results_panel->Refresh( );
    }
    else if ( event.GetSelection( ) == 4 ) {
        refinement_results_panel->Layout( );
        refinement_results_panel->Refresh( );
    }
#endif
}
