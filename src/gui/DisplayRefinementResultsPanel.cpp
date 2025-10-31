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

DisplayRefinementResultsPanel::DisplayRefinementResultsPanel(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name)
    : DisplayRefinementResultsPanelParent(parent, id, pos, size, style) {
    ShowOrthDisplayPanel->EnableStartWithFourierScaling( );
    ShowOrthDisplayPanel->EnableDoNotShowStatusBar( );
    ShowOrthDisplayPanel->Initialise( );
}

void DisplayRefinementResultsPanel::Clear( ) {
    ShowOrthDisplayPanel->Clear( );
    AngularPlotPanel->Clear( );
    FSCResultsPanel->Clear( );
}

DisplayCTFRefinementResultsPanel::DisplayCTFRefinementResultsPanel(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name)
    : DisplayCTFRefinementResultsPanelParent(parent, id, pos, size, style) {

    ShowOrthDisplayPanel->EnableStartWithFourierScaling( );
    ShowOrthDisplayPanel->EnableDoNotShowStatusBar( );
    ShowOrthDisplayPanel->Initialise( );
}

void DisplayCTFRefinementResultsPanel::Clear( ) {
    ShowOrthDisplayPanel->Clear( );
    DefocusHistorgramPlotPanel->Clear( );
    FSCResultsPanel->Clear( );
}
