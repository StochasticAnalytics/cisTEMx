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

#ifndef __DISPLAYREFINEMENTRESULTS_PANEL_H__
#define __DISPLAYREFINEMENTRESULTS_PANEL_H__

class DisplayRefinementResultsPanel : public DisplayRefinementResultsPanelParent {
  public:
    DisplayRefinementResultsPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL, const wxString& name = wxPanelNameStr);

    void Clear( );
};

class DisplayCTFRefinementResultsPanel : public DisplayCTFRefinementResultsPanelParent {
  public:
    DisplayCTFRefinementResultsPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL, const wxString& name = wxPanelNameStr);

    void Clear( );
};

#endif
