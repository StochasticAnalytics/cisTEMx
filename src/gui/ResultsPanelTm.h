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

#ifndef _SRC_GUI_RESULTS_PANEL_TM_H_
#define _SRC_GUI_RESULTS_PANEL_TM_H_

// Temporary forward declaration
class ResultsPanelParent;

class ResultsPanelTm : public ResultsPanelParent {
  public:
    ResultsPanelTm(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize(500, 300), long style = wxTAB_TRAVERSAL);
    void OnResultsBookPageChanged(wxBookCtrlEvent& event);
};

#endif