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

#ifndef _gui_ActionsPanelTm_h_
#define _gui_ActionsPanelTm_h_

class ActionsPanelTm : public ActionsPanelParent {
  public:
    ActionsPanelTm(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize(500, 300), long style = wxTAB_TRAVERSAL);
    ~ActionsPanelTm( );
    virtual void OnActionsBookPageChanged(wxListbookEvent& event) override;
};

#endif
