#ifndef _SRC_gui_ActionsPanelRx_h_
#define _SRC_gui_ActionsPanelRx_h_

class ActionsPanelRx : public ActionsPanelParent {
  public:
    ActionsPanelRx(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize(500, 300), long style = wxTAB_TRAVERSAL);
    void OnActionsBookPageChanged(wxBookCtrlEvent& event);

    const char* Type( ) const { return "ActionsPanelRx"; };
};

#endif
