class ActionsPanel : public ActionsPanelParent {
  public:
    ActionsPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize(500, 300), long style = wxTAB_TRAVERSAL);
    void OnActionsBookPageChanged(wxBookCtrlEvent& event);

    const char* Type( ) const { return "ActionsPanel"; };
};
