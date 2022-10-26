class ResultsPanel : public ResultsPanelParent {
  public:
    ResultsPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize(500, 300), long style = wxTAB_TRAVERSAL);
    void OnResultsBookPageChanged(wxBookCtrlEvent& event);

    const char* Type( ) const { return "ResultsPanel"; };
};