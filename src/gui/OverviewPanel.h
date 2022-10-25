#ifndef __MyOverviewPanel__
#define __MyOverviewPanel__

class OverviewPanel : public OverviewPanelParent {
  public:
    OverviewPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize(500, 300), long style = wxTAB_TRAVERSAL);

    void SetWelcomeInfo( );
    void SetProjectInfo( );
    void OnInfoURL(wxTextUrlEvent& event);

    const char* Type( ) const { return "OverviewPanel"; };
};

#endif // __MyOverviewPanel__
