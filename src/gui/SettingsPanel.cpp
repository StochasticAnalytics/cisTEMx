#include "../core/gui_core_headers.h"

SettingsPanel::SettingsPanel(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
    : SettingsPanelParent(parent, id, pos, size, style) {
    // Bind OnListBookPageChanged from
    Bind(wxEVT_LISTBOOK_PAGE_CHANGED, wxBookCtrlEventHandler(SettingsPanel::OnSettingsBookPageChanged), this);
}

// TODO: destructor

void SettingsPanel::OnSettingsBookPageChanged(wxBookCtrlEvent& event) {
    extern MyRunProfilesPanel* run_profiles_panel;

#ifdef __WXOSX__
    // Necessary for MacOS to refresh the panels
    if ( event.GetSelection( ) == 0 ) {
        run_profiles_panel->Layout( );
        run_profiles_panel->Refresh( );
    }
#endif
}
