#include "../core/gui_core_headers.h"

ActionsPanelTm::ActionsPanelTm(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
    : ActionsPanelParent(parent, id, pos, size, style) {
    // Bind OnListBookPageChanged from
    Bind(wxEVT_LISTBOOK_PAGE_CHANGED, wxBookCtrlEventHandler(ActionsPanelTm::OnActionsBookPageChanged), this);
}

// TODO: destructor

void ActionsPanelTm::OnActionsBookPageChanged(wxBookCtrlEvent& event) {

    extern MyAlignMoviesPanel*        align_movies_panel;
    extern FitCTFPanel*               fitctf_panel;
    extern MatchTemplatePanel*        match_template_panel;
    extern MatchTemplateResultsPanel* match_template_results_panel;
    extern RefineTemplatePanel*       refine_template_panel;
    extern Generate3DPanel*           generate_3d_panel;
    extern Sharpen3DPanel*            sharpen_3d_panel;
}
