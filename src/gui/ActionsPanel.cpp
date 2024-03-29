#include "../core/gui_core_headers.h"

ActionsPanel::ActionsPanel(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
    : ActionsPanelParent(parent, id, pos, size, style) {
    // Bind OnListBookPageChanged from
    Bind(wxEVT_LISTBOOK_PAGE_CHANGED, wxBookCtrlEventHandler(ActionsPanel::OnActionsBookPageChanged), this);
}

// TODO: destructor

void ActionsPanel::OnActionsBookPageChanged(wxBookCtrlEvent& event) {
    extern MyAlignMoviesPanel*   align_movies_panel;
    extern FitCTFPanel*          fitctf_panel;
    extern MyFindParticlesPanel* findparticles_panel;
    extern MyRefine2DPanel*      classification_panel;
    extern AbInitio3DPanel*      ab_initio_3d_panel;
    extern AutoRefine3DPanelSpa* auto_refine_3d_panel_spa;
    extern AutoRefine3DPanelRx*  auto_refine_3d_panel_rx;
    extern MyRefine3DPanel*      refine_3d_panel;
    extern RefineCTFPanel*       refine_ctf_panel;
    extern Generate3DPanel*      generate_3d_panel;
    extern Sharpen3DPanel*       sharpen_3d_panel;
}
