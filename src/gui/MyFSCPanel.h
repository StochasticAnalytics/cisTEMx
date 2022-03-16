#ifndef __MyFSCPanel__
#define __MyFSCPanel__

/**
@file
Subclass of FSCPanel, which is generated by wxFormBuilder.
*/

#include "ProjectX_gui.h"

//// end generated include

/** Implementing FSCPanel */
class MyFSCPanel : public FSCPanel {
  protected:
    // Handlers for FSCPanel events.
    //void OnClassComboBoxChange( wxCommandEvent& event );
    Refinement* my_refinement;
    int         highlighted_class;

  public:
    /** Constructor */
    MyFSCPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize(500, 300), long style = wxTAB_TRAVERSAL);
    void AddRefinement(Refinement* refinement_to_plot);
    void Clear( );

    void PopupTextClick(wxCommandEvent& event);
    void SaveImageClick(wxCommandEvent& event);
    void HighlightClass(int wanted_class);

    //// end generated class members
};

#endif // __MyFSCPanel__
