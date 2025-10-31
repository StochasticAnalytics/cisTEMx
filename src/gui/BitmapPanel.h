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

#ifndef __BITMAP_PANEL_H__
#define __BITMAP_PANEL_H__

#include <wx/panel.h>

class BitmapPanel : public wxPanel {
  public:
    Image PanelImage;
    //wxBitmap PanelBitmap; // buffer for the panel size
    wxString panel_text;
    wxString title_text;
    bool     use_auto_contrast;

    BitmapPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL, const wxString& name = wxPanelNameStr);
    ~BitmapPanel( );

    void OnPaint(wxPaintEvent& evt);
    void OnEraseBackground(wxEraseEvent& event);
    //	void SetupPanelBitmap();
    void Clear( );

    bool  should_show;
    float font_size_multiplier;
};

#endif
