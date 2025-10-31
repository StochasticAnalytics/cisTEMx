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

#ifndef __CTF1DPLOT_PANEL_H__
#define __CTF1DPLOT_PANEL_H__

#include <vector>
#include "../gui/mathplot.h"
#include <wx/panel.h>

class
        CTF1DPanel : public wxPanel {

    wxBoxSizer* GraphSizer;

  public:
    CTF1DPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL, const wxString& name = wxPanelNameStr);

    void Clear( );
    void AddPoint(double spatial_frequency, double ctf_fit, double quality_of_fit, double aplitude_spectrum);
    void Draw( );

    std::vector<double> current_spatial_frequency;
    std::vector<double> current_ctf_fit;
    std::vector<double> current_quality_of_fit;
    std::vector<double> current_amplitude_spectrum;

    mpWindow*        current_plot_window;
    mpTopInfoLegend* legend;
    mpTitle*         title;

    mpFXYVector* current_ctf_fit_vector_layer;
    mpFXYVector* current_quality_of_fit_vector_layer;
    mpFXYVector* current_amplitude_vector_layer;
};

#endif
