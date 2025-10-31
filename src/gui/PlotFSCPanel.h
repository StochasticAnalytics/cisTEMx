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

#ifndef __PLOTFSC_PANEL_H__
#define __PLOTFSC_PANEL_H__

#include <vector>
#include "../gui/mathplot.h"
#include <wx/panel.h>

WX_DEFINE_ARRAY(mpFXYVector*, ArrayofmpFXYVectors);

class RefinementLimit : public mpFY {

  public:
    double spatial_frequency;

    RefinementLimit(float wanted_spatial_frequency) : mpFY(wxT(" Min. refinement limit"), mpALIGN_TOP) { spatial_frequency = wanted_spatial_frequency; }

    void SetSpatialFrequency(float wanted_spatial_frequency) { spatial_frequency = wanted_spatial_frequency; }

    virtual double GetX(double y) { return spatial_frequency; }

    virtual double GetMinX( ) { return -0.05; }

    virtual double GetMinY( ) { return 1.05; }
};

class
        PlotFSCPanel : public wxPanel {

    wxBoxSizer* GraphSizer;
    int         number_of_added_fscs;

  public:
    PlotFSCPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL, const wxString& name = wxPanelNameStr);
    ~PlotFSCPanel( );

    void Clear(bool update_display = true);
    void AddPartFSC(ResolutionStatistics* statistics_to_add, float wanted_nyquist);
    void Draw(float nyquist);
    void SetupBaseLayers( );
    void HighlightClass(int wanted_class);

    ArrayofmpFXYVectors FSC_Layers;

    float current_refinement_resolution_limit;

    float            current_nyquist;
    mpWindow*        current_plot_window;
    mpTitle*         title;
    mpScaleX*        current_xaxis;
    mpScaleY*        current_yaxis;
    RefinementLimit* refinement_limit;
};

#endif
