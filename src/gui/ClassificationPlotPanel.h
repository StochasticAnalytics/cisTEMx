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

#ifndef __ClassificationPlotPanel__
#define __ClassificationPlotPanel__

class
        ClassificationPlotPanel : public ClassificationPlotPanelParent {

    wxBoxSizer* LikelihoodSizer;
    wxBoxSizer* SigmaSizer;
    wxBoxSizer* PercentageMovedSizer;

  public:
    ClassificationPlotPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);
    ~ClassificationPlotPanel( );

    void Clear( );
    void AddPoints(float round, float likelihood, float sigma, float percentage_moved);
    void Draw( );

    std::vector<double> round_data;
    std::vector<double> likelihood_data;
    std::vector<double> sigma_data;
    std::vector<double> percentage_moved_data;

    mpWindow* likelihood_plot_window;
    mpWindow* sigma_plot_window;
    mpWindow* percentage_moved_plot_window;

    mpScaleX* likelihood_xaxis;
    mpScaleX* sigma_xaxis;
    mpScaleX* percentage_moved_xaxis;

    mpScaleY* likelihood_yaxis;
    mpScaleY* sigma_yaxis;
    mpScaleY* percentage_moved_yaxis;

    mpFXYVector* likelihood_vector_layer;
    mpFXYVector* sigma_vector_layer;
    mpFXYVector* percentage_moved_vector_layer;
};

#endif
