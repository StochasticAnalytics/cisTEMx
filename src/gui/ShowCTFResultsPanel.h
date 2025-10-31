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

#ifndef __SHOWCTF_RESULTS_PANEL_H__
#define __SHOWCTF_RESULTS_PANEL_H__

#include <vector>
#include "../gui/mathplot.h"
#include <wx/panel.h>
#include "../gui/job_panel.h"
#include "../gui/ProjectX_gui.h"

class
        ShowCTFResultsPanel : public ShowCTFResultsPanelParent {
  public:
    ShowCTFResultsPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);
    ~ShowCTFResultsPanel( );

    void OnFitTypeRadioButton(wxCommandEvent& event);
    void Clear( );
    void Draw(wxString diagnostic_filename, bool find_additional_phase_shift, float defocus1, float defocus2, float defocus_angle, float phase_shift, float score, float fit_res, float alias_res, float iciness, float tilt_angle, float tilt_axis, float sample_thickness, wxString ImageFile);
};

#endif
