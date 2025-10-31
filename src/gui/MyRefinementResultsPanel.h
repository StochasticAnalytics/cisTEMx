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

class MyRefinementResultsPanel : public RefinementResultsPanel {

  public:
    bool refinement_package_is_dirty;
    bool input_params_are_dirty;

    long        refinement_id_of_buffered_refinement;
    Refinement* currently_displayed_refinement;
    Refinement* buffered_full_refinement;

    MyRefinementResultsPanel(wxWindow* parent);
    void FillRefinementPackageComboBox(void);
    void FillInputParametersComboBox(void);
    void FillAngles(int wanted_class);
    void DrawOrthViews( );

    void OnUpdateUI(wxUpdateUIEvent& event);

    void OnRefinementPackageComboBox(wxCommandEvent& event);
    void OnInputParametersComboBox(wxCommandEvent& event);
    void OnDisplayTabChange(wxAuiNotebookEvent& event);
    void OnJobDetailsToggle(wxCommandEvent& event);

    int current_class;

    void OnClassComboBoxChange(wxCommandEvent& event);
    void AngularPlotPopupClick(wxCommandEvent& event);
    void PopupParametersClick(wxCommandEvent& event);

    void UpdateCachedRefinement( );
    void UpdateBufferedFullRefinement( );

    void WriteJobInfo(int wanted_class);
    void ClearJobInfo( );

    void Clear( );
};
