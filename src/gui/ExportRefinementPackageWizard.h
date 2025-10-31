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

#ifndef __ExportRefinementPackageWizard__
#define __ExportRefinementPackageWizard__

class ExportRefinementPackageWizard : public ExportRefinementPackageWizardParent {
  protected:
    void OnStackBrowseButtonClick(wxCommandEvent& event);
    void OnMetaBrowseButtonClick(wxCommandEvent& event);

  public:
    /** Constructor */
    ExportRefinementPackageWizard(wxWindow* parent);
    ~ExportRefinementPackageWizard( );
    //// end generated class members

    void OnFinished(wxWizardEvent& event);
    void OnPageChanged(wxWizardEvent& event);
    void OnPathChange(wxCommandEvent& event);
    void OnUpdateUI(wxUpdateUIEvent& event);

    void OnParamsComboBox(wxCommandEvent& event);

    void CheckPaths( );

    void inline DisableNextButton( ) {
        wxWindow* win = wxWindow::FindWindowById(wxID_FORWARD);
        if ( win )
            win->Enable(false);
    }

    void inline EnableNextButton( ) {
        wxWindow* win = wxWindow::FindWindowById(wxID_FORWARD);
        if ( win )
            win->Enable(true);
    }

    RefinementPackage* current_package;
};

#endif
