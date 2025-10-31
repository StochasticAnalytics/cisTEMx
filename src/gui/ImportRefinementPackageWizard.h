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

#ifndef __src_gui_ImportRefinementPackageWizard__
#define __src_gui_ImportRefinementPackageWizard__

class ImportRefinementPackageWizard : public ImportRefinementPackageWizardParent {
  protected:
    void OnStackBrowseButtonClick(wxCommandEvent& event);
    void OnMetaBrowseButtonClick(wxCommandEvent& event);

  public:
    /** Constructor */
    ImportRefinementPackageWizard(wxWindow* parent);
    //// end generated class members

    void OnFinished(wxWizardEvent& event);
    void OnPageChanged(wxWizardEvent& event);
    void OnPathChange(wxCommandEvent& event);
    void OnUpdateUI(wxUpdateUIEvent& event);

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

  private:
    template <typename StarFileSource_t>
    void ImportRefinementPackage(StarFileSource_t& input_params_file, const int stack_x_size, const int stack_num_images);
};
#endif