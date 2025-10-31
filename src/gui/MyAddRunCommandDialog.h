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

#ifndef __MyAddRunCommandDialog__
#define __MyAddRunCommandDialog__

/** Implementing AddRunCommandDialog */
class MyAddRunCommandDialog : public AddRunCommandDialog {
    MyRunProfilesPanel* my_parent;

  public:
    /** Constructor */
    MyAddRunCommandDialog(MyRunProfilesPanel* parent);

    void ProcessResult( );

    void OnOKClick(wxCommandEvent& event);
    void OnCancelClick(wxCommandEvent& event);
    void OnEnter(wxCommandEvent& event);
    void OnOverrideCheckbox(wxCommandEvent& event);

    //// end generated class members
};

#endif // __MyAddRunCommandDialog__
