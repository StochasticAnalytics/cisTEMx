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

#ifndef __ClassumSelectionCopyFromDialog__
#define __ClassumSelectionCopyFromDialog__

class ClassumSelectionCopyFromDialog : public ClassumSelectionCopyFromDialogParent {
  public:
    ClassumSelectionCopyFromDialog(wxWindow* parent);
    void OnOKButtonClick(wxCommandEvent& event);
    void OnCancelButtonClick(wxCommandEvent& event);
    void FillWithSelections(int number_of_classes);

    int ReturnSelectedPosition( ) { return selected_selection_array_position; };

    wxArrayInt original_array_positions;
    int        selected_selection_array_position;
};

#endif // __ClassumSelectionCopyFromDialog__
