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

#ifndef __MyVolumeChooserDialog__
#define __MyVolumeChooserDialog__

#include "ProjectX_gui.h"

class MyVolumeChooserDialog : public VolumeChooserDialog {

  public:
    long     selected_volume_id;
    wxString selected_volume_name;

    MyVolumeChooserDialog(wxWindow* parent);
    virtual void OnCancelClick(wxCommandEvent& event);
    virtual void OnRenameClick(wxCommandEvent& event);
};

#endif
