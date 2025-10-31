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

#ifndef _SRC_GUI_DATABASE_UDPATE_DIALOG_H_
#define _SRC_GUI_DATABASE_UPDATE_DIALOG_H_

#include "ProjectX_gui_main.h"

class DatabaseUpdateDialog : public DatabaseUpdateDialogParent {
  public:
    // DatabaseUpdateDialog(wxWindow* parent);
    DatabaseUpdateDialog(wxWindow* parent, wxString db_changes);

    void OnButtonClicked(wxCommandEvent& event);

  private:
    wxString schema_changes;
};
#endif