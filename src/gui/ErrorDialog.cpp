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

//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

MyErrorDialog::MyErrorDialog(wxWindow* parent)
    : ErrorDialog(parent) {
}

void MyErrorDialog::OnClickOK(wxCommandEvent& event) {
    Destroy( );
}
