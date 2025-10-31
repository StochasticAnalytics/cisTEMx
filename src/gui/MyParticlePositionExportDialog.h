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

#ifndef __MyParticlePositionExportDialog__
#define __MyParticlePositionExportDialog__

class MyParticlePositionExportDialog : public ParticlePositionExportDialog {
  public:
    MyParticlePositionExportDialog(wxWindow* parent);

    void OnCancelButtonClick(wxCommandEvent& event);
    void OnExportButtonClick(wxCommandEvent& event);
    void OnDirChanged(wxFileDirPickerEvent& event);
};

#endif
