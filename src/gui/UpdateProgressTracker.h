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

#ifndef _SRC_GUI_UPDATE_PROGRESS_TRACKER_H
#define _SRC_GUI_UPDATE_PROGRESS_TRACKER_H

// This class exists only as an interface for overriding in MainFrame.cpp.
// Its main purpose is to allow the GUI to track database schema update
// progress without exposing any more GUI code to the database than is
// strictly necessary to avoid any weird dependency and other issues,
// such as substantially increased compile time.
// It is possible to generalize this for database-releated tracking
class UpdateProgressTracker {
  public:
    virtual void OnUpdateProgress(int progress, wxString new_msg, bool& should_update_text) = 0;
    virtual void OnCompletion( )                                                            = 0;
};
#endif