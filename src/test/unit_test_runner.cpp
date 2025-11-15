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

#define CATCH_CONFIG_RUNNER
#include "../../include/catch2/catch.hpp"
#include <wx/app.h>

// Minimal wxApp for unit tests - provides traits/locale for wxString, wxFileName, etc.
class MinimalTestApp : public wxAppConsole {
  public:
    virtual bool OnInit( ) override { return true; }
};

IMPLEMENT_APP_NO_MAIN(MinimalTestApp)

int main(int argc, char* argv[]) {
    // Initialize wxWidgets for console mode
    wxApp::SetInstance(new MinimalTestApp( ));
    wxEntryStart(argc, argv);
    wxTheApp->CallOnInit( );

    // Run Catch2 tests
    int result = Catch::Session( ).run(argc, argv);

    // Cleanup wxWidgets
    wxTheApp->OnExit( );
    wxEntryCleanup( );

    return result;
}