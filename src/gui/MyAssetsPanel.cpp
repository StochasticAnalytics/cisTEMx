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

#include "../core/gui_core_headers.h"

MyAssetsPanel::MyAssetsPanel(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
    : AssetsPanel(parent, id, pos, size, style) {
    // Bind OnListBookPageChanged from
    Bind(wxEVT_LISTBOOK_PAGE_CHANGED, wxBookCtrlEventHandler(MyAssetsPanel::OnAssetsBookPageChanged), this);
}

// TODO: destructor

void MyAssetsPanel::OnAssetsBookPageChanged(wxBookCtrlEvent& event) {
    extern MyMovieAssetPanel*             movie_asset_panel;
    extern MyImageAssetPanel*             image_asset_panel;
    extern MyParticlePositionAssetPanel*  particle_position_asset_panel;
    extern MyVolumeAssetPanel*            volume_asset_panel;
    extern AtomicCoordinatesAssetPanel*   atomic_coordinates_asset_panel;
    extern MyRefinementPackageAssetPanel* refinement_package_asset_panel;

#ifdef __WXOSX__
    // Necessary for MacOS to refresh the panels
    if ( event.GetSelection( ) == 0 ) {
        movie_asset_panel->Layout( );
        movie_asset_panel->Refresh( );
    }
    else if ( event.GetSelection( ) == 1 ) {
        image_asset_panel->Layout( );
        image_asset_panel->Refresh( );
    }
    else if ( event.GetSelection( ) == 2 ) {
        particle_position_asset_panel->Layout( );
        particle_position_asset_panel->Refresh( );
    }
    else if ( event.GetSelection( ) == 3 ) {
        volume_asset_panel->Layout( );
        volume_asset_panel->Refresh( );
    }
    else if ( event.GetSelection( ) == 4 ) {
        refinement_package_asset_panel->Layout( );
        refinement_package_asset_panel->Refresh( );
    }
#endif
}
