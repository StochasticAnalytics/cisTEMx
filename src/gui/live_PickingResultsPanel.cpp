//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

live_PickingResultsPanel::live_PickingResultsPanel(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
    : live_PickingResultsPanelParent(parent, id, pos, size, style) {
    //Bind(wxEVT_COMBOBOX, &ShowPickingResultsPanel::OnFitTypeRadioButton, this);
    //Bind(wxEVT_COMBOBOX, &ShowPickingResultsPanel::OnFitTypeRadioButton, this);

    //CTF2DResultsPanel->font_size_multiplier = 1.5;

    PickingResultsImagePanel->UnsetToolTip( );

    LowResFilterTextCtrl->SetMinMaxValue(0.1, FLT_MAX);
    LowResFilterTextCtrl->SetPrecision(0);
}

live_PickingResultsPanel::~live_PickingResultsPanel( ) {
    //Unbind(wxEVT_COMBOBOX, &ShowCTFResultsPanel::OnFitTypeRadioButton, this);
    //Unbind(wxEVT_COMBOBOX, &ShowCTFResultsPanel::OnFitTypeRadioButton, this);
    PickingResultsImagePanel->UnsetToolTip( );
}

void live_PickingResultsPanel::Clear( ) {
    PickingResultsImagePanel->should_show = false;
    PickingResultsImagePanel->UnsetToolTip( );
    Refresh( );
}

void live_PickingResultsPanel::Draw(const wxString& image_filename, ArrayOfParticlePositionAssets& array_of_assets, const float particle_radius_in_angstroms, const float pixel_size_in_angstroms, CTF micrograph_ctf, int image_asset_id, float iciness) {

    // Don't do this - it deallocates the images
    //PickingResultsImagePanel->Clear();

    PickingResultsImagePanel->SetImageFilename(image_filename, pixel_size_in_angstroms, micrograph_ctf);
    PickingResultsImagePanel->SetParticleCoordinatesAndRadius(array_of_assets, particle_radius_in_angstroms);
    PickingResultsImagePanel->UpdateScalingAndDimensions( );
    PickingResultsImagePanel->UpdateImageInBitmap( );
    PickingResultsImagePanel->should_show = true;
    PickingResultsImagePanel->SetToolTip(wxString::Format(wxT("%i coordinates picked"), int(array_of_assets.GetCount( ))));
    SetNumberOfPickedCoordinates(int(array_of_assets.Count( )));
    SetImageAssetID(image_asset_id);
    SetDefocus(0.5 * (micrograph_ctf.GetDefocus1( ) + micrograph_ctf.GetDefocus2( )) * pixel_size_in_angstroms);
    SetIciness(iciness);

    PickingResultsImagePanel->Refresh( );
}

void live_PickingResultsPanel::OnCirclesAroundParticlesCheckBox(wxCommandEvent& event) {
    if ( CirclesAroundParticlesCheckBox->IsChecked( ) ) {
        if ( ! PickingResultsImagePanel->draw_circles_around_particles ) {
            PickingResultsImagePanel->draw_circles_around_particles = true;
            PickingResultsImagePanel->Refresh( );
        }
    }
    else {
        if ( PickingResultsImagePanel->draw_circles_around_particles ) {
            PickingResultsImagePanel->draw_circles_around_particles = false;
            PickingResultsImagePanel->Refresh( );
        }
    }
}

void live_PickingResultsPanel::OnScaleBarCheckBox(wxCommandEvent& event) {
    if ( ScaleBarCheckBox->IsChecked( ) ) {
        if ( ! PickingResultsImagePanel->draw_scale_bar ) {
            PickingResultsImagePanel->draw_scale_bar = true;
            PickingResultsImagePanel->Refresh( );
        }
    }
    else {
        if ( PickingResultsImagePanel->draw_scale_bar ) {
            PickingResultsImagePanel->draw_scale_bar = false;
            PickingResultsImagePanel->Refresh( );
        }
    }
}

void live_PickingResultsPanel::OnLowPassEnter(wxCommandEvent& event) {
    if ( PickingResultsImagePanel->should_low_pass == true ) {
        PickingResultsImagePanel->low_res_filter_value = LowResFilterTextCtrl->ReturnValue( );
        PickingResultsImagePanel->UpdateImageInBitmap(true);
        PickingResultsImagePanel->Refresh( );
    }
    event.Skip( );
}

void live_PickingResultsPanel::OnLowPassKillFocus(wxFocusEvent& event) {
    if ( PickingResultsImagePanel->should_low_pass == true ) {
        PickingResultsImagePanel->low_res_filter_value = LowResFilterTextCtrl->ReturnValue( );
        PickingResultsImagePanel->UpdateImageInBitmap(true);
        PickingResultsImagePanel->Refresh( );
    }
    event.Skip( );
}

void live_PickingResultsPanel::OnHighPassFilterCheckBox(wxCommandEvent& event) {
    if ( HighPassFilterCheckBox->IsChecked( ) ) {
        if ( ! PickingResultsImagePanel->should_high_pass ) {
            PickingResultsImagePanel->should_high_pass = true;
            PickingResultsImagePanel->UpdateImageInBitmap(true);
            PickingResultsImagePanel->Refresh( );
        }
    }
    else {
        if ( PickingResultsImagePanel->should_high_pass ) {
            PickingResultsImagePanel->should_high_pass = false;
            PickingResultsImagePanel->UpdateImageInBitmap(true);
            PickingResultsImagePanel->Refresh( );
        }
    }
}

void live_PickingResultsPanel::OnWienerFilterCheckBox(wxCommandEvent& event) {
    if ( WienerFilterCheckBox->IsChecked( ) ) {
        if ( ! PickingResultsImagePanel->should_wiener_filter ) {
            PickingResultsImagePanel->should_wiener_filter = true;
            PickingResultsImagePanel->UpdateImageInBitmap(true);
            PickingResultsImagePanel->Refresh( );
        }
    }
    else {
        if ( PickingResultsImagePanel->should_wiener_filter ) {
            PickingResultsImagePanel->should_wiener_filter = false;
            PickingResultsImagePanel->UpdateImageInBitmap(true);
            PickingResultsImagePanel->Refresh( );
        }
    }
}

void live_PickingResultsPanel::OnLowPassFilterCheckBox(wxCommandEvent& event) {
    if ( LowPassFilterCheckBox->IsChecked( ) ) {
        LowResFilterTextCtrl->Enable(true);
        LowAngstromStatic->Enable(true);

        PickingResultsImagePanel->low_res_filter_value = LowResFilterTextCtrl->ReturnValue( );

        PickingResultsImagePanel->should_low_pass = true;
        PickingResultsImagePanel->UpdateImageInBitmap(true);
        PickingResultsImagePanel->Refresh( );
    }
    else {
        PickingResultsImagePanel->low_res_filter_value = -1.0;
        PickingResultsImagePanel->should_low_pass      = false;
        PickingResultsImagePanel->UpdateImageInBitmap(true);
        PickingResultsImagePanel->Refresh( );
    }
}

void live_PickingResultsPanel::OnUndoButtonClick(wxCommandEvent& event) {
    PickingResultsImagePanel->StepBackwardInHistoryOfParticleCoordinates( );
}

void live_PickingResultsPanel::OnRedoButtonClick(wxCommandEvent& event) {
    PickingResultsImagePanel->StepForwardInHistoryOfParticleCoordinates( );
}

void live_PickingResultsPanel::SetNumberOfPickedCoordinates(int number_of_coordinates) {
    NumberOfPicksStaticText->SetLabel(wxString::Format(wxT("%i picked coordinates"), number_of_coordinates));
}

void live_PickingResultsPanel::SetImageAssetID(int image_asset_id) {
    ImageIDStaticText->SetLabel(wxString::Format(wxT("Image ID: %i"), image_asset_id));
}

void live_PickingResultsPanel::SetIciness(float iciness) {
    IcinessStaticText->SetLabel(wxString::Format(wxT("Iciness: %.2f"), iciness));
}

void live_PickingResultsPanel::SetDefocus(float defocus_in_angstroms) {
    DefocusStaticText->SetLabel(wxString::Format(wxT("Defocus: %.2f μm"), defocus_in_angstroms / 10000.0));
}
