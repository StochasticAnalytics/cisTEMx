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

MyAddRunCommandDialog::MyAddRunCommandDialog(MyRunProfilesPanel* parent)
    : AddRunCommandDialog(parent) {
    my_parent = parent;
}

void MyAddRunCommandDialog::OnOKClick(wxCommandEvent& event) {
    ProcessResult( );
}

void MyAddRunCommandDialog::OnOverrideCheckbox(wxCommandEvent& event) {
    if ( OverrideCheckBox->IsChecked( ) == true )
        OverridenNoCopiesSpinCtrl->Enable(true);
    else
        OverridenNoCopiesSpinCtrl->Enable(false);
}

void MyAddRunCommandDialog::OnCancelClick(wxCommandEvent& event) {
    EndModal(0);
    Destroy( );
}

void MyAddRunCommandDialog::OnEnter(wxCommandEvent& event) {
    ProcessResult( );
}

void MyAddRunCommandDialog::ProcessResult( ) {

    // check that the text box contains $command..

    if ( CommandTextCtrl->GetValue( ).Find("$command") == wxNOT_FOUND ) {
        ErrorStaticText->Show(true);
        Layout( );
        Fit( );
    }
    else {
        if ( my_parent->buffer_profile.run_commands[my_parent->selected_command].command_to_run == CommandTextCtrl->GetValue( ) && my_parent->buffer_profile.run_commands[my_parent->selected_command].number_of_copies == NumberCopiesSpinCtrl->GetValue( ) && my_parent->buffer_profile.run_commands[my_parent->selected_command].delay_time_in_ms == DelayTimeSpinCtrl->GetValue( ) && my_parent->buffer_profile.run_commands[my_parent->selected_command].number_of_threads_per_copy == NumberThreadsSpinCtrl->GetValue( ) && my_parent->buffer_profile.run_commands[my_parent->selected_command].overriden_number_of_copies == OverridenNoCopiesSpinCtrl->GetValue( ) && my_parent->buffer_profile.run_commands[my_parent->selected_command].override_total_copies == OverrideCheckBox->GetValue( ) ) {
            // nothing changed just exit..
            EndModal(0);
            Destroy( );
        }
        else {
            //Update and exit..

            my_parent->buffer_profile.run_commands[my_parent->selected_command].command_to_run             = CommandTextCtrl->GetValue( );
            my_parent->buffer_profile.run_commands[my_parent->selected_command].number_of_copies           = NumberCopiesSpinCtrl->GetValue( );
            my_parent->buffer_profile.run_commands[my_parent->selected_command].delay_time_in_ms           = DelayTimeSpinCtrl->GetValue( );
            my_parent->buffer_profile.run_commands[my_parent->selected_command].number_of_threads_per_copy = NumberThreadsSpinCtrl->GetValue( );
            my_parent->buffer_profile.run_commands[my_parent->selected_command].overriden_number_of_copies = OverridenNoCopiesSpinCtrl->GetValue( );
            my_parent->buffer_profile.run_commands[my_parent->selected_command].override_total_copies      = OverrideCheckBox->GetValue( );

            my_parent->FillCommandsBox( );
            my_parent->command_panel_has_changed = true;
            EndModal(0);
            Destroy( );
        }
    }
}
