#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

extern MyMainFrame*        main_frame;
extern MyRunProfilesPanel* run_profiles_panel;

GenericRunnerPanel::GenericRunnerPanel(wxWindow* parent, wxWindowID id,
                                       const wxPoint& pos, const wxSize& size, long style)
    : JobPanel(parent, id, pos, size, style) {

    run_profiles_are_dirty = true;
    expected_result_count  = 0;
    received_result_count  = 0;
    passed_result_count    = 0;

    wxBoxSizer* main_sizer = new wxBoxSizer(wxVERTICAL);

    // Run profile selection + start button
    wxBoxSizer* top_sizer = new wxBoxSizer(wxHORIZONTAL);
    top_sizer->Add(new wxStaticText(this, wxID_ANY, "Run Profile:"), 0, wxALIGN_CENTER_VERTICAL | wxALL, 5);
    RunProfileComboBox = new wxComboBox(this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY);
    top_sizer->Add(RunProfileComboBox, 1, wxALL | wxEXPAND, 5);
    StartButton = new wxButton(this, wxID_ANY, "Start");
    top_sizer->Add(StartButton, 0, wxALL, 5);

    main_sizer->Add(top_sizer, 0, wxEXPAND);

    // Output display
    output_textctrl = new wxTextCtrl(this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_MULTILINE | wxTE_READONLY | wxTE_RICH2);
    main_sizer->Add(output_textctrl, 1, wxALL | wxEXPAND, 5);

    SetSizer(main_sizer);
    Layout( );

    StartButton->Bind(wxEVT_BUTTON, &GenericRunnerPanel::OnStartButtonClick, this);
    Bind(wxEVT_UPDATE_UI, &GenericRunnerPanel::OnUpdateUI, this);
}

GenericRunnerPanel::~GenericRunnerPanel( ) {
}

void GenericRunnerPanel::FillRunProfileComboBox( ) {
    RunProfileComboBox->Clear( );
    for ( int i = 0; i < run_profiles_panel->run_profile_manager.number_of_run_profiles; i++ ) {
        RunProfileComboBox->Append(run_profiles_panel->run_profile_manager.run_profiles[i].name);
    }
    if ( RunProfileComboBox->GetCount( ) > 0 )
        RunProfileComboBox->SetSelection(0);
    run_profiles_are_dirty = false;
}

void GenericRunnerPanel::OnUpdateUI(wxUpdateUIEvent& event) {
    if ( run_profiles_are_dirty ) {
        FillRunProfileComboBox( );
    }
}

void GenericRunnerPanel::OnStartButtonClick(wxCommandEvent& event) {
    if ( RunProfileComboBox->GetSelection( ) == wxNOT_FOUND ) {
        WriteErrorText("Please select a run profile first.");
        return;
    }

    RunProfile active_run_profile = run_profiles_panel->run_profile_manager.run_profiles[RunProfileComboBox->GetSelection( )];
    int        number_of_jobs     = active_run_profile.ReturnTotalJobs( );

    output_textctrl->Clear( );
    expected_result_count = number_of_jobs;
    received_result_count = 0;
    passed_result_count   = 0;
    WriteInfoText(wxString::Format("Starting generic_runner with %i jobs...", number_of_jobs));

    // Reset with profile and program name, then add jobs
    current_job_package.Reset(active_run_profile, "generic_runner", number_of_jobs);

    // Send just the binary name — the worker resolves the full path relative to
    // its own executable location. This works across machines with different
    // filesystem layouts (e.g. local vs AWS) because the worker and test binary
    // are always co-located in the same build/install directory.
    for ( int job_counter = 0; job_counter < number_of_jobs; job_counter++ ) {
        current_job_package.AddJob("ti", "console_test", number_of_jobs);
    }

    // Launch via job controller
    my_job_id = main_frame->job_controller.AddJob(this,
                                                  active_run_profile.manager_command,
                                                  active_run_profile.gui_address);

    if ( my_job_id != -1 ) {
        running_job = true;
        StartButton->Enable(false);
        SetNumberConnectedTextToZeroAndStartTracking( );
    }
    else {
        WriteErrorText("Failed to start job controller.");
    }
}

void GenericRunnerPanel::OnSocketJobResultQueueMsg(ArrayofJobResults& received_queue) {
    // Receives batched results relayed by the master via AddJobToResultQueue.
    // Encoding: float[0]=exit_code, float[1]=job_number, float[2]=string_length,
    //           float[3]=checksum, float[4..N]=packed stdout bytes (4 chars per float)
    for ( int i = 0; i < received_queue.GetCount( ); i++ ) {
        if ( received_queue[i].result_size >= 4 ) {
            int          exit_code     = static_cast<int>(received_queue[i].result_data[0]);
            int          job_number    = static_cast<int>(received_queue[i].result_data[1]);
            int          string_len    = static_cast<int>(received_queue[i].result_data[2]);
            unsigned int sent_checksum = static_cast<unsigned int>(received_queue[i].result_data[3]);
            received_result_count++;

            wxString status = (exit_code == 0) ? "PASSED" : "FAILED";
            if ( exit_code == 0 )
                passed_result_count++;
            WriteInfoText("");
            WriteInfoText("");
            WriteInfoText("");
            WriteInfoText(wxString::Format("=== Worker %i result (%s, exit code %i, %i bytes) ===",
                                           job_number, status, exit_code, string_len));

            // Decode packed stdout string and verify checksum
            if ( string_len > 0 && received_queue[i].result_size >= 4 + (string_len + 3) / 4 ) {
                char* decoded = new char[string_len + 1];
                memcpy(decoded, &received_queue[i].result_data[4], string_len);
                decoded[string_len] = '\0';

                // Verify round-trip checksum
                unsigned int computed_checksum = 0;
                for ( int c = 0; c < string_len; c++ ) {
                    computed_checksum += static_cast<unsigned char>(decoded[c]);
                }

                if ( computed_checksum == sent_checksum ) {
                    WriteInfoText(wxString::Format("[Worker %i] Checksum verified (%u) - %i bytes round-tripped intact",
                                                   job_number, computed_checksum, string_len));
                }
                else {
                    WriteErrorText(wxString::Format("[Worker %i] CHECKSUM MISMATCH: sent=%u received=%u",
                                                    job_number, sent_checksum, computed_checksum));
                }

                // Display stdout with worker prefix on each line
                wxString          output_str = wxString::FromUTF8(decoded);
                wxStringTokenizer tokenizer(output_str, "\n");
                while ( tokenizer.HasMoreTokens( ) ) {
                    wxString line = tokenizer.GetNextToken( );
                    if ( ! line.IsEmpty( ) ) {
                        WriteInfoText(wxString::Format("  [W%i] %s", job_number, line));
                    }
                }

                delete[] decoded;
            }
        }
    }
}

void GenericRunnerPanel::HandleSocketDisconnect(wxSocketBase* connected_socket) {
    // Re-enable StartButton on disconnect so panel is not stuck
    WriteErrorText("Error: Controller disconnected.");
    StartButton->Enable(true);
    JobPanel::HandleSocketDisconnect(connected_socket);
}

void GenericRunnerPanel::WriteInfoText(wxString text_to_write) {
    output_textctrl->SetDefaultStyle(wxTextAttr(*wxBLACK));
    output_textctrl->AppendText(text_to_write);
    if ( text_to_write.EndsWith("\n") == false )
        output_textctrl->AppendText("\n");
}

void GenericRunnerPanel::WriteErrorText(wxString text_to_write) {
    output_textctrl->SetDefaultStyle(wxTextAttr(*wxRED));
    output_textctrl->AppendText(text_to_write);
    if ( text_to_write.EndsWith("\n") == false )
        output_textctrl->AppendText("\n");
}

void GenericRunnerPanel::SetNumberConnectedText(wxString wanted_text) {
    WriteInfoText(wanted_text);
}

void GenericRunnerPanel::SetTimeRemainingText(wxString wanted_text) {
    // Minimal panel: no time display
}

void GenericRunnerPanel::OnSocketAllJobsFinished( ) {
    int failed_count = received_result_count - passed_result_count;

    WriteInfoText("");
    WriteInfoText("");
    WriteInfoText("");
    WriteInfoText(wxString::Format("Network test complete: %i/%i workers reported.",
                                   received_result_count, expected_result_count));
    WriteInfoText(wxString::Format("  PASSED: %i  FAILED: %i", passed_result_count, failed_count));

    if ( received_result_count != expected_result_count ) {
        WriteErrorText(wxString::Format("  Warning: %i workers did not report results.",
                                        expected_result_count - received_result_count));
    }

    running_job = false;
    StartButton->Enable(true);
}
