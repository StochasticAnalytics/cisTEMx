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

class RunCommand {

  public:
    RunCommand( );
    ~RunCommand( );

    wxString command_to_run;
    int      number_of_copies;
    int      number_of_threads_per_copy;
    bool     override_total_copies;
    int      overriden_number_of_copies;
    int      delay_time_in_ms;

    bool operator==(const RunCommand& other) const;
    bool operator!=(const RunCommand& other) const;

    void SetCommand(wxString wanted_command, int wanted_number_of_copies, int wanted_number_of_threads_per_copy, bool wanted_override_total_copies, int wanted_overriden_number_of_copies, int wanted_delay_time_in_ms);
};