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

/*  \brief  Master Job Controller for the gui, for tracking and communicating with all running jobs.


*/

#define MAX_GUI_JOBS 1000

class GuiJob {

  public:
    bool          is_active;
    unsigned char job_code[SOCKET_CODE_SIZE];

    JobPanel*     parent_panel;
    wxSocketBase* socket;

    wxString launch_command;
    wxString gui_address;

    GuiJob( );
    GuiJob(JobPanel* wanted_parent_panel);

    void LaunchJob( );
    void KillJob( );
};

class GuiJobController {

  public:
    long job_index_tracker;
    long number_of_running_jobs;

    GuiJob job_list[MAX_GUI_JOBS]; // FIXED NUMBER - BAD!

    GuiJobController( );

    long AddJob(JobPanel* wanted_parent_panel, wxString wanted_launch_command, wxString wanted_gui_address);
    //void LaunchJob(unsigned char *job_code,  wxString launch_command);
    bool          LaunchJob(GuiJob* job_to_launch);
    long          FindFreeJobSlot( );
    long          ReturnJobNumberFromJobCode(unsigned char* job_code);
    void          GenerateJobCode(unsigned char* job_code);
    void          KillJob(int job_to_kill);
    void          KillJobIfSocketExists(wxSocketBase* socket);
    wxSocketBase* ReturnJobSocketPointer(int wanted_job);
};
