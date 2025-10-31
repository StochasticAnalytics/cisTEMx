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

#include "../constants/constants.h"

class Project {

  public:
    Database database;

    bool     is_open;
    wxString project_name;

    wxFileName project_directory;
    wxFileName movie_asset_directory;
    wxFileName image_asset_directory;
    wxFileName template_matching_asset_directory;
    wxFileName phase_difference_asset_directory;
    wxFileName volume_asset_directory;
    wxFileName ctf_asset_directory;
    wxFileName particle_position_asset_directory;
    wxFileName particle_stack_directory;
    wxFileName class_average_directory;

    wxFileName parameter_file_directory;
    wxFileName scratch_directory;

    double total_cpu_hours;
    int    total_jobs_run;

    int      integer_database_version;
    wxString cistem_version_text;
    wxString current_workflow; // It would be better to connect this somehow to main_frame.current_workflow or vice-versa

    Project( );
    ~Project( );

    void Close(bool remove_lock = true, bool update_statistics = true);
    bool CreateNewProject(wxFileName database_file, wxString project_directory, wxString project_name);
    bool OpenProjectFromFile(wxFileName file_to_open);
    bool ReadMasterSettings( );
    void WriteProjectStatisticsToDatabase( );

    inline bool RecordCurrentWorkflowInDB(wxString workflow) { return database.RecordCurrentWorkflowInDB(workflow); }
};
