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

class JobTracker {

  public:
    int total_number_of_jobs;
    int total_running_processes;
    int total_number_of_finished_jobs;
    int last_update_time_total_running_processes;

    long start_time;
    long last_update_time;

    float last_update_seconds_per_job_per_process;
    long  time_of_last_remaining_time_call;

    int        old_percent_complete;
    wxTimeSpan old_time_remaining;
    wxTimeSpan time_remaining;

    JobTracker( );
    ~JobTracker( );

    void       StartTracking(int wanted_total_number_of_jobs);
    void       AddConnection( );
    void       MarkJobFinished( );
    wxTimeSpan ReturnRemainingTime( );
    wxTimeSpan ReturnTimeSinceStart( );

    inline int ReturnPercentCompleted( ) {
        int percent_completed = myround((float(total_number_of_finished_jobs) / float(total_number_of_jobs)) * 100.0);
        if ( percent_completed > 100 )
            percent_completed = 100;
        else if ( percent_completed < 0 )
            percent_completed = 0;

        return percent_completed;
    }

    bool ShouldUpdate( );
};
