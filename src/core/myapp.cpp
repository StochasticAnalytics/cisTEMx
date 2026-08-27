#include "core_headers.h"
#include <wx/evtloop.h>
#include <unistd.h> // sysconf, for the RSS diagnostics in HandleSocketProgramDefinedResult

wxDEFINE_EVENT(wxEVT_COMMAND_MYTHREAD_COMPLETED, wxThreadEvent);
wxDEFINE_EVENT(wxEVT_COMMAND_MYTHREAD_ENDING, wxThreadEvent);
wxDEFINE_EVENT(wxEVT_COMMAND_MYTHREAD_SEND_IMAGE_RESULT, wxThreadEvent);
wxDEFINE_EVENT(wxEVT_COMMAND_MYTHREAD_SENDERROR, wxThreadEvent);
wxDEFINE_EVENT(wxEVT_COMMAND_MYTHREAD_SENDINFO, wxThreadEvent);
wxDEFINE_EVENT(wxEVT_COMMAND_MYTHREAD_INTERMEDIATE_RESULT_AVAILABLE, wxThreadEvent);
wxDEFINE_EVENT(wxEVT_COMMAND_MYTHREAD_SEND_PROGRAM_DEFINED_RESULT, ReturnProgramDefinedResultEvent);

#define THREAD_START_NEXT_JOB 0
#define THREAD_DIE 1
#define THREAD_SLEEP 2

bool MyApp::OnInit( ) {
    long counter;
    thread_next_action = THREAD_SLEEP;

    number_of_dispatched_jobs         = 0;
    number_of_finished_jobs           = 0;
    number_of_timing_results_received = 0;

    max_number_of_connected_workers = 0;

    zombie_timer = NULL;
    // Never assigned on the non-compute leader (it starts no CalculateThread), and every
    // teardown path calls work_thread->Kill() behind a != NULL guard - uninitialized stack
    // garbage here made that a wild pthread kill (the leader-only teardown segfault, and a
    // likely source of the historical intermittent heap corruption).
    work_thread            = NULL;
    queue_timer            = NULL;
    queue_timer_set        = false;
    master_queue_timer_set = false;

    controller_socket = NULL;
    master_socket     = NULL;

    connected_to_the_master   = false;
    currently_running_a_job   = false;
    i_am_a_non_compute_leader = false;

    // i_am_the_master was never initialized anywhere: the only assignment in this file is
    // "= true" in HandleSocketYouAreTheMaster, so until a process is elected master this
    // read indeterminate memory. It happens to work because a fresh heap is usually zero,
    // but every "am I the master?" branch in this file was resting on that. Same bug class
    // as the work_thread one above.
    i_am_the_master = false;

    time_of_last_queue_send        = 0;
    time_of_last_master_queue_send = 0;
    number_of_results_sent         = 0;

    total_milliseconds_spent_on_threads = 0;

    socket_to_worker_job_pointer_hash.clear( );

    worker_liveness_timer = NULL;
    liveness_ping_sent_at_ms.clear( );
    number_of_abandoned_jobs = 0;
    run_unfinishable_strikes = 0;

    jobs_to_redispatch.clear( );
    job_dispatch_attempts.clear( );
    max_job_redispatch_tries = ReturnMaxJobRedispatchTries( );

    inter_thread_message_queue.Post(0);

    ActivateMKLDebugForNonIntelCPU( ); // if not Intel CPU and if using the MKL attempt to set an environment variable that can lead to substanstial speedup.
    ProgramSpecificInit( );

    return true;
}

int MyApp::OnExit( ) {
    ProgramSpecificCleanUp( );
    return 0;
}

void MyApp::OnEventLoopEnter(wxEventLoopBase* loop) {
    if ( loop->IsMain( ) == true ) {
        // initialise sockets, and set the event handler for SocketCommunicator

        wxSocketBase::Initialize( );
        brother_event_handler = this;

        int  parse_status;
        int  number_of_arguments;
        int  counter;
        long temp_long;

        wxString      current_address;
        wxArrayString possible_controller_addresses;
        wxIPV4address junk_address;

        socket_to_worker_job_pointer_hash.clear( );

        // Bind the thread events

        Bind(wxEVT_COMMAND_MYTHREAD_COMPLETED, &MyApp::OnThreadComplete, this); // Called when DoCalculation finishes
        Bind(wxEVT_COMMAND_MYTHREAD_SENDERROR, &MyApp::OnThreadSendError, this);
        Bind(wxEVT_COMMAND_MYTHREAD_SENDINFO, &MyApp::OnThreadSendInfo, this);
        Bind(wxEVT_COMMAND_MYTHREAD_ENDING, &MyApp::OnThreadEnding, this); // When thread is about to die
        Bind(wxEVT_COMMAND_MYTHREAD_INTERMEDIATE_RESULT_AVAILABLE, &MyApp::OnThreadIntermediateResultAvailable, this);
        Bind(wxEVT_COMMAND_MYTHREAD_SEND_IMAGE_RESULT, &MyApp::OnThreadSendImageResult, this);
        Bind(wxEVT_COMMAND_MYTHREAD_SEND_PROGRAM_DEFINED_RESULT, &MyApp::OnThreadSendProgramDefinedResult, this);

        // Connect to the controller program..

        command_line_parser.SetCmdLine(argc, argv);
        command_line_parser.AddParam("controller_address", wxCMD_LINE_VAL_STRING, wxCMD_LINE_PARAM_OPTIONAL);
        command_line_parser.AddParam("controller_port", wxCMD_LINE_VAL_NUMBER, wxCMD_LINE_PARAM_OPTIONAL);
        command_line_parser.AddParam("job_code", wxCMD_LINE_VAL_STRING, wxCMD_LINE_PARAM_OPTIONAL);
        command_line_parser.AddParam("wanted_number_of_threads", wxCMD_LINE_VAL_NUMBER, wxCMD_LINE_PARAM_OPTIONAL);

        // Let the app add options
        AddCommandLineOptions( );

        //wxPrintf("\n");

        parse_status        = command_line_parser.Parse(true);
        number_of_arguments = command_line_parser.GetParamCount( );

        if ( parse_status != 0 ) {
            wxPrintf("\n\n");
            ExitMainLoop( );
            exit(0);
            return;
        }

        // if we have no arguments run interactively.. if we have 4 continue as though we have network info, else error..

        if ( number_of_arguments == 0 ) {
            is_running_locally = true;
            DoInteractiveUserInput( );
            stopwatch.Start( );
            DoCalculation( );
            total_milliseconds_spent_on_threads += stopwatch.Time( );
            MyInteractiveProgramCleanup( );
            fftwf_cleanup( ); // this is needed to stop valgrind reporting memory leaks..
            exit(0);
        }
        else if ( number_of_arguments != 4 ) {
            command_line_parser.Usage( );
            wxPrintf("\n\n");
            ExitMainLoop( );
            exit(0);
            return;
        }

        is_running_locally = false;

        // get the address and port of the job controller (should be command line options).

        wxStringTokenizer controller_ip_address_tokens(command_line_parser.GetParam(0), ",");

        while ( controller_ip_address_tokens.HasMoreTokens( ) == true ) {
            current_address = controller_ip_address_tokens.GetNextToken( );
            possible_controller_addresses.Add(current_address);
            if ( junk_address.Hostname(current_address) == false ) {
                MyDebugPrint(" Error: Address (%s) - not recognized as an IP or hostname\n\n", current_address);
                exit(-1);
            };
        }

        if ( command_line_parser.GetParam(1).ToLong(&controller_port) == false ) {
            MyPrintWithDetails(" Error: Port (%s) - not recognized as a port\n\n", command_line_parser.GetParam(1));
            exit(-1);
        }

        if ( command_line_parser.GetParam(2).Len( ) != SOCKET_CODE_SIZE ) {
            {
                MyPrintWithDetails(" Error: Code (%s) - is the incorrect length (%li instead of %i)\n\n", command_line_parser.GetParam(2), command_line_parser.GetParam(2).Len( ), SOCKET_CODE_SIZE);
                exit(-1);
            }
        }

        if ( command_line_parser.GetParam(3).ToLong(&temp_long) == false ) {
            MyPrintWithDetails(" Error: No. of Threads (%s) - not recognized as a number\n\n", command_line_parser.GetParam(3));
            exit(-1);
        }

        number_of_threads_requested_on_command_line = temp_long;
        if ( number_of_threads_requested_on_command_line < 1 )
            number_of_threads_requested_on_command_line = 1;

        // copy over job code.

        for ( counter = 0; counter < SOCKET_CODE_SIZE; counter++ ) {
            current_job_code[counter] = command_line_parser.GetParam(2).GetChar(counter);
        }

        // Attempt to connect to the controller..

        active_controller_address.Service(controller_port);
        is_connected = false;

        //MyDebugPrint("\n JOB : Trying to connect to %s:%i (timeout = 30 sec) ...\n", controller_address.IPAddress(), controller_address.Service());
        controller_socket = new wxSocketClient( );
        controller_socket->SetFlags(SOCKET_FLAGS);
        controller_socket->Notify(false);

        for ( counter = 0; counter < possible_controller_addresses.GetCount( ); counter++ ) {
            active_controller_address.Hostname(possible_controller_addresses.Item(counter));
            controller_socket->Connect(active_controller_address, false);
            controller_socket->WaitOnConnect(30);

            if ( controller_socket->IsConnected( ) == false ) {
                controller_socket->Close( );
                //wxPrintf("Connection Failed.\n\n");
            }
            else {
                break;
            }
        }

        controller_socket->SetFlags(SOCKET_FLAGS);

        if ( controller_socket->IsConnected( ) == false || controller_socket->IsOk( ) == false ) {
            controller_socket->Close( );
            MyDebugPrint(" JOB : Failed ! Unable to connect\n");
            ExitMainLoop( );
            exit(0);
            return;
        }

        // Monitor this connection..

        MonitorSocket(controller_socket);

        // we are apparently connected, but this can be a lie as a certain number of connections appear to just be accepted by the operating
        // system - if the port if valid.  So if we don't get any events from this socket within 10 seconds, we are going to try again...

        number_of_failed_connections = 0;
        i_am_a_zombie                = true;

        zombie_timer = new wxTimer(this, 1);
        zombie_timer->StartOnce(20000);

        // timer events

        Bind(wxEVT_TIMER, wxTimerEventHandler(MyApp::OnQueueTimer), this, 2);
        Bind(wxEVT_TIMER, wxTimerEventHandler(MyApp::OnZombieTimer), this, 1);
    }
}

// Placeholder (to be overridden) function to add options to the command line
void MyApp::AddCommandLineOptions( ) {
    return;
}

// How many times may a single job be re-sent after its first attempt?
// CISTEM_EXPERIMENTAL_FAILED_WORKER_RESUBMIT_TRIES, default 1. Zero disables
// re-dispatch entirely (pre-2026-08 behaviour: an orphaned job is lost and the
// run can never reach N of N). Negative or unparseable values fall back to the
// default rather than becoming an unbounded retry loop.
long MyApp::ReturnMaxJobRedispatchTries( ) {
    const long default_tries = 1;
    // Bounded above as well as below. Without a ceiling a single env var re-enables the
    // unbounded retry loop this exists to prevent, and max + 1 (used in the logs and the
    // budget arithmetic) is signed overflow at LONG_MAX. There is no real workflow that
    // wants one job attempted more times than this.
    const long  maximum_tries = 1000;
    const char* from_env      = getenv("CISTEM_EXPERIMENTAL_FAILED_WORKER_RESUBMIT_TRIES");

    if ( from_env == NULL )
        return default_tries;

    long     parsed_value;
    wxString value_as_string(from_env);

    if ( value_as_string.Trim( ).Trim(false).ToLong(&parsed_value) == false || parsed_value < 0 ) {
        wxPrintf("Warning: CISTEM_EXPERIMENTAL_FAILED_WORKER_RESUBMIT_TRIES = '%s' is not a non-negative integer - using the default of %li\n", from_env, default_tries);
        return default_tries;
    }

    if ( parsed_value > maximum_tries ) {
        wxPrintf("Warning: CISTEM_EXPERIMENTAL_FAILED_WORKER_RESUBMIT_TRIES = '%s' exceeds the maximum of %li - using %li\n", from_env, maximum_tries, maximum_tries);
        return maximum_tries;
    }

    return parsed_value;
}

// CISTEM_EXPERIMENTAL_WORKER_LIVENESS_TIMEOUT: seconds a job-holding worker may leave a
// liveness ping unanswered before the master presumes it dead and releases its job.
// Default 900 s - generous against every stall actually observed (the worst master merge
// backlog measured is ~6 minutes, and master-side stalls delay the ping and the deadline
// check identically, so they cannot false-positive). 0 disables the sweep entirely.
long MyApp::ReturnWorkerLivenessTimeoutSeconds( ) {
    const long  default_seconds = 900;
    const long  maximum_seconds = 86400;
    const char* from_env        = getenv("CISTEM_EXPERIMENTAL_WORKER_LIVENESS_TIMEOUT");

    if ( from_env == NULL )
        return default_seconds;

    long     parsed_value;
    wxString value_as_string(from_env);

    if ( value_as_string.Trim( ).Trim(false).ToLong(&parsed_value) == false || parsed_value < 0 ) {
        wxPrintf("Warning: CISTEM_EXPERIMENTAL_WORKER_LIVENESS_TIMEOUT = '%s' is not a non-negative integer - using the default of %li\n", from_env, default_seconds);
        return default_seconds;
    }

    if ( parsed_value > maximum_seconds ) {
        wxPrintf("Warning: CISTEM_EXPERIMENTAL_WORKER_LIVENESS_TIMEOUT = '%s' exceeds the maximum of %li - using %li\n", from_env, maximum_seconds, maximum_seconds);
        return maximum_seconds;
    }

    return parsed_value;
}

// Any process that is pinged answers from its MAIN thread (the monitor thread dispatches
// here via CallAfter), so a pong proves the process exists AND its event loop is alive. A
// worker whose calculation thread is busy computing still answers immediately.
void MyApp::HandleSocketLivenessPing(wxSocketBase* connected_socket) {
    WriteToSocket(connected_socket, socket_liveness_pong, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
}

void MyApp::HandleSocketLivenessPong(wxSocketBase* connected_socket) {
    liveness_ping_sent_at_ms.erase(connected_socket);
}

// Master only, every 3 minutes: (1) ping every socket that holds a job and presume dead
// any that has not answered within the liveness timeout - a worker that died behind a
// broken tunnel leaves the master's socket half-open forever, because the master only
// ever reads from a busy worker and a socket that is never written to never errors. The
// presumed-dead socket goes through HandleSocketDisconnect, so its job is released into
// the normal orphan/re-dispatch machinery. (2) Check whether the run can still finish at
// all, and end it loudly if not (see FailRunAsUnfinishable).
void MyApp::OnWorkerLivenessTimer(wxTimerEvent& event) {
    const long timeout_seconds = ReturnWorkerLivenessTimeoutSeconds( );

    if ( timeout_seconds > 0 ) {
        const long now_ms = long(wxGetLocalTimeMillis( ).GetValue( ));

        // Collect first: HandleSocketDisconnect mutates the hash this iterates.
        std::vector<wxSocketBase*> presumed_dead;

        for ( SocketJobPointerHash::iterator hash_entry = socket_to_worker_job_pointer_hash.begin( ); hash_entry != socket_to_worker_job_pointer_hash.end( ); ++hash_entry ) {
            wxSocketBase* worker_socket = hash_entry->first;

            if ( worker_socket == NULL )
                continue;

            // A socket the OS already knows is dead does not need the full ping timeout -
            // and writing a ping to it would just spray write errors into the log.
            if ( worker_socket->IsConnected( ) == false ) {
                presumed_dead.push_back(worker_socket);
                continue;
            }

            auto ping_entry = liveness_ping_sent_at_ms.find(worker_socket);

            if ( ping_entry == liveness_ping_sent_at_ms.end( ) ) {
                liveness_ping_sent_at_ms[worker_socket] = now_ms;
                WriteToSocket(worker_socket, socket_liveness_ping, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
            }
            else if ( now_ms - ping_entry->second > timeout_seconds * 1000 ) {
                presumed_dead.push_back(worker_socket);
            }
        }

        for ( wxSocketBase* dead_socket : presumed_dead ) {
            SocketSendError(wxString::Format("Error: a worker holding a job has not answered a liveness ping for over %li seconds - presuming it dead and releasing its job.", timeout_seconds));
            liveness_ping_sent_at_ms.erase(dead_socket);
            HandleSocketDisconnect(dead_socket);
        }
    }

    // Can this run still finish? Two ways it provably cannot:
    //   (1) every job not yet finished has been abandoned at the re-dispatch cap;
    //   (2) all jobs are dispatched, no worker is connected (so no one will ever ask for
    //       a re-dispatch), nothing is queued for merge, and the finish gate is short.
    // Both must hold on two consecutive ticks (>= 3 minutes apart) before acting, so an
    // in-flight late result or a straggler connection gets a grace window.
    bool run_is_unfinishable = false;

    if ( number_of_abandoned_jobs > 0 && number_of_finished_jobs + number_of_abandoned_jobs >= current_job_package.number_of_jobs )
        run_is_unfinishable = true;
    else if ( current_job_package.number_of_jobs > 0 &&
              number_of_dispatched_jobs >= current_job_package.number_of_jobs &&
              worker_socket_pointers.GetCount( ) == 0 &&
              number_of_finished_jobs < current_job_package.number_of_jobs &&
              queued_result_payload_count.load( ) == 0 )
        run_is_unfinishable = true;

    if ( run_is_unfinishable == true ) {
        run_unfinishable_strikes++;
        if ( run_unfinishable_strikes >= 2 )
            FailRunAsUnfinishable(wxString::Format("%li of %i jobs are finished, %li are permanently abandoned, and %i workers remain - no path to completion exists.",
                                                   number_of_finished_jobs, current_job_package.number_of_jobs, number_of_abandoned_jobs, int(worker_socket_pointers.GetCount( ))));
    }
    else {
        run_unfinishable_strikes = 0;
    }
}

// The all-done gate needs number_of_finished_jobs == number_of_jobs; when that has become
// unreachable the old behavior was to wait in the event loop forever, holding the fleet
// and its GPU slots until someone killed the run by hand (2026-08-27: an hour, 22 slots).
// End it instead: tell everyone to die, tear down, and exit with a code that says why.
void MyApp::FailRunAsUnfinishable(wxString reason) {
    SocketSendError("Error: this run can no longer finish - shutting it down. " + reason);
    wxPrintf("MASTER: run is unfinishable - shutting down. %s\n", reason);

    for ( int counter = 0; counter < worker_socket_pointers.GetCount( ); counter++ ) {
        if ( worker_socket_pointers[counter] == NULL )
            continue;
        WriteToSocket(worker_socket_pointers[counter], socket_time_to_die, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
        StopMonitoringAndDestroySocket(worker_socket_pointers[counter]);
    }
    worker_socket_pointers.Clear( );

    if ( controller_socket != NULL ) {
        StopMonitoringAndDestroySocket(controller_socket);
        controller_socket = NULL;
    }

    ShutDownServer( );
    ShutDownSocketMonitor( );

    if ( work_thread != NULL )
        work_thread->Kill( );

    ExitMainLoop( );
    exit(cistem::exit_code::run_unfinishable);
}

// Count one dispatch of job_index and return the resulting attempt number (1 on the
// first dispatch), or -1 if job_index is not a job in the current package.
//
// -1, not 1. Returning 1 for an unaccountable index looks conservative and is the exact
// opposite: the counter never advances, so "attempts <= max" stays true forever and the
// cap becomes an unbounded retry loop - the failure this whole mechanism exists to
// prevent. Callers must treat a negative as "do not retry": an index we cannot account
// for is precisely where a runaway would start, so it fails closed.
int MyApp::RecordJobDispatchAttempt(int job_index) {
    if ( job_index < 0 || job_index >= int(job_dispatch_attempts.size( )) )
        return -1;

    job_dispatch_attempts[job_index]++;
    return job_dispatch_attempts[job_index];
}

// How many times has job_index been dispatched so far? -1 if it is not a job in the
// current package; see RecordJobDispatchAttempt for why that must fail closed.
int MyApp::ReturnJobDispatchAttempts(int job_index) {
    if ( job_index < 0 || job_index >= int(job_dispatch_attempts.size( )) )
        return -1;

    return job_dispatch_attempts[job_index];
}

void MyApp::SendNextJobTo(wxSocketBase* socket) {
    // Jobs orphaned by a worker that died mid-flight are re-issued first: without this a
    // death after full dispatch loses the job forever and the run can never finish (the
    // all-done gate needs number_of_finished_jobs == number_of_jobs). Late-connecting
    // stragglers - which otherwise just get told to die - become the recovery path.
    if ( ! jobs_to_redispatch.empty( ) ) {
        RunJob* orphaned_job = jobs_to_redispatch.back( );
        jobs_to_redispatch.pop_back( );

        // Count this attempt and say so: a retried job that keeps coming back is
        // the signature of a poison job, and the log is how that gets noticed.
        // job_number IS the index into the package (JobPackage::AddJob sets
        // jobs[i].job_number = i), so use it rather than subtracting pointers - that
        // subtraction is only defined while the pointer is still inside this package's
        // array, which is not something this code can promise.
        const int attempts_for_this_job = RecordJobDispatchAttempt(orphaned_job->job_number);

        if ( attempts_for_this_job > 0 ) {
            SocketSendInfo(wxString::Format("Re-dispatching job %i (attempt %i of %li) after a worker was lost.",
                                            orphaned_job->job_number, attempts_for_this_job, max_job_redispatch_tries + 1));
        }
        else {
            // HandleSocketDisconnect refuses to queue a job it cannot account for, so
            // this should be unreachable; say so loudly rather than retry uncounted.
            SocketSendError(wxString::Format("Error: Re-dispatching job %i, but it has no entry in the attempt table - its retries are NOT being counted.",
                                             orphaned_job->job_number));
        }

        orphaned_job->SendJob(socket);
        socket_to_worker_job_pointer_hash[socket] = orphaned_job;
        // The worker just asked for work: that is proof of life, so clear any outstanding
        // liveness ping. This also prevents a recycled socket address inheriting a stale
        // ping timestamp from a previous worker and being presumed dead on arrival.
        liveness_ping_sent_at_ms.erase(socket);
        return;
    }

    // if we haven't dispatched all jobs yet, then send it, otherwise tell the worker to die..

    if ( number_of_dispatched_jobs < current_job_package.number_of_jobs ) {
        // See RunJob::SendJob() Doxygen for encoding order specification
        current_job_package.jobs[number_of_dispatched_jobs].SendJob(socket);
        socket_to_worker_job_pointer_hash[socket] = &current_job_package.jobs[number_of_dispatched_jobs];
        // Proof of life + guards against a recycled socket address inheriting a stale
        // ping timestamp (see the re-dispatch branch above).
        liveness_ping_sent_at_ms.erase(socket);

        RecordJobDispatchAttempt(int(number_of_dispatched_jobs));

        number_of_dispatched_jobs++;
    }
    else {
        WriteToSocket(socket, socket_time_to_die, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
        // stop monitoring the socket..
        //StopMonitoringSocket(socket); stopped doing this for timings

        // Remember that this socket doesn't have a job anymore
        socket_to_worker_job_pointer_hash.erase(socket);
        liveness_ping_sent_at_ms.erase(socket);
    }
}

void MyApp::SendJobFinished(int job_number) {
    //MyDebugAssertTrue(i_am_the_master == true, "SendJobFinished called by a worker!");

    WriteToSocket(controller_socket, socket_job_finished, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
    // send the job number of the current job..
    WriteToSocket(controller_socket, &job_number, sizeof(int), true, "SendJobNumber", FUNCTION_DETAILS_AS_WXSTRING);
}

void MyApp::SendJobResult(JobResult* result) {
    //MyDebugAssertTrue(i_am_the_master == true, "SendJobResult called by a worker!");

    WriteToSocket(controller_socket, socket_job_result, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
    // See JobResult::SendToSocket() Doxygen for encoding order specification
    result->SendToSocket(controller_socket);
}

void MyApp::SendJobResultQueue(ArrayofJobResults& queue_to_send) {
    //MyDebugAssertTrue(i_am_the_master == true, "SendJobResultQueue called by a worker!");

    WriteToSocket(controller_socket, socket_job_result_queue, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
    // See SendResultQueueToSocket() Doxygen for encoding order specification
    SendResultQueueToSocket(controller_socket, queue_to_send);
}

void MyApp::MasterSendIntenalQueue( ) {
    SendJobResultQueue(master_job_queue);
    master_job_queue.Clear( );
    time_of_last_master_queue_send = time(NULL);
}

void MyApp::SendAllJobsFinished( ) {
    //MyDebugAssertTrue(i_am_the_master == true, "SendAllJobsFinished called by a worker!");

    // we will send all jobs finished - but first we need to ensure we have sent any results in the result queue
    // wait for 5 seconds to give workers times to send in their last jobs..

    wxSleep(1);
    //Yield();

    if ( master_job_queue.GetCount( ) != 0 )
        MasterSendIntenalQueue( );

    WriteToSocket(controller_socket, socket_all_jobs_finished, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
    WriteToSocket(controller_socket, &total_milliseconds_spent_on_threads, sizeof(long), true, "SendTotalMillisecondsSpentOnThreads", FUNCTION_DETAILS_AS_WXSTRING);
}

void MyApp::OnZombieTimer(wxTimerEvent& event) {
    if ( i_am_a_zombie == true ) {
        number_of_failed_connections++;

        // Note this used to call ExitMainLoop( ) and then fall straight through into
        // another reconnect attempt below - the give-up was decided and then ignored.
        // Exit here, and non-zero, so the batch system records a failed worker rather
        // than the 0 that falling out of the event loop produces.
        if ( number_of_failed_connections >= 5 ) {
            wxPrintf("Worker: giving up after %i failed connection attempts.\n", number_of_failed_connections);
            ExitMainLoop( );
            exit(cistem::exit_code::reconnect_failed);
        }

        if ( connected_to_the_master == true ) {
            master_socket->Close( );
            master_socket->Connect(active_controller_address, false);
            master_socket->WaitOnConnect(30);

            if ( master_socket->IsConnected( ) == false ) {
                master_socket->Close( );
            }

            master_socket->SetFlags(SOCKET_FLAGS);

            if ( master_socket->IsConnected( ) == false || master_socket->IsOk( ) == false ) {
                master_socket->Close( );
                MyDebugPrint(" JOB : Failed ! Unable to connect\n");
                ExitMainLoop( );
                exit(cistem::exit_code::reconnect_failed);
            }

            if ( i_am_the_master == false )
                controller_socket = master_socket;
        }
        else {
            controller_socket->Close( );
            controller_socket->Connect(active_controller_address, false);
            controller_socket->WaitOnConnect(30);

            if ( controller_socket->IsConnected( ) == false ) {
                controller_socket->Close( );
            }

            controller_socket->SetFlags(SOCKET_FLAGS);

            if ( controller_socket->IsConnected( ) == false || controller_socket->IsOk( ) == false ) {
                controller_socket->Close( );
                MyDebugPrint(" JOB : Failed ! Unable to connect\n");
                ExitMainLoop( );
                exit(cistem::exit_code::reconnect_failed);
            }
        }
        // once again, we are aparently connected, but this can be a lie as a certain number of connections appear to just be accepted by the operating
        // system - if the port if valid.  So if we don't get any events from this socket within 10 seconds, we are going to try again...

        zombie_timer = new wxTimer(this, 1);
        zombie_timer->StartOnce(20000);
    }
}

void MyApp::OnMasterQueueTimer(wxTimerEvent& event) {
    if ( master_job_queue.GetCount( ) > 0 ) {
        MasterSendIntenalQueue( );
    }

    master_queue_timer_set = false;
    delete master_queue_timer;
}

void MyApp::OnQueueTimer(wxTimerEvent& event) {
    SendAllResultsFromResultQueue( );

    queue_timer_set = false;
    delete queue_timer;
}

void MyApp::OnThreadComplete(wxThreadEvent& my_event) {
    //SETUP_SOCKET_CODES

    // The compute thread is finished.. get the next job
    // thread should be dead, or nearly dead..

    //work_thread = NULL;
    SendAllResultsFromResultQueue( );

    // get the next job..
    WriteToSocket(master_socket, socket_send_next_job, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);

    // if there is a result - send it to the gui..
    my_result.job_number = my_current_job.job_number;
    my_result.SendToSocket(master_socket);
}

// Create and start the calculation thread if there is not one already. Deliberately does
// NOT touch the stopwatch: on a re-arm (a job arriving after the previous thread idled
// out) the worker-lifetime clock keeps running from the original start, so the timing the
// worker eventually reports stays a lifetime figure. Returns false, with work_thread back
// at NULL, if the thread could not be created - the caller decides how fatal that is.
bool MyApp::EnsureWorkThreadIsRunning( ) {
    if ( work_thread != NULL )
        return true;

    work_thread = new CalculateThread(this, GetMaxJobWaitTimeInSeconds( ));

    if ( work_thread->Run( ) != wxTHREAD_NO_ERROR ) {
        MyPrintWithDetails("Can't create the thread!");
        delete work_thread;
        work_thread = NULL;
        return false;
    }

    return true;
}

void MyApp::OnThreadEnding(wxThreadEvent& my_event) {
    SendAllResultsFromResultQueue( );

    work_thread = NULL;

    // This handler only ever runs for a thread that broke out of its loop on the idle
    // timeout (job_wait_time): both told-to-die paths (HandleSocketTimeToDie, worker-side
    // HandleSocketDisconnect) exit the process before this event could be processed.

    // A job can arrive in the gap between the thread deciding to end and this handler
    // running: HandleSocketReadyToSendSingleJob sets thread_next_action and both it and
    // this handler run on the main thread, but the job's event can be queued first. If
    // that happened the flag is set with no thread left to read it, and the job is
    // silently swallowed - the master's books say assigned, this process never runs it,
    // and the run wedges one result short (observed 2026-08-27: run nbryBR5B5GseKMFp
    // stuck at 57 of 58 chunks of its final image). Re-arm instead of exiting.
    bool have_pending_job = false;
    {
        wxMutexLocker lock(job_lock);
        if ( lock.IsOk( ) == true )
            have_pending_job = (thread_next_action == THREAD_START_NEXT_JOB);
    }

    if ( have_pending_job == true ) {
        if ( EnsureWorkThreadIsRunning( ) == false ) {
            SocketSendError("Error: worker has a pending job but cannot recreate its calculation thread - exiting so the job is re-dispatched.");
            exit(cistem::exit_code::thread_creation_failed);
        }
        return;
    }

    // No job pending: this process has nothing to do and no thread to ever ask for more.
    // A networked worker that lingers here is a connected-but-useless husk that holds its
    // batch slot (and GPU) until the run ends or someone kills it - 22 of them held slots
    // for an hour on 2026-08-27. Exit and release the slot instead. Send the thread
    // timing first, exactly like the told-to-die path, so the master's bookkeeping
    // (worker_socket_pointers erase + timing tally) treats this as an orderly departure
    // rather than a mid-run death, and the GUI is not spammed with false errors.
    if ( is_running_locally == false && i_am_the_master == false ) {
        wxPrintf("Worker: calculation thread idled out with no job pending - exiting to release this slot.\n");

        if ( master_socket != NULL && master_socket->IsConnected( ) == true ) {
            long milliseconds_spent_by_thread = stopwatch.Time( );
            WriteToSocket(master_socket, socket_send_thread_timing, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
            WriteToSocket(master_socket, &milliseconds_spent_by_thread, sizeof(long), true, "SendMillisecondsSpentByThread", FUNCTION_DETAILS_AS_WXSTRING);
            StopMonitoringAndDestroySocket(master_socket);
        }

        ShutDownSocketMonitor( );
        exit(cistem::exit_code::idle_worker_timeout);
    }
}

void MyApp::OnThreadSendError(wxThreadEvent& my_event) {
    SocketSendError(my_event.GetString( ));
    //MyDebugPrint("ThreadSendError");
}

void MyApp::OnThreadSendInfo(wxThreadEvent& my_event) {
    SocketSendInfo(my_event.GetString( ));
    //MyDebugPrint("ThreadSendError");
}

void MyApp::OnThreadIntermediateResultAvailable(wxThreadEvent& my_event) {
    //	wxPrintf("MyApp::Received result available event..\n");
    //SendAllResultsFromResultQueue();

    if ( queue_timer_set == false ) {
        queue_timer_set = true;
        queue_timer     = new wxTimer(this, 2);
        queue_timer->StartOnce(1000);
    }
}

void MyApp::OnThreadSendImageResult(wxThreadEvent& my_event) {
    //MyDebugAssertTrue(i_am_the_master == false, "OnThreadSendImageResult called by master!");

    Image image_to_send;
    image_to_send              = my_event.GetPayload<Image>( );
    int      position_in_stack = my_event.GetInt( );
    wxString filename_to_write = my_event.GetString( );
    int      details[3];

    details[0] = image_to_send.logical_x_dimension;
    details[1] = image_to_send.logical_y_dimension;
    details[2] = position_in_stack;

    WriteToSocket(master_socket, socket_result_with_image_to_write, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
    WriteToSocket(master_socket, details, sizeof(int) * 3, true, "SendResultImageDetailsFromWorkerToMaster", FUNCTION_DETAILS_AS_WXSTRING);
    WriteToSocket(master_socket, image_to_send.real_values, image_to_send.real_memory_allocated * sizeof(float), true, "SendResultImageDataFromWorkerToMaster", FUNCTION_DETAILS_AS_WXSTRING);
    SendwxStringToSocket(&filename_to_write, master_socket);

    // post a message to the message queue to allow the calulcation thread to send the next image..
    inter_thread_message_queue.Post(0);
}

void MyApp::OnThreadSendProgramDefinedResult(ReturnProgramDefinedResultEvent& my_event) {
    //MyDebugAssertTrue(i_am_the_master == false, "OnThreadSendImageResult called by master!");

    float* array_to_send              = my_event.GetResultData( );
    long   size_of_array              = my_event.GetSizeOfResultData( );
    int    number_of_expected_results = my_event.GetNumberOfExpectedResults( );
    int    result_number              = my_event.GetResultNumber( );

    int details[3];

    details[0] = size_of_array;
    details[1] = result_number;
    details[2] = number_of_expected_results;

    WriteToSocket(master_socket, socket_program_defined_result, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
    WriteToSocket(master_socket, details, sizeof(int) * 3, true, "SendProgramDefinedResultDetailsFromWorkerToMaster", FUNCTION_DETAILS_AS_WXSTRING);
    WriteToSocket(master_socket, array_to_send, size_of_array * sizeof(float), true, "SendProgramDefinedResultArrayFromWorkerToMaster", FUNCTION_DETAILS_AS_WXSTRING);

    delete[] array_to_send;
}

void MyApp::SendAllResultsFromResultQueue( ) {
    // have we sent results within the last second? if so wait 1s

    ArrayofJobResults my_queue_array;

    // we want to pop off all the jobs, and send them in one big lump..

    wxMutexLocker* lock = new wxMutexLocker(job_lock);

    if ( lock->IsOk( ) == true ) {
        while ( 1 == 1 ) {

            JobResult* popped_job = PopJobFromResultQueue( );

            if ( popped_job == NULL ) {
                break;
            }
            else {
                my_queue_array.Add(*popped_job);
                delete popped_job;
            }
        }
    }
    else {
        SocketSendError("Job Lock Error!");
        MyPrintWithDetails("Can't get job lock!");
    }

    delete lock;

    // ok, send them all..

    if ( my_queue_array.GetCount( ) > 0 ) {
        /*		if (time(NULL) - time_of_last_queue_send < 1)
		{
			wxSleep(1);
		}*/

        SendIntermediateResultQueue(my_queue_array);
        time_of_last_queue_send = time(NULL);
    }
}

void MyApp::SendIntermediateResultQueue(ArrayofJobResults& queue_to_send) {
    //MyDebugAssertTrue(i_am_the_master == false, "SendIntermediateResultQueue called by master!");

    if ( queue_to_send.GetCount( ) > 0 ) {
        WriteToSocket(master_socket, socket_job_result_queue, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
        SendResultQueueToSocket(master_socket, queue_to_send);
    }

    time_of_last_queue_send = time(NULL);
}

void MyApp::SocketSendError(wxString error_to_send) {
    // The master also mirrors to stdout, which the on-disk job_control log captures.
    // Without this, every master-side decision reported through here (worker deaths,
    // job re-dispatches, the retry-cap give-up) exists only as transient text in the
    // GUI panel - during the 2026-08-26 OOM post-mortem there was no on-disk record of
    // any of it. Workers are not mirrored: their stdout is streamed per-job condor
    // output, where these messages would be duplicates of what the master reports.
    if ( i_am_the_master == true )
        wxPrintf("MASTER %s | %s\n", wxDateTime::Now( ).FormatISOTime( ), error_to_send);

    // send the error message flag

    if ( is_running_locally == false ) {
        WriteToSocket(controller_socket, socket_i_have_an_error, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
        SendwxStringToSocket(&error_to_send, controller_socket);
    }
}

void MyApp::SocketSendInfo(wxString info_to_send) {
    // See SocketSendError above for why the master mirrors to stdout.
    if ( i_am_the_master == true )
        wxPrintf("MASTER %s | %s\n", wxDateTime::Now( ).FormatISOTime( ), info_to_send);

    // send the info message flag

    if ( is_running_locally == false ) {
        WriteToSocket(controller_socket, socket_i_have_info, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
        SendwxStringToSocket(&info_to_send, controller_socket);
    }
}

void MyApp::SendError(wxString error_to_send) {
    if ( is_running_locally == true ) {
        wxPrintf("\nError : %s\n", error_to_send);
    }
    else if ( work_thread != NULL ) {
        work_thread->QueueError(error_to_send);
    }
    else {
        SocketSendError("SendError with null work thread!");
    }
}

void MyApp::SendErrorAndCrash(wxString error_to_send) {
    SendError(error_to_send);
    if ( ! is_running_locally )
        wxSleep(2); // wait for the main thread to actually send the error
    DEBUG_ABORT;
}

void MyApp::SendInfo(wxString info_to_send) {
    if ( is_running_locally == true ) {
        wxPrintf("\nInfo : %s\n", info_to_send);
    }
    else if ( work_thread != NULL ) {
        work_thread->QueueInfo(info_to_send);
    }
    else {
        SocketSendError("SendInfo with null work thread!");
    }
}

void MyApp::AddJobToResultQueue(JobResult* result_to_add) {
    //	wxPrintf("Adding Job to result Queue\n");
    wxMutexLocker* lock = new wxMutexLocker(job_lock);

    if ( lock->IsOk( ) == true )
        job_queue.Add(result_to_add);
    else {
        SocketSendError("Job Lock Error!");
        MyPrintWithDetails("Can't get job lock!");
    }

    delete lock;

    if ( work_thread != NULL ) {
        //		wxPrintf("MyApp::Marking Intermediate Result Available...\n");
        work_thread->MarkIntermediateResultAvailable( );
    }
    else {
        wxPrintf("Work thread is NULL!\n");
    }
}

void MyApp::SendProcessedImageResult(Image* image_to_send, int position_in_stack, wxString filename_to_save) {
    if ( work_thread != NULL ) {
        work_thread->SendProcessedImageResult(image_to_send, position_in_stack, filename_to_save);
    }
    else {
        wxPrintf("Work thread is NULL!\n");
    }
}

void MyApp::SendProgramDefinedResultToMaster(float* array_to_send, long size_of_array, int result_number, int number_of_expected_results) {
    if ( work_thread != NULL ) {
        work_thread->SendProgramDefinedResultToMaster(array_to_send, size_of_array, result_number, number_of_expected_results);
    }
    else {
        wxPrintf("Work thread is NULL!\n");
    }
}

JobResult* MyApp::PopJobFromResultQueue( ) // MAKE SURE THE MUTEX JOB_LOCK IS LOCKED BEFORE CALLING THIS!
{
    JobResult* popped_job = NULL;

    if ( job_queue.GetCount( ) > 0 )
        popped_job = job_queue.Detach(0);
    return popped_job;
}

// Main execution in this thread..

wxThread::ExitCode CalculateThread::Entry( ) {
    int  thread_action_copy;
    long millis_sleeping = 0;

    while ( 1 == 1 ) {
        wxMutexLocker* lock = new wxMutexLocker(main_thread_pointer->job_lock);

        if ( lock->IsOk( ) == true ) {
            //wxPrintf("Thread next action = %i\n", main_thread_pointer->thread_next_action);
            thread_action_copy = main_thread_pointer->thread_next_action;
        }
        else {
            QueueError("Job Lock Error!");
        }

        if ( main_thread_pointer->thread_next_action == THREAD_START_NEXT_JOB ) {
            main_thread_pointer->thread_next_action = THREAD_SLEEP;
            millis_sleeping                         = 0;
        }

        delete lock;

        if ( thread_action_copy == THREAD_START_NEXT_JOB ) {
            bool           success         = main_thread_pointer->DoCalculation( ); // This should be overrided per app..
            wxThreadEvent* my_thread_event = new wxThreadEvent(wxEVT_COMMAND_MYTHREAD_COMPLETED);

            if ( success == true )
                my_thread_event->SetInt(1);
            else
                my_thread_event->SetInt(0);
            wxQueueEvent(main_thread_pointer, my_thread_event);
        }
        else if ( thread_action_copy == THREAD_SLEEP ) {
            wxMilliSleep(100);
            millis_sleeping += 100;

            if ( millis_sleeping > job_wait_time * 1000 ) {
                // we have been waiting for 10 seconds, something probably went wrong - so die.
                wxPrintf("Calculation thread has been waiting for something to do for %.2f seconds - going to finish\n", job_wait_time);
                QueueError(wxString::Format("Calculation thread has been waiting for something to do for %.2f seconds - going to finish", job_wait_time));
                break;
            }
        }
        else if ( thread_action_copy == THREAD_DIE )
            break;
    }

    wxThreadEvent* my_thread_event = new wxThreadEvent(wxEVT_COMMAND_MYTHREAD_ENDING);
    wxQueueEvent(main_thread_pointer, my_thread_event);

    fftwf_cleanup( ); // this is needed to stop valgrind reporting memory leaks..
    return (wxThread::ExitCode)0; // success
}

void CalculateThread::QueueError(wxString error_to_queue) {
    wxThreadEvent* test_event = new wxThreadEvent(wxEVT_COMMAND_MYTHREAD_SENDERROR);
    test_event->SetString(error_to_queue);
    wxQueueEvent(main_thread_pointer, test_event);
}

void CalculateThread::QueueInfo(wxString info_to_queue) {
    wxThreadEvent* test_event = new wxThreadEvent(wxEVT_COMMAND_MYTHREAD_SENDINFO);
    test_event->SetString(info_to_queue);
    wxQueueEvent(main_thread_pointer, test_event);
}

void CalculateThread::MarkIntermediateResultAvailable( ) {
    wxThreadEvent* test_event = new wxThreadEvent(wxEVT_COMMAND_MYTHREAD_INTERMEDIATE_RESULT_AVAILABLE);
    wxQueueEvent(main_thread_pointer, test_event);
    //wxPrintf("CalculateThread::Queueing Result Available Event..\n");
}

void CalculateThread::SendProcessedImageResult(Image* image_to_send, int position_in_stack, wxString filename_to_save) {
    char message;
    if ( main_thread_pointer->inter_thread_message_queue.ReceiveTimeout(300000, message) == wxMSGQUEUE_TIMEOUT ) // timeout after 5 minutes;
    {
        QueueError("Timed out waiting for message queue");
    }

    wxThreadEvent* test_event = new wxThreadEvent(wxEVT_COMMAND_MYTHREAD_SEND_IMAGE_RESULT);
    test_event->SetInt(position_in_stack);
    test_event->SetString(filename_to_save);
    test_event->SetPayload(*image_to_send);
    wxQueueEvent(main_thread_pointer, test_event);
}

void CalculateThread::SendProgramDefinedResultToMaster(float* array_to_send, long size_of_array, int result_number, int number_of_expected_results) {
    ReturnProgramDefinedResultEvent* test_event = new ReturnProgramDefinedResultEvent(wxEVT_COMMAND_MYTHREAD_SEND_PROGRAM_DEFINED_RESULT);
    test_event->SetResultData(array_to_send);
    test_event->SetSizeOfResultData(size_of_array);
    test_event->SetResultNumber(result_number);
    test_event->SetNumberOfExpectedResults(number_of_expected_results);
    wxQueueEvent(main_thread_pointer, test_event);
}

CalculateThread::~CalculateThread( ) {
    //wxCriticalSectionLocker enter(m_pHandler->m_pThreadCS);
    // the thread is being destroyed; make sure not to leave dangling pointers around

    wxMutexLocker* lock = new wxMutexLocker(main_thread_pointer->job_lock);

    if ( lock->IsOk( ) == true ) {
        main_thread_pointer->work_thread = NULL;
    }
    else {
        QueueError("Job Lock Error!");
    }

    delete lock;

    main_thread_pointer = NULL;
}

///////////////////////////////////////////////////////////////////////////////////
//                              SOCKET HANDLING                                  //
///////////////////////////////////////////////////////////////////////////////////

// These should be from guix_job_control :-

//void MyApp::HandleSocketJobPackage(wxSocketBase *connected_socket, JobPackage *received_package)
//{
//	current_job_package = *received_package;
//	delete received_package;
//}

void MyApp::HandleSocketYouAreTheMaster(wxSocketBase* connected_socket, JobPackage* received_package) {

    current_job_package = *received_package;
    delete received_package;

    // Per-job dispatch counters for orphaned-job recovery (see SendNextJobTo and
    // HandleSocketDisconnect). Sized here because this is where the master first learns
    // how many jobs the package holds. This is sizing, not a reset: the controller elects
    // a master exactly once per process (guix_job_control's have_assigned_master is set
    // true and never cleared), so this handler runs once and there is no stale state to
    // clear. Anything that makes it re-entrant has a much bigger problem to solve first -
    // the assignment above delete[]s the old jobs array, which dangles every RunJob* in
    // socket_to_worker_job_pointer_hash and jobs_to_redispatch.
    job_dispatch_attempts.assign(current_job_package.number_of_jobs, 0);

    // we got real communication, so we are not a zombie

    i_am_a_zombie = false;
    if ( zombie_timer != NULL ) {
        delete zombie_timer;
        zombie_timer = NULL;
    }

    i_am_the_master = true;

    // CISTEM_EXPERIMENTAL_LEADER_NON_COMPUTE: when set (and non-empty) this process only serves,
    // aggregates and reports - it does not connect itself as a worker or run a CalculateThread -
    // and its server must bind the first free port of a 4-port window derived from the job code,
    // so that pre-established tunnels can forward to a known port.

    const char* leader_non_compute_env = getenv("CISTEM_EXPERIMENTAL_LEADER_NON_COMPUTE");
    i_am_a_non_compute_leader          = (leader_non_compute_env != NULL && leader_non_compute_env[0] != '\0' && strcmp(leader_non_compute_env, "0") != 0); // "0" means off

    // we need to start a server so that the workers can connect..

    if ( i_am_a_non_compute_leader == true ) {
        // derive the port window from the job code with 32-bit FNV-1a:
        // base = 41000 + (hash mod 7996), window = base .. base + 3

        wxUint32 job_code_hash = 2166136261u;

        for ( int counter = 0; counter < SOCKET_CODE_SIZE; counter++ ) {
            job_code_hash = (job_code_hash ^ (wxUint32)current_job_code[counter]) * 16777619u;
        }

        const int base_port = 41000 + (int)(job_code_hash % 7996u);
        int       derived_ports[4];

        for ( int counter = 0; counter < 4; counter++ ) {
            derived_ports[counter] = base_port + counter;
        }

        if ( SetupServer(derived_ports, 4) == false ) {
            wxPrintf("SSH_TUNNEL_ERROR: LEADER_NON_COMPUTE master could not bind any port of the derived window %i-%i - aborting\n", base_port, base_port + 3);
            SocketSendError(wxString::Format("SSH_TUNNEL_ERROR: LEADER_NON_COMPUTE master could not bind any port of the derived window %i-%i - aborting", base_port, base_port + 3));
            ExitMainLoop( );
            exit(-1);
            return;
        }

        wxPrintf("LEADER_NON_COMPUTE: master server bound port %s (window %i-%i)\n", ReturnServerPortString( ), base_port, base_port + 3);
    }
    else {
        SetupServer( );
    }

    // bind to the master queue timer..

    Bind(wxEVT_TIMER, wxTimerEventHandler(MyApp::OnMasterQueueTimer), this, 3);

    // Periodic liveness sweep over job-holding workers, and the run-unfinishable check.
    // Recurring, master-lifetime: during a long main-thread stall (a merge backlog) the
    // tick events just queue up and are processed - with correct arithmetic - when the
    // stall clears, so a stalled master cannot false-positive its own workers. Guarded so
    // a re-entered master handler cannot double-Bind or leak a second timer.
    if ( worker_liveness_timer == NULL ) {
        Bind(wxEVT_TIMER, wxTimerEventHandler(MyApp::OnWorkerLivenessTimer), this, 4);
        worker_liveness_timer = new wxTimer(this, 4);
        worker_liveness_timer->Start(180000); // 3 minutes
    }

    my_port        = ReturnServerPort( );
    my_port_string = ReturnServerPortString( );

    my_ip_address = ReturnIPAddressFromSocket(connected_socket);

    master_ip_address  = my_ip_address;
    master_port_string = my_port_string;
    master_port        = my_port;

    if ( i_am_a_non_compute_leader == false ) {
        // connect myself as a worker..

        master_socket = new wxSocketClient( );
        master_socket->SetFlags(SOCKET_FLAGS);
        master_socket->Notify(false);

        active_controller_address.Hostname("localhost");
        active_controller_address.Service(master_port);

        master_socket->Connect(active_controller_address, false);
        master_socket->WaitOnConnect(30);

        master_socket->SetFlags(SOCKET_FLAGS);

        if ( master_socket->IsConnected( ) == false ) {
            master_socket->Close( );
            MyDebugPrint("JOB : Failed ! Unable to connect\n");
        }
        else {
            MonitorSocket(master_socket);

            // Start the worker thread..
            stopwatch.Start( );
            work_thread = new CalculateThread(this, GetMaxJobWaitTimeInSeconds( ));

            if ( work_thread->Run( ) != wxTHREAD_NO_ERROR ) {
                MyPrintWithDetails("Can't create the thread!");
                delete work_thread;
                work_thread = NULL;
                ExitMainLoop( );
            }
        }
    }
    else {
        wxPrintf("LEADER_NON_COMPUTE: not connecting myself as a worker, serving/aggregating only\n");
    }

    // I have to send my ip address to the controller..

    // This is possibly dodgy as it's not being controlled by SocketCommunicator - hopefully it is ok, as this socket is not yet
    // being monitored in the corresponding read in guix_job_control - but it's a possible point of failure.

    SendwxStringToSocket(&my_ip_address, connected_socket);
    SendwxStringToSocket(&my_port_string, connected_socket);

    if ( i_am_a_non_compute_leader == true ) {
        // Tell the controller not to count this process as a connected worker:
        // it serves and aggregates only, and counting it would both misreport
        // the connected total to the GUI and make the controller shut its
        // server down one real worker early (see SendNumberofConnections).
        // Sent after the ip/port strings above, so it sits in the socket
        // buffer until the controller starts monitoring this socket.
        WriteToSocket(connected_socket, socket_i_am_a_dedicated_master, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
    }

    // ok, now get the job details from the conduit controller

    //WriteToSocket(connected_socket, socket_send_job_details, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
}

void MyApp::HandleSocketYouAreAWorker(wxSocketBase* connected_socket, wxString master_ip_address, wxString master_port_string) {

    // we got real communication, so we are not a zombie

    i_am_a_zombie = false;
    if ( zombie_timer != NULL ) {
        delete zombie_timer;
        zombie_timer = NULL;
    }

    long received_port;
    //i_am_the_master = false;

    master_port_string.ToLong(&received_port);

    // CISTEM_WORKER_PORT_REMAP ("<remote_base>:<local_base>:<count>"): the launch wrapper
    // holds PRIVATE local tunnel forwards instead of sharing the wire-advertised ports with
    // sibling jobs on the same node (shared forwards die with whichever sibling exits
    // first). The master port arrives over the wire in the remote window; map it into this
    // worker's private local window before connecting.
    const char* port_remap_env = getenv("CISTEM_WORKER_PORT_REMAP");
    if ( port_remap_env != NULL && port_remap_env[0] != '\0' ) {
        long              remap_remote_base, remap_local_base, remap_count;
        wxStringTokenizer remap_tokens(wxString::FromUTF8(port_remap_env), ":");
        if ( remap_tokens.CountTokens( ) == 3 &&
             remap_tokens.GetNextToken( ).ToLong(&remap_remote_base) &&
             remap_tokens.GetNextToken( ).ToLong(&remap_local_base) &&
             remap_tokens.GetNextToken( ).ToLong(&remap_count) &&
             received_port >= remap_remote_base && received_port < remap_remote_base + remap_count ) {
            const long remapped_port = remap_local_base + (received_port - remap_remote_base);
            wxPrintf("WORKER: master port %li remapped to private local forward %li (CISTEM_WORKER_PORT_REMAP=%s)\n", received_port, remapped_port, port_remap_env);
            received_port = remapped_port;
        }
    }

    master_port = (short int)received_port;

    // remove this socket from monitoring and destroy it..

    StopMonitoringAndDestroySocket(connected_socket);
    IfSocketIsAKeySocketSetItToNull(connected_socket);

    // connect to the new master..

    master_socket = new wxSocketClient( );
    master_socket->SetFlags(SOCKET_FLAGS);
    master_socket->Notify(false);

    active_controller_address.Hostname(master_ip_address);
    active_controller_address.Service(master_port);

    // Under a fleet-wide reconnect (a batch advance releases every seat at once) the
    // master's accept queue can be momentarily full; a single failed attempt here
    // otherwise costs a full zombie cycle through the controller - another
    // identification (inflating the GUI's connection counter) and another 20 s.
    // Retry with pid-staggered backoff so the fleet de-synchronizes.
    for ( int connect_attempt = 0;; connect_attempt++ ) {
        master_socket->Connect(active_controller_address, false);
        master_socket->WaitOnConnect(30);
        if ( master_socket->IsConnected( ) == true || connect_attempt >= 9 )
            break;
        master_socket->Close( );
        wxMilliSleep(1000 + (wxGetProcessId( ) % 7) * 500);
    }

    master_socket->SetFlags(SOCKET_FLAGS);

    if ( master_socket->IsConnected( ) == false ) {
        master_socket->Close( );
        wxPrintf("WORKER: cannot connect to master at %s:%i\n", master_ip_address, (int)master_port);
    }

    // otherwise we should be connected.. so start monitoring..

    MonitorSocket(master_socket);
    if ( i_am_the_master == false )
        controller_socket = master_socket;

    // Start the worker thread.. (only once: this handler re-runs on every zombie
    // reconnect cycle, and each pass used to leak another detached CalculateThread,
    // all counting down the same idle timeout and printing their own exit spam)
    if ( work_thread == NULL ) {
        stopwatch.Start( );
        if ( EnsureWorkThreadIsRunning( ) == false )
            ExitMainLoop( );
    }

    // we are apparently connected again, but this can be a lie = a certain number of connections appear to just be accepted by the operating
    // system - if the port if valid.  So if we don't get any events from this socket with 30 seconds, we are going to assume something
    // went wrong and die...

    i_am_a_zombie = true;
    zombie_timer  = new wxTimer(this, 1);
    zombie_timer->StartOnce(20000);
}

void MyApp::HandleSocketTimeToDie(wxSocketBase* connected_socket) // This can be sent to a worker or the master, need to check which it is.
{
    if ( i_am_the_master == true && connected_socket == controller_socket ) {
        // tell any connected workers to die. then exit..

        for ( int counter = 0; counter < worker_socket_pointers.GetCount( ); counter++ ) {
            WriteToSocket(worker_socket_pointers[counter], socket_time_to_die, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
        }

        worker_socket_pointers.Clear( );
    }
    else //Worker
    {
        // Timing stuff here
        long milliseconds_spent_by_thread = stopwatch.Time( );

        WriteToSocket(master_socket, socket_send_thread_timing, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
        WriteToSocket(master_socket, &milliseconds_spent_by_thread, sizeof(long), true, "SendMillisecondsSpentByThread", FUNCTION_DETAILS_AS_WXSTRING);

        // time to die!
        wxMutexLocker* lock = new wxMutexLocker(job_lock);

        if ( lock->IsOk( ) == true ) {
            thread_next_action = THREAD_DIE;
        }
        else {
            SocketSendError("Job Lock Error!");
            MyPrintWithDetails("Can't get job lock!");
        }

        delete lock;
        StopMonitoringAndDestroySocket(master_socket);
        if ( i_am_the_master == false )
            ShutDownSocketMonitor( );

        // give the thread some time to die..
        wxSleep(2);

        // process thread events in case it has done something
        // Yield(); //(wxEVT_CATEGORY_THREAD);

        if ( work_thread != NULL )
            work_thread->Kill( );

        if ( i_am_the_master == false ) // don't die if we are also the master
        {
            ExitMainLoop( );
            exit(0);
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////
//                        FROM WORKERS WHEN I AM THE MASTER                       //
///////////////////////////////////////////////////////////////////////////////////

void MyApp::HandleSocketSendNextJob(wxSocketBase* connected_socket, JobResult* received_result) {
    SendNextJobTo(connected_socket);

    // Send info that the job has finished, and if necessary the result..

    if ( received_result->job_number != -1 ) {
        // Count each job ONCE. A worker falsely declared dead (stale-pointer disconnect,
        // dropped tunnel that recovers) gets its in-flight job re-dispatched; if the
        // original worker was in fact alive, BOTH completions arrive. Without this guard
        // each one increments number_of_finished_jobs, so the all-done gate could pass
        // while other jobs had never run and the run would be declared complete with
        // holes in it. Latent hole found by inspection of the re-dispatch path; the
        // first completion was already forwarded, so duplicates are simply dropped.
        if ( current_job_package.jobs[received_result->job_number].has_been_run == true ) {
            delete received_result;
            return;
        }

        if ( received_result->result_size > 0 ) {
            SendJobResult(received_result);
        }
        else // just say job finished..
        {
            SendJobFinished(received_result->job_number);
        }

        number_of_finished_jobs++;
        current_job_package.jobs[received_result->job_number].has_been_run = true;

        // check if we have all timings, and all results (this is chctedecked in two places - socket send timing and receive results as it is not certain will happen last)

        if ( number_of_finished_jobs == current_job_package.number_of_jobs && number_of_timing_results_received == max_number_of_connected_workers ) {

            SendAllJobsFinished( );

            if ( current_job_package.ReturnNumberOfJobsRemaining( ) != 0 ) {
                SocketSendError("All jobs should be finished, but job package is not empty.");
            }

            // time to die!

            StopMonitoringAndDestroySocket(connected_socket);
            ShutDownSocketMonitor( );
            delete received_result;

            if ( work_thread != NULL )
                work_thread->Kill( );

            ExitMainLoop( );
            return;
        }
    }

    delete received_result;
}

void MyApp::HandleSocketIHaveAnError(wxSocketBase* connected_socket, wxString error_message) {
    SocketSendError(error_message);
}

void MyApp::HandleSocketIHaveInfo(wxSocketBase* connected_socket, wxString info_message) {
    SocketSendInfo(info_message);
}

void MyApp::HandleSocketJobResult(wxSocketBase* connected_socket, JobResult* received_result) {
    SendJobResult(received_result);
    delete received_result;
}

void MyApp::HandleSocketJobResultQueue(wxSocketBase* connected_socket, ArrayofJobResults* received_queue) {
    // copy these results to our own result queue

    for ( int counter = 0; counter < received_queue->GetCount( ); counter++ ) {
        master_job_queue.Add(received_queue->Item(counter));
    }

    delete received_queue;

    // if there is no timer running, start one.

    if ( master_queue_timer_set == false ) {
        master_queue_timer_set = true;
        master_queue_timer     = new wxTimer(this, 3);
        master_queue_timer->StartOnce(1000);
    }
}

void MyApp::HandleSocketResultWithImageToWrite(wxSocketBase* connected_socket, wxString filename_to_write_to, int position_in_stack) {
    //	if (master_output_file.IsOpen() == false || master_output_file.filename != filename_to_write_to)
    //	{
    //		// if we are writing a file, close it..
    //		if (master_output_file.IsOpen() == true) master_output_file.CloseFile();
    //		master_output_file.OpenFile(filename_to_write_to.ToStdString(), true);
    //		image_to_write->WriteSlice(&master_output_file, 1); // to setup the file..
    //	}
    //
    //	image_to_write->WriteSlice(&master_output_file, position_in_stack);
    //	delete image_to_write;

    float temp_float;
    temp_float = position_in_stack;

    JobResult job_to_queue;
    job_to_queue.SetResult(1, &temp_float);
    master_job_queue.Add(job_to_queue);

    if ( master_queue_timer_set == false ) {
        master_queue_timer_set = true;
        master_queue_timer     = new wxTimer(this, 3);
        master_queue_timer->StartOnce(1000);
    }
    else {
        if ( time(NULL) - time_of_last_master_queue_send > 2 ) {
            // must be a lot of queued event, so the timer is not being called -  send the current result queue anyway so the gui gets updated;

            MasterSendIntenalQueue( );
        }
    }
}

// This process's resident set size in bytes, from /proc/self/statm. 0 if unreadable
// (non-Linux, or /proc unavailable) - callers print it as best-effort diagnostics only.
static long ReturnOwnResidentSetBytes( ) {
    long  size_pages     = 0;
    long  resident_pages = 0;
    FILE* statm          = fopen("/proc/self/statm", "r");
    if ( statm == NULL )
        return 0;
    if ( fscanf(statm, "%ld %ld", &size_pages, &resident_pages) != 2 )
        resident_pages = 0;
    fclose(statm);
    return resident_pages * sysconf(_SC_PAGESIZE);
}

// The tightest memory limit over this process's cgroup-v2 ancestry, and the usage
// charged at that same level. This is what the kernel's OOM killer actually enforces:
// on a shared login node the binding limit is typically the per-user slice, shared with
// every other process of the same user - so a process can be killed at an RSS well
// below the limit because its siblings hold the rest. Returns false if nothing was
// readable (non-Linux, cgroup v1, no limit set anywhere).
static bool ReturnCgroupMemoryCurrentAndLimit(long& current_bytes, long& limit_bytes) {
    char  cgroup_path[1024] = "";
    FILE* self_cgroup       = fopen("/proc/self/cgroup", "r");
    if ( self_cgroup == NULL )
        return false;

    char line[1024];
    while ( fgets(line, sizeof(line), self_cgroup) != NULL ) {
        if ( strncmp(line, "0::", 3) == 0 ) {
            strncpy(cgroup_path, line + 3, sizeof(cgroup_path) - 1);
            cgroup_path[strcspn(cgroup_path, "\n")] = '\0';
            break;
        }
    }
    fclose(self_cgroup);
    if ( cgroup_path[0] == '\0' )
        return false;

    // Walk from our own cgroup up toward the root, keeping the smallest numeric
    // memory.max seen and the memory.current charged at that same level.
    bool           found_a_limit = false;
    long           best_limit    = 0;
    long           best_current  = 0;
    const wxString cgroup_root   = "/sys/fs/cgroup";
    wxString       walk_path     = cgroup_root + wxString::FromUTF8(cgroup_path);

    while ( walk_path.Length( ) > cgroup_root.Length( ) ) {
        long  level_limit = -1;
        FILE* max_file    = fopen((walk_path + "/memory.max").ToUTF8( ).data( ), "r");
        if ( max_file != NULL ) {
            char max_text[64] = "";
            if ( fgets(max_text, sizeof(max_text), max_file) != NULL && strncmp(max_text, "max", 3) != 0 )
                level_limit = atol(max_text);
            fclose(max_file);
        }

        if ( level_limit > 0 && (found_a_limit == false || level_limit < best_limit) ) {
            long  level_current = 0;
            FILE* current_file  = fopen((walk_path + "/memory.current").ToUTF8( ).data( ), "r");
            if ( current_file != NULL ) {
                char current_text[64] = "";
                if ( fgets(current_text, sizeof(current_text), current_file) != NULL )
                    level_current = atol(current_text);
                fclose(current_file);
            }
            found_a_limit = true;
            best_limit    = level_limit;
            best_current  = level_current;
        }

        walk_path = walk_path.BeforeLast('/');
    }

    if ( found_a_limit == false )
        return false;
    current_bytes = best_current;
    limit_bytes   = best_limit;
    return true;
}

void MyApp::HandleSocketProgramDefinedResult(wxSocketBase* connected_socket, float* data_array, int size_of_data_array, int result_number, int number_of_expected_results) {
    const wxLongLong merge_start_ms = wxGetLocalTimeMillis( );

    MasterHandleProgramDefinedResult(data_array, size_of_data_array, result_number, number_of_expected_results);
    delete[] data_array;

    queued_result_payload_count.fetch_sub(1);
    queued_result_payload_bytes.fetch_sub(long(size_of_data_array) * long(sizeof(float)));

    // One diagnostic line per merged payload, to stdout so it lands in the on-disk
    // job_control log. Written AFTER this payload is consumed, so "backlog" is what is
    // still queued behind it. This is the line that was missing on 2026-08-26, when the
    // leader was OOM-killed at 7.6 GB against the 8 GB per-user cgroup with a deep
    // unread-payload backlog and no record of merge times, memory, or queue depth in
    // any log.
    const long merge_milliseconds = (wxGetLocalTimeMillis( ) - merge_start_ms).ToLong( );
    long       cgroup_current     = 0;
    long       cgroup_limit       = 0;
    wxString   cgroup_report      = "n/a";
    if ( ReturnCgroupMemoryCurrentAndLimit(cgroup_current, cgroup_limit) == true )
        cgroup_report = wxString::Format("%.2f/%.2f GB", double(cgroup_current) / 1073741824.0, double(cgroup_limit) / 1073741824.0);

    wxPrintf("MASTER DIAG %s | result for image %i (payload %.0f MB) merged in %li ms | backlog %i payloads / %.2f GB | rss %.2f GB | cgroup %s\n",
             wxDateTime::Now( ).FormatISOTime( ), result_number,
             double(long(size_of_data_array) * long(sizeof(float))) / 1048576.0,
             merge_milliseconds,
             queued_result_payload_count.load( ), double(queued_result_payload_bytes.load( )) / 1073741824.0,
             double(ReturnOwnResidentSetBytes( )) / 1073741824.0, cgroup_report);
}

void MyApp::HandleSocketSendThreadTiming(wxSocketBase* connected_socket, long received_timing_in_milliseconds) {
    total_milliseconds_spent_on_threads += received_timing_in_milliseconds;
    StopMonitoringAndDestroySocket(connected_socket);
    //connected_socket->Destroy();

    // A worker departing on its idle timeout (exit_code::idle_worker_timeout) can cross
    // in flight with a job this master just dispatched to it: the worker sends its timing
    // and exits without ever seeing the job. Erasing the hash entry below would then lose
    // that job with no recovery, so re-queue it first, exactly as a mid-run death would.
    if ( socket_to_worker_job_pointer_hash.count(connected_socket) != 0 && socket_to_worker_job_pointer_hash[connected_socket] != NULL &&
         socket_to_worker_job_pointer_hash[connected_socket]->has_been_run == false ) {
        RunJob*   orphaned_job    = socket_to_worker_job_pointer_hash[connected_socket];
        const int attempts_so_far = ReturnJobDispatchAttempts(orphaned_job->job_number);

        if ( attempts_so_far >= 0 && attempts_so_far <= max_job_redispatch_tries ) {
            SocketSendInfo(wxString::Format("A worker departed (idle timeout) just as job %i was dispatched to it; the job has been re-queued.", orphaned_job->job_number));
            jobs_to_redispatch.push_back(orphaned_job);
        }
        else {
            number_of_abandoned_jobs++;
            SocketSendError(wxString::Format("Error: Job %i was in flight to a departing worker and its retry budget is exhausted - it is abandoned.", orphaned_job->job_number));
        }
    }

    // This worker is done and its socket is going away: forget it, so the controller-disconnect /
    // time-to-die loops do not write to a destroyed socket, and the disconnect the monitor thread
    // may still deliver for it (the worker closes right after sending its timing) is not treated
    // as a worker dying mid-run.
    socket_to_worker_job_pointer_hash.erase(connected_socket);
    liveness_ping_sent_at_ms.erase(connected_socket);
    // Guard the Remove: during a mass teardown (GUI kill mid-run) or a stale-pointer
    // disconnect, this handler can fire for a socket the disconnect path already removed
    // (or that IfSocketIsAKeySocketSetItToNull NULLed in place); an unguarded Remove then
    // asserts "removing inexistent element" (41 of them in one killed-run teardown log).
    if ( worker_socket_pointers.Index(connected_socket) != wxNOT_FOUND )
        worker_socket_pointers.Remove(connected_socket);

    number_of_timing_results_received++;

    SendLiveWorkerCountToController( );

    // check if we have all timings, and all results (this is checked in two places - socket send timing and receive results as it is not certain will happen last)

    if ( number_of_finished_jobs == current_job_package.number_of_jobs && number_of_timing_results_received == max_number_of_connected_workers ) {
        SendAllJobsFinished( );

        if ( current_job_package.ReturnNumberOfJobsRemaining( ) != 0 ) {
            SocketSendError("All jobs should be finished, but job package is not empty.");
        }

        // time to die!

        wxSleep(5);

        controller_socket->Destroy( );
        controller_socket = NULL;

        ShutDownServer( );
        ShutDownSocketMonitor( );
        ExitMainLoop( );
        return;
    }
}

void MyApp::SendLiveWorkerCountToController( ) {
    if ( i_am_the_master == false || controller_socket == NULL )
        return;

    // worker_socket_pointers entries can be NULLed in place (IfSocketIsAKeySocketSetItToNull),
    // so count the live ones rather than trusting GetCount().
    int live_worker_count = 0;
    for ( int counter = 0; counter < worker_socket_pointers.GetCount( ); counter++ ) {
        if ( worker_socket_pointers[counter] != NULL )
            live_worker_count++;
    }

    // Same wire format the controller uses toward the GUI (4-byte count after the code);
    // the controller relays it verbatim in HandleSocketNumberOfConnections.
    WriteToSocket(controller_socket, socket_number_of_connections, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
    WriteToSocket(controller_socket, &live_worker_count, 4, true, "SendLiveWorkerCount", FUNCTION_DETAILS_AS_WXSTRING);
}

///////////////////////////////////////////////////////////////////////////////////
//              SERVER CONNECTIONS FROM WORKERS WHEN I AM THE MASTER              //
///////////////////////////////////////////////////////////////////////////////////

void MyApp::HandleNewSocketConnection(wxSocketBase* new_connection, unsigned char* identification_code) {
    if ( new_connection == NULL )
        return;

    if ( (memcmp(identification_code, current_job_code, SOCKET_CODE_SIZE) != 0) ) {
        SendError("Unknown Job ID (Job Control), leftover from a previous job? - Closing Connection");
        new_connection->Destroy( ); // we should not be monitoring this socket, just destroy it.
        new_connection = NULL;
    }
    else {
        // start monitoring this socket..
        MonitorSocket(new_connection);
        worker_socket_pointers.Add(new_connection);
        max_number_of_connected_workers++;

        // tell it is is connected..
        WriteToSocket(new_connection, socket_you_are_connected, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);

        int number_of_commands_to_run;
        if ( current_job_package.number_of_jobs + 1 < current_job_package.my_profile.ReturnTotalJobs( ) )
            number_of_commands_to_run = current_job_package.number_of_jobs + 1;
        else
            number_of_commands_to_run = current_job_package.my_profile.ReturnTotalJobs( );

        // A computing master occupies one of the launched seats and also connects to its own
        // server, so it expects number_of_commands_to_run - 1 remote workers. A non-compute
        // leader is launched outside the run-profile seats and does not self-connect, so every
        // expected connection is a real worker.
        int expected_worker_connections = i_am_a_non_compute_leader ? number_of_commands_to_run : number_of_commands_to_run - 1;
        if ( worker_socket_pointers.GetCount( ) == expected_worker_connections ) {
            SocketSendInfo("All workers have re-connected to the master.");
        }

        SendLiveWorkerCountToController( );
    }

    delete[] identification_code;
}

///////////////////////////////////////////////////////////////////////////////////
//                      FROM THE MASTER WHEN I AM A WORKER                        //
///////////////////////////////////////////////////////////////////////////////////

// Time to die is above as it could be for worker or master

void MyApp::HandleSocketYouAreConnected(wxSocketBase* connected_socket) {

    // if we got here, we are not a zombie..

    i_am_a_zombie = false;
    if ( zombie_timer != NULL ) {
        delete zombie_timer;
        zombie_timer = NULL;
    }

    // we are connected, request the first job..
    is_connected = true;

    WriteToSocket(connected_socket, socket_send_next_job, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
    JobResult temp_result; // dummy result for the initial request - not reallt very nice
    temp_result.job_number  = -1;
    temp_result.result_size = 0;
    temp_result.SendToSocket(connected_socket);
}

void MyApp::HandleSocketReadyToSendSingleJob(wxSocketBase* connected_socket, RunJob* received_job) {
    MyDebugAssertTrue(currently_running_a_job == false, "Received a new job, when already running a job!");
    my_current_job = *received_job;
    delete received_job;

    // The calculation thread may have idled out (job_wait_time) while this job's request
    // was queued behind a busy master - the answer can arrive minutes after it was asked
    // for. Setting the flag with no thread to read it silently swallows the job: the
    // master's books say assigned, this process never runs it, and the run wedges one
    // result short (observed 2026-08-27: run nbryBR5B5GseKMFp stuck at 57 of 58 chunks
    // of its final image behind a ~6 minute merge backlog). Re-arm the thread first.
    if ( EnsureWorkThreadIsRunning( ) == false ) {
        SocketSendError("Error: worker received a job but cannot create a calculation thread to run it - exiting so the job is re-dispatched.");
        exit(cistem::exit_code::thread_creation_failed);
    }

    wxMutexLocker* lock = new wxMutexLocker(job_lock);

    if ( lock->IsOk( ) == true ) {
        MyDebugAssertFalse(thread_next_action == THREAD_START_NEXT_JOB, "Thread action is already start job");
        thread_next_action = THREAD_START_NEXT_JOB;
    }
    else {
        SocketSendError("Job Lock Error!");
        MyPrintWithDetails("Can't get job lock!");
    }

    delete lock;
}

void MyApp::HandleSocketDisconnect(wxSocketBase* connected_socket) {
    if ( connected_socket == controller_socket && i_am_the_master == true ) // kill everything..
    {
        MyDebugPrint("Master received disconnect from controller");

        for ( int counter = 0; counter < worker_socket_pointers.GetCount( ); counter++ ) {
            WriteToSocket(worker_socket_pointers[counter], socket_time_to_die, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
            StopMonitoringAndDestroySocket(worker_socket_pointers[counter]);
            worker_socket_pointers[counter] = NULL;
        }

        worker_socket_pointers.Clear( );

        StopMonitoringAndDestroySocket(controller_socket);
        controller_socket = NULL;

        ShutDownServer( );
        ShutDownSocketMonitor( );

        if ( work_thread != NULL )
            work_thread->Kill( );

        ExitMainLoop( );
        return;
    }
    else if ( i_am_the_master == true && connected_socket != master_socket ) // a worker died..
    {
        // A socket that never identified itself with the job code was never a worker - it is
        // a port-liveness probe (the condor shim's tunnel watchdog re-verifies the forwarded
        // master ports every few seconds by connecting and closing). Destroy it quietly:
        // reporting it as a dead worker spams the GUI with a false error per probe.
        if ( worker_socket_pointers.Index(connected_socket) == wxNOT_FOUND ) {
            StopMonitoringAndDestroySocket(connected_socket);
            return;
        }

        // The hash only holds sockets with a job currently assigned: operator[] on a socket
        // that never requested a job (or whose entry was erased when it was told to die)
        // would insert a NULL and the print below would crash the master.
        const bool disconnected_worker_had_a_job = (socket_to_worker_job_pointer_hash.count(connected_socket) != 0 && socket_to_worker_job_pointer_hash[connected_socket] != NULL);

        // Report EVERY death that takes an unfinished job with it - the old guard
        // (number_of_dispatched_jobs < number_of_jobs) made a death AFTER full dispatch,
        // i.e. exactly the endgame case, silent: the job vanished with no message and the
        // run hung with N-1 of N results recorded. Queue the orphaned job for re-dispatch
        // so the next worker that asks (including a late straggler) picks it up.
        if ( disconnected_worker_had_a_job && socket_to_worker_job_pointer_hash[connected_socket]->has_been_run == false ) {
            RunJob*   orphaned_job    = socket_to_worker_job_pointer_hash[connected_socket];
            const int attempts_so_far = ReturnJobDispatchAttempts(orphaned_job->job_number);

            SocketSendInfo("The disconnected worker was running a job with the following arguments:\n" + orphaned_job->PrintAllArgumentsTowxString( ));

            // Re-dispatch is capped. A job that always kills its worker (a pathological
            // pixel producing NaNs that trips an assert, say) would otherwise be handed to
            // worker after worker forever. Everything below counts ATTEMPTS, not retries:
            // attempts_so_far is how many times this job has been sent out, and the budget
            // is max_job_redispatch_tries + 1 attempts.
            if ( attempts_so_far < 0 ) {
                SocketSendError(wxString::Format("Error: A worker has disconnected with unfinished job %i, but that job has no entry in this package's attempt table, so its retries could not be bounded. It will NOT be re-dispatched.",
                                                 orphaned_job->job_number));
            }
            else if ( attempts_so_far <= max_job_redispatch_tries ) {
                SocketSendError(wxString::Format("Error: A worker has disconnected with unfinished job %i on attempt %i of %li; it will be re-dispatched.",
                                                 orphaned_job->job_number, attempts_so_far, max_job_redispatch_tries + 1));
                jobs_to_redispatch.push_back(orphaned_job);
            }
            else {
                // The job is now permanently abandoned. Count it: when every remaining job
                // is abandoned, the liveness tick's unfinishable check fails the run loudly
                // (FailRunAsUnfinishable) instead of leaving the master waiting in its
                // event loop to be killed by hand. The two-tick grace there is also the
                // hedge for a worker falsely declared dead whose result is still in flight
                // (see the duplicate-result guard in HandleSocketJobResult) - if it lands,
                // the run completes normally.
                number_of_abandoned_jobs++;
                SocketSendError(wxString::Format("Error: Job %i has now failed on all %li permitted attempt(s) (CISTEM_EXPERIMENTAL_FAILED_WORKER_RESUBMIT_TRIES = %li) and will not be re-dispatched. Unless a result for it is still in flight from a worker wrongly presumed dead, this run cannot reach %i of %i finished jobs, and the master will shut the run down.",
                                                 orphaned_job->job_number, max_job_redispatch_tries + 1, max_job_redispatch_tries,
                                                 current_job_package.number_of_jobs, current_job_package.number_of_jobs));
            }
        }
        else if ( number_of_dispatched_jobs < current_job_package.number_of_jobs ) {
            SocketSendError("Error: A worker has disconnected before all jobs are finished.");
            SocketSendInfo("The disconnected worker had no job assigned.");
        }

        socket_to_worker_job_pointer_hash.erase(connected_socket);
        liveness_ping_sent_at_ms.erase(connected_socket);
        // Remove the entry as well as destroying the socket: a stale pointer left in the
        // array aliases future sockets malloc'd at the recycled address (watchdog probes
        // then masquerade as this dead worker, one false disconnect per probe).
        worker_socket_pointers.Remove(connected_socket);
        StopMonitoringAndDestroySocket(connected_socket);

        SendLiveWorkerCountToController( );

        // This worker is gone and will never send its thread timing, so it must not stay
        // counted by the all-done gate (number_of_timing_results_received ==
        // max_number_of_connected_workers). Workers that FINISHED were already erased from
        // worker_socket_pointers in HandleSocketSendThreadTiming, so this branch only sees
        // genuinely unfinished deaths - e.g. an idle-queue straggler that condor matches
        // after the work is done, which connects and dies at the handshake. Without the
        // decrement one such death wedges the run at the finish line: all results are in,
        // but the gate arithmetic is permanently one short and socket_all_jobs_finished is
        // never sent (observed 2026-08-23: 58/58 images done, 4 straggler EOF-deaths, GUI
        // meter frozen). Decrement and re-evaluate the gate here, since this death may be
        // exactly what completes it.
        max_number_of_connected_workers--;
        if ( number_of_finished_jobs == current_job_package.number_of_jobs && number_of_timing_results_received == max_number_of_connected_workers ) {
            SendAllJobsFinished( );

            if ( current_job_package.ReturnNumberOfJobsRemaining( ) != 0 ) {
                SocketSendError("All jobs should be finished, but job package is not empty.");
            }

            // time to die! (mirrors the gate sites in HandleSocketSendNextJob / HandleSocketSendThreadTiming)
            ShutDownServer( );
            ShutDownSocketMonitor( );

            if ( work_thread != NULL )
                work_thread->Kill( );

            ExitMainLoop( );
            return;
        }
    }
    else // i am a worker and the master died.. time to die
    {
        StopMonitoringAndDestroySocket(connected_socket);
        if ( i_am_the_master == false )
            ShutDownSocketMonitor( );

        if ( work_thread != NULL )
            work_thread->Kill( );

        // Exit NON-zero: this is a failure, and the exit code is the only thing the batch
        // system ever learns about it. Falling out through ExitMainLoop( ) reported 0 -
        // identical to a worker that finished its work - so every abnormal worker death in
        // a run's condor history read as a success.
        ExitMainLoop( );
        exit(cistem::exit_code::master_disconnected);
    }
}

// Mainly for when we destroy worker sockets without directly using the array, we want them to be set to NULL;

void MyApp::IfSocketIsAKeySocketSetItToNull(wxSocketBase* socket_to_check) {
    for ( int counter = 0; counter < worker_socket_pointers.GetCount( ); counter++ ) {
        if ( worker_socket_pointers[counter] == socket_to_check )
            worker_socket_pointers[counter] = NULL;
    }

    if ( controller_socket == socket_to_check )
        controller_socket = NULL;

    if ( master_socket == socket_to_check )
        master_socket = NULL;
}
