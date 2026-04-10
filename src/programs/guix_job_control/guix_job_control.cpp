#include <wx/wx.h>
#include <wx/app.h>
#include <wx/cmdline.h>
#include <cstdio>
#include <map>
#include "wx/socket.h"
#include <wx/evtloop.h>

#include "../../core/core_headers.h"
#include "../../core/socket_communication_utils/socket_codes.h"

wxDEFINE_EVENT(wxEVT_COMMAND_MYTHREAD_LAUNCHJOB, wxThreadEvent);
wxDEFINE_EVENT(wxEVT_COMMAND_MYTHREAD_SENDINFO, wxThreadEvent);

SETUP_SOCKET_CODES

#ifdef __WXOSX__
class
        JobControlApp : public wxApp,
                        public SocketCommunicator
#else
class
        JobControlApp : public wxAppConsole,
                        public SocketCommunicator
#endif
{
    bool          have_assigned_master;
    wxArrayString possible_gui_addresses;

    wxIPV4address active_gui_address;
    wxIPV4address my_address;
    wxIPV4address master_process_address;

    wxSocketClient* gui_socket;
    wxSocketBase*   master_socket;

    int number_of_received_jobs;

    long gui_port;
    long master_process_port;

    bool all_jobs_are_finished;

    short int     my_port;
    wxArrayString my_possible_ip_addresses;
    wxString      my_port_string;
    wxString      my_active_ip_address;
    wxString      master_ip_address;
    wxString      master_port;

    long number_of_workers_already_connected;

    // Revert-debug-ga1: populated by LaunchJobThread after tunnel verification.
    // Lists ssh_target names that have a reverse tunnel for the controller port.
    // Read by HandleNewSocketConnection after master election, to set up a
    // second reverse tunnel per target for the master port, and to decide
    // whether to rewrite the master address sent to tunneled workers.
    wxArrayString remote_tunnel_targets;
    bool          master_tunnels_established;
    // Remote-side port chosen at LaunchRemoteJob time, verified free on every
    // remote_tunnel_targets entry. Reused as BOTH the remote listen port
    // (ssh -f -N -R <port>:127.0.0.1:<master_port>) AND the port sent to the
    // tunneled worker (master=127.0.0.1:<port>). Must NOT reuse master_port
    // because a stale cisTEM worker from another user can silently collide
    // on the remote (seen: sshastri zombie bound 0.0.0.0:3002 on GA1, ssh -R
    // fell back to [::1]:3002 with ExitOnForwardFailure=yes still returning 0,
    // and IPv4 127.0.0.1 traffic routed into the zombie).
    wxString master_tunnel_remote_port;

    friend class LaunchJobThread;

    // Socket Handling overrides..

    void HandleNewSocketConnection(wxSocketBase* new_connection, unsigned char* identification_code);
    void HandleSocketYouAreConnected(wxSocketBase* connected_socket);
    void HandleSocketJobPackage(wxSocketBase* connected_socket, JobPackage* received_package);
    void HandleSocketTimeToDie(wxSocketBase* connected_socket);
    void HandleSocketIHaveAnError(wxSocketBase* connected_socket, wxString error_message);
    void HandleSocketIHaveInfo(wxSocketBase* connected_socket, wxString info_message);
    void HandleSocketJobResult(wxSocketBase* connected_socket, JobResult* received_result);
    void HandleSocketJobResultQueue(wxSocketBase* connected_socket, ArrayofJobResults* received_queue);
    //void HandleSocketSendJobDetails(wxSocketBase *connected_socket);
    void HandleSocketJobFinished(wxSocketBase* connected_socket, int finished_job_number);
    void HandleSocketAllJobsFinished(wxSocketBase* connected_socket, long received_timing_in_milliseconds);
    void HandleSocketDisconnect(wxSocketBase* connected_socket);
    void HandleSocketTemplateMatchResultReady(wxSocketBase* connected_socket, int& image_number, float& high_res_limit_used, float& threshold_used, ArrayOfTemplateMatchFoundPeakInfos& peak_infos, ArrayOfTemplateMatchFoundPeakInfos& peak_changes);

    // end

    void SendError(wxString error_to_send);
    void SendInfo(wxString info_to_send);

    void SendJobFinished(int job_number);
    void SendJobResult(JobResult* result_to_send);
    void SendJobResultQueue(ArrayofJobResults& queue_to_send);

    void SendAllJobsFinished(long total_timing_from_master);
    void SendNumberofConnections( );

    void OnThreadLaunchJob(wxThreadEvent& event);
    void OnThreadSendInfo(wxThreadEvent& event);

  public:
    virtual bool OnInit( );
    void         OnEventLoopEnter(wxEventLoopBase* loop);
};

class LaunchJobThread : public wxThread {
  public:
    LaunchJobThread(JobControlApp* handler, RunProfile wanted_run_profile, wxString wanted_ip_address, wxString wanted_port, const unsigned char* wanted_job_code, long wanted_actual_number_of_jobs) : wxThread(wxTHREAD_DETACHED) {
        main_thread_pointer   = handler;
        current_run_profile   = wanted_run_profile;
        ip_address            = wanted_ip_address;
        port_number           = wanted_port;
        actual_number_of_jobs = wanted_actual_number_of_jobs;

        for ( int counter = 0; counter < SOCKET_CODE_SIZE; counter++ ) {
            job_code[counter] = wanted_job_code[counter];
        }
    }

    //~LaunchJobThread();

  protected:
    JobControlApp* main_thread_pointer;
    RunProfile     current_run_profile;
    wxString       ip_address;
    wxString       port_number;
    long           actual_number_of_jobs;
    unsigned char  job_code[SOCKET_CODE_SIZE];

    void             LaunchRemoteJob( );
    void             QueueInfo(wxString info_to_queue);
    virtual ExitCode Entry( );
};

wxThread::ExitCode LaunchJobThread::Entry( ) {
    MyDebugPrintGA1("LaunchJobThread::Entry begin pid=%d", (int)getpid( ));
    MyDebugPrintGA1("LaunchJobThread::Entry ip_address=%s port_number=%s actual_number_of_jobs=%ld", ip_address, port_number, actual_number_of_jobs);
    MyDebugPrintGA1("LaunchJobThread::Entry calling LaunchRemoteJob()");
    LaunchRemoteJob( );
    MyDebugPrintGA1("LaunchJobThread::Entry LaunchRemoteJob() returned");
    return (wxThread::ExitCode)0; // success
}

IMPLEMENT_APP(JobControlApp)

bool JobControlApp::OnInit( ) {
    MyDebugPrintGA1("JobControlApp::OnInit begin pid=%d", (int)getpid( ));
    number_of_received_jobs    = 0;
    all_jobs_are_finished      = false;
    master_tunnels_established = false;
    master_tunnel_remote_port  = "";
    remote_tunnel_targets.Empty( );
    MyDebugPrintGA1("JobControlApp::OnInit counters reset");

    MyDebugPrint("Job Controller: Running...\n");
    MyDebugPrintGA1("JobControlApp::OnInit returning true");

    // initialise sockets
    wxSocketBase::Initialize( );

    return true;
}

void JobControlApp::OnEventLoopEnter(wxEventLoopBase* loop) {
    if ( loop->IsMain( ) == true ) {
        long          counter;
        wxIPV4address my_address;
        wxIPV4address junk_address;
        wxString      current_address;
        wxArrayString buffer_addresses;
        have_assigned_master = false;

        // Set brother event handler, this is a nastly little hack so that the socket communicator can use the event handler, and it will work whether the "brother" is a console app or gui panel.
        // it needs to be done in all classes that derive from SocketCommunicator

        brother_event_handler = this;

        // set up the parameters for passing the gui address..

        static const wxCmdLineEntryDesc command_line_descriptor[] =
                {
                        {wxCMD_LINE_PARAM, "a", "address", "gui_address", wxCMD_LINE_VAL_STRING, wxCMD_LINE_OPTION_MANDATORY},
                        {wxCMD_LINE_PARAM, "p", "port", "gui_port", wxCMD_LINE_VAL_NUMBER, wxCMD_LINE_OPTION_MANDATORY},
                        {wxCMD_LINE_PARAM, "j", "job_code", "job_code", wxCMD_LINE_VAL_STRING, wxCMD_LINE_OPTION_MANDATORY},
                        {wxCMD_LINE_NONE}};

        wxCmdLineParser command_line_parser(command_line_descriptor, argc, argv);

        if ( command_line_parser.Parse(true) != 0 ) {
            wxPrintf("\n\n");
            exit(0);
        }

        // get the address and port of the gui (should be command line options).

        wxStringTokenizer gui_ip_address_tokens(command_line_parser.GetParam(0), ",");

        while ( gui_ip_address_tokens.HasMoreTokens( ) == true ) {
            current_address = gui_ip_address_tokens.GetNextToken( );
            possible_gui_addresses.Add(current_address);
            if ( junk_address.Hostname(current_address) == false ) {
                MyDebugPrint(" Error: Address (%s) - not recognized as an IP or hostname\n\n", current_address);
                exit(-1);
            };
        }

        if ( command_line_parser.GetParam(1).ToLong(&gui_port) == false ) {
            MyDebugPrint(" Error: Port (%s) - not recognized as a port\n\n", command_line_parser.GetParam(1));
            exit(-1);
        }

        if ( command_line_parser.GetParam(2).Len( ) != SOCKET_CODE_SIZE ) {
            {
                MyDebugPrint(" Error: Code (%s) - is the incorrect length(%i instead of %i)\n\n", command_line_parser.GetParam(2), command_line_parser.GetParam(2).Len( ), SOCKET_CODE_SIZE);
                exit(-1);
            }
        }

        // copy over job code.

        for ( counter = 0; counter < SOCKET_CODE_SIZE; counter++ ) {
            current_job_code[counter] = command_line_parser.GetParam(2).GetChar(counter);
        }

        // Attempt to connect to the gui..

        active_gui_address.Service(gui_port);
        gui_socket = new wxSocketClient( );
        gui_socket->Notify(false);

        for ( counter = 0; counter < possible_gui_addresses.GetCount( ); counter++ ) {
            active_gui_address.Hostname(possible_gui_addresses.Item(counter));
            MyDebugPrint("\nJob Controller: Trying to connect to %s:%i (timeout = 4 sec) ...\n", active_gui_address.IPAddress( ), active_gui_address.Service( ));

            gui_socket->Connect(active_gui_address, false);
            gui_socket->WaitOnConnect(20);

            if ( gui_socket->IsConnected( ) == false ) {
                gui_socket->Close( );
                MyDebugPrint("Connection Failed.\n\n");
            }
            else {
                break;
            }
        }

        if ( gui_socket->IsConnected( ) == false ) {
            gui_socket->Close( );
            wxPrintf("All connections Failed! Exiting...\n");
            ExitMainLoop( );
            return;
        }

        gui_socket->SetFlags(SOCKET_FLAGS);
        MyDebugPrint("Job Controller: Succeeded - Connection established!\n\n");

        number_of_workers_already_connected = 0;

        // monitor gui socket..

        MonitorSocket(gui_socket);

        // Job launching event..

        Bind(wxEVT_COMMAND_MYTHREAD_LAUNCHJOB, &JobControlApp::OnThreadLaunchJob, this);
        Bind(wxEVT_COMMAND_MYTHREAD_SENDINFO, &JobControlApp::OnThreadSendInfo, this);
    }
}

// BEGIN SSH_TUNNEL_HACK — reverse tunnel for remote jobs over untrusted networks
// Set to false to disable all tunnel logic and revert to original behavior.
static const bool USE_SSH_TUNNEL = true;

// Run a command and capture stdout. Returns empty string on failure.
static wxString RunAndCapture(const wxString& cmd) {
    wxString result;
    FILE*    fp = popen(cmd.ToUTF8( ).data( ), "r");
    if ( fp ) {
        char buf[512];
        while ( fgets(buf, sizeof(buf), fp) != NULL ) {
            result += buf;
        }
        pclose(fp);
    }
    result.Trim( );
    return result;
}

// Extract the SSH target (user@host or hostname) from a command string.
// Parses: ssh [-flags...] <target> "..."
// Returns the first non-flag token after "ssh".
static wxString ExtractSSHTarget(const wxString& cmd) {
    wxStringTokenizer tok(cmd, " ");
    bool              past_ssh = false;
    while ( tok.HasMoreTokens( ) ) {
        wxString token = tok.GetNextToken( );
        if ( token == "ssh" ) {
            past_ssh = true;
            continue;
        }
        if ( past_ssh ) {
            if ( token.StartsWith("-") )
                continue;
            return token;
        }
    }
    return "";
}

// END SSH_TUNNEL_HACK

void LaunchJobThread::LaunchRemoteJob( ) {
    MyDebugPrintGA1("LaunchRemoteJob ENTRY pid=%d tid=%ld", (int)getpid( ), (long)wxThread::GetCurrentId( ));
    MyDebugPrintGA1("LaunchRemoteJob ip_address='%s' port_number='%s' actual_jobs=%ld", ip_address, port_number, actual_number_of_jobs);
    MyDebugPrintGA1("LaunchRemoteJob run_profile.number_of_run_commands=%ld controller_address='%s'", current_run_profile.number_of_run_commands, current_run_profile.controller_address);
    MyDebugPrintGA1("LaunchRemoteJob run_profile.executable_name='%s'", current_run_profile.executable_name);
    long counter;
    long command_counter;
    long process_counter;
    long number_of_commands_to_run;
    long number_of_commands_run = 0;
    long number_to_run_for_this_command;

    wxIPV4address address;

    // for n processes (specified in the job package) we need to launch the specified command, along with our
    // IP address, port and job code..

    wxString executable;
    wxString executable_with_threads;
    wxString execution_command;

    // BEGIN SSH_TUNNEL
    // When USE_SSH_TUNNEL is true and a run command uses SSH, we set up
    // a reverse tunnel per SSH target so the remote worker can connect
    // back to salina via 127.0.0.1:<tunnel_port>.
    //
    // This works for ALL remote hosts (grantahlquist1, aws-host-*, etc.)
    // regardless of whether they have Tailscale installed.

    // Per-target tunnel info: ssh_target -> {effective_ip, effective_port}
    struct TunnelInfo {
        wxString ip;
        wxString port;
    };

    std::map<wxString, TunnelInfo> tunnel_map;

    // Convention: SSH config aliases prefixed with "remote-" need reverse
    // tunnels (the worker can't route back to salina directly).
    // Everything else is assumed local (salina, etna, siracusa, etc.)
    //
    // Example ~/.ssh/config entries:
    //   Host remote-grantahlquist1   ->  tunnel
    //   Host remote-aws-1            ->  tunnel
    //   Host remote-aws-2            ->  tunnel
    //   Host etna                    ->  no tunnel (local VLAN)
    static const wxString REMOTE_PREFIX = "remote-";

    auto NeedsTunnel = [&](const wxString& target) -> bool {
        wxString host = target;
        if ( host.Contains("@") )
            host = host.AfterFirst('@');
        return host.StartsWith(REMOTE_PREFIX);
    };

    int next_tunnel_port = 43210;

    if ( USE_SSH_TUNNEL ) {
        MyDebugPrintGA1("LaunchRemoteJob beginning tunnel scan loop, num_run_commands=%ld", current_run_profile.number_of_run_commands);
        // Scan all commands for SSH targets that need tunnels
        for ( long cc = 0; cc < current_run_profile.number_of_run_commands; cc++ ) {
            wxString cmd = current_run_profile.run_commands[cc].command_to_run;
            MyDebugPrintGA1("tunnel scan cc=%ld cmd='%s'", cc, cmd);
            wxString target = ExtractSSHTarget(cmd);
            MyDebugPrintGA1("tunnel scan cc=%ld extracted target='%s' empty=%d", cc, target, (int)target.IsEmpty( ));
            if ( target.IsEmpty( ) )
                continue;
            if ( ! NeedsTunnel(target) ) {
                MyDebugPrintGA1("tunnel scan cc=%ld target='%s' is LOCAL, skipping", cc, target);
                wxPrintf("SSH_TUNNEL: %s is local, no tunnel needed\n", target);
                continue;
            }
            if ( tunnel_map.find(target) != tunnel_map.end( ) ) {
                MyDebugPrintGA1("tunnel scan cc=%ld target='%s' already has tunnel, skipping", cc, target);
                continue; // Already have a tunnel for this target
            }

            MyDebugPrintGA1("tunnel scan cc=%ld target='%s' NEW REMOTE TARGET, setting up", cc, target);
            wxPrintf("SSH_TUNNEL: Setting up tunnel for %s\n", target);

            // 1. Verify connectivity
            MyDebugPrintGA1("tunnel target='%s' running ssh connectivity test", target);
            wxString ssh_test = RunAndCapture(wxString::Format(
                    "ssh -o ConnectTimeout=10 %s \"echo ok\" 2>/dev/null", target));
            MyDebugPrintGA1("tunnel target='%s' ssh_test result='%s'", target, ssh_test);
            if ( ssh_test != "ok" ) {
                MyDebugPrintGA1("tunnel target='%s' SSH TEST FAILED, aborting LaunchRemoteJob", target);
                QueueInfo(wxString::Format("SSH_TUNNEL_ERROR: Cannot SSH to %s _ aborting", target));
                return;
            }
            wxPrintf("SSH_TUNNEL: Connectivity to %s confirmed\n", target);
            MyDebugPrintGA1("tunnel target='%s' connectivity confirmed, beginning port try loop next_tunnel_port=%d", target, next_tunnel_port);

            // 2. Establish reverse tunnel
            // Pre-check: ssh -R with ExitOnForwardFailure still returns 0 when
            // the IPv4 loopback port is held by another process and sshd falls
            // back to binding [::1]:PORT only -- leaving the worker's IPv4
            // connect() silently routed to the conflicting process. Verify the
            // port is free on the remote's 127.0.0.1 BEFORE trusting ssh -R.
            bool     tunnel_ok = false;
            wxString remote_port;
            for ( int try_port = next_tunnel_port; try_port < next_tunnel_port + 40; try_port++ ) {
                wxString probe = RunAndCapture(wxString::Format(
                        "ssh -o ConnectTimeout=5 %s \"bash -c 'exec 3<>/dev/tcp/127.0.0.1/%d 2>/dev/null && echo in_use || echo free'\" 2>/dev/null",
                        target, try_port));
                MyDebugPrintGA1("tunnel target='%s' precheck port %d result='%s'", target, try_port, probe);
                if ( ! probe.Contains("free") ) {
                    // in_use or ambiguous -- skip to next port
                    continue;
                }
                MyDebugPrintGA1("tunnel target='%s' trying port %d (forwarding to localhost:%s)", target, try_port, port_number);
                wxString tunnel_cmd = wxString::Format(
                        "ssh -f -N -o ExitOnForwardFailure=yes -R %d:localhost:%s %s 2>/dev/null",
                        try_port, port_number, target);
                int ret = system(tunnel_cmd.ToUTF8( ).data( ));
                MyDebugPrintGA1("tunnel target='%s' port %d ssh -f -N returned %d", target, try_port, ret);
                if ( ret == 0 ) {
                    remote_port      = wxString::Format("%d", try_port);
                    next_tunnel_port = try_port + 1;
                    tunnel_ok        = true;
                    MyDebugPrintGA1("tunnel target='%s' SUCCESS on port %d", target, try_port);
                    break;
                }
            }
            if ( ! tunnel_ok ) {
                QueueInfo(wxString::Format("SSH_TUNNEL_ERROR: No free port for %s _ aborting", target));
                return;
            }

            tunnel_map[target] = {wxString("127.0.0.1"), remote_port};
            QueueInfo(wxString::Format("SSH_TUNNEL: %s -> 127.0.0.1:%s -> localhost:%s",
                                       target, remote_port, port_number));
        }

        // 3. Verify all tunnels are working before releasing to command loop.
        //    SSH to each remote and confirm the tunnel port is listening.
        //    This replaces manual launch ordering and delay hacks.
        MyDebugPrintGA1("tunnel verification phase: %zu tunnel(s) to verify", tunnel_map.size( ));
        for ( auto& kv : tunnel_map ) {
            wxString target = kv.first;
            wxString tport  = kv.second.port;
            MyDebugPrintGA1("verify target='%s' port=%s", target, tport);
            wxPrintf("SSH_TUNNEL: Verifying tunnel to %s (port %s)...\n", target, tport);

            bool verified = false;
            for ( int attempt = 0; attempt < 5; attempt++ ) {
                // Check that the tunnel port is listening on the remote (use bash /dev/tcp, no nc dependency)
                wxString check = RunAndCapture(wxString::Format(
                        "ssh -o ConnectTimeout=5 %s \"bash -c 'exec 3<>/dev/tcp/127.0.0.1/%s && echo tunnel_ok || echo tunnel_fail' 2>/dev/null\" 2>/dev/null",
                        target, tport));
                MyDebugPrintGA1("verify target='%s' attempt=%d check_output='%s'", target, attempt + 1, check);
                if ( check.Contains("tunnel_ok") ) {
                    wxPrintf("SSH_TUNNEL: Tunnel to %s verified OK\n", target);
                    MyDebugPrintGA1("verify target='%s' OK on attempt=%d", target, attempt + 1);
                    verified = true;
                    break;
                }
                wxPrintf("SSH_TUNNEL: Tunnel to %s not ready, retrying (%d/5)...\n", target, attempt + 1);
                wxMilliSleep(1000);
            }
            if ( ! verified ) {
                // Tunnel established but can't reach the port -- warn but continue
                // (the tunnel SSH process may just need a moment)
                wxPrintf("SSH_TUNNEL: WARNING: Could not verify tunnel to %s -- proceeding anyway\n", target);
                MyDebugPrintGA1("verify target='%s' UNVERIFIED, proceeding anyway", target);
            }
        }

        if ( tunnel_map.empty( ) ) {
            wxPrintf("SSH_TUNNEL: No remote SSH targets, using original ip/port\n");
            MyDebugPrintGA1("tunnel_map empty, no remote targets");
        }
        else {
            wxPrintf("SSH_TUNNEL: All %zu tunnel(s) ready\n", tunnel_map.size( ));
            MyDebugPrintGA1("tunnel_map ready, count=%zu", tunnel_map.size( ));
        }
    }
    else {
        wxPrintf("SSH_TUNNEL: Disabled (USE_SSH_TUNNEL=false)\n");
    }
    // END SSH_TUNNEL

    // Build executable strings.
    // executable_default uses original ip/port (local workers).
    // For SSH targets with tunnels, we build per-command executables in the loop below.
    wxString executable_default;

    if ( current_run_profile.controller_address == "" ) {
        executable_default = current_run_profile.executable_name + " " + ip_address + " " + port_number + " ";
    }
    else {
        executable_default = current_run_profile.executable_name + " " + current_run_profile.controller_address + " " + port_number + " ";
    }

    for ( counter = 0; counter < SOCKET_CODE_SIZE; counter++ ) {
        executable_default += job_code[counter];
    }

    wxPrintf("SSH_TUNNEL: executable_default: %s\n", executable_default);
    for ( auto& kv : tunnel_map ) {
        wxPrintf("SSH_TUNNEL: tunnel[%s]: 127.0.0.1:%s\n", kv.first, kv.second.port);
    }

    // Revert-debug-ga1: publish tunnel target list to JobControlApp so
    // HandleNewSocketConnection can set up master-port reverse tunnels to
    // these same hosts once master election completes. Must be done BEFORE
    // workers start connecting back (first socket accept fires from the
    // first system() call below).
    main_thread_pointer->remote_tunnel_targets.Empty( );
    for ( auto& kv : tunnel_map ) {
        main_thread_pointer->remote_tunnel_targets.Add(kv.first);
    }
    main_thread_pointer->master_tunnels_established = false;
    main_thread_pointer->master_tunnel_remote_port  = "";
    MyDebugPrintGA1("published remote_tunnel_targets count=%d", (int)main_thread_pointer->remote_tunnel_targets.GetCount( ));

    // Revert-debug-ga1: reserve ONE high port that is free on every remote
    // target. This port is used on BOTH sides of the master-port reverse
    // tunnel (the remote listen port AND the port we tell tunneled workers
    // to connect to). Scanning uniformly means every tunneled worker gets
    // the same master address, so we don't need to identify which peer
    // 127.0.0.1 connection belongs to which remote in the worker branch.
    if ( ! tunnel_map.empty( ) ) {
        bool reserved = false;
        for ( int try_port = next_tunnel_port; try_port < next_tunnel_port + 80; try_port++ ) {
            bool free_on_all = true;
            for ( auto& kv : tunnel_map ) {
                wxString target = kv.first;
                wxString probe  = RunAndCapture(wxString::Format(
                         "ssh -o ConnectTimeout=5 %s \"bash -c 'exec 3<>/dev/tcp/127.0.0.1/%d 2>/dev/null && echo in_use || echo free'\" 2>/dev/null",
                         target, try_port));
                MyDebugPrintGA1("master-port reserve probe target='%s' port=%d result='%s'", target, try_port, probe);
                if ( ! probe.Contains("free") ) {
                    free_on_all = false;
                    break;
                }
            }
            if ( free_on_all ) {
                main_thread_pointer->master_tunnel_remote_port = wxString::Format("%d", try_port);
                next_tunnel_port                               = try_port + 1;
                reserved                                       = true;
                MyDebugPrintGA1("master-port reserve SUCCESS port=%d", try_port);
                wxPrintf("SSH_TUNNEL: reserved master-tunnel port %d (uniform across %zu target(s))\n", try_port, tunnel_map.size( ));
                break;
            }
        }
        if ( ! reserved ) {
            QueueInfo("SSH_TUNNEL_ERROR: could not reserve a master-tunnel port free on all remotes _ aborting");
            MyDebugPrintGA1("master-port reserve FAILED, aborting");
            return;
        }
    }

    wxMilliSleep(2000);

    if ( actual_number_of_jobs < current_run_profile.ReturnTotalJobs( ) )
        number_of_commands_to_run = actual_number_of_jobs;
    else
        number_of_commands_to_run = current_run_profile.ReturnTotalJobs( );

    MyDebugPrintGA1("LaunchRemoteJob entering command launch loop num_commands=%ld total_to_run=%ld", current_run_profile.number_of_run_commands, number_of_commands_to_run);
    for ( command_counter = 0; command_counter < current_run_profile.number_of_run_commands; command_counter++ ) {
        MyDebugPrintGA1("cmd_loop iter=%ld num_copies=%i delay_ms=%i", command_counter, current_run_profile.run_commands[command_counter].number_of_copies, current_run_profile.run_commands[command_counter].delay_time_in_ms);
        if ( number_of_commands_to_run - number_of_commands_run < current_run_profile.run_commands[command_counter].number_of_copies )
            number_to_run_for_this_command = number_of_commands_to_run - number_of_commands_run;
        else
            number_to_run_for_this_command = current_run_profile.run_commands[command_counter].number_of_copies;

        // Pick the right executable for this command's target.
        // If the command has an SSH target with a tunnel, use 127.0.0.1:<tunnel_port>.
        // Otherwise use the default ip/port (local workers).
        wxString cmd_template   = current_run_profile.run_commands[command_counter].command_to_run;
        wxString cmd_ssh_target = ExtractSSHTarget(cmd_template);
        MyDebugPrintGA1("cmd_loop iter=%ld template='%s' ssh_target='%s'", command_counter, cmd_template, cmd_ssh_target);

        if ( ! cmd_ssh_target.IsEmpty( ) && tunnel_map.find(cmd_ssh_target) != tunnel_map.end( ) ) {
            TunnelInfo& ti = tunnel_map[cmd_ssh_target];
            MyDebugPrintGA1("cmd_loop iter=%ld picking TUNNEL executable target=%s tunnel_ip=%s tunnel_port=%s", command_counter, cmd_ssh_target, ti.ip, ti.port);
            if ( current_run_profile.controller_address == "" ) {
                executable = current_run_profile.executable_name + " " + ti.ip + " " + ti.port + " ";
            }
            else {
                executable = current_run_profile.executable_name + " " + current_run_profile.controller_address + " " + ti.port + " ";
            }
            for ( long jc = 0; jc < SOCKET_CODE_SIZE; jc++ ) {
                executable += job_code[jc];
            }
            wxPrintf("SSH_TUNNEL: Command %ld -> tunnel (%s via 127.0.0.1:%s)\n",
                     command_counter, cmd_ssh_target, ti.port);
        }
        else {
            executable = executable_default;
            MyDebugPrintGA1("cmd_loop iter=%ld picking DEFAULT executable (target empty or not in tunnel_map)", command_counter);
            wxPrintf("SSH_TUNNEL: Command %ld -> default\n", command_counter);
        }

        execution_command       = current_run_profile.run_commands[command_counter].command_to_run;
        executable_with_threads = executable + wxString::Format(" %i", current_run_profile.run_commands[command_counter].number_of_threads_per_copy);
        execution_command.Replace("$command", executable_with_threads);
        execution_command.Replace("$program_name", current_run_profile.executable_name);

        execution_command += "&";

        MyDebugPrintGA1("cmd_loop iter=%ld final execution_command='%s' copies=%ld", command_counter, execution_command, number_to_run_for_this_command);
        for ( process_counter = 0; process_counter < number_to_run_for_this_command; process_counter++ ) {

            wxMilliSleep(current_run_profile.run_commands[command_counter].delay_time_in_ms);

            if ( process_counter == 0 )
                QueueInfo(wxString::Format("Job Control : Executing '%s' %li times.", execution_command, number_to_run_for_this_command));

            //wxThreadEvent *test_event = new wxThreadEvent(wxEVT_COMMAND_MYTHREAD_LAUNCHJOB);
            //test_event->SetString(execution_command);

            //wxQueueEvent(main_thread_pointer, test_event);
            //wxExecute(execution_command);
            MyDebugPrintGA1("about to system() iter=%ld copy=%ld", command_counter, process_counter);
            int sysret = system(execution_command.ToUTF8( ).data( ));
            MyDebugPrintGA1("system() returned %d for iter=%ld copy=%ld", sysret, command_counter, process_counter);
            number_of_commands_run++;
        }
    }

    MyDebugPrintGA1("LaunchRemoteJob END num_commands_run=%ld waiting on socket events", number_of_commands_run);
    // now we wait for the connections - this is taken care of as server events..
}

void LaunchJobThread::QueueInfo(wxString info_to_queue) {
    wxThreadEvent* test_event = new wxThreadEvent(wxEVT_COMMAND_MYTHREAD_SENDINFO);
    test_event->SetString(info_to_queue);

    wxQueueEvent(main_thread_pointer, test_event);
}

void JobControlApp::OnThreadLaunchJob(wxThreadEvent& event) {
    if ( wxExecute(event.GetString( )) == 0 ) {
        SendError(wxString::Format("Error: Failed to launch (%s)", event.GetString( )));
    }
}

void JobControlApp::OnThreadSendInfo(wxThreadEvent& my_event) {
    SendInfo(my_event.GetString( ));
}

void JobControlApp::SendError(wxString error_to_send) {
    WriteToSocket(gui_socket, socket_i_have_an_error, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
    SendwxStringToSocket(&error_to_send, gui_socket);
}

void JobControlApp::SendInfo(wxString info_to_send) {
    WriteToSocket(gui_socket, socket_i_have_info, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
    SendwxStringToSocket(&info_to_send, gui_socket);
}

void JobControlApp::SendJobFinished(int job_number) {
    WriteToSocket(gui_socket, socket_job_finished, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
    WriteToSocket(gui_socket, &job_number, sizeof(int), true, "SendJobNumber", FUNCTION_DETAILS_AS_WXSTRING);
}

void JobControlApp::SendJobResult(JobResult* job_to_send) {
    WriteToSocket(gui_socket, socket_job_result, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
    job_to_send->SendToSocket(gui_socket);
}

void JobControlApp::SendJobResultQueue(ArrayofJobResults& queue_to_send) {
    WriteToSocket(gui_socket, socket_job_result_queue, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
    SendResultQueueToSocket(gui_socket, queue_to_send);
}

void JobControlApp::SendAllJobsFinished(long total_timing_from_master) {
    WriteToSocket(gui_socket, socket_all_jobs_finished, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
    WriteToSocket(gui_socket, &total_timing_from_master, sizeof(long), true, "SendTotalMillisecondsSpentOnThreads", FUNCTION_DETAILS_AS_WXSTRING);
}

void JobControlApp::SendNumberofConnections( ) {
    WriteToSocket(gui_socket, socket_number_of_connections, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
    WriteToSocket(gui_socket, &number_of_workers_already_connected, 4, true, "SendNumberOfConnections", FUNCTION_DETAILS_AS_WXSTRING);

    int number_of_commands_to_run;

    if ( current_job_package.number_of_jobs + 1 < current_job_package.my_profile.ReturnTotalJobs( ) )
        number_of_commands_to_run = current_job_package.number_of_jobs + 1;
    else
        number_of_commands_to_run = current_job_package.my_profile.ReturnTotalJobs( );

    if ( number_of_workers_already_connected == number_of_commands_to_run ) {

        ShutDownServer( );
        MyDebugPrint("Socket Server is now shutdown\n");
    }
}

//////////////////////////////////////////////////////////////////////////////////////
//                      SOCKET OVERRIDES  - FIRST SERVER                            //
//////////////////////////////////////////////////////////////////////////////////////

void JobControlApp::HandleNewSocketConnection(wxSocketBase* new_connection, unsigned char* identification_code) {
    MyDebugPrintGA1("HandleNewSocketConnection ENTRY socket=%p", (void*)new_connection);

    if ( new_connection == NULL ) {
        MyDebugPrintGA1("HandleNewSocketConnection NULL socket, returning");
        return;
    }

    wxString peer_ip = ReturnIPAddressFromSocket(new_connection);
    MyDebugPrintGA1("HandleNewSocketConnection peer_ip='%s'", peer_ip);

    if ( (memcmp(identification_code, current_job_code, SOCKET_CODE_SIZE) != 0) ) {
        MyDebugPrintGA1("HandleNewSocketConnection BAD JOB ID from peer=%s", peer_ip);
        SendError("Unknown Job ID (Job Control), leftover from a previous job? - Closing Connection");
        new_connection->Destroy( ); // should not be monitoring so can just destroy it
        new_connection = NULL;
    }
    else {
        MyDebugPrintGA1("HandleNewSocketConnection valid job id, peer=%s have_assigned_master=%d", peer_ip, (int)have_assigned_master);
        // one of the workers has connected to us.  If it is the first one then
        // we need to make it the master, tell it to start a socket server
        // and send us the address so we can pass it on to all future workers.
        // If we have already assigned the master, then we just need to send it
        // the masters address.

        if ( have_assigned_master == false ) // we don't have a master, so assign it
        {
            MyDebugPrintGA1("HandleNewSocketConnection ASSIGNING MASTER peer=%s", peer_ip);

            master_socket        = new_connection;
            have_assigned_master = true;

            WriteToSocket(new_connection, socket_you_are_the_master, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
            MyDebugPrintGA1("HandleNewSocketConnection sent socket_you_are_the_master, sending job package");
            // See JobPackage::SendJobPackage() Doxygen for encoding order specification
            current_job_package.SendJobPackage(new_connection);
            MyDebugPrintGA1("HandleNewSocketConnection job package sent, waiting for master to report ip/port");

            bool no_error;
            master_ip_address = ReceivewxStringFromSocket(new_connection, no_error);
            master_port       = ReceivewxStringFromSocket(new_connection, no_error);
            MyDebugPrintGA1("HandleNewSocketConnection MASTER REPORTED ip='%s' port='%s' no_error=%d", master_ip_address, master_port, (int)no_error);

            // Revert-debug-ga1: set up reverse tunnel for master port to each
            // remote target, so tunneled workers can reach the master via
            // 127.0.0.1:<master_tunnel_remote_port> on their own loopback.
            // The REMOTE listen port is master_tunnel_remote_port (pre-scanned
            // in LaunchRemoteJob as a high port free on every target) -- it
            // MUST NOT be the same as master_port because another user's
            // stale cisTEM worker could silently hold master_port=3002 on a
            // remote, letting ssh -R bind only [::1]:3002 and shadow the
            // IPv4 loopback we actually need. The SALINA side of the forward
            // still targets 127.0.0.1:<master_port> (where the just-elected
            // master actually listens).
            if ( ! master_tunnels_established && ! remote_tunnel_targets.IsEmpty( ) ) {
                if ( master_tunnel_remote_port.IsEmpty( ) ) {
                    wxPrintf("SSH_TUNNEL: ERROR: no master_tunnel_remote_port reserved, tunneled workers will fail\n");
                    MyDebugPrintGA1("master-port reverse tunnel skip: no reserved port");
                }
                else {
                    MyDebugPrintGA1("setting up master-port reverse tunnels count=%d remote_port=%s -> salina master_port=%s", (int)remote_tunnel_targets.GetCount( ), master_tunnel_remote_port, master_port);
                    for ( size_t ii = 0; ii < remote_tunnel_targets.GetCount( ); ii++ ) {
                        wxString target     = remote_tunnel_targets[ii];
                        wxString tunnel_cmd = wxString::Format(
                                "ssh -f -N -o ExitOnForwardFailure=yes -R %s:127.0.0.1:%s %s 2>/dev/null",
                                master_tunnel_remote_port, master_port, target);
                        MyDebugPrintGA1("master tunnel target='%s' cmd='%s'", target, tunnel_cmd);
                        int ret = system(tunnel_cmd.ToUTF8( ).data( ));
                        MyDebugPrintGA1("master tunnel target='%s' ssh returned %d", target, ret);
                        if ( ret != 0 ) {
                            wxPrintf("SSH_TUNNEL: WARNING: master tunnel to %s failed (ret=%d)\n", target, ret);
                        }
                        else {
                            wxPrintf("SSH_TUNNEL: master tunnel to %s established (remote %s -> salina 127.0.0.1:%s)\n", target, master_tunnel_remote_port, master_port);
                        }
                    }
                    master_tunnels_established = true;
                    MyDebugPrintGA1("master-port reverse tunnels done");
                }
            }

            // monitor the master socket..

            MonitorSocket(new_connection);

            number_of_workers_already_connected++;
            MyDebugPrintGA1("HandleNewSocketConnection master count now %d", (int)number_of_workers_already_connected);
            SendNumberofConnections( );
        }
        else // we have a master, tell this worker who it's master is.
        {
            // Revert-debug-ga1: if the worker connected via a reverse tunnel
            // (peer_ip == 127.0.0.1) the master VLAN IP we received from the
            // master is not routable from the remote host. Substitute both
            // the ip AND the port: the worker must connect to
            // 127.0.0.1:<master_tunnel_remote_port> on its own loopback, which
            // the reverse tunnel forwards back to salina 127.0.0.1:<master_port>.
            wxString effective_master_ip   = master_ip_address;
            wxString effective_master_port = master_port;
            if ( peer_ip == "127.0.0.1" ) {
                effective_master_ip = "127.0.0.1";
                if ( ! master_tunnel_remote_port.IsEmpty( ) ) {
                    effective_master_port = master_tunnel_remote_port;
                }
                MyDebugPrintGA1("tunneled worker detected, rewriting master '%s':'%s' -> '%s':'%s'", master_ip_address, master_port, effective_master_ip, effective_master_port);
            }
            MyDebugPrintGA1("HandleNewSocketConnection ASSIGNING WORKER peer=%s, telling it master='%s':'%s'", peer_ip, effective_master_ip, effective_master_port);
            WriteToSocket(new_connection, socket_you_are_a_worker, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
            SendwxStringToSocket(&effective_master_ip, new_connection);
            SendwxStringToSocket(&effective_master_port, new_connection);
            MyDebugPrintGA1("HandleNewSocketConnection worker peer=%s told master, monitoring socket", peer_ip);

            // that should be the end of our interactions with the worker
            // it should disconnect itself. We have to monitor it however so that
            // we can destroy it once it disconnects..

            MonitorSocket(new_connection);

            number_of_workers_already_connected++;
            MyDebugPrintGA1("HandleNewSocketConnection worker count now %d", (int)number_of_workers_already_connected);
            SendNumberofConnections( );
        }
    }

    delete[] identification_code;
}

//////////////////////////////////////////////////////////////////////////////////////
//                      THESE SHOULD BE FROM THE GUI                                //
//////////////////////////////////////////////////////////////////////////////////////

void JobControlApp::HandleSocketYouAreConnected(wxSocketBase* connected_socket) {
    // we are connected to the gui, ask it to send job details..

    WriteToSocket(connected_socket, socket_send_job_details, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
}

void JobControlApp::HandleSocketJobPackage(wxSocketBase* connected_socket, JobPackage* received_package) {
    MyDebugPrintGA1("HandleSocketJobPackage ENTRY received package from gui");
    // we have a job package from the gui, copy it.. start a server and launch the jobs..

    current_job_package = *received_package;
    delete received_package;
    MyDebugPrintGA1("HandleSocketJobPackage copied package, profile.num_run_commands=%ld total_jobs=%i", current_job_package.my_profile.number_of_run_commands, current_job_package.number_of_jobs);

    SetupServer( );
    my_port        = ReturnServerPort( );
    my_port_string = ReturnServerPortString( );
    MyDebugPrintGA1("HandleSocketJobPackage SetupServer done my_port=%ld my_port_string='%s'", my_port, my_port_string);

    wxString      current_address_according_to_gui;
    wxString      ip_address_string;
    wxArrayString buffer_addresses = ReturnServerAllIpAddresses( );

    current_address_according_to_gui = ReturnIPAddressFromSocket(gui_socket);

    my_possible_ip_addresses.Clear( );
    if ( current_address_according_to_gui.IsEmpty( ) == false )
        my_possible_ip_addresses.Add(current_address_according_to_gui);

    for ( int counter = 0; counter < buffer_addresses.GetCount( ); counter++ ) {
        if ( buffer_addresses.Item(counter) != current_address_according_to_gui )
            my_possible_ip_addresses.Add(buffer_addresses.Item(counter));
    }

    for ( int counter = 0; counter < my_possible_ip_addresses.GetCount( ); counter++ ) {
        if ( counter != 0 )
            ip_address_string += ",";
        ip_address_string += my_possible_ip_addresses.Item(counter);
    }

    MyDebugPrintGA1("HandleSocketJobPackage final ip_address_string='%s' creating LaunchJobThread", ip_address_string);
    LaunchJobThread* launch_thread = new LaunchJobThread(this, current_job_package.my_profile, ip_address_string, my_port_string, current_job_code, current_job_package.number_of_jobs);
    MyDebugPrintGA1("HandleSocketJobPackage LaunchJobThread created, calling Run()");

    if ( launch_thread->Run( ) != wxTHREAD_NO_ERROR ) {
        MyPrintWithDetails("Can't create the launch thread!");
        delete launch_thread;
        ExitMainLoop( );
        return;
    }
}

void JobControlApp::HandleSocketTimeToDie(wxSocketBase* connected_socket) {
    if ( have_assigned_master == true ) {
        WriteToSocket(master_socket, socket_time_to_die, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
        StopMonitoringAndDestroySocket(master_socket);
    }

    ShutDownServer( );

    // close GUI connection

    StopMonitoringAndDestroySocket(gui_socket);

    ShutDownSocketMonitor( );

    // die..

    ExitMainLoop( );
    exit(0);
    return;
}

//////////////////////////////////////////////////////////////////////////////////////
//                    THESE SHOULD BE FROM THE MASTER                               //
//////////////////////////////////////////////////////////////////////////////////////

//void JobControlApp::HandleSocketSendJobDetails(wxSocketBase *connected_socket)
//{
// send it the job package we should have already received from the gui..

//	WriteToSocket(connected_socket, socket_sending_job_package, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
//	current_job_package.SendJobPackage(connected_socket);
//}

void JobControlApp::HandleSocketIHaveAnError(wxSocketBase* connected_socket, wxString error_message) {
    // pass the error message up to the GUI..

    SendError(error_message);
}

void JobControlApp::HandleSocketIHaveInfo(wxSocketBase* connected_socket, wxString info_message) {
    // pass the error message up to the GUI..

    SendInfo(info_message);
}

void JobControlApp::HandleSocketJobResult(wxSocketBase* connected_socket, JobResult* received_result) {
    // pass the result onto the GUI...

    SendJobResult(received_result);

    // delete it..

    delete received_result;
}

void JobControlApp::HandleSocketJobResultQueue(wxSocketBase* connected_socket, ArrayofJobResults* received_queue) {
    // pass it on and delete..

    SendJobResultQueue(*received_queue);
    number_of_received_jobs += received_queue->GetCount( );

    delete received_queue;
}

void JobControlApp::HandleSocketJobFinished(wxSocketBase* connected_socket, int finished_job_number) {
    // pass on to the gui..

    SendJobFinished(finished_job_number);
}

void JobControlApp::HandleSocketAllJobsFinished(wxSocketBase* connected_socket, long received_timing_in_milliseconds) {
    // pass on to the gui..

    SendAllJobsFinished(received_timing_in_milliseconds);
    all_jobs_are_finished = true;

    // don't die, wait for GUI to kill me..
}

void JobControlApp::HandleSocketTemplateMatchResultReady(wxSocketBase* connected_socket, int& image_number, float& high_res_limit_used, float& threshold_used, ArrayOfTemplateMatchFoundPeakInfos& peak_infos, ArrayOfTemplateMatchFoundPeakInfos& peak_changes) {
    // pass on to the gui..

    SendTemplateMatchingResultToSocket(gui_socket, image_number, high_res_limit_used, threshold_used, peak_infos, peak_changes);
}

void JobControlApp::HandleSocketDisconnect(wxSocketBase* connected_socket) {
    MyDebugPrintGA1("HandleSocketDisconnect ENTRY socket=%p gui_socket=%p master_socket=%p", (void*)connected_socket, (void*)gui_socket, (void*)master_socket);
    // what disconnected..

    if ( connected_socket == gui_socket ) {
        MyDebugPrintGA1("HandleSocketDisconnect GUI DISCONNECT, aborting");
        MyDebugPrint("Got a disconnect from the GUI - aborting");

        if ( have_assigned_master == true ) {
            WriteToSocket(master_socket, socket_time_to_die, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
            StopMonitoringAndDestroySocket(master_socket);
        }

        ShutDownServer( );

        // close GUI connection

        StopMonitoringAndDestroySocket(gui_socket);

        ShutDownSocketMonitor( );

        // die..

        ExitMainLoop( );
        exit(0);
        return;
    }
    else if ( connected_socket == master_socket ) {
        //must be from the master..
        //Have we send all jobs are finished?

        if ( all_jobs_are_finished == false ) // something went wrong
        {
            MyDebugPrint("Controller got disconnect from master...");
            SendError("Controller received a disconnect from the master before the job was finished..");

            ShutDownServer( );

            // close GUI connection

            StopMonitoringAndDestroySocket(gui_socket);
            ShutDownSocketMonitor( );

            // die..

            ExitMainLoop( );
            exit(0);
            return;
        }

        // otherwise we don't care.. still wait for gui to kill us..
    }
    else // Must be a worker dropping us to connect to the master
    {
        StopMonitoringAndDestroySocket(connected_socket);
    }
}
