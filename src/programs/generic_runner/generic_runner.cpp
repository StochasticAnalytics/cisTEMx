#include "../../core/core_headers.h"
#include <sys/wait.h>

class
        GenericRunnerApp : public MyApp {
  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

    // Master-side override: consolidates results from all workers
    // Signature must exactly match myapp.h:230 — use override keyword
    void MasterHandleProgramDefinedResult(float* result_array, long array_size,
                                          int result_number,
                                          int number_of_expected_results) override;

  private:
    bool IsAllowedBinary(const wxString& binary_name);

    // Track consolidated results (master only)
    int number_of_received_results;
    int total_expected_results;
};

IMPLEMENT_APP(GenericRunnerApp)

bool GenericRunnerApp::IsAllowedBinary(const wxString& binary_name) {
    // Spec item 2: Only whitelisted test binaries are allowed
    // Extract just the filename for comparison (binary_name may be a full path)
    wxFileName binary_path(binary_name);
    wxString   name_only = binary_path.GetFullName( );
    return (name_only == "unit_test_runner" || name_only == "console_test");
}

void GenericRunnerApp::DoInteractiveUserInput( ) {
    UserInput* my_input = new UserInput("GenericRunner", 1.00);

    wxString binary_to_run = my_input->GetFilenameFromUser(
            "Binary to run",
            "Name of the whitelisted binary to execute",
            "console_test", true);

    delete my_input;

    // Format: "ti" = text (binary name) + int (total number of jobs)
    // In interactive mode, there is 1 job
    my_current_job.ManualSetArguments("ti",
                                      binary_to_run.ToUTF8( ).data( ),
                                      1);
}

bool GenericRunnerApp::DoCalculation( ) {
    // Read job arguments — works identically whether set by
    // DoInteractiveUserInput() (local) or by GUI panel AddJob() (sockets)
    wxString binary_to_run  = my_current_job.arguments[0].ReturnStringArgument( );
    int      number_of_jobs = my_current_job.arguments[1].ReturnIntegerArgument( );

    // Spec item 2: Whitelist enforcement BEFORE execution
    if ( ! IsAllowedBinary(binary_to_run) ) {
        SendError(wxString::Format("Whitelist violation: '%s' is not an allowed binary. "
                                   "Only 'unit_test_runner' and 'console_test' are permitted.",
                                   binary_to_run));
        return false;
    }

    int my_job_number = my_current_job.job_number;

    // Resolve binary path relative to this executable's directory.
    // The test binary is co-located with generic_runner in the same build/install dir.
    // This works across machines with different filesystem layouts.
    wxFileName my_exe(wxStandardPaths::Get( ).GetExecutablePath( ));
    wxString   binary_full_path = my_exe.GetPath( ) + wxFileName::GetPathSeparator( ) + binary_to_run;

    SendInfo(wxString::Format("Worker %i: Starting execution of '%s'", my_job_number, binary_full_path));

    // Execute binary via popen() to capture stdout
    wxString captured_output;
    int      exit_code = -1;

    FILE* pipe = popen(binary_full_path.ToUTF8( ).data( ), "r");
    if ( pipe != NULL ) {
        char line_buffer[1024];
        while ( fgets(line_buffer, sizeof(line_buffer), pipe) != NULL ) {
            captured_output += line_buffer;
        }
        int wait_status = pclose(pipe);
        exit_code       = WIFEXITED(wait_status) ? WEXITSTATUS(wait_status) : -1;
    }

    SendInfo(wxString::Format("Worker %i: %s exited with code %i", my_job_number, binary_full_path, exit_code));

    // Socket-based result paths — only when launched by the job controller
    if ( is_running_locally == false ) {
        // Spec item 4: Send results via SendProgramDefinedResultToMaster()
        // MUST be heap-allocated — the event infrastructure takes ownership and calls delete[]
        float* result_data = new float[2];
        result_data[0]     = static_cast<float>(exit_code);
        result_data[1]     = static_cast<float>(my_job_number);

        SendProgramDefinedResultToMaster(result_data, 2, my_job_number, number_of_jobs);
    }

    // Send result + captured stdout to GUI via AddJobToResultQueue.
    // Encoding: float[0] = exit_code, float[1] = job_number,
    //           float[2] = string_length, float[3] = checksum,
    //           float[4..N] = packed bytes (4 chars per float)
    if ( is_running_locally == false ) {
        const char* output_cstr       = captured_output.ToUTF8( ).data( );
        int         string_len        = strlen(output_cstr);
        int         floats_for_string = (string_len + 3) / 4; // ceil(len/4)
        int         total_floats      = 4 + floats_for_string;

        // Compute checksum for round-trip verification
        unsigned int checksum = 0;
        for ( int c = 0; c < string_len; c++ ) {
            checksum += static_cast<unsigned char>(output_cstr[c]);
        }

        float* gui_data = new float[total_floats];
        gui_data[0]     = static_cast<float>(exit_code);
        gui_data[1]     = static_cast<float>(my_job_number);
        gui_data[2]     = static_cast<float>(string_len);
        gui_data[3]     = static_cast<float>(checksum);

        // Pack string bytes into floats (4 bytes per float via memcpy)
        memset(&gui_data[4], 0, floats_for_string * sizeof(float));
        memcpy(&gui_data[4], output_cstr, string_len);

        JobResult* gui_result  = new JobResult;
        gui_result->job_number = my_job_number;
        gui_result->SetResult(total_floats, gui_data);
        AddJobToResultQueue(gui_result);

        delete[] gui_data;
    }

    return true;
}

void GenericRunnerApp::MasterHandleProgramDefinedResult(float* result_array, long array_size,
                                                        int result_number,
                                                        int number_of_expected_results) {
    // Master receives results from each worker
    int exit_code  = static_cast<int>(result_array[0]);
    int job_number = static_cast<int>(result_array[1]);

    number_of_received_results++;
    total_expected_results = number_of_expected_results;
}
