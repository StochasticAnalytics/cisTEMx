#ifndef _SRC_GUI_TEMPLATEMATCHQUEUELOGGER_H_
#define _SRC_GUI_TEMPLATEMATCHQUEUELOGGER_H_

#include <wx/wx.h>
#include <wx/log.h>
#include <wx/datetime.h>
#include <wx/filename.h>
#include <wx/ffile.h>
#include <memory>

#define cisTEM_QM_LOGGING

// Only enable logging functionality when cisTEM_QM_LOGGING is defined
#ifdef cisTEM_QM_LOGGING

#include <fstream>
#include <sstream>
#include <iomanip>

// Define which logging categories are enabled at compile time
// Comment out any you don't want to log
#define QM_TRACE_METHOD     // Method entry/exit with timestamps
#define QM_TRACE_DB         // Database transaction initiation and results
// #define QM_TRACE_DB_VALUES  // Detailed database values (verbose) - disabled by default
#define QM_TRACE_DB_SCHEMA  // Database schema validation and table existence checks
#define QM_TRACE_STATE      // Queue state changes and transitions
#define QM_TRACE_UI         // UI updates and user interactions
#define QM_TRACE_SEARCH     // Search execution flow and status
#define QM_TRACE_ERROR      // Error conditions and warnings
// #define QM_TRACE_DEBUG      // Detailed debug output (queue dumps, etc.) - disabled by default

// Simple direct file logger
class QueueManagerLogger {
  private:
    static std::ofstream log_file;
    static wxDateTime    start_time;
    static bool          is_enabled;
    static wxString      log_file_path;

  public:
    static bool Initialize() {
        if (is_enabled) return true;

        // Create log file in /tmp directory for debugging
        wxDateTime now = wxDateTime::Now();
        log_file_path = wxString::Format("/tmp/QM_log_%s.txt",
                                        now.Format("%Y%m%d_%H%M%S"));

        log_file.open(log_file_path.ToStdString(), std::ios::out | std::ios::app);
        if (!log_file.is_open()) {
            return false;
        }

        start_time = now;
        is_enabled = true;

        log_file << "\n=== Queue Manager Logging Session Started: "
                 << now.Format("%Y-%m-%d %H:%M:%S").ToStdString()
                 << " ===\n" << std::flush;

        return true;
    }

    static void Shutdown() {
        if (!is_enabled) return;

        wxDateTime end_time = wxDateTime::Now();
        wxTimeSpan duration = end_time - start_time;

        log_file << "=== Session Ended: "
                 << end_time.Format("%Y-%m-%d %H:%M:%S").ToStdString()
                 << " (Duration: " << duration.Format("%H:%M:%S").ToStdString()
                 << ") ===\n\n" << std::flush;

        log_file.close();
        is_enabled = false;
    }

    static void Log(const wxString& category, const wxString& message) {
        if (!is_enabled) return;

        wxDateTime now = wxDateTime::Now();
        wxTimeSpan elapsed = now - start_time;

        log_file << "[" << now.Format("%H:%M:%S").ToStdString()
                 << " +" << elapsed.Format("%M:%S").ToStdString()
                 << "." << std::setfill('0') << std::setw(3) << now.GetMillisecond()
                 << "] [" << category.ToStdString() << "] "
                 << message.ToStdString() << "\n" << std::flush;
    }

    static bool IsEnabled() { return is_enabled; }
    static wxString GetLogPath() { return log_file_path; }
};

// Simplified manager that just wraps our direct logger
class QueueManagerLogManager {
  public:
    static void EnableLogging(bool enable = true) {
        if (enable) {
            QueueManagerLogger::Initialize();
        } else {
            DisableLogging();
        }
    }

    static void DisableLogging() {
        QueueManagerLogger::Shutdown();
    }

    static bool IsLoggingEnabled() {
        return QueueManagerLogger::IsEnabled();
    }

    static wxString GetLogFilePath() {
        return QueueManagerLogger::GetLogPath();
    }
};

// Convenience macros for logging with different categories
// Each macro only logs if its corresponding QM_TRACE_* is defined at compile time

#ifdef QM_TRACE_METHOD
#define QM_LOG_METHOD_ENTRY(method) QueueManagerLogger::Log("METHOD", wxString::Format("ENTER: %s", method))
#define QM_LOG_METHOD_EXIT(method) QueueManagerLogger::Log("METHOD", wxString::Format("EXIT: %s", method))
#else
#define QM_LOG_METHOD_ENTRY(method) ((void)0)
#define QM_LOG_METHOD_EXIT(method) ((void)0)
#endif

#ifdef QM_TRACE_DB
#define QM_LOG_DB(msg, ...) QueueManagerLogger::Log("DB", wxString::Format(msg, ##__VA_ARGS__))
#else
#define QM_LOG_DB(msg, ...) ((void)0)
#endif

#ifdef QM_TRACE_DB_VALUES
#define QM_LOG_DB_VALUES(msg, ...) QueueManagerLogger::Log("DB_VALUES", wxString::Format(msg, ##__VA_ARGS__))
#else
#define QM_LOG_DB_VALUES(msg, ...) ((void)0)
#endif

#ifdef QM_TRACE_STATE
#define QM_LOG_STATE(msg, ...) QueueManagerLogger::Log("STATE", wxString::Format(msg, ##__VA_ARGS__))
#else
#define QM_LOG_STATE(msg, ...) ((void)0)
#endif

#ifdef QM_TRACE_UI
#define QM_LOG_UI(msg, ...) QueueManagerLogger::Log("UI", wxString::Format(msg, ##__VA_ARGS__))
#else
#define QM_LOG_UI(msg, ...) ((void)0)
#endif

#ifdef QM_TRACE_SEARCH
#define QM_LOG_SEARCH(msg, ...) QueueManagerLogger::Log("SEARCH", wxString::Format(msg, ##__VA_ARGS__))
#else
#define QM_LOG_SEARCH(msg, ...) ((void)0)
#endif

#ifdef QM_TRACE_ERROR
#define QM_LOG_ERROR(msg, ...) QueueManagerLogger::Log("ERROR", wxString::Format(msg, ##__VA_ARGS__))
#else
#define QM_LOG_ERROR(msg, ...) ((void)0)
#endif

#ifdef QM_TRACE_DEBUG
#define QM_LOG_DEBUG(msg, ...) QueueManagerLogger::Log("DEBUG", wxString::Format(msg, ##__VA_ARGS__))
#else
#define QM_LOG_DEBUG(msg, ...) ((void)0)
#endif

#ifdef QM_TRACE_DB_SCHEMA
#define QM_LOG_DB_SCHEMA(msg, ...) QueueManagerLogger::Log("DB_SCHEMA", wxString::Format(msg, ##__VA_ARGS__))
#else
#define QM_LOG_DB_SCHEMA(msg, ...) ((void)0)
#endif

#else // cisTEM_QM_LOGGING not defined

// Define empty macros when logging is disabled
#define QM_LOG_METHOD_ENTRY(method) ((void)0)
#define QM_LOG_METHOD_EXIT(method) ((void)0)
#define QM_LOG_DB(msg, ...) ((void)0)
#define QM_LOG_DB_VALUES(msg, ...) ((void)0)
#define QM_LOG_STATE(msg, ...) ((void)0)
#define QM_LOG_UI(msg, ...) ((void)0)
#define QM_LOG_SEARCH(msg, ...) ((void)0)
#define QM_LOG_ERROR(msg, ...) ((void)0)
#define QM_LOG_DEBUG(msg, ...) ((void)0)
#define QM_LOG_DB_SCHEMA(msg, ...) ((void)0)

// Stub class when logging is disabled
class QueueManagerLogManager {
  public:
    static void EnableLogging(bool enable = true) {}

    static void DisableLogging( ) {}

    static void EnableAllTraceMasks( ) {}

    static void DisableAllTraceMasks( ) {}

    static void SetTraceMask(const wxString& mask, bool enable) {}

    static bool IsLoggingEnabled( ) { return false; }

    static wxString GetLogFilePath( ) { return wxEmptyString; }
};

#endif // cisTEM_QM_LOGGING

#endif // _SRC_GUI_TEMPLATEMATCHQUEUELOGGER_H_