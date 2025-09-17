# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

cisTEM is a scientific computing application for cryo-electron microscopy (cryo-EM) image processing and 3D reconstruction. It's written primarily in C++ with CUDA GPU acceleration support and includes both command-line programs and a wxWidgets-based GUI.

## Build System

cisTEM uses GNU Autotools as the primary build system with Intel MKL for optimized FFT operations.

### Developer Build Process

For development builds, follow this sequence from the project root:

1. **Initial setup after clean install:**

   ```bash
   ./regenerate_containers.sh
   ./regenerate_project.b
   ```

2. **Configure and build using VS Code tasks:**
   - Use VS Code Command Palette → Tasks: Run Task
   - Default profiles:
     - `Configure cisTEM DEBUG build`
     - `BUILD cisTEM DEBUG`

3. **Manual build process:**

   ```bash
   # Example debug build with Intel compiler and GPU support
   mkdir -p build/intel-gpu-debug-static
   cd build/intel-gpu-debug-static
   CC=icc CXX=icpc ../../configure --enable-debugmode --enable-gpu-debug \
     --with-wx-config=/opt/WX/icc-static/bin/wx-config \
     --enable-staticmode --with-cuda=/usr/local/cuda \
     --enable-experimental --enable-openmp
   make -j8
   ```

### CMake (Alternative)

```bash
mkdir build && cd build
cmake -DBUILD_STATIC_BINARIES=ON -DBUILD_EXPERIMENTAL_FEATURES=OFF ..
make -j$(nproc)
```

Build options for CMake:

- `BUILD_STATIC_BINARIES=ON/OFF` - Static vs dynamic linking
- `BUILD_EXPERIMENTAL_FEATURES=ON/OFF` - Include experimental code
- `BUILD_OpenMP=ON/OFF` - Enable OpenMP multithreading

### Docker Development Environment
The project uses a Docker container for cross-platform development. Container definitions are in `scripts/containers/` with base and top layer architecture.

## Architecture

### Core Components

- **src/core/** - Core libraries and data structures
  - Image processing classes (`image.h`, `mrc_file.h`)
  - Mathematical utilities (`matrix.h`, `functions.h`)
  - Database interface (SQLite integration)
  - GPU acceleration headers and CUDA code

- **src/gui/** - wxWidgets-based graphical interface
  - Main application framework
  - Panel components for different workflows
  - Icon resources and UI elements

- **src/programs/** - Command-line executables
  - Individual processing programs (ctffind, unblur, refine3d, etc.)
  - Each program is self-contained with its own main()

### Key Dependencies

- **Intel MKL** - Primary FFT library for optimized performance
- **FFTW** - Alternative FFT library (maintained for portability but not officially supported due to restrictive licensing)
- **wxWidgets** - GUI framework (typically 3.0.5 stable)
- **LibTIFF** - TIFF image file support
- **SQLite** - Database backend
- **CUDA** - GPU acceleration (optional)
- **Intel C++ Compiler (icc/icpc)** - Primary compiler for performance builds

## Development Commands

### Testing

cisTEM has a multi-tiered testing approach:

```bash
# Unit tests - Test individual methods and functions
./unit_test_runner

# Console tests - Mid-complexity tests of single methods
./console_test

# Functional tests - Test complete workflows and image processing tasks
./samples_functional_testing

# Quick test executable
./quick_test
```

**Testing hierarchy:**

- `unit_test_runner` - Basic unit tests for core functionality
- `console_test` - Intermediate complexity, testing individual methods with embedded test data
- `samples_functional_testing` - Full workflow tests simulating real image processing tasks

Refer to `.github/workflows/` for CI test configurations and current testing priorities.

### GPU Development

The project includes CUDA code for GPU acceleration. GPU-related files are primarily in:

- Core extensions for GPU operations
- Specialized GPU kernels for image processing
- CUDA FFT implementations

### Code Structure Notes

- Most core functionality is in header-only or heavily templated C++ code
- Image processing uses custom Image class with MRC file format support
- Database schema is defined for project management
- Extensive use of wxWidgets for cross-platform GUI components
- Legacy features mean style isn't fully coherent, but the project aims to unify as code is modified

## Code Style and Standards

- **Formatting:** Project uses `.clang-format` in the root directory for consistent code formatting
- **wxWidgets Printf Formatting:** Be careful with format specifiers in wxPrintf and debug assertions. Common issue: `long` vs `int` mismatches can cause segfaults in wxFormatConverterBase. Always match format specifiers exactly to variable types (e.g., `%ld` for `long`, `%d` for `int`, `%f` for `float`)
- **Temporary Debugging Changes:** All temporary debugging code (debug prints, commented-out code, test modifications) must be marked with `// revert - <description of change and reason>` to ensure cleanup before commits. Search for "revert" to find all temporary changes.
- **Philosophy:** Incremental modernization - update and unify style as code is modified rather than wholesale changes
- **Legacy Compatibility:** Many legacy features exist; maintain compatibility while gradually improving
- **Preprocessor Defines:** All project-specific preprocessor defines should be prefixed with `cisTEM_` to avoid naming collisions (e.g., `cisTEM_ENABLE_FEATURE` not `ENABLE_FEATURE`)
- **Include Guards:** Use the full path from project root in uppercase with underscores for header file include guards (e.g., `_SRC_GUI_MYHEADER_H_` for `src/gui/MyHeader.h`, not `__MyHeader__`)
- **Temporary Files:** All temporary files (scripts, plans, documentation drafts) should be created in `.claude/cache/` directory. Create this directory if it doesn't exist. This keeps the project root clean and makes it easy to identify Claude-generated temporary content

## wxWidgets Best Practices for cisTEM

### Memory Management
- **Widget Ownership:** Create widgets with clear parent-child relationships. Parent widgets automatically delete their children.
- **Avoid Complex Member Widgets:** Don't use `std::unique_ptr` or member variables for widgets that may outlive workflow switches. Instead, create them locally in dialogs.
- **Dialog-Scoped Resources:** For temporary UI elements (like queue managers), create them as children of dialogs rather than panel members.

Example:
```cpp
// GOOD: Dialog owns the widget
wxDialog* dialog = new wxDialog(parent, ...);
MyWidget* widget = new MyWidget(dialog);  // Dialog will delete it

// AVOID: Complex lifecycle management
class Panel {
    std::unique_ptr<MyWidget> persistent_widget;  // Risky during workflow switches
};
```

### Database Access Patterns
- **Defer Database Operations:** Never access the database in constructors, especially during workflow switching when `main_frame` might be invalid.
- **Use Lazy Loading:** Implement a flag-based approach for database operations.

Example:
```cpp
// GOOD: Lazy loading pattern
class MyWidget {
    bool needs_database_load = true;

    void OnFirstUse() {
        if (needs_database_load && main_frame && main_frame->current_project.is_open) {
            LoadFromDatabase();
            needs_database_load = false;
        }
    }
};
```

### Workflow Switching Robustness
- **Design for Destruction:** Panels are destroyed and recreated during workflow switches. Don't assume persistence.
- **State in Database:** Keep complex state in the database rather than in memory.
- **Avoid Destructor Logic:** Don't put complex logic in destructors; wxWidgets handles most cleanup automatically.

### Build System Tips
- **Parallel Builds:** Use `make -j16` (or available thread count) for faster compilation
- **Force Rebuilds:** Delete dependency files to force rebuild: `rm gui/.deps/cisTEM-TargetFile.*`
- **VS Code Quirks:** Git diffs may need manual refresh in VS Code after file changes

## Environment Variables

- `WX_CONFIG` - Path to wx-config for specifying wxWidgets installation
- CUDA environment variables for GPU builds
- Various build flags configured in `.vscode/tasks.json`

## IDE Configuration

The project is designed for development with Visual Studio Code using Docker containers:

- VS Code settings linked via `.vscode` symlink to `.vscode_shared/CistemDev`
- Container environment managed through `regenerate_containers.sh`
- Build tasks pre-configured for different compiler and configuration combinations

## Template Matching Queue Development Patterns

### Static Members for Cross-Dialog Persistence
When implementing features that need to persist across dialog instances (like queues), use static members rather than complex lifecycle management:
```cpp
// In header
static std::deque<QueueItem> execution_queue;
static long currently_running_id;

// In cpp file - define static members
std::deque<QueueItem> QueueManager::execution_queue;
long QueueManager::currently_running_id = -1;
```

### Job Completion Tracking Pattern
For async job tracking in panels, store the job ID when starting and check it in completion callbacks:
```cpp
// In job panel header
long running_queue_job_id;  // -1 if not from queue

// In job start
running_queue_job_id = job.template_match_id;

// In ProcessAllJobsFinished or similar
if (running_queue_job_id > 0) {
    UpdateQueueStatus(running_queue_job_id, "complete");
    running_queue_job_id = -1;
}
```

This pattern allows proper status updates without tight coupling between components.
