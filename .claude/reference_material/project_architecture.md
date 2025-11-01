# Project Architecture Reference
## cisTEMx Codebase Structure and Components

### Overview
cisTEMx is a scientific computing application for cryo-electron microscopy (cryo-EM) image processing and 3D reconstruction. It combines high-performance computing with an intuitive GUI.

### Directory Structure

```
cisTEMx/
├── src/
│   ├── core/           # Core libraries and data structures
│   ├── gui/            # wxWidgets-based graphical interface
│   ├── programs/       # Command-line executables
│   └── test/           # Unit tests
├── scripts/            # Build and utility scripts
├── .claude/            # Claude AI configuration and agents
├── .github/            # GitHub workflows and CI
├── build/              # Build output (not in repo)
│   ├── debug/
│   └── release/
└── test_data/          # Test datasets
```

### Core Components

#### src/core/
**Purpose**: Core functionality, data structures, and algorithms

**Key Classes**:
- `Image`: 2D/3D image data and operations
- `ReconstructedVolume`: 3D reconstruction data
- `Particle`: Particle picking and classification
- `SocketCommunicator`: Network communication
- `Database`: SQLite database interface

**Documentation**: `src/core/CLAUDE.md`

**Architecture Highlights**:
- Template-heavy for performance
- CUDA kernels for GPU acceleration
- FFT wrappers (MKL primary, FFTW fallback)
- Memory-mapped file I/O for large datasets

#### src/gui/
**Purpose**: wxWidgets-based graphical user interface

**Key Components**:
- `MainFrame`: Application main window
- `ProjectPanel`: Project management interface
- `ResultsPanel`: Display processing results
- Various dialog classes for user interaction

**Documentation**: `src/gui/CLAUDE.md`

**Architecture Highlights**:
- Parent-child widget hierarchy
- Event-driven architecture
- Model-View separation
- Custom drawing for scientific visualization

#### src/programs/
**Purpose**: Command-line programs for batch processing

**Categories**:
- **Processing**: refine3d, refine2d, reconstruct3d
- **Utilities**: console_test, samples_functional_testing
- **Analysis**: calculate_fsc, estimate_ctf
- **Data Management**: import_particles, merge_star

**Documentation**: `src/programs/CLAUDE.md`

**Architecture Highlights**:
- Shared core library usage
- MPI support for cluster execution
- Pipeline-friendly I/O
- Parameter file driven

#### scripts/
**Purpose**: Build, deployment, and utility scripts

**Key Scripts**:
- `regenerate_project.b`: Autotools regeneration
- `regenerate_containers.sh`: Docker container setup
- Linting and static analysis tools
- Deployment and packaging scripts

**Documentation**: `scripts/CLAUDE.md`

### Data Flow Architecture

```
Input Images → Preprocessing → Particle Picking
                                      ↓
                              Classification
                                      ↓
                              Alignment & CTF
                                      ↓
                              3D Reconstruction
                                      ↓
                              Refinement
                                      ↓
                              Final Volume
```

### Communication Architecture

#### GUI ↔ Processing
- Socket-based IPC for local processing
- Job queue management
- Real-time progress updates
- Result streaming

#### Cluster Execution
- MPI for distributed processing
- Master-worker pattern
- Shared filesystem for data
- SSH for remote job submission

### Memory Management Strategy

#### Image Data
- Large allocations with explicit new/delete
- Memory-mapped files for out-of-core processing
- GPU memory staging for CUDA operations
- Reference counting for shared data

#### GUI Components
- Parent-child ownership model
- Automatic cleanup through wxWidgets
- No manual memory management needed

#### Processing Pipeline
- Pool allocators for frequently allocated objects
- RAII wrappers for resources
- Smart pointers for complex ownership

### Thread Safety

#### Core Library
- Most classes are NOT thread-safe
- Explicit synchronization required
- Thread-local storage for temporary buffers

#### GUI
- Main thread only for wxWidgets calls
- Worker threads for processing
- Message passing for thread communication

### Build System Architecture

#### Configuration
- GNU Autotools for build configuration
- Support for multiple compilers (Intel, GCC)
- Optional features (CUDA, MPI, etc.)

#### Dependencies
- Intel MKL (required): FFT and linear algebra
- wxWidgets (required): GUI framework
- SQLite (embedded): Database
- CUDA (optional): GPU acceleration
- MPI (optional): Distributed processing

### Extension Points

#### Adding New Programs
1. Create in `src/programs/new_program/`
2. Add to `src/programs/Makefile.am`
3. Link against core library
4. Follow existing program patterns

#### Adding GUI Features
1. Extend existing panels or create new ones
2. Follow parent-child widget pattern
3. Use event binding for interactivity
4. Update `src/gui/Makefile.am`

#### Adding Core Functionality
1. Implement in `src/core/`
2. Provide both CPU and GPU versions if applicable
3. Add unit tests
4. Update relevant documentation

### Performance Considerations

#### Bottlenecks
- FFT operations → Use MKL/CUDA
- Memory bandwidth → Optimize access patterns
- I/O operations → Use memory mapping
- GUI responsiveness → Offload to worker threads

#### Optimization Strategy
1. Profile first (Intel VTune, nvprof)
2. Optimize algorithms before code
3. Use appropriate data structures
4. Leverage parallelism (OpenMP, CUDA)

### Related Documentation
- Component-specific CLAUDE.md files in each directory
- Build system: `build_system.md`
- Testing: `testing_practices.md`
- Code style: `code_style_standards.md`