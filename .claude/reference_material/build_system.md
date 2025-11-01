# Build System Reference
## cisTEMx Build Configuration and Dependencies

### Overview
cisTEMx uses GNU Autotools as the primary build system with Intel MKL for optimized FFT operations.

### Quick Start Commands

```bash
# Initial setup (run once after cloning)
./regenerate_containers.sh
./regenerate_project.b

# Configure and build using VS Code
# Command Palette → Tasks: Run Task → BUILD cisTEMx DEBUG

# Or manually:
mkdir -p build/debug && cd build/debug
../../configure --enable-debugmode
make -j16
```

### Build Configurations

#### Debug Build
```bash
../../configure --enable-debugmode
```

#### Release Build
```bash
../../configure
```

#### With CUDA Support
```bash
../../configure --enable-cuda
```

### Key Dependencies

| Dependency | Purpose | Version Notes |
|------------|---------|---------------|
| **Intel MKL** | Primary FFT library for optimized performance | Required |
| **wxWidgets** | GUI framework | Typically 3.0.5 stable |
| **SQLite** | Database backend | Embedded |
| **CUDA** | GPU acceleration | Optional |
| **Intel C++ Compiler** | Primary compiler for performance builds | icc/icpc |

### VS Code Integration

The project includes pre-configured VS Code build tasks:
- Build tasks for different compiler/configuration combinations
- Symlinked `.vscode` directory with project settings
- Docker container development environment

### Build Output Locations

- Debug builds: `build/debug/`
- Release builds: `build/release/`
- Executables: `build/<config>/src/programs/`
- Libraries: `build/<config>/src/core/`

### Common Build Issues

#### MKL Not Found
Ensure Intel MKL is properly installed and environment variables are set:
```bash
source /opt/intel/mkl/bin/mklvars.sh intel64
```

#### wxWidgets Configuration
The build system expects wxWidgets 3.0.5. Check with:
```bash
wx-config --version
```

### Related Documentation
- See `scripts/CLAUDE.md` for detailed build script documentation
- Check `.github/workflows/` for CI build configurations