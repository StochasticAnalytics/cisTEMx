# GitHub Workflows for cisTEMx

This directory contains GitHub Actions workflows that implement the CI/CD pipeline for the cisTEMx project.

## Overview

The cisTEMx CI/CD system uses a **hierarchical trigger architecture** where all builds start from a single entry point (`check_formatting.yml`) that enforces code formatting standards before triggering parallel build configurations. This design ensures consistent code style while optimizing resource usage through strategic runner selection.

### Workflow Trigger Flow

```
check_formatting.yml (Entry Point - on push/PR)
    ├── Checks C++ formatting with clang-format
    └── If formatting passes, triggers in parallel:
        ├── debug_build.yml (GPU debug)
        ├── debug_build_cpu_only.yml (CPU debug)
        ├── release_build_cpu_only.yml (CPU release)
        ├── release_build.yml (GPU release)
        ├── release_build_GNU_mkl.yml (GPU release with GNU/MKL)
        └── release_build_clang_noFastFFT.yml (GPU release without FastFFT)
```

### Runner Strategy & Cost Optimization

The workflow system uses three types of runners strategically to minimize costs:

1. **`ubuntu-latest` (GitHub free tier)**: Used for CPU-only builds and lightweight tasks
   - CPU builds are small enough to fit within free tier constraints
   - All container management and maintenance workflows use this

2. **`cpu_2core_runner` (Paid runner)**: Required for GPU builds
   - GPU builds are too large for free tier runners
   - Compiles CUDA code and creates artifacts for GPU testing

3. **`gpu_runner` (GPU-enabled runner)**: Used only for GPU test execution
   - Minimizes expensive GPU time by only running tests, not compilation
   - Receives pre-built artifacts from CPU runners

## CI/CD Build Workflows

### Entry Point & Orchestration

#### `check_formatting.yml`

- **Purpose**: Entry point for all CI/CD builds
- **Triggers**: On push to main, branches ending with `_with_ci`, or any PR
- **Runner**: `ubuntu-latest` (free tier)
- **Features**:
  - Enforces C++ code formatting using clang-format
  - Uses concurrency control to cancel in-progress runs on same branch
  - Triggers all build configurations in parallel if formatting passes
  - Acts as a quality gate before resource-intensive builds

### Reusable Build Template

#### `run_builds.yml`

- **Purpose**: Parameterized workflow template for all build configurations
- **Container**: `ghcr.io/stochasticanalytics/cistem_build_env:v3.0.3`
- **Parameters**:
  - `run_tests`: `'cpu'` or `'gpu'` - Determines test execution strategy
  - `compiler_cc` / `compiler_cxx`: Compiler selection (clang/gcc)
  - `conf_options`: Configure script options
  - `runner`: Which runner type to use
  - `make_verbose`: Build verbosity
- **Jobs**:
  1. **`build`**: Compiles code and optionally runs CPU tests
  2. **`run_gpu_tests`**: Downloads artifacts and runs tests on GPU (only if `run_tests: 'gpu'`)

### Build Configurations

All build configurations use the `run_builds.yml` template with different parameters:

#### GPU Builds (Paid Runners)

These builds require `cpu_2core_runner` due to size constraints:

1. **`debug_build.yml`** - GPU Debug Build
   - Compiler: clang/clang++
   - Options: `--enable-gpu-debug --enable-debugmode --with-cuda`
   - Tests: GPU (via artifacts)
   - Use case: Debugging GPU-specific issues with full symbols

2. **`release_build.yml`** - GPU Release Build
   - Compiler: clang/clang++
   - Options: `--with-cuda`
   - Tests: GPU (via artifacts)
   - Use case: Standard GPU production build

3. **`release_build_GNU_mkl.yml`** - GPU Release with GNU/MKL
   - Compiler: gcc/g++
   - Options: `--with-cuda` + Intel MKL paths
   - Tests: GPU (via artifacts)
   - Use case: Testing GNU compiler compatibility and MKL optimization

4. **`release_build_clang_noFastFFT.yml`** - GPU Release without FastFFT
   - Compiler: clang/clang++
   - Options: `--with-cuda --disable-FastFFT`
   - Tests: CPU only (despite GPU build)
   - Use case: Testing compatibility without FastFFT acceleration

#### CPU Builds (Free Tier)

These builds use `ubuntu-latest` runners as they fit within free tier limits:

5. **`debug_build_cpu_only.yml`** - CPU Debug Build
   - Compiler: clang/clang++
   - Options: `--enable-debugmode`
   - Tests: CPU
   - Use case: Debugging CPU-only functionality

6. **`release_build_cpu_only.yml`** - CPU Release Build
   - Compiler: clang/clang++
   - Options: Standard (no special flags)
   - Tests: CPU
   - Use case: Standard CPU-only production build

### Artifact-Based GPU Testing

GPU builds use a cost-optimized testing strategy:

1. **Build Phase** (on `cpu_2core_runner`):
   - Compiles code in container environment
   - Uploads test executables as artifacts:
     - `console_test` - Core functionality tests
     - `samples_functional_testing` - Workflow tests
     - `unit_test_runner` - Unit test suite

2. **Test Phase** (on `gpu_runner`):
   - Downloads artifacts from build phase
   - Executes tests on actual GPU hardware
   - Reports results back to workflow

This pattern minimizes expensive GPU runner usage by only using it for test execution, not compilation.

## Container Management Workflows

### `build_base_container.yml`

- **Purpose**: Builds base container with OS-level dependencies
- **Trigger**: Manual dispatch only
- **Version Source**: `.vscode/CONTAINER_VERSION_BASE`
- **Runner**: `ubuntu-latest`
- **Output**: Pushes to GitHub Container Registry (GHCR)

### `build_top_container.yml`

- **Purpose**: Builds application container with wxWidgets, libtorch, etc.
- **Trigger**: Manual dispatch only
- **Version Source**: `.vscode/CONTAINER_VERSION_TOP`
- **Runner**: `ubuntu-latest`
- **Features**:
  - Highly configurable via dispatch inputs
  - Options for wxWidgets version, npm, Claude support, libtorch, documentation tools
  - Builds on top of base container

### `push_container_to_ghcr.yml` (DEPRECATED)

- **Status**: ⚠️ DEPRECATED - Use `build_base_container.yml` or `build_top_container.yml` instead
- **Purpose**: Legacy workflow for pushing containers
- **Trigger**: Manual dispatch only

## Validation Workflows

### `validate_sync.yml`

- **Purpose**: Ensures hardcoded values match their source of truth
- **Trigger**: On every push and PR
- **Runner**: `ubuntu-latest`
- **Validates**:
  - Container version in `run_builds.yml` matches `.vscode/CONTAINER_VERSION_TOP`
  - Other hardcoded values across the repository
- **Script**: `.github/scripts/validate_sync.py`
- **Important**: When updating container versions, change both locations

## Maintenance Workflows

### `cleanup_actions.yml`

- **Purpose**: Delete workflow runs older than specified days
- **Trigger**: Manual dispatch only
- **Runner**: `ubuntu-latest`
- **Parameters**:
  - `days_to_keep`: Number of days to retain (default: 30)
  - Supports fractional days (e.g., 0.5 for 12 hours)
- **Usage**:
  1. Go to "Actions" tab in GitHub
  2. Select "Cleanup Old Actions" workflow
  3. Click "Run workflow"
  4. Enter days to keep
  5. Click "Run workflow"

## Test Suites

All workflows that run tests execute three test suites:

1. **Console Test** (`./console_test`)
   - Tests core cisTEM functionality
   - Validates basic operations and algorithms

2. **Samples Functional Testing** (`./samples_functional_testing`)
   - Tests complete processing workflows
   - Validates end-to-end functionality

3. **Unit Test Runner** (`./unit_test_runner`)
   - Runs unit test suite
   - Tests individual components in isolation

## Troubleshooting

### Common Issues

1. **Container Version Mismatch**
   - Error: `validate_sync.yml` fails
   - Solution: Update both `run_builds.yml` line 54 and `.vscode/CONTAINER_VERSION_TOP`

2. **Build Fails on Free Runner**
   - Error: Runner out of disk space
   - Solution: GPU builds must use `cpu_2core_runner`, not `ubuntu-latest`

3. **GPU Tests Not Running**
   - Check: Build workflow sets `run_tests: 'gpu'`
   - Check: Artifacts uploaded successfully from build job
   - Check: `gpu_runner` is available

4. **Formatting Check Blocks All Builds**
   - Run: `clang-format -i src/**/*.cpp src/**/*.h` locally
   - Commit formatted changes before pushing

### Adding New Build Configurations

1. Create new workflow file (e.g., `my_build.yml`)
2. Use `run_builds.yml` as template via `uses: ./.github/workflows/run_builds.yml`
3. Set appropriate parameters for your configuration
4. Add trigger job to `check_formatting.yml`:

   ```yaml
   build-my-configuration:
     needs: check-format
     uses: ./.github/workflows/my_build.yml
   ```

### Container Updates

When updating container versions:

1. Update version file (`.vscode/CONTAINER_VERSION_BASE` or `_TOP`)
2. Run appropriate container build workflow
3. Update hardcoded reference in `run_builds.yml` line 54
4. Verify with `validate_sync.yml`

## Best Practices

1. **Always test locally** before pushing to avoid wasting CI resources
2. **Use appropriate runners** - Don't use GPU runners for CPU-only tasks
3. **Monitor artifact size** - Large artifacts slow down GPU test startup
4. **Keep workflows DRY** - Use `run_builds.yml` template for new configurations
5. **Document changes** - Update this README when adding/modifying workflows

## Workflow File Reference

| File | Purpose | Runner | Trigger |
|------|---------|--------|---------|
| `check_formatting.yml` | Entry point, format check | `ubuntu-latest` | Push/PR |
| `run_builds.yml` | Reusable build template | Parameterized | Called by others |
| `debug_build.yml` | GPU debug build | `cpu_2core_runner` | Via check_formatting |
| `debug_build_cpu_only.yml` | CPU debug build | `ubuntu-latest` | Via check_formatting |
| `release_build.yml` | GPU release build | `cpu_2core_runner` | Via check_formatting |
| `release_build_cpu_only.yml` | CPU release build | `ubuntu-latest` | Via check_formatting |
| `release_build_GNU_mkl.yml` | GPU GNU/MKL build | `cpu_2core_runner` | Via check_formatting |
| `release_build_clang_noFastFFT.yml` | GPU no-FastFFT build | `cpu_2core_runner` | Via check_formatting |
| `build_base_container.yml` | Base container build | `ubuntu-latest` | Manual |
| `build_top_container.yml` | Top container build | `ubuntu-latest` | Manual |
| `push_container_to_ghcr.yml` | Legacy container push | `ubuntu-latest` | Manual (deprecated) |
| `validate_sync.yml` | Validation checks | `ubuntu-latest` | Push/PR |
| `cleanup_actions.yml` | Cleanup old runs | `ubuntu-latest` | Manual |
