#!/bin/bash
#
# Local CI Emulation Script for cisTEMx
#
# This script emulates the GitHub Actions CI pipeline locally using Docker.
# It runs formatting checks followed by all 6 build configurations with tests.
#
# Usage:
#   ./scripts/local_ci.sh          Run full CI pipeline
#   ./scripts/local_ci.sh --cleanup Clean up old build directories
#   ./scripts/local_ci.sh --help    Show this help
#
# Features:
# - Fail-fast behavior (stops on first failure)
# - Full detailed logs + clean summary log
# - Uses same Docker container as GitHub Actions
# - Builds in /tmp/cistemx_ci for speed
# - Uses 8 cores for builds
#

set -e  # Exit on any error
set -o pipefail

# Parse command line arguments
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Local CI Emulation Script for cisTEMx"
    echo ""
    echo "Usage:"
    echo "  $0           Run full CI pipeline"
    echo "  $0 --cleanup  Clean up old build directories in /tmp/cistemx_ci"
    echo "  $0 --help     Show this help"
    echo ""
    echo "Environment variables:"
    echo "  GPU_DEVICE    GPU device to use (default: 0, can be 'all' or specific device number)"
    echo ""
    echo "Examples:"
    echo "  GPU_DEVICE=1 $0              # Use GPU 1"
    echo "  GPU_DEVICE=all $0            # Use all GPUs"
    echo ""
    exit 0
fi

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER_VERSION=$(cat "$REPO_ROOT/.vscode/CONTAINER_VERSION_TOP" | tr -d '\n')
DOCKER_IMAGE="ghcr.io/stochasticanalytics/cistem_build_env:v${CONTAINER_VERSION}"
BUILD_BASE="/tmp/cistemx_ci"
LOG_DIR="$REPO_ROOT/ci_logs"
N_THREADS=8
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# GPU configuration (can be overridden via environment variable)
# Examples: GPU_DEVICE=0 (default), GPU_DEVICE=1, GPU_DEVICE=all
GPU_DEVICE="${GPU_DEVICE:-0}"

# Create log directory
mkdir -p "$LOG_DIR"

# Initialize summary log
SUMMARY_LOG="$LOG_DIR/summary_${TIMESTAMP}.log"
echo "=====================================" > "$SUMMARY_LOG"
echo "cisTEMx Local CI Run - $TIMESTAMP" >> "$SUMMARY_LOG"
echo "=====================================" >> "$SUMMARY_LOG"
echo "" >> "$SUMMARY_LOG"

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
    echo "[INFO] $1" >> "$SUMMARY_LOG"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
    echo "[✓] $1" >> "$SUMMARY_LOG"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
    echo "[✗] $1" >> "$SUMMARY_LOG"
}

log_step() {
    echo ""
    echo -e "${YELLOW}==>${NC} $1"
    echo "" >> "$SUMMARY_LOG"
    echo "==> $1" >> "$SUMMARY_LOG"
}

# Safe remove function with sanity checks
safe_remove() {
    local target="$1"

    # Sanity checks to prevent accidental deletion of wrong directories
    if [ -z "$target" ]; then
        log_error "safe_remove: Empty path provided"
        return 1
    fi

    # Must be under /tmp/cistemx_ci or be a temp directory
    if [[ "$target" != /tmp/cistemx_ci* ]] && [[ "$target" != /tmp/tmp.* ]]; then
        log_error "safe_remove: Path '$target' is not in allowed cleanup locations"
        return 1
    fi

    # Prevent deleting root directories
    if [ "$target" == "/" ] || [ "$target" == "/tmp" ] || [ "$target" == "/tmp/" ]; then
        log_error "safe_remove: Refusing to delete root directory '$target'"
        return 1
    fi

    # Only remove if it exists
    if [ -e "$target" ]; then
        rm -rf "$target"
        return 0
    fi

    return 0
}

# Cleanup function for --cleanup option
cleanup_old_builds() {
    echo "Cleaning up old CI build directories..."

    if [ ! -d "$BUILD_BASE" ]; then
        echo "Nothing to clean - $BUILD_BASE does not exist"
        exit 0
    fi

    # Calculate total size before cleanup
    TOTAL_SIZE=$(du -sh "$BUILD_BASE" 2>/dev/null | cut -f1)
    echo "Current size of $BUILD_BASE: $TOTAL_SIZE"

    # List what will be removed
    echo ""
    echo "Directories to be removed:"
    find "$BUILD_BASE" -mindepth 1 -maxdepth 1 -type d -exec du -sh {} \; 2>/dev/null || echo "  (none)"

    echo ""
    read -p "Proceed with cleanup? [y/N] " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        safe_remove "$BUILD_BASE"
        echo "Cleanup complete!"
    else
        echo "Cleanup cancelled"
    fi

    exit 0
}

# Handle --cleanup option
if [ "$1" == "--cleanup" ]; then
    cleanup_old_builds
fi

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed or not in PATH"
    exit 1
fi

# Pull the Docker image
log_step "Pulling Docker image: $DOCKER_IMAGE"
if docker pull "$DOCKER_IMAGE" &> "$LOG_DIR/docker_pull.log"; then
    log_success "Docker image pulled successfully"
else
    log_error "Failed to pull Docker image"
    cat "$LOG_DIR/docker_pull.log"
    exit 1
fi

# Step 1: Formatting Check
log_step "Step 1: Formatting Check"

FORMAT_DIR="$BUILD_BASE/format_check"
FORMAT_LOG="$LOG_DIR/format_check_${TIMESTAMP}.log"

# Clean up any previous formatting check
safe_remove "$FORMAT_DIR"
mkdir -p "$FORMAT_DIR"

log_info "Cloning repository to $FORMAT_DIR"
{
    cd "$FORMAT_DIR"
    git clone --depth 1 "$REPO_ROOT" repo
    cd repo

    log_info "Running clang-format-14 check"

    # Run clang-format check in Docker using the same logic as pre-commit hook
    docker run --rm \
        -v "$FORMAT_DIR/repo:/workspace" \
        -w /workspace \
        "$DOCKER_IMAGE" \
        bash -c '
            set -e

            # Function to check if file should be excluded
            should_exclude_file() {
                local file="$1"

                # Exclude files in include/ directory (third-party headers)
                if [[ "$file" == include/* ]]; then
                    return 0
                fi

                # Exclude files in src/gui/wxformbuilder (input .fbp files)
                if [[ "$file" == src/gui/wxformbuilder/* ]]; then
                    return 0
                fi

                # Exclude icon files (auto-generated binary data)
                if [[ "$file" == src/gui/icons/* ]]; then
                    return 0
                fi

                # Exclude files with ProjectX_gui in the name (generated by wxFormBuilder)
                if [[ "$file" == *ProjectX_gui*.cpp ]] || [[ "$file" == *ProjectX_gui*.h ]]; then
                    return 0
                fi

                # Check file header for wxFormBuilder warning
                if [ -f "$file" ]; then
                    if head -n 10 "$file" | grep -q "PLEASE DO \*NOT\* EDIT THIS FILE\|DO NOT EDIT THIS FILE\|Generated by wxFormBuilder"; then
                        return 0
                    fi
                fi

                return 1
            }

            # Find all C++ and CUDA files
            FILES=$(find src -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.cc" -o -name "*.cxx" -o -name "*.hpp" -o -name "*.cu" -o -name "*.cuh" \))

            FORMAT_ISSUES=()
            TEMP_DIR=$(mktemp -d)

            # Check each file
            for file in $FILES; do
                # Skip excluded files
                if should_exclude_file "$file"; then
                    continue
                fi

                if [ -f "$file" ]; then
                    # Get the formatted version
                    clang-format-14 "$file" > "$TEMP_DIR/$(basename $file).formatted"

                    # Compare with original
                    if ! diff -q "$file" "$TEMP_DIR/$(basename $file).formatted" > /dev/null 2>&1; then
                        FORMAT_ISSUES+=("$file")
                    fi
                fi
            done

            # Clean up temp directory
            rm -rf "$TEMP_DIR"

            # Report results
            if [ ${#FORMAT_ISSUES[@]} -gt 0 ]; then
                echo ""
                echo "ERROR: The following files have formatting issues:"
                for file in "${FORMAT_ISSUES[@]}"; do
                    echo "  - $file"
                done
                echo ""
                echo "Fix with: clang-format-14 -i <file>"
                exit 1
            fi

            echo "All C++ and CUDA files are properly formatted."
            exit 0
        '
} &> "$FORMAT_LOG"

if [ $? -eq 0 ]; then
    log_success "Formatting check passed"
else
    log_error "Formatting check failed"
    echo ""
    echo "See full log: $FORMAT_LOG"
    tail -n 50 "$FORMAT_LOG"
    exit 1
fi

# Clean up formatting check directory
safe_remove "$FORMAT_DIR"

# Step 2: Build all configurations
log_step "Step 2: Building all configurations"

# Define build configurations
# Format: "build_type|CC|CXX|configure_options|run_tests_on"
declare -a BUILD_CONFIGS=(
    "GPU_release|clang|clang++|--with-cuda --enable-openmp --enable-experimental --enable-staticmode --disable-multiple-global-refinements --with-wx-config=/opt/WX/wx305-clang-static-gtk2/bin/wx-config|gpu"
    "GPU_release_GNU_MKL|gcc|g++|--with-cuda --disable-FastFFT --enable-openmp --enable-experimental --enable-staticmode --with-wx-config=/opt/WX/wx305-gcc-static-gtk2/bin/wx-config --disable-multiple-global-refinements|gpu"
    "GPU_release_no_FastFFT|clang|clang++|--with-cuda --disable-FastFFT --enable-openmp --enable-experimental --enable-staticmode --with-wx-config=/opt/WX/wx305-clang-static-gtk2/bin/wx-config --disable-multiple-global-refinements|cpu"
    "GPU_debug|clang|clang++|--enable-gpu-debug --enable-debugmode --with-cuda --disable-FastFFT --enable-openmp --enable-experimental --enable-staticmode --disable-multiple-global-refinements --with-wx-config=/opt/WX/wx305-clang-static-gtk2/bin/wx-config|gpu"
    "cpu_release|clang|clang++|--disable-FastFFT --enable-openmp --enable-experimental --enable-staticmode --disable-multiple-global-refinements --with-wx-config=/opt/WX/wx305-clang-static-gtk2/bin/wx-config|cpu"
    "cpu_debug|clang|clang++|--enable-debugmode --disable-FastFFT --enable-openmp --enable-experimental --enable-staticmode --disable-multiple-global-refinements --with-wx-config=/opt/WX/wx305-clang-static-gtk2/bin/wx-config|cpu"
)

TOTAL_BUILDS=${#BUILD_CONFIGS[@]}
CURRENT_BUILD=0

for config in "${BUILD_CONFIGS[@]}"; do
    CURRENT_BUILD=$((CURRENT_BUILD + 1))

    # Parse configuration
    IFS='|' read -r BUILD_TYPE CC CXX CONFIGURE_OPTS RUN_TESTS <<< "$config"

    log_step "Build $CURRENT_BUILD/$TOTAL_BUILDS: $BUILD_TYPE"

    BUILD_DIR="$BUILD_BASE/$BUILD_TYPE"
    FULL_LOG="$LOG_DIR/${BUILD_TYPE}_full_${TIMESTAMP}.log"

    # Clean up any previous build
    safe_remove "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"

    START_TIME=$(date +%s)

    {
        echo "======================================"
        echo "Building: $BUILD_TYPE"
        echo "CC: $CC"
        echo "CXX: $CXX"
        echo "Configure options: $CONFIGURE_OPTS"
        echo "======================================"
        echo ""

        log_info "Cloning repository to $BUILD_DIR"
        cd "$BUILD_DIR"
        git clone --depth 1 "$REPO_ROOT" repo
        cd repo

        log_info "Running build in Docker container"

        # Run the full build and test process in Docker
        # Configure GPU access based on GPU_DEVICE setting
        if [ "$GPU_DEVICE" == "all" ]; then
            GPU_ARG="--gpus all"
        else
            GPU_ARG="--gpus \"device=$GPU_DEVICE\""
        fi

        docker run --rm \
            --user $(id -u):$(id -g) \
            $GPU_ARG \
            -v "$BUILD_DIR/repo:/workspace" \
            -w /workspace \
            -e CC="$CC" \
            -e CXX="$CXX" \
            -e PATH="/usr/bin:\$PATH" \
            "$DOCKER_IMAGE" \
            bash -c "
                set -e

                echo '=== Running regenerate_project.sh ==='
                ./regenerate_project.sh

                echo ''
                echo '=== Creating build directory ==='
                mkdir -p build/${BUILD_TYPE}
                cd build/${BUILD_TYPE}

                echo ''
                echo '=== Running configure ==='
                ../../configure $CONFIGURE_OPTS

                echo ''
                echo '=== Building with make -j ${N_THREADS} ==='
                mkdir -p tmp
                export TMPDIR=\$(pwd)/tmp
                make -j ${N_THREADS}

                echo ''
                echo '=== Preparing test binaries ==='
                mkdir -p artifacts
                mv src/unit_test_runner src/samples_functional_testing src/console_test artifacts/
                chmod +x artifacts/*

                echo ''
                echo '=== Cleaning up build artifacts ==='
                rm -rf src/core src/gui src/programs tmp

                echo ''
                echo '=== Running tests ==='
                cd artifacts

                echo 'Running console_test...'
                ./console_test

                echo ''
                echo 'Running samples_functional_testing...'
                ./samples_functional_testing

                echo ''
                echo 'Running unit_test_runner...'
                ./unit_test_runner

                echo ''
                echo '=== All tests passed for ${BUILD_TYPE} ==='
            "
    } &> "$FULL_LOG"

    BUILD_EXIT_CODE=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    if [ $BUILD_EXIT_CODE -eq 0 ]; then
        log_success "$BUILD_TYPE: Build and tests passed (${DURATION}s)"
        # Clean up successful build
        safe_remove "$BUILD_DIR"
    else
        log_error "$BUILD_TYPE: Build or tests failed (${DURATION}s)"
        echo ""
        echo "Full log: $FULL_LOG"
        echo ""
        echo "Last 100 lines of output:"
        tail -n 100 "$FULL_LOG"
        exit 1
    fi
done

# Final summary
log_step "CI Run Complete"
log_success "All formatting checks, builds, and tests passed!"
echo ""
echo "Summary log: $SUMMARY_LOG"
echo "Detailed logs: $LOG_DIR/*_${TIMESTAMP}.log"
echo ""

exit 0
