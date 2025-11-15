#!/bin/bash

#########################################################################################################################################################################################
### PURPOSE:
#####################
# Build Apptainer/Singularity .sif container directly from Dockerfile
# This enables creation of distribution images with compiled binaries, test data, and aggressive cleanup
# without requiring intermediate Docker registry push
#########################################################################################################################################################################################

#########################################################################################################################################################################################
### USAGE:
#####################
# This script must be run on a system with Apptainer/Singularity installed
# It will build a .sif file directly from a Dockerfile
#
# Examples:
#   ./build_apptainer.sh --dockerfile distribution_image/Dockerfile
#   ./build_apptainer.sh --dockerfile top_image/Dockerfile --output custom.sif
#   ./build_apptainer.sh --dockerfile ../my_build/Dockerfile --output /scratch/$USER/container.sif
#########################################################################################################################################################################################

set -e  # Exit on error

# Source shared validation library
source "$(dirname "$0")/validate_apptainer_args.sh"

# Default values
DOCKERFILE=""
OUTPUT_FILE=""
FORCE_OVERWRITE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      echo ""
      echo "Usage: build_apptainer.sh [OPTIONS]"
      echo ""
      echo "Build Apptainer/Singularity .sif container directly from Dockerfile"
      echo ""
      echo "Options:"
      echo "  -h, --help                   Show this help message"
      echo "  --dockerfile PATH            Path to Dockerfile (required)"
      echo "  --output FILE                Output .sif file name (default: cistem_<dirname>.sif)"
      echo "  --force                      Overwrite existing .sif file"
      echo ""
      echo "Examples:"
      echo "  # Build from distribution Dockerfile"
      echo "  ./build_apptainer.sh --dockerfile distribution_image/Dockerfile"
      echo ""
      echo "  # Specify custom output location"
      echo "  ./build_apptainer.sh --dockerfile top_image/Dockerfile --output custom.sif"
      echo ""
      echo "  # Write to HPC scratch space"
      echo "  ./build_apptainer.sh --dockerfile ../my_build/Dockerfile --output /scratch/\$USER/container.sif"
      echo ""
      echo "  # Force overwrite existing file"
      echo "  ./build_apptainer.sh --dockerfile distribution_image/Dockerfile --force"
      echo ""
      exit 0
      ;;
    --dockerfile)
      DOCKERFILE="$2"
      shift 2
      ;;
    --output)
      OUTPUT_FILE="$2"
      shift 2
      ;;
    --force)
      FORCE_OVERWRITE=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Try --help for usage information"
      exit 1
      ;;
  esac
done

# Validate required arguments
if [[ -z "$DOCKERFILE" ]]; then
    echo "Error: --dockerfile is required"
    echo "Try --help for usage information"
    exit 1
fi

# Validate Dockerfile exists and is readable
if [[ ! -f "$DOCKERFILE" ]]; then
    echo "Error: Dockerfile not found: $DOCKERFILE"
    exit 1
fi

if [[ ! -r "$DOCKERFILE" ]]; then
    echo "Error: Dockerfile is not readable: $DOCKERFILE"
    exit 1
fi

# Validate OUTPUT_FILE if specified
if [[ -n "$OUTPUT_FILE" ]]; then
    validate_output_file "$OUTPUT_FILE" || exit 1
fi

# Check if apptainer or singularity is installed
check_container_command || exit 1

# Set default output filename if not specified
if [[ -z "$OUTPUT_FILE" ]]; then
    # Extract a clean name from the Dockerfile path
    # If Dockerfile is in a directory, use directory name
    # Otherwise use "build"
    DOCKERFILE_DIR=$(dirname "$DOCKERFILE")
    DOCKERFILE_DIR_BASENAME=$(basename "$DOCKERFILE_DIR")

    if [[ "$DOCKERFILE_DIR" == "." ]]; then
        OUTPUT_FILE="cistem_build.sif"
    else
        OUTPUT_FILE="cistem_${DOCKERFILE_DIR_BASENAME}.sif"
    fi
fi

# Validate output file (now that we have a default)
validate_output_file "$OUTPUT_FILE" || exit 1

# Check if output file exists
if [[ -f "$OUTPUT_FILE" && "$FORCE_OVERWRITE" == "false" ]]; then
    echo "Error: Output file '$OUTPUT_FILE' already exists"
    echo "Use --force to overwrite or specify a different --output filename"
    exit 1
fi

# Display build information
echo ""
echo "==================================================================="
echo "Building Apptainer/Singularity container from Dockerfile"
echo "==================================================================="
echo "Dockerfile:  $DOCKERFILE"
echo "Output:      $OUTPUT_FILE"
echo "Command:     $CONTAINER_CMD"
echo "==================================================================="
echo ""

# Warn about potential size
echo "Note: This may take several minutes and produce a large file (several GB)"
echo "The .sif file will contain all layers from the Dockerfile build"
echo ""

# Check available disk space (warn if less than 10GB)
check_disk_space 10 || exit 1

# Perform the build
# The 'build' command creates a SIF file from a Dockerfile
echo "Running: $CONTAINER_CMD build $OUTPUT_FILE $DOCKERFILE"
echo ""

# Disable set -e temporarily to capture build exit code
set +e
$CONTAINER_CMD build "$OUTPUT_FILE" "$DOCKERFILE"
BUILD_EXIT_CODE=$?
set -e

# Check if successful
if [[ $BUILD_EXIT_CODE -eq 0 ]]; then
    echo ""
    echo "==================================================================="
    echo "SUCCESS: Container built successfully"
    echo "==================================================================="
    echo "Output file: $OUTPUT_FILE"
    ls -lh "$OUTPUT_FILE"
    echo ""
    echo "Usage examples:"
    echo "  # Run interactive shell"
    echo "  $CONTAINER_CMD shell $OUTPUT_FILE"
    echo ""
    echo "  # Execute a command"
    echo "  $CONTAINER_CMD exec $OUTPUT_FILE <command>"
    echo ""
    echo "  # Run a build"
    echo "  $CONTAINER_CMD exec --bind /path/to/cistem:/workspace $OUTPUT_FILE bash -c 'cd /workspace && ./configure && make'"
    echo ""
else
    echo ""
    echo "==================================================================="
    echo "ERROR: Container build failed (exit code: $BUILD_EXIT_CODE)"
    echo "==================================================================="
    echo "Please check the error messages above from $CONTAINER_CMD"
    echo ""
    echo "Common issues:"
    echo "  - Dockerfile contains errors or invalid syntax"
    echo "  - Base image not accessible or doesn't exist"
    echo "  - Network connectivity problems"
    echo "  - Insufficient disk space for build layers"
    echo "  - Build context missing required files (COPY/ADD sources)"
    echo ""
    exit 1
fi
