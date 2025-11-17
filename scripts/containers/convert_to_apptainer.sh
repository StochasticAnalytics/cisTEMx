#!/bin/bash

#########################################################################################################################################################################################
### PURPOSE:
#####################
# Convert cisTEMx Docker container to Apptainer/Singularity format (.sif file)
# This enables use of the development environment on HPC systems that use Apptainer/Singularity instead of Docker
#########################################################################################################################################################################################

#########################################################################################################################################################################################
### USAGE:
#####################
# This script must be run on a system with Apptainer/Singularity installed
# It will pull the Docker image and convert it to a .sif file
#
# Examples:
#   ./convert_to_apptainer.sh                                    # Use versions from .vscode config
#   ./convert_to_apptainer.sh --version 3.0.3                    # Specify version explicitly
#   ./convert_to_apptainer.sh --source docker://custom/repo:tag  # Specify custom source
#   ./convert_to_apptainer.sh --output my-container.sif          # Custom output file name
#########################################################################################################################################################################################

set -e  # Exit on error

# Source shared validation library
source "$(dirname "$0")/validate_apptainer_args.sh"

usr_path="../../.vscode"

# Default values
VERSION=""
REGISTRY="ghcr.io"  # Default to GitHub Container Registry (used in CI)
OUTPUT_FILE=""
DOCKER_SOURCE=""
FORCE_OVERWRITE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      echo ""
      echo "Usage: convert_to_apptainer.sh [OPTIONS]"
      echo ""
      echo "Convert cisTEMx Docker container to Apptainer/Singularity .sif format"
      echo ""
      echo "Options:"
      echo "  -h, --help                   Show this help message"
      echo "  --version VERSION            Specify container version (default: from .vscode/CONTAINER_VERSION_TOP)"
      echo "  --source SOURCE              Specify full Docker source URI (e.g., docker://repo:tag)"
      echo "  --registry REGISTRY          Docker registry (default: ghcr.io)"
      echo "                               Options: ghcr.io, docker.io"
      echo "  --output FILE                Output .sif file name (default: cistem_build_env_vVERSION.sif)"
      echo "  --force                      Overwrite existing .sif file"
      echo ""
      echo "Examples:"
      echo "  # Use default configuration from .vscode (pulls from ghcr.io)"
      echo "  ./convert_to_apptainer.sh"
      echo ""
      echo "  # Specify version explicitly"
      echo "  ./convert_to_apptainer.sh --version 3.0.3"
      echo ""
      echo "  # Use Docker Hub instead of GHCR"
      echo "  ./convert_to_apptainer.sh --registry docker.io"
      echo ""
      echo "  # Custom output file"
      echo "  ./convert_to_apptainer.sh --output my-dev-env.sif"
      echo ""
      echo "  # Force overwrite existing file"
      echo "  ./convert_to_apptainer.sh --force"
      echo ""
      exit 0
      ;;
    --version)
      VERSION="$2"
      validate_version "$VERSION" || exit 1
      shift 2
      ;;
    --source)
      DOCKER_SOURCE="$2"
      validate_docker_source "$DOCKER_SOURCE" || exit 1
      shift 2
      ;;
    --registry)
      REGISTRY="$2"
      if [[ "$REGISTRY" != "docker.io" && "$REGISTRY" != "ghcr.io" ]]; then
        echo "Error: Registry must be ghcr.io or docker.io"
        exit 1
      fi
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

# Validate OUTPUT_FILE if specified
if [[ -n "$OUTPUT_FILE" ]]; then
    validate_output_file "$OUTPUT_FILE" || exit 1
fi

# Check if apptainer or singularity is installed
check_container_command || exit 1

# If no custom source specified, build from repository and version
if [[ -z "$DOCKER_SOURCE" ]]; then
    # Get version from .vscode if not specified
    if [[ -z "$VERSION" ]]; then
        if [[ ! -f "${usr_path}/CONTAINER_VERSION_TOP" ]]; then
            echo "Error: CONTAINER_VERSION_TOP file not found at ${usr_path}/CONTAINER_VERSION_TOP"
            echo "Please run this script from scripts/containers/ or specify --version"
            exit 1
        fi
        VERSION=$(cat "${usr_path}/CONTAINER_VERSION_TOP")
    fi

    # Get repository name
    if [[ ! -f "${usr_path}/CONTAINER_REPO_NAME" ]]; then
        echo "Error: CONTAINER_REPO_NAME file not found at ${usr_path}/CONTAINER_REPO_NAME"
        echo "Please run this script from scripts/containers/ or specify --source"
        exit 1
    fi
    REPO_NAME=$(cat "${usr_path}/CONTAINER_REPO_NAME" | tr '[:upper:]' '[:lower:]')

    # Strip registry from REPO_NAME if present (it may already include registry/org/name)
    # Remove leading ghcr.io/ or docker.io/ if present
    REPO_NAME="${REPO_NAME#ghcr.io/}"
    REPO_NAME="${REPO_NAME#docker.io/}"

    # Build Docker source URI
    DOCKER_TAG="v${VERSION}"
    DOCKER_SOURCE="docker://${REGISTRY}/${REPO_NAME}:${DOCKER_TAG}"
fi

# Set default output filename if not specified
if [[ -z "$OUTPUT_FILE" ]]; then
    # Extract a clean name from the source
    if [[ -n "$VERSION" ]]; then
        OUTPUT_FILE="cistem_build_env_v${VERSION}.sif"
    else
        # Try to extract version from source
        OUTPUT_FILE="cistem_build_env.sif"
    fi
fi

# Check if output file exists
if [[ -f "$OUTPUT_FILE" && "$FORCE_OVERWRITE" == "false" ]]; then
    echo "Error: Output file '$OUTPUT_FILE' already exists"
    echo "Use --force to overwrite or specify a different --output filename"
    exit 1
fi

# Display build information
echo ""
echo "==================================================================="
echo "Converting Docker container to Apptainer/Singularity format"
echo "==================================================================="
echo "Source:      $DOCKER_SOURCE"
echo "Output:      $OUTPUT_FILE"
echo "Command:     $CONTAINER_CMD"
echo "==================================================================="
echo ""

# Warn about potential size
echo "Note: This may take several minutes and produce a large file (several GB)"
echo "The .sif file will contain all layers of the Docker container"
echo ""

# Check available disk space (warn if less than 10GB)
check_disk_space 10 || exit 1

# Perform the conversion
# The 'build' command creates a SIF file from a Docker URI
echo "Running: $CONTAINER_CMD build $OUTPUT_FILE $DOCKER_SOURCE"
echo ""

# Disable set -e temporarily to capture build exit code
set +e
$CONTAINER_CMD build "$OUTPUT_FILE" "$DOCKER_SOURCE"
BUILD_EXIT_CODE=$?
set -e

# Check if successful
if [[ $BUILD_EXIT_CODE -eq 0 ]]; then
    echo ""
    echo "==================================================================="
    echo "SUCCESS: Container converted successfully"
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
    echo "ERROR: Container conversion failed (exit code: $BUILD_EXIT_CODE)"
    echo "==================================================================="
    echo "Please check the error messages above from $CONTAINER_CMD"
    echo ""
    echo "Common issues:"
    echo "  - Docker source not accessible or doesn't exist"
    echo "  - Network connectivity problems"
    echo "  - Insufficient disk space for .sif file"
    echo "  - Invalid Docker URI format"
    echo ""
    exit 1
fi
