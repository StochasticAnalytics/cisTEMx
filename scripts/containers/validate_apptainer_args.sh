#!/bin/bash

#########################################################################################################################################################################################
### PURPOSE:
#####################
# Shared validation library for Apptainer/Singularity scripts
# Provides common validation functions used by both convert_to_apptainer.sh and build_apptainer.sh
#
# USAGE:
#   source "$(dirname "$0")/validate_apptainer_args.sh"
#   validate_output_file "$OUTPUT_FILE"
#   validate_version "$VERSION"
#   # etc.
#
# This file should be SOURCED, not executed directly
#########################################################################################################################################################################################

# Validate output file path for security and correctness
# Usage: validate_output_file "$OUTPUT_FILE"
# Returns: 0 on success, 1 on validation failure
validate_output_file() {
    local OUTPUT_FILE="$1"

    if [[ -z "$OUTPUT_FILE" ]]; then
        echo "Error: OUTPUT_FILE is empty"
        return 1
    fi

    # Check absolute paths - only allow whitelisted directories
    if [[ "$OUTPUT_FILE" = /* ]]; then
        # Whitelist for absolute paths (HPC scratch directories, temp directories, etc.)
        if [[ ! "$OUTPUT_FILE" =~ ^(/scratch/|/tmp/) ]]; then
            echo "Error: Absolute paths only allowed for whitelisted directories"
            echo "Allowed: /scratch/*, /tmp/*"
            echo "Your path: $OUTPUT_FILE"
            echo "Use a relative path or a whitelisted absolute path"
            return 1
        fi
    fi

    # Reject parent directory references
    if [[ "$OUTPUT_FILE" =~ \.\. ]]; then
        echo "Error: Parent directory references (..) not allowed in output path"
        echo "Output file must be in the current directory or subdirectory"
        return 1
    fi

    # Validate filename characters (alphanumeric, dash, underscore, dot, forward slash only)
    if [[ ! "$OUTPUT_FILE" =~ ^[a-zA-Z0-9._/-]+$ ]]; then
        echo "Error: Invalid characters in output filename"
        echo "Allowed characters: letters, numbers, dash, underscore, dot, forward slash"
        return 1
    fi

    # Enforce .sif extension
    if [[ ! "$OUTPUT_FILE" =~ \.sif$ ]]; then
        echo "Error: Output file must have .sif extension"
        echo "Example: my-container.sif or containers/dev-env.sif"
        return 1
    fi

    # Canonicalize path and verify it stays in current directory tree
    # Only apply this check to relative paths (absolute whitelisted paths are intentionally outside)
    if [[ "$OUTPUT_FILE" != /* ]]; then
        local OUTPUT_DIR=$(dirname "$OUTPUT_FILE")

        # Create directory if needed and get canonical path
        if [[ "$OUTPUT_DIR" != "." ]]; then
            mkdir -p "$OUTPUT_DIR" 2>/dev/null || {
                echo "Error: Cannot create output directory: $OUTPUT_DIR"
                return 1
            }
            local CANONICAL_DIR=$(cd "$OUTPUT_DIR" && pwd)
            local CURRENT_DIR=$(pwd)

            # Verify canonical path is within current directory
            if [[ "$CANONICAL_DIR" != "$CURRENT_DIR"* ]]; then
                echo "Error: Output path escapes current directory"
                echo "Canonical path: $CANONICAL_DIR"
                echo "Current dir: $CURRENT_DIR"
                return 1
            fi
        fi
    fi

    return 0
}

# Validate version string format
# Usage: validate_version "$VERSION"
# Returns: 0 on success, 1 on validation failure
validate_version() {
    local VERSION="$1"

    if [[ -z "$VERSION" ]]; then
        echo "Error: VERSION is empty"
        return 1
    fi

    # Validate semantic version format: X.Y.Z or X.Y.Z-suffix
    if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9]+)?$ ]]; then
        echo "Error: Invalid version format: $VERSION"
        echo "Expected format: X.Y.Z or X.Y.Z-suffix"
        echo "Examples: 3.0.3, 1.0.0-beta, 2.1.3-rc1"
        return 1
    fi

    return 0
}

# Validate Docker source URI format
# Usage: validate_docker_source "$DOCKER_SOURCE"
# Returns: 0 on success, 1 on validation failure
# Note: Warnings are printed to stderr but don't cause failure
validate_docker_source() {
    local DOCKER_SOURCE="$1"

    if [[ -z "$DOCKER_SOURCE" ]]; then
        echo "Error: DOCKER_SOURCE is empty"
        return 1
    fi

    # Validate docker:// URI format
    if [[ ! "$DOCKER_SOURCE" =~ ^docker://[a-zA-Z0-9._-]+/[a-zA-Z0-9._/-]+:[a-zA-Z0-9._-]+$ ]]; then
        echo "Error: Invalid Docker source format: $DOCKER_SOURCE"
        echo "Expected format: docker://registry/repo:tag"
        echo "Examples:"
        echo "  docker://ghcr.io/stochasticanalytics/cistemx:v3.0.3"
        echo "  docker://docker.io/library/ubuntu:22.04"
        return 1
    fi

    # Additional check: Verify registry is trusted (warning only, not a failure)
    if [[ ! "$DOCKER_SOURCE" =~ ^docker://(ghcr\.io|docker\.io)/ ]]; then
        echo "Warning: Source uses non-standard registry" >&2
        echo "Trusted registries: ghcr.io, docker.io" >&2
        echo "Proceeding with user-specified source: $DOCKER_SOURCE" >&2
    fi

    return 0
}

# Check available disk space and prompt if below threshold
# Usage: check_disk_space [min_gb]
# Default min_gb: 10
# Returns: 0 if sufficient or user confirms, 1 if user aborts
check_disk_space() {
    local MIN_GB="${1:-10}"  # Default 10GB

    local AVAILABLE_SPACE_KB=$(df -k . | awk 'NR==2 {print $4}')
    local AVAILABLE_SPACE_GB=$((AVAILABLE_SPACE_KB / 1024 / 1024))

    if [[ $AVAILABLE_SPACE_GB -lt $MIN_GB ]]; then
        echo "WARNING: Low disk space available"
        echo "Available: ${AVAILABLE_SPACE_GB} GB"
        echo "Recommended: At least ${MIN_GB} GB"
        echo ""
        read -p "Continue anyway? [y/N] " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted by user"
            return 1
        fi
    fi

    return 0
}

# Check if apptainer or singularity is installed
# Usage: check_container_command
# Sets global variable: CONTAINER_CMD (to "apptainer" or "singularity")
# Returns: 0 on success, 1 if neither found
check_container_command() {
    if command -v apptainer &> /dev/null; then
        CONTAINER_CMD="apptainer"
    elif command -v singularity &> /dev/null; then
        CONTAINER_CMD="singularity"
    else
        echo "Error: Neither 'apptainer' nor 'singularity' command found"
        echo "Please install Apptainer/Singularity to use this script"
        echo ""
        echo "Installation instructions:"
        echo "  Ubuntu/Debian: https://apptainer.org/docs/admin/main/installation.html#install-ubuntu-packages"
        echo "  RHEL/CentOS: https://apptainer.org/docs/admin/main/installation.html#install-rpm"
        echo "  From source: https://apptainer.org/docs/admin/main/installation.html#install-from-source"
        return 1
    fi

    echo "Using container command: $CONTAINER_CMD"
    return 0
}
