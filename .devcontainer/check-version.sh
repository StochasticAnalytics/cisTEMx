#!/usr/bin/env bash
# Check if devcontainer image version matches CONTAINER_VERSION_TOP
# Fails by default - only passes on explicit version match

# EARLY MARKER: Write immediately to prove script was invoked
{
    echo "=========================================="
    echo "INVOKED: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "PWD: $(pwd)"
    echo "Script path: $0"
    echo "BASH_SOURCE: ${BASH_SOURCE[0]}"
    echo "=========================================="
} >> /tmp/devcontainer-check-invoked.log

set -e  # Exit on any error

# Setup logging - write to file directly at each echo
LOG_FILE="/tmp/devcontainer-version-check-$(date +%s).log"

# Helper function to log to both stdout and file
log() {
    #echo "$@"
    echo "$@" >> "${LOG_FILE}"
}

# Initialize log file
echo "Script started: $(date '+%Y-%m-%d %H:%M:%S')" > "${LOG_FILE}"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(dirname "$SCRIPT_DIR")"

log "=================================================="
log "DevContainer Version Check"
log "=================================================="
log "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
log "Log file: ${LOG_FILE}"
log ""
log "📂 Workspace root: ${WORKSPACE_ROOT}"

VERSION_FILE="${WORKSPACE_ROOT}/.vscode/CONTAINER_VERSION_TOP"
REPO_FILE="${WORKSPACE_ROOT}/.vscode/CONTAINER_REPO_NAME"
DEVCONTAINER_FILE="${WORKSPACE_ROOT}/.devcontainer/devcontainer.json"

log "📄 Checking version file: ${VERSION_FILE}"
log "📄 Checking repo file: ${REPO_FILE}"
log "📄 Checking devcontainer file: ${DEVCONTAINER_FILE}"

# Verify required files exist
if [ ! -f "$VERSION_FILE" ]; then
    log ""
    log "❌ ERROR: CONTAINER_VERSION_TOP file not found at: $VERSION_FILE"
    log "   This file is required to validate container version."
    exit 1
fi

if [ ! -f "$REPO_FILE" ]; then
    log ""
    log "❌ ERROR: CONTAINER_REPO_NAME file not found at: $REPO_FILE"
    log "   This file is required to validate container repository."
    exit 1
fi

if [ ! -f "$DEVCONTAINER_FILE" ]; then
    log ""
    log "❌ ERROR: devcontainer.json not found at: $DEVCONTAINER_FILE"
    exit 1
fi

# Read expected values from config files
EXPECTED_VERSION=$(cat "$VERSION_FILE" | tr -d '[:space:]')
EXPECTED_REPO=$(cat "$REPO_FILE" | tr -d '[:space:]')

if [ -z "$EXPECTED_VERSION" ]; then
    log ""
    log "❌ ERROR: CONTAINER_VERSION_TOP file is empty"
    log "   The version file must contain a valid version number."
    exit 1
fi

if [ -z "$EXPECTED_REPO" ]; then
    log ""
    log "❌ ERROR: CONTAINER_REPO_NAME file is empty"
    log "   The repo file must contain a valid repository path."
    exit 1
fi

# Construct expected image string
EXPECTED_IMAGE="${EXPECTED_REPO}:v${EXPECTED_VERSION}"

# Extract current image from devcontainer.json
CURRENT_IMAGE=$(grep -oP '"image":\s*"\K[^"]+' "$DEVCONTAINER_FILE")

log ""
log "📋 CONTAINER_REPO_NAME specifies: ${EXPECTED_REPO}"
log "📋 CONTAINER_VERSION_TOP specifies: v${EXPECTED_VERSION}"
log "📋 Expected image: ${EXPECTED_IMAGE}"
log ""
log "🔧 devcontainer.json image field: ${CURRENT_IMAGE}"

log ""
log "🔍 Comparing images..."
log "   Expected: ${EXPECTED_IMAGE}"
log "   Current:  ${CURRENT_IMAGE}"

# Compare images
if [ "$EXPECTED_IMAGE" != "$CURRENT_IMAGE" ]; then
    log ""
    log "❌ ERROR: Image mismatch detected"
    log "   devcontainer.json uses: ${CURRENT_IMAGE}"
    log "   Expected image: ${EXPECTED_IMAGE}"
    log ""
    log "   Action required:"
    log "   Update devcontainer.json image field to: ${EXPECTED_IMAGE}"
    exit 1
fi

# Success case - images match
log ""
log "✅ Container image check passed: ${EXPECTED_IMAGE}"
log "=================================================="
exit 0
