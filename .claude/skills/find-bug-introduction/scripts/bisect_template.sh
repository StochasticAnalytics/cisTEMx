#!/bin/bash
# Basic git bisect test script template
#
# Usage: git bisect run ./bisect_template.sh
#
# Exit codes:
#   0   = Good (bug not present)
#   1   = Bad (bug present)
#   125 = Skip (can't test this commit)

set -e  # Exit on error (will be caught below)

# ==============================================================================
# Configuration
# ==============================================================================

# Build command (adjust for your project)
BUILD_CMD="make clean && make"

# Test command (adjust for your specific test)
TEST_CMD="./run_tests"

# Optional: Set timeouts
BUILD_TIMEOUT=300  # 5 minutes
TEST_TIMEOUT=60    # 1 minute

# ==============================================================================
# Helper Functions
# ==============================================================================

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

# ==============================================================================
# Main Bisect Logic
# ==============================================================================

COMMIT=$(git rev-parse --short HEAD)
log "Testing commit $COMMIT"

# Step 1: Try to build
log "Building..."
if ! timeout ${BUILD_TIMEOUT} bash -c "$BUILD_CMD" > /dev/null 2>&1; then
    log "Build failed - skipping commit $COMMIT"
    exit 125  # Skip this commit
fi

log "Build successful"

# Step 2: Run test
log "Running test..."
if timeout ${TEST_TIMEOUT} bash -c "$TEST_CMD" > /dev/null 2>&1; then
    log "Test PASSED - commit $COMMIT is GOOD"
    exit 0  # Bug not present
else
    log "Test FAILED - commit $COMMIT is BAD"
    exit 1  # Bug present
fi
