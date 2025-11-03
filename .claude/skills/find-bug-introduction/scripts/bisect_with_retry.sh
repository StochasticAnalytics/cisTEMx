#!/bin/bash
# Git bisect test script with retry logic for flaky tests
#
# Usage: git bisect run ./bisect_with_retry.sh
#
# This script handles flaky tests by running them multiple times
# and using majority voting to determine good/bad status.

set -e

# ==============================================================================
# Configuration
# ==============================================================================

BUILD_CMD="make clean && make"
TEST_CMD="./run_tests"

# Retry configuration
RETRY_COUNT=10           # Number of times to run test
FAILURE_THRESHOLD=6      # Failures needed to mark as bad (>50%)

BUILD_TIMEOUT=300
TEST_TIMEOUT=60

# Optional: Log file for debugging
LOG_FILE="${LOG_FILE:-bisect_test.log}"

# ==============================================================================
# Helper Functions
# ==============================================================================

log() {
    local msg="[$(date +'%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

# ==============================================================================
# Main Logic
# ==============================================================================

COMMIT=$(git rev-parse --short HEAD)
log "========================================="
log "Testing commit $COMMIT with retry logic"
log "========================================="

# Step 1: Build
log "Building..."
if ! timeout ${BUILD_TIMEOUT} bash -c "$BUILD_CMD" >> "$LOG_FILE" 2>&1; then
    log "Build failed - skipping commit $COMMIT"
    exit 125
fi

log "Build successful"

# Step 2: Run test multiple times
log "Running test $RETRY_COUNT times..."
failures=0
successes=0

for i in $(seq 1 $RETRY_COUNT); do
    log "  Run $i/$RETRY_COUNT..."

    if timeout ${TEST_TIMEOUT} bash -c "$TEST_CMD" >> "$LOG_FILE" 2>&1; then
        ((successes++))
        log "    ✓ Pass"
    else
        ((failures++))
        log "    ✗ Fail"
    fi
done

# Step 3: Determine result based on majority
log ""
log "Results: $successes passed, $failures failed"

if [ $failures -ge $FAILURE_THRESHOLD ]; then
    log "Majority FAILED ($failures/$RETRY_COUNT) - commit $COMMIT is BAD"
    exit 1
else
    log "Majority PASSED ($successes/$RETRY_COUNT) - commit $COMMIT is GOOD"
    exit 0
fi
