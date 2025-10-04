#!/usr/bin/env bash
# Run Tier 0 (Blocker) static analysis - must-fix issues before commit
# Usage: ./analyze_blocker.sh [path]
# Default path: src/core/tensor/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build/clang-tidy-debug"
TARGET_PATH="${1:-src/core/tensor/}"

# Tier 0 checks: Critical bugs that must be fixed
TIER0_CHECKS="bugprone-use-after-move,bugprone-dangling-handle,bugprone-undelegated-constructor,performance-move-const-arg,cert-err52-cpp"

cd "$PROJECT_ROOT"

if [[ ! -f "$BUILD_DIR/compile_commands.json" ]]; then
    echo "Error: Compilation database not found at $BUILD_DIR/compile_commands.json"
    echo "Run: Tasks → 'Build with Bear (Generate compile_commands.json)'"
    exit 1
fi

echo "=== Tier 0 (Blocker) Static Analysis ==="
echo "Target: $TARGET_PATH"
echo "Checks: $TIER0_CHECKS"
echo ""

# Find C++ files (not .cu, those need special handling)
find "$TARGET_PATH" \( -name '*.cpp' -o -name '*.h' \) -print0 | \
    xargs -0 -r clang-tidy-14 \
        -p "$BUILD_DIR" \
        --checks="$TIER0_CHECKS" \
        --header-filter='src/core/tensor/.*'

echo ""
echo "=== Tier 0 Analysis Complete ==="
