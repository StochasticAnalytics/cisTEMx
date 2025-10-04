#!/usr/bin/env bash
# Run Tier 3 (Deep) static analysis - all checks from .clang-tidy config
# Usage: ./analyze_deep.sh [path]
# Default path: src/core/tensor/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build/clang-tidy-debug"
TARGET_PATH="${1:-src/core/tensor/}"

cd "$PROJECT_ROOT"

if [[ ! -f "$BUILD_DIR/compile_commands.json" ]]; then
    echo "Error: Compilation database not found at $BUILD_DIR/compile_commands.json"
    echo "Run: Tasks → 'Build with Bear (Generate compile_commands.json)'"
    exit 1
fi

echo "=== Tier 3 (Deep - All Checks) Static Analysis ==="
echo "Target: $TARGET_PATH"
echo "Using full .clang-tidy configuration"
echo ""

# No --checks override - uses full config from .clang-tidy
find "$TARGET_PATH" \( -name '*.cpp' -o -name '*.h' \) -print0 | \
    xargs -0 -r clang-tidy-14 \
        -p "$BUILD_DIR" \
        --header-filter='src/core/tensor/.*'

echo ""
echo "=== Tier 3 Analysis Complete ==="
