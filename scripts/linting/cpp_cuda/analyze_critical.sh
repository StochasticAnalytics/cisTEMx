#!/usr/bin/env bash
# Run Tier 0+1 (Critical) static analysis - important correctness/performance issues
# Usage: ./analyze_critical.sh [path]
# Default path: src/core/tensor/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build/clang-tidy-debug"
TARGET_PATH="${1:-src/core/tensor/}"

# Tier 0+1 checks: Blocker + Critical
TIER01_CHECKS="bugprone-*,-bugprone-easily-swappable-parameters,-bugprone-narrowing-conversions,-bugprone-implicit-widening-of-multiplication-result,performance-unnecessary-copy-initialization,performance-for-range-copy,performance-noexcept-move-constructor,performance-inefficient-vector-operation,performance-move-constructor-init,performance-move-const-arg,modernize-use-nullptr,modernize-use-override,modernize-use-emplace,modernize-use-default-member-init,cppcoreguidelines-special-member-functions,cert-err52-cpp"

cd "$PROJECT_ROOT"

if [[ ! -f "$BUILD_DIR/compile_commands.json" ]]; then
    echo "Error: Compilation database not found at $BUILD_DIR/compile_commands.json"
    echo "Run: Tasks → 'Build with Bear (Generate compile_commands.json)'"
    exit 1
fi

echo "=== Tier 0+1 (Critical) Static Analysis ==="
echo "Target: $TARGET_PATH"
echo ""

find "$TARGET_PATH" \( -name '*.cpp' -o -name '*.h' \) -print0 | \
    xargs -0 -r clang-tidy-14 \
        -p "$BUILD_DIR" \
        --checks="$TIER01_CHECKS" \
        --header-filter='src/core/tensor/.*'

echo ""
echo "=== Tier 0+1 Analysis Complete ==="
