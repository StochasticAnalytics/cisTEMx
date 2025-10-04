#!/usr/bin/env bash
# Run Tier 0-2 (Standard) static analysis - CI-level checks
# Usage: ./analyze_standard.sh [path]
# Default path: src/core/tensor/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build/clang-tidy-debug"
TARGET_PATH="${1:-src/core/tensor/}"

# Tier 0-2 checks: Blocker + Critical + Important
TIER012_CHECKS="bugprone-*,-bugprone-easily-swappable-parameters,-bugprone-narrowing-conversions,-bugprone-implicit-widening-of-multiplication-result,performance-*,modernize-*,-modernize-use-trailing-return-type,-modernize-use-nodiscard,-modernize-avoid-c-arrays,-modernize-concat-nested-namespaces,cert-err*,cert-flp30-c,cert-msc*,cert-oop*,concurrency-mt-unsafe,misc-unconventional-assign-operator,misc-new-delete-overloads,misc-misplaced-const,cppcoreguidelines-owning-memory,cppcoreguidelines-no-malloc,cppcoreguidelines-special-member-functions"

cd "$PROJECT_ROOT"

if [[ ! -f "$BUILD_DIR/compile_commands.json" ]]; then
    echo "Error: Compilation database not found at $BUILD_DIR/compile_commands.json"
    echo "Run: Tasks → 'Build with Bear (Generate compile_commands.json)'"
    exit 1
fi

echo "=== Tier 0-2 (Standard - CI Level) Static Analysis ==="
echo "Target: $TARGET_PATH"
echo ""

find "$TARGET_PATH" \( -name '*.cpp' -o -name '*.h' \) -print0 | \
    xargs -0 -r clang-tidy-14 \
        -p "$BUILD_DIR" \
        --checks="$TIER012_CHECKS" \
        --header-filter='src/core/tensor/.*'

echo ""
echo "=== Tier 0-2 Analysis Complete ==="
