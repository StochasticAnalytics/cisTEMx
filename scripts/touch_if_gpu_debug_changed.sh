#!/bin/bash
# Compares ENABLE_GPU_DEBUG value and touches affected files if changed

BUILD_DIR="$1"
NEW_VALUE="$2"
STATE_FILE="$BUILD_DIR/.gpu_debug_level"
SOURCE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Read previous value
if [ -f "$STATE_FILE" ]; then
    OLD_VALUE=$(cat "$STATE_FILE")
else
    OLD_VALUE="<none>"
fi

# Compare
if [ "$OLD_VALUE" != "$NEW_VALUE" ]; then
    echo "ENABLE_GPU_DEBUG changed: $OLD_VALUE -> $NEW_VALUE"
    echo "Finding files that reference ENABLE_GPU_DEBUG..."

    # Find all source files in src/ that reference ENABLE_GPU_DEBUG
    FILES=$(cd "$SOURCE_ROOT" && find src/ \( -name "*.cu" -o -name "*.cuh" -o -name "*.cpp" -o -name "*.h" \) -exec grep -l "ENABLE_GPU_DEBUG" {} +)

    if [ -n "$FILES" ]; then
        FILE_COUNT=$(echo "$FILES" | wc -l)
        echo "Touching $FILE_COUNT files..."
        cd "$SOURCE_ROOT" && echo "$FILES" | xargs touch
        echo "Files marked for rebuild"
    else
        echo "No files found using ENABLE_GPU_DEBUG"
    fi

    # Save new value
    echo "$NEW_VALUE" > "$STATE_FILE"
else
    echo "ENABLE_GPU_DEBUG unchanged ($NEW_VALUE), no rebuild needed"
fi
