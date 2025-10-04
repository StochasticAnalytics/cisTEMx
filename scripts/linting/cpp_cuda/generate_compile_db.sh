#!/usr/bin/env bash
# Generate compilation database with Bear for static analysis
# Usage: ./generate_compile_db.sh [build_dir] [clean]
# Default build_dir: build/clang-tidy-debug
# clean: "true" for clean build, "false" for incremental (default)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BUILD_DIR="${1:-build/clang-tidy-debug}"
CLEAN_BUILD="${2:-false}"

cd "$PROJECT_ROOT"

if [[ ! -d "$BUILD_DIR" ]]; then
    echo "Creating build directory: $BUILD_DIR"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    echo "Configuring with Clang..."
    CC=clang CXX=clang++ ../../configure \
        --enable-debugmode \
        --with-cuda=/usr/local/cuda \
        --with-wx-config=/opt/WX/clang-static/bin/wx-config \
        --enable-experimental \
        --enable-openmp \
        --disable-build-all \
        --enable-profiling \
        --enable-fp16-particlestacks \
        --disable-multiple-global-refinements \
        --enable-build-scale-with-mask \
        --enable-build-create-mask \
        --enable-build-resample \
        --enable-build-resize
    cd "$PROJECT_ROOT"
fi

cd "$BUILD_DIR"

if [[ "$CLEAN_BUILD" == "true" ]]; then
    echo "Clean build requested..."
    make clean
fi

echo "Building with Bear to generate compilation database..."
bear -- make -j"$(nproc)"

echo ""
echo "Compilation database generated: $BUILD_DIR/compile_commands.json"
echo "Total entries: $(jq '. | length' compile_commands.json)"

# Create symlink in project root for IDE integration
cd "$PROJECT_ROOT"
ln -sf "$BUILD_DIR/compile_commands.json" .
echo "Linked to project root for IDE integration"
echo ""
echo "✓ Ready for static analysis!"
