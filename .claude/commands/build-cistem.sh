#!/bin/bash
# Slash command for building cisTEM in debug mode
# This executes the exact build sequence from cpp-build-expert agent


# STEP 1: Get Git Project Root
git_root=$(git rev-parse --show-toplevel)

# STEP 2: Extract Build Directory from VS Code Tasks
build_subdir=$(grep -A 2 '"BUILD cisTEM DEBUG"' "$git_root/.vscode/tasks.json" | grep '"command"' | sed 's/.*cd \${build_dir}\/\([^ ]*\) &&.*/\1/')

# STEP 3: Construct Absolute Build Path
build_path="$git_root/build/$build_subdir"

# STEP 4: Determine Optimal Parallelism
core_count=$(lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l)
if [[ $core_count -gt 16 ]]; then
    core_count=16
fi

# STEP 5: Verify Variables Are Set
echo "Build Configuration:"
echo "  Git Root: $git_root"
echo "  Build Subdir: $build_subdir"
echo "  Build Path: $build_path"
echo "  Core Count: $core_count"
echo ""

# STEP 6: Execute Build
echo "Starting build..."
cd "$build_path"
make -j"$core_count"
