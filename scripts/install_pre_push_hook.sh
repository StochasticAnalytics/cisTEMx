#!/bin/bash
# Install pre-push hook for cisTEMx development
# This script should be run from the project root or called by regenerate_project.sh

set -e

# Find git directory (handles both regular repos and worktrees)
GIT_DIR=$(git rev-parse --git-dir)
HOOKS_DIR="$GIT_DIR/hooks"

# For worktrees, git hooks go in the main repo's hooks directory
if [[ "$GIT_DIR" == *"/worktrees/"* ]]; then
    # Extract main repo path from worktree git dir
    MAIN_GIT_DIR=$(echo "$GIT_DIR" | sed 's|/\.git/worktrees/.*|/.git|')
    HOOKS_DIR="$MAIN_GIT_DIR/hooks"
fi

echo "Installing pre-push hook to: $HOOKS_DIR"

# Create hooks directory if it doesn't exist
mkdir -p "$HOOKS_DIR"

# Create pre-push hook
cat > "$HOOKS_DIR/pre-push" << 'EOF'
#!/bin/bash
# Pre-push hook for cisTEMx
# Runs build and full test suite before allowing push
# Override with: git push --no-verify

set -e  # Exit on error (will be temporarily disabled for specific checks)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project paths
PROJECT_ROOT=$(git rev-parse --show-toplevel)
HOOK_CONFIG="$PROJECT_ROOT/.git/hooks/.pre-push-config"
TASKS_JSON="$PROJECT_ROOT/.vscode/tasks.json"

# Timing
START_TIME=$(date +%s)

echo -e "${BLUE}=== cisTEMx Pre-Push Hook ===${NC}"
echo ""

# Function to print elapsed time
print_elapsed() {
    local end_time=$(date +%s)
    local elapsed=$((end_time - START_TIME))
    local minutes=$((elapsed / 60))
    local seconds=$((elapsed % 60))
    echo -e "${BLUE}Total elapsed time: ${minutes}m ${seconds}s${NC}"
}

# Function to extract build directory from command
extract_build_dir() {
    local command="$1"
    # Extract the path after "cd " and before " &&"
    # Example: "cd ${build_dir}/intel-gpu-debug-static && make ..."
    # -> "intel-gpu-debug-static"
    echo "$command" | sed -n 's|.*cd \${build_dir}/\([^ ]*\) .*|\1|p'
}

# Function to select build configuration
select_build_config() {
    echo -e "${YELLOW}Selecting build configuration for testing...${NC}"
    echo ""

    if [ ! -f "$TASKS_JSON" ]; then
        echo -e "${RED}ERROR: tasks.json not found at $TASKS_JSON${NC}"
        exit 1
    fi

    # Check if jq is available
    if ! command -v jq &> /dev/null; then
        echo -e "${RED}ERROR: jq is required for parsing tasks.json${NC}"
        exit 1
    fi

    # Strip comments from tasks.json (VS Code allows comments in JSON)
    # Remove // comment lines and inline // comments
    CLEAN_JSON=$(grep -v '^\s*//' "$TASKS_JSON" | sed 's|//.*$||')

    # Extract BUILD tasks (exclude Configure tasks)
    mapfile -t labels < <(echo "$CLEAN_JSON" | jq -r '.tasks[] | select(.label | contains("BUILD")) | select(.label | contains("Configure") | not) | .label')
    mapfile -t commands < <(echo "$CLEAN_JSON" | jq -r '.tasks[] | select(.label | contains("BUILD")) | select(.label | contains("Configure") | not) | .command')

    if [ ${#labels[@]} -eq 0 ]; then
        echo -e "${RED}ERROR: No BUILD tasks found in tasks.json${NC}"
        exit 1
    fi

    # Display menu
    echo "Available build configurations:"
    for i in "${!labels[@]}"; do
        echo "  $((i+1))) ${labels[$i]}"
    done
    echo ""

    # Get user selection
    while true; do
        read -p "Select configuration (1-${#labels[@]}): " selection </dev/tty
        if [[ "$selection" =~ ^[0-9]+$ ]] && [ "$selection" -ge 1 ] && [ "$selection" -le ${#labels[@]} ]; then
            break
        fi
        echo -e "${RED}Invalid selection. Please try again.${NC}"
    done

    # Store selection (0-indexed)
    local idx=$((selection - 1))
    local selected_label="${labels[$idx]}"
    local selected_command="${commands[$idx]}"
    local build_subdir=$(extract_build_dir "$selected_command")

    if [ -z "$build_subdir" ]; then
        echo -e "${RED}ERROR: Could not extract build directory from command${NC}"
        exit 1
    fi

    local build_dir="$PROJECT_ROOT/build/$build_subdir"

    echo ""
    echo -e "${GREEN}Selected: $selected_label${NC}"
    echo -e "${BLUE}Build directory: $build_dir${NC}"

    # Save configuration
    cat > "$HOOK_CONFIG" <<EOFCONFIG
# Pre-push hook configuration
# Generated: $(date)
BUILD_LABEL=$selected_label
BUILD_DIR=$build_dir
BUILD_COMMAND=$selected_command
EOFCONFIG

    echo ""
}

# Function to load build configuration
load_build_config() {
    if [ ! -f "$HOOK_CONFIG" ]; then
        return 1
    fi

    source "$HOOK_CONFIG"

    # Verify configuration is still valid
    if [ -z "$BUILD_DIR" ] || [ ! -d "$BUILD_DIR" ]; then
        echo -e "${YELLOW}WARNING: Saved build directory no longer exists${NC}"
        return 1
    fi

    echo -e "${BLUE}Using saved configuration: $BUILD_LABEL${NC}"
    echo -e "${BLUE}Build directory: $BUILD_DIR${NC}"

    # Ask if user wants to reselect
    read -p "Use this configuration? (Y/n): " response </dev/tty
    if [[ "$response" =~ ^[Nn] ]]; then
        return 1
    fi

    echo ""
    return 0
}

# Load or select build configuration
if ! load_build_config; then
    select_build_config
    # Reload the newly created config
    source "$HOOK_CONFIG"
fi

# Phase 1: Build
echo -e "${YELLOW}=== Phase 1: Building cisTEMx ===${NC}"
echo -e "${BLUE}Command: cd $BUILD_DIR && make -j$(nproc)${NC}"
echo ""

BUILD_START=$(date +%s)

# Execute build command
cd "$BUILD_DIR"
set +e  # Don't exit on error, we want to capture it
make -j$(nproc)
BUILD_EXIT=$?
set -e

BUILD_END=$(date +%s)
BUILD_TIME=$((BUILD_END - BUILD_START))

if [ $BUILD_EXIT -ne 0 ]; then
    echo ""
    echo -e "${RED}============================================${NC}"
    echo -e "${RED}BUILD FAILED (exit code: $BUILD_EXIT)${NC}"
    echo -e "${RED}Build time: ${BUILD_TIME}s${NC}"
    echo -e "${RED}============================================${NC}"
    echo ""
    echo -e "${YELLOW}To push anyway (not recommended):${NC}"
    echo -e "  git push --no-verify"
    echo ""
    print_elapsed
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Build succeeded (${BUILD_TIME}s)${NC}"
echo ""

# Phase 2: Verify test executables exist
echo -e "${YELLOW}=== Phase 2: Verifying test executables ===${NC}"

TEST_EXECUTABLES=(
    "src/test/unit_test_runner"
    "src/programs/console_test/console_test"
    "src/programs/samples/samples_functional_testing"
)

ALL_EXIST=true
for exe in "${TEST_EXECUTABLES[@]}"; do
    full_path="$BUILD_DIR/$exe"
    if [ ! -x "$full_path" ]; then
        echo -e "${RED}✗ Not found or not executable: $exe${NC}"
        ALL_EXIST=false
    else
        echo -e "${GREEN}✓ Found: $exe${NC}"
    fi
done

if [ "$ALL_EXIST" = false ]; then
    echo ""
    echo -e "${RED}============================================${NC}"
    echo -e "${RED}TEST EXECUTABLES MISSING${NC}"
    echo -e "${RED}Build succeeded but test executables not found${NC}"
    echo -e "${RED}============================================${NC}"
    echo ""
    echo -e "${YELLOW}To push anyway (not recommended):${NC}"
    echo -e "  git push --no-verify"
    echo ""
    print_elapsed
    exit 1
fi

echo -e "${GREEN}✓ All test executables found${NC}"
echo ""

# Phase 3: Run tests
echo -e "${YELLOW}=== Phase 3: Running test suites ===${NC}"
echo ""

# Test 1: unit_test_runner
echo -e "${BLUE}[1/3] Running unit_test_runner...${NC}"
TEST1_START=$(date +%s)
set +e
"$BUILD_DIR/src/test/unit_test_runner"
TEST1_EXIT=$?
set -e
TEST1_END=$(date +%s)
TEST1_TIME=$((TEST1_END - TEST1_START))

if [ $TEST1_EXIT -ne 0 ]; then
    echo ""
    echo -e "${RED}============================================${NC}"
    echo -e "${RED}UNIT TESTS FAILED (exit code: $TEST1_EXIT)${NC}"
    echo -e "${RED}Test time: ${TEST1_TIME}s${NC}"
    echo -e "${RED}============================================${NC}"
    echo ""
    echo -e "${YELLOW}To push anyway (not recommended):${NC}"
    echo -e "  git push --no-verify"
    echo ""
    print_elapsed
    exit 1
fi
echo -e "${GREEN}✓ unit_test_runner passed (${TEST1_TIME}s)${NC}"
echo ""

# Test 2: console_test
echo -e "${BLUE}[2/3] Running console_test...${NC}"
TEST2_START=$(date +%s)
set +e
"$BUILD_DIR/src/programs/console_test/console_test"
TEST2_EXIT=$?
set -e
TEST2_END=$(date +%s)
TEST2_TIME=$((TEST2_END - TEST2_START))

if [ $TEST2_EXIT -ne 0 ]; then
    echo ""
    echo -e "${RED}============================================${NC}"
    echo -e "${RED}CONSOLE TEST FAILED (exit code: $TEST2_EXIT)${NC}"
    echo -e "${RED}Test time: ${TEST2_TIME}s${NC}"
    echo -e "${RED}============================================${NC}"
    echo ""
    echo -e "${YELLOW}To push anyway (not recommended):${NC}"
    echo -e "  git push --no-verify"
    echo ""
    print_elapsed
    exit 1
fi
echo -e "${GREEN}✓ console_test passed (${TEST2_TIME}s)${NC}"
echo ""

# Test 3: samples_functional_testing
echo -e "${BLUE}[3/3] Running samples_functional_testing...${NC}"
TEST3_START=$(date +%s)
set +e
"$BUILD_DIR/src/programs/samples/samples_functional_testing"
TEST3_EXIT=$?
set -e
TEST3_END=$(date +%s)
TEST3_TIME=$((TEST3_END - TEST3_START))

if [ $TEST3_EXIT -ne 0 ]; then
    echo ""
    echo -e "${RED}============================================${NC}"
    echo -e "${RED}FUNCTIONAL TESTS FAILED (exit code: $TEST3_EXIT)${NC}"
    echo -e "${RED}Test time: ${TEST3_TIME}s${NC}"
    echo -e "${RED}============================================${NC}"
    echo ""
    echo -e "${YELLOW}To push anyway (not recommended):${NC}"
    echo -e "  git push --no-verify"
    echo ""
    print_elapsed
    exit 1
fi
echo -e "${GREEN}✓ samples_functional_testing passed (${TEST3_TIME}s)${NC}"
echo ""

# Success!
TOTAL_TEST_TIME=$((TEST1_TIME + TEST2_TIME + TEST3_TIME))
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}ALL TESTS PASSED${NC}"
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}Build time: ${BUILD_TIME}s${NC}"
echo -e "${GREEN}Test time:  ${TOTAL_TEST_TIME}s (unit: ${TEST1_TIME}s, console: ${TEST2_TIME}s, functional: ${TEST3_TIME}s)${NC}"
echo ""
print_elapsed
echo ""
echo -e "${GREEN}Proceeding with push...${NC}"
echo ""

exit 0
EOF

# Make the hook executable
chmod +x "$HOOKS_DIR/pre-push"

echo "Pre-push hook installed successfully!"
