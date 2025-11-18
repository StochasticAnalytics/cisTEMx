#!/bin/bash
# Script to create a new git worktree for cisTEMx with proper submodule setup
# Usage: ./create_worktree.sh branch_name

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Auto-detect repository root
detect_repo_root() {
    local detected_root

    # Try to detect using git
    if detected_root=$(git rev-parse --show-toplevel 2>/dev/null); then
        # Validate this is actually cisTEMx repo by checking for marker files
        # We check for core_headers.h and scripts directory as cisTEMx-specific markers
        if [[ -f "$detected_root/src/core/core_headers.h" ]] && [[ -d "$detected_root/scripts" ]]; then
            echo "$detected_root"
            return 0
        fi
    fi

    return 1
}

# Configuration - Auto-detect repo root
# Try auto-detection first, fall back to environment variable for backward compatibility
if MAIN_REPO=$(detect_repo_root); then
    echo -e "${YELLOW}INFO: Auto-detected repository root: $MAIN_REPO${NC}"
elif [[ -n "${cistemx_main_repo:-}" ]]; then
    MAIN_REPO="$cistemx_main_repo"
    echo -e "${YELLOW}INFO: Using cistemx_main_repo environment variable: $MAIN_REPO${NC}"
else
    echo -e "${RED}ERROR: Could not detect cisTEMx repository root.${NC}" >&2
    echo -e "${RED}Please run this script from within the cisTEMx repository,${NC}" >&2
    echo -e "${RED}or set the cistemx_main_repo environment variable.${NC}" >&2
    exit 1
fi
WORKTREE_DIR="$MAIN_REPO/worktrees"

# List of unofficial submodules (SSH paths)
UNOFFICIAL_SUBMODULES=(
    "git@github.com:StochasticAnalytics/core_knowledge_graph.git:core_knowledge_graph"
)

# Track what we've created for cleanup
CREATED_WORKTREE=""
CREATED_BRANCH=""

# Helper functions
log_error() {
    echo -e "${RED}ERROR: $1${NC}" >&2
}

log_success() {
    echo -e "${GREEN}SUCCESS: $1${NC}"
}

log_info() {
    echo -e "${YELLOW}INFO: $1${NC}"
}

cleanup_on_failure() {
    if [[ -n "$CREATED_WORKTREE" && -d "$CREATED_WORKTREE" ]]; then
        log_info "Cleaning up: removing worktree $CREATED_WORKTREE"
        cd "$MAIN_REPO"
        git worktree remove --force "$CREATED_WORKTREE" 2>/dev/null || true
    fi

    if [[ -n "$CREATED_BRANCH" ]]; then
        log_info "Cleaning up: removing branch $CREATED_BRANCH"
        cd "$MAIN_REPO"
        git branch -D "$CREATED_BRANCH" 2>/dev/null || true
    fi

    # Restore symlinks in main repo if they were unlinked
    cd "$MAIN_REPO"
    handle_symlinks_in_repo "$MAIN_REPO" "restore"
}

fail_with_cleanup() {
    log_error "$1"
    cleanup_on_failure
    exit 1
}

# Set up trap for cleanup on any error
trap 'cleanup_on_failure' ERR

# Validate/Fix helper functions for setup verification
validate_ckg_exists() {
    local ckg_path="$MAIN_REPO/core_knowledge_graph"
    local validation_failed=0

    if [[ ! -d "$ckg_path" ]]; then
        log_error "core_knowledge_graph directory not found at: $ckg_path"
        return 1
    fi

    if [[ ! -d "$ckg_path/.git" ]]; then
        log_error "core_knowledge_graph exists but is not a git repository"
        return 1
    fi

    log_success "core_knowledge_graph exists and is a git repository"
    return 0
}

validate_symlinks() {
    local validation_failed=0

    cd "$MAIN_REPO"

    for item in .claude CLAUDE.md; do
        local expected_target="core_knowledge_graph/$item"

        if [[ ! -e "$item" ]]; then
            log_error "$item does not exist (should link to $expected_target)"
            validation_failed=1
        elif [[ ! -L "$item" ]]; then
            log_error "$item exists but is not a symlink (should link to $expected_target)"
            validation_failed=1
        else
            local actual_target=$(readlink "$item")
            if [[ "$actual_target" != "$expected_target" ]]; then
                log_error "$item points to '$actual_target' but should point to '$expected_target'"
                validation_failed=1
            else
                log_success "$item -> $actual_target (correct)"
            fi
        fi
    done

    return $validation_failed
}

fix_symlinks() {
    cd "$MAIN_REPO"

    log_info "Fixing symlinks..."

    for item in .claude CLAUDE.md; do
        local target="core_knowledge_graph/$item"

        # Remove existing item if it's not a correct symlink
        if [[ -e "$item" || -L "$item" ]]; then
            if [[ -L "$item" ]]; then
                local current_target=$(readlink "$item")
                if [[ "$current_target" == "$target" ]]; then
                    log_info "$item already correctly linked"
                    continue
                fi
            fi
            log_info "Removing incorrect $item"
            rm -f "$item"
        fi

        # Create the symlink
        log_info "Creating $item -> $target"
        ln -sf "$target" "$item"
    done

    log_success "Symlinks fixed"
    return 0
}

# Validation functions
check_main_repo() {
    log_info "Checking we're in main repository..."

    if [[ "$(pwd)" != "$MAIN_REPO" ]]; then
        fail_with_cleanup "Not in main repository. Current: $(pwd), Expected: $MAIN_REPO"
    fi

    if [[ ! -d .git ]]; then
        fail_with_cleanup "Not a git repository"
    fi

    log_success "In correct repository"
}

check_clean_git_state() {
    log_info "Checking git state is clean..."

    if [[ -n $(git status --porcelain) ]]; then
        fail_with_cleanup "Git working tree is not clean. Commit or stash changes first."
    fi

    log_success "Git state is clean"
}

check_subrepo_state() {
    local subrepo_path="$1"
    local main_branch="$2"

    log_info "Checking $subrepo_path sub-repository state..."

    # Check if the sub-repo exists
    if [[ ! -d "$MAIN_REPO/$subrepo_path" ]]; then
        log_error "Sub-repository not found: $subrepo_path"
        return 1
    fi

    # Navigate to sub-repo
    cd "$MAIN_REPO/$subrepo_path"

    # Check if it's a git repository
    if [[ ! -d .git ]]; then
        fail_with_cleanup "$subrepo_path is not a git repository"
    fi

    # Check if working tree is clean
    if [[ -n $(git status --porcelain) ]]; then
        fail_with_cleanup "$subrepo_path has uncommitted changes. Commit or stash changes first."
    fi

    # Get current branch
    local current_branch=$(git branch --show-current)

    if [[ -z "$current_branch" ]]; then
        fail_with_cleanup "$subrepo_path is in detached HEAD state"
    fi

    # Check if branch matches main repo branch
    if [[ "$current_branch" != "$main_branch" ]]; then
        fail_with_cleanup "$subrepo_path is on branch '$current_branch', but main repo is on '$main_branch'. They must match."
    fi

    log_success "$subrepo_path is clean and on branch '$current_branch'"

    # Return to main repo
    cd "$MAIN_REPO"
}

validate_branch_name() {
    local branch_name="$1"

    log_info "Validating branch name format..."

    # Check it matches snake_case pattern
    if [[ ! "$branch_name" =~ ^[a-z][a-z0-9_]*$ ]]; then
        fail_with_cleanup "Branch name must be snake_case (lowercase, underscores, starting with letter): $branch_name"
    fi

    log_success "Branch name format is valid"
}

check_worktree_dir() {
    log_info "Checking worktree directory..."

    if [[ ! -d "$WORKTREE_DIR" ]]; then
        fail_with_cleanup "Worktree directory does not exist: $WORKTREE_DIR"
    fi

    if [[ ! -w "$WORKTREE_DIR" ]]; then
        fail_with_cleanup "Worktree directory is not writable: $WORKTREE_DIR"
    fi

    log_success "Worktree directory exists and is writable"
}

check_branch_not_exists() {
    local branch_name="$1"

    log_info "Checking branch doesn't already exist..."

    if git show-ref --verify --quiet "refs/heads/$branch_name"; then
        fail_with_cleanup "Branch already exists: $branch_name"
    fi

    log_success "Branch name is available"
}

check_dir_not_exists() {
    local branch_name="$1"
    local target_dir="$WORKTREE_DIR/$branch_name"

    log_info "Checking directory doesn't already exist..."

    if [[ -e "$target_dir" ]]; then
        fail_with_cleanup "Directory already exists: $target_dir"
    fi

    log_success "Directory name is available"
}

handle_symlinks_in_repo() {
    local repo_dir="$1"
    local action="$2"  # "unlink" or "restore"

    cd "$repo_dir"

    for item in .claude CLAUDE.md; do
        if [[ "$action" == "unlink" ]]; then
            if [[ -e "$item" ]]; then
                log_info "Checking $item in $repo_dir..."

                if [[ ! -L "$item" ]]; then
                    fail_with_cleanup "$item exists but is not a symlink in $repo_dir"
                fi

                # Store the link target before unlinking
                local target=$(readlink "$item")
                echo "$target" > ".${item}_link_target"

                log_info "Unlinking $item (target: $target)"
                unlink "$item"
            fi
        elif [[ "$action" == "restore" ]]; then
            if [[ -f ".${item}_link_target" ]]; then
                local target=$(cat ".${item}_link_target")
                log_info "Restoring $item -> $target"
                ln -s "$target" "$item"
                rm ".${item}_link_target"
            fi
        fi
    done
}

create_worktree() {
    local branch_name="$1"
    local target_dir="$WORKTREE_DIR/$branch_name"

    log_info "Creating worktree and branch..."

    # Temporarily unlink .claude and CLAUDE.md in main repo
    handle_symlinks_in_repo "$MAIN_REPO" "unlink"

    # Create the worktree
    if ! git worktree add -b "$branch_name" "$target_dir"; then
        # Restore links if worktree creation failed
        handle_symlinks_in_repo "$MAIN_REPO" "restore"
        fail_with_cleanup "Failed to create worktree"
    fi

    # Track what we created for potential cleanup
    CREATED_WORKTREE="$target_dir"
    CREATED_BRANCH="$branch_name"

    # Restore links in main repo
    handle_symlinks_in_repo "$MAIN_REPO" "restore"

    log_success "Worktree created at $target_dir"
    echo "$target_dir"
}

checkout_unofficial_submodules() {
    local worktree_dir="$1"
    local source_branch="$2"
    local new_branch="$3"

    cd "$worktree_dir"

    log_info "Checking out unofficial submodules..."

    for submodule_entry in "${UNOFFICIAL_SUBMODULES[@]}"; do
        IFS=':' read -r repo_url subdir <<< "$submodule_entry"

        log_info "Cloning $subdir from $repo_url (branch: $source_branch)..."

        if [[ -d "$subdir" ]]; then
            log_info "$subdir already exists, skipping"
            continue
        fi

        # Clone from the matching branch
        if ! git clone --branch "$source_branch" "$repo_url" "$subdir"; then
            log_error "Failed to clone $repo_url (branch: $source_branch) into $subdir"
            continue
        fi

        log_success "Cloned $subdir from branch '$source_branch'"

        # Navigate into the sub-repo and create the new branch
        cd "$subdir"

        log_info "Creating new branch '$new_branch' in $subdir..."

        if ! git checkout -b "$new_branch"; then
            log_error "Failed to create branch '$new_branch' in $subdir"
            cd "$worktree_dir"
            continue
        fi

        log_success "Created and checked out branch '$new_branch' in $subdir"

        # Return to worktree directory
        cd "$worktree_dir"
    done
}

setup_worktree_symlinks() {
    local worktree_dir="$1"

    log_info "Setting up symlinks in worktree..."

    # Check what symlinks exist in main repo
    cd "$MAIN_REPO"

    for item in .claude CLAUDE.md; do
        if [[ -L "$item" ]]; then
            local target=$(readlink "$item")
            log_info "Found $item -> $target in main repo"

            cd "$worktree_dir"

            if [[ "$target" == /* ]]; then
                # Absolute path - use directly
                log_info "Creating $item -> $target (absolute path)"
                ln -s "$target" "$item"
            else
                # Relative path - assume same relative structure works in worktree
                # This works because we clone submodules in same relative location
                log_info "Creating $item -> $target (relative path)"

                # Verify target will exist
                if [[ ! -e "$target" ]]; then
                    log_error "Warning: $target does not exist yet in worktree"
                    log_error "This is expected if it's in a submodule that needs to be initialized"
                fi

                ln -s "$target" "$item"
            fi

            cd "$MAIN_REPO"
        fi
    done

    log_success "Symlinks configured"
}

# Mode functions
show_usage() {
    cat << EOF
Usage: $0 <mode> [arguments]

Modes:
  validate              Check cisTEMx setup (repo root, core_knowledge_graph, symlinks)
  fix                   Validate and automatically fix any issues
  worktree <branch> [base]   Create a new worktree
                        - branch: Name for the new branch (snake_case)
                        - base: Optional base branch (defaults to main)

Examples:
  $0 validate
  $0 fix
  $0 worktree my_feature
  $0 worktree my_feature main

For worktree mode, the script will:
  1. Validate the current setup
  2. Create a new git worktree with the specified branch name
  3. Clone core_knowledge_graph into the worktree
  4. Set up symlinks for .claude and CLAUDE.md
EOF
}

mode_validate() {
    echo "========================================"
    echo "cisTEMx Setup Validation"
    echo "========================================"
    echo ""

    local validation_passed=0

    # Check core_knowledge_graph
    if ! validate_ckg_exists; then
        validation_passed=1
    fi

    # Check symlinks
    if ! validate_symlinks; then
        validation_passed=1
    fi

    echo ""
    if [[ $validation_passed -eq 0 ]]; then
        log_success "All validation checks passed"
        return 0
    else
        log_error "Some validation checks failed"
        echo ""
        echo "Run '$0 fix' to automatically fix these issues"
        return 1
    fi
}

mode_fix() {
    echo "========================================"
    echo "cisTEMx Setup Fix"
    echo "========================================"
    echo ""

    # Check core_knowledge_graph (cannot auto-fix if missing)
    if ! validate_ckg_exists; then
        echo ""
        log_error "core_knowledge_graph is missing or not a git repository"
        log_error "This must be fixed manually. Please clone core_knowledge_graph:"
        echo ""
        echo "  git clone git@github.com:StochasticAnalytics/core_knowledge_graph.git $MAIN_REPO/core_knowledge_graph"
        echo ""
        return 1
    fi

    # Fix symlinks
    if ! fix_symlinks; then
        log_error "Failed to fix symlinks"
        return 1
    fi

    echo ""
    log_success "Setup fixed successfully"
    return 0
}

mode_worktree() {
    local branch_name="$1"
    local base_branch="${2:-}"

    echo "========================================"
    echo "cisTEMx Worktree Creation"
    echo "========================================"
    echo ""

    # Run all validations
    check_main_repo
    check_clean_git_state
    validate_branch_name "$branch_name"
    check_worktree_dir
    check_branch_not_exists "$branch_name"
    check_dir_not_exists "$branch_name"

    # Determine the base branch
    cd "$MAIN_REPO"
    if [[ -n "$base_branch" ]]; then
        # Verify the specified base branch exists
        if ! git show-ref --verify --quiet "refs/heads/$base_branch"; then
            fail_with_cleanup "Specified base branch does not exist: $base_branch"
        fi
        log_info "Using specified base branch: $base_branch"
    else
        # Default to main
        base_branch="main"
        if ! git show-ref --verify --quiet "refs/heads/$base_branch"; then
            fail_with_cleanup "Default base branch 'main' does not exist. Please specify a base branch."
        fi
        log_info "Using default base branch: $base_branch"
    fi

    # Check sub-repository state (core_knowledge_graph)
    check_subrepo_state "core_knowledge_graph" "$base_branch"

    echo ""
    log_info "All validations passed. Creating worktree..."
    echo ""

    # Create the worktree
    local worktree_path
    worktree_path=$(create_worktree "$branch_name")

    # Checkout unofficial submodules with branch synchronization
    checkout_unofficial_submodules "$worktree_path" "$base_branch" "$branch_name"

    # Setup symlinks in the new worktree
    setup_worktree_symlinks "$worktree_path"

    # Clear the error trap since we succeeded
    trap - ERR

    echo ""
    echo "========================================"
    log_success "Worktree setup complete!"
    echo "========================================"
    echo ""
    echo "Worktree location: $worktree_path"
    echo "Branch (main repo): $branch_name"
    echo "Branch (sub-repos): $branch_name (based on $base_branch)"
    echo ""
    echo "To start working:"
    echo "  cd $worktree_path"
    echo ""
    echo "To remove this worktree later:"
    echo "  git worktree remove $worktree_path"
    echo "  git branch -d $branch_name  # if you want to delete the branch too"
    echo ""
}

# Main script
main() {
    if [[ $# -lt 1 ]]; then
        show_usage
        exit 1
    fi

    local mode="$1"
    shift

    case "$mode" in
        validate)
            mode_validate
            ;;
        fix)
            mode_fix
            ;;
        worktree)
            if [[ $# -lt 1 ]]; then
                log_error "worktree mode requires a branch name"
                echo ""
                show_usage
                exit 1
            fi
            mode_worktree "$@"
            ;;
        -h|--help|help)
            show_usage
            exit 0
            ;;
        *)
            log_error "Unknown mode: $mode"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

main "$@"
