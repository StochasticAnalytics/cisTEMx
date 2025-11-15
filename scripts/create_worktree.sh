#!/bin/bash
# Script to create a new git worktree for cisTEMx with proper submodule setup
# Usage: ./create_worktree.sh branch_name

set -euo pipefail

# Configuration
if [[ -z "${cistemx_main_repo}" ]]; then
    echo "cistemx_main_repo not set, using default"
    exit 1
else
    MAIN_REPO="$cistemx_main_repo"
fi
WORKTREE_DIR="$MAIN_REPO/worktrees"

# List of unofficial submodules (SSH paths)
UNOFFICIAL_SUBMODULES=(
    "git@github.com:StochasticAnalytics/core_knowledge_graph.git:core_knowledge_graph"
)

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

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

# Main script
main() {
    if [[ $# -ne 1 ]]; then
        log_error "Usage: $0 branch_name"
        log_error "Example: $0 my_new_feature"
        exit 1
    fi

    local branch_name="$1"

    echo "========================================"
    echo "cisTEMx Worktree Creation Script"
    echo "========================================"
    echo ""

    # Run all validations
    check_main_repo
    check_clean_git_state
    validate_branch_name "$branch_name"
    check_worktree_dir
    check_branch_not_exists "$branch_name"
    check_dir_not_exists "$branch_name"

    # Get the current branch of the main repo
    cd "$MAIN_REPO"
    local main_branch=$(git branch --show-current)

    if [[ -z "$main_branch" ]]; then
        fail_with_cleanup "Main repository is in detached HEAD state. Please checkout a branch first."
    fi

    log_info "Main repository is on branch: $main_branch"

    # Check sub-repository state (core_knowledge_graph)
    check_subrepo_state "core_knowledge_graph" "$main_branch"

    echo ""
    log_info "All validations passed. Creating worktree..."
    echo ""

    # Create the worktree
    local worktree_path
    worktree_path=$(create_worktree "$branch_name")

    # Checkout unofficial submodules with branch synchronization
    checkout_unofficial_submodules "$worktree_path" "$main_branch" "$branch_name"

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
    echo "Branch (sub-repos): $branch_name (based on $main_branch)"
    echo ""
    echo "To start working:"
    echo "  cd $worktree_path"
    echo ""
    echo "To remove this worktree later:"
    echo "  git worktree remove $worktree_path"
    echo "  git branch -d $branch_name  # if you want to delete the branch too"
    echo ""
}

main "$@"
