# cisTEMx Worktree Creation Script

## Overview

This script provides three modes for managing cisTEMx development environment:
1. **validate** - Verify repository setup (core_knowledge_graph, symlinks)
2. **fix** - Automatically repair common setup issues
3. **worktree** - Create git worktrees with proper submodule synchronization

Features:
- Auto-detection of repository root (no environment variables required)
- Symlink management for `.claude` and `CLAUDE.md`
- Unofficial submodule checkout (e.g., `core_knowledge_graph`)
- Proper validation and error handling with automatic cleanup
- Optional base branch specification for worktrees

## Requirements

- Run from anywhere within the cisTEMx repository (auto-detects root)
- For worktree mode: Git working tree must be clean (no uncommitted changes)
- For worktree mode: `worktrees/` directory must exist and be writable
- SSH access to unofficial submodule repositories

## Usage

```bash
./create_worktree.sh <mode> [arguments]
```

### Modes

#### validate mode
Check cisTEMx setup without making changes:

```bash
./create_worktree.sh validate
```

Validates:
- Repository root detection
- core_knowledge_graph exists and is a git repository
- .claude and CLAUDE.md symlinks point to correct targets

#### fix mode
Validate and automatically fix issues:

```bash
./create_worktree.sh fix
```

Fixes:
- Broken or missing .claude and CLAUDE.md symlinks
- Reports if core_knowledge_graph is missing (requires manual clone)

#### worktree mode
Create a new worktree with branch synchronization:

```bash
./create_worktree.sh worktree <branch_name> [base_branch]
```

Arguments:
- `branch_name` - Name for new branch (must be snake_case)
- `base_branch` - Optional base branch (defaults to main)

### Examples

```bash
# Validate current setup
./create_worktree.sh validate

# Fix symlink issues
./create_worktree.sh fix

# Create worktree from current branch
./create_worktree.sh worktree my_new_feature

# Create worktree from specific base branch
./create_worktree.sh worktree my_feature main

# Show help
./create_worktree.sh --help
```

### Worktree Mode Creates:
- New branch: `my_new_feature`
- New worktree at: `<repo_root>/worktrees/my_new_feature`
- Cloned submodules in the worktree
- Proper symlinks for `.claude` and `CLAUDE.md`

## What It Does

### Auto-Detection Phase
- Uses `git rev-parse --show-toplevel` to find repository root
- Validates this is cisTEMx by checking for marker files (`src/core/core_headers.h`, `scripts/`)
- Falls back to `cistemx_main_repo` environment variable if needed (backward compatibility)
- Works from any subdirectory within the repository

### validate Mode
1. Checks core_knowledge_graph directory exists and is a git repository
2. Verifies .claude and CLAUDE.md symlinks point to correct targets
3. Reports status without making changes

### fix Mode
1. Runs validation checks
2. Automatically fixes broken or missing symlinks
3. Reports if core_knowledge_graph needs manual cloning

### worktree Mode

#### 1. Validation Phase
- Checks git state is clean
- Validates branch name format (snake_case)
- Verifies worktrees directory exists and is writable
- Ensures branch name doesn't already exist
- Ensures target directory doesn't already exist
- Verifies base branch exists (defaults to main, or uses specified base branch)
- Checks `core_knowledge_graph` sub-repository is clean (no uncommitted changes)
- Verifies `core_knowledge_graph` is on the same branch as the base branch

#### 2. Worktree Creation
- Temporarily unlinks `.claude` and `CLAUDE.md` in main repo (if they exist as symlinks)
- Creates new git worktree with `git worktree add -b branch_name worktrees/branch_name`
- Restores symlinks in main repo

### 3. Submodule Setup (with Branch Synchronization)
- Clones unofficial submodules (currently `core_knowledge_graph`) into the worktree
- Uses SSH URLs defined in the script configuration
- **Branch synchronization**: Clones from the same branch as the main repository (e.g., if main repo is on `main`, clones from `main`)
- Creates a new branch in each sub-repository matching the worktree branch name
- This ensures sub-repositories track the same feature branches as the main repo

### 4. Symlink Configuration
- Recreates `.claude` and `CLAUDE.md` symlinks in the worktree
- Handles both relative and absolute symlink targets
- Preserves the same link structure as main repo

## Error Handling

The script includes robust error handling:
- Automatic cleanup on failure (removes worktree and branch)
- Restores symlinks in main repo if anything fails
- Clear error messages for each validation failure
- Trap for unexpected errors

## Cleanup

To remove a worktree later:

```bash
# Remove the worktree
git worktree remove /sa_shared/git/cisTEMx/worktrees/branch_name

# Delete the branch (if desired)
git branch -d branch_name  # or -D to force
```

## Configuration

### Repository Auto-Detection

The script automatically detects the cisTEMx repository root, so no environment variables are required. It:
1. Uses `git rev-parse --show-toplevel` to find the git root
2. Validates cisTEMx-specific marker files exist
3. Falls back to `cistemx_main_repo` environment variable if set (for backward compatibility)

This means you can run the script from any directory within cisTEMx.

### Adding New Unofficial Submodules

Edit the `UNOFFICIAL_SUBMODULES` array at the top of the script:

```bash
UNOFFICIAL_SUBMODULES=(
    "git@github.com:StochasticAnalytics/core_knowledge_graph.git:core_knowledge_graph"
    "git@github.com:StochasticAnalytics/another_repo.git:another_directory"
)
```

Format: `"ssh_url:target_directory"`

## Symlink Behavior

The script handles `.claude` and `CLAUDE.md` specially:

1. **During worktree creation**: These must be symlinks in main repo (or not exist). They are temporarily unlinked to avoid conflicts during `git worktree add`.

2. **In the worktree**: The same symlinks are recreated, pointing to the same targets. This works because:
   - If targets are relative paths (e.g., `core_knowledge_graph/.claude`), they work the same way in the worktree
   - If targets are absolute paths, they work identically everywhere

3. **Validation**: If `.claude` or `CLAUDE.md` exist but are NOT symlinks, the script fails with an error.

## Branch Synchronization Feature

### How It Works

The script ensures that sub-repositories (like `core_knowledge_graph`) stay synchronized with the main repository across branches:

1. **Pre-validation**: Before creating the worktree, the script checks:
   - Main repo is on a branch (not detached HEAD)
   - `core_knowledge_graph` exists and is a git repository
   - `core_knowledge_graph` has no uncommitted changes
   - `core_knowledge_graph` is on the **same branch** as the main repo

2. **Clone from matching branch**: When creating the worktree, sub-repositories are cloned from the branch that matches the main repo's current branch:
   ```bash
   # If main repo is on 'main', clone core_knowledge_graph from 'main'
   git clone --branch main git@github.com:StochasticAnalytics/core_knowledge_graph.git
   ```

3. **Create feature branch**: After cloning, a new branch is created in the sub-repository matching the worktree's branch name:
   ```bash
   # In the cloned sub-repository
   git checkout -b my_new_feature
   ```

### Example Workflow

Suppose:
- Main repo (`cisTEMx`) is on branch `main`
- `core_knowledge_graph` sub-repo in main directory is also on branch `main`
- You want to create a worktree for feature `my_feature`

The script will:
1. Verify both repos are on `main` and clean
2. Create worktree `worktrees/my_feature` with new branch `my_feature`
3. Clone `core_knowledge_graph` from branch `main`
4. Create and checkout branch `my_feature` in the cloned `core_knowledge_graph`

Result:
- Main repo worktree: on `my_feature` (branched from `main`)
- Sub-repo in worktree: on `my_feature` (branched from `main`)

### Why This Matters

This ensures that:
- Feature branches in sub-repos are based on the correct parent branch
- Development work in worktrees doesn't accidentally mix changes from different branches
- Sub-repositories maintain consistent branching structure with the main repo
- You can work on multiple features in parallel without branch conflicts

## Design Rationale

### Why temporarily unlink symlinks?

`git worktree add` copies tracked files from the main repo. If `.claude` or `CLAUDE.md` were tracked files (not symlinks), they'd be copied. By temporarily unlinking them, we ensure:
- No conflicts during worktree creation
- Clean separation between main repo and worktree
- Ability to recreate symlinks with worktree-specific targets if needed

### Why check for clean git state?

Creating a worktree from a dirty git state can lead to unexpected behavior. The requirement ensures:
- Clear starting point
- No accidental inclusion of uncommitted changes
- Predictable worktree contents

### Why use snake_case for branch names?

Consistency and readability:
- Matches Python/C++ naming conventions common in cisTEMx
- Avoids shell quoting issues
- Easy to type and recognize
- Natural mapping to directory names

## Troubleshooting

### "Branch already exists"
```bash
git branch -d old_branch_name  # or -D to force
```

### "Directory already exists"
```bash
# If it's a stale worktree
git worktree remove /sa_shared/git/cisTEMx/worktrees/dir_name

# Or just remove manually
rm -rf /sa_shared/git/cisTEMx/worktrees/dir_name
```

### "Not a symlink"
If `.claude` or `CLAUDE.md` exist but aren't symlinks:
```bash
# Back them up first if needed
mv .claude .claude.backup
# Then create proper symlink
ln -s core_knowledge_graph/.claude .claude
```

### Submodule clone fails
- Check SSH key is loaded: `ssh-add -l`
- Test SSH access: `ssh -T git@github.com`
- Verify repository URL in script configuration

### "Main repository is in detached HEAD state"
This means the main repo isn't on a branch. Fix by checking out a branch:
```bash
git checkout main  # or whatever branch you want
```

### "core_knowledge_graph has uncommitted changes"
The sub-repo must be clean before creating a worktree:
```bash
cd core_knowledge_graph
git status
# Commit or stash changes
git add .
git commit -m "Save work in progress"
# or
git stash
```

### "core_knowledge_graph is on branch 'X', but main repo is on 'Y'"
The sub-repo must be on the same branch as the main repo:
```bash
cd core_knowledge_graph
git checkout main  # or whatever branch the main repo is on
cd ..
```

### Clone fails with "branch 'X' not found"
The sub-repository doesn't have a branch matching the main repo's current branch. You need to:
1. Create that branch in the sub-repo first, or
2. Switch the main repo to a branch that exists in both repos
