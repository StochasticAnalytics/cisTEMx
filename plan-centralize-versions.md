# Plan: Centralize Tool Versions via Sourced Shell File + Docker ENV

## Context

After the clang-14 → clang-18 upgrade, version-specific references are scattered across 22+ files. Changing a tool version requires editing scripts, VS Code settings, tasks.json, Dockerfiles, and linting configurations. The solution: a single version-controlled shell file that is sourced during Docker builds, installed into the container for runtime use, and sourced by scripts directly.

**Key constraint**: Avoid `.bashrc` modifications for environment setup — use `/etc/profile.d/` and Docker `ENV` instead.

## Approach

### Step 1: Create `scripts/containers/cistem-versions.sh`

**New file**: `scripts/containers/cistem-versions.sh`

This is the **single source of truth** for all pinned versions and paths:

```bash
#!/bin/bash
# ============================================================================
# cisTEMx Centralized Version and Path Configuration
# Single source of truth — sourced by Dockerfiles, scripts, and shells.
# Installed to /etc/profile.d/ in the container for interactive shells.
# ============================================================================

# --- Container Identity ---
export CISTEM_CONTAINER_REPO=ghcr.io/stochasticanalytics/cistem_build_env
export CISTEM_CONTAINER_VERSION_BASE=3.1.0
export CISTEM_CONTAINER_VERSION_TOP=3.1.2

# --- LLVM/Clang Toolchain ---
# clang-format pinned at 14 to avoid reformatting codebase
# Compiler/clangd/clang-tidy at 18 for CUDA 13 compatibility
export CISTEM_CLANG_FORMAT_VERSION=14
export CISTEM_CLANG_COMPILER_VERSION=18

# --- GCC ---
export CISTEM_GCC_VERSION=11

# --- CUDA ---
export CISTEM_CUDA_VERSION=12.3.2
export CISTEM_CUDA_DRIVER_VERSION=545.23.08
export CISTEM_CUDA_DIR=/usr/local/cuda
export CISTEM_OLDEST_GPU_ARCH=70
export CISTEM_TARGET_GPU_ARCH=86

# --- wxWidgets ---
export CISTEM_WX_DEFAULT_CONFIG=/opt/WX/wx305-clang-static-gtk2/bin/wx-config

# --- Intel MKL ---
export CISTEM_MKLROOT=/opt/intel/oneapi/mkl/latest
export CISTEM_TBBROOT=/opt/intel/oneapi/tbb/latest

# --- Build ---
export CISTEM_BUILD_THREADS=16
```

### Step 2: Source in base Dockerfile + install to `/etc/profile.d/`

**File**: `scripts/containers/base_image/Dockerfile`

```dockerfile
# Copy centralized versions file — single source of truth for all tool versions
COPY cistem-versions.sh /etc/profile.d/cistem-versions.sh

# Make build-time variables available from the versions file
# Docker ENV persists into the running container for non-login shells
# /etc/profile.d/ covers login shells (VS Code terminal, interactive bash)
RUN . /etc/profile.d/cistem-versions.sh && \
    echo "Container configured with CLANG_FORMAT=${CISTEM_CLANG_FORMAT_VERSION}, COMPILER=${CISTEM_CLANG_COMPILER_VERSION}"
```

For Docker `ENV` declarations (needed for non-login shell contexts like VS Code tasks):
- Rather than duplicating values, source the file and use it within each RUN that needs the variables
- Add a small set of critical ENV declarations that tasks.json needs via `${env:VAR}`

```dockerfile
# Critical ENVs for VS Code task variable expansion (${env:CISTEM_*})
# Values must match cistem-versions.sh — update there first, then here
ENV CISTEM_CUDA_DIR=/usr/local/cuda \
    CISTEM_OLDEST_GPU_ARCH=70 \
    CISTEM_TARGET_GPU_ARCH=86 \
    CISTEM_CLANG_FORMAT_VERSION=14 \
    CISTEM_CLANG_COMPILER_VERSION=18 \
    CISTEM_WX_DEFAULT_CONFIG=/opt/WX/wx305-clang-static-gtk2/bin/wx-config
```

Refactor existing Dockerfile to use these in RUN commands:
- Clang install: `. /etc/profile.d/cistem-versions.sh && apt-get install -y clang-format-${CISTEM_CLANG_FORMAT_VERSION} clang-${CISTEM_CLANG_COMPILER_VERSION} ...`
- CUDA install: `ARG CUDA_VER` initialized from the file, or sourced in RUN
- wxWidgets builds: Can reference vars for documentation, but the paths are already parameterized by the build loop

### Step 3: Top layer sources same file + overrides

**File**: `scripts/containers/top_image/Dockerfile`

The top layer can source the same file and override `CISTEM_CONTAINER_VERSION_TOP` if needed:

```dockerfile
# Top layer reads base versions, overrides are set here
ENV CISTEM_CONTAINER_VERSION_TOP=3.1.2
```

The migration install block should source the versions file:
```dockerfile
RUN . /etc/profile.d/cistem-versions.sh && \
    apt-get update && \
    apt-get install -y clang-${CISTEM_CLANG_COMPILER_VERSION} clangd-${CISTEM_CLANG_COMPILER_VERSION} ...
```

### Step 4: Update scripts to source the versions file

Scripts source from the project-relative path (for running outside the container) or the installed path:

```bash
# At the top of each script:
VERSIONS_FILE="${CISTEM_VERSIONS_FILE:-$(git rev-parse --show-toplevel 2>/dev/null)/scripts/containers/cistem-versions.sh}"
[ -f "$VERSIONS_FILE" ] && . "$VERSIONS_FILE"
```

**Files to update**:

| File | Change |
|------|--------|
| `scripts/install_clang_format_hook.sh` | Source versions file; use `$CISTEM_CLANG_FORMAT_VERSION` in hook template |
| `scripts/local_ci.sh` | Source versions file; replace `clang-format-14` with `clang-format-${CISTEM_CLANG_FORMAT_VERSION}` |
| `scripts/linting/cpp_cuda/analyze_blocker.sh` | Replace `clang-tidy-14` with `clang-tidy-${CISTEM_CLANG_COMPILER_VERSION}` |
| `scripts/linting/cpp_cuda/analyze_critical.sh` | Same |
| `scripts/linting/cpp_cuda/analyze_standard.sh` | Same |
| `scripts/linting/cpp_cuda/analyze_deep.sh` | Same |

**Dead code to remove**:
| File | Reason |
|------|--------|
| `scripts/run_clang_format.sh` | Not referenced anywhere — superseded by the pre-commit hook |

### Step 5: Update VS Code settings and tasks

**`.vscode/settings.json`**:
- `"C_Cpp.clang_format_path": "/usr/bin/clang-format-14"` → `"/usr/bin/clang-format"` (use the symlink)

**`.vscode/tasks.json`**:
- Replace `options.env` hardcoded values with ENV references:
  ```json
  "options": {
      "env": {
          "cuda_dir": "${env:CISTEM_CUDA_DIR}",
          "oldest_gpu_arch": "${env:CISTEM_OLDEST_GPU_ARCH}",
          "target_gpu_arch": "${env:CISTEM_TARGET_GPU_ARCH}",
          ...
      }
  }
  ```
- Replace versioned tool references:
  - `run-clang-tidy-14` → `run-clang-tidy-${env:CISTEM_CLANG_COMPILER_VERSION}`
  - `clang-tidy-diff-14.py` → `clang-tidy-diff-${env:CISTEM_CLANG_COMPILER_VERSION}.py`
  - `scan-build-14` → `scan-build-${env:CISTEM_CLANG_COMPILER_VERSION}`

### Step 6: Consolidate CONTAINER_VERSION files

Currently 3 files: `CONTAINER_VERSION_TOP`, `CONTAINER_VERSION_BASE`, `CONTAINER_REPO_NAME`

These are read by `check-version.sh` **before** the container starts (host-side), so they can't come from Docker ENV. Consolidate into a single `.vscode/CONTAINER_CONFIG`:

```
REPO=ghcr.io/stochasticanalytics/cistem_build_env
VERSION_BASE=3.1.0
VERSION_TOP=3.1.2
```

Update `check-version.sh` to parse this single file. Delete the 3 individual files.

The values in `CONTAINER_CONFIG` should match the values in `cistem-versions.sh`. A validation script (or the existing `validate_sync.py` referenced in the pre-push hook) can verify they stay in sync.

## Files Modified

| File | Change |
|------|--------|
| `scripts/containers/cistem-versions.sh` | **NEW** — single source of truth |
| `scripts/containers/base_image/Dockerfile` | COPY + source versions file; add ENV block; refactor clang/cuda installs |
| `scripts/containers/top_image/Dockerfile` | Source versions file in migration block |
| `scripts/install_clang_format_hook.sh` | Source versions; use `$CISTEM_CLANG_FORMAT_VERSION` |
| `scripts/local_ci.sh` | Source versions; use `$CISTEM_CLANG_FORMAT_VERSION` |
| `scripts/linting/cpp_cuda/analyze_*.sh` (4 files) | Source versions; use `$CISTEM_CLANG_COMPILER_VERSION` |
| `scripts/run_clang_format.sh` | **DELETE** — dead code |
| `.vscode/settings.json` | clang-format path → use symlink |
| `.vscode/tasks.json` | Use `${env:CISTEM_*}` for tool versions and paths |
| `.vscode/CONTAINER_CONFIG` | **NEW** — consolidates 3 version files |
| `.vscode/CONTAINER_VERSION_TOP` | **DELETE** — merged into CONTAINER_CONFIG |
| `.vscode/CONTAINER_VERSION_BASE` | **DELETE** — merged into CONTAINER_CONFIG |
| `.vscode/CONTAINER_REPO_NAME` | **DELETE** — merged into CONTAINER_CONFIG |
| `.devcontainer/check-version.sh` | Parse new CONTAINER_CONFIG format |
| `.devcontainer/devcontainer.json` | Update `initializeCommand` if needed |

## Verification

1. **Container build**: `docker build` top layer — cistem-versions.sh is sourced, clang installed with correct version
2. **ENV in container**: `env | grep CISTEM_` shows all variables
3. **Scripts**: `scripts/install_clang_format_hook.sh` — generates hook using `$CISTEM_CLANG_FORMAT_VERSION`
4. **VS Code tasks**: Run "Build with Bear" — resolves `${env:CISTEM_*}` variables
5. **Version check**: `check-version.sh` parses consolidated `CONTAINER_CONFIG`
6. **clang-format**: VS Code format-on-save uses `/usr/bin/clang-format` symlink

## Out of Scope (future)

- wxWidgets path parameterization in tasks.json configure commands
- c_cpp_properties.json parameterization
- MKL documentation extraction from Dockerfile
- `install_clang_format_hook.sh` hook template: the generated hook embeds version values at install time (heredoc), so the hook itself doesn't need to source the file at runtime — it bakes in the version from install time

---

# Part 2: Claude Code Persistent Configuration

## Context

Claude Code stores its runtime state (plans, sessions, memory, credentials) at `~/.claude/`. In this devcontainer, `~` is ephemeral — lost on container rebuild. Plans, memory, and session history disappear between sessions. The fix: redirect `CLAUDE_CONFIG_DIR` to a persistent, version-controlled location on the bind-mounted `/sa_shared/` volume, and set Claude Code environment variables via `remoteEnv` in devcontainer.json.

Additionally, plan output should always go to the project-local `.claude/cache/` as reviewable markdown files, never dumped to the terminal.

## Step 7: Create persistent git repo for Claude Code config

**New repo**: `/sa_shared/git/claude_code_config`

```bash
mkdir -p /sa_shared/git/claude_code_config
cd /sa_shared/git/claude_code_config
git init
```

**`.gitignore`** — track plans/memory, ignore secrets and ephemeral data:

```gitignore
# Secrets — NEVER commit
.credentials.json

# Large ephemeral session data
debug/
telemetry/
cache/
paste-cache/
shell-snapshots/
session-env/
ide/
file-history/
todos/
tasks/
downloads/
history.jsonl

# Session transcripts (can be 1-2MB+ each)
projects/*/*.jsonl
projects/*/subagents/
projects/*/tool-results/

# Keep auto-memory files (the whole point of version controlling this)
!projects/*/memory/
!projects/*/memory/**
```

**`README.md`**:

```markdown
# Claude Code Configuration

Persistent storage for Claude Code runtime state, redirected via `CLAUDE_CONFIG_DIR`.

## What's tracked
- `plans/` — Plan mode markdown files
- `projects/*/memory/` — Auto-memory files (MEMORY.md, topic files)
- `plugins/` — Plugin configurations

## What's ignored
- Credentials, session transcripts, debug logs, telemetry
- See .gitignore for full list

## Setup
Set `CLAUDE_CONFIG_DIR=/sa_shared/git/claude_code_config` in the devcontainer environment.
This is configured via `remoteEnv` in `.devcontainer/devcontainer.json`.
```

## Step 8: Set Claude Code environment variables

**File**: `.devcontainer/devcontainer.json`

Add Claude Code env vars to the existing `remoteEnv` block:

```json
"remoteEnv": {
    "WORKSPACE_BASENAME": "${localWorkspaceFolderBasename}",
    "WORKSPACE_CONTAINER_NAME": "cisTEMx-${localEnv:USER}-${localWorkspaceFolderBasename}",
    "CLAUDE_CONFIG_DIR": "/sa_shared/git/claude_code_config",
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "0",
    "CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY": "1",
    "DISABLE_TELEMETRY": "1"
}
```

**Why `remoteEnv`**: These are evaluated before any process starts in the container. Unlike Dockerfile `.bashrc` additions, they're available to all processes (not just interactive shells) and they're workspace-scoped — different devcontainer projects could use different config dirs.

**Vars NOT set** (using defaults):
- `CLAUDE_CODE_TMPDIR` — default `/tmp` is fine (ephemeral is correct for temp files)
- `CLAUDE_CODE_ENABLE_TELEMETRY` — left unset (off by default)
- `CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS` — default token limit

## Step 9: Add `clc` alias with plan-location rules

**File**: `scripts/containers/top_image/Dockerfile`

Add alongside existing `claude` and `ccc` aliases (after line 157):

```dockerfile
RUN echo "alias clc='claude --verbose --append-system-prompt \"Plans are ALWAYS written as markdown files to the project .claude/cache/ directory. NEVER output plan content directly to the terminal. When sharing a plan, provide ONLY the clickable file path to the .claude/cache/ plan file for review and inline feedback.\"'" >> /home/cisTEMdev/.bashrc
```

This appends to the system prompt without replacing Claude Code's built-in instructions. The `--verbose` flag is included since the existing `claude` alias already uses it.

**Note**: The existing `alias claude='claude --verbose'` (line 133 of Dockerfile) remains unchanged. `clc` is a separate alias that adds the plan-location system prompt on top.

## Step 10: Migrate existing plan to new config dir

Copy the current plan from the ephemeral location to the persistent repo:

```bash
mkdir -p /sa_shared/git/claude_code_config/plans
cp /workspaces/cisTEMx/.claude/cache/plan-centralize-versions.md /sa_shared/git/claude_code_config/plans/
```

Also ensure the project-level auto-memory directory structure exists:

```bash
mkdir -p /sa_shared/git/claude_code_config/projects/-workspaces-cisTEMx/memory
```

## Files Modified (Part 2)

| File | Change |
|------|--------|
| `/sa_shared/git/claude_code_config/` | **NEW** — git repo for persistent Claude Code config |
| `/sa_shared/git/claude_code_config/.gitignore` | **NEW** — track plans/memory, ignore secrets/ephemeral |
| `/sa_shared/git/claude_code_config/README.md` | **NEW** — repo documentation |
| `.devcontainer/devcontainer.json` | Add Claude Code env vars to `remoteEnv` block |
| `scripts/containers/top_image/Dockerfile` | Add `clc` alias with `--append-system-prompt` |

## Verification (Part 2)

1. **Config dir**: After restart, `echo $CLAUDE_CONFIG_DIR` → `/sa_shared/git/claude_code_config`
2. **Auto memory**: `echo $CLAUDE_CODE_DISABLE_AUTO_MEMORY` → `0`
3. **Plans persist**: Create a plan, rebuild container, verify plan file still exists at `/sa_shared/git/claude_code_config/plans/`
4. **Git tracking**: `cd /sa_shared/git/claude_code_config && git status` shows plans/ and memory/ tracked
5. **clc alias**: Run `clc` in terminal, verify system prompt includes plan-location rules
6. **No surveys**: Verify "How is Claude doing?" surveys don't appear

## Implementation Order

Parts 1 and 2 can be implemented independently. Suggested order:

1. **Part 2 first** (Steps 7-10) — immediate quality-of-life improvement, no container rebuild needed for the git repo + alias
2. **Part 1 later** (Steps 1-6) — requires container rebuild for full version centralization

Note: Step 8 (devcontainer.json `remoteEnv` changes) only take effect on next container rebuild or reopen. The git repo (Step 7) and plan migration (Step 10) are immediately usable.
