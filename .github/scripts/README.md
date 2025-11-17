# CI/CD Scripts

This directory contains scripts used by CI/CD workflows and development tooling.

## Sync Validation System

The sync validation system ensures hardcoded values across the codebase stay synchronized with their source of truth.

### Overview

Some values in the codebase must be hardcoded in multiple locations (e.g., container versions in CI workflows and devcontainer configs). GitHub Actions doesn't support dynamic container images, and other constraints make it impractical to reference these values programmatically everywhere.

The sync validator:

1. Maintains a **rules database** (JSON) describing WHERE source and target values are, and HOW to extract them
2. **Does NOT store the actual values** - only the structural metadata (file paths and regex patterns)
3. Reads current values from source files and compares to target locations
4. Reports mismatches and fails if values are out of sync

### Files

- **`sync_validation_rules.json`** - Rules database describing synchronization relationships
- **`validate_sync.py`** - Python script that performs validation
- **`.github/workflows/validate_sync.yml`** - CI workflow that runs validation on every push/PR

### Running Validation

**Automatic:**

- Runs in pre-push hook (installed by `./regenerate_project.sh`)
- Runs in CI on every push and pull request

**Manual:**

```bash
python3 .github/scripts/validate_sync.py
```

Exit codes:

- `0` - All values in sync
- `1` - One or more values out of sync
- `2` - Configuration or file read error

### Adding New Rules

To add a new synchronization rule, edit `sync_validation_rules.json`:

```json
{
  "rules": [
    {
      "name": "rule_identifier",
      "description": "Human-readable description of what must stay in sync",
      "source": {
        "file": "path/to/source/file",
        "pattern": "regex_pattern_with_(capture_group)",
        "capture_group": 1
      },
      "targets": [
        {
          "file": "path/to/target/file",
          "pattern": "regex_pattern_with_(capture_group)",
          "capture_group": 1,
          "context": "Description of where this appears"
        }
      ]
    }
  ]
}
```

**Key principles:**

- **Source** is the single source of truth for the value
- **Targets** are locations that must match the source
- **Patterns** are Python regex with capture groups to extract values
- **No hardcoded values** - rules describe structure only, validator extracts current values

### Example

The `container_version_top` rule ensures the container version in `.vscode/CONTAINER_VERSION_TOP` matches:

- CI workflow container image (`.github/workflows/run_builds.yml`)
- Devcontainer image (`.devcontainer/devcontainer.json`)

When the source version changes to `3.0.3`, targets must be updated manually, and the validator will catch if any are missed.

### Regex Pattern Tips

- Use raw strings in JSON: `"pattern": "version: v(.+)"`
- Escape special regex characters: `.` becomes `\\.`
- Use capture groups to extract values: `"pattern": "version: v(.+)"` captures everything after `v`
- Set `capture_group` to match your parentheses (1-indexed)
- Test patterns with actual file content to ensure they match

### Design Philosophy

This system embodies a key principle: **when you can't make it dynamic, make it validated**.

Rather than fight constraints (GitHub Actions, JSON configs, etc.) that prevent dynamic value references, we:

1. Accept that some values must be hardcoded
2. Document the relationships in machine-readable form
3. Validate automatically that changes propagate correctly
4. Fail fast if synchronization breaks

This approach scales to many "source of truth → multiple dependents" patterns throughout the codebase.
