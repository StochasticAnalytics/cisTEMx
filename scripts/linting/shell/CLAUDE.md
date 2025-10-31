# Shell Script Linting

Static analysis and linting for shell scripts in cisTEMx.

## Status

✅ **Active - Basic shellcheck integration implemented**

Minimal shellcheck linting is now available for shell script validation.

## Planned Tools

### shellcheck

**Purpose:** Static analysis for shell scripts
**Website:** <https://www.shellcheck.net/>

**What it detects:**

- Common scripting errors (unquoted variables, etc.)
- Portability issues (bashisms in sh scripts)
- Security vulnerabilities (injection risks)
- Deprecated syntax
- Performance anti-patterns

**Example usage:**

```bash
# Check single script
shellcheck scripts/regenerate_project.b

# Check all scripts in directory
find scripts/ -name "*.sh" -exec shellcheck {} +

# With specific shell dialect
shellcheck -s bash scripts/build_helper.sh
```

### shfmt

**Purpose:** Shell script formatter
**Website:** <https://github.com/mvdan/sh>

**What it does:**

- Consistent indentation (2 or 4 spaces)
- Standardized spacing
- Simplified syntax where possible

**Example usage:**

```bash
# Format in-place
shfmt -w scripts/*.sh

# Check formatting (CI)
shfmt -d scripts/

# With specific indent
shfmt -i 4 -w scripts/*.sh
```

## Scope

Shell scripts to be linted:

- **`scripts/regenerate_project.b`** - Autotools regeneration
- **`scripts/regenerate_containers.sh`** - Docker container rebuild
- **`scripts/testing/`** - Test runner scripts
- **`scripts/build/`** - Build helper scripts
- **Container scripts** - `scripts/containers/*/install_*.sh`
- **Linting scripts** - `scripts/linting/cpp_cuda/*.sh`

## Planned Configuration

### `.shellcheckrc`

Located at project root:

```bash
# Disable checks that conflict with cisTEMx patterns
disable=SC2086  # Allow word splitting for build flags
disable=SC2046  # Allow word splitting in command substitution

# Set shell dialect
shell=bash

# Source paths for external scripts
source-path=SCRIPTDIR
```

### VS Code Tasks (Planned)

```json
{
  "label": "Lint Shell Scripts (shellcheck)",
  "type": "shell",
  "command": "find scripts/ -name '*.sh' -exec shellcheck {} +",
  "problemMatcher": {
    "owner": "shellcheck",
    "pattern": {
      "regexp": "^(.*):(\\d+):(\\d+):\\s+(warning|error):\\s+(.*)$",
      "file": 1,
      "line": 2,
      "column": 3,
      "severity": 4,
      "message": 5
    }
  }
}
```

## Planned Workflow

**Pre-commit:**

```bash
# Quick check on modified scripts
git diff --name-only --cached | grep '\.sh$' | xargs shellcheck
```

**CI Integration:**

```bash
# Check all scripts
find scripts/ -name '*.sh' -o -name '*.b' | xargs shellcheck

# Check formatting
shfmt -d scripts/
```

## Common Shell Issues to Detect

**Unquoted variables:**

```bash
# BAD
for file in $FILES; do

# GOOD
for file in "$FILES"; do
```

**Missing error checking:**

```bash
# BAD
cd /some/path
rm -rf *

# GOOD
cd /some/path || exit 1
rm -rf ./*
```

**Useless use of cat:**

```bash
# BAD
cat file.txt | grep pattern

# GOOD
grep pattern file.txt
```

**Portability:**

```bash
# BAD (bashism in sh script)
#!/bin/sh
if [[ -f file ]]; then

# GOOD
#!/bin/sh
if [ -f file ]; then
```

## Installation (Future)

Add to `scripts/containers/top_image/Dockerfile`:

```dockerfile
# Shell script linting tools
RUN apt-get update && apt-get install -y \
    shellcheck \
    && rm -rf /var/lib/apt/lists/*

# Install shfmt (binary release)
RUN wget -O /usr/local/bin/shfmt \
    https://github.com/mvdan/sh/releases/download/v3.7.0/shfmt_v3.7.0_linux_amd64 \
    && chmod +x /usr/local/bin/shfmt
```

## Next Steps

1. Install shellcheck and shfmt in development container
2. Create `.shellcheckrc` configuration
3. Run baseline analysis on existing scripts
4. Fix critical issues
5. Add VS Code tasks
6. Create pre-commit hook
7. Integrate into CI

## References

- [ShellCheck Wiki](https://github.com/koalaman/shellcheck/wiki)
- [ShellCheck Gallery of Bad Code](https://github.com/koalaman/shellcheck/blob/master/README.md#gallery-of-bad-code)
- [shfmt Documentation](https://github.com/mvdan/sh)
- [Google Shell Style Guide](https://google.github.io/styleguide/shellguide.html)

---

**Last Updated:** 2025-10-04
**Implementation Priority:** Medium (after C++/CUDA linting stabilizes)
