#!/usr/bin/env bash
# Run shellcheck on shell scripts
# Usage: ./analyze_shell.sh [path]
# Default: analyze all .sh files in project

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TARGET_PATH="${1:-$PROJECT_ROOT}"

cd "$PROJECT_ROOT"

# Check if shellcheck is installed
if ! command -v shellcheck &> /dev/null; then
    echo "Error: shellcheck is not installed"
    echo "Install with: apt-get install shellcheck"
    exit 1
fi

echo "=== Shell Script Analysis with shellcheck ==="
echo "Target: $TARGET_PATH"
echo "Config: .shellcheckrc"
echo ""

# Find all shell scripts
# Look for .sh files and files with #!/bin/bash or #!/usr/bin/env bash shebang
find "$TARGET_PATH" -type f -name "*.sh" -print0 | while IFS= read -r -d '' script; do
    echo "Checking: $script"
    shellcheck "$script" || true
done

echo ""
echo "=== shellcheck analysis complete ==="
