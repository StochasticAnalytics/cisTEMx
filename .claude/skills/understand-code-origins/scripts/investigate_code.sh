#!/bin/bash
# Automated code investigation script
# Combines blame, pickaxe, and log to understand code origins
#
# Usage: ./investigate_code.sh <file> [search_term]

set -e

FILE="${1:?Usage: $0 <file> [search_term]}"
SEARCH_TERM="${2:-}"

echo "==================================="
echo "Code Investigation: $FILE"
echo "==================================="
echo

# Check if file exists
if [ ! -f "$FILE" ]; then
    echo "Error: File $FILE not found"
    exit 1
fi

# Function to show section header
section() {
    echo
    echo "--- $1 ---"
    echo
}

# 1. Basic blame (who currently owns the code)
section "Current Ownership (blame -w)"
git blame -w "$FILE" | head -20
echo "... (showing first 20 lines)"
echo

# 2. Blame with move detection
section "With Move Detection (blame -wCCC)"
git blame -wCCC "$FILE" | head -20
echo "... (showing first 20 lines)"
echo

# 3. If search term provided, investigate it
if [ -n "$SEARCH_TERM" ]; then
    section "Searching for: $SEARCH_TERM"

    # Pickaxe search
    echo "When was it added/removed? (log -S)"
    git log -S"$SEARCH_TERM" --oneline -- "$FILE" | head -10

    echo
    echo "Any modifications? (log -G)"
    git log -G"$SEARCH_TERM" --oneline -- "$FILE" | head -10

    # Show in current file
    echo
    echo "Current occurrences:"
    git blame -wCCC "$FILE" | grep -i "$SEARCH_TERM" | head -5
fi

# 4. File history summary
section "File History Summary"
echo "Total commits: $(git log --oneline -- "$FILE" | wc -l)"
echo "First commit: $(git log --reverse --oneline -- "$FILE" | head -1)"
echo "Last commit: $(git log --oneline -- "$FILE" | head -1)"
echo

# 5. Top contributors
section "Top 5 Contributors"
git shortlog -sn -- "$FILE" | head -5
echo

# 6. Recent activity
section "Last 10 Commits"
git log --oneline --date=short --format="%h %ad %an %s" -- "$FILE" | head -10
echo

# 7. Check for renames
section "Rename History"
if git log --follow --name-status --oneline -- "$FILE" | grep -q "^R"; then
    echo "File was renamed:"
    git log --follow --name-status --oneline -- "$FILE" | grep "^R"
else
    echo "No renames detected"
fi
echo

echo "==================================="
echo "Investigation complete!"
echo "==================================="
echo
echo "Next steps:"
echo "  - For detailed commit: git show <commit-hash>"
echo "  - For function history: git log -L :function:$FILE"
echo "  - For interactive drilling: git gui blame $FILE"
