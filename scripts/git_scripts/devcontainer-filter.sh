#!/bin/bash


#!/bin/bash
echo "FILTER: $1" >> /tmp/git-filter-debug.log
cat -
exit 0


# Usage: 
# For smudge: .git/filters/devcontainer-filter.sh smudge
# For clean:  .git/filters/devcontainer-filter.sh clean

MODE=${1}

echo "Here"
echo "1 is $1"

if [ "$MODE" = "smudge" ]; then
    # Early return: if cistem_mounts not set, just pass through unchanged
    echo "smudging mounts"
    if [ -z "${cistem_mounts}" ]; then
        cat
        exit 0
    fi
    
    # Smudge: Replace // mount with actual mounts
    while IFS= read -r line; do
        if echo "$line" | grep -q "^[[:space:]]*// mount"; then
            # Output the mount line with the env variable value
            echo "    \"mounts\": [${cistem_mounts}],"
        else
            echo "$line"
        fi
    done
    
elif [ "$MODE" = "clean" ]; then
    # Clean: Replace mounts array with // mount placeholder
    echo "Cleaning mounts"
    in_mounts=false
    while IFS= read -r line; do
        if echo "$line" | grep -q '"mounts"[[:space:]]*:[[:space:]]*\['; then
            # Start of mounts array - replace with placeholder
            echo "    // mounts"
            in_mounts=true
            # Check if it's a single-line array
            if echo "$line" | grep -q '\]'; then
                in_mounts=false
            fi
        elif [ "$in_mounts" = true ]; then
            # Skip lines inside mounts array
            if echo "$line" | grep -q '\]'; then
                in_mounts=false
            fi
        else
            echo "$line"
        fi
    done
fi
