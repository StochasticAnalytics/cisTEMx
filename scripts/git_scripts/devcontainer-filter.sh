#!/bin/bash

# Usage: 
# For smudge: .git/filters/devcontainer-filter.sh smudge
# For clean:  .git/filters/devcontainer-filter.sh clean



MODE=${1}


if [ "$MODE" = "smudge" ]; then
    # Early return: if cistem_mounts not set, just pass through unchanged

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
    in_mounts=false
    while IFS= read -r line; do
        if echo "$line" | grep -q '"mounts"[[:space:]]*:[[:space:]]*\['; then
            # Start of mounts array - replace with placeholder
            # only works for single lines
            echo "    // mounts"
        else
            echo "$line"
        fi
    done
else
 echo "$1 not recognized"  >> /tmp/git-filter-debug.log
fi
