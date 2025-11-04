#!/bin/bash

cache_dir=${HOME}/.claude/projects/-workspaces-cisTEMx
readarray -t last_file < <(grep -l ':\"Skill' ${cache_dir}/*.jsonl 2>/dev/null)

if [[ ${#last_file[@]} -eq 0 ]] ; then
    echo "No cache files with Skill invocations found"
    exit 0
else
    echo "Found ${#last_file[@]} cache file(s) with Skill invocations:"
    printf '%s\n' "${last_file[@]}"
fi

for file in "${last_file[@]}" ; do
    echo "Working on file: $file"
    first_skill_line=$(grep -n ':\"Skill' "$file" | cut -f1 -d: | head -n 1)
    if [[ -n "$first_skill_line" ]] ; then
        echo "First Skill invocation at line: $first_skill_line"
        # Create backup
        cp "$file" "${file}.bak"
        # Keep only lines before the Skill invocation
        head -n $((first_skill_line - 1)) "${file}.bak" > "$file"
        echo "Deleted lines from $first_skill_line to end of file"
    else
        echo "No Skill line found"
    fi
done
