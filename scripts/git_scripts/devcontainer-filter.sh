#!/usr/bin/env bash


# We need cistem_mounts variable defined in bashrc, but the env is not passed so we check for the file, then parse it
if [ -f ${HOME}/.bashrc ] ; then
    # get the line that defines cistem_mounts
    cistem_mounts_line=$(grep '^cistem_mounts=' ${HOME}/.bashrc)
    # if the line is not empty evaluate it to define the variable
    if [ -n "$cistem_mounts_line" ] ; then
        eval $cistem_mounts_line
    fi
fi


# Usage: 
# For smudge: .git/filters/devcontainer-filter.sh smudge
# For clean:  .git/filters/devcontainer-filter.sh clean



MODE=${1}


begin_tag="// begin_mounts"
end_tag="// end_mounts"
# Add a couple simple functions to check for the begin and end of mounts block tags
check_begin_mounts() {
    # make this insensitive to leading spaces
    if echo "$1" | grep -q "[[:space:]]*${begin_tag}"; then
        return 0
    else
        return 1
    fi
}
check_end_mounts() {
    if echo "$1" | grep -q "[[:space:]]*${end_tag}"; then
        return 0
    else
        return 1
    fi
}

if [ "$MODE" = "smudge" ]; then
    # Early return: if cistem_mounts not set, just pass through unchanged
    if [ -v "${cistem_mounts}" ]; then
        cat
        exit 0
    fi
    
    # Smudge: Replace // mount with actual mounts
    in_ mounts=false
    while IFS= read -r line; do
        if check_begin_mounts "$line"; then
            in_mounts=true
            # Start of mounts array was empyty mounts : [] set to open the bracket
            echo "    ${begin_tag}"
            echo "    \"mounts\": ["
            # Now insert the mounts from the cistem_mounts array
            first_mount=true # no comma
            for mount in ${cistem_mounts[@]}; do
                if [ "$first_mount" = true ] ; then
                    first_mount=false
                    echo "        \"$mount\","
                else
                    echo "        \"$mount\""
                fi
            done
            echo "    ],"
        else
            if [ $in_mounts == true ] ; then
                if check_end_mounts "$line"; then
                    echo "    ${end_tag}"
                fi
                in_mounts=false
                # Skip all lines in mounts block
                continue
            fi
            echo "$line"
        fi
    done
    
elif [ "$MODE" = "clean" ]; then
    
    in_mounts=false
    while IFS= read -r line; do
        if [ $in_mounts = true ] ; then
            # We are in mounts block, check for end
            if check_end_mounts "$line"; then
                in_mounts=false
                echo "    ${end_tag}"
            fi
            # Skip all lines in mounts block
            continue
        fi

        if check_begin_mounts "$line"; then
            # Start of mounts array - replace with empty mounts placeholder
            echo "    ${begin_tag}"
            echo "    \"mounts\" : [],"  
            in_mounts=true
        else
            echo "$line"
        fi
    done
else
 echo "$1 not recognized"  >> /tmp/git-filter-debug.log
fi
