#!/bin/bash
#
# claude_notifications.sh - Send push notifications via PushCut API
#
# DEVELOPMENT STATUS: This script is in early development and currently configured
# for a specific development environment. It is not yet production-ready.
#
# REQUIREMENTS:
# - PushCut app (available on Apple App Store: https://www.pushcut.io/)
# - PushCut API key stored in configuration file
#
# CONFIGURATION:
# This script reads the PushCut API key from: /sa_shared/software/.push_cut
#
# The .push_cut file should contain ONLY your PushCut API key (no newlines, quotes, etc.)
# To get an API key:
#   1. Install PushCut from the Apple App Store
#   2. Open the app and go to Account settings
#   3. Generate an API key
#   4. Save the key to the configuration file location
#
# KNOWN ISSUES / TODO:
# TODO: The configuration file path is currently HARD-CODED and specific to one
#       development environment. This path should be configurable via an environment
#       variable (e.g., PUSHCUT_CONFIG_PATH or CLAUDE_NOTIFICATION_CONFIG) to make
#       this script portable across different systems and users.
#
#       For now, users wanting to use this script will need to either:
#       a) Create /sa_shared/software/.push_cut with their API key, OR
#       b) Modify the hard-coded path in this script (lines below)
#
# EXPIRATION NOTICE:
# This script contains a time-bomb mechanism to prevent indefinite use with hard-coded
# paths. See date check below for details.

set -e

# =============================================================================
# TIME BOMB: Hard-coded configuration path check
# =============================================================================
# EXPIRY DATE: 2025-12-07 (30 days from 2025-11-07)
#
# This script will REFUSE TO RUN after the expiry date to force addressing the
# hard-coded configuration path issue. This is intentional technical debt management.
#
# To extend the deadline:
#   1. Update EXPIRY_DATE below (format: YYYY-MM-DD)
#   2. New date should be NO MORE than 30 days in the future
#   3. Add a comment explaining why the extension is needed
#
# The proper fix is to replace the hard-coded path with an environment variable.
# =============================================================================

EXPIRY_DATE="2025-12-07"
CURRENT_DATE=$(date +%Y-%m-%d)

if [[ "$CURRENT_DATE" > "$EXPIRY_DATE" ]]; then
    echo "╔════════════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                            ║"
    echo "║   ⚠️  ERROR: This script has EXPIRED and will not run! ⚠️                 ║"
    echo "║                                                                            ║"
    echo "║   This script contains a HARD-CODED configuration path that must be       ║"
    echo "║   replaced with an environment variable.                                  ║"
    echo "║                                                                            ║"
    echo "║   HARD-CODED PATH: /sa_shared/software/.push_cut                          ║"
    echo "║                                                                            ║"
    echo "║   REQUIRED ACTION:                                                         ║"
    echo "║   Replace the hard-coded path with an environment variable like:          ║"
    echo "║   \${PUSHCUT_CONFIG_PATH:-\$HOME/.config/pushcut/api_key}                   ║"
    echo "║                                                                            ║"
    echo "║   TEMPORARY WORKAROUND (not recommended):                                 ║"
    echo "║   You can extend the deadline by updating EXPIRY_DATE on line 54,         ║"
    echo "║   but it should NOT be more than 30 days in the future.                   ║"
    echo "║                                                                            ║"
    echo "║   Expiry date: $EXPIRY_DATE                                         ║"
    echo "║   Current date: $CURRENT_DATE                                      ║"
    echo "║                                                                            ║"
    echo "╚════════════════════════════════════════════════════════════════════════════╝"
    exit 1
fi

# Check for PushCut API key configuration file
if [ ! -f /sa_shared/software/.push_cut ] ; then
    echo "WARNING: pushcut configuration not found at /sa_shared/software/.push_cut"
    echo "Not sending notification"
    exit 0
fi

is_terminal_visible_here () {
    # Return codes:
    # 0: terminal IS visible (don't send notification)
    # 1: terminal is NOT visible (send notification)
    # 2: cannot determine - tools missing
    # 3: cannot determine - workspace container name not set
    # 4: cannot determine - window not found for workspace container

    # check that wmctrl exists
    if ! command -v wmctrl &> /dev/null; then
        return 2
    fi
    # check that xdotool exists
    if ! command -v xdotool &> /dev/null; then
        return 2
    fi
    # check that workspace container name is set
    if [ -z "$WORKSPACE_CONTAINER_NAME" ]; then
        return 3
    fi

    window_id=$(wmctrl -lpx | grep $WORKSPACE_CONTAINER_NAME  | awk '{print $1}')
    # check if window_id is empty
    if [ -z "$window_id" ]; then
        return 4
    fi
    # use xdotool to get the active desktop for that window
    window_desktop=$(xdotool get_desktop_for_window $window_id)
    # make sure active_desktop is not empty
    if [ -z "$window_desktop" ]; then
        return 4
    fi
    # get the current desktop
    current_desktop=$(xdotool get_desktop)
    # compare the two
    if [ "$window_desktop" -eq "$current_desktop" ]; then
        return 0
    else
        return 1
    fi
}

if [ -n "$NOTIFY_PUSHCUT_SILENT" ]; then
    echo "pushcut is silenced"
    exit 0
fi

# Check if argument is provided
if [ $# -eq 0 ]; then
    echo "WARNING: Usage: $0 TITLE [DESCRIPTION]"
    echo "No arguments provided - not sending notification"
    exit 0
fi

# Set title and text
TITLE="$1"
TEXT="${2:-$1}"  # If text is not provided, use title as text

# Check terminal visibility and handle accordingly
# Temporarily disable set -e to capture the return code
set +e
is_terminal_visible_here
visibility_status=$?
set -e

send_notification=false

case $visibility_status in
    0)
        echo "Terminal is visible - not sending notification"
        ;;
    1)
        echo "Terminal is NOT visible - sending notification"
        send_notification=true
        ;;
    2)
        if [[ "x${PUSHCUT_SEND_ALL}" == "xtrue" ]]; then
            echo "Cannot determine visibility (wmctrl or xdotool not found), but PUSHCUT_SEND_ALL=true - sending notification"
            send_notification=true
        elif [[ -n "${PUSHCUT_SEND_ALL}" ]]; then
            echo "WARNING: PUSHCUT_SEND_ALL='${PUSHCUT_SEND_ALL}' but must be 'true' to override"
            echo "Cannot determine terminal visibility - wmctrl and/or xdotool not found"
            echo "Install with: sudo apt-get install wmctrl xdotool"
            echo "Not sending notification"
            exit 0
        else
            echo "WARNING: Cannot determine terminal visibility - wmctrl and/or xdotool not found"
            echo "Install with: sudo apt-get install wmctrl xdotool"
            echo "Or set PUSHCUT_SEND_ALL=true to send notifications regardless"
            echo "Not sending notification"
            exit 0
        fi
        ;;
    3)
        if [[ "x${PUSHCUT_SEND_ALL}" == "xtrue" ]]; then
            echo "Cannot determine visibility (WORKSPACE_CONTAINER_NAME not set), but PUSHCUT_SEND_ALL=true - sending notification"
            send_notification=true
        elif [[ -n "${PUSHCUT_SEND_ALL}" ]]; then
            echo "WARNING: PUSHCUT_SEND_ALL='${PUSHCUT_SEND_ALL}' but must be 'true' to override"
            echo "Cannot determine terminal visibility - WORKSPACE_CONTAINER_NAME is not set"
            echo "Set WORKSPACE_CONTAINER_NAME environment variable"
            echo "Not sending notification"
            exit 0
        else
            echo "WARNING: Cannot determine terminal visibility - WORKSPACE_CONTAINER_NAME is not set"
            echo "Set WORKSPACE_CONTAINER_NAME environment variable or set PUSHCUT_SEND_ALL=true"
            echo "Not sending notification"
            exit 0
        fi
        ;;
    4)
        if [[ "x${PUSHCUT_SEND_ALL}" == "xtrue" ]]; then
            echo "Cannot determine visibility (window not found for WORKSPACE_CONTAINER_NAME='$WORKSPACE_CONTAINER_NAME'), but PUSHCUT_SEND_ALL=true - sending notification"
            send_notification=true
        elif [[ -n "${PUSHCUT_SEND_ALL}" ]]; then
            echo "WARNING: PUSHCUT_SEND_ALL='${PUSHCUT_SEND_ALL}' but must be 'true' to override"
            echo "Cannot determine terminal visibility - no window found for WORKSPACE_CONTAINER_NAME='$WORKSPACE_CONTAINER_NAME'"
            echo "Check that the workspace container name is correct"
            echo "Not sending notification"
            exit 0
        else
            echo "WARNING: Cannot determine terminal visibility - no window found for WORKSPACE_CONTAINER_NAME='$WORKSPACE_CONTAINER_NAME'"
            echo "Check that the workspace container name is correct or set PUSHCUT_SEND_ALL=true"
            echo "Not sending notification"
            exit 0
        fi
        ;;
    *)
        echo "WARNING: Unexpected return code from is_terminal_visible_here: $visibility_status"
        echo "Not sending notification"
        exit 0
        ;;
esac

if [ "$send_notification" = true ]; then
    # Send notification to Pushcut - using printf to handle quotes properly
    curl -s -X POST "https://api.pushcut.io/$(cat /sa_shared/software/.push_cut)/notifications/TEST1" \
         -H 'Content-Type: application/json' \
         -d "$(printf '{"title":"%s","text":"%s"}' "${TITLE//\"/\\\"}" "${TEXT//\"/\\\"}")"
fi