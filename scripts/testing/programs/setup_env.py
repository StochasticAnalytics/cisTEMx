#!/usr/bin/env python3
"""
Environmental setup utility for cisTEM test scripts.

This script modifies the Python sys.path to include the necessary directories
so that test scripts can find and import cistem_test_utils without requiring
annoying_hack.py or special environment variables.
"""

import os
import sys

def setup_cistem_env():
    """
    Add the necessary directories to sys.path for cisTEM testing.
    
    This function is designed to be imported and called at the beginning of
    test scripts to ensure all the required modules are available.
    """
    # Get the absolute path to the programs directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check if the directories are already in sys.path
    if script_dir not in sys.path:
        sys.path.append(script_dir)

    return True

if __name__ == "__main__":
    # If run directly, just report success
    if setup_cistem_env():
        print("cisTEM environment setup successfully.")
    else:
        print("Failed to set up cisTEM environment.")
        sys.exit(1)