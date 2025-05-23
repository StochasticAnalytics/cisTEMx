"""
Simple module to replace annoying_hack.py for cisTEM test scripts.

This module should be imported at the start of any test script that needs
access to cistem_test_utils. It adds the programs directory to the Python path.

Usage:
    import cistem_path_setup  # Add this as the first import in test scripts
    
    # Then import cistem_test_utils modules as normal
    import cistem_test_utils.args as tmArgs
"""

import os
import sys

# Get the absolute path to the programs directory (parent of this file)
cistem_programs_path = os.path.dirname(os.path.abspath(__file__))

# Add to Python path if not already there
if cistem_programs_path not in sys.path:
    sys.path.append(cistem_programs_path)