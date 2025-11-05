"""
Shared utilities for Claude Code skills.

This module provides common functionality used across multiple skills,
particularly for path validation and security.
"""

from .path_validation import (
    validate_path_within_project,
    validate_build_dir_in_gitignore,
    find_git_root
)

__all__ = [
    'validate_path_within_project',
    'validate_build_dir_in_gitignore',
    'find_git_root'
]
