#!/usr/bin/env python3
"""
Path validation utilities for skill scripts.

Provides security-focused path validation to prevent:
- Path traversal attacks
- Writing outside project boundaries
- Git pollution from build artifacts

Developed in response to lab tech security review (Nov 2025).
"""

import re
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple


def find_git_root() -> Optional[Path]:
    """
    Find the git project root directory.

    Returns:
        Path to git root, or None if not in a git repository
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True
        )
        return Path(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def validate_path_within_project(
    path: Path,
    git_root: Optional[Path] = None,
    path_description: str = "Path"
) -> Tuple[bool, str]:
    """
    Validate that a path is within project boundaries.

    Prevents path traversal attacks and accidental operations outside project.

    Args:
        path: Path to validate
        git_root: Project root (auto-detected if None)
        path_description: Human-readable description for error messages

    Returns:
        Tuple of (is_valid, error_message)
        - (True, "") if valid
        - (False, error_message) if invalid

    Example:
        >>> valid, error = validate_path_within_project(
        ...     Path("/project/build/debug"),
        ...     Path("/project"),
        ...     "Build directory"
        ... )
        >>> if not valid:
        ...     print(error, file=sys.stderr)
        ...     return 1
    """
    if git_root is None:
        git_root = find_git_root()
        if git_root is None:
            return False, f"Error: Not in a git repository"

    try:
        # Resolve both paths to handle symlinks and relative paths
        resolved_path = path.resolve()
        resolved_root = git_root.resolve()

        # Check if path is within project
        resolved_path.relative_to(resolved_root)
        return True, ""

    except ValueError:
        error_msg = f"Error: {path_description} must be within project directory\n"
        error_msg += f"  Path: {path}\n"
        error_msg += f"  Project root: {git_root}"
        return False, error_msg


def validate_no_path_traversal(path_component: str) -> Tuple[bool, str]:
    """
    Validate that a path component doesn't contain traversal sequences.

    Checks for common path traversal patterns:
    - '..' (parent directory)
    - Absolute paths (starting with '/')
    - Hidden traversal attempts

    Args:
        path_component: String to validate (e.g., build subdirectory name)

    Returns:
        Tuple of (is_valid, error_message)

    Example:
        >>> valid, error = validate_no_path_traversal("../../etc")
        >>> assert not valid

        >>> valid, error = validate_no_path_traversal("debug-build")
        >>> assert valid
    """
    if not path_component:
        return False, "Error: Path component cannot be empty"

    if '..' in path_component:
        return False, f"Error: Path traversal detected: '{path_component}' contains '..'"

    if path_component.startswith('/'):
        return False, f"Error: Absolute path not allowed: '{path_component}'"

    # Check for encoded traversal attempts
    if '%2e%2e' in path_component.lower():
        return False, f"Error: Encoded path traversal detected: '{path_component}'"

    return True, ""


def validate_build_dir_in_gitignore(
    build_dir: Path,
    git_root: Optional[Path] = None
) -> Tuple[bool, str]:
    """
    Validate that a build directory is covered by .gitignore.

    Prevents git pollution from misparse of build directory paths.

    Args:
        build_dir: Build directory path to check
        git_root: Project root (auto-detected if None)

    Returns:
        Tuple of (is_covered, warning_message)
        - (True, "") if covered by .gitignore
        - (False, warning) if not covered

    Note:
        This is a heuristic check, not exhaustive. It checks for common
        patterns like 'build/' but may not catch all .gitignore rules.

    Example:
        >>> covered, warning = validate_build_dir_in_gitignore(
        ...     Path("/project/build/debug-typo")
        ... )
        >>> if not covered:
        ...     print(f"WARNING: {warning}")
    """
    if git_root is None:
        git_root = find_git_root()
        if git_root is None:
            return False, "Cannot verify: not in git repository"

    gitignore_path = git_root / ".gitignore"
    if not gitignore_path.exists():
        return False, f".gitignore not found at {gitignore_path}"

    try:
        with open(gitignore_path, 'r') as f:
            gitignore_patterns = f.read().splitlines()
    except Exception as e:
        return False, f"Could not read .gitignore: {e}"

    # Get relative path from git root
    try:
        rel_path = build_dir.resolve().relative_to(git_root.resolve())
    except ValueError:
        return False, f"Build directory outside project: {build_dir}"

    # Check if path matches any .gitignore pattern
    # This is a simplified check - doesn't handle all gitignore complexity
    path_str = str(rel_path)
    path_parts = path_str.split('/')

    for pattern in gitignore_patterns:
        pattern = pattern.strip()

        # Skip comments and empty lines
        if not pattern or pattern.startswith('#'):
            continue

        # Remove leading/trailing slashes for comparison
        pattern = pattern.strip('/')

        # Check for exact match or prefix match
        if pattern == path_str:
            return True, ""

        if pattern.endswith('/') and path_str.startswith(pattern):
            return True, ""

        # Check if any path component matches (e.g., 'build/' matches 'build/foo')
        if path_str.startswith(pattern + '/'):
            return True, ""

        # Check for wildcard patterns (basic support)
        if '*' in pattern:
            regex_pattern = pattern.replace('*', '.*')
            if re.match(regex_pattern, path_str):
                return True, ""

    warning = f"Build directory may not be in .gitignore: {rel_path}\n"
    warning += "  This could lead to accidentally committing build artifacts.\n"
    warning += "  Common patterns to add:\n"
    warning += f"    {path_parts[0]}/  (if consistent build directory)\n"
    warning += f"    {rel_path}/  (specific to this directory)"

    return False, warning


def validate_build_directory(
    build_subdir: str,
    git_root: Optional[Path] = None,
    strict: bool = True
) -> Tuple[bool, Path, str]:
    """
    Comprehensive validation for build directory from configuration.

    Combines multiple validation checks for build directories:
    - No path traversal
    - Within project bounds
    - Covered by .gitignore

    Args:
        build_subdir: Build subdirectory name from configuration
        git_root: Project root (auto-detected if None)
        strict: If True, .gitignore check is required; if False, it's a warning

    Returns:
        Tuple of (is_valid, build_path, error_or_warning_message)
        - (True, path, "") if fully valid
        - (True, path, warning) if valid but with gitignore warning
        - (False, None, error) if invalid

    Example:
        >>> valid, path, msg = validate_build_directory("debug-build")
        >>> if not valid:
        ...     print(msg, file=sys.stderr)
        ...     return 1
        >>> if msg:  # Warning
        ...     print(f"WARNING: {msg}", file=sys.stderr)
    """
    if git_root is None:
        git_root = find_git_root()
        if git_root is None:
            return False, None, "Error: Not in a git repository"

    # Check for path traversal
    valid, error = validate_no_path_traversal(build_subdir)
    if not valid:
        return False, None, error

    # Construct full build path
    build_path = git_root / "build" / build_subdir

    # Validate within project
    valid, error = validate_path_within_project(
        build_path,
        git_root,
        "Build directory"
    )
    if not valid:
        return False, None, error

    # Check .gitignore coverage
    covered, warning = validate_build_dir_in_gitignore(build_path, git_root)

    if not covered:
        if strict:
            return False, None, warning
        else:
            return True, build_path, warning  # Valid but with warning

    return True, build_path, ""


if __name__ == "__main__":
    # Quick self-test
    print("Testing path validation utilities...")

    # Test 1: Find git root
    root = find_git_root()
    print(f"✓ Git root: {root}")

    # Test 2: Path traversal detection
    valid, _ = validate_no_path_traversal("../../etc")
    assert not valid, "Should detect path traversal"
    valid, _ = validate_no_path_traversal("debug-build")
    assert valid, "Should allow normal directory name"
    print("✓ Path traversal detection working")

    # Test 3: Within project validation
    if root:
        valid, _ = validate_path_within_project(root / "build" / "debug", root)
        assert valid, "Should allow paths within project"
        print("✓ Within-project validation working")

    print("\nAll self-tests passed!")
