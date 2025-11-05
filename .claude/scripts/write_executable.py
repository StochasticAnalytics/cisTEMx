#!/usr/bin/env python3
"""Write file with executable permissions atomically.

This utility creates files with execute permissions set from creation,
avoiding the two-step approval process of Write + chmod.

Usage:
    echo "content" | python write_executable.py <path>
    cat file | python write_executable.py <path>
    python write_executable.py <path> < input.txt
"""

import os
import sys
from pathlib import Path


def write_executable(path: str, content: str, mode: int = 0o755) -> None:
    """
    Write file with specific permissions atomically.

    Args:
        path: Path to file to create
        content: File content to write
        mode: Unix permissions (default: 0o755 = rwxr-xr-x)
    """
    path_obj = Path(path)

    # Ensure parent directory exists
    if path_obj.parent != Path('.'):
        path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Check if file already exists
    file_existed = path_obj.exists()

    # Create/overwrite file with proper permissions
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    umask_original = os.umask(0)
    try:
        fd = os.open(path, flags, mode)
        with os.fdopen(fd, 'w') as f:
            f.write(content)
    finally:
        os.umask(umask_original)

    # If file existed, os.open() doesn't reset permissions, so chmod explicitly
    if file_existed:
        os.chmod(path, mode)


def main() -> int:
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: write_executable.py <path> [mode]", file=sys.stderr)
        print("Content read from stdin", file=sys.stderr)
        print("Optional mode in octal (default: 755)", file=sys.stderr)
        print("\nExample:", file=sys.stderr)
        print('  cat script.sh | python write_executable.py path/to/script.sh', file=sys.stderr)
        return 1

    path = sys.argv[1]

    # Optional mode argument
    mode = 0o755
    if len(sys.argv) > 2:
        try:
            mode = int(sys.argv[2], 8)  # Parse as octal
        except ValueError:
            print(f"Error: Invalid mode '{sys.argv[2]}' (must be octal)", file=sys.stderr)
            return 1

    # Read content from stdin
    try:
        content = sys.stdin.read()
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        return 130

    # Write file
    try:
        write_executable(path, content, mode)
        print(f"Created executable file: {path} (mode: {oct(mode)})")
        return 0
    except OSError as e:
        print(f"Error writing file: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
