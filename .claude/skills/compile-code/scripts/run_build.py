#!/usr/bin/env python3
"""
Execute cisTEMx compilation with comprehensive logging.

This script extracts build configuration from VS Code tasks.json,
determines optimal parallelism, and executes make with full output
capture to a timestamped log file.
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


def find_git_root() -> Optional[Path]:
    """Find the git project root."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True
        )
        return Path(result.stdout.strip())
    except subprocess.CalledProcessError:
        return None


def get_core_count() -> int:
    """Determine optimal core count for parallel compilation (max 16)."""
    try:
        result = subprocess.run(
            ["lscpu", "-b", "-p=Core,Socket"],
            capture_output=True,
            text=True,
            check=True
        )
        # Count unique core,socket pairs (excluding comment lines)
        cores = len([
            line for line in result.stdout.splitlines()
            if not line.startswith('#')
        ])
        # Cap at 16 to avoid overwhelming the system
        return min(cores, 16)
    except subprocess.CalledProcessError:
        # Fallback to a reasonable default
        return 8


def extract_build_directory(
    tasks_json_path: Path,
    config_name: str
) -> Optional[str]:
    """
    Extract build subdirectory from tasks.json for specified config.
    
    Args:
        tasks_json_path: Path to .vscode/tasks.json
        config_name: Build configuration name (e.g., "DEBUG", "RELEASE")
    
    Returns:
        Build subdirectory name or None if not found
    """
    try:
        with open(tasks_json_path, 'r') as f:
            # Remove comments for JSON parsing
            content = f.read()
            # Simple comment removal (doesn't handle all edge cases but works for tasks.json)
            content = re.sub(r'//.*', '', content)
            tasks = json.loads(content)
        
        # Look for "BUILD cisTEMx <CONFIG>" task
        task_label = f"BUILD cisTEMx {config_name}"
        
        for task in tasks.get('tasks', []):
            if task.get('label') == task_label:
                command = task.get('command', '')
                # Extract directory from: cd ${build_dir}/SUBDIR && make
                match = re.search(r'cd\s+\$\{build_dir\}/([^\s&]+)', command)
                if match:
                    return match.group(1)
        
        return None
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing tasks.json: {e}", file=sys.stderr)
        return None


def run_make(
    build_dir: Path,
    core_count: int,
    log_file: Path
) -> Tuple[int, Path]:
    """
    Execute make with output logging.
    
    Args:
        build_dir: Absolute path to build directory
        core_count: Number of parallel jobs
        log_file: Path to log file
    
    Returns:
        Tuple of (exit_code, log_file_path)
    """
    print(f"Build Configuration:")
    print(f"  Build Directory: {build_dir}")
    print(f"  Parallel Jobs: {core_count}")
    print(f"  Log File: {log_file}")
    print()
    print("Starting compilation...")
    print()
    
    # Ensure log directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Run make with output capture
    try:
        with open(log_file, 'w') as log:
            process = subprocess.Popen(
                ["make", f"-j{core_count}"],
                cwd=build_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Stream output to both console and log file
            for line in process.stdout:
                print(line, end='')
                log.write(line)
            
            process.wait()
            return process.returncode, log_file
            
    except Exception as e:
        print(f"Error executing make: {e}", file=sys.stderr)
        return 1, log_file


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Execute cisTEMx compilation with logging'
    )
    parser.add_argument(
        '--config',
        default='DEBUG',
        help='Build configuration from tasks.json (default: DEBUG)'
    )
    parser.add_argument(
        '--cores',
        type=int,
        help='Override automatic core count detection'
    )
    parser.add_argument(
        '--print-build-dir',
        action='store_true',
        help='Print build directory path and exit (no compilation)'
    )

    args = parser.parse_args()
    
    # Find git root
    git_root = find_git_root()
    if not git_root:
        print("Error: Not in a git repository", file=sys.stderr)
        return 1
    
    # Locate tasks.json
    tasks_json = git_root / ".vscode" / "tasks.json"
    if not tasks_json.exists():
        print(f"Error: {tasks_json} not found", file=sys.stderr)
        return 1
    
    # Extract build directory
    build_subdir = extract_build_directory(tasks_json, args.config)
    if not build_subdir:
        print(f"Error: Could not find build config '{args.config}' in tasks.json",
              file=sys.stderr)
        print(f"Available configs: DEBUG, RELEASE, etc.", file=sys.stderr)
        return 1
    
    build_dir = git_root / "build" / build_subdir

    # If user just wants the build directory path, print and exit
    if args.print_build_dir:
        print(build_dir)
        return 0

    if not build_dir.exists():
        print(f"Error: Build directory does not exist: {build_dir}", file=sys.stderr)
        print("Run the corresponding 'Configure' task first", file=sys.stderr)
        return 1
    
    # Determine core count
    core_count = args.cores if args.cores else get_core_count()
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = git_root / ".claude" / "cache" / f"build_{timestamp}.log"
    
    # Execute build
    exit_code, log_path = run_make(build_dir, core_count, log_file)
    
    # Report result
    print()
    if exit_code == 0:
        print("✓ BUILD SUCCESS")
        print(f"  Log: {log_path}")
        return 0
    else:
        print("✗ BUILD FAILED")
        print(f"  Log: {log_path}")
        print()
        print("Run analyze_build_log.py to generate error summary:")
        print(f"  python .claude/skills/compile-code/scripts/analyze_build_log.py {log_path}")
        return exit_code


if __name__ == "__main__":
    sys.exit(main())
