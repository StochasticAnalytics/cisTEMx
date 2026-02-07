#!/usr/bin/env python3
"""Launch cisTEMx development container and attach Cursor.

This script replicates the functionality of devcontainer.json for editors
that don't support the Dev Containers extension (like Cursor).

Default behavior: Start the container and launch Cursor attached to it.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def get_workspace_info() -> dict:
    """Determine workspace paths and container name.

    Returns:
        dict with keys: workspace_root, workspace_basename, container_name
    """
    # Script is in .devcontainer/, workspace is parent
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent
    workspace_basename = workspace_root.name

    user = os.environ.get("USER", "unknown")
    container_name = f"cisTEMx-{user}-{workspace_basename}"

    return {
        "workspace_root": workspace_root,
        "workspace_basename": workspace_basename,
        "container_name": container_name,
        "user": user,
    }


def read_image_config(workspace_root: Path) -> str:
    """Read image version from .vscode/ config files.

    Args:
        workspace_root: Path to workspace root directory

    Returns:
        Full image name with tag (e.g., ghcr.io/stochasticanalytics/cistem_build_env:v3.1.0)
    """
    version_file = workspace_root / ".vscode" / "CONTAINER_VERSION_TOP"
    repo_file = workspace_root / ".vscode" / "CONTAINER_REPO_NAME"

    if not version_file.exists():
        print(f"Error: Version file not found: {version_file}", file=sys.stderr)
        sys.exit(1)
    if not repo_file.exists():
        print(f"Error: Repo file not found: {repo_file}", file=sys.stderr)
        sys.exit(1)

    version = version_file.read_text().strip()
    repo = repo_file.read_text().strip()

    return f"{repo}:v{version}"


def run_version_check(workspace_root: Path) -> bool:
    """Execute the existing check-version.sh validation.

    Args:
        workspace_root: Path to workspace root directory

    Returns:
        True if validation passes, False otherwise
    """
    check_script = workspace_root / ".devcontainer" / "check-version.sh"

    if not check_script.exists():
        print(f"Warning: Version check script not found: {check_script}", file=sys.stderr)
        return True  # Don't fail if script doesn't exist

    result = subprocess.run(
        [str(check_script)],
        cwd=str(workspace_root),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("Version check failed:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return False

    return True


def get_container_status(container_name: str) -> str:
    """Check if container is running, stopped, or non-existent.

    Args:
        container_name: Name of the Docker container

    Returns:
        One of: "running", "stopped", "none"
    """
    # Check if container exists at all
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Status}}", container_name],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return "none"

    status = result.stdout.strip()
    if status == "running":
        return "running"
    else:
        return "stopped"


def stop_container(container_name: str) -> bool:
    """Stop the running container.

    Args:
        container_name: Name of the Docker container

    Returns:
        True if stopped successfully or wasn't running
    """
    status = get_container_status(container_name)

    if status == "none":
        print(f"Container '{container_name}' does not exist.")
        return True

    if status == "stopped":
        print(f"Container '{container_name}' is already stopped.")
        return True

    print(f"Stopping container '{container_name}'...")
    result = subprocess.run(
        ["docker", "stop", container_name],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Failed to stop container: {result.stderr}", file=sys.stderr)
        return False

    print(f"Container '{container_name}' stopped.")
    return True


def remove_container(container_name: str) -> bool:
    """Remove a stopped container.

    Args:
        container_name: Name of the Docker container

    Returns:
        True if removed successfully or didn't exist
    """
    result = subprocess.run(
        ["docker", "rm", container_name],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def start_container(info: dict, image: str, interactive: bool = False) -> bool:
    """Start the container with all mounts and configuration.

    Args:
        info: Workspace info dict from get_workspace_info()
        image: Full Docker image name with tag
        interactive: If True, run interactively with TTY

    Returns:
        True if container is running after this call
    """
    container_name = info["container_name"]
    workspace_root = info["workspace_root"]
    workspace_basename = info["workspace_basename"]

    status = get_container_status(container_name)

    # If running and not interactive, we're done
    if status == "running" and not interactive:
        print(f"Container '{container_name}' is already running.")
        return True

    # If stopped, start it
    if status == "stopped":
        print(f"Starting stopped container '{container_name}'...")
        result = subprocess.run(
            ["docker", "start", container_name],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Failed to start container: {result.stderr}", file=sys.stderr)
            return False
        print(f"Container '{container_name}' started.")
        return True

    # Container doesn't exist, create it
    print(f"Creating container '{container_name}'...")

    # Build docker run command
    cmd = ["docker", "run"]

    if interactive:
        cmd.extend(["-it", "--rm"])
    else:
        cmd.append("-d")

    # Container name
    cmd.extend(["--name", container_name])

    # Network
    cmd.extend(["--network", "host"])

    # User (don't use --user, let the container handle it via entrypoint)
    # The container's default user is cisTEMdev

    # Environment variables
    display = os.environ.get("DISPLAY", ":0")
    cmd.extend(["-e", f"DISPLAY={display}"])
    cmd.extend(["-e", f"WORKSPACE_BASENAME={workspace_basename}"])
    cmd.extend(["-e", f"WORKSPACE_CONTAINER_NAME={container_name}"])

    # Workspace mount
    cmd.extend(["-v", f"{workspace_root}:/workspaces/cisTEMx"])

    # Additional mounts from devcontainer.json
    cmd.extend(["-v", "/scratch:/scratch"])
    cmd.extend(["-v", "/sa_shared:/sa_shared"])

    # X11 forwarding
    xauthority = os.environ.get("XAUTHORITY", os.path.expanduser("~/.Xauthority"))
    cmd.extend(["-v", f"{xauthority}:/home/cisTEMdev/.Xauthority"])
    cmd.extend(["-v", "/tmp/.X11-unix:/tmp/.X11-unix"])

    # Working directory inside container
    cmd.extend(["-w", "/workspaces/cisTEMx"])

    # Image
    cmd.append(image)

    # For detached mode, run a long-lived process
    if not interactive:
        cmd.extend(["sleep", "infinity"])

    print(f"Running: {' '.join(cmd)}")

    if interactive:
        # For interactive, replace current process
        os.execvp("docker", cmd)
    else:
        # Don't capture output - let Docker show pull progress and other messages
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Failed to create container (exit code {result.returncode})", file=sys.stderr)
            return False
        print(f"Container '{container_name}' created and running.")
        return True


def attach_cursor(info: dict) -> None:
    """Launch Cursor attached to the running container.

    Args:
        info: Workspace info dict from get_workspace_info()
    """
    container_name = info["container_name"]

    # Hex-encode the container name for the URI
    hex_name = container_name.encode().hex()

    # Build the remote URI
    # Format: vscode-remote://attached-container+<hex-name>/path
    remote_uri = f"vscode-remote://attached-container+{hex_name}/workspaces/cisTEMx"

    print(f"Launching Cursor attached to '{container_name}'...")
    print(f"URI: {remote_uri}")

    # Launch Cursor with the remote URI
    subprocess.Popen(
        ["cursor", "--folder-uri", remote_uri],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    print("Cursor launched. It may take a moment to connect.")


def show_status(info: dict) -> None:
    """Display container status information.

    Args:
        info: Workspace info dict from get_workspace_info()
    """
    container_name = info["container_name"]
    status = get_container_status(container_name)

    print(f"Container: {container_name}")
    print(f"Status: {status}")

    if status == "running":
        # Get additional info
        result = subprocess.run(
            ["docker", "inspect", "--format",
             "Created: {{.Created}}\nImage: {{.Config.Image}}",
             container_name],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(result.stdout)


def main() -> int:
    """Parse arguments and execute requested action."""
    parser = argparse.ArgumentParser(
        description="Launch cisTEMx development container for Cursor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s              Start container and launch Cursor
  %(prog)s --no-cursor  Start container only
  %(prog)s --stop       Stop the container
  %(prog)s --restart    Restart container and launch Cursor
  %(prog)s --shell      Start container and get interactive shell
  %(prog)s --status     Show container status
        """,
    )

    parser.add_argument(
        "--no-cursor",
        action="store_true",
        help="Start container without launching Cursor",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop the running container",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Restart the container (stop, remove, start fresh)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show container status",
    )
    parser.add_argument(
        "--shell",
        action="store_true",
        help="Start container and attach an interactive shell (implies --no-cursor)",
    )

    args = parser.parse_args()

    # Get workspace info
    info = get_workspace_info()

    # Handle status command
    if args.status:
        show_status(info)
        return 0

    # Handle stop command
    if args.stop:
        return 0 if stop_container(info["container_name"]) else 1

    # Read image configuration
    image = read_image_config(info["workspace_root"])
    print(f"Using image: {image}")

    # Run version check
    if not run_version_check(info["workspace_root"]):
        print("Version check failed. Please fix the configuration.", file=sys.stderr)
        return 1

    # Handle restart command
    if args.restart:
        stop_container(info["container_name"])
        remove_container(info["container_name"])

    # Handle shell command (interactive mode)
    if args.shell:
        start_container(info, image, interactive=True)
        # If we get here, exec failed
        return 1

    # Start container (default)
    if not start_container(info, image):
        return 1

    # Launch Cursor unless --no-cursor specified
    if not args.no_cursor:
        attach_cursor(info)

    return 0


if __name__ == "__main__":
    sys.exit(main())
