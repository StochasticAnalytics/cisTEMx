"""
Filesystem coordination primitives for lab-tech agents.

Implements Purple Team Defenses 1, 2, 3, 10:
- Atomic ticket claiming via hardlinks
- Filesystem compatibility layer
- Advisory locks with timeout and cleanup
- Atomic artifact writing with checksums
"""

import os
import fcntl
import time
import socket
import hashlib
import uuid
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import contextmanager
import json


class FilesystemCoordination:
    """
    Filesystem operations guaranteed to be atomic and cross-platform.
    Detects filesystem capabilities and adapts operations.
    """

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.fs_type = self._detect_filesystem()
        self.atomic_ops = self._configure_atomic_ops()

    def _detect_filesystem(self) -> str:
        """Detect NFS, local, Docker volume, etc."""
        # Basic detection - can be enhanced
        try:
            # Check if running in Docker
            if os.path.exists('/.dockerenv'):
                return 'docker_volume'

            # Check /proc/mounts on Linux
            if os.path.exists('/proc/mounts'):
                with open('/proc/mounts', 'r') as f:
                    mounts = f.read()
                    if 'nfs' in mounts:
                        return 'nfs'

            return 'local_posix'
        except Exception:
            return 'unknown'

    def _configure_atomic_ops(self) -> Dict[str, Any]:
        """Configure atomic operation strategy based on filesystem."""
        if self.fs_type == 'nfs':
            return {
                'move': self._nfs_atomic_move,
                'lock': self._fcntl_lock,
            }
        else:  # local_posix, docker_volume, unknown
            return {
                'move': self._posix_atomic_rename,
                'lock': self._fcntl_lock,
            }

    def atomic_move(self, src: Path, dst: Path) -> None:
        """Guaranteed atomic move for this filesystem type."""
        return self.atomic_ops['move'](src, dst)

    def _nfs_atomic_move(self, src: Path, dst: Path) -> None:
        """NFS-safe atomic move using hardlink + unlink pattern."""
        temp_name = dst.parent / f".tmp_{uuid.uuid4()}_{dst.name}"
        try:
            # 1. Hardlink src to temp
            os.link(src, temp_name)
            # 2. Rename temp to dst (atomic on NFS if same dir)
            os.rename(temp_name, dst)
            # 3. Unlink src
            os.unlink(src)
        except Exception as e:
            # Cleanup temp if it exists
            if temp_name.exists():
                temp_name.unlink()
            raise

    def _posix_atomic_rename(self, src: Path, dst: Path) -> None:
        """Simple rename for POSIX local filesystems."""
        os.rename(src, dst)

    def _fcntl_lock(self, fd: int, timeout: int = 30) -> None:
        """Acquire lock using fcntl (advisory)."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                time.sleep(0.1)
        raise TimeoutError(f"Could not acquire lock within {timeout}s")


def atomic_claim_ticket(
    ticket_id: str,
    agent_name: str,
    session_dir: Path
) -> Optional[Dict[str, Any]]:
    """
    Atomically claim a ticket using hardlink test-and-set.

    Args:
        ticket_id: Ticket identifier
        agent_name: Name of claiming agent
        session_dir: Session directory path

    Returns:
        Ticket data if successful, None if already claimed

    Raises:
        FileNotFoundError: If ticket doesn't exist
    """
    ticket_path = session_dir / "tickets" / "pending" / f"{ticket_id}.json"
    claim_path = session_dir / "agents" / agent_name / "inbox" / f"{ticket_id}.json"
    lock_path = session_dir / "locks" / f"{ticket_id}.lock"

    try:
        # Atomic test-and-set via hardlink (works on NFS, Windows, POSIX)
        os.link(ticket_path, lock_path)
    except FileExistsError:
        # Another agent already claimed it
        return None
    except FileNotFoundError:
        # Ticket doesn't exist
        return None

    # We won the race, now move to our inbox
    try:
        with open(ticket_path, 'r') as f:
            ticket_data = json.load(f)

        # Write to claim path
        claim_path.parent.mkdir(parents=True, exist_ok=True)
        with open(claim_path, 'w') as f:
            json.dump(ticket_data, f)

        # Remove from pending
        os.unlink(ticket_path)

        return ticket_data
    except Exception as e:
        # Release lock on failure
        try:
            os.unlink(lock_path)
        except FileNotFoundError:
            pass
        raise


def release_ticket(ticket_id: str, session_dir: Path) -> None:
    """
    Release ticket lock after completion or failure.

    Args:
        ticket_id: Ticket identifier
        session_dir: Session directory path
    """
    lock_path = session_dir / "locks" / f"{ticket_id}.lock"
    try:
        os.unlink(lock_path)
    except FileNotFoundError:
        pass  # Already released


@contextmanager
def ticket_lock(
    ticket_id: str,
    session_dir: Path,
    timeout_seconds: int = 30,
    holder_id: Optional[str] = None
):
    """
    Advisory lock with timeout and crash recovery.

    Args:
        ticket_id: Ticket to lock
        session_dir: Session directory
        timeout_seconds: Max time to wait for lock
        holder_id: Process/agent ID (for crash recovery)

    Yields:
        lock_fd: File descriptor (for fcntl operations)

    Raises:
        TimeoutError: If lock not acquired within timeout
    """
    lock_path = session_dir / "locks" / f"{ticket_id}.lock"
    lock_info_path = session_dir / "locks" / f"{ticket_id}.info.json"

    # Ensure locks directory exists
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    # Create lock file with holder info
    lock_info = {
        'holder_id': holder_id or os.getpid(),
        'acquired_at': time.time(),
        'hostname': socket.gethostname(),
    }

    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    start_time = time.time()

    try:
        # Try non-blocking first
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with open(lock_info_path, 'w') as f:
                json.dump(lock_info, f)
            yield fd
            return
        except BlockingIOError:
            pass

        # Someone else holds lock, check if stale
        if lock_info_path.exists():
            with open(lock_info_path, 'r') as f:
                existing_lock = json.load(f)
            age = time.time() - existing_lock['acquired_at']

            if age > 300:  # 5 minutes = stale lock
                # Try to steal lock (previous holder crashed)
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with open(lock_info_path, 'w') as f:
                    json.dump(lock_info, f)
                yield fd
                return

        # Wait with timeout
        while time.time() - start_time < timeout_seconds:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with open(lock_info_path, 'w') as f:
                    json.dump(lock_info, f)
                yield fd
                return
            except BlockingIOError:
                time.sleep(0.1)

        raise TimeoutError(
            f"Could not acquire lock on {ticket_id} within {timeout_seconds}s"
        )

    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        try:
            os.unlink(lock_info_path)
        except FileNotFoundError:
            pass


def write_artifact_atomic(content: str, session_dir: Path) -> Path:
    """
    Write artifact atomically to prevent partial reads.

    Args:
        content: Artifact content
        session_dir: Session directory

    Returns:
        Path to written artifact
    """
    artifact_id = str(uuid.uuid4())
    artifacts_dir = session_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    temp_path = artifacts_dir / f".tmp_{artifact_id}.md"
    final_path = artifacts_dir / f"{artifact_id}.md"
    checksum_path = artifacts_dir / f"{artifact_id}.md.sha256"

    # Write to temp file
    with open(temp_path, 'w') as f:
        f.write(content)

    # Calculate checksum
    checksum = hashlib.sha256(content.encode()).hexdigest()
    with open(checksum_path, 'w') as f:
        f.write(checksum)

    # Atomic rename (only visible when complete)
    os.rename(temp_path, final_path)

    return final_path


def read_artifact_validated(artifact_path: Path) -> str:
    """
    Read artifact and validate checksum.

    Args:
        artifact_path: Path to artifact

    Returns:
        Validated artifact content

    Raises:
        ValueError: If checksum missing or mismatch
    """
    checksum_path = Path(str(artifact_path) + ".sha256")

    if not checksum_path.exists():
        raise ValueError(f"Artifact {artifact_path} missing checksum")

    with open(checksum_path, 'r') as f:
        expected_checksum = f.read().strip()

    with open(artifact_path, 'r') as f:
        content = f.read()

    actual_checksum = hashlib.sha256(content.encode()).hexdigest()

    if actual_checksum != expected_checksum:
        raise ValueError(
            f"Artifact {artifact_path} checksum mismatch: "
            f"expected {expected_checksum}, got {actual_checksum}"
        )

    return content
