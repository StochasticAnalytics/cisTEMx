"""
Session lifecycle management for lab-tech coordination.

Implements Purple Team Defenses 4, 6, 8, 9:
- TTL enforcement via Lead monitoring
- Two-phase shutdown protocol
- Session garbage collection
- Ticket requeue tracking
"""

import os
import time
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from glob import glob
import json


class LeadOrchestrator:
    """Lead orchestrator with session lifecycle management."""

    def __init__(self, session_dir: Path):
        self.session_dir = Path(session_dir)

    def monitor_ticket_health(self) -> None:
        """
        Periodic health check for tickets.
        Called every 30 seconds by Lead's monitoring loop.
        """
        now = time.time()

        # Check active tickets
        active_dir = self.session_dir / "tickets" / "active"
        if not active_dir.exists():
            return

        for ticket_file in active_dir.glob("*.json"):
            with open(ticket_file, 'r') as f:
                ticket = json.load(f)

            age = now - self._parse_timestamp(ticket['created_at'])
            ttl = ticket.get('ttl_seconds', 300)

            if age > ttl:
                # Ticket expired
                ticket_id = ticket['id']
                self._handle_expired_ticket(ticket, ticket_file)

    def _parse_timestamp(self, timestamp_str: str) -> float:
        """Parse ISO timestamp to epoch seconds."""
        from datetime import datetime
        try:
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            return dt.timestamp()
        except:
            # Fallback: treat as epoch seconds if already a number
            return float(timestamp_str)

    def _handle_expired_ticket(
        self,
        ticket: Dict[str, Any],
        ticket_file: Path
    ) -> None:
        """Handle an expired ticket."""
        ticket_id = ticket['id']
        ttl = ticket.get('ttl_seconds', 300)

        # Check if lock still held
        lock_path = self.session_dir / "locks" / f"{ticket_id}.lock"
        if lock_path.exists():
            # Agent crashed or stuck, force-release lock
            try:
                os.unlink(lock_path)
            except FileNotFoundError:
                pass

        # Move to failed for manual review
        failed_dir = self.session_dir / "tickets" / "failed"
        failed_dir.mkdir(parents=True, exist_ok=True)

        ticket['failure_reason'] = f"TTL expired ({ttl}s)"
        ticket['expired_at'] = time.time()

        failed_path = failed_dir / f"{ticket_id}.json"
        with open(failed_path, 'w') as f:
            json.dump(ticket, f, indent=2)

        os.unlink(ticket_file)

    def initiate_shutdown(self, reason: str = "converged") -> None:
        """
        Two-phase shutdown:
        Phase 1: Signal shutdown, wait for agent acknowledgment
        Phase 2: After all agents ACK, perform cleanup
        """
        shutdown_signal = {
            'shutdown_initiated': True,
            'reason': reason,
            'timestamp': time.time(),
            'agents': ['red', 'blue'],
            'acks_required': ['red', 'blue'],
            'acks_received': [],
        }

        signal_path = self.session_dir / "shutdown_signal.json"
        with open(signal_path, 'w') as f:
            json.dump(shutdown_signal, f, indent=2)

        # Wait for ACKs with timeout
        timeout = 30  # seconds
        start = time.time()

        while time.time() - start < timeout:
            with open(signal_path, 'r') as f:
                signal = json.load(f)

            if set(signal['acks_received']) == set(signal['acks_required']):
                # All agents acknowledged
                self.perform_cleanup()
                return

            time.sleep(1)

        # Timeout: Some agents didn't ACK
        print(f"Warning: Shutdown timeout, forcing cleanup")
        self.perform_cleanup()

    def perform_cleanup(self) -> None:
        """Actually delete session directory."""
        # Move to temporary archived location first
        cache_dir = self.session_dir.parent
        archive_path = cache_dir / f"archived_{self.session_dir.name}"

        shutil.move(str(self.session_dir), str(archive_path))

        # Archive will be cleaned up by garbage collection later


def check_shutdown_signal(session_dir: Path, agent_name: str) -> bool:
    """
    Check if Lead has initiated shutdown.

    Args:
        session_dir: Session directory
        agent_name: Name of this agent (red, blue, etc.)

    Returns:
        True if shutdown signal detected, False otherwise
    """
    signal_path = session_dir / "shutdown_signal.json"

    if not signal_path.exists():
        return False

    with open(signal_path, 'r') as f:
        signal = json.load(f)

    if signal.get('shutdown_initiated'):
        # ACK shutdown
        if agent_name not in signal['acks_received']:
            signal['acks_received'].append(agent_name)
            with open(signal_path, 'w') as f:
                json.dump(signal, f, indent=2)

        return True

    return False


def cleanup_abandoned_sessions(cache_dir: Path = None) -> None:
    """
    Scan for session directories that have been abandoned.
    Run this periodically (e.g., on agent startup or nightly).

    Args:
        cache_dir: Cache directory to scan (defaults to .claude/cache)
    """
    if cache_dir is None:
        cache_dir = Path(".claude/cache")

    pattern = str(cache_dir / "lab_tech_coordination_*")

    for session_dir_str in glob(pattern):
        session_dir = Path(session_dir_str)

        if not session_dir.is_dir():
            continue

        # Check session age
        session_id_file = session_dir / "session_id.txt"
        if not session_id_file.exists():
            # Incomplete session, delete immediately
            shutil.rmtree(session_dir, ignore_errors=True)
            continue

        age = time.time() - session_id_file.stat().st_mtime

        # Check if session is still active
        state_files = list(session_dir.glob("agents/*/state.json"))
        if not state_files:
            # No agents have checked in, likely abandoned
            if age > 1800:  # 30 minutes
                abandoned_path = cache_dir / f"abandoned_{session_dir.name}"
                try:
                    shutil.move(str(session_dir), str(abandoned_path))
                except Exception:
                    pass  # May have been cleaned up concurrently
            continue

        # Check last state update
        last_activity = max(f.stat().st_mtime for f in state_files)
        idle_time = time.time() - last_activity

        if idle_time > 1800:  # 30 minutes of inactivity
            # Likely abandoned - move to abandoned/ for inspection
            abandoned_path = cache_dir / f"abandoned_{session_dir.name}"
            try:
                shutil.move(str(session_dir), str(abandoned_path))
            except Exception:
                pass  # May have been cleaned up concurrently


class TicketManager:
    """Manages ticket requeue tracking and failure handling."""

    def __init__(self, session_dir: Path):
        self.session_dir = Path(session_dir)

    def handle_failed_ticket(
        self,
        ticket: Dict[str, Any],
        failure_reason: str = "unknown"
    ) -> None:
        """
        Handle a failed ticket with requeue policy.

        Args:
            ticket: Ticket data
            failure_reason: Reason for failure
        """
        # Record failure
        failure_record = {
            'failed_at': time.time(),
            'failure_reason': failure_reason,
            'requeued': False,
        }

        if 'failure_history' not in ticket:
            ticket['failure_history'] = []
        ticket['failure_history'].append(failure_record)

        # Check requeue count
        requeue_count = ticket.get('requeue_count', 0)
        max_requeues = ticket.get('max_requeues', 1)

        if requeue_count < max_requeues:
            # Requeue
            ticket['requeue_count'] = requeue_count + 1
            ticket['failure_history'][-1]['requeued'] = True

            # Reset state and move back to pending
            ticket['state'] = 'pending'
            pending_dir = self.session_dir / "tickets" / "pending"
            pending_dir.mkdir(parents=True, exist_ok=True)

            pending_path = pending_dir / f"{ticket['id']}.json"
            with open(pending_path, 'w') as f:
                json.dump(ticket, f, indent=2)

            print(f"Info: Requeued ticket {ticket['id']} "
                  f"(attempt {requeue_count + 2})")
        else:
            # Permanently failed
            ticket['state'] = 'permanently_failed'
            perm_failed_dir = self.session_dir / "tickets" / "permanently_failed"
            perm_failed_dir.mkdir(parents=True, exist_ok=True)

            perm_failed_path = perm_failed_dir / f"{ticket['id']}.json"
            with open(perm_failed_path, 'w') as f:
                json.dump(ticket, f, indent=2)

            print(f"Error: Ticket {ticket['id']} permanently failed "
                  f"after {requeue_count + 1} attempts")
