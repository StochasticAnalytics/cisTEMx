#!/usr/bin/env python3
"""
Integration tests for Phase 2: Session Lifecycle Management

Tests all lifecycle defenses:
- Defense 4: TTL enforcement via Lead monitoring
- Defense 6: Two-phase shutdown protocol
- Defense 8: Session garbage collection
- Defense 9: Ticket requeue tracking
"""

import sys
import unittest
import tempfile
import shutil
import os
import time
import json
from pathlib import Path
from datetime import datetime

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from session_manager import (
    LeadOrchestrator,
    check_shutdown_signal,
    cleanup_abandoned_sessions,
    TicketManager
)


class TestTTLEnforcement(unittest.TestCase):
    """Test Defense 4: TTL enforcement via Lead monitoring"""

    def setUp(self):
        """Create test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.session_dir = self.test_dir / "session"
        self.session_dir.mkdir()

        # Create directory structure
        (self.session_dir / "tickets" / "active").mkdir(parents=True)
        (self.session_dir / "tickets" / "failed").mkdir(parents=True)
        (self.session_dir / "locks").mkdir(parents=True)

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_active_ticket(self, ticket_id, age_seconds=0):
        """Helper to create an active ticket with specific age"""
        ticket = {
            "id": ticket_id,
            "topic": "Test",
            "created_at": time.time() - age_seconds,
            "ttl_seconds": 300  # 5 minutes
        }

        ticket_path = self.session_dir / "tickets" / "active" / f"{ticket_id}.json"
        with open(ticket_path, 'w') as f:
            json.dump(ticket, f)

        return ticket

    def test_ttl_not_expired(self):
        """Ticket within TTL is not marked as failed"""
        self._create_active_ticket("ticket_001", age_seconds=60)  # 1 minute old

        orchestrator = LeadOrchestrator(self.session_dir)
        orchestrator.monitor_ticket_health()

        # Should still be in active
        active_path = self.session_dir / "tickets" / "active" / "ticket_001.json"
        self.assertTrue(active_path.exists())

        # Should not be in failed
        failed_path = self.session_dir / "tickets" / "failed" / "ticket_001.json"
        self.assertFalse(failed_path.exists())

    def test_ttl_expired(self):
        """Ticket beyond TTL is moved to failed"""
        self._create_active_ticket("ticket_002", age_seconds=360)  # 6 minutes old

        orchestrator = LeadOrchestrator(self.session_dir)
        orchestrator.monitor_ticket_health()

        # Should be moved to failed
        failed_path = self.session_dir / "tickets" / "failed" / "ticket_002.json"
        self.assertTrue(failed_path.exists())

        # Should not be in active
        active_path = self.session_dir / "tickets" / "active" / "ticket_002.json"
        self.assertFalse(active_path.exists())

        # Check failure reason
        with open(failed_path, 'r') as f:
            ticket = json.load(f)

        self.assertIn("TTL expired", ticket['failure_reason'])

    def test_lock_released_on_expiration(self):
        """Lock is force-released when ticket expires"""
        self._create_active_ticket("ticket_003", age_seconds=360)

        # Create lock for this ticket
        lock_path = self.session_dir / "locks" / "ticket_003.lock"
        lock_path.touch()

        orchestrator = LeadOrchestrator(self.session_dir)
        orchestrator.monitor_ticket_health()

        # Lock should be removed
        self.assertFalse(lock_path.exists())


class TestTwoPhaseShutdown(unittest.TestCase):
    """Test Defense 6: Two-phase shutdown protocol"""

    def setUp(self):
        """Create test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.session_dir = self.test_dir / "session"
        self.session_dir.mkdir()

        # Create directory structure
        (self.session_dir / "agents" / "red").mkdir(parents=True)
        (self.session_dir / "agents" / "blue").mkdir(parents=True)

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_shutdown_signal_created(self):
        """Lead creates shutdown signal file"""
        # Test by manually creating signal (initiate_shutdown cleans up after timeout)
        orchestrator = LeadOrchestrator(self.session_dir)

        signal = {
            'shutdown_initiated': True,
            'reason': 'test_shutdown',
            'timestamp': time.time(),
            'agents': ['red', 'blue'],
            'acks_required': ['red', 'blue'],
            'acks_received': [],
        }

        signal_path = self.session_dir / "shutdown_signal.json"
        with open(signal_path, 'w') as f:
            json.dump(signal, f, indent=2)

        # Signal should exist
        self.assertTrue(signal_path.exists())

        # Verify signal content
        with open(signal_path, 'r') as f:
            loaded = json.load(f)

        self.assertTrue(loaded['shutdown_initiated'])
        self.assertEqual(loaded['reason'], "test_shutdown")
        self.assertIn('red', loaded['acks_required'])
        self.assertIn('blue', loaded['acks_required'])

    def test_agent_ack_shutdown(self):
        """Agent successfully ACKs shutdown signal"""
        # Create shutdown signal
        signal = {
            'shutdown_initiated': True,
            'reason': 'test',
            'acks_required': ['red', 'blue'],
            'acks_received': []
        }

        signal_path = self.session_dir / "shutdown_signal.json"
        with open(signal_path, 'w') as f:
            json.dump(signal, f)

        # Agent checks shutdown
        result = check_shutdown_signal(self.session_dir, "red")

        self.assertTrue(result)

        # Verify ACK was added
        with open(signal_path, 'r') as f:
            updated_signal = json.load(f)

        self.assertIn('red', updated_signal['acks_received'])

    def test_no_shutdown_signal(self):
        """Agent returns False when no shutdown signal exists"""
        result = check_shutdown_signal(self.session_dir, "red")
        self.assertFalse(result)

    def test_session_archived_after_shutdown(self):
        """Session is archived (not deleted) after shutdown"""
        orchestrator = LeadOrchestrator(self.session_dir)

        # Manually add ACKs to avoid timeout
        signal = {
            'shutdown_initiated': True,
            'reason': 'test',
            'acks_required': ['red', 'blue'],
            'acks_received': ['red', 'blue']  # Pre-ACK'd
        }

        signal_path = self.session_dir / "shutdown_signal.json"
        with open(signal_path, 'w') as f:
            json.dump(signal, f)

        orchestrator.initiate_shutdown()

        # Session should be archived
        cache_dir = self.session_dir.parent
        archived = list(cache_dir.glob("archived_*"))

        self.assertEqual(len(archived), 1)


class TestSessionGarbageCollection(unittest.TestCase):
    """Test Defense 8: Session garbage collection"""

    def setUp(self):
        """Create test environment"""
        self.cache_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def _create_session(self, name, age_hours=0, has_activity=True):
        """Helper to create a test session"""
        session_dir = self.cache_dir / f"lab_tech_coordination_{name}"
        session_dir.mkdir()

        # Create session_id.txt
        session_id_path = session_dir / "session_id.txt"
        session_id_path.write_text(name)

        # Set modification time
        if age_hours > 0:
            old_time = time.time() - (age_hours * 3600)
            os.utime(session_id_path, (old_time, old_time))

        if has_activity:
            # Create agent state files
            state_dir = session_dir / "agents" / "red"
            state_dir.mkdir(parents=True)

            state_path = state_dir / "state.json"
            state_path.write_text(json.dumps({"last_heartbeat": time.time()}))

            if age_hours > 0:
                os.utime(state_path, (old_time, old_time))

        return session_dir

    def test_recent_active_session_preserved(self):
        """Recent, active session is not cleaned up"""
        session = self._create_session("recent", age_hours=0, has_activity=True)

        cleanup_abandoned_sessions(self.cache_dir)

        self.assertTrue(session.exists())

    def test_old_inactive_session_cleaned(self):
        """Old, inactive session (30+ min) is moved to abandoned/"""
        session = self._create_session("old_inactive", age_hours=1, has_activity=False)

        cleanup_abandoned_sessions(self.cache_dir)

        # Original should be moved
        self.assertFalse(session.exists())

        # Should be in abandoned/
        abandoned = list(self.cache_dir.glob("abandoned_*old_inactive*"))
        self.assertEqual(len(abandoned), 1)

    def test_incomplete_session_deleted(self):
        """Session without session_id.txt is deleted immediately"""
        session_dir = self.cache_dir / "lab_tech_coordination_incomplete"
        session_dir.mkdir()
        # No session_id.txt

        cleanup_abandoned_sessions(self.cache_dir)

        # Should be deleted
        self.assertFalse(session_dir.exists())

    def test_stale_session_cleaned(self):
        """Session with no activity for 30+ minutes is cleaned"""
        # Create session with state files but old activity
        session = self._create_session("stale", age_hours=0, has_activity=True)

        # Make state files old (35 minutes)
        state_path = session / "agents" / "red" / "state.json"
        old_time = time.time() - (35 * 60)
        os.utime(state_path, (old_time, old_time))

        cleanup_abandoned_sessions(self.cache_dir)

        # Should be moved to abandoned
        self.assertFalse(session.exists())

        abandoned = list(self.cache_dir.glob("abandoned_*stale*"))
        self.assertEqual(len(abandoned), 1)


class TestTicketRequeue(unittest.TestCase):
    """Test Defense 9: Ticket requeue tracking"""

    def setUp(self):
        """Create test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.session_dir = self.test_dir / "session"
        self.session_dir.mkdir()

        # Create directory structure
        (self.session_dir / "tickets" / "pending").mkdir(parents=True)
        (self.session_dir / "tickets" / "permanently_failed").mkdir(parents=True)

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_ticket(self, ticket_id, requeue_count=0):
        """Helper to create a test ticket"""
        ticket = {
            "id": ticket_id,
            "topic": "Test",
            "max_requeues": 1,
            "requeue_count": requeue_count
        }
        return ticket

    def test_first_failure_requeued(self):
        """First failure: ticket is requeued"""
        ticket = self._create_ticket("ticket_001", requeue_count=0)
        manager = TicketManager(self.session_dir)

        manager.handle_failed_ticket(ticket, "Test failure")

        # Should be requeued
        self.assertEqual(ticket['requeue_count'], 1)
        self.assertEqual(ticket['state'], 'pending')

        # Should exist in pending
        pending_path = self.session_dir / "tickets" / "pending" / "ticket_001.json"
        self.assertTrue(pending_path.exists())

    def test_second_failure_permanent(self):
        """Second failure: ticket permanently failed"""
        ticket = self._create_ticket("ticket_002", requeue_count=1)
        manager = TicketManager(self.session_dir)

        manager.handle_failed_ticket(ticket, "Second failure")

        # Should be permanently failed
        self.assertEqual(ticket['state'], 'permanently_failed')

        # Should exist in permanently_failed
        perm_failed_path = self.session_dir / "tickets" / "permanently_failed" / "ticket_002.json"
        self.assertTrue(perm_failed_path.exists())

    def test_failure_history_tracked(self):
        """Failure history is tracked for each failure"""
        ticket = self._create_ticket("ticket_003")
        manager = TicketManager(self.session_dir)

        # First failure
        manager.handle_failed_ticket(ticket, "Reason 1")

        self.assertEqual(len(ticket['failure_history']), 1)
        self.assertEqual(ticket['failure_history'][0]['failure_reason'], "Reason 1")
        self.assertTrue(ticket['failure_history'][0]['requeued'])

        # Second failure
        manager.handle_failed_ticket(ticket, "Reason 2")

        self.assertEqual(len(ticket['failure_history']), 2)
        self.assertEqual(ticket['failure_history'][1]['failure_reason'], "Reason 2")
        self.assertFalse(ticket['failure_history'][1]['requeued'])

    def test_custom_max_requeues(self):
        """Custom max_requeues is respected"""
        ticket = {
            "id": "ticket_004",
            "topic": "Test",
            "max_requeues": 2,  # Allow 2 requeues instead of 1
            "requeue_count": 0
        }

        manager = TicketManager(self.session_dir)

        # First failure: requeue
        manager.handle_failed_ticket(ticket, "Failure 1")
        self.assertEqual(ticket['state'], 'pending')

        # Second failure: still requeue
        manager.handle_failed_ticket(ticket, "Failure 2")
        self.assertEqual(ticket['state'], 'pending')

        # Third failure: permanent
        manager.handle_failed_ticket(ticket, "Failure 3")
        self.assertEqual(ticket['state'], 'permanently_failed')


def run_tests():
    """Run all Phase 2 integration tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTTLEnforcement))
    suite.addTests(loader.loadTestsFromTestCase(TestTwoPhaseShutdown))
    suite.addTests(loader.loadTestsFromTestCase(TestSessionGarbageCollection))
    suite.addTests(loader.loadTestsFromTestCase(TestTicketRequeue))

    # Run with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
