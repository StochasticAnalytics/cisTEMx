#!/usr/bin/env python3
"""
Comprehensive unit tests for Phase 1: Core Safety

Tests all defensive implementations:
- Defense 1: Atomic ticket claiming via hardlinks
- Defense 2: Filesystem compatibility layer
- Defense 3: Advisory locks with timeout and cleanup
- Defense 10: Atomic artifact writing with checksums
"""

import sys
import unittest
import tempfile
import shutil
import os
import time
import json
import hashlib
from pathlib import Path
from multiprocessing import Process, Queue
import fcntl

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from filesystem_coordination import (
    atomic_claim_ticket,
    release_ticket,
    ticket_lock,
    write_artifact_atomic,
    read_artifact_validated,
    FilesystemCoordination
)


class TestAtomicTicketClaiming(unittest.TestCase):
    """Test Defense 1: Atomic ticket claiming via hardlinks"""

    def setUp(self):
        """Create test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.session_dir = self.test_dir / "session"

        # Create session structure
        (self.session_dir / "tickets" / "pending").mkdir(parents=True)
        (self.session_dir / "agents" / "red" / "inbox").mkdir(parents=True)
        (self.session_dir / "agents" / "blue" / "inbox").mkdir(parents=True)
        (self.session_dir / "locks").mkdir(parents=True)

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_ticket(self, ticket_id="test_001"):
        """Helper to create a test ticket"""
        ticket = {
            "id": ticket_id,
            "topic": "Test",
            "instructions": "Test instructions"
        }
        ticket_path = self.session_dir / "tickets" / "pending" / f"{ticket_id}.json"
        with open(ticket_path, 'w') as f:
            json.dump(ticket, f)
        return ticket

    def test_single_agent_claim_success(self):
        """Single agent successfully claims ticket"""
        self._create_ticket("ticket_001")

        result = atomic_claim_ticket("ticket_001", "red", self.session_dir)

        self.assertIsNotNone(result)
        self.assertEqual(result['id'], "ticket_001")

        # Verify ticket moved to inbox
        inbox_path = self.session_dir / "agents" / "red" / "inbox" / "ticket_001.json"
        self.assertTrue(inbox_path.exists())

        # Verify ticket removed from pending
        pending_path = self.session_dir / "tickets" / "pending" / "ticket_001.json"
        self.assertFalse(pending_path.exists())

        # Verify lock created
        lock_path = self.session_dir / "locks" / "ticket_001.lock"
        self.assertTrue(lock_path.exists())

    def test_double_claim_prevented(self):
        """Second agent cannot claim already-claimed ticket"""
        self._create_ticket("ticket_001")

        # Red claims
        result1 = atomic_claim_ticket("ticket_001", "red", self.session_dir)
        self.assertIsNotNone(result1)

        # Recreate ticket in pending (simulate race condition)
        self._create_ticket("ticket_001")

        # Blue tries to claim (should fail due to lock)
        result2 = atomic_claim_ticket("ticket_001", "blue", self.session_dir)
        self.assertIsNone(result2)

    def test_claim_nonexistent_ticket(self):
        """Claiming nonexistent ticket returns None"""
        result = atomic_claim_ticket("nonexistent", "red", self.session_dir)
        self.assertIsNone(result)

    def test_concurrent_claim_race_condition(self):
        """Multiple agents racing for same ticket - exactly one wins"""
        self._create_ticket("ticket_race")

        def attempt_claim(agent_name, result_queue):
            """Worker function for concurrent claim attempt"""
            result = atomic_claim_ticket("ticket_race", agent_name, self.session_dir)
            result_queue.put((agent_name, result is not None))

        # Launch 5 agents concurrently
        result_queue = Queue()
        processes = []
        for i in range(5):
            p = Process(target=attempt_claim, args=(f"agent_{i}", result_queue))
            p.start()
            processes.append(p)

        # Wait for all processes
        for p in processes:
            p.join(timeout=5)

        # Collect results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        # Verify exactly one succeeded
        successes = [r for r in results if r[1]]
        self.assertEqual(len(successes), 1,
                        f"Expected exactly 1 success, got {len(successes)}")

    def test_release_ticket(self):
        """Releasing ticket removes lock file"""
        self._create_ticket("ticket_001")
        atomic_claim_ticket("ticket_001", "red", self.session_dir)

        lock_path = self.session_dir / "locks" / "ticket_001.lock"
        self.assertTrue(lock_path.exists())

        release_ticket("ticket_001", self.session_dir)

        self.assertFalse(lock_path.exists())

    def test_release_nonexistent_ticket(self):
        """Releasing nonexistent ticket doesn't raise exception"""
        # Should not raise
        release_ticket("nonexistent", self.session_dir)


class TestFilesystemCompatibility(unittest.TestCase):
    """Test Defense 2: Filesystem compatibility layer"""

    def setUp(self):
        """Create test environment"""
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_filesystem_detection(self):
        """Filesystem type is detected"""
        fs = FilesystemCoordination(self.test_dir)

        # Should detect one of the known types
        self.assertIn(fs.fs_type, ['docker_volume', 'local_posix', 'nfs', 'unknown'])

    def test_atomic_move_same_directory(self):
        """Atomic move within same directory"""
        fs = FilesystemCoordination(self.test_dir)

        src = self.test_dir / "source.txt"
        dst = self.test_dir / "dest.txt"

        src.write_text("test content")

        fs.atomic_move(src, dst)

        self.assertFalse(src.exists())
        self.assertTrue(dst.exists())
        self.assertEqual(dst.read_text(), "test content")

    def test_atomic_move_cross_directory(self):
        """Atomic move across directories"""
        fs = FilesystemCoordination(self.test_dir)

        src_dir = self.test_dir / "src"
        dst_dir = self.test_dir / "dst"
        src_dir.mkdir()
        dst_dir.mkdir()

        src = src_dir / "file.txt"
        dst = dst_dir / "file.txt"

        src.write_text("test content")

        fs.atomic_move(src, dst)

        self.assertFalse(src.exists())
        self.assertTrue(dst.exists())
        self.assertEqual(dst.read_text(), "test content")

    def test_atomic_move_preserves_content(self):
        """Atomic move preserves file content exactly"""
        fs = FilesystemCoordination(self.test_dir)

        content = "A" * 10000  # 10KB
        src = self.test_dir / "large.txt"
        dst = self.test_dir / "large_moved.txt"

        src.write_text(content)

        fs.atomic_move(src, dst)

        self.assertEqual(dst.read_text(), content)


class TestAdvisoryLocks(unittest.TestCase):
    """Test Defense 3: Advisory locks with timeout and cleanup"""

    def setUp(self):
        """Create test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.session_dir = self.test_dir / "session"
        self.session_dir.mkdir()

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_lock_acquisition_and_release(self):
        """Lock can be acquired and released"""
        with ticket_lock("test_lock", self.session_dir) as fd:
            self.assertIsNotNone(fd)

            # Lock file should exist
            lock_path = self.session_dir / "locks" / "test_lock.lock"
            self.assertTrue(lock_path.exists())

            # Lock info should exist
            info_path = self.session_dir / "locks" / "test_lock.info.json"
            self.assertTrue(info_path.exists())

        # After context exit, info should be cleaned up
        self.assertFalse(info_path.exists())

    def test_lock_timeout(self):
        """Lock times out when already held"""
        # First lock holder
        with ticket_lock("test_lock", self.session_dir, timeout_seconds=30):
            # Second lock attempt with short timeout
            with self.assertRaises(TimeoutError):
                with ticket_lock("test_lock", self.session_dir, timeout_seconds=1):
                    pass

    def test_lock_info_contains_metadata(self):
        """Lock info file contains holder metadata"""
        with ticket_lock("test_lock", self.session_dir, holder_id="test_agent"):
            info_path = self.session_dir / "locks" / "test_lock.info.json"

            with open(info_path, 'r') as f:
                info = json.load(f)

            self.assertEqual(info['holder_id'], "test_agent")
            self.assertIn('acquired_at', info)
            self.assertIn('hostname', info)

    def test_stale_lock_recovery(self):
        """Stale lock (>5min) can be stolen"""
        lock_path = self.session_dir / "locks" / "test_lock.lock"
        info_path = self.session_dir / "locks" / "test_lock.info.json"

        # Create locks directory
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        # Create stale lock (6 minutes ago)
        with open(lock_path, 'w') as f:
            pass

        stale_info = {
            'holder_id': 'crashed_agent',
            'acquired_at': time.time() - 360,  # 6 minutes ago
            'hostname': 'old_host'
        }
        with open(info_path, 'w') as f:
            json.dump(stale_info, f)

        # Should be able to acquire stale lock
        with ticket_lock("test_lock", self.session_dir, timeout_seconds=5, holder_id="new_agent"):
            # Successfully acquired
            with open(info_path, 'r') as f:
                new_info = json.load(f)

            self.assertEqual(new_info['holder_id'], "new_agent")

    def test_lock_cleanup_on_exception(self):
        """Lock is cleaned up even when exception occurs"""
        info_path = self.session_dir / "locks" / "test_lock.info.json"

        try:
            with ticket_lock("test_lock", self.session_dir):
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Lock info should be cleaned up despite exception
        self.assertFalse(info_path.exists())


class TestAtomicArtifactWriting(unittest.TestCase):
    """Test Defense 10: Atomic artifact writing with checksums"""

    def setUp(self):
        """Create test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.session_dir = self.test_dir / "session"
        self.session_dir.mkdir()

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_write_and_read_artifact(self):
        """Artifact can be written and read back"""
        content = "# Test Artifact\n\nThis is test content."

        artifact_path = write_artifact_atomic(content, self.session_dir)

        self.assertTrue(artifact_path.exists())

        read_content = read_artifact_validated(artifact_path)

        self.assertEqual(read_content, content)

    def test_checksum_file_created(self):
        """Checksum file is created alongside artifact"""
        content = "Test content"

        artifact_path = write_artifact_atomic(content, self.session_dir)
        checksum_path = Path(str(artifact_path) + ".sha256")

        self.assertTrue(checksum_path.exists())

        # Verify checksum is correct
        expected_checksum = hashlib.sha256(content.encode()).hexdigest()
        actual_checksum = checksum_path.read_text().strip()

        self.assertEqual(actual_checksum, expected_checksum)

    def test_corruption_detected(self):
        """Corrupted artifact is detected via checksum mismatch"""
        content = "Original content"

        artifact_path = write_artifact_atomic(content, self.session_dir)

        # Corrupt the artifact
        with open(artifact_path, 'w') as f:
            f.write("Corrupted content")

        # Reading should raise ValueError
        with self.assertRaises(ValueError) as context:
            read_artifact_validated(artifact_path)

        self.assertIn("checksum mismatch", str(context.exception))

    def test_missing_checksum_detected(self):
        """Missing checksum file is detected"""
        content = "Test content"

        artifact_path = write_artifact_atomic(content, self.session_dir)
        checksum_path = Path(str(artifact_path) + ".sha256")

        # Delete checksum
        checksum_path.unlink()

        # Reading should raise ValueError
        with self.assertRaises(ValueError) as context:
            read_artifact_validated(artifact_path)

        self.assertIn("missing checksum", str(context.exception))

    def test_large_artifact(self):
        """Large artifact is written correctly"""
        # 1MB artifact
        content = "A" * (1024 * 1024)

        artifact_path = write_artifact_atomic(content, self.session_dir)
        read_content = read_artifact_validated(artifact_path)

        self.assertEqual(len(read_content), len(content))
        self.assertEqual(read_content, content)

    def test_artifact_atomicity(self):
        """Artifact is not visible until write completes"""
        # This is implicitly tested by the write implementation
        # (temp file + rename), but we can verify the pattern

        artifacts_dir = self.session_dir / "artifacts"

        # Before write, no artifacts
        self.assertFalse(artifacts_dir.exists() or
                        list(artifacts_dir.glob("*.md")) if artifacts_dir.exists() else False)

        # Write artifact
        artifact_path = write_artifact_atomic("content", self.session_dir)

        # After write, exactly one artifact (no temp files)
        artifacts = list(artifacts_dir.glob("*.md"))
        self.assertEqual(len(artifacts), 1)

        # No temp files
        temp_files = list(artifacts_dir.glob(".tmp_*.md"))
        self.assertEqual(len(temp_files), 0)


def run_tests():
    """Run all Phase 1 unit tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestAtomicTicketClaiming))
    suite.addTests(loader.loadTestsFromTestCase(TestFilesystemCompatibility))
    suite.addTests(loader.loadTestsFromTestCase(TestAdvisoryLocks))
    suite.addTests(loader.loadTestsFromTestCase(TestAtomicArtifactWriting))

    # Run with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
