#!/usr/bin/env python3
"""
Smoke test for lab-tech coordination primitives.
Validates basic functionality of Phase 1-3 implementations.
"""

import sys
import tempfile
import shutil
from pathlib import Path
import json
import time

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
from session_manager import (
    cleanup_abandoned_sessions,
    check_shutdown_signal,
    TicketManager
)
from convergence import (
    parse_session_from_prompt,
    select_convergence_strategy
)


def test_filesystem_detection():
    """Test filesystem type detection."""
    print("Testing filesystem detection...")
    temp_dir = Path(tempfile.mkdtemp())
    try:
        fs = FilesystemCoordination(temp_dir)
        print(f"  ✓ Detected filesystem type: {fs.fs_type}")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_atomic_ticket_claiming():
    """Test atomic ticket claiming with hardlinks."""
    print("\nTesting atomic ticket claiming...")
    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Setup session structure
        session_dir = temp_dir / "session"
        tickets_dir = session_dir / "tickets" / "pending"
        tickets_dir.mkdir(parents=True)

        agents_dir = session_dir / "agents"
        (agents_dir / "red" / "inbox").mkdir(parents=True)
        (agents_dir / "blue" / "inbox").mkdir(parents=True)

        locks_dir = session_dir / "locks"
        locks_dir.mkdir(parents=True)

        # Create a test ticket
        ticket = {
            "id": "test_ticket_001",
            "topic": "Test topic",
            "instructions": "Test instructions"
        }

        ticket_path = tickets_dir / "test_ticket_001.json"
        with open(ticket_path, 'w') as f:
            json.dump(ticket, f)

        # Attempt to claim ticket
        claimed = atomic_claim_ticket("test_ticket_001", "red", session_dir)

        if claimed is None:
            print(f"  ✗ Failed to claim ticket")
            return False

        print(f"  ✓ Red agent claimed ticket: {claimed['id']}")

        # Verify ticket no longer in pending
        if ticket_path.exists():
            print(f"  ✗ Ticket still in pending directory")
            return False

        # Verify ticket in red's inbox
        red_inbox = agents_dir / "red" / "inbox" / "test_ticket_001.json"
        if not red_inbox.exists():
            print(f"  ✗ Ticket not in red's inbox")
            return False

        print(f"  ✓ Ticket moved to red's inbox")

        # Verify lock exists
        lock_path = locks_dir / "test_ticket_001.lock"
        if not lock_path.exists():
            print(f"  ✗ Lock file not created")
            return False

        print(f"  ✓ Lock file created")

        # Try to claim same ticket with blue (should fail)
        # First recreate the ticket in pending for this test
        with open(ticket_path, 'w') as f:
            json.dump(ticket, f)

        claimed_blue = atomic_claim_ticket("test_ticket_001", "blue", session_dir)
        if claimed_blue is not None:
            print(f"  ✗ Blue agent should not have claimed locked ticket")
            return False

        print(f"  ✓ Blue agent correctly prevented from claiming locked ticket")

        # Release the lock
        release_ticket("test_ticket_001", session_dir)

        if lock_path.exists():
            print(f"  ✗ Lock not released")
            return False

        print(f"  ✓ Lock released successfully")

        return True

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_atomic_artifact_writing():
    """Test atomic artifact writing with checksums."""
    print("\nTesting atomic artifact writing...")
    temp_dir = Path(tempfile.mkdtemp())

    try:
        session_dir = temp_dir / "session"
        session_dir.mkdir()

        # Write an artifact
        content = "# Test Artifact\n\nThis is test content."
        artifact_path = write_artifact_atomic(content, session_dir)

        print(f"  ✓ Artifact written: {artifact_path.name}")

        # Verify artifact exists
        if not artifact_path.exists():
            print(f"  ✗ Artifact file not created")
            return False

        # Verify checksum file exists
        checksum_path = Path(str(artifact_path) + ".sha256")
        if not checksum_path.exists():
            print(f"  ✗ Checksum file not created")
            return False

        print(f"  ✓ Checksum file created")

        # Read and validate
        read_content = read_artifact_validated(artifact_path)

        if read_content != content:
            print(f"  ✗ Content mismatch")
            return False

        print(f"  ✓ Content validated with checksum")

        # Test corruption detection
        with open(artifact_path, 'w') as f:
            f.write("corrupted content")

        try:
            read_artifact_validated(artifact_path)
            print(f"  ✗ Failed to detect corrupted artifact")
            return False
        except ValueError as e:
            print(f"  ✓ Corruption detected: {str(e)[:50]}...")

        return True

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_ticket_lock():
    """Test advisory lock with timeout."""
    print("\nTesting advisory lock with timeout...")
    temp_dir = Path(tempfile.mkdtemp())

    try:
        session_dir = temp_dir / "session"
        session_dir.mkdir()

        # Acquire lock
        with ticket_lock("test_lock", session_dir, timeout_seconds=5) as fd:
            print(f"  ✓ Lock acquired (fd={fd})")

            # Verify lock info file created
            lock_info_path = session_dir / "locks" / "test_lock.info.json"
            if not lock_info_path.exists():
                print(f"  ✗ Lock info file not created")
                return False

            print(f"  ✓ Lock info file created")

        # Verify lock released (info file deleted)
        if lock_info_path.exists():
            print(f"  ✗ Lock info file not cleaned up")
            return False

        print(f"  ✓ Lock released and cleaned up")

        return True

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_session_discovery():
    """Test session discovery from prompt."""
    print("\nTesting session discovery...")

    try:
        prompt = "[SESSION_DIR:/tmp/test_session] Perform analysis..."
        session_dir = parse_session_from_prompt(prompt)

        if str(session_dir) != "/tmp/test_session":
            print(f"  ✗ Incorrect session dir: {session_dir}")
            return False

        print(f"  ✓ Session directory parsed: {session_dir}")

        # Test missing session dir
        try:
            parse_session_from_prompt("No session dir here")
            print(f"  ✗ Should have raised ValueError")
            return False
        except ValueError:
            print(f"  ✓ ValueError raised for missing session dir")

        return True

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_convergence_strategy_selection():
    """Test convergence strategy selection."""
    print("\nTesting convergence strategy selection...")

    try:
        # Test adversarial review strategy
        strategy = select_convergence_strategy('adversarial_review', {})
        print(f"  ✓ Adversarial review strategy created: {type(strategy).__name__}")

        # Test parallel decomposition strategy
        strategy = select_convergence_strategy('parallel_decomposition', {'num_agents': 12})
        print(f"  ✓ Parallel decomposition strategy created: {type(strategy).__name__}")

        # Test unknown strategy
        try:
            select_convergence_strategy('unknown_strategy', {})
            print(f"  ✗ Should have raised ValueError for unknown strategy")
            return False
        except ValueError:
            print(f"  ✓ ValueError raised for unknown strategy")

        return True

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ticket_requeue():
    """Test ticket requeue tracking."""
    print("\nTesting ticket requeue tracking...")
    temp_dir = Path(tempfile.mkdtemp())

    try:
        session_dir = temp_dir / "session"
        session_dir.mkdir()

        manager = TicketManager(session_dir)

        # Create test ticket
        ticket = {
            "id": "test_requeue",
            "topic": "Test",
            "max_requeues": 1
        }

        # First failure - should requeue
        manager.handle_failed_ticket(ticket, "Test failure")

        if ticket['requeue_count'] != 1:
            print(f"  ✗ Requeue count incorrect: {ticket['requeue_count']}")
            return False

        print(f"  ✓ Ticket requeued (count=1)")

        # Second failure - should permanently fail
        manager.handle_failed_ticket(ticket, "Second failure")

        if ticket['state'] != 'permanently_failed':
            print(f"  ✗ Ticket state incorrect: {ticket['state']}")
            return False

        print(f"  ✓ Ticket permanently failed after max requeues")

        # Verify failure history
        if len(ticket['failure_history']) != 2:
            print(f"  ✗ Failure history incorrect length")
            return False

        print(f"  ✓ Failure history tracked")

        return True

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("Lab-Tech Coordination Smoke Tests")
    print("=" * 60)

    tests = [
        ("Filesystem Detection", test_filesystem_detection),
        ("Atomic Ticket Claiming", test_atomic_ticket_claiming),
        ("Atomic Artifact Writing", test_atomic_artifact_writing),
        ("Ticket Lock", test_ticket_lock),
        ("Session Discovery", test_session_discovery),
        ("Convergence Strategy Selection", test_convergence_strategy_selection),
        ("Ticket Requeue", test_ticket_requeue),
    ]

    results = []
    for name, test_func in tests:
        result = test_func()
        results.append((name, result))

    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    passed = sum(1 for _, r in results if r)
    total = len(results)

    print(f"\nTotal: {passed}/{total} tests passed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
