#!/usr/bin/env python3
"""
Setup a test session for lab-tech coordination validation.

Creates complete directory structure with all required files for
testing the iterative convergence workflow.
"""

import sys
import json
import uuid
from pathlib import Path
from datetime import datetime


def setup_test_session(base_dir: Path = None) -> Path:
    """
    Create a complete lab-tech coordination session directory.

    Args:
        base_dir: Base directory for session (defaults to .claude/cache/)

    Returns:
        Absolute path to created session directory
    """
    if base_dir is None:
        base_dir = Path("/workspaces/cisTEMx/.claude/cache")

    # Create timestamped session directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = base_dir / f"lab_tech_session_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating session directory: {session_dir}")

    # Create ticket directories
    (session_dir / "tickets" / "pending").mkdir(parents=True)
    (session_dir / "tickets" / "active").mkdir(parents=True)
    (session_dir / "tickets" / "completed").mkdir(parents=True)
    (session_dir / "tickets" / "failed").mkdir(parents=True)
    (session_dir / "tickets" / "permanently_failed").mkdir(parents=True)
    print("  ✓ Ticket directories created")

    # Create agent directories
    for agent in ["red", "blue", "lead"]:
        (session_dir / "agents" / agent / "inbox").mkdir(parents=True)
        (session_dir / "agents" / agent / "outbox").mkdir(parents=True)

        # Create initial state file
        state_file = session_dir / "agents" / agent / "state.json"
        state_file.write_text(json.dumps({
            "agent_name": agent,
            "status": "initialized",
            "last_heartbeat": datetime.now().isoformat()
        }, indent=2))
    print("  ✓ Agent directories created (red, blue, lead)")

    # Create coordination directories
    (session_dir / "artifacts").mkdir()
    (session_dir / "locks").mkdir()
    print("  ✓ Coordination directories created")

    # Create session ID file
    session_id = str(uuid.uuid4())
    session_id_file = session_dir / "session_id.txt"
    session_id_file.write_text(session_id)
    print(f"  ✓ Session ID created: {session_id}")

    # Create convergence configuration
    convergence_config = {
        "strategy": "adversarial_review",
        "max_iterations": 3,
        "quality_threshold": 0.8,
        "convergence_gates": [
            {
                "id": "comprehensive_coverage",
                "description": "Both perspectives address all key aspects (≥3 sections)",
                "met": False
            },
            {
                "id": "specific_examples",
                "description": "Concrete examples provided (code blocks or 'Example:' text)",
                "met": False
            },
            {
                "id": "actionable_recommendations",
                "description": "Clear actionable guidance (should, must, implement, fix, etc.)",
                "met": False
            }
        ],
        "current_iteration": 1,
        "current_quality": 0.0,
        "converged": False
    }

    convergence_file = session_dir / "convergence.json"
    convergence_file.write_text(json.dumps(convergence_config, indent=2))
    print("  ✓ Convergence configuration created")

    # Create session metadata
    metadata = {
        "session_id": session_id,
        "created_at": datetime.now().isoformat(),
        "status": "active",
        "review_target": "methodological-skill-creation",
        "agents": ["red", "blue", "lead"]
    }

    metadata_file = session_dir / "session_metadata.json"
    metadata_file.write_text(json.dumps(metadata, indent=2))
    print("  ✓ Session metadata created")

    print(f"\n✅ Session ready: {session_dir.absolute()}")
    return session_dir.absolute()


if __name__ == "__main__":
    session_path = setup_test_session()
    print(f"\nSession directory: {session_path}")
    print(f"\nUse this in agent prompts:")
    print(f"[SESSION_DIR:{session_path}]")
