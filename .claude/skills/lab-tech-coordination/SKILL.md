---
name: lab-tech-coordination
description: Filesystem-based multi-agent coordination for lab-tech sub-agents (Lead, Red, Blue). Provides deterministic ticket claiming, session management, and convergence evaluation. FOR SUB-AGENTS ONLY - main agent coordinates but does not load this skill.
---

# Lab Tech Coordination

Deterministic coordination primitives for lab-tech agents operating via filesystem-based message passing.

## Purpose

This skill provides security-hardened implementations for:
- Atomic ticket claiming (prevents race conditions)
- Session lifecycle management (graceful shutdown, garbage collection)
- Convergence evaluation (quorum, timeout, pluggable strategies)

**Audience**: Lab Tech Lead, Lab Tech Red, Lab Tech Blue
**NOT for**: Main Claude agent (coordinates but doesn't execute primitives)

## Core Safety (Phase 1 - BLOCKING)

To claim tickets atomically, use `scripts/filesystem_coordination.py::atomic_claim_ticket()`.
To acquire locks with timeout, use `scripts/filesystem_coordination.py::ticket_lock()`.
To write artifacts atomically, use `scripts/filesystem_coordination.py::write_artifact_atomic()`.
To detect filesystem type, use `scripts/filesystem_coordination.py::FilesystemCoordination`.

For atomicity guarantees and race condition analysis, see `resources/phase1_core_safety.md`.

## Lifecycle Management (Phase 2 - REQUIRED)

To initiate graceful shutdown, use `scripts/session_manager.py::initiate_shutdown()`.
To check for shutdown signals, use `scripts/session_manager.py::check_shutdown_signal()`.
To clean up abandoned sessions, use `scripts/session_manager.py::cleanup_abandoned_sessions()`.
To handle failed tickets, use `scripts/session_manager.py::handle_failed_ticket()`.

For shutdown protocol and TTL enforcement, see `resources/phase2_lifecycle.md`.

## Convergence & Coordination (Phase 3 - REQUIRED)

To parse session from prompt, use `scripts/convergence.py::parse_session_from_prompt()`.
To wait for agent results, use `scripts/convergence.py::wait_for_agent_result()`.
To evaluate convergence, use `scripts/convergence.py::ConvergenceEvaluator`.
To select strategy, use `scripts/convergence.py::select_convergence_strategy()`.

For convergence strategies and quorum policies, see `resources/phase3_convergence.md`.

## Testing

To run unit tests, use `scripts/test_coordination.py`.

For integration test scenarios, see `resources/testing_guide.md`.

## Templates

- `templates/ticket_schema.json` - Ticket structure with requeue tracking
- `templates/convergence_schema.json` - Convergence criteria structure
- `templates/protocol_schema.json` - Session protocol structure

## Available Resources

- `resources/phase1_core_safety.md` - Atomicity, filesystem compatibility, locks
- `resources/phase2_lifecycle.md` - Shutdown, garbage collection, TTL enforcement
- `resources/phase3_convergence.md` - Quorum, strategies, session discovery
- `resources/testing_guide.md` - Validation protocols
- `resources/citations.md` - Purple team report, Anthropic docs
