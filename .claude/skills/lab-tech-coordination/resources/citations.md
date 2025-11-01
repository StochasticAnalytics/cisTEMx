# Skill Citations and Sources

## Purpose

Tracks external sources consulted during skill creation for maintenance and evolution tracking.

## Primary Source

### Purple Team Security Review

**Source**: `.claude/cache/purple_team_review_lab_tech_parallelization.md`
**Date**: 2025-11-01
**Review Type**: Adversarial security, safety, and robustness assessment

**Relevant Findings**:
- 15 defensive mitigations organized into 4 phases (Core Safety, Lifecycle, Convergence, Scalability)
- Concrete Python implementations for all critical operations
- Race condition analysis and atomicity guarantees
- Session lifecycle protocols
- Convergence strategies for different coordination patterns

**Linked Content**:
- `scripts/filesystem_coordination.py` - Defenses 1, 2, 3, 10
- `scripts/session_manager.py` - Defenses 4, 6, 8, 9
- `scripts/convergence.py` - Defenses 5, 7, 12
- All resource files document purple team findings

### Original Architecture Plan

**Source**: `.claude/cache/lab_tech_coordination_plan_v2.md`
**Date**: 2025-11-01
**Content**: Filesystem-based multi-agent coordination architecture

**Key Concepts**:
- Ephemeral coordination directories
- Ticket-based iteration
- Main agent as launcher pattern
- Skill-based orchestration

**Linked Content**:
- `templates/ticket_schema.json`
- `templates/convergence_schema.json`
- `templates/protocol_schema.json`

## Platform Dependencies

### Anthropic Skills

**Source**: https://github.com/anthropics/skills
**Accessed**: 2025-11-01
**Relevant Patterns**:
- Scripts directory for deterministic operations
- Progressive disclosure through resources
- Sub-agent-only skills pattern

**Linked Content**:
- `SKILL.md` - "FOR SUB-AGENTS ONLY" designation

### Python Standard Library

**Modules Used**:
- `os`, `fcntl`: Filesystem operations and locking
- `pathlib`: Path manipulation
- `json`: Data serialization
- `hashlib`: SHA256 checksums for artifact validation
- `contextlib`: Context managers for lock management
- `abc`: Abstract base classes for strategy pattern

**Linked Content**:
- `scripts/filesystem_coordination.py`
- `scripts/session_manager.py`
- `scripts/convergence.py`

## Review Protocol

When reviewing this skill for updates:

1. Check purple team report for new findings or mitigations
2. Verify Python stdlib APIs haven't changed (rare but possible)
3. Check Anthropic skills platform for new patterns
4. Review CWE classifications for updated security guidance
5. Validate that defenses remain effective against identified threats

## Version History

- 2025-11-01: Initial creation based on purple team review
- Next review: After Phase 1-3 testing validates implementations
