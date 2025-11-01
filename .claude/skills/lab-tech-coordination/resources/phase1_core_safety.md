# Phase 1: Core Safety - Filesystem Coordination

## Purpose

Explains atomic operations, race conditions, and filesystem compatibility for lab-tech agents using filesystem-based coordination.

## When You Need This

When you need to understand:
- Why atomic operations matter
- How to prevent race conditions
- Filesystem compatibility issues
- Lock management strategies

## Atomic Ticket Claiming (Defense 1)

### The Problem: Race Conditions

Without atomic claiming, this happens:

```
T=0.000s: Red reads pending/ticket_123.json
T=0.001s: Blue reads pending/ticket_123.json (ALSO sees it)
T=0.002s: Red claims ticket
T=0.003s: Blue claims ticket (DUPLICATE WORK!)
```

### The Solution: Hardlink Test-and-Set

Using `atomic_claim_ticket()`:
1. Attempts `os.link(ticket_path, lock_path)` - atomic on all filesystems
2. First agent to successfully create hardlink wins
3. Loser gets `FileExistsError` and knows ticket is claimed
4. Winner moves ticket to inbox and proceeds

**Why hardlinks**: `os.link()` is atomic even on NFS, Docker volumes, Windows.

### Usage

```python
ticket = atomic_claim_ticket("ticket_123", "red", session_dir)
if ticket is None:
    # Already claimed by someone else
    pass
else:
    # You won the race, process ticket
    process_ticket(ticket)
```

## Filesystem Compatibility (Defense 2)

### The Problem: Different Atomicity Guarantees

- **Local POSIX**: `os.rename()` is atomic within same filesystem
- **NFS**: Cross-directory rename may not be atomic
- **Docker volumes**: Varies by volume driver
- **Windows**: Different semantics than Linux

### The Solution: FilesystemCoordination

Detects filesystem type and adapts operations:

```python
fs = FilesystemCoordination(session_dir)
# Automatically uses correct atomic move for this filesystem
fs.atomic_move(src, dst)
```

**Detection logic**:
- Checks for `/.dockerenv` → Docker
- Parses `/proc/mounts` for NFS
- Falls back to local POSIX

**NFS-safe pattern**: Hardlink → Rename → Unlink (guarantees atomicity)

## Advisory Locks with Timeout (Defense 3)

### The Problem: Deadlocks and Abandoned Locks

Without timeouts:
```python
lock = acquire_lock("ticket_123")
# Agent crashes here
# Lock held forever, other agents blocked
```

### The Solution: ticket_lock() Context Manager

```python
with ticket_lock("ticket_123", session_dir, timeout_seconds=30) as fd:
    # Process ticket
    # Lock automatically released even if exception raised
```

**Features**:
- **Timeout**: 30-second default, prevents indefinite waiting
- **Stale detection**: Locks older than 5 minutes can be stolen (crash recovery)
- **Lock info**: Tracks who holds lock (PID, hostname) for debugging
- **Cleanup**: Context manager ensures release even on exceptions

**Lock types**:
- **Advisory** (fcntl.LOCK_EX): Cooperative - agents must check locks
- Not mandatory - won't prevent bad actors, but prevents honest mistakes

## Atomic Artifact Writing (Defense 10)

### The Problem: Partial Reads

Without atomicity:
```python
# Red agent writing artifact
write("artifact.md", large_content)  # Crashes mid-write
# Blue reads "artifact.md" - gets corrupted partial content
```

### The Solution: Temp + Rename + Checksum

Using `write_artifact_atomic()`:
1. Writes to `.tmp_<uuid>.md` (invisible to readers)
2. Calculates SHA256 checksum
3. Writes checksum to `<uuid>.md.sha256`
4. Renames `.tmp_<uuid>.md` → `<uuid>.md` (atomic visibility)

**Result**: Artifact only visible when complete AND validated.

### Usage

```python
# Write
artifact_path = write_artifact_atomic(content, session_dir)

# Read and validate
content = read_artifact_validated(artifact_path)
# Raises ValueError if checksum mismatch (corruption detected)
```

## Race Condition Scenarios Prevented

### Scenario 1: Double Ticket Claiming
**Without atomicity**: Both agents claim same ticket
**With hardlinks**: One succeeds, one gets `FileExistsError`
**CWE**: CWE-362 (Concurrent Execution using Shared Resource)

### Scenario 2: Partial File Move
**Without atomicity**: Ticket visible in both `pending/` and `active/`
**With atomic rename**: Ticket in exactly one location
**CWE**: CWE-367 (Time-of-check Time-of-use)

### Scenario 3: Corrupted Artifacts
**Without validation**: Partial writes read as complete
**With checksums**: Corruption detected, fails loudly
**CWE**: CWE-354 (Improper Validation of Integrity Check Value)

## Filesystem Compatibility Matrix

| Operation | Local POSIX | NFS | Docker | Windows | Recommendation |
|-----------|-------------|-----|--------|---------|----------------|
| `os.link()` | Atomic | Atomic | Atomic | Atomic (NTFS) | ✅ Safe for claiming |
| `os.rename()` same-dir | Atomic | Atomic | Atomic | Atomic | ✅ Safe |
| `os.rename()` cross-dir | Atomic | ⚠️ May not be | Varies | Atomic | ⚠️ Use hardlink pattern |
| `fcntl.flock()` | Advisory | Advisory | Advisory | Not available | ⚠️ Use filelock lib |
| Write+rename | Atomic visibility | Atomic visibility | Atomic visibility | Atomic visibility | ✅ Safe for artifacts |

## Troubleshooting

### "FileExistsError when claiming ticket"
**Cause**: Another agent already claimed it (expected behavior)
**Action**: Try next ticket in inbox

### "TimeoutError acquiring lock"
**Cause**: Another agent holds lock, or stale lock
**Action**: Check lock info file for holder details, consider longer timeout

### "ValueError: checksum mismatch"
**Cause**: Artifact corrupted (disk error or agent crash)
**Action**: Report to Lead, artifact should be regenerated

### "Permission denied on hardlink"
**Cause**: Cross-filesystem link (hardlinks require same filesystem)
**Action**: Ensure session_dir is on same filesystem as locks/

## Related Purple Team Findings

From purple team review:
- **Critical**: Ticket race conditions (Defense 1 addresses)
- **Critical**: Non-atomic rename on NFS (Defense 2 addresses)
- **High**: Lock implementation gaps (Defense 3 addresses)
- **Medium**: Partial artifact corruption (Defense 10 addresses)

See `.claude/cache/purple_team_review_lab_tech_parallelization.md` for full analysis.
