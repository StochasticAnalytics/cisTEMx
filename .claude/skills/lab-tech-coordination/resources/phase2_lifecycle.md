# Phase 2: Session Lifecycle Management

## Purpose

Explains session initialization, shutdown protocols, garbage collection, and failure recovery for lab-tech agents.

## When You Need This

When you need to understand:
- How sessions are created and cleaned up
- Shutdown coordination between agents
- TTL enforcement and ticket expiration
- Ticket requeue policies and failure tracking

## Two-Phase Shutdown Protocol (Defense 6)

### The Problem: Cleanup Races

Without coordination:
```
T=100s: Lead decides to synthesize, deletes session_dir
T=100.5s: Red finishing ticket, tries to write artifact
Result: FileNotFoundError, partial results lost
```

### The Solution: Signal + ACK + Cleanup

**Phase 1: Signal Shutdown**
Lead writes `shutdown_signal.json`:
```json
{
  "shutdown_initiated": true,
  "reason": "converged",
  "acks_required": ["red", "blue"],
  "acks_received": []
}
```

**Phase 2: Agents ACK**
Each agent:
1. Checks `shutdown_signal.json` periodically
2. Finishes current ticket if nearly done (>90% complete)
3. Adds self to `acks_received`
4. Exits gracefully

**Phase 3: Lead Cleans Up**
After all ACKs received (or 30s timeout):
- Moves session_dir to `archived_{name}` (not deleted immediately)
- Archived sessions cleaned up later by garbage collection

### Usage for Agents

```python
# In agent main loop
if check_shutdown_signal(session_dir, "red"):
    # Shutdown initiated
    if current_ticket_progress > 0.9:
        finish_current_ticket()
    # Already ACK'd by check_shutdown_signal()
    sys.exit(0)
```

### Usage for Lead

```python
orchestrator = LeadOrchestrator(session_dir)
orchestrator.initiate_shutdown(reason="converged")
# Waits for ACKs, then archives session
```

## TTL Enforcement (Defense 4)

### The Problem: Stuck Tickets

Tickets can get stuck when:
- Agent crashes mid-processing
- Infinite loop in analysis
- Network partition (on NFS)

Without TTL, stuck tickets block progress forever.

### The Solution: Lead Monitors Expiration

**Ticket TTL**: Each ticket has `ttl_seconds` (default: 300 = 5 minutes)

**Lead monitoring loop**:
```python
# Every 30 seconds
orchestrator.monitor_ticket_health()
```

**When ticket expires**:
1. Force-release lock (agent crashed or stuck)
2. Move ticket to `failed/` (not deleted, preserved for analysis)
3. Add `failure_reason: "TTL expired (300s)"`
4. Lead can requeue or handle in synthesis

### TTL Values

- **Initial analysis**: 300s (5 min) - enough for thorough first pass
- **Iteration**: 180s (3 min) - focused refinement
- **Synthesis**: 600s (10 min) - aggregating multiple artifacts

Adjust based on task complexity.

## Session Garbage Collection (Defense 8)

### The Problem: Orphaned Sessions

Sessions can be abandoned when:
- Main agent exits unexpectedly (Ctrl+C)
- Claude timeout (2 min default for Task)
- System crash
- Bug in coordination logic

Without cleanup, `.claude/cache/` fills with orphaned directories.

### The Solution: Periodic Scanning

**Run on Lead startup**:
```python
cleanup_abandoned_sessions()
```

**Detection criteria**:
- **Incomplete sessions**: No `session_id.txt` → delete immediately
- **Old sessions**: No activity for 1 hour → delete
- **Idle sessions**: No state updates for 30 minutes → move to `abandoned/`

**Safety**:
- Checks state file timestamps (agents write heartbeats)
- Moves to quarantine before deletion (allows inspection)
- Runs automatically, no manual cleanup needed

### Avoiding False Positives

Agents should update `state.json` periodically:
```python
state = {
    'agent': 'red',
    'last_heartbeat': time.time(),
    'current_ticket': ticket_id
}
write_json(f"{session_dir}/agents/red/state.json", state)
```

## Ticket Requeue Tracking (Defense 9)

### The Problem: Infinite Retry Loops

Without limits:
```
Agent crashes → ticket requeued
Agent crashes again → ticket requeued again
Repeat forever...
```

### The Solution: Requeue Counter + Failure History

**Ticket schema** includes:
```json
{
  "requeue_count": 0,
  "max_requeues": 1,
  "failure_history": [
    {
      "failed_at": "...",
      "failure_reason": "Agent timeout",
      "requeued": true
    }
  ]
}
```

**Policy**:
1. First failure: Requeue (attempt 2)
2. Second failure: Permanently failed
3. Preserve in `permanently_failed/` for debugging

### Usage

```python
manager = TicketManager(session_dir)
manager.handle_failed_ticket(ticket, failure_reason="Analysis timeout")
# Automatically checks requeue_count and handles appropriately
```

### When Tickets Permanently Fail

**Lead's responsibility**:
- Check `permanently_failed/` directory
- Note missing perspectives in synthesis
- Include warning: "Red perspective unavailable due to permanent failure"
- Provide partial synthesis if valuable

**Don't**: Silently ignore permanently failed tickets

## Session Lifecycle Flow

```
1. Main Agent invokes Lab Tech Lead
   ↓
2. Lead creates session directory:
   lab_tech_coordination_2025-11-01T12-00-00_<uuid>/
   ↓
3. Lead initializes structures:
   - tickets/{pending,active,completed,failed,permanently_failed}/
   - agents/{red,blue}/{inbox,outbox}/
   - locks/
   - artifacts/
   - session_id.txt, convergence.json, protocol.json
   ↓
4. Lead prepares initial tickets, signals ready
   ↓
5. Main agent invokes Red and Blue in parallel
   ↓
6. Agents check inboxes, claim tickets, process
   ↓
7. Lead monitors: TTL checks, convergence evaluation
   ↓
8. If converged or max iterations:
   Lead initiates shutdown
   ↓
9. Agents ACK shutdown, exit gracefully
   ↓
10. Lead archives session, synthesis complete
   ↓
11. Garbage collector cleans up old archives (daily)
```

## Failure Recovery Scenarios

### Scenario 1: Agent Crashes Mid-Ticket

**Detection**: TTL expires (ticket in `active/` for >300s)
**Recovery**:
1. Lead force-releases lock
2. Moves ticket to `failed/`
3. Requeues if requeue_count < max_requeues
4. New agent claims requeued ticket

### Scenario 2: Main Agent Exits Early

**Detection**: No state updates for 30 minutes
**Recovery**:
1. Garbage collector detects idle session
2. Moves to `abandoned/` (preserves work)
3. Human can inspect and resume if needed
4. Auto-deleted after 24 hours

### Scenario 3: Lead Crashes Before Synthesis

**Detection**: Lead state.json not updated
**Recovery**:
1. Main agent sees Lead timeout
2. Can re-invoke Lead with same session_dir
3. Lead resumes: reads artifacts, synthesizes
4. Or: Main agent reads artifacts directly

## Timeouts Reference

| Component | Timeout | Reason |
|-----------|---------|--------|
| Ticket TTL | 300s (5 min) | Prevents stuck agents |
| Lock acquisition | 30s | Prevents deadlocks |
| Stale lock | 300s (5 min) | Crash recovery |
| Shutdown ACK | 30s | Graceful exit |
| Session idle | 1800s (30 min) | Garbage collection |
| Session age | 3600s (1 hr) | Incomplete cleanup |

## Troubleshooting

### "Shutdown timeout, forcing cleanup"
**Cause**: Agent didn't ACK within 30s (crashed or stuck)
**Action**: Normal - Lead forces cleanup anyway
**Prevention**: Agents should check shutdown signal frequently

### "Ticket permanently failed after N attempts"
**Cause**: Agent consistently failing on this ticket
**Action**: Review failure_history, debug agent issue
**Prevention**: Fix root cause, increase TTL if needed

### "Session directory already exists"
**Cause**: UUID collision (astronomically rare) or cleanup failed
**Action**: Use different session_dir name
**Prevention**: Include timestamp + UUID in directory name

## Related Purple Team Findings

From purple team review:
- **Critical**: Session cleanup races (Defense 6 addresses)
- **High**: Orphaned agents (Defense 8 addresses)
- **Medium**: TTL enforcement unclear (Defense 4 addresses)
- **Medium**: Failed ticket recovery (Defense 9 addresses)

See `.claude/cache/purple_team_review_lab_tech_parallelization.md` for full analysis.
