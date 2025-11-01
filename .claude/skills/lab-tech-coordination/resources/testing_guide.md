# Testing Guide for Lab-Tech Coordination

## Purpose

Provides testing protocols and validation strategies for lab-tech coordination primitives.

## When You Need This

When you need to:
- Validate coordination implementations
- Debug coordination failures
- Create new tests for coordination features
- Verify purple team defenses work correctly

## Running Smoke Tests

### Quick Validation

```bash
python3 .claude/skills/lab-tech-coordination/scripts/smoke_test.py
```

**Expected output**: `Total: 7/7 tests passed`

**What's tested**:
- Filesystem detection
- Atomic ticket claiming with race condition prevention
- Atomic artifact writing with checksum validation
- Advisory locks with timeout
- Session discovery from prompts
- Convergence strategy selection
- Ticket requeue tracking

### If Tests Fail

**FileNotFoundError during tests**:
- Cause: Permission issues in `/tmp`
- Fix: Check write permissions, try different temp directory

**Lock acquisition failures**:
- Cause: Stale locks from previous test run
- Fix: `rm -rf /tmp/tmp*` to clean up

**Import errors**:
- Cause: Python path issues
- Fix: Run from repo root, or `export PYTHONPATH=.`

## Unit Testing Strategy

### Phase 1 Tests (Core Safety)

**Test: Atomic ticket claiming**
```python
def test_race_condition():
    # Launch 10 agents concurrently
    # All attempt to claim same ticket
    # Verify: Exactly 1 succeeds, 9 get None
    # Verify: Lock file exists for winner
```

**Test: Filesystem compatibility**
```python
def test_nfs_atomic_move():
    # Simulate NFS by forcing hardlink+unlink pattern
    # Verify: File moved atomically
    # Verify: Never visible in two locations
```

**Test: Lock timeout**
```python
def test_lock_timeout():
    # Agent 1 acquires lock
    # Agent 2 attempts with 1s timeout
    # Verify: Agent 2 gets TimeoutError
```

**Test: Checksum validation**
```python
def test_artifact_corruption_detection():
    # Write artifact with checksum
    # Corrupt artifact file
    # Verify: read_artifact_validated() raises ValueError
```

### Phase 2 Tests (Lifecycle)

**Test: Two-phase shutdown**
```python
def test_shutdown_protocol():
    # Lead initiates shutdown
    # Verify: shutdown_signal.json created
    # Agents ACK
    # Verify: All ACKs received
    # Lead performs cleanup
    # Verify: Session archived
```

**Test: TTL enforcement**
```python
def test_ttl_expiration():
    # Create ticket with TTL=1s
    # Wait 2s
    # monitor_ticket_health()
    # Verify: Ticket in failed/ directory
    # Verify: Lock released
```

**Test: Garbage collection**
```python
def test_abandoned_session_cleanup():
    # Create session 2 hours old with no activity
    # Run cleanup_abandoned_sessions()
    # Verify: Session deleted
```

**Test: Requeue limits**
```python
def test_max_requeues():
    # Fail ticket once -> requeued
    # Fail ticket twice -> permanently_failed
    # Verify: requeue_count incremented
    # Verify: failure_history tracked
```

### Phase 3 Tests (Convergence)

**Test: Session discovery**
```python
def test_session_from_prompt():
    # Parse valid prompt
    # Verify: Correct path extracted
    # Parse invalid prompt
    # Verify: ValueError raised
```

**Test: Convergence quorum**
```python
def test_wait_for_both_agents():
    # Red writes result at T=10s
    # Blue writes result at T=20s
    # Verify: wait returns both after 20s
    # Verify: Doesn't return early with only Red
```

**Test: Strategy selection**
```python
def test_strategy_types():
    # Select adversarial_review
    # Verify: AdversarialReviewConvergence instance
    # Select parallel_decomposition
    # Verify: ParallelDecompositionConvergence instance
```

## Integration Testing Strategy

### Purple Team Scenarios

From `.claude/cache/purple_team_review_lab_tech_parallelization.md`:

**Scenario 1: Happy Path (2 agents, 2 iterations, converge)**
```python
def test_happy_path():
    # 1. Lead creates session, tickets
    # 2. Red and Blue claim, analyze
    # 3. Lead evaluates: not converged
    # 4. Lead creates iteration 2 tickets
    # 5. Red and Blue refine
    # 6. Lead evaluates: converged
    # 7. Shutdown protocol
    # Verify: Synthesis contains both perspectives
```

**Scenario 2: Agent Failure (Red crashes, ticket requeued)**
```python
def test_agent_crash_recovery():
    # 1. Red claims ticket
    # 2. Simulate crash (force exit without cleanup)
    # 3. TTL expires
    # 4. Lead requeues ticket
    # 5. New Red instance claims requeued ticket
    # 6. Completes successfully
    # Verify: Final synthesis includes recovered work
```

**Scenario 3: Timeout (Blue takes 5 minutes, TTL triggers)**
```python
def test_ttl_timeout():
    # 1. Blue claims ticket
    # 2. Simulate slow processing (sleep 350s)
    # 3. Lead's monitor_ticket_health() expires ticket
    # 4. Verify: Ticket in failed/
    # 5. Verify: Lead continues with Red's results only
```

**Scenario 4: Concurrent Sessions (No interference)**
```python
def test_concurrent_sessions():
    # 1. Start session A (reviewing skill_1)
    # 2. Start session B (reviewing skill_2)
    # 3. Run both in parallel
    # Verify: Red_A doesn't claim tickets from session_B
    # Verify: Artifacts don't mix between sessions
```

**Scenario 5: Abandoned Session (Main agent exits)**
```python
def test_abandoned_session():
    # 1. Lead creates session, starts Red/Blue
    # 2. Simulate main agent exit (kill process)
    # 3. Wait 35 minutes
    # 4. Run garbage collector
    # Verify: Session moved to abandoned/
    # Verify: Session cleaned up after 24h
```

**Scenario 6: Scalability (12 readers, parallel decomposition)**
```python
def test_parallel_decomposition():
    # 1. Lead creates 12 tickets (file chunks)
    # 2. Launch 12 reader agents
    # 3. All claim tickets, process in parallel
    # 4. Lead waits for quorum (10/12)
    # 5. Synthesize available results
    # Verify: Parallelism (all agents run simultaneously)
    # Verify: No duplicate work
```

## Performance Testing

### Metrics to Track

**Coordination overhead**:
- Time to claim ticket: Target <10ms
- Time to write artifact: Target <50ms
- Time to evaluate convergence: Target <100ms

**Scalability limits**:
- Max concurrent agents: Test up to 20
- Max session size: Test up to 100MB artifacts
- Filesystem operations/sec: Measure on NFS

### Profiling

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Run coordination operation
atomic_claim_ticket(...)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 slowest operations
```

## Test Data Generators

### Create Test Session

```python
def create_test_session():
    session_dir = Path(f"/tmp/test_session_{uuid.uuid4()}")
    session_dir.mkdir()

    # Create directory structure
    (session_dir / "tickets" / "pending").mkdir(parents=True)
    (session_dir / "agents" / "red" / "inbox").mkdir(parents=True)
    (session_dir / "agents" / "blue" / "inbox").mkdir(parents=True)
    (session_dir / "locks").mkdir()
    (session_dir / "artifacts").mkdir()

    return session_dir
```

### Create Test Ticket

```python
def create_test_ticket(session_dir, ticket_id="test_001"):
    ticket = {
        "id": ticket_id,
        "topic": "Test analysis",
        "instructions": "Perform test analysis",
        "created_at": time.time(),
        "ttl_seconds": 300,
        "requeue_count": 0,
        "max_requeues": 1
    }

    ticket_path = session_dir / "tickets" / "pending" / f"{ticket_id}.json"
    with open(ticket_path, 'w') as f:
        json.dump(ticket, f)

    return ticket
```

## Debugging Coordination Issues

### Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('lab_tech_coordination')
```

### Inspect Session State

```bash
# List all tickets
find session_dir/tickets -name "*.json" -exec cat {} \;

# Check locks
ls -la session_dir/locks/

# Review artifacts
cat session_dir/artifacts/*.md
```

### Common Issues

**"Lock file exists but no agent is working"**:
- Check lock info: `cat session_dir/locks/ticket_123.info.json`
- Age > 5 min = stale lock
- Force release: `rm session_dir/locks/ticket_123.*`

**"Convergence never reached despite quality work"**:
- Check gate evaluation logic
- Print artifact content, manually verify gates
- Lower quality_threshold temporarily

**"Agent claims ticket but doesn't process"**:
- Check agent logs for exceptions
- Verify session_dir is correct
- Check TTL hasn't expired

## Test Coverage Goals

**Phase 1 (Core Safety)**: 95% coverage
- All race conditions tested
- All filesystem types tested
- All lock scenarios tested

**Phase 2 (Lifecycle)**: 90% coverage
- All shutdown scenarios tested
- All TTL scenarios tested
- All requeue scenarios tested

**Phase 3 (Convergence)**: 85% coverage
- Both strategies tested
- All gate types tested
- Partial convergence tested

**Overall**: 90% line coverage minimum

## Continuous Testing

### Pre-commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit
python3 .claude/skills/lab-tech-coordination/scripts/smoke_test.py
if [ $? -ne 0 ]; then
    echo "Smoke tests failed, commit aborted"
    exit 1
fi
```

### CI Pipeline

```yaml
test:
  script:
    - python3 smoke_test.py
    - python3 unit_tests.py
    - python3 integration_tests.py
  artifacts:
    reports:
      coverage: coverage.xml
```

## Related Resources

- `phase1_core_safety.md` - Concepts behind safety tests
- `phase2_lifecycle.md` - Concepts behind lifecycle tests
- `phase3_convergence.md` - Concepts behind convergence tests
- Purple team review (`.claude/cache/purple_team_review_lab_tech_parallelization.md`) - Original security analysis
