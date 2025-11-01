# Phase 3: Convergence & Coordination Strategies

## Purpose

Explains session discovery, convergence evaluation, and pluggable coordination strategies for multi-agent reviews.

## When You Need This

When you need to understand:
- How agents find their session directory
- When a review has "converged" (is complete)
- Different coordination patterns (adversarial review vs. parallel decomposition)
- Quorum policies and timeout handling

## Session Discovery (Defense 5)

### The Problem: Session Isolation

Multiple concurrent lab-tech reviews:
```
Session A: reviewing skill_1
Session B: reviewing skill_2
Session C: reviewing skill_3
```

How does Red agent know which session to join?

### The Solution: Prompt Injection

**Main agent includes session directory in prompt**:
```
"[SESSION_DIR:/path/to/session_dir] Perform critical analysis of..."
```

**Agent extracts session directory**:
```python
session_dir = parse_session_from_prompt(prompt)
# Returns: Path("/path/to/session_dir")
```

**Benefits**:
- **Explicit** - No ambient authority, no guessing
- **Isolated** - Each agent gets exactly the right session
- **Concurrent** - Multiple sessions don't interfere
- **Absolute paths** - No CWD issues

### Usage

```python
# Agent initialization
def main(prompt: str):
    try:
        session_dir = parse_session_from_prompt(prompt)
    except ValueError:
        print("ERROR: No session directory in prompt")
        sys.exit(1)

    # Now use session_dir for all coordination operations
    ticket = atomic_claim_ticket("ticket_1", "red", session_dir)
```

## Convergence Quorum (Defense 7)

### The Problem: Premature Convergence

Without quorum:
```
T=10s: Red submits comprehensive analysis
T=30s: Blue still working...
T=31s: Lead evaluates, sees Red's work, declares "converged"
Result: Missing Blue's valuable input
```

### The Solution: Wait for Both + Timeout

**Lead's convergence check**:
```python
red_result = wait_for_agent_result('red', session_dir, timeout=120)
blue_result = wait_for_agent_result('blue', session_dir, timeout=120)

if not (red_result and blue_result):
    # One or both timed out
    if red_result or blue_result:
        # Partial convergence (one succeeded)
        return evaluate_partial_convergence(red_result, blue_result)
    else:
        # Both failed
        raise CoordinationFailure("Both agents timed out")
```

**Key features**:
- **Quorum**: Requires both perspectives by default
- **Timeout**: 2 minutes per agent (prevents indefinite waiting)
- **Graceful degradation**: Partial synthesis if one agent succeeds
- **Fail-fast**: Error if both agents fail

### Timeout Strategies

- **Per-agent timeout**: Each agent gets full timeout
- **Total timeout**: Sum of agent timeouts for iteration
- **Adaptive**: Increase timeout if agents consistently timing out

## Convergence Gates

### What Are Gates?

Gates are specific quality criteria that must be met for convergence:

```json
{
  "id": "comprehensive_coverage",
  "description": "Both perspectives address all key aspects",
  "met": false
}
```

### Standard Gates for Adversarial Review

1. **Comprehensive Coverage**
   - Check: Multiple sections (##) in artifact
   - Threshold: ≥3 sections
   - Why: Ensures depth, not superficial analysis

2. **Specific Examples**
   - Check: Code blocks (```) or "Example:" text
   - Threshold: At least one concrete example
   - Why: Prevents vague generalities

3. **Actionable Recommendations**
   - Check: Imperative keywords (should, must, implement, fix)
   - Threshold: At least one recommendation
   - Why: Ensures practical value

### Gate Evaluation

```python
def check_gate_in_artifact(gate_id, artifact_path):
    content = read_file(artifact_path)

    if gate_id == 'comprehensive_coverage':
        return content.count('##') >= 3
    elif gate_id == 'specific_examples':
        return '```' in content or 'Example:' in content
    elif gate_id == 'actionable_recommendations':
        keywords = ['should', 'must', 'implement', 'fix', 'add', 'remove']
        return any(kw in content.lower() for kw in keywords)
```

**Gate passed**: Both Red AND Blue must address it
**Convergence**: Quality score ≥ threshold (default: 0.8)

## Pluggable Convergence Strategies (Defense 12)

### Why Pluggable?

Different coordination patterns need different convergence logic:
- **Adversarial review**: Red vs Blue, iterate to agreement
- **Parallel decomposition**: 12 readers, no interaction, quorum = all done
- **Hierarchical**: Sub-Leads coordinate workers, Main Lead aggregates

### Strategy Pattern

```python
class ConvergenceStrategy(ABC):
    @abstractmethod
    def initialize(self, session_dir, config):
        """Set up convergence criteria"""

    @abstractmethod
    def check_convergence(self, session_dir, iteration):
        """Return True if converged"""

    @abstractmethod
    def synthesize(self, session_dir):
        """Generate final output"""
```

### Strategy 1: Adversarial Review

**For**: Red/Blue reviews (2 agents, iterative)

**Convergence**:
- Both agents must submit
- Both must pass quality gates
- OR: Max iterations reached (3 default)

**Config**:
```python
{
  'max_iterations': 3,
  'quality_threshold': 0.8,
  'gates': ['comprehensive_coverage', 'specific_examples', 'actionable_recommendations']
}
```

**Usage**:
```python
strategy = select_convergence_strategy('adversarial_review', config)
strategy.initialize(session_dir, config)

if strategy.check_convergence(session_dir, iteration=1):
    synthesis = strategy.synthesize(session_dir)
```

### Strategy 2: Parallel Decomposition

**For**: N parallel agents (e.g., 12 file readers, no interaction)

**Convergence**:
- Quorum of agents complete (default: all N)
- OR: 80% of quorum + timeout
- No quality gates (each agent independent)

**Config**:
```python
{
  'num_agents': 12,
  'quorum': 12,  # All must complete
  'timeout': 300  # 5 min total
}
```

**Usage**:
```python
strategy = select_convergence_strategy('parallel_decomposition', {
    'num_agents': 12,
    'quorum': 10  # 10/12 required
})
```

### Selecting Strategy

**Lead's responsibility**:
```python
# Based on review type
if review_type == 'skill_review':
    strategy = select_convergence_strategy('adversarial_review', config)
elif review_type == 'parallel_analysis':
    strategy = select_convergence_strategy('parallel_decomposition', config)
```

## Partial Convergence

### When One Agent Fails

If Red succeeds but Blue times out:

**Option 1: Use partial results**
```python
def evaluate_partial_convergence(red_result, blue_result):
    if red_result and not blue_result:
        # Only Red succeeded
        return partial_synthesis(red_only=red_result)
    # etc.
```

**Option 2: Fail iteration, retry**
```
Iteration 1: Blue timeout
Lead: Requeue Blue's ticket with increased TTL
Iteration 2: Both succeed, converge
```

**Decision criteria**:
- **Critical review**: Require both (fail iteration)
- **Informational**: Partial OK (note missing perspective)
- **Time pressure**: Use what's available

## Convergence Flow Example

### Adversarial Review - 2 Iterations

**Iteration 1**:
```
T=0: Lead creates initial tickets for Red and Blue
T=1: Red claims, analyzes, writes artifact_red_1.md (60s)
T=1: Blue claims, analyzes, writes artifact_blue_1.md (80s)
T=80: Lead evaluates both artifacts
  - comprehensive_coverage: ✓ (both have 4+ sections)
  - specific_examples: ✗ (Red missing examples)
  - actionable_recommendations: ✓
  - Quality: 2/3 = 0.67 < 0.8 (threshold)
  - NOT CONVERGED
```

**Iteration 2**:
```
T=90: Lead creates refined tickets
  - Red: "Add concrete examples for each finding"
  - Blue: "Expand on implementation details"
T=91: Both claim, refine, write artifact_red_2.md, artifact_blue_2.md
T=150: Lead evaluates
  - comprehensive_coverage: ✓
  - specific_examples: ✓ (Red added examples)
  - actionable_recommendations: ✓
  - Quality: 3/3 = 1.0 ≥ 0.8
  - CONVERGED
T=151: Lead synthesizes, initiates shutdown
```

## Quality Threshold Tuning

**0.8 (default)**: Strict - most gates must pass
**0.6**: Moderate - majority of gates
**1.0**: Perfect - all gates required
**0.5**: Lenient - half of gates

**When to adjust**:
- Early iterations: Lower threshold (0.6) to make progress
- Final iteration: Raise threshold (1.0) for quality
- Time-critical: Lower threshold to avoid infinite iteration
- High-stakes: Raise threshold to ensure thoroughness

## Troubleshooting

### "Convergence never reached"
**Cause**: Gates too strict, agents can't satisfy them
**Action**: Review gate definitions, lower threshold temporarily
**Prevention**: Test gates with sample artifacts first

### "Premature convergence (low quality)"
**Cause**: Threshold too low or gates not strict enough
**Action**: Raise threshold, add more specific gates
**Prevention**: Calibrate gates with known good/bad examples

### "Agent timeout before convergence"
**Cause**: Agent too slow or TTL too short
**Action**: Increase TTL, optimize agent logic
**Prevention**: Profile agent performance, set realistic timeouts

### "Mixed perspectives (Red aggressive, Blue superficial)"
**Cause**: Gates don't enforce balance
**Action**: Add balance-specific gates (e.g., "both perspectives have examples")
**Prevention**: Design gates that require both agents to contribute

## Related Purple Team Findings

From purple team review:
- **High**: Convergence brittleness (Defense 7 addresses with quorum)
- **High**: Criteria don't generalize (Defense 12 addresses with strategies)
- **Medium**: Session discovery unclear (Defense 5 addresses with prompt injection)

See `.claude/cache/purple_team_review_lab_tech_parallelization.md` for full analysis.
