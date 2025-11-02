# Red Team Plan Perspective

Critical analysis framework for identifying plan risks, gaps, and failure modes.

## Critical Analysis Approach

### Challenge Everything

**Assume the plan is wrong until proven otherwise.**

This isn't pessimism - it's rigorous analysis. The goal is to find problems BEFORE they cause failures, not after.

**Questions to ask**:
- What if this assumption doesn't hold?
- What have we overlooked?
- Where are the hidden dependencies?
- How does this break in practice?
- What's being taken for granted?

## Risky Assumptions

### Common Assumption Failures

**"This library/tool will work as documented"**

**Reality check**:
- Documentation may be outdated
- Edge cases not documented
- Performance not as claimed
- Breaking changes in updates
- Platform-specific issues

**Failure mode**: Integration takes 5x longer than planned

**Example**:
```
Plan: "Use wxWidgets for cross-platform GUI (1 week)"

Risk: wxWidgets platform differences require workarounds
      macOS behavior differs from Linux
      Threading model varies by platform

Reality: GUI work takes 5 weeks, not 1
```

---

**"We'll just use the existing codebase pattern"**

**Reality check**:
- Existing pattern may be flawed
- Technical debt may force workarounds
- Pattern may not scale to new use case
- Dependencies may have changed

**Failure mode**: Forced to refactor mid-implementation

**Example**:
```
Plan: "Follow existing GPU memory management pattern"

Risk: Existing pattern doesn't handle our memory size
      Assumes synchronous operations (we need async)
      No error handling for out-of-memory

Reality: Need to redesign memory management
```

---

**"Team has expertise in this technology"**

**Reality check**:
- "Expertise" may mean "used it once"
- Technology may have evolved since last use
- Deep knowledge gaps may exist
- Learning curve underestimated

**Failure mode**: Implementation stalls on unknowns

**Example**:
```
Plan: "Developer has C++ experience, GPU work straightforward"

Risk: CUDA programming is different from CPU C++
      Memory model understanding required
      Kernel optimization is specialized skill

Reality: Weeks lost learning CUDA fundamentals
```

---

**"This will be straightforward to integrate"**

**Reality check**:
- Integration points are never straightforward
- API mismatches require adapters
- Data format conversions add complexity
- Error handling spans boundaries

**Failure mode**: Integration consumes 50% of timeline

---

**"Performance will be adequate"**

**Reality check**:
- "Adequate" not quantified
- No profiling done
- Algorithmic complexity overlooked
- Resource constraints not measured

**Failure mode**: Late-stage performance crisis

**Example**:
```
Plan: "Process images in batch, should be fast enough"

Risk: No profiling done
      "Fast enough" not defined
      Memory usage not calculated

Reality: Runs out of memory, processing too slow for production
```

---

**"Testing will be easy"**

**Reality check**:
- Test data may not exist
- Test infrastructure may need building
- Edge cases hard to reproduce
- GPU/parallel tests are harder than sequential

**Failure mode**: Inadequate testing, bugs in production

## Missing Elements

### What Plans Often Omit

**Error handling strategy**:
```
Plan includes:
✓ Happy path implementation
✗ Error detection
✗ Error recovery
✗ Graceful degradation
✗ User error messaging

Result: First error crashes entire system
```

**Logging and observability**:
```
Plan includes:
✓ Feature implementation
✗ Debug logging
✗ Performance metrics
✗ Error tracking
✗ User action logging

Result: Production issues impossible to diagnose
```

**Documentation**:
```
Plan includes:
✓ Code implementation
✗ API documentation
✗ Architecture docs
✗ User guide
✗ Maintenance runbook

Result: Only original developer understands code
```

**Rollback strategy**:
```
Plan includes:
✓ Deployment process
✗ Rollback procedure
✗ Data migration rollback
✗ Compatibility with old version

Result: Cannot revert if deployment fails
```

**Performance testing**:
```
Plan includes:
✓ Functional testing
✗ Load testing
✗ Memory profiling
✗ Bottleneck identification
✗ Scalability validation

Result: Production load reveals critical issues
```

**Dependency management**:
```
Plan includes:
✓ External libraries listed
✗ Version pinning
✗ Dependency update strategy
✗ License compliance
✗ Vulnerability monitoring

Result: Builds break, security issues introduced
```

## Dependency Risks

### Types of Dependencies

**Technical dependencies**:
```
Dependency: Plan requires CUDA 12.0 features
Risk: Users may have CUDA 11.x
      Upgrading CUDA may break other software
      May not be available on all platforms
Mitigation needed: Fallback to CPU or older CUDA API
```

**Resource dependencies**:
```
Dependency: Plan requires GPU with 16GB memory
Risk: Users may have 8GB GPUs
      Future models may need more
      Multiple users sharing GPU
Mitigation needed: Memory-efficient algorithm or graceful degradation
```

**Timeline dependencies**:
```
Dependency: GUI work starts after core engine complete
Risk: Engine may be delayed
      Interface may need engine changes
      Parallel work could save time
Mitigation needed: Define stable interface, work in parallel
```

**Knowledge dependencies**:
```
Dependency: Plan assumes familiarity with wxWidgets threading
Risk: Developer may need to learn this
      Documentation may be poor
      Learning time not budgeted
Mitigation needed: Allocate learning time, have expert review
```

**External dependencies**:
```
Dependency: Plan relies on third-party library
Risk: Library may be unmaintained
      May have bugs affecting us
      May change API in update
Mitigation needed: Vendor/fork library, version pinning
```

### Single Points of Failure

**One person knows the system**:
```
Risk: "Only Alice understands the GPU pipeline"
Failure: Alice unavailable → work stalls
Mitigation: Documentation, knowledge sharing, pair programming
```

**One critical component**:
```
Risk: "Entire system depends on this parser working perfectly"
Failure: Parser bug → system unusable
Mitigation: Defense in depth, input validation, error handling
```

**One external service**:
```
Risk: "We call external API for critical function"
Failure: API down → system broken
Mitigation: Caching, fallback, graceful degradation
```

## Failure Scenarios

### Common Failure Patterns

**The Cascade Failure**:
```
Scenario: Small delay causes total collapse

Week 1: Library integration takes 2 days instead of 1
Week 2: Interface work delayed, uncovers library limitation
Week 3: Need to find alternative library, start over
Week 4: New library has different API, more refactoring
Week 5: Testing delayed, finds new issues
Week 6: Originally planned completion, now 50% done

Trigger: Underestimated integration complexity
Cascade: Each delay compounds, reveals new problems
Impact: 6-week project becomes 12 weeks
```

**The Hidden Complexity**:
```
Scenario: "Simple" task reveals iceberg of complexity

Task: "Add GPU acceleration (2 days)"

Reality:
- Day 1: Setup CUDA development environment
- Day 2: Port algorithm to CUDA
- Day 3: Debug memory errors
- Day 4: Discover algorithm not parallelizable as-is
- Week 2: Redesign algorithm for GPU
- Week 3: Optimize kernel performance
- Week 4: Handle edge cases, error paths
- Week 5: Testing reveals race conditions

Trigger: Insufficient upfront investigation
Impact: 2-day task becomes 5 weeks
```

**The Integration Nightmare**:
```
Scenario: Components work alone, fail together

Module A: Works perfectly in isolation (tested)
Module B: Works perfectly in isolation (tested)
Integration: Complete failure

Issues:
- Different error handling assumptions
- Timing assumptions (A expects immediate response, B is async)
- Data format mismatches
- Resource contention (both want GPU)
- State synchronization problems

Trigger: No integration testing until late
Impact: Last-minute crisis, major refactoring
```

**The Performance Cliff**:
```
Scenario: Works fine in testing, fails in production

Testing: Process 10 images, works great
Production: Process 1000 images, crashes

Issues:
- O(n²) algorithm acceptable for n=10, not n=1000
- Memory usage scales poorly
- Resource leaks accumulate
- Cache thrashing at scale
- Synchronization overhead

Trigger: No performance testing at scale
Impact: Production failure, emergency refactoring
```

## Timeline Risks

### Estimation Anti-Patterns

**Best-case estimation**:
```
Task: "Implement feature X"
Estimate: 2 days (if everything goes perfectly)

Reality:
- Assumes no blockers
- Assumes perfect knowledge
- Assumes no debugging needed
- Assumes no interruptions

Actual: 4-6 days typical
```

**Missing buffer time**:
```
Plan:
Week 1: Task A (5 days)
Week 2: Task B (5 days)
Week 3: Task C (5 days)

Reality:
- No time for code review
- No time for bug fixes
- No time for refactoring
- No time for testing
- No time for unexpected issues

Actual: Need 4-5 weeks, not 3
```

**Parallel work illusion**:
```
Plan: "Alice does A, Bob does B, done in 1 week"

Reality:
- A and B have integration points requiring coordination
- Alice and Bob need to sync frequently
- Integration work not scheduled
- Testing integration not scheduled

Actual: More like 2 weeks, not 1
```

**Dependency chains underestimated**:
```
Task sequence:
A → B → C → D (1 day each, 4 days total)

Reality:
- A delayed 1 day → B starts late
- B uncovers issue in A → rework
- C discovers B incomplete → blocked
- D waiting on C → idle time

Actual: Critical path is longer, delays cascade
```

### Red Flags in Timelines

**"Just" tasks**:
```
❌ "Just add logging" (1 hour)
❌ "Just integrate with API" (half day)
❌ "Just add tests" (2 hours)

Reality: "Just" tasks are never just. They're underestimated by 3-5x.
```

**No slack time**:
```
❌ Every day fully scheduled
❌ No buffer for unknowns
❌ No time for code review
❌ No time for refactoring

Reality: You need 20-30% slack for unexpected issues.
```

**Optimistic parallelization**:
```
❌ "These 4 tasks can be done in parallel, 1 week total"

Reality: Coordination overhead, integration time, dependencies.
         More like 2-3 weeks.
```

## Risk Assessment Framework

### Risk Scoring

**Likelihood × Impact**:

**Likelihood**:
- **High**: Has happened before, likely to happen again
- **Medium**: Could happen, some factors point to it
- **Low**: Unlikely but possible

**Impact**:
- **Severe**: Project failure, major refactoring, timeline doubles
- **Moderate**: Delay of 1-2 weeks, significant rework
- **Minor**: A few days delay, small adjustments

**Combined Risk**:
```
             │ Minor  │ Moderate │ Severe
─────────────┼────────┼──────────┼────────
High         │ MEDIUM │ HIGH     │ CRITICAL
Medium       │ LOW    │ MEDIUM   │ HIGH
Low          │ LOW    │ LOW      │ MEDIUM
```

**CRITICAL risks**: Address immediately, may be blockers
**HIGH risks**: Must mitigate before proceeding
**MEDIUM risks**: Monitor and plan mitigation
**LOW risks**: Accept or minor mitigation

### Risk Identification Checklist

**Technical risks**:
- [ ] Technology not fully understood by team?
- [ ] Dependencies on experimental/unstable libraries?
- [ ] Performance requirements not validated?
- [ ] Platform-specific issues not investigated?
- [ ] Integration points not well-defined?

**Resource risks**:
- [ ] Key person single point of failure?
- [ ] Hardware/infrastructure assumptions?
- [ ] Competing priorities for resources?
- [ ] External dependencies on other teams?

**Timeline risks**:
- [ ] Estimates based on best-case scenarios?
- [ ] No buffer time allocated?
- [ ] Critical path dependencies not mapped?
- [ ] Learning curves underestimated?
- [ ] Testing/debugging time insufficient?

**Scope risks**:
- [ ] Requirements still evolving?
- [ ] "Just one more feature" creep?
- [ ] Unclear success criteria?
- [ ] Stakeholder alignment issues?

## Output Template

```markdown
# Plan Review - Critical Analysis

## Executive Summary

**Overall risk level**: [Critical/High/Medium/Low]
**Primary concern**: [One sentence summary of biggest risk]
**Recommendation**: [Proceed/Revise plan/Major rework needed]

## Risky Assumptions

### [Assumption 1]
- **Assumption**: [What's being assumed]
- **Reality check**: [Why this may not hold]
- **Failure mode**: [What breaks if wrong]
- **Likelihood**: [High/Medium/Low that assumption fails]
- **Impact**: [Severe/Moderate/Minor if it fails]
- **Risk level**: [Critical/High/Medium/Low]

[Repeat for each risky assumption]

## Missing Elements

### Critical Gaps
- **Missing**: [What's absent from plan]
- **Why critical**: [Impact of absence]
- **Consequence**: [What fails without this]

### Important Gaps
- **Missing**: [What's absent]
- **Why important**: [Impact]

[Repeat for each gap]

## Dependency Risks

### [Dependency 1]
- **Type**: [Technical/Resource/Timeline/External/Knowledge]
- **Description**: [What we depend on]
- **Risk**: [What can go wrong]
- **Single point of failure?**: [Yes/No]
- **Impact if fails**: [Severe/Moderate/Minor]
- **Likelihood of failure**: [High/Medium/Low]
- **Risk level**: [Critical/High/Medium/Low]

[Repeat for each dependency]

## Failure Scenarios

### Most Likely Failure: [Scenario name]
- **Trigger**: [What initiates failure]
- **Progression**: [How failure develops]
- **Cascade effects**: [What else breaks]
- **Final impact**: [End result]
- **Likelihood**: [High/Medium/Low]
- **Mitigation**: [Brief suggestion - Blue will expand]

### Other Significant Failures
[Repeat format for 2-3 more scenarios]

## Timeline Risks

### Estimation Issues
- **Task**: [Which task]
- **Estimated**: [Plan's estimate]
- **Likely actual**: [Realistic estimate]
- **Delta**: [Difference]
- **Reason**: [Why underestimated]

### Critical Path Concerns
- **Path**: [Task sequence]
- **Bottleneck**: [Constraint]
- **Slack**: [Buffer time available]
- **Risk**: [What threatens timeline]

## Risk Assessment Summary

### Critical Risks (Address now)
1. [Risk]: [Brief description]
2. [Risk]: [Brief description]

### High Risks (Mitigate before starting)
1. [Risk]: [Brief description]
2. [Risk]: [Brief description]

### Medium Risks (Monitor and plan)
1. [Risk]: [Brief description]
2. [Risk]: [Brief description]

### Low Risks (Accept or minor mitigation)
1. [Risk]: [Brief description]

## Recommendations

**Before proceeding**:
- [ ] [Critical action 1]
- [ ] [Critical action 2]

**Early in execution**:
- [ ] [Important action 1]
- [ ] [Important action 2]

**Monitor throughout**:
- [ ] [Ongoing concern 1]
- [ ] [Ongoing concern 2]
```

## Red Team Mindset

### Assume Failure

**Default assumption**: This plan will fail

**Then ask**: What specific failure modes exist?
**Then ask**: Which failures are most likely?
**Then ask**: Which failures have most impact?
**Then ask**: Can these be mitigated?

### Trust but Verify

**Assumptions in plan**: Question each one
**Estimates in plan**: Add buffer
**Dependencies in plan**: Map them all
**"Simple" tasks**: Investigate complexity

### Learn from History

**What failed before?**
- Integration tasks always take longer
- Performance issues emerge late
- "Just" tasks are never just
- Dependencies hide complexity

**What patterns repeat?**
- Underestimating learning curves
- Optimistic parallelization
- Missing error handling
- Inadequate testing time

### Checklist

Before completing red team analysis:

- [ ] Challenged every major assumption
- [ ] Identified missing elements
- [ ] Mapped all dependencies
- [ ] Described failure scenarios
- [ ] Assessed timeline realism
- [ ] Scored risks (likelihood × impact)
- [ ] Identified critical vs. high vs. medium vs. low risks
- [ ] Provided specific examples
- [ ] Made concrete recommendations
