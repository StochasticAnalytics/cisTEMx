# Planning Frameworks

Best practices, anti-patterns, and reference frameworks for effective project planning.

## Planning Principles

### Iterative, Not Waterfall

**Anti-pattern**:
```
1. Complete all planning (weeks 1-2)
2. Execute according to plan (weeks 3-10)
3. Realize plan was wrong (week 11)
4. Panic and improvise (week 12)
```

**Best practice**:
```
1. Plan Phase 1 in detail (week 1)
2. Execute Phase 1, learn (weeks 2-3)
3. Adjust plan based on learnings (week 4)
4. Plan Phase 2 in detail (week 4)
5. Execute Phase 2, learn (weeks 5-6)
6. Repeat
```

**Why it works**: Early feedback, adaptation, reduced waste

### Evidence-Based Estimation

**Anti-pattern**: Guessing
```
"This seems like a 2-day task"
```

**Better**: Reference class forecasting
```
"Last similar task took 4 days
 This one has comparable complexity
 Estimate: 3-5 days (accounting for learning from last time)"
```

**Best**: Probabilistic estimation
```
Best case: 2 days (if everything goes perfectly)
Likely case: 4 days (typical)
Worst case: 8 days (if major issues)

Confidence: 70% chance done in 4 days, 90% in 6 days
```

### Explicit Risk Management

**Anti-pattern**: Hope for the best
```
Plan: [Tasks and timeline]
Risks: [Not mentioned]
```

**Best practice**: Risk-driven planning
```
Plan:
- Task A (2 days)
- Task B (3 days)

Risks:
- Task A depends on library X (risk: API may not work as documented)
  Mitigation: 1-day spike to validate library first

- Task B requires GPU (risk: developer machine may lack GPU)
  Mitigation: Set up cloud GPU environment beforehand

Adjusted plan:
- Day 0: Set up cloud GPU, validate library X
- Days 1-2: Task A
- Days 3-5: Task B
- Day 6: Buffer for unknowns
```

## Estimation Frameworks

### Three-Point Estimation

**Formula**: (Best + 4×Likely + Worst) / 6

**Example**:
```
Task: Implement GPU kernel

Best case: 1 day (if trivial to port)
Likely case: 3 days (typical)
Worst case: 10 days (if major algorithm changes needed)

Expected: (1 + 4×3 + 10) / 6 = 3.8 days ≈ 4 days
```

**Benefits**:
- Acknowledges uncertainty
- Weights toward likely case
- Accounts for worst case
- More realistic than single-point estimates

### Story Points (Relative Sizing)

**Instead of absolute time**:
```
❌ "This task is 2 days"
```

**Use relative complexity**:
```
✓ "This task is 5 points (compared to baseline task which is 3 points)"
```

**Baseline examples**:
```
1 point: Trivial change (add logging line)
2 points: Simple task (add parameter to function)
3 points: Straightforward feature (add validation)
5 points: Moderate feature (implement new algorithm)
8 points: Complex feature (integrate with external system)
13 points: Very complex (redesign major component)
```

**Benefits**:
- Removes time pressure from estimation
- Focuses on complexity
- Velocity emerges from historical data
- Less political than time estimates

### Cone of Uncertainty

**Reality**: Estimates get more accurate as project progresses

```
Project start:    Estimate accuracy: 0.25x - 4x actual
After design:     Estimate accuracy: 0.5x - 2x actual
Mid-implementation: Estimate accuracy: 0.67x - 1.5x actual
Near completion:  Estimate accuracy: 0.8x - 1.25x actual
```

**Implications**:
- Early estimates are very uncertain → add large buffers
- Don't commit to fixed scope/time/cost early → constrain one, flex others
- Re-estimate as you learn → update plan regularly

**Example**:
```
Week 0 estimate: "Project will take 2-8 weeks"
Week 2 estimate: "Project will take 4-6 weeks" (more data available)
Week 4 estimate: "Project will take 5-6 weeks" (almost done)
```

## Dependency Management

### Dependency Types

**Technical dependencies**:
```
Task B depends on Task A completing
→ Critical path: A must finish before B starts
→ Schedule: Can't parallelize these tasks
→ Risk: A delays → B delays
```

**Resource dependencies**:
```
Task C and Task D both need GPU workstation
→ Constraint: Can't do both simultaneously
→ Schedule: Sequence or acquire second GPU
→ Risk: Resource contention slows both
```

**Knowledge dependencies**:
```
Task E requires understanding of wxWidgets threading
→ Prerequisite: Learning or expert consultation
→ Schedule: Add learning time or involve expert
→ Risk: Learning curve longer than expected
```

### Dependency Mapping

**Technique**: Create dependency graph

```
   A
   ↓
   B → D
   ↓   ↓
   C   E
       ↓
       F

Critical path: A → B → C (longest path)
Parallelizable: D and B (no dependency)
Bottleneck: A (blocks everything)
```

**Analysis**:
- **Critical path**: A→B→C determines minimum timeline
- **Slack**: D has slack (can be delayed without affecting timeline)
- **Risk**: A is single point of failure (all work blocked if A fails)

**Mitigation**:
- Start A immediately (highest priority)
- Can delay D if needed (has slack)
- Define clear interface between B and D (enable parallel work)

## Risk Assessment Frameworks

### Risk Matrix (Likelihood × Impact)

```
             │ Minor  │ Moderate │ Severe
─────────────┼────────┼──────────┼────────
High         │ MEDIUM │ HIGH     │ CRITICAL
Medium       │ LOW    │ MEDIUM   │ HIGH
Low          │ LOW    │ LOW      │ MEDIUM
```

**How to use**:
1. Identify each risk
2. Assess likelihood (High/Medium/Low)
3. Assess impact (Severe/Moderate/Minor)
4. Place in matrix
5. Prioritize mitigation: CRITICAL first, then HIGH, then MEDIUM

**Example**:
```
Risk: "GPU library API may not work as documented"
Likelihood: Medium (APIs sometimes differ from docs)
Impact: Severe (blocks entire GPU work)
→ HIGH risk → mitigate early with spike work
```

### Risk Mitigation Strategies

**Avoid**: Eliminate the risk
```
Risk: "New library may be unstable"
Avoid: Use well-established library instead
```

**Reduce**: Lower likelihood or impact
```
Risk: "Developer inexperienced with CUDA"
Reduce likelihood: Provide training
Reduce impact: Pair with experienced developer
```

**Transfer**: Make it someone else's problem
```
Risk: "Infrastructure might fail"
Transfer: Use managed cloud service with SLA
```

**Accept**: Acknowledge and monitor
```
Risk: "Users might request feature X"
Accept: We'll address if requested, not proactively
```

### Pre-Mortem Technique

**Process**:
1. Assume project has failed catastrophically
2. Team brainstorms: "Why did we fail?"
3. Identify failure scenarios
4. Plan mitigations for most likely/impactful

**Example**:
```
Pre-mortem session:

Failure: "GPU acceleration project failed completely"

Why it failed (team brainstorm):
- "We discovered our algorithm can't parallelize well" (HIGH)
- "GPU library had critical bugs" (MEDIUM)
- "Performance gains weren't worth the complexity" (MEDIUM)
- "Team couldn't learn CUDA fast enough" (LOW)

Mitigations:
- HIGH: Do 1-week spike to validate algorithm is parallelizable
- MEDIUM: Evaluate library thoroughly, have backup library
- MEDIUM: Define minimum performance gain threshold (5x)
- LOW: Allocate learning time, pair programming
```

## Incremental Delivery

### Vertical Slicing

**Principle**: Each increment should deliver end-to-end value

**Bad (horizontal slicing)**:
```
Sprint 1: Database layer
Sprint 2: Business logic
Sprint 3: API layer
Sprint 4: UI
→ No working system until Sprint 4
```

**Good (vertical slicing)**:
```
Sprint 1: End-to-end for User Login (DB + logic + API + UI)
Sprint 2: End-to-end for User Profile (DB + logic + API + UI)
Sprint 3: End-to-end for User Settings (DB + logic + API + UI)
→ Working system after each sprint
```

**Benefits**:
- Integration happens every sprint (not deferred)
- Can demo to users early (get feedback)
- Can ship after any sprint if priorities change

### Minimum Viable Product (MVP)

**Goal**: Smallest version that delivers value

**Not**:
- Minimum *marketable* product (that's different)
- Minimum *sellable* product (that's different)
- Prototype or proof-of-concept (those aren't delivered to users)

**MVP is**: Smallest *useful* version for real users

**Example**:
```
Full vision: "GPU-accelerated image processing with 20 filters"

MVP: "GPU-accelerated Gaussian blur filter"
- One filter (most-requested)
- GPU acceleration (core value proposition)
- Fallback to CPU (robustness)
- Tested and documented

Deferred to v2+:
- Other filters
- Multi-GPU support
- Advanced optimizations
```

**Why MVP first**:
- Validate core assumption (users want GPU acceleration)
- Learn from real usage (which filters matter most?)
- Deliver value quickly (users get blur filter now)
- Pivot if needed (before investing in all 20 filters)

### Walking Skeleton

**Concept**: Minimal end-to-end implementation

**Example**:
```
Goal: GPU-accelerated image processing system

Walking skeleton:
- Load image (simplest format, smallest size)
- Pass to GPU (simplest kernel: copy data)
- Return from GPU (no processing, just copy)
- Save image (same format)

Result: No value to users, but proves:
- Image I/O works
- GPU integration works
- Data pipeline works
- End-to-end system works
```

**When to use**:
- Complex integrations (prove they work before building features)
- New technology (validate integration before investing)
- Distributed systems (ensure components can communicate)

**Then**: Add features incrementally to working skeleton

## Timeline Planning

### Buffer Time

**Reality**: Estimates are wrong

**Hofstadter's Law**: "It always takes longer than you expect, even when you take into account Hofstadter's Law"

**Solution**: Add buffer

**How much buffer?**
```
Individual task level: 20-30% buffer
- Task estimate: 3 days
- With buffer: 4 days
- Reason: Local unknowns, debugging, distractions

Project level: 30-50% buffer
- Sum of tasks: 20 days
- With buffer: 26-30 days
- Reason: Integration, cross-task issues, scope adjustments

Organizational level: 50-100% buffer
- Project estimate: 30 days
- With organizational reality: 45-60 days
- Reason: Dependencies on other teams, priority shifts, meetings
```

**Where to put buffer**:
```
❌ Don't: Add buffer to each task estimate
   (Teams will use it all, then ask for more)

✓ Do: Add buffer at project/phase level
   (Protects against aggregate uncertainty)
```

### Critical Path Method (CPM)

**Goal**: Identify longest path through dependency graph

**Example**:
```
Tasks:
A: 2 days
B: 3 days (depends on A)
C: 4 days (depends on A)
D: 1 day (depends on B and C)

Paths:
A → B → D: 2 + 3 + 1 = 6 days
A → C → D: 2 + 4 + 1 = 7 days ← CRITICAL PATH

Insight: C is on critical path
- Delay in C delays entire project
- A and B have no slack
- Optimizing D (1 day) won't help (not on critical path)
```

**Use critical path for**:
- Identifying highest-priority tasks (on critical path)
- Finding tasks that can be delayed (off critical path)
- Determining minimum project duration (critical path length)

### Parkinson's Law

**Law**: "Work expands to fill the time available"

**Implication**:
```
If you give a 2-day task a 5-day deadline:
→ It will take 5 days

If you give a 2-day task a 2-day deadline:
→ It will take 2-3 days (with focus)
```

**Application**:
- Don't pad estimates too much (encourages inefficiency)
- Use time constraints to drive focus
- Balance: Enough time, not excessive time

## Common Planning Anti-Patterns

### The Waterfall Trap

**Anti-pattern**:
```
Months 1-2: Complete all requirements
Months 3-4: Complete all design
Months 5-8: Complete all implementation
Month 9: Test
Month 10: Realize requirements were wrong
```

**Why it fails**:
- No feedback until late
- Requirements change during implementation
- Integration issues discovered late
- Can't pivot once committed

**Alternative**: Iterative approach with frequent feedback

### The Scope Creep

**Anti-pattern**:
```
Week 1: Plan to build feature A
Week 2: "Let's add feature B, it's just a small addition"
Week 3: "Feature C would be nice to have"
Week 4: "We should really include D while we're at it"
Week 8: Project at 40% complete, timeline blown
```

**Prevention**:
- Define MVP strictly
- Track all additions as scope changes (not "just small additions")
- Defer nice-to-haves to v2
- Require formal approval for scope changes

### The Optimistic Estimate

**Anti-pattern**:
```
"This will take 2 weeks if everything goes perfectly"
→ Plan for 2 weeks
→ Reality: 4-6 weeks
```

**Why it fails**:
- Estimates based on best case
- Doesn't account for unknowns
- No buffer for problems
- Pressure to meet unrealistic deadline

**Alternative**: Estimate for likely case, add buffer

### The Big Bang Integration

**Anti-pattern**:
```
Months 1-3: Team A builds component A
Months 1-3: Team B builds component B (in isolation)
Month 4: Try to integrate A and B
Month 4: Discover A and B don't work together
Months 5-6: Frantic integration fixes
```

**Alternative**: Continuous integration
- Define interfaces early
- Integrate frequently (weekly or daily)
- Walking skeleton approach
- Catch integration issues early

### The Expert Dependency

**Anti-pattern**:
```
"Only Alice knows how the GPU pipeline works"
→ Alice goes on vacation
→ GPU work completely blocked
```

**Prevention**:
- Pair programming (knowledge sharing)
- Documentation
- Code reviews (multiple people understand code)
- Cross-training

### The Sunk Cost Fallacy

**Anti-pattern**:
```
"We've invested 6 weeks in approach A
 It's not working well
 But we can't stop now, we've invested too much"
→ Continue another 6 weeks
→ Fail
```

**Correct approach**:
```
"We've invested 6 weeks in approach A
 It's not working well
 Sunk cost: 6 weeks (can't recover)
 Remaining cost if continue: 6 more weeks + uncertain outcome
 Alternative: Pivot to approach B, 3 weeks + likely success
 Decision: Pivot (don't throw good money after bad)"
```

## Planning Checklists

### Plan Review Checklist

- [ ] Clear objectives defined
- [ ] Success criteria quantified
- [ ] Scope explicitly defined (in-scope and out-of-scope)
- [ ] Dependencies identified and mapped
- [ ] Risks identified and assessed
- [ ] Mitigation strategies for high risks
- [ ] Realistic estimates (not best-case)
- [ ] Buffer time allocated (30-50%)
- [ ] Critical path identified
- [ ] Incremental delivery plan (phases or sprints)
- [ ] Decision points defined (go/no-go criteria)
- [ ] Resource requirements clear
- [ ] Assumptions documented
- [ ] Fallback plans for high-risk items

### Estimation Review Checklist

- [ ] Based on historical data (not guesses)
- [ ] Accounts for uncertainty (best/likely/worst or ranges)
- [ ] Includes buffer time
- [ ] Accounts for integration and testing
- [ ] Accounts for code review and documentation
- [ ] Accounts for learning curve if new technology
- [ ] Validated against similar past tasks
- [ ] Reviewed by someone other than implementer

### Risk Management Checklist

- [ ] Risks identified (technical, resource, timeline, scope)
- [ ] Likelihood assessed (High/Medium/Low)
- [ ] Impact assessed (Severe/Moderate/Minor)
- [ ] Risk level calculated (likelihood × impact)
- [ ] Mitigation strategy for each High/Critical risk
- [ ] Risk owner assigned
- [ ] Mitigation cost estimated
- [ ] Residual risk after mitigation identified
- [ ] Monitoring plan for ongoing risks

## Decision Frameworks

### Go/No-Go Criteria

**Use case**: Deciding whether to proceed after investigation phase

**Template**:
```
Go criteria (must have all):
- ✓ Spike demonstrates 5x performance improvement
- ✓ Library API works as needed
- ✓ Team comfortable with technology
- ✓ Timeline realistic (3-4 weeks)

No-Go criteria (any one triggers):
- ✗ Performance improvement < 2x
- ✗ Library has critical bugs
- ✗ Team needs > 2 weeks training
- ✗ Timeline > 6 weeks

Decision: [Go/No-Go based on criteria met]
```

### Cost-Benefit Analysis

**Framework**:
```
Option A: GPU acceleration
Costs:
- Implementation: 4 weeks
- Learning curve: 1 week
- Ongoing maintenance: 0.5 day/month
- Total: ~5 weeks upfront, 0.5 day/month

Benefits:
- User time savings: 5 min → 1 min per image
- 100 users × 10 images/day × 4 min = 4000 min/day saved
- = 67 hours/day user time saved

ROI: Very high (5 weeks investment, ongoing large benefit)

Option B: CPU optimization
Costs:
- Implementation: 2 weeks
- Total: 2 weeks upfront

Benefits:
- User time savings: 5 min → 3 min per image
- 100 users × 10 images/day × 2 min = 2000 min/day saved
- = 33 hours/day user time saved

ROI: High (2 weeks investment, ongoing medium benefit)

Decision: Option A if resources allow (higher benefit)
          Option B if time-constrained (faster to deliver)
```

## References

### Classic Project Management

- **PERT**: Program Evaluation and Review Technique
- **CPM**: Critical Path Method
- **Gantt charts**: Timeline visualization
- **WBS**: Work Breakdown Structure

### Agile/Iterative Approaches

- **Scrum**: Sprints, daily standups, retrospectives
- **Kanban**: Visualize work, limit WIP, optimize flow
- **XP**: Pair programming, TDD, continuous integration
- **Lean**: Eliminate waste, amplify learning, decide late

### Risk Management

- **FMEA**: Failure Mode and Effects Analysis
- **Pre-mortem**: Assume failure, work backwards
- **Risk matrix**: Likelihood × Impact assessment
- **Monte Carlo**: Probabilistic simulation

### Estimation Techniques

- **Planning poker**: Team-based estimation
- **Wideband Delphi**: Expert consensus
- **Reference class forecasting**: Historical data
- **Three-point estimation**: Best/Likely/Worst
