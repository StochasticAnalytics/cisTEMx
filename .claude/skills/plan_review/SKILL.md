---
name: plan_review
description: Red/blue team plan analysis - critical risk assessment and constructive opportunity identification
---

# Plan Review

Framework for evaluating plans and architectures through critical analysis (red team) and constructive improvement (blue team).

## Purpose

Systematic plan evaluation using:
- **Red team perspective**: Identify risks, gaps, unrealistic assumptions, what will fail
- **Blue team perspective**: Identify opportunities, strengths, how to de-risk, incremental improvements

## When to Use

- Reviewing technical designs before implementation
- Evaluating project plans and timelines
- Assessing architecture proposals
- Analyzing risk mitigation strategies
- Validating requirement specifications
- Pre-mortem analysis of approaches

## For Red Team (Critical Analysis)

Load `resources/red_perspective.md` for:
- **Risk identification**: What can go wrong, what's been overlooked
- **Assumption testing**: What's assumed that may not hold
- **Dependency analysis**: Hidden dependencies, single points of failure
- **Failure mode analysis**: How this plan breaks in practice

### Key Questions Red Asks

- What assumptions are risky?
- What's missing from this plan?
- Where are the dependencies we haven't acknowledged?
- How does this fail?
- What's the most likely point of breakdown?

## For Blue Team (Constructive Analysis)

Load `resources/blue_perspective.md` for:
- **Strength identification**: What's well-planned, solid approaches
- **Opportunity recognition**: How to amplify good ideas
- **Risk mitigation**: How to address red team concerns
- **Incremental improvement**: How to evolve the plan iteratively

### Key Questions Blue Asks

- What's working well in this plan?
- How can we de-risk the risky parts?
- What quick wins can we achieve?
- How do we iterate safely?
- What's the minimum viable version?

## Shared Reference Material

Load `resources/planning_frameworks.md` for:
- Common planning anti-patterns
- Risk assessment frameworks
- Estimation best practices
- Dependency mapping techniques
- Incremental delivery strategies

## Output Format

### Red Team Output

```markdown
# Plan Review - Critical Analysis

## Executive Summary
[1-2 sentences: highest risk findings]

## Risky Assumptions

### Assumption: [What's assumed]
- **Reality check**: [Why this may not hold]
- **Failure mode**: [What breaks if assumption fails]
- **Likelihood**: [High/Medium/Low]
- **Impact**: [Severe/Moderate/Minor]

[Repeat for each risky assumption]

## Missing Elements

- **Missing**: [What's not in the plan]
- **Why it matters**: [Impact of absence]
- **Consequence**: [What fails without this]

## Dependency Risks

### Dependency: [What plan depends on]
- **Type**: [Technical/Resource/Timeline/External]
- **Risk**: [What can go wrong]
- **Single point of failure?**: [Yes/No]
- **Mitigation**: [Brief suggestion - Blue will expand]

## Failure Scenarios

### Scenario: [How plan breaks]
- **Trigger**: [What causes failure]
- **Cascade**: [How failure propagates]
- **Impact**: [Extent of damage]
- **Likelihood**: [High/Medium/Low]

## Risk Assessment

**Overall risk level**: [High/Medium/Low]
**Most likely failure point**: [Specific element]
**Biggest gap**: [What's most critically missing]
```

### Blue Team Output

```markdown
# Plan Review - Constructive Analysis

## Strengths

- **[Strength]**: [What's well-planned], [Why this is good]

## Opportunities

- **[Opportunity]**: [How to enhance the plan], [Expected benefit]

## Risk Mitigation Strategies

### For [Red Team Risk]
**Mitigation approach**: [How to address]
**Implementation**: [Specific steps]
**Cost/effort**: [Resource estimate]
**Residual risk**: [What remains after mitigation]

[Repeat for each major risk]

## Incremental Delivery Plan

**Phase 1** (Minimum viable):
- [Deliverable 1]
- [Deliverable 2]
- **De-risks**: [Which risks this addresses]

**Phase 2** (Enhanced):
- [Deliverable 3]
- [Deliverable 4]
- **De-risks**: [Which risks this addresses]

**Phase 3** (Complete):
- [Deliverable 5]
- [Deliverable 6]
- **De-risks**: [Which risks this addresses]

## Quick Wins

1. **[Improvement]**: [Low effort, high value change]
   - Effort: [time estimate]
   - Impact: [benefit]

[Repeat for 3-5 quick wins]

## Recommendations

**Priority 0** (Address before starting):
- [Blocker or critical gap]

**Priority 1** (Address in early phases):
- [Important risk mitigation]

**Priority 2** (Monitor and adjust):
- [Lower priority concerns]
```

## Progressive Disclosure

**Level 1** (this file): Framework and approach
**Level 2**: Load red_perspective.md or blue_perspective.md based on your role
**Level 3**: Load planning_frameworks.md for reference on specific techniques

## Version

1.0 - Initial unified plan review framework
