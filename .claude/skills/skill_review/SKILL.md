---
name: skill_review
description: Red/blue team quality analysis for Claude Code skills - critical and constructive perspectives
---

# Skill Review

Framework for evaluating skill quality through critical analysis (red team) and constructive improvement (blue team).

## Purpose

Systematic skill evaluation using:
- **Red team perspective**: Identify what's broken, unclear, missing, will fail
- **Blue team perspective**: Identify what's strong, improvement opportunities, patterns worth replicating

## When to Use

- Reviewing new skills before deployment
- Quality assessment of existing skills
- Identifying improvement opportunities
- Validating progressive disclosure effectiveness
- Ensuring alignment with Anthropic best practices

## For Red Team (Critical Analysis)

Load `resources/red_perspective.md` for:
- **Clarity assessment**: What's unclear, ambiguous, contradictory
- **Completeness check**: What's missing, what gaps exist
- **Failure mode analysis**: What will break, edge cases not handled
- **Usability problems**: What makes the skill hard to use

### Key Questions Red Asks

- What assumptions does this skill make?
- Where will users get confused?
- What's missing that will cause failures?
- What contradictions exist?
- How will this break in practice?

## For Blue Team (Constructive Analysis)

Load `resources/blue_perspective.md` for:
- **Strength identification**: What works well, effective patterns
- **Improvement opportunities**: How to enhance without breaking
- **Pattern replication**: What's worth copying to other skills
- **Quick wins**: High-impact, low-effort improvements

### Key Questions Blue Asks

- What's working well here?
- What patterns should we replicate?
- How can we incrementally improve?
- What are the quick wins?
- What's the long-term vision?

## Shared Reference Material

Load `resources/skill_best_practices.md` for:
- Anthropic's progressive disclosure guidelines
- Skill structure best practices
- Common skill anti-patterns
- SKILL.md formatting conventions
- Context preservation strategies

## Evaluation Criteria

### Progressive Disclosure (3 Levels)

**Level 1 - SKILL.md header** (~100 lines):
- Framework and approach
- When to use / when not to use
- Key concepts overview

**Level 2 - Resources** (loaded on-demand):
- Detailed methodologies
- Reference materials
- Examples and templates

**Level 3 - External references** (via web fetch):
- Official documentation
- Deep-dive articles

### Quality Gates

1. **Clarity**: Can users understand what the skill does?
2. **Completeness**: Does it provide enough information?
3. **Correctness**: Is the information accurate?
4. **Usability**: Is it easy to apply?
5. **Maintainability**: Will it stay current?

## Output Format

### Red Team Output

```markdown
# Skill Review - Critical Analysis

## Clarity Issues
- **[Issue]**: [Location], [Problem], [Impact]

## Completeness Gaps
- **Missing**: [What's absent], [Why it matters]

## Failure Modes
- **[Scenario]**: [What breaks], [How to trigger]

## Usability Problems
- **[Problem]**: [Difficulty], [User impact]

## Contradictions
- **[Location 1]** vs **[Location 2]**: [Conflict]

## Severity Assessment
- Critical: [Issues requiring immediate fix]
- Major: [Significant problems]
- Minor: [Small improvements]
```

### Blue Team Output

```markdown
# Skill Review - Constructive Analysis

## Strengths
- **[Strength]**: [What works], [Why it's effective]

## Patterns Worth Replicating
- **[Pattern]**: [Description], [Where else to use]

## Improvement Opportunities

### Quick Wins (< 1 hour)
- [Improvement]: [How to implement]

### Medium-Term (1 day)
- [Enhancement]: [Implementation approach]

### Long-Term Vision
- [Strategic improvement]: [Benefits]

## Enhancement Recommendations
[Prioritized list with effort estimates]
```

## Progressive Disclosure

**Level 1** (this file): Framework and approach
**Level 2**: Load red_perspective.md or blue_perspective.md based on your role
**Level 3**: Load skill_best_practices.md for Anthropic guidelines

## Version

1.0 - Initial unified skill review framework
