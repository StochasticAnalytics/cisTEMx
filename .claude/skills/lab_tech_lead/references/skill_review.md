# Skill Review Framework - Lab Tech Lead Orchestration

## Review Objectives

When reviewing a skill, coordinate the team to assess:

1. **Purpose Clarity**: Is the skill's purpose and audience well-defined?
2. **Structure Effectiveness**: Does the organization support progressive disclosure?
3. **Context Preservation**: Does it properly delegate to resources?
4. **Discoverability**: Can users find and understand when to use it?
5. **Maintainability**: Will it remain useful as the project evolves?

## Orchestration Checklist

### Pre-Review Setup
- [ ] Identify the skill under review
- [ ] Load the SKILL.md and scan directory structure
- [ ] Note any immediate structural observations
- [ ] Prepare specific areas for Red and Blue to examine

### Review Coordination

**Phase 1: Initial Assessment**
- Direct Red to examine: gaps, violations, missing elements
- Direct Blue to examine: strengths, patterns, opportunities
- Time box: ~40% of review effort

**Phase 2: Deep Dive**
- Have Red stress-test edge cases and failure modes
- Have Blue identify enhancement paths and elegant patterns
- Time box: ~40% of review effort

**Phase 3: Synthesis**
- Facilitate lunch discussion between perspectives
- Find consensus on critical issues
- Merge improvement suggestions with risk mitigation
- Time box: ~20% of review effort

## Key Review Dimensions

### 1. YAML Frontmatter Compliance
- **name**: kebab-case, ≤64 chars
- **description**: Clear when/what/who, ≤1024 chars
- Only allowed properties used
- No validation errors

### 2. Progressive Disclosure Pattern
- SKILL.md remains concise (entry point)
- Resources provide depth
- Templates enable reuse
- Scripts automate tasks

### 3. Audience Appropriateness
- Language matches audience expertise
- Examples relevant to use cases
- Complexity appropriate to skill level
- Clear about human vs sub-agent use

### 4. Structural Organization
```
skill-name/
├── SKILL.md              # Concise, delegates to resources
├── resources/           # Detailed documentation
│   ├── topic_1.md      # Specific knowledge area
│   ├── topic_2.md      # Another knowledge area
│   └── citations.md    # External dependencies
├── templates/          # Reusable patterns
└── scripts/            # Automation tools
```

### 5. Knowledge Preservation
- Captures institutional knowledge
- Documents decision rationale
- Preserves context for future iterations
- Maintains citation tracking

## Synthesis Framework

After Red and Blue report, synthesize findings using:

### Severity Classification
1. **Critical**: Breaks skill functionality or violates core principles
2. **Major**: Significantly impairs effectiveness
3. **Minor**: Improvement opportunity
4. **Enhancement**: Nice-to-have addition

### Action Prioritization
1. **Must Fix**: Critical issues blocking skill use
2. **Should Fix**: Major issues degrading experience
3. **Consider**: Minor improvements worth doing
4. **Future**: Enhancements for later iterations

### Consensus Building
- Where Red and Blue agree → Strong confidence in finding
- Where they differ → Nuanced trade-off to discuss
- Where both see opportunity → High-value improvement

## Output Template

```markdown
## Lab Tech Team Skill Review: [Skill Name]

### Executive Summary
[2-3 sentences capturing overall assessment]

### Consensus Findings
- [What all techs agree on]

### Critical Issues (Must Fix)
- [Issue]: [Red's concern + Blue's solution approach]

### Improvements (Should Consider)
- [Area]: [Current state → Suggested enhancement]

### Strengths to Preserve
- [What's working well and why]

### Recommended Actions
1. [Highest priority action]
2. [Next priority action]
3. [Additional considerations]

### Evolution Path
[How this skill could grow over time]
```

## Remember

You're facilitating a collaborative review, not judging. Your role is to:
- Extract maximum insight from both perspectives
- Build consensus without suppressing dissent
- Transform criticism into action
- Ensure review produces improvement, not just assessment