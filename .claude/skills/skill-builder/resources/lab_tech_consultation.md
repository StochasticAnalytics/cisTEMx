# Lab Tech Consultation Guidelines

## When to Consult Lab Techs

### Complexity Triggers

**High Complexity Indicators**:
- Multi-domain skill spanning 3+ technical areas
- Requires expertise you don't fully possess
- Architecture decisions with long-term implications
- Performance-critical or security-sensitive operations
- Novel patterns not covered in existing skills

**Uncertainty Indicators**:
- Multiple viable approaches with unclear trade-offs
- Conflicting best practices from different sources
- Edge cases that could significantly impact design
- Integration with external systems or APIs
- Regulatory or compliance considerations

### Consultation Thresholds

| Skill Type | Complexity | Uncertainty | Consult? |
|------------|------------|-------------|----------|
| Reference | Low | Low | No |
| Reference | High | Any | Yes |
| Operational | Low | Low | No |
| Operational | Medium | High | Yes |
| Operational | High | Any | Yes |
| Decision | Any | High | Yes |
| Coordination | Medium+ | Any | Yes |

## How to Engage Lab Techs

### Preparation Before Consultation

1. **Document Current Understanding**:
   - What you're trying to accomplish
   - What you've already determined
   - Where you're uncertain

2. **Prepare Specific Questions**:
   - Not "Is this good?" but "Should I use pattern X or Y because..."
   - Include context about constraints
   - Specify what kind of feedback you need

3. **Gather Relevant Materials**:
   - Draft SKILL.md if it exists
   - Related documentation
   - Similar existing skills for comparison

### Invocation Pattern

```python
# Use the Task tool with lab-tech-lead agent
prompt = """
Review my skill design for {skill-name}.

Context: {brief description of skill purpose}

Specific Questions:
1. {Focused question about design choice}
2. {Question about implementation approach}
3. {Question about edge case handling}

Current Draft:
{paste draft SKILL.md or description}

Please provide:
- Critical analysis of potential issues
- Constructive suggestions for improvement
- Recommendation on whether to proceed or iterate
"""

subagent_type = "lab-tech-lead"
```

### What Lab Techs Provide

**Red Team (Critical Analysis)**:
- Identifies missing error handling
- Points out security vulnerabilities
- Highlights performance bottlenecks
- Questions assumptions
- Finds edge cases you haven't considered

**Blue Team (Constructive Guidance)**:
- Suggests proven patterns
- Provides implementation strategies
- Offers alternative approaches
- Shares relevant examples
- Identifies reusable components

**Lead (Synthesis)**:
- Balanced recommendation
- Prioritized action items
- Clear go/no-go decision
- Risk assessment
- Success criteria

## Common Consultation Scenarios

### Scenario 1: Architecture Review
**When**: Designing coordination between multiple components
**Ask About**:
- Communication patterns
- State management
- Error propagation
- Scalability considerations

### Scenario 2: Performance Optimization
**When**: Skill operates on large datasets or in tight loops
**Ask About**:
- Algorithm selection
- Caching strategies
- Parallel processing opportunities
- Memory management

### Scenario 3: Security Validation
**When**: Skill handles sensitive data or external inputs
**Ask About**:
- Input validation requirements
- Authorization patterns
- Audit logging needs
- Threat modeling

### Scenario 4: API Design
**When**: Skill provides interfaces for other skills
**Ask About**:
- Parameter design
- Error handling contracts
- Versioning strategy
- Backward compatibility

## Post-Consultation Actions

### Incorporating Feedback

1. **Prioritize Critical Issues**: Address Red team's critical findings first
2. **Implement Quick Wins**: Apply Blue team's easy improvements
3. **Document Decisions**: Record why you accepted or rejected specific suggestions
4. **Plan Iterations**: Schedule follow-up for complex changes

### When to Re-Consult

- After major design changes
- When new requirements emerge
- Before finalizing complex skills
- If implementation reveals new challenges

## Lab Tech Limitations

### What They DON'T Decide

- Business requirements (that's your/stakeholder's decision)
- Project priorities (consult advisor)
- Whether to create skill at all (use decision framework)
- Timeline commitments (your assessment)

### What They CAN'T Do

- Write the entire skill for you
- Make decisions without sufficient context
- Override security or compliance requirements
- Guarantee perfect solutions

## Examples of Good Consultations

### Example 1: Clear Technical Question
"I'm designing a caching skill. Should I use LRU or LFU eviction for scientific computation results where access patterns vary by workflow phase?"

### Example 2: Architecture Validation
"This coordination skill manages 5 parallel workers. Here's my state machine design. What failure modes am I not handling?"

### Example 3: Pattern Selection
"Converting a complex CLAUDE.md. It has 15 sections. Should I create one skill with extensive resources, or decompose into 3 specialized skills?"

## Consultation Anti-Patterns

❌ **Too Vague**: "Is this skill good?"
✅ **Better**: "Does this error handling pattern adequately cover network failures?"

❌ **No Context**: "How should I implement caching?"
✅ **Better**: "For caching 10GB of image tiles with 100ms access requirement, should I use..."

❌ **After the Fact**: Consulting after full implementation
✅ **Better**: Consult during design phase when changes are cheap

Remember: Lab techs are a resource for complex decisions, not a rubber stamp for simple ones. Use them wisely to get maximum value from their expertise.