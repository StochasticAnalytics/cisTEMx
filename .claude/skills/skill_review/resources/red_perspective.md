# Red Team Skill Perspective

Critical analysis framework for identifying skill weaknesses and failure modes.

## Critical Analysis Approach

### Attack the Assumptions

**What does this skill assume about**:
- User's prior knowledge
- Available context window
- Agent capabilities
- Task complexity
- Information availability

**Common broken assumptions**:
- "Users will read the whole file" → They won't, they'll skim
- "This is self-explanatory" → It isn't
- "Progressive disclosure is obvious" → It needs explicit structure
- "Examples aren't necessary" → They're critical for understanding
- "This will only be used one way" → Users are creative

### Identify Confusion Points

**Where will users get lost?**

**Jargon without definition**:
```markdown
# Bad
"Use progressive disclosure with L1/L2/L3 resources"

# Better
"Use progressive disclosure (3-level loading):
- L1 (SKILL.md): Core framework (~100 lines)
- L2 (resources/*.md): Detailed methodology (loaded on-demand)
- L3 (web fetch): External documentation (when needed)"
```

**Unclear entry points**:
```markdown
# Bad
"## Usage"
[50 lines of different scenarios with no hierarchy]

# Better
"## Quick Start (First-Time Users)
[One canonical example]

## Common Scenarios
[Organized by use case]

## Advanced Usage
[Edge cases and complex scenarios]"
```

**Ambiguous instructions**:
```markdown
# Bad
"Load the appropriate resource when needed"

# Better
"When analyzing security vulnerabilities, load resources/red_perspective.md.
When designing mitigations, load resources/blue_perspective.md.
Load resources/vulnerability_database.md as reference for both."
```

## Completeness Gaps

### Missing Information

**"When NOT to use" section**:
- Every "when to use" needs a "when NOT to use"
- Prevents misapplication
- Sets appropriate scope boundaries

**Prerequisites**:
```markdown
# Missing
"## Purpose
This skill helps you..."

# Complete
"## Prerequisites
Before using this skill, you should:
- Have completed [related skill] if applicable
- Understand [key concept]
- Have access to [required resources]

## Purpose
This skill helps you..."
```

**Exit criteria**:
- How do users know they're done?
- What's the expected output?
- How to validate success?

### Missing Examples

**No examples = no understanding**

**Pattern**: For each instruction, provide:
1. **Bad example**: What not to do (with explanation)
2. **Good example**: What to do (with explanation)
3. **Context**: Why the good approach works

**Example of good example**:
```markdown
## Progressive Disclosure

**Bad approach** (everything in SKILL.md):
```markdown
# Skill Name (500 lines)
[All methodology, all examples, all edge cases]
```
**Problem**: Consumes entire context window, overwhelms users

**Good approach** (3 levels):
```markdown
# SKILL.md (~100 lines)
[Framework, when to use, what resources exist]

# resources/methodology.md (loaded when needed)
[Detailed steps, decision trees]

# Web fetch (external docs, loaded when needed)
[Official documentation, deep dives]
```
**Why it works**: Preserves context, progressive complexity
```

## Failure Mode Analysis

### How Will This Break?

**Circular dependencies**:
```markdown
# Will break
Skill A: "For advanced usage, see Skill B"
Skill B: "First complete Skill A"
```

**Resource explosion**:
```markdown
# Will consume entire context
SKILL.md references 10 resources
Each resource references 5 more resources
User loads 1 + 10 + 50 = 61 files
```

**Ambiguous resource loading**:
```markdown
# Unclear
"Load additional resources as needed"

# Clear
"If analyzing memory safety (C/C++), load resources/memory_safety.md
If analyzing injection attacks, load resources/injection_patterns.md
Load resources/cvss_scoring.md only when assigning severity scores"
```

### Edge Cases Not Handled

**What happens when**:
- Required information isn't available?
- User has incomplete context?
- Dependencies aren't met?
- Output format requirements conflict?
- Multiple skills overlap in scope?

**Document the boundaries**:
```markdown
## Scope Limitations

**This skill covers**: [Explicit list]
**This skill does NOT cover**: [Explicit list]
**For those cases, use**: [Alternative skill or approach]
```

## Usability Problems

### Cognitive Load Issues

**Too many decisions at once**:
```markdown
# Overload
"Choose between approach A, B, C, D, or E based on context X, Y, or Z,
considering factors 1, 2, 3, 4, 5, and constraints α, β, γ"

# Manageable
"## Decision Tree

If [simple condition], use approach A.
Otherwise, if [simple condition], use approach B.
For complex cases, load resources/decision_guide.md"
```

**No clear starting point**:
```markdown
# Confusing
[Multiple sections of equal visual weight, no hierarchy]

# Clear
"## Quick Start (90% of use cases)
[Canonical example]

## Advanced Topics
[Load resources as needed]"
```

### Structure Problems

**Wall of text**:
- Break up long paragraphs
- Use bulleted lists
- Add subheadings every 5-10 lines
- Use code blocks for examples
- Add visual hierarchy with **bold** and *emphasis*

**Inconsistent formatting**:
```markdown
# Inconsistent
## Section 1
- Item
- Item

## section 2
* item
* item

## SECTION 3
- ITEM
- ITEM
```

**No visual landmarks**:
- Users should be able to scan and find relevant sections quickly
- Consistent heading hierarchy (## for major, ### for minor)
- Consistent code block formatting
- Clear delineation between sections

## Contradictions

### Internal Conflicts

**SKILL.md says one thing, resource says another**:
```markdown
# SKILL.md
"Always use approach A"

# resources/methodology.md
"Approach B is preferred in most cases"
```

**Examples contradict instructions**:
```markdown
# Instruction
"Keep resource files under 200 lines"

# Actual resources
methodology.md: 450 lines
examples.md: 380 lines
```

### Conflicts with Anthropic Guidelines

**Check against**:
- Progressive disclosure best practices
- Context preservation strategies
- Sub-agent delegation patterns
- Tool usage policies
- Prompt engineering guidelines

**Common violations**:
- Loading too much at once
- Not using progressive disclosure
- Reinventing Anthropic patterns poorly
- Ignoring context limits
- Poor tool delegation

## Output Template

```markdown
# Skill Review - Critical Analysis

## Executive Summary
[1-2 sentences: most critical issues found]

## Clarity Issues

### Confusing Instructions
- **Location**: [Section/line]
- **Problem**: [What's unclear]
- **Impact**: [How this confuses users]
- **Example**: [Specific case where confusion occurs]

### Jargon/Assumptions
- **Term/Assumption**: [What's assumed]
- **Problem**: [Why problematic]
- **Impact**: [User confusion or failure]

[Repeat for each clarity issue]

## Completeness Gaps

### Missing Sections
- **Missing**: [What's absent]
- **Why it matters**: [Impact of absence]
- **Suggested content**: [What should be added]

### Missing Examples
- **Instruction without example**: [Which instruction]
- **Why example needed**: [Concept is abstract/complex/error-prone]

[Repeat for each gap]

## Failure Modes

### Breaking Scenarios
- **Scenario**: [What situation breaks this]
- **Failure**: [What breaks, how]
- **Likelihood**: [High/Medium/Low]
- **Impact**: [Severity of failure]

### Edge Cases Not Handled
- **Case**: [Edge case description]
- **Current behavior**: [What happens now - usually undefined]
- **Recommended handling**: [How to address]

[Repeat for each failure mode]

## Usability Problems

### Cognitive Load Issues
- **Problem**: [Too many choices/unclear entry/information overload]
- **Location**: [Where it occurs]
- **Impact**: [User paralysis, confusion, errors]

### Structure Issues
- **Problem**: [Wall of text/no hierarchy/inconsistent format]
- **Location**: [Which section]
- **Impact**: [User can't find information, gives up]

[Repeat for each usability problem]

## Contradictions

### Internal Conflicts
- **Conflict**: [SKILL.md section] contradicts [Resource section]
- **Details**: [Specific contradiction]
- **Resolution needed**: [How to reconcile]

### Conflicts with Best Practices
- **Violation**: [Which Anthropic guideline violated]
- **Current approach**: [What skill does]
- **Recommended approach**: [Align with guidelines]

[Repeat for each contradiction]

## Severity Assessment

### Critical (Fix before deployment)
- [Issue 1]: [Brief description, why critical]
- [Issue 2]: [Brief description, why critical]

### Major (Fix in next iteration)
- [Issue 1]: [Brief description, why major]
- [Issue 2]: [Brief description, why major]

### Minor (Improvements for polish)
- [Issue 1]: [Brief description]
- [Issue 2]: [Brief description]

## Risk Assessment

**Deployment readiness**: [Ready/Not ready/Needs revision]
**Likelihood of user confusion**: [High/Medium/Low]
**Likelihood of misuse**: [High/Medium/Low]
**Maintenance burden**: [High/Medium/Low]
```

## Red Team Mindset

### Questions to Ask

1. **Clarity**: If I knew nothing, would I understand this?
2. **Completeness**: What's missing that will cause failure?
3. **Robustness**: How does this break?
4. **Usability**: Where will users get stuck?
5. **Consistency**: What contradictions exist?

### Assume the Worst

- Users will skim, not read carefully
- Users will misunderstand ambiguous instructions
- Users will miss prerequisites
- Users will use this skill for unintended purposes
- Users will combine this skill with others in unexpected ways
- Resources will be loaded in random order
- Context window will be nearly full when skill loads

### Look for Patterns of Failure

**From other skills**:
- What caused problems before?
- What do users consistently misunderstand?
- What edge cases keep appearing?

**From Anthropic docs**:
- What anti-patterns are documented?
- What guidelines exist for a reason (learned from failures)?

## Checklist

Before marking review complete, verify:

- [ ] Every "when to use" has a "when NOT to use"
- [ ] Every instruction has an example
- [ ] Progressive disclosure is explicit (not assumed)
- [ ] Jargon is defined or eliminated
- [ ] Prerequisites are clearly stated
- [ ] Exit criteria are defined
- [ ] Edge cases are documented
- [ ] Failure modes are identified
- [ ] No circular dependencies
- [ ] No contradictions (internal or with Anthropic)
- [ ] Visual hierarchy is clear
- [ ] Cognitive load is manageable
- [ ] Resources are scoped appropriately (not 500+ lines each)
