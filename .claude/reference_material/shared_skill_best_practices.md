# Skill Best Practices

**Shared Reference Material for Skills**

**Referenced by**:
- `lab_tech_red` - Critical skill review analysis
- `lab_tech_blue` - Constructive skill review analysis

**Note**: This is shared reference material. Changes to this file impact multiple skills listed above.

---

Anthropic's guidelines and patterns for effective Claude Code skills.

## Progressive Disclosure (Core Principle)

### The Three-Level Model

**Level 1: SKILL.md** (~100 lines, always loaded)
- **Purpose**: What this skill does, why it exists
- **When to use**: Clear use cases
- **When NOT to use**: Scope boundaries
- **Framework**: Core concepts and approach
- **Resource map**: What each resource contains

**Level 2: resources/*.md** (loaded on-demand)
- **Methodology**: Detailed step-by-step processes
- **Examples**: Comprehensive examples and templates
- **Reference**: Lookup information, tables, checklists
- **Domain knowledge**: Specialized information

**Level 3: External** (web fetch when needed)
- **Official docs**: Language specs, library documentation
- **Deep dives**: Detailed articles, research papers
- **Current info**: Latest versions, recent changes

### Why Progressive Disclosure Matters

**Context window is limited**:
- Claude has finite context (~200k tokens, but ~50k effective working memory)
- Every line loaded is a line not available for task work
- Skills compete with user code, conversation history, tool outputs

**Cognitive load is real**:
- Users can't process 500 lines at once
- Key information gets lost in walls of text
- Clear structure helps scanability

**Different users need different depth**:
- Beginners need quick start
- Experts need advanced techniques
- Everyone needs reference material sometimes

### Anti-Pattern: The Monolith

```markdown
# ❌ BAD: Everything in SKILL.md (500 lines)

---
name: security_review
---

# Security Review

[50 lines of purpose and scope]

## Methodology

[200 lines of detailed steps]

## Examples

[150 lines of examples]

## Reference

[100 lines of vulnerability database]
```

**Problems**:
- Consumes 500 lines of context immediately
- Overwhelming to read
- Hard to find specific information
- Can't choose depth based on need

### Best Practice: Progressive Structure

```markdown
# ✅ GOOD: SKILL.md (100 lines)

---
name: security_review
---

# Security Review

[Framework and approach - 40 lines]

## Progressive Disclosure

**Level 1** (this file): Framework, when to use
**Level 2**: Load resources/red_perspective.md for vulnerability analysis
**Level 3**: Load resources/vulnerability_database.md for CWE/CVE reference

[Resource map - 30 lines]

## Quick Start

[One canonical example - 30 lines]
```

**Resources** (loaded as needed):
- `resources/red_perspective.md` (200 lines, load when analyzing vulnerabilities)
- `resources/blue_perspective.md` (200 lines, load when designing mitigations)
- `resources/vulnerability_database.md` (400 lines, load as reference)

**Benefits**:
- Initial load: 100 lines vs 500
- User controls complexity
- Context preserved for actual work

## SKILL.md Structure

### Frontmatter (Required)

```markdown
---
name: skill_name
description: One-line description for skill discovery
---
```

**Guidelines**:
- `name`: lowercase, underscores, descriptive
- `description`: 50-80 characters, what the skill does

### Recommended Sections

**1. Title and Purpose** (10-20 lines)
```markdown
# Skill Name

One paragraph: What this skill helps you do and why it exists.

## Purpose

- Key benefit 1
- Key benefit 2
- Key benefit 3
```

**2. When to Use** (10-20 lines)
```markdown
## When to Use

- Specific scenario 1
- Specific scenario 2
- Specific scenario 3

## When NOT to Use

- Wrong scenario 1 (use [other skill] instead)
- Wrong scenario 2 (this is out of scope)
- Wrong scenario 3 (prerequisites not met)
```

**3. Quick Start** (20-30 lines)
```markdown
## Quick Start

**Most common use case**: [Description]

**Steps**:
1. [First action]
2. [Second action]
3. [Expected result]

**Example**:
```
[Minimal working example]
```
```

**4. Progressive Disclosure Map** (10-20 lines)
```markdown
## Progressive Disclosure

**Level 1** (this file): [What's here]
**Level 2**: Load resources/[name].md for [purpose]
**Level 3**: Web fetch [external docs] when [condition]
```

**5. Version** (2-5 lines)
```markdown
## Version

1.0 - Initial release
```

### Total: ~80-100 lines

## Resource File Design

### Resources Should Be

**1. Focused**:
- One topic per resource
- 150-300 lines typical
- Over 400 lines? Consider splitting

**2. Self-contained**:
- Don't require reading other resources first (except SKILL.md)
- Include necessary context
- Can reference other resources for "see also"

**3. Clearly named**:
```markdown
# Good names
resources/methodology.md
resources/examples.md
resources/reference_guide.md
resources/troubleshooting.md

# Bad names
resources/misc.md
resources/part2.md
resources/stuff.md
```

**4. Structured**:
```markdown
# Resource Title

Brief intro (what this resource contains)

## Section 1

Content

## Section 2

Content

[etc.]
```

### Resource Organization Patterns

**By purpose**:
```
resources/
  methodology.md      # How to do the task
  examples.md         # Example implementations
  reference.md        # Lookup information
  troubleshooting.md  # Common issues
```

**By perspective** (red/blue teams):
```
resources/
  red_perspective.md    # Critical/adversarial view
  blue_perspective.md   # Constructive/defensive view
  shared_reference.md   # Common information
```

**By domain**:
```
resources/
  memory_safety.md      # C/C++ memory issues
  injection_attacks.md  # SQL/command injection
  crypto_vulns.md       # Cryptographic issues
```

## Common Anti-Patterns

### 1. The Kitchen Sink

**Problem**: Trying to cover every possible scenario
```markdown
## When to Use

- Scenario 1
- Scenario 2
- Scenario 3
[... 20 more scenarios ...]
```

**Solution**: Focus on core use cases, mention "and similar scenarios"
```markdown
## When to Use

- Core scenario 1
- Core scenario 2
- Core scenario 3
- Similar situations requiring [core capability]

## When NOT to Use

[Explicit boundaries]
```

### 2. No Examples

**Problem**: Instructions without concrete examples
```markdown
## Methodology

1. Analyze the code structure
2. Identify key patterns
3. Extract relevant information
4. Generate documentation
```

**Solution**: Always include at least one complete example
```markdown
## Methodology

**Example: Documenting a security vulnerability**

1. **Analyze the code**:
```cpp
char buf[256];
strcpy(buf, user_input);  // No bounds checking
```

2. **Identify pattern**: CWE-119 buffer overflow

3. **Extract details**:
   - Location: file.cpp:42
   - Input: user_input (untrusted)
   - Risk: Code execution

4. **Generate documentation**:
```markdown
**CWE-119: Buffer Overflow**
Location: file.cpp:42
Impact: Remote code execution
Mitigation: Use strncpy or std::string
```
```

### 3. Jargon Without Definition

**Problem**: Assuming user knowledge
```markdown
"Use CVSS v3.1 scoring with proper AV, AC, PR, and UI metrics"
```

**Solution**: Define terms or provide reference
```markdown
"Use CVSS v3.1 scoring (Common Vulnerability Scoring System):
- **AV** (Attack Vector): Network/Adjacent/Local/Physical
- **AC** (Attack Complexity): Low/High
- **PR** (Privileges Required): None/Low/High
- **UI** (User Interaction): None/Required

For full CVSS guide, load resources/cvss_reference.md"
```

### 4. Circular Dependencies

**Problem**: Resources require each other
```markdown
# SKILL.md
"First read resources/intro.md"

# resources/intro.md
"This assumes you've read SKILL.md sections 3-5"
```

**Solution**: Linear dependency chain
```markdown
# SKILL.md
[Self-contained framework]
"For details, load resources/intro.md"

# resources/intro.md
[Self-contained methodology]
"Build on framework from SKILL.md"
```

### 5. Inconsistent Formatting

**Problem**: No visual consistency
```markdown
## Section One
Content

### subsection
content

## SECTION TWO
Content

#### subsubsection
content
```

**Solution**: Consistent hierarchy
```markdown
## Major Section

Brief intro

### Subsection

Content

### Another Subsection

Content

## Another Major Section

Brief intro

### Subsection

Content
```

## Naming Conventions

### Skill Names

**Pattern**: `topic_focus`
- `security_review`
- `skill_review`
- `plan_review`
- `methodological_skill_creation`

**Not**:
- `SecurityReview` (no CamelCase)
- `security-review` (no dashes)
- `review` (too vague)
- `the_security_review_skill` (redundant)

### Resource Names

**Pattern**: `purpose.md` or `topic_focus.md`
- `methodology.md`
- `examples.md`
- `red_perspective.md`
- `vulnerability_database.md`

**Not**:
- `part1.md` (not descriptive)
- `misc.md` (not specific)
- `RedPerspective.md` (no CamelCase)

## Context Preservation Strategies

### Calculate Your Budget

**Typical skill context budget**:
- Target: 80-120 lines for SKILL.md
- Maximum: 200 lines (going over suggests refactoring needed)
- Resources: 150-400 lines each (loaded only when needed)

**Context math**:
```
Good progressive disclosure:
  SKILL.md: 100 lines
  Resource 1 (when needed): 200 lines
  Resource 2 (when needed): 250 lines
  Total loaded: 100 + one resource = 100-350 lines

Bad monolith:
  SKILL.md: 600 lines
  Total loaded: 600 lines (always)
  Context waste: 250-500 lines
```

### Resource Loading Guidance

**Tell users WHEN to load**:
```markdown
## Which Resource to Load?

**Analyzing vulnerabilities?**
→ Load resources/red_perspective.md

**Designing mitigations?**
→ Load resources/blue_perspective.md

**Need CWE/CVE lookup?**
→ Load resources/vulnerability_database.md

**Don't load all at once** - preserve context for your work
```

### Sub-Agent Delegation

**Pattern**: Skills designed for sub-agents
```markdown
# SKILL.md (designed for sub-agents, not main agent)

This skill is loaded by the [specific-agent] sub-agent when invoked.

Main agent: Invoke via Task tool
Sub-agent: Loads this skill and relevant resources

[Focused instructions for sub-agent role]
```

**Why**: Sub-agents have isolated context, allowing:
- Main agent preserves context
- Sub-agent gets full skill + resources
- Specialized processing without main agent overhead

## Testing Skills

### Usability Test

**Can a new user**:
1. Understand what the skill does? (read title + purpose)
2. Know when to use it? (read "when to use")
3. Know when NOT to use it? (read "when NOT to use")
4. Get started quickly? (follow quick start)
5. Find detailed info? (use resource map)

### Completeness Test

**Check for**:
- [ ] Frontmatter (name, description)
- [ ] Purpose section
- [ ] When to use
- [ ] When NOT to use
- [ ] At least one example
- [ ] Progressive disclosure map
- [ ] Version number

### Progressive Disclosure Test

**Verify**:
- [ ] SKILL.md under 200 lines
- [ ] Each resource under 500 lines
- [ ] No circular dependencies
- [ ] Clear guidance on what to load when
- [ ] Resources are self-contained

### Clarity Test

**Read aloud**:
- Does it sound natural?
- Are there confusing sentences?
- Is jargon defined?
- Are examples clear?

## Version Control

### Semantic Versioning

**Pattern**: `MAJOR.MINOR`
- **MAJOR**: Breaking changes (change in scope, major restructure)
- **MINOR**: Additions (new resources, new examples, enhancements)

**Example**:
```markdown
## Version

2.1 - Added CUDA-specific vulnerability patterns (2025-11-15)
2.0 - Restructured into red/blue perspectives (2025-11-01)
1.2 - Added troubleshooting section (2025-10-20)
1.1 - Fixed progressive disclosure (2025-10-15)
1.0 - Initial release (2025-10-01)
```

### Deprecation

**If skill becomes obsolete**:
```markdown
---
name: old_skill
description: ⚠️ DEPRECATED - Use new_skill instead
deprecated: true
replacement: new_skill
---

# ⚠️ DEPRECATED: Old Skill

**This skill is deprecated as of 2025-11-01.**

**Use instead**: `new_skill`

**Migration guide**: [How to migrate]

**This file will be removed**: 2025-12-01

---

[Original content kept for reference]
```

## Skill Discovery

### Description Best Practices

**Good descriptions** (50-80 chars):
```yaml
description: Red/blue team security analysis for vulnerability assessment
description: Quality review framework for Claude Code skills
description: Systematic plan review with critical and constructive perspectives
```

**Bad descriptions**:
```yaml
description: A skill  # Too vague
description: This skill helps you perform comprehensive security analysis including vulnerability assessment, threat modeling, penetration testing, and defense-in-depth evaluation  # Too long
description: Stuff  # Not descriptive
```

### Skill Categories

**Consider organizing**:
```
.claude/skills/
  security/
    security_review/
    penetration_testing/
  quality/
    skill_review/
    code_review/
  planning/
    plan_review/
    architecture_review/
  meta/
    methodological_skill_creation/
    skill_discovery/
```

## Maintenance

### Review Triggers

**Review skill when**:
- First 10 uses completed (capture early confusion points)
- Underlying tech changes (language update, library change)
- Anthropic releases new guidance
- Related skills created (check for overlap/gaps)
- Users report confusion
- Every 3-6 months (freshness check)

### Update Checklist

- [ ] Content still accurate?
- [ ] Examples still work?
- [ ] External references still valid?
- [ ] Progressive disclosure still appropriate?
- [ ] New anti-patterns discovered?
- [ ] Version number updated?
- [ ] Change documented in version section?

## References

### Official Anthropic Guidance

- Progressive disclosure guidelines (Claude Code documentation)
- Context preservation strategies (Claude Code documentation)
- Skill creation best practices (Anthropic documentation)
- Prompt engineering guide (Anthropic documentation)

### Skill Design Patterns

**Available in this project**:
- `methodological_skill_creation` skill: Meta-skill for creating skills
- Existing skills as examples: `security_review`, `skill_review`, `plan_review`

### Community Resources

- Claude Code community discussions
- Skill library examples
- User feedback and common issues
