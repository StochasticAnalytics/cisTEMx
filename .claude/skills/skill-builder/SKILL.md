---
name: skill-builder
description: Systematic methodology for creating effective skills from domain knowledge. Use when designing new skills, converting documentation to skills, or teaching sub-agents skill creation patterns. Provides templates, decision frameworks, and quality criteria.
---

# Skill Builder

This skill provides a systematic approach to creating high-quality skills that preserve context through progressive disclosure and delegation.

## Core Principle

**Skills are context preservation mechanisms.** They package specialized knowledge for on-demand loading, enable delegation to sub-agents, and create institutional knowledge that persists across sessions.

## Quick Start

To create a new skill:
1. Analyze the knowledge domain and audience
2. **Conduct deep research if needed** (see Research Pattern below)
3. **Build in `.claude/cache/skills/<skill-name>/`** (avoids permission requests)
4. Design the directory structure
5. Write concise SKILL.md with references to resources
6. Validate the skill structure
7. **Move to `.claude/skills/` when complete** (single approval)
8. Document in journal

For detailed methodology on each step, consult `resources/five_phase_methodology.md`.

## Build Location Pattern

**Always build skills in `.claude/cache/skills/` first:**
- Avoids repeated permission requests for file creation
- Allows rapid iteration and testing
- When complete, move entire directory to `.claude/skills/` (one approval)

Example workflow:
```bash
# Build here (no permissions needed)
mkdir -p .claude/cache/skills/my-skill/{resources,scripts,templates}

# Create all files freely in cache...

# When ready, move to production (single approval)
mv .claude/cache/skills/my-skill .claude/skills/
```

## Complete Workflow Sequence

1. **Research** - Delegate to parallel Task() agents with `model='haiku'` if broad research needed
2. **Build in cache** - Create in `.claude/cache/skills/<skill-name>/` to avoid permission requests
3. **Initial draft** - Write SKILL.md, resources, templates, scripts
4. **Red/Blue review** - 1-2 iterations using serial pattern (see lab_tech_consultation.md)
5. **Implement changes** - Apply converged proposals from review
6. **Move to production** - `mv .claude/cache/skills/<name> .claude/skills/` (single approval)
7. **Validation phase** - Use unit-testing skill to validate code templates (SEPARATE step)
8. **Final commit** - Commit validated skill

**Key timing:** Validation happens AFTER move to production, not before. Directory move is near end of workflow, after Red/Blue review and implementation.

## Research Pattern for Skills Requiring External Knowledge

**Use parallel Task() invocations for deep, broad research:**

1. **Launch multiple general-purpose agents** in parallel (single message, multiple Task calls)
2. **Each agent** conducts autonomous multi-step research on a specific subtopic
3. **Each agent writes** to a uniquely-named file to avoid race conditions:
   - Pattern: `.claude/cache/<topic>_<subtopic>_research.md`
   - Example: `git_history_bisect_research.md`, `git_history_churn_research.md`
4. **After completion**, synthesize findings into comprehensive document

This maximizes research depth and breadth while respecting context limits.

## Key Constraints

**YAML Frontmatter Requirements:**
- `name`: Required, max 64 characters, kebab-case
- `description`: Required, max 1024 characters, be specific about what/when/who
- Only allowed properties: `name`, `description`, `license`, `allowed-tools`, `metadata`
- Other properties will cause validation errors

**Structure:**
```
skill-name/
├── SKILL.md              # Concise entry point (required)
├── resources/           # Detailed documentation
├── scripts/            # Automation tools
└── templates/          # Reusable patterns
```

## Decision Frameworks

### When to Create New vs. Enhance Existing

For guidance on skill boundaries, see `resources/skill_boundaries.md`.

### One Skill or Multiple?

For splitting criteria and examples, see `resources/skill_decomposition.md`.

## Common Patterns

### The Workflow Skill
Step-by-step process with decision points. See `templates/workflow_skill_template.md`.

### The Reference Skill
Quick lookup for specific information. See `templates/reference_skill_template.md`.

### The Decision Skill
Framework for making choices. See `templates/decision_skill_template.md`.

## Lab Tech Consultation

When creating complex skills, invoke lab techs for technical discussion:
1. Present your analysis and proposed structure
2. Lab techs will identify gaps and suggest enhancements
3. Incorporate synthesized recommendations

For detailed protocol, see `resources/lab_tech_consultation.md`.

## Quality Checklist

Before finalizing a skill:
- [ ] Description clearly indicates when to use (max 1024 chars)
- [ ] SKILL.md stays concise, delegates to resources
- [ ] Resources provide progressive detail
- [ ] Tested for discovery and execution
- [ ] Documented in journal with lessons learned
- [ ] Citations documented in `resources/citations.md` for maintainability

## Maintenance Patterns

### Coupled Reference Tracking

When adding cross-file references (markdown anchor links, resource pointers):

```markdown
<!-- coupled -->
See [Section Name](resources/file.md#section-anchor)
```

**Purpose:** Marks coupling points for future CI/linting verification when references break.

**Usage:**
- Add before resource references in SKILL.md
- Add in progressive disclosure navigation sections
- Enables automated maintenance burden tracking

### Validation Tags for Code Templates

Mark code templates requiring validation testing:

```python
#!/usr/bin/env python3
"""
Template description

<!-- needs validation -->
"""
```

```cpp
/**
 * Template description
 *
 * <!-- needs validation -->
 */
```

**Purpose:** Identifies templates for validation phase with unit-testing skill.

**Usage:**
- Add to all executable code templates (Python, C++, Bash)
- Validation happens in step 7 of workflow (after production move)

## Available Resources

- `resources/five_phase_methodology.md` - Comprehensive skill creation process
- `resources/skill_boundaries.md` - When to create new vs. enhance existing
- `resources/skill_decomposition.md` - Splitting complex domains into multiple skills
- `resources/claude_md_conversion.md` - Converting CLAUDE.md files (Week 1 priority)
- `resources/script_integration_pattern.md` - How to properly include scripts in skills
- `resources/lab_tech_consultation.md` - Protocol for technical discussion
- `templates/workflow_skill_template.md` - Template for procedural skills
- `templates/reference_skill_template.md` - Template for lookup skills
- `templates/decision_skill_template.md` - Template for decision frameworks
- `scripts/validate_skill.py` - Validate YAML frontmatter and structure

## For Skill Review Only

The following resource is for the skill-review-skill to check for updates to external dependencies:

- `resources/citations.md` - External sources and platform evolution tracking
