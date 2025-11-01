---
name: methodological-skill-creation
description: Systematic methodology for creating effective skills from domain knowledge. Use when designing new skills, converting documentation to skills, or teaching sub-agents skill creation patterns. Provides templates, decision frameworks, and quality criteria.
---

# Methodological Skill Creation

This skill provides a systematic approach to creating high-quality skills that preserve context through progressive disclosure and delegation.

## Core Principle

**Skills are context preservation mechanisms.** They package specialized knowledge for on-demand loading, enable delegation to sub-agents, and create institutional knowledge that persists across sessions.

## Quick Start

To create a new skill:
1. Analyze the knowledge domain and audience
2. Design the directory structure
3. Write concise SKILL.md with references to resources
4. Test discovery and execution
5. Document in journal

For detailed methodology on each step, consult `resources/five_phase_methodology.md`.

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

## Available Resources

- `resources/five_phase_methodology.md` - Comprehensive skill creation process
- `resources/skill_boundaries.md` - When to create new vs. enhance existing
- `resources/skill_decomposition.md` - Splitting complex domains into multiple skills
- `resources/claude_md_conversion.md` - Converting CLAUDE.md files (Week 1 priority)
- `resources/pattern_library.md` - Successful skill patterns with examples
- `resources/anti_patterns.md` - Common pitfalls and how to avoid them
- `resources/lab_tech_consultation.md` - Protocol for technical discussion
- `templates/workflow_skill_template.md` - Template for procedural skills
- `templates/reference_skill_template.md` - Template for lookup skills
- `templates/decision_skill_template.md` - Template for decision frameworks
- `scripts/validate_skill.py` - Validate YAML frontmatter and structure

## For Skill Review Only

The following resource is for the skill-review-skill to check for updates to external dependencies:

- `resources/citations.md` - External sources and platform evolution tracking
