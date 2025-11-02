# Skill Citations and Sources

## Purpose

This file tracks external sources consulted during skill creation for the skill-review-skill to verify currency and identify platform updates. Not for routine skill operation.

## Citations

### Anthropic Skills Core Documentation

**Source**: https://github.com/anthropics/skills
**Accessed**: 2024-11-01
**Relevant Learnings**:
- Skills are folders with SKILL.md containing YAML frontmatter + instructions
- Progressive disclosure through bundled resources
- Skills preserve context through delegation

**Linked Content**:
- `SKILL.md` - Frontmatter constraints
- `resources/five_phase_methodology.md` - Progressive disclosure implementation

### YAML Frontmatter Constraints

**Source**: https://github.com/anthropics/skills/issues/37
**Source**: https://github.com/anthropics/skills/issues/9817
**Accessed**: 2024-11-01
**Critical Constraints**:
- name: max 64 characters, kebab-case
- description: max 1024 characters
- Only allowed properties: name, description, license, allowed-tools, metadata
- Other properties cause validation errors
- Formatting sensitivity: multi-line descriptions may be silently ignored

**Linked Content**:
- `SKILL.md` - Key Constraints section
- `resources/five_phase_methodology.md` - Phase 2: YAML Frontmatter Design

### Skill Structure Examples

**Source**: https://github.com/anthropics/skills/blob/main/skill-creator/SKILL.md
**Accessed**: 2024-11-01
**Key Patterns**:
- Keep SKILL.md concise (<100 lines ideal)
- Use "For X, see resources/Y" delegation pattern
- Imperative voice: "To accomplish X, do Y"
- Best practice: Move detailed reference to resources/

**Linked Content**:
- `SKILL.md` - Overall structure
- `resources/five_phase_methodology.md` - Phase 3: SKILL.md Guidelines

**Source**: https://github.com/anthropics/skills/blob/main/document-skills/pdf/SKILL.md
**Accessed**: 2024-11-01
**Production Skill Pattern**:
- Quick Start section with minimal example
- Progressive detail through resource references
- Task-to-tool mapping tables for quick lookup

**Linked Content**:
- `templates/workflow_skill_template.md` - Quick Start pattern

### Platform News and Updates

**Source**: https://www.anthropic.com/news/skills
**Source**: https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills
**Accessed**: 2024-11-01
**Key Concepts**:
- Skills as "structured prompts that can be created, shared, and reused"
- Three-level loading system (metadata → SKILL.md → resources)
- Context efficiency through progressive disclosure

**Linked Content**:
- `resources/five_phase_methodology.md` - Progressive Disclosure Layers

## Review Protocol

When skill-review-skill processes this file:

1. **Check each source URL** for updates/changes
2. **Compare constraints** against current implementation
3. **Identify new features** or deprecated patterns
4. **Generate update recommendations** if divergence found
5. **Flag breaking changes** that require immediate attention

## Version History

- 2024-11-01: Initial creation during methodological_skill_creation development
- Sources reflect Anthropic Skills platform as of this date
- Next review recommended: When skill-review-skill is implemented

## Notes

- Citations are for maintainability, not operational use
- This pattern should be replicated in all skills for long-term health
- External dependencies can evolve; regular review ensures alignment