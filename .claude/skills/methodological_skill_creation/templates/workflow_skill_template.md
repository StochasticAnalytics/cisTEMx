# Workflow Skill Template

Use this template for skills that guide through multi-step processes.

## SKILL.md Structure

```yaml
---
name: [verb-noun-workflow]
description: [What it does]. Use when [trigger scenario]. For [audience]. Enables [outcome].
---
```

```markdown
# [Skill Title]

[One paragraph explaining what this workflow accomplishes and why it matters.]

## When to Use This Workflow

- [Specific trigger 1]
- [Specific trigger 2]
- [Specific trigger 3]

## Prerequisites

- [Required knowledge/tools]
- [Required access/permissions]

## Quick Start

[Minimal example showing the core workflow]

## Workflow Steps

1. **[Phase Name]** - [Brief description]
   - See `resources/phase1_details.md` for comprehensive guide

2. **[Phase Name]** - [Brief description]
   - See `resources/phase2_details.md` for comprehensive guide

3. **[Phase Name]** - [Brief description]
   - See `resources/phase3_details.md` for comprehensive guide

## Validation

How to verify the workflow completed successfully:
- [ ] [Validation check 1]
- [ ] [Validation check 2]

## Common Issues

For troubleshooting, see `resources/troubleshooting.md`.

## Available Resources

- `resources/phase1_details.md` - [Description]
- `resources/phase2_details.md` - [Description]
- `resources/phase3_details.md` - [Description]
- `resources/troubleshooting.md` - Common problems and solutions
- `templates/[template].md` - [Reusable templates]
```

## Resource File Structure

```markdown
# [Phase Name] Details

## Overview
[What this phase accomplishes]

## Steps

### Step 1: [Action]
[Detailed instructions]

### Step 2: [Action]
[Detailed instructions]

## Decision Points

**If [condition]:**
- [Action A]

**If [other condition]:**
- [Action B]

## Examples

### Example 1: [Scenario]
[Concrete example]

### Example 2: [Scenario]
[Concrete example]

## Validation
- [ ] [Check specific to this phase]
- [ ] [Another check]

## Next Phase
Proceed to [next phase] when [criteria met].
```

## Example Usage

### Git Feature Workflow

**SKILL.md snippet:**
```yaml
---
name: git-feature-workflow
description: Structured workflow for feature development with git. Use when starting new features or fixing bugs. For all developers. Enables clean, reviewable git history.
---
```

**Resources structure:**
```
resources/
├── branch_creation.md
├── commit_practices.md
├── pr_preparation.md
└── troubleshooting.md
```