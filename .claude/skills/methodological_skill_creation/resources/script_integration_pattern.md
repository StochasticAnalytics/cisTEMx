# Script Integration Pattern for Skills

## Purpose

Documents how to properly integrate executable scripts into skills for deterministic, token-efficient operations.

## When to Include Scripts

Include scripts in `scripts/` directory when:
- Code would be repeatedly rewritten in sessions (wastes tokens)
- Operations must be deterministic and reliable
- Implementation is complex but usage is simple
- Token efficiency is critical

## Directory Structure

```
skill-name/
├── SKILL.md
├── resources/
├── scripts/              # Executable implementations (Python/Bash/etc.)
│   ├── operations.py
│   └── test_operations.py
└── templates/
```

## Referencing Scripts in SKILL.md

**Critical**: Scripts must be explicitly referenced in SKILL.md so Claude knows they're available.

**Pattern:**
```markdown
To accomplish X, use `scripts/module_name.py::function_name()`.

For detailed methodology, see `resources/methodology.md`.
```

## Script Benefits

- **Token efficient**: Stay out of context until needed for patching or environment adjustments
- **Deterministic**: Reliable execution without LLM variability
- **Reusable**: Same script used across invocations without reloading
- **Testable**: Standard unit testing frameworks apply

## Basic Script Design

```python
"""Brief module purpose."""

def operation(param1, param2):
    """
    Brief description.

    Args:
        param1: Description
        param2: Description

    Returns:
        Description
    """
    # Implementation
```

## Integration Checklist

- [ ] Scripts referenced in SKILL.md with module::function notation
- [ ] Type hints and docstrings present
- [ ] Platform differences handled (if applicable)
- [ ] Test scripts included

## Source

**Based on**: Anthropic Skills skill-creator example
**Pattern**: Scripts are "executable code for tasks requiring deterministic reliability or repeatedly rewritten code"
