# Commit Message Best Practices

Synthesized from recent LLM and human-centered research (2024-2025).

## Core Principles

### 1. Conciseness
- Summary line: **≤50 characters**
- Brief enough for quick scanning
- Research shows LLM-generated messages are perceived as more concise in 52% of cases

### 2. Imperative Mood
- Use "Add feature" not "Added feature" or "Adds feature"
- Matches git's own convention ("Merge branch", "Revert commit")
- More direct and action-oriented

### 3. Intent Over Description
- **Why** the change was made, not just **what** changed
- Code diffs show the "what" - commit messages should explain the "why"
- This is the hardest part and where human insight is most valuable

### 4. Conventional Types
Use prefixes to categorize commits for indexability:

- `feat:` - New feature for the user
- `fix:` - Bug fix
- `refactor:` - Code restructuring without behavior change
- `perf:` - Performance improvement
- `test:` - Adding or updating tests
- `docs:` - Documentation only
- `build:` - Build system or dependencies
- `style:` - Formatting, whitespace (not CSS)
- `chore:` - Maintenance tasks

### 5. Expressiveness
- Grammatically correct
- Clear and fluent
- Avoid jargon unless domain-specific

## Structure

```
<type>: <summary in ≤50 chars>

[Optional body: explain why, provide context]
[Wrap at 72 characters for body]

[Optional footer: references, breaking changes]
```

## Evaluation Criteria

Good commits score well on:
1. **Rationality**: Logical explanation for the change
2. **Comprehensiveness**: Covers what changed with relevant details
3. **Conciseness**: Brief and to the point
4. **Expressiveness**: Grammatically correct and fluent

## Examples

**Good:**
```
feat: Add GPU-accelerated FFT for 3D reconstruction

Implements CUDA kernels for Fourier transforms, reducing
processing time by 60% for large volumes (>1024³).

Refs: #234
```

**Good (simple):**
```
fix: Prevent crash on empty image stack
```

**Too vague:**
```
Update files
```

**Too detailed (put in body instead):**
```
fix: Change line 42 in image.cpp from foo() to bar() because...
```

## Frequency

- **Commit often**: Every discrete logical change
- Small, focused commits are easier to review and revert
- Better for git bisect when debugging

## cisTEMx Patterns

Based on recent successful commits:
- Focus on conciseness and clarity
- Explain architectural decisions in body when warranted
- Reference issues/PRs when applicable
- Scientific code benefits from explaining algorithmic choices

## Sources

- "Context Conquers Parameters: Outperforming Proprietary LLM in Commit Message Generation" (2024)
- "Automated Commit Message Generation with Large Language Models: An Empirical Study" (2024)
- Conventional Commits specification
- cisTEMx git history analysis
