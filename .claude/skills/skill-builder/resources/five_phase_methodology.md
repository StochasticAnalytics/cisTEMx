# Five-Phase Skill Creation Methodology

A comprehensive guide to creating effective skills that preserve context and enable delegation.

## Phase 1: Analysis & Discovery

### Objective
Understand what knowledge exists, who uses it, and how it should be packaged.

### Key Questions
1. **Domain**: What specific capability or knowledge area?
2. **Audience**: You, sub-agents, or both?
3. **Frequency**: How often accessed?
4. **Context**: What decisions does it enable?
5. **Dependencies**: What other knowledge required?

### Audit Process
```
1. Map the knowledge domain
2. Identify natural boundaries
3. Determine access patterns
4. Define success criteria
5. Check for existing similar skills
```

### Research Requirements

**When creating skills that require external knowledge**, conduct thorough research:

1. **Deep Domain Research**:
   - Review ALL relevant information shared in the user's prompt
   - Consult official documentation (language specs, library docs, tool manuals)
   - Search trusted web resources (Stack Overflow, official blogs, authoritative tutorials)
   - Look for expert practices, common patterns, and known pitfalls
   - Document findings in `.claude/cache/` for reference

2. **Parallel Research for Breadth**:
   - You may use **multiple parallel Task() invocations** to research different aspects simultaneously
   - **CRITICAL: Use unique filenames to avoid race conditions**
   - Naming pattern: `.claude/cache/<topic>_<subtopic>_research.md`
   - Example for git-history skill:
     - `.claude/cache/git_history_bisect_research.md`
     - `.claude/cache/git_history_churn_research.md`
     - `.claude/cache/git_history_testing_research.md`
   - Each parallel task writes to its own file
   - After all parallel tasks complete, synthesize into final document
   - This maximizes research depth and breadth while preventing file conflicts

3. **Research Source Priority** (from CLAUDE.md):
   - Official documentation (language specs, library docs)
   - Project documentation (CLAUDE.md files, architecture docs)
   - Git history (what actually worked/failed)
   - Your own notes (check before reinventing!)
   - Trusted community (highly-voted StackOverflow, official issues)
   - General internet (verify against official sources)

4. **Document Your Research**:
   - Save individual research findings to unique files (see naming above)
   - Include sources, dates accessed, key learnings
   - Note any conflicting information or open questions
   - Synthesize all parallel findings into `.claude/cache/<topic>_comprehensive_research.md`
   - This becomes foundation for `resources/citations.md`

### Output
- Clear problem statement
- Audience specification
- Success criteria
- Initial structure proposal
- Research findings document(s) (if external knowledge required)

## Phase 2: Design & Structure

### Objective
Create clear architecture with progressive disclosure.

### Directory Planning
```
skill-name/                    # kebab-case naming
├── SKILL.md                   # Entry point (concise!)
├── resources/                 # Detailed content
│   ├── methodology.md        # How-to procedures
│   ├── reference.md          # Quick lookups
│   ├── examples.md           # Patterns/samples
│   └── troubleshooting.md    # Problem resolution
├── templates/                 # Reusable structures
└── scripts/                   # Automation tools
```

### YAML Frontmatter Design

**Constraints:**
- `name`: Max 64 characters
- `description`: Max 1024 characters
- No other properties without explicit documentation

**Description Formula:**
```
What it does (capability) +
When to use it (triggers) +
Who uses it (audience) +
Key outcome (what it enables)
```

**Example:**
```yaml
---
name: git-workflow
description: Structured git workflow for feature development. Use when starting new features, fixing bugs, or reviewing code. For all developers. Enables consistent, clean git history.
---
```

### Progressive Disclosure Layers

1. **Metadata** (always loaded): name + description
2. **SKILL.md** (on activation): Overview + resource pointers
3. **Resources** (on demand): Detailed procedures
4. **Templates/Scripts** (when needed): Tools and patterns

## Phase 3: Implementation

### Objective
Write clear, actionable content at appropriate detail levels.

### SKILL.md Guidelines

**Keep it concise** - Aim for < 100 lines:
- Brief overview (1 paragraph)
- When to use (bullet list)
- Quick start (minimal example)
- Resource directory (what's available)

**Delegate details** - Point to resources:
```markdown
For detailed methodology, see `resources/methodology.md`.
For examples, consult `resources/examples.md`.
```

### Resource File Structure

Each resource should be self-contained:
```markdown
# Resource Title

## Purpose
What this resource provides.

## When You Need This
Specific scenarios.

## Content
[Detailed information]

## Related Resources
Links to other files.
```

### Writing Style
- Use imperative form: "To X, do Y"
- Be concrete and specific
- Include examples
- Provide validation steps

## Phase 4: Testing & Validation

### Objective
Ensure skill loads correctly and provides value.

### Testing Protocol

#### 1. Discovery Test
```bash
# Create triggering scenario
# Verify skill is suggested
# Check description clarity
```

#### 2. Execution Test
- Follow instructions exactly
- Note any ambiguities
- Verify resource loading
- Check for missing steps

#### 3. Context Test
- Load skill with other skills
- Check for context overflow
- Verify no circular dependencies

#### 4. Sub-agent Test (if applicable)
- Invoke with sub-agent
- Verify completion possible
- Check context efficiency

### Common Issues

| Issue | Solution |
|-------|----------|
| Not discovered | Refine description triggers |
| Too much context | Move detail to resources |
| Unclear instructions | Add concrete examples |
| Missing information | Add troubleshooting guide |

## Phase 5: Documentation & Evolution

### Objective
Create institutional knowledge and enable continuous improvement.

### Citation Documentation

**Every skill must include `resources/citations.md`** to track:
- External sources consulted (URLs, documentation, examples)
- Platform constraints discovered
- Patterns learned from other skills
- Date accessed for future currency checks

This enables the future skill-review-skill to maintain skill health as platforms evolve.

**Format:**
```markdown
### [Source Category]
**Source**: [URL]
**Accessed**: [YYYY-MM-DD]
**Relevant Learnings**: [What we learned]
**Linked Content**: [Where it's applied in this skill]
```

### Journal Entry Template
```markdown
## Skill Created: [name]

**Source**: [Original documentation/need]
**Created**: [Date]
**Audience**: [You/sub-agents/both]
**Dependencies**: [Other skills required]

**Design Decisions**:
- Why structured this way
- Trade-offs considered
- Patterns used

**Lessons Learned**:
- What worked well
- What was challenging
- Future improvements

**Testing Results**:
- Discovery success
- Execution clarity
- Context efficiency
```

### Evolution Tracking
1. Version major changes in git
2. Document why changes made
3. Update dependent skills
4. Share patterns with lab techs

### Continuous Improvement
- Review usage patterns monthly
- Gather feedback from sessions
- Update based on failures
- Extract common patterns

## Success Metrics

A well-created skill exhibits:
- **High discovery rate** when relevant
- **Low context usage** relative to value
- **Clear execution** without ambiguity
- **Graceful degradation** with missing resources
- **Easy maintenance** with clear structure

## Next Steps

After creating a skill:
1. Test thoroughly
2. Document in journal
3. Share patterns with lab techs
4. Update pattern library
5. Consider related skills needed