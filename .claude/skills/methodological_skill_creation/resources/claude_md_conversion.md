# CLAUDE.md to Skills Conversion Guide

A temporary but critical process for Week 1 priorities. After completion, only the main CLAUDE.md at project root will remain.

## Context

During Week 1, you need to:
1. Audit all existing CLAUDE.md files in the codebase
2. Convert them to appropriate skills
3. Preserve only the root CLAUDE.md
4. Create skills catalog for future reference

## Conversion Process

### Step 1: Discovery & Cataloging

```bash
# Find all CLAUDE.md files
find . -name "CLAUDE.md" -type f | sort > claude_md_inventory.txt

# Examine each for themes
for file in $(cat claude_md_inventory.txt); do
    echo "=== $file ==="
    head -20 "$file"
done
```

### Step 2: Analysis Matrix

For each CLAUDE.md, document:

| File Path | Domain | Audience | Frequency | Skill Candidate | Priority |
|-----------|--------|----------|-----------|-----------------|----------|
| src/core/CLAUDE.md | Core architecture | You & sub-agents | High | core-architecture | 1 |
| src/gui/CLAUDE.md | GUI patterns | Sub-agents | Medium | gui-development | 2 |
| scripts/CLAUDE.md | Build system | You | High | build-system | 1 |

### Step 3: Grouping Strategy

**Merge related content** when:
- Same domain (e.g., all testing-related)
- Same audience
- Natural workflow progression
- Combined size reasonable

**Keep separate** when:
- Different domains
- Different audiences
- Independent usage patterns
- Would exceed context limits

### Step 4: Conversion Patterns

#### Pattern A: Direct Conversion
When CLAUDE.md is focused and cohesive:
```
CLAUDE.md → skill-name/SKILL.md (condensed)
         → skill-name/resources/details.md (expanded)
```

#### Pattern B: Split Conversion
When CLAUDE.md covers multiple topics:
```
CLAUDE.md → skill-1/SKILL.md (topic 1)
         → skill-2/SKILL.md (topic 2)
         → shared/resources/common.md
```

#### Pattern C: Reference Consolidation
When multiple CLAUDE.md files cover same domain:
```
dir1/CLAUDE.md + dir2/CLAUDE.md → unified-skill/SKILL.md
                                → unified-skill/resources/combined.md
```

### Step 5: Implementation

For each conversion:

1. **Create skill directory**:
```bash
mkdir -p .claude/skills/[skill-name]/resources
```

2. **Write SKILL.md**:
- Extract core purpose
- Identify key triggers
- Summarize capabilities
- Reference detailed resources

3. **Move detailed content**:
- Procedures → resources/methodology.md
- Examples → resources/examples.md
- Reference → resources/reference.md

4. **Validate**:
```bash
# Check YAML frontmatter
head -10 SKILL.md | grep -E "^(name|description):"

# Verify character limits
echo -n "$(grep '^name:' SKILL.md | cut -d: -f2-)" | wc -c  # < 64
echo -n "$(grep '^description:' SKILL.md | cut -d: -f2-)" | wc -c  # < 1024
```

### Step 6: Documentation

Create conversion map:
```markdown
# CLAUDE.md → Skills Conversion Map

## Completed Conversions

### src/core/CLAUDE.md
- Created: core-architecture (2024-11-01)
- Audience: You & sub-agents
- Resources: architecture.md, patterns.md
- Notes: Split GPU content into separate skill

### src/gui/CLAUDE.md
- Created: gui-development (2024-11-01)
- Audience: Sub-agents primarily
- Resources: wxwidgets.md, patterns.md
- Notes: Merged with dialog patterns

[Continue for each conversion...]
```

## Special Considerations

### High-Value Conversions

Prioritize CLAUDE.md files that:
1. You reference frequently
2. Contain procedural workflows
3. Have complex decision trees
4. Would benefit sub-agents

### Archive Strategy

Before deleting CLAUDE.md files:
1. Ensure all content is captured
2. Test the new skill works
3. Document the conversion
4. Commit the removal with clear message

### Lab Tech Consultation

Invoke lab techs for:
- Complex multi-file consolidations
- Unclear domain boundaries
- Audience determination
- Priority conflicts

## Timeline

**Week 1 Goals**:
- Day 1-2: Inventory and analysis
- Day 3-4: High-priority conversions
- Day 5-6: Remaining conversions
- Day 7: Validation and cleanup

## Success Criteria

- All CLAUDE.md files (except root) converted
- Skills are discoverable and functional
- No knowledge lost in transition
- Clear documentation trail
- Lab techs consulted for complex cases

## After Week 1

This process becomes obsolete. New knowledge goes directly into:
- Skills (for procedural/reference content)
- Root CLAUDE.md (for identity/framework)
- Journal (for observations/learning)