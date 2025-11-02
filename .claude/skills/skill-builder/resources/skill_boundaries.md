# Skill Boundaries Guide

Determining when to create new skills versus enhancing existing ones.

## Core Principle

Skills should have **high cohesion** (everything inside relates) and **low coupling** (minimal dependencies between skills).

## Create NEW Skill When

### Different Domain
The knowledge covers a fundamentally different area:
- Build system vs. Testing
- GUI development vs. Core algorithms
- Documentation vs. Security

### Different Audience
The primary users differ:
- You need strategic overview
- Sub-agents need detailed procedures
- Both need different perspectives

### Different Trigger Context
Activated under different circumstances:
- "Starting new feature" vs. "Debugging production issue"
- "Code review" vs. "Performance optimization"
- "Planning phase" vs. "Execution phase"

### Would Overwhelm Existing Skill
Adding would make the skill:
- Too large (> 50KB total)
- Too complex (> 5 major topics)
- Too ambiguous (unclear when to use)

## Enhance EXISTING Skill When

### Natural Extension
The new content:
- Extends existing capability
- Uses same mental model
- Follows same workflow

### Same Activation Context
Both triggered by same scenarios:
- Additional git workflows → enhance git-workflow
- New testing patterns → enhance test-architecture
- More build configurations → enhance build-system

### Shared Resources
Content shares significant resources:
- Common templates
- Same reference materials
- Shared scripts/tools

### Small Addition
The new content is:
- < 20% of existing size
- Single focused topic
- Clear subsection of existing

## Decision Framework

```
1. What triggers this knowledge need?
   - Same as existing skill → ENHANCE
   - Different trigger → NEW

2. Who is the primary audience?
   - Same as existing skill → ENHANCE
   - Different audience → NEW

3. How related to existing content?
   - Natural extension → ENHANCE
   - Different domain → NEW

4. What's the cognitive load?
   - Fits naturally → ENHANCE
   - Adds confusion → NEW
```

## Examples

### Example 1: Build System Knowledge

**Scenario**: You have `build-system` skill. New knowledge about cross-compilation.

**Decision**: ENHANCE
- Same domain (building)
- Same audience (developers)
- Natural extension of build configurations

**Implementation**:
```
build-system/
├── SKILL.md (add reference to cross-compilation)
└── resources/
    ├── standard-builds.md
    └── cross-compilation.md (NEW)
```

### Example 2: Testing Patterns

**Scenario**: You have `unit-testing` skill. New knowledge about integration testing.

**Decision**: NEW skill `integration-testing`
- Different trigger context (unit vs. integration)
- Different tools and patterns
- Different complexity level

**Implementation**:
```
unit-testing/          (existing)
integration-testing/   (new)
```

### Example 3: Security Considerations

**Scenario**: Multiple skills need security notes.

**Decision**: NEW skill `security-practices` + references
- Cross-cutting concern
- Different expertise required
- Can be referenced from other skills

**Implementation**:
```
security-practices/    (central skill)
build-system/SKILL.md  (references security-practices)
api-design/SKILL.md    (references security-practices)
```

## Anti-Patterns to Avoid

### The Kitchen Sink
Adding everything tangentially related:
```
❌ git-workflow/
    ├── git-commands.md
    ├── github-api.md
    ├── ci-cd-pipelines.md
    ├── docker-deployment.md
    └── kubernetes-orchestration.md
```

**Fix**: Separate by primary purpose.

### The False Split
Artificially dividing cohesive content:
```
❌ testing-setup/
   testing-execution/
   testing-validation/
```

**Fix**: Keep natural workflows together.

### The Circular Dependency
Skills that require each other:
```
❌ skill-a: "For details, see skill-b"
   skill-b: "First, complete skill-a"
```

**Fix**: Extract shared content or merge skills.

## Lab Tech Consultation

When unsure about boundaries:
1. Present both options (new vs. enhance)
2. Describe the knowledge to add
3. Lab techs will evaluate:
   - Cognitive load
   - Usage patterns
   - Maintenance burden
4. Implement consensus recommendation

## Quick Decision Tree

```
Is it the same domain?
├─ NO → Create NEW skill
└─ YES → Same audience?
         ├─ NO → Create NEW skill
         └─ YES → Same triggers?
                  ├─ NO → Create NEW skill
                  └─ YES → Would it double size?
                           ├─ YES → Create NEW skill
                           └─ NO → ENHANCE existing
```

## Remember

When in doubt:
- Prefer multiple focused skills over one complex skill
- Skills can reference each other
- Evolution is expected - refactor when patterns emerge
- Document boundary decisions in journal