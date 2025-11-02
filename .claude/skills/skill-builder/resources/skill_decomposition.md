# Skill Decomposition Guidelines

## When to Create Multiple Skills

### Indicators for Decomposition

**Token Budget Pressure**:
- SKILL.md exceeding 200 lines despite delegation
- Resources folder growing beyond 10 files
- Context window warnings during testing

**Cognitive Complexity**:
- More than 3 distinct operational modes
- Requires different mental models for different tasks
- Users consistently using only subset of functionality

**Architectural Signals**:
- Natural workflow stages that rarely interact
- Different user personas using different parts
- Conflicting optimization strategies needed

### Decomposition Patterns

**1. Pipeline Pattern**
```
skill-analyze → skill-process → skill-report
```
- Each skill handles one pipeline stage
- Clear handoff points between skills
- State passed through artifacts or parameters

**2. Specialization Pattern**
```
skill-base (shared core)
├── skill-variant-a (specialized for use case A)
├── skill-variant-b (specialized for use case B)
└── skill-variant-c (specialized for use case C)
```
- Common functionality in base skill
- Specialized variants for different contexts
- Inheritance through shared resources

**3. Orchestrator Pattern**
```
skill-orchestrator
├── skill-worker-1
├── skill-worker-2
└── skill-worker-3
```
- Orchestrator manages workflow
- Workers handle specific tasks
- Similar to lab-tech team structure

## Decomposition Process

### Step 1: Map Functionality
- List all current capabilities
- Group by cognitive similarity
- Identify natural boundaries

### Step 2: Analyze Dependencies
- Document data flow between groups
- Identify shared resources
- Map coordination needs

### Step 3: Design Interfaces
- Define clear handoff points
- Specify data formats
- Create coordination protocols

### Step 4: Implement Incrementally
- Start with highest-value split
- Test integration thoroughly
- Document composition patterns

## Anti-Patterns to Avoid

**Over-Decomposition**:
- Creating skills for single functions
- Excessive coordination overhead
- Lost context between related operations

**Unclear Boundaries**:
- Overlapping responsibilities
- Ambiguous handoff points
- Circular dependencies

**Premature Decomposition**:
- Splitting before understanding full scope
- Optimizing before measuring bottlenecks
- Creating complexity without clear benefit

## When NOT to Decompose

**Keep as Single Skill When**:
- Total size under 300 lines with resources
- Single coherent mental model
- Tightly coupled operations
- Same user for all functions
- Decomposition adds more complexity than it removes

## Refactoring Existing Skills

### Recognition Triggers
- Repeated "For X, see Y" delegations for same topic
- Users consistently asking for subset of functionality
- Performance issues from loading unnecessary resources
- Conflicting requirements in different use cases

### Safe Refactoring Process
1. Create decomposed skills alongside original
2. Test thoroughly with parallel runs
3. Migrate users gradually
4. Deprecate original after validation
5. Document migration in git history

## Examples from cisTEMx

**Good Decomposition**: Lab-tech team
- Lead, Red, Blue have distinct perspectives
- Clear orchestration model
- Each skill focused on one approach

**Good Monolith**: methodological_skill_creation
- All phases tightly related
- Single workflow from start to finish
- Shared mental model throughout

## Coordination Mechanisms

When skills are decomposed, use:
- File-based artifacts for data passing
- Session directories for shared state
- Clear naming conventions for discovery
- Documentation of integration patterns

Remember: Decomposition is a tool for managing complexity, not a goal in itself. A well-organized monolithic skill is better than a poorly decomposed system.