# Fundamentals of Refactoring Prioritization

## Purpose

This resource explains the core concepts behind data-driven refactoring prioritization: behavioral code analysis, code churn, hotspots, and the ROI framework.

## When You Need This

- New to behavioral code analysis
- Understanding why git history matters for code quality
- Justifying refactoring decisions to stakeholders
- Learning the theoretical foundation

---

## Behavioral Code Analysis

**Definition**: Using version control history combined with static analysis to identify code quality issues and predict future problems.

**Key Insight**: Code alone doesn't reveal:
- How often it changes
- Who works on it
- Communication patterns required
- Evolutionary trends
- Coordination bottlenecks

Git history provides the **social and temporal dimension** that static analysis misses.

**Origin**: Popularized by Adam Tornhill in "Your Code as a Crime Scene" (2015) and "Software Design X-Rays" (2018).

---

## The Three Critical Questions

Behavioral code analysis answers:

### 1. Where does technical debt carry the highest interest rate?

Not all debt matters equally. Some legacy code sits untouched for years (low interest). Other code changes constantly, multiplying maintenance costs (high interest).

**Focus on high-interest debt.**

### 2. Does architecture align with system evolution?

Initial architecture doesn't always match how the system evolved. Misalignment creates:
- Excessive coupling between modules
- Cross-team coordination overhead
- Repeated bug fixes in the same areas

**Detect architectural decay through change patterns.**

### 3. What productivity bottlenecks exist?

Some code requires coordination between many developers or teams. Conway's Law predicts this creates communication overhead.

**Identify and minimize coordination hotspots.**

---

## Core Metrics Explained

### Change Frequency (Churn)

**What**: How often a file/module changes over time

**Measured by**: Number of commits touching the file

**Indicates**:
- Instability
- Active development
- Maintenance burden
- Possible quality issues

**Healthy pattern**:
```
Commits: ███████░░░░░░░░░  (burst during development, then stable)
Time:    ─────────────────→
```

**Unhealthy pattern**:
```
Commits: ██████████████████  (sustained high frequency, never stabilizes)
Time:    ─────────────────→
```

### Complexity

**What**: Difficulty of understanding and modifying code

**Measured by**:
- Lines of Code (LOC)
- Cyclomatic Complexity (CCN) - number of decision paths
- Indentation depth
- Function length

**Indicates**:
- Cognitive load
- Testing difficulty
- Bug risk
- Modification risk

**Complexity alone is NOT a problem.** Complex stable code is fine.

### Code Churn (Detailed)

**What**: Total lines added + deleted over time

**Differs from change frequency**: One commit could have high churn (1000 lines) or low churn (1 line).

**Research finding**: Pre-release code churn is a strong predictor of post-release defect density.

**High churn reasons**:
- Repeated bug fixes → quality issue
- Incremental features → missing abstraction
- Configuration tweaks → poor separation of concerns
- Multiple developers → communication issues

### Temporal Coupling

**What**: Modules that tend to change together

**Example**: If `auth.cpp` and `database.cpp` change together in 75% of commits, they have 75% temporal coupling.

**Reveals**:
- Hidden dependencies not visible in code
- Architectural issues
- Missing abstractions
- Copy-paste duplication

### Code Age

**What**: Time since last modification

**Indicates**:
- Stability (old, unchanged)
- Active development (recently changed)
- Maturity

**Combined with churn**: Old, low-churn code = stable, well-designed. New, high-churn code = under development or problematic.

---

## The Hotspot Concept

**Definition**: Code that is both **complex** AND **frequently modified**.

**Rationale**: "If the code never changes, it's not costing us money."

### The Four-Quadrant Framework

```
High │   Stable     │  HOTSPOTS   │
Cmplx│   Ignore     │  PRIORITY   │
     │              │   FOCUS     │
     ├──────────────┼─────────────┤
Low  │    OK        │     OK      │
Cmplx│              │             │
     └──────────────┴─────────────┘
       Low Change     High Change
        Frequency      Frequency
```

**Top-Left (Complex, Low Change)**: Legacy code that works. Leave alone unless you must modify it.

**Top-Right (Complex, High Change)**: HOTSPOTS. Costs time and money with every change. **Priority refactoring target.**

**Bottom-Left (Simple, Low Change)**: Ideal. Well-designed, stable code.

**Bottom-Right (Simple, High Change)**: Active development or configuration. Usually fine.

### Hotspot Thresholds

| Metric | Low | Medium | High | Critical |
|--------|-----|--------|------|----------|
| Churn (commits/year) | <5 | 5-20 | 20-50 | >50 |
| CCN (cyclomatic complexity) | <10 | 10-20 | 20-50 | >50 |
| LOC (lines of code) | <200 | 200-500 | 500-1000 | >1000 |

**Context matters**: These are guidelines, not absolute rules. A 2000-line config file might be fine. A 200-line function with CCN 60 is alarming.

---

## The ROI Framework

**Principle**: Prioritize refactoring based on **return on investment**, not gut feeling.

### Technical Debt as Financial Liability

**Principal**: Cost to refactor/rewrite the problematic code

**Interest Rate**: Annual drag on productivity
- Extra development time (features take longer)
- Bug fix cost (repeated issues)
- Incident cost (outages, customer impact)

**Formula**:
```
Annual Interest = (Extra Dev Time) + (Bug Fix Cost) + (Incident Cost)
Interest Rate = Annual Interest / Principal
```

**Strategy**: Pay down debt with highest interest rate first.

### Example

**Component A**:
- Principal: 2 weeks to refactor
- Annual cost: 8 weeks of slowdowns + 4 weeks of bug fixes = 12 weeks
- Interest rate: 600% (12 weeks / 2 weeks)

**Component B**:
- Principal: 4 weeks to refactor
- Annual cost: 6 weeks of slowdowns
- Interest rate: 150% (6 weeks / 4 weeks)

**Prioritize Component A** despite larger Component B refactoring effort.

---

## The Pareto Principle (80/20 Rule)

**Observation**: In most codebases, 20% of files account for 80% of problems.

**Power law distribution**: Small number of files have disproportionate impact.

**Implication**: You don't need to refactor the whole codebase. Focus on the critical 20% (or even 5%).

**Real example from Tornhill**:
- 200,000 line codebase
- Hotspot analysis identified 2,300 line file
- X-Ray analysis narrowed to 186 critical lines
- **Result**: <0.1% of codebase refactored, major productivity improvement

---

## When NOT to Refactor

Avoid refactoring when:

1. **Code is stable** (low churn) even if complex
   - It works, don't break it

2. **Code is simple** even if high churn
   - Easy to maintain, not a priority

3. **Scheduled for replacement**
   - Don't refactor what you'll delete

4. **Team lacks domain expertise**
   - Refactoring requires understanding

5. **No test coverage**
   - Refactor tests first (or add tests during refactoring)

6. **Unclear business value**
   - Must justify effort

**Remember**: "If the code never changes, it's not costing us money."

---

## Validation Strategies

Data-driven doesn't mean ignoring human insight. Validate with:

### 1. Developer Survey
Ask team: "What code slows you down most?"

Compare with hotspot analysis. High agreement = confidence. Mismatches = investigate.

### 2. Issue Tracker Analysis
Count bugs/issues by module. Does it correlate with hotspots?

### 3. Feature Velocity
Which modules slow down feature delivery? Cross-reference with churn.

### 4. Code Review Time
Which files require longest review discussions? Often matches hotspots.

### 5. Onboarding Feedback
New team members struggle with hotspots first. Ask where they got stuck.

**Best practice**: Combine metrics + human experience for robust prioritization.

---

## Measuring Refactoring Success

Track these metrics before and after refactoring:

### Churn Reduction
Does refactored code change less frequently?

**Target**: 30-50% reduction in commits after refactoring.

### Complexity Reduction
Lower CCN or LOC?

**Target**: Reduce CCN by 40-60%, or split large files into smaller ones.

### Defect Reduction
Fewer bugs post-refactoring?

**Track**: Count bugs for 3 months before/after.

### Velocity Improvement
Do features ship faster?

**Measure**: Story points or cycle time for features touching refactored code.

### Developer Satisfaction
Does team feel more productive?

**Survey**: "On scale 1-10, how easy is this code to work with?"

---

## Budget Allocation

**Industry benchmarks**:
- 10-15% of sprint capacity: Continuous debt reduction
- 20% of every sprint: Balance new features + debt (common agile practice)
- 15% of IT budget: Debt remediation (leading firms)

**Recommendation**: Allocate fixed percentage, like a "sinking fund," to prevent debt accumulation.

**Anti-pattern**: Waiting for "cleanup sprint" that never comes. Make it continuous.

---

## Related Resources

- **`churn_analysis.md`**: Detailed churn measurement techniques
- **`hotspot_analysis.md`**: Step-by-step hotspot identification
- **`temporal_coupling.md`**: Finding hidden dependencies
- **`practical_workflow.md`**: Complete implementation guide
