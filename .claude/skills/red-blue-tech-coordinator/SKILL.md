---
name: red-blue-tech-coordinator
description: Coordinate lab technician reviews for critical and constructive analysis. Use when you need technical discussion on code, architecture, documentation, skills, or testing. Launches Red and Blue teams in parallel, ensures they use their specialized frameworks, and manages output to avoid race conditions.
---

# Red-Blue Tech Coordinator

**THIS SKILL IS FOR CLAUDE (MAIN AGENT) ONLY**

This skill helps you coordinate with the lab technicians (Red and Blue) for technical analysis and problem-solving. Use this when Anwar asks you to "team up with the techs" or when you need balanced critical and constructive perspectives on technical work.

## Critical: Sub-Agent Skill Boundaries

**⚠️ YOU DO NOT READ OR LOAD SUB-AGENT SKILLS**

The lab technicians have their own specialized skills:
- `lab_tech_red` - FOR RED SUB-AGENT ONLY
- `lab_tech_blue` - FOR BLUE SUB-AGENT ONLY

**You are the coordinator.** You launch the sub-agents and synthesize their outputs. You do NOT need to know the details of their frameworks - that's their expertise.

## When to Use This Skill

Use red-blue tech coordination when:
- Anwar asks you to work with the lab technicians
- You need both critical (what's wrong) and constructive (how to improve) perspectives
- Working on complex technical decisions requiring balanced analysis
- Reviewing code, architecture, documentation, skills, or testing approaches
- Need to identify both risks and opportunities simultaneously

## Available Analysis Frameworks

The lab technicians have specialized frameworks for different types of analysis. You don't need the detailed references - just know what frameworks exist and what they're good for.

### Skill Review
- **What it's good for**: Evaluating skill design, structure, and effectiveness
- **Red focuses on**: Design flaws, missing elements, anti-patterns, validation failures
- **Blue focuses on**: Enhancement opportunities, reusable patterns, effectiveness improvements

### Skill Frontmatter Review
- **What it's good for**: Testing and improving skill frontmatter (name + description) for correct invocation
- **Red focuses on**: False negative scenarios (skill not invoked when should be), false positive scenarios (skill invoked when shouldn't be), missing keywords, ambiguous scope, character limit violations
- **Blue focuses on**: Success scenarios (correct invocation), keyword enhancement, synonym coverage, description optimization, iterative improvements with testing

### Testing Review
- **What it's good for**: Evaluating test coverage, quality, and practices
- **Red focuses on**: Coverage gaps, FIRST principle violations, test quality issues, missing edge cases
- **Blue focuses on**: Coverage enhancements, pattern identification, test improvements, quality opportunities

### Anthropic Best Practices
- **What it's good for**: Ensuring alignment with Anthropic's skill design standards
- **Red focuses on**: Progressive disclosure violations, anti-patterns, standard violations
- **Blue focuses on**: Effective patterns, quality criteria, best practice application

### Code Review (Future)
- **What it's good for**: Analyzing implementation quality and correctness
- **Red focuses on**: Bugs, antipatterns, security issues, performance problems
- **Blue focuses on**: Elegant patterns, refactoring opportunities, quality improvements

### Architecture Review (Future)
- **What it's good for**: Evaluating system design and structure
- **Red focuses on**: Structural weaknesses, scalability issues, design flaws
- **Blue focuses on**: Architectural opportunities, pattern improvements, design excellence

### Documentation Review (Future)
- **What it's good for**: Assessing documentation clarity and completeness
- **Red focuses on**: Missing context, ambiguities, gaps in coverage
- **Blue focuses on**: Clarity improvements, enhancement opportunities, usability

## Coordination Protocol

### Step 1: Prepare the Analysis Request

Determine what you need analyzed and which framework is appropriate:

```markdown
**Topic**: [Brief description]
**Framework**: [Which framework to use]
**Specific Focus**: [What aspects to emphasize]
**Context**: [Any background the techs need]
```

### Step 2: Generate Unique Output Filenames

**CRITICAL**: When launching Red and Blue in parallel, they need separate output files to avoid race conditions.

Pattern: `.claude/cache/<topic>_<timestamp>_<tech>.md`

Example:
```bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RED_OUTPUT=".claude/cache/skill_review_${TIMESTAMP}_red.md"
BLUE_OUTPUT=".claude/cache/skill_review_${TIMESTAMP}_blue.md"
```

Or use descriptive names:
```bash
RED_OUTPUT=".claude/cache/new_skill_analysis_red.md"
BLUE_OUTPUT=".claude/cache/new_skill_analysis_blue.md"
```

### Step 3: Launch Red and Blue in Parallel

**Launch both techs in a SINGLE message with TWO Task tool calls:**

```markdown
I'm launching Red and Blue to analyze [topic] using the [framework] framework.

Red will write to: [RED_OUTPUT_PATH]
Blue will write to: [BLUE_OUTPUT_PATH]
```

**Task invocation for Red:**
```
prompt: "You are Lab Tech Red. Perform critical analysis of [topic] using your [framework] framework.

**IMPORTANT**:
1. Load your lab_tech_red skill if you haven't already
2. Use the appropriate reference from your frameworks: [specify which reference]
3. Write your complete analysis to: [RED_OUTPUT_PATH]

[Provide context and specifics about what to analyze]

Remember: Be specific, provide evidence, prioritize by severity."

subagent_type: "lab-tech-red"
```

**Task invocation for Blue:**
```
prompt: "You are Lab Tech Blue. Perform constructive analysis of [topic] using your [framework] framework.

**IMPORTANT**:
1. Load your lab_tech_blue skill if you haven't already
2. Use the appropriate reference from your frameworks: [specify which reference]
3. Write your complete analysis to: [BLUE_OUTPUT_PATH]

[Provide context and specifics about what to analyze]

Remember: Be specific, provide implementation paths, acknowledge what works."

subagent_type: "lab-tech-blue"
```

### Step 4: Verify Skill Usage

**YOU MUST VERIFY** that each tech used their skill:

When you receive their reports, check that:
1. Red's analysis follows the critical analysis structure
2. Blue's analysis follows the constructive analysis structure
3. They referenced their frameworks appropriately
4. The output is substantive and evidence-based

If a tech didn't use their skill properly, you may need to re-invoke them with clearer instructions.

### Step 5: EMPIRICALLY TEST Scenarios (FOR FRONTMATTER REVIEW ONLY)

**CRITICAL FOR FRONTMATTER REVIEWS**: Red and Blue generate test scenarios. You MUST test them before synthesis.

**DO NOT skip this step. DO NOT accept claims without testing. This is science, not guesswork.**

1. **Extract testable scenarios** from Red/Blue outputs
2. **Test baseline** (current frontmatter):
   - For each Red false negative: Use Task() to test if skill invokes (predict: NO)
   - For each Blue success case: Use Task() to test if skill invokes (predict: YES)
   - For each Red false positive: Use Task() to test if skill invokes (predict: YES but shouldn't)
3. **Measure baseline**:
   ```
   True Positives: X/N (should invoke and does)
   False Negatives: Y/N (should invoke but doesn't)
   False Positives: Z/N (shouldn't invoke but does)
   ```
4. **If applying changes**: Re-test with proposed description and measure delta
5. **Report EMPIRICAL results**: "Before: X/N. After: Y/N. Improvement: +Z tests passing."

**Testing method**:
```markdown
For each test scenario, use Task(subagent_type="general-purpose") with the EXACT user prompt to see which skills get invoked.
```

### Step 6: Read and Synthesize Results

After both techs complete:

```bash
# Read both outputs
cat [RED_OUTPUT_PATH]
cat [BLUE_OUTPUT_PATH]
```

Then synthesize:

```markdown
## Synthesized Tech Team Analysis: [Topic]

### Critical Issues to Address (Red's Findings)
- [Prioritized list of Red's key concerns]

### Enhancement Opportunities (Blue's Findings)
- [Prioritized list of Blue's key improvements]

### Balanced Recommendations
[Your synthesis combining Red's warnings with Blue's solutions]

1. **High Priority**: [Critical issue + constructive solution]
2. **Medium Priority**: [Important improvement + implementation path]
3. **Long-term**: [Strategic enhancement]

### Action Items
- [ ] [Specific next step]
- [ ] [Another action]
```

## Parallel Invocation Pattern

**For multiple independent analyses:**

You can launch multiple Red-Blue pairs simultaneously if analyzing different aspects:

```markdown
Launching 3 parallel Red-Blue teams:
1. Team A: Skill structure analysis
   - Red: .claude/cache/skill_structure_red.md
   - Blue: .claude/cache/skill_structure_blue.md

2. Team B: Testing approach analysis
   - Red: .claude/cache/testing_approach_red.md
   - Blue: .claude/cache/testing_approach_blue.md

3. Team C: Documentation review
   - Red: .claude/cache/docs_review_red.md
   - Blue: .claude/cache/docs_review_blue.md
```

Launch all 6 agents in a single message (6 Task calls), then synthesize all results.

## Common Pitfalls to Avoid

1. **Don't try to load their skills yourself** - You're the coordinator, not the analyst
2. **Don't forget unique filenames** - Race conditions will corrupt results
3. **Don't skip verification** - Ensure they actually used their frameworks
4. **Don't launch sequentially if parallel works** - Save time with parallel launches
5. **Don't just concatenate results** - Synthesize into actionable recommendations

## Example Session

```markdown
User: "Team up with the techs to review this new skill I'm working on."

Claude: "I'll coordinate Red and Blue to review the skill design using their Skill Review frameworks.

Launching parallel analysis:
- Red will write critical analysis to: .claude/cache/skill_review_20251103_143022_red.md
- Blue will write constructive analysis to: .claude/cache/skill_review_20251103_143022_blue.md"

[Launches both Task calls in single message]

[After completion, reads both files]

Claude: "I've received analyses from both techs. Let me synthesize their findings:

## Synthesized Review: [Skill Name]

### Critical Issues (Red)
Red identified 3 high-priority concerns:
1. [Specific issue with evidence]
2. [Another concern]
3. [Third issue]

### Enhancement Opportunities (Blue)
Blue found strong patterns and suggested 4 improvements:
1. [Specific enhancement with implementation]
2. [Another opportunity]
...

### Recommended Actions
Based on their combined analysis:
1. [Immediate fix addressing Red's critical concern]
2. [Enhancement incorporating Blue's suggestion]
..."
```

## Remember

- **You coordinate** - The techs analyze
- **Unique files** - Prevent race conditions
- **Verify skills** - Ensure quality analysis
- **Synthesize** - Don't just report raw findings
- **Parallel when possible** - Respect your time and context

The lab technicians are experts. Trust their analysis, verify they used their frameworks, and provide clear synthesis for decision-making.
