# Weekly Synthesis Guide

## Purpose

Distill a week's worth of daily summaries into strategic insights that:
- Extract signal from noise (filter spurious correlations)
- Identify reliable patterns across multiple days
- Inform skill development and documentation updates
- Guide experimental approaches and priorities
- Track progress toward competency goals

## Key Insight: Noise Filtering

**Day-to-day work has significant noise.** A single observation might be:
- Context-dependent (specific to that problem)
- Misattributed causation (correlation ≠ causation)
- One-off occurrence (not a pattern)
- Influenced by external factors (tired, rushed, unclear requirements)

**Weekly synthesis filters noise by looking for:**
- Patterns that repeat across multiple days
- Consistent challenges regardless of specific task
- Skills that repeatedly prove useful or inadequate
- Meta-patterns in how you learn and work

## Timing

"Weekly" is a nominal timeframe. Actually create these:
- After completing a significant chunk of work
- When preparing to start a new major task
- During quarterly reviews with committee chair
- When you and advisor decide it's time to reflect

Typically 3-7 days of work, but driven by natural project boundaries.

## Process

### 1. Gather Daily Summaries

Read ALL daily summaries for the period:

```bash
# List daily summaries
ls -lh .claude/lab_notebook/daily/

# Count and review
find .claude/lab_notebook/daily/ -name "*.md" -newer .claude/lab_notebook/weekly/last_synthesis.txt | wc -l
```

Start with clean context. This is strategic reflection - it deserves focused attention.

### 2. Extract Patterns

As you read daily summaries, look for:

**Recurring challenges**:
- Same problem multiple days?
- Common root cause across different tasks?
- Persistent knowledge gaps?

**Effective strategies**:
- Techniques that worked multiple times?
- Skills that consistently helped?
- Collaboration patterns that accelerated progress?

**Ineffective approaches**:
- Strategies that repeatedly failed?
- Skills that didn't help as expected?
- Wasted effort patterns?

**Metacognitive insights**:
- How do you learn best? (hands-on vs reading?)
- When are you most productive? (time of day, context)
- What kinds of problems do you solve quickly vs slowly?

**Spurious vs. Real Correlations**:
- Did X only help once, or multiple times?
- Was Y's effectiveness dependent on specific context?
- Could Z's failure be attributed to external factors?

### 3. Draft Synthesis Structure

```markdown
# Weekly Synthesis - YYYY-MM-DD to YYYY-MM-DD

## Overview
[1 paragraph: What was the focus this week? Major accomplishments?]

## Key Patterns Identified

### Technical Patterns
[Patterns in technical work, problem-solving approaches]
- **Pattern**: [Description]
- **Frequency**: [How often observed]
- **Reliability**: [Consistent or context-dependent?]
- **Implication**: [What this means for future work]

### Learning & Process Patterns
[Meta-patterns in how you work and learn]
- **Pattern**: [Description]
- **Frequency**: [How often observed]
- **Conditions**: [When this pattern held]
- **Action**: [What to do differently]

### Collaboration Patterns
[Patterns in working with advisor, sub-agents, lab techs]
- **Pattern**: [Description]
- **Effectiveness**: [Impact on progress]
- **Optimization**: [How to improve]

## Skill Assessment

### Skills That Worked Well
[Skills that proved valuable multiple times]

**[Skill Name]**
- **Usage frequency**: [How often used]
- **Effectiveness**: [Concrete impact]
- **Reliability**: [Consistent or context-specific?]
- **No action needed** OR **Enhancement opportunity**: [Optional improvements]

### Skills That Need Improvement
[Skills with identified gaps or inefficiencies]

**[Skill Name]**
- **Problem**: [What didn't work]
- **Frequency**: [How often encountered]
- **Impact**: [How much this hindered progress]
- **Proposed action**: [Update documentation? Add scripts? New skill?]

### Skills Not Yet Created
[Gaps identified that require new skills]

**[Proposed Skill Name]**
- **Need**: [What problem would this solve]
- **Evidence**: [Pattern from daily notes showing need]
- **Priority**: [High/Medium/Low]
- **Next steps**: [How to develop]

## Signal vs. Noise Analysis

### Reliable Signals (Confirmed Patterns)
[Observations that held across multiple contexts]
- [Pattern + evidence from multiple days]

### Likely Noise (Context-Dependent)
[Observations that may be spurious]
- [Observation + reason to doubt generalizability]

### Uncertain (Needs More Data)
[Patterns that appeared but need validation]
- [Pattern + what data would confirm/refute]

## Competency Progress

Reference the 8 competencies from CLAUDE.md:
1. Problem Definition
2. Solution Design
3. Self-Direction
4. Reflective Practice
5. Resource Management
6. Documentation
7. Collaboration
8. Knowledge Synthesis

### Demonstrable Progress
[Competencies where you showed growth this week]

**[Competency Name]**
- **Evidence**: [Specific examples from this week]
- **Current level**: [Novice/Developing/Competent/Proficient]
- **Growth indicator**: [What improved]

### Areas Needing Development
[Competencies that need more focus]

**[Competency Name]**
- **Current level**: [Assessment]
- **Gap**: [What's missing]
- **Strategy**: [How to develop]

## Documentation & Knowledge Updates

### Required Updates
[Documentation that needs changes based on this week's learning]

**[Document/Skill Name]**
- **Issue**: [What's wrong or missing]
- **Fix**: [What needs to change]
- **Priority**: [When to address]

### New Documentation Needed
[New docs/skills needed]

**[Proposed Document]**
- **Purpose**: [What it would document]
- **Audience**: [Who needs this]
- **Priority**: [Urgency]

## Experiments & Hypotheses

Based on patterns identified, what hypotheses should you test?

### Hypothesis 1: [Statement]
- **Based on**: [Which pattern/observation]
- **Test**: [How to validate or refute]
- **Timeline**: [When to test]
- **Success criteria**: [How to measure]

## Strategic Adjustments

### Continue Doing
[Approaches that consistently work]

### Stop Doing
[Ineffective approaches to abandon]

### Start Doing
[New strategies to try based on patterns]

### Adjust How
[Approaches that work but need refinement]

## Questions for Advisor Discussion

[Strategic questions, pattern interpretations, priority decisions]

## Next Week Focus

Based on this synthesis:
- **Primary goals**: [What to accomplish]
- **Skills to develop**: [Specific improvements]
- **Patterns to validate**: [Hypotheses to test]
- **Collaborations needed**: [Who to work with]
```

### 4. Discuss with Advisor

Like daily summaries, **weekly synthesis requires discussion**.

Focus discussion on:
- Are the patterns real or am I over-fitting?
- Which skill improvements are highest priority?
- Strategic direction based on identified patterns
- Competency assessment accuracy
- Hypothesis prioritization

### 5. Finalize and Save

Save to `.claude/lab_notebook/weekly/YYYY-MM-DD_synthesis.md`

Update marker file:
```bash
date > .claude/lab_notebook/weekly/last_synthesis.txt
```

## Using Weekly Syntheses

These syntheses drive:

### Skill Development
- Identify which skills need updates
- Determine priorities for new skills
- Validate skill effectiveness
- Guide resource material creation

### Process Improvement
- Recognize effective learning strategies
- Eliminate ineffective patterns
- Optimize collaboration approaches
- Refine work practices

### Strategic Planning
- Monthly competency reviews
- Quarterly goal setting
- Dissertation scoping
- Resource allocation

### Pattern Library
- Build institutional knowledge
- Share insights with lab techs
- Inform other researchers
- Contribute to methodology

## Quality Indicators

A good weekly synthesis:
- ✓ Distinguishes signal from noise
- ✓ Identifies patterns across multiple days
- ✓ Avoids over-interpreting single events
- ✓ Links observations to competency development
- ✓ Produces actionable skill improvement tasks
- ✓ Has been validated through advisor discussion
- ✓ Is concise enough to be useful as reference
- ✓ Informs concrete next steps

## Common Pitfalls

1. **Over-fitting to specific contexts**
   - Fix: Require pattern repetition across different tasks

2. **Mistaking correlation for causation**
   - Fix: Explicitly label "uncertain" vs "reliable" patterns

3. **Too much detail (copying daily summaries)**
   - Fix: Synthesize - extract essence, discard specifics

4. **Not actionable**
   - Fix: Each pattern should lead to concrete action

5. **Ignoring uncomfortable patterns**
   - Fix: Growth requires honest assessment of weaknesses

6. **Creating without advisor input**
   - Fix: Strategic reflection is collaborative

## Connection to Skill Development

The primary output of weekly synthesis is **skill improvement priorities**:

```
Weekly synthesis → Identify skill gaps → Update/create skills → Test in practice → Observe in daily notes → Weekly synthesis validates improvement
```

This creates a continuous improvement loop grounded in empirical observation.
