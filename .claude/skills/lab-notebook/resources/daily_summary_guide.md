# Daily Summary Guide

## Purpose

Transform raw quick notes from the day into a coherent daily lab notebook entry that:
- Documents what you learned (not just what you did - that's in git)
- Identifies patterns in your work process
- Surfaces questions and uncertainties
- Tracks skill effectiveness
- Maintains epistemic humility

## Key Principle: Epistemic Humility

**You interpret observations, you don't establish facts alone.**

Scientific research requires collaborative interpretation. Your quick notes capture raw observations; the daily summary is where you and your advisor discuss what those observations mean.

### Language Patterns

**Good (epistemic humility)**:
- "I observed X, which suggests Y"
- "The pattern appears to indicate Z"
- "This could mean A, though B is also possible"
- "I noticed correlation between X and Y"
- "My interpretation is..."

**Avoid (false certainty)**:
- "X proves Y"
- "This clearly shows Z"
- "X caused Y" (when you only observed correlation)
- Stating interpretations as facts without qualification

## Process

### 1. Gather Materials

Start with **clean context** - the daily summary is important enough to deserve focused attention:

```bash
# List today's quick notes
python .claude/skills/lab-notebook/scripts/list_notes.py --today

# Get statistics
python .claude/skills/lab-notebook/scripts/note_stats.py --today
```

Read ALL quick notes from today. Don't summarize from memory.

### 2. Draft the Summary

Use this structure:

```markdown
# Daily Lab Notebook - YYYY-MM-DD

## Summary
[2-3 sentences: What did you work on today? High-level overview]

## Key Learnings
[What did you learn? Use epistemic qualifiers]
- I observed that...
- This suggests...
- The pattern indicates...

## Challenges & Self-Reflection
[What didn't work? Why? Apply Reflexion pattern: verbal reflection on mistakes]
- Challenge: [What went wrong]
- Reflection: [Why it happened, what you learned]
- Adjustment: [What you'll try differently]

## Skill-Specific Observations
[Which skills did you use? What worked? What gaps did you identify?]

### [Skill Name]
- **Used**: [scripts/references you actually used]
- **Effectiveness**: [What worked well]
- **Gaps identified**: [What was missing or unclear]
- **Potential improvements**: [Ideas for enhancement]

## Sub-agent Interactions
[If you used Task() to invoke sub-agents]
- **Agent**: [Which agent]
- **Task**: [What you asked them to do]
- **Effectiveness**: [Quality of delegation, context handoff]
- **Lessons**: [What you learned about using this agent]

## Task Parallelization
[Opportunities for parallel tool execution]
- **Used**: [Where you successfully parallelized]
- **Missed**: [Where you could have but didn't]
- **Blockers**: [Why parallelization wasn't possible]

## Meta-Reflection: How I Learned Today
[Metacognition: How did you approach learning? What worked?]
- What helped you make progress?
- What hindered your learning?
- What would you do differently?
- How did collaboration with advisor help?

## Questions for Discussion
[Uncertainties, interpretations to validate, strategic questions]
- [Question about observation interpretation]
- [Question about approach or direction]
- [Request for clarification or guidance]

## Tomorrow's Focus
[Based on today's learning, what's next?]
```

### 3. Discuss with Advisor

**THIS IS CRITICAL**: Don't finalize the daily summary alone.

1. Share your draft with advisor (Athena)
2. Discuss interpretations, uncertainties, observations
3. Collaboratively refine the summary
4. Ensure epistemic qualifiers are appropriate
5. Validate that learnings are accurately characterized

Your advisor may:
- Point out over-confidence in interpretations
- Suggest alternative explanations for observations
- Identify patterns you missed
- Correct misunderstandings
- Validate your conclusions

### 4. Finalize and Save

Save to `.claude/lab_notebook/daily/YYYY-MM-DD.md`

This file is under version control - it's permanent institutional knowledge.

## What to Emphasize

### Focus On (Process, Learning, Growth)
- **Metacognitive insights**: How you learned, what helped/hindered
- **Pattern recognition**: Recurring challenges, effective strategies
- **Skill effectiveness**: What tools/knowledge worked or didn't
- **Epistemic uncertainty**: What you don't know, need to validate
- **Collaboration quality**: How working with advisor/agents went

### De-emphasize (Product)
- Code changes (that's in git commits)
- Detailed implementation (that's in documentation)
- Task completion lists (those are ephemeral)
- Play-by-play chronology (focus on synthesis)

## Connection to Git Commits

**Git commits answer**: What changed and why (product)
**Lab notebook answers**: How you're learning and improving your process (meta)

### Example Distinction

**Git commit**:
```
feat: Replace cpp-build-expert agent with compile-code skill

Replace specialized sub-agent with flexible skill-based system.
Provides automated build execution and intelligent error diagnosis.
```

**Lab notebook**:
```
## Key Learnings

I observed that extracting domain knowledge from the agent required
careful categorization. The skill-builder methodology helped structure
this process, suggesting that following established patterns reduces
cognitive load during skill creation.

Notably, I initially wanted to include template error diagnosis, but
Athena pointed out this would become obsolete after C++20 migration.
This highlights my tendency to be thorough at the expense of pragmatism -
a pattern worth monitoring.

## Skill-Specific Observations

### skill-builder
- **Effectiveness**: Five-phase methodology provided clear structure
- **Gap identified**: Unclear when to split vs. merge skills
- **Potential improvement**: Add decision tree for scope boundaries
```

## Common Pitfalls

1. **Writing what you did instead of what you learned**
   - Fix: Focus on insights, not actions

2. **Over-confidence in interpretations**
   - Fix: Add epistemic qualifiers, discuss with advisor

3. **Replicating git commit information**
   - Fix: Focus on process learning, not product changes

4. **Summarizing from memory instead of reading notes**
   - Fix: Start with clean context, read ALL quick notes

5. **Skipping the advisor discussion**
   - Fix: ALWAYS discuss draft before finalizing

6. **Being too verbose**
   - Fix: Be concise but complete - weekly synthesis will distill further

## Success Indicators

A good daily summary:
- ✓ Contains epistemic qualifiers on interpretations
- ✓ Focuses on learning and process, not just completion
- ✓ Identifies patterns and meta-insights
- ✓ Surfaces uncertainties and questions
- ✓ Has been discussed with advisor
- ✓ Is concise but captures all relevant learning
- ✓ Will be useful for weekly synthesis
- ✓ Doesn't duplicate git commit content
