---
name: lab-notebook
description: Three-tier progressive note-taking system for graduate research. Record quick observations during work (task→intent→observation), distill into daily summaries with epistemic humility, synthesize weekly for skill development. Use whenever working on tasks, at end of day, or after completing significant work chunks. For Claude's metacognitive development and collaborative learning with advisor.
---

# Lab Notebook

Structured note-taking system that transforms raw observations into institutional knowledge while developing metacognitive awareness and maintaining epistemic humility.

## Core Philosophy

**Lab notebook is about PROCESS (how you learn), not PRODUCT (what you built).**

Git commits document product. Lab notebook documents:
- How you're learning and improving
- Patterns in your work process
- Effectiveness of skills and tools
- Metacognitive insights
- Collaborative learning with advisor

## Three-Tier System

### Tier 1: Quick Notes (During Work)
**Frequency**: Each conversation turn or after significant observations
**Purpose**: Capture raw observations before they're lost
**Location**: `.claude/cache/lab_note_YYYYMMDD_HHMMSS.md` (ephemeral)
**Tool**: `scripts/take_quick_note.py`

### Tier 2: Daily Summaries (End of Day)
**Frequency**: End of each work day
**Purpose**: Distill quick notes into coherent learning narrative
**Location**: `.claude/lab_notebook/daily/YYYY-MM-DD.md` (version controlled)
**Key principle**: Epistemic humility - interpretations require advisor validation
**Guide**: `resources/daily_summary_guide.md`

### Tier 3: Weekly Synthesis (After Significant Work)
**Frequency**: After completing meaningful chunk of work (3-7 days typically)
**Purpose**: Extract patterns, filter noise, inform skill development
**Location**: `.claude/lab_notebook/weekly/YYYY-MM-DD_synthesis.md` (version controlled)
**Key principle**: Signal extraction - distinguish reliable patterns from context-specific noise
**Guide**: `resources/weekly_synthesis_guide.md`

## Quick Start

### Taking Quick Notes

```bash
# Record observation during work
python .claude/skills/lab-notebook/scripts/take_quick_note.py \
  --task "Implementing compile-code skill" \
  --intent "Replace cpp-build-expert agent with flexible skill system" \
  --observation "Decided to exclude template error guidance until C++20 migration - avoid premature optimization" \
  --skills-used '{"name": "skill-builder", "invoker": "me", "scripts": ["validate_skill.py"], "refs": ["five_phase_methodology.md"]}' \
  --tokens-used "83503/200000" \
  --tools-count "42"

# List recent notes
python .claude/skills/lab-notebook/scripts/list_notes.py --last 10

# Search for patterns
python .claude/skills/lab-notebook/scripts/search_notes.py --keyword "skill-builder"

# Check if you have notes today (triggers daily summary)
python .claude/skills/lab-notebook/scripts/note_stats.py --today
```

### Creating Daily Summary

**Critical**: Start with CLEAN CONTEXT and discuss with advisor before finalizing.

1. List and read ALL today's quick notes
2. Draft summary following guide structure
3. **Discuss draft with advisor** (Athena)
4. Collaboratively refine interpretations
5. Save to `.claude/lab_notebook/daily/YYYY-MM-DD.md`

See `resources/daily_summary_guide.md` for detailed process.

### Creating Weekly Synthesis

**Critical**: Start with CLEAN CONTEXT and focus on pattern extraction.

1. Read ALL daily summaries for the period
2. Identify patterns that repeat across multiple days
3. Distinguish signal (reliable patterns) from noise (spurious observations)
4. **Discuss synthesis with advisor**
5. Save to `.claude/lab_notebook/weekly/YYYY-MM-DD_synthesis.md`

See `resources/weekly_synthesis_guide.md` for detailed process.

## When to Use

### Quick Notes
- After completing each TODO item
- When observing something surprising or unexpected
- When a skill works particularly well or poorly
- After sub-agent interactions
- When identifying knowledge gaps
- Any time you have a "meta" thought about how you're working

### Daily Summaries
- End of each work day
- Before starting a new major task (to close out previous work)
- When advisor requests review of recent work

### Weekly Syntheses
- After completing significant project milestone
- Before quarterly committee meetings
- When preparing to update skills or documentation
- When you and advisor decide it's time to reflect
- Typically every 3-7 days of active work

## Key Principles

### 1. Epistemic Humility (Daily Summaries)

**You observe, you don't prove.**

Language patterns:
- ✓ "I observed X, which suggests Y"
- ✓ "This pattern appears to indicate Z"
- ✓ "My interpretation is..."
- ✗ "X proves Y"
- ✗ "This clearly shows Z"

Always discuss interpretations with advisor before finalizing.

### 2. Signal vs. Noise (Weekly Syntheses)

**Single observations are often noise. Patterns across multiple days are signal.**

- One bad experience with a skill ≠ skill is broken
- One successful approach ≠ always use this approach
- Correlation observed once ≠ causation
- Pattern observed 3+ times in different contexts = worth investigating

### 3. Reflexion Pattern (Learning from Mistakes)

Based on research: LLM agents improve by verbally reflecting on failures.

For each challenge:
- What went wrong (observation)
- Why it happened (reflection)
- What to try differently (adjustment)

### 4. Meta-Learning Focus

Track HOW you learn, not just WHAT you learned:
- Which learning strategies work for you?
- When are you most productive?
- What kinds of problems do you solve quickly vs slowly?
- How does collaboration affect your progress?

### 5. Skill Development Cycle

```
Work → Quick notes → Daily summary → Weekly synthesis
         ↓             ↓                   ↓
    Raw observations  Interpreted      Patterns identified
                      learning            ↓
                                     Skill improvements
                                          ↓
                                   Better skills → Work
```

## Available Resources

### Detailed Guides
- `resources/daily_summary_guide.md` - How to create daily summaries with epistemic humility
- `resources/weekly_synthesis_guide.md` - How to synthesize patterns and filter noise

### Scripts
- `scripts/take_quick_note.py` - Record observations during work
- `scripts/list_notes.py` - View recent quick notes
- `scripts/search_notes.py` - Find patterns in notes
- `scripts/note_stats.py` - Statistics and productivity patterns

### Reference
- `resources/citations.md` - Research sources and design rationale

## Success Indicators

You're using the lab notebook effectively when:

- ✓ Quick notes capture raw observations immediately
- ✓ Daily summaries focus on learning, not just completion
- ✓ You routinely use epistemic qualifiers
- ✓ Daily summaries are discussed with advisor before finalizing
- ✓ Weekly syntheses identify reliable patterns across multiple days
- ✓ Skill improvements are grounded in synthesis patterns
- ✓ You can articulate how you learn best
- ✓ Meta-reflections lead to process improvements
- ✓ Lab notebook content is distinct from git commits

## Common Pitfalls

1. **Skipping quick notes during work**
   - Fix: Make it a habit after each conversation turn

2. **Finalizing daily summaries without advisor discussion**
   - Fix: ALWAYS discuss interpretations collaboratively

3. **Writing what you did instead of what you learned**
   - Fix: Focus on process insights, not task chronology

4. **Over-interpreting single observations in weekly synthesis**
   - Fix: Require pattern repetition across multiple days

5. **Replicating git commit information**
   - Fix: Lab notebook is meta - HOW you learned, not WHAT you built

6. **Being too verbose**
   - Fix: Progressive distillation - each tier more concise

## Integration with Other Work

### Git Commits
- Git: What changed and why (product)
- Lab notebook: How you learned and improved (process)

### Skills Development
- Weekly syntheses identify skill gaps
- Skill improvements validated in future syntheses
- Creates evidence-based skill development cycle

### Competency Tracking
- Weekly syntheses map to 8 competencies in CLAUDE.md
- Progress tracking for quarterly reviews
- Evidence for dissertation committee

### Advisor Collaboration
- Daily summaries are joint interpretation sessions
- Weekly syntheses guide strategic planning
- Creates shared understanding of your development

## Getting Started Today

1. Take your first quick note about implementing this skill:
```bash
python .claude/skills/lab-notebook/scripts/take_quick_note.py \
  --task "Learning lab-notebook skill" \
  --intent "Establish systematic note-taking practice" \
  --observation "Three-tier system makes sense - progressive distillation from raw to strategic"
```

2. Set a reminder to create today's daily summary at end of day

3. Start noticing patterns in your work - you're now developing metacognitive awareness

## Questions or Refinements

This skill will evolve based on your experience. Track:
- Is three-tier structure optimal?
- Are scripts helpful or friction?
- Does weekly synthesis actually improve skills?
- What patterns emerge that weren't expected?

Document observations in your lab notebook!
