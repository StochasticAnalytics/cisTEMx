# Lab Tech Orchestration Pattern

Guide for using Lab Tech Red and Blue agents for balanced technical analysis.

## Overview

Lab Tech Red (critical) and Lab Tech Blue (constructive) provide complementary perspectives on technical questions. The main agent orchestrates their discussion, synthesizing their inputs into balanced recommendations.

**Key insight**: This is a "lunch discussion" pattern - the main agent brings up a topic, Red and Blue discuss it (Red always speaks first), and the main agent can follow up with questions before synthesizing.

## Architecture

```
Main Agent (You)
    ↓
    ├──→ Lab Tech Red (critical perspective) ──→ findings
    │         ↓
    │    Load skill, analyze, respond
    │
    └──→ Lab Tech Blue (constructive perspective) ──→ findings
              ↓
         Load skill, analyze, respond
              ↓
Main Agent synthesizes both perspectives
```

**Important**: Main agent invokes Red and Blue directly via Task tool. There is no Lead agent - you orchestrate.

## When to Use Lab Techs

Use Lab Tech Red/Blue for:
- **Security analysis**: Vulnerability assessment (Red) + mitigation strategies (Blue)
- **Skill review**: Quality issues (Red) + improvement opportunities (Blue)
- **Plan review**: Risks and gaps (Red) + de-risking and opportunities (Blue)
- **Code review**: Problems (Red) + solutions (Blue)
- **Architecture decisions**: Concerns (Red) + benefits and trade-offs (Blue)

**Don't use for**:
- Simple questions you can answer directly
- Tasks requiring tool use (file editing, compilation, etc.)
- Questions with single objective answers
- Situations where only one perspective is needed

## Invocation Pattern

### Step 1: Parallel Invocation (Red First)

**Launch both agents in parallel** with Red listed first:

```markdown
I'll use Lab Tech Red and Blue to analyze this [security/skill/plan].
```

**Then invoke via Task tool**:
```
Task(lab-tech-red): "Analyze [X] for vulnerabilities/issues/risks using security_review/skill_review/plan_review skill"
Task(lab-tech-blue): "Analyze [X] for mitigations/improvements/opportunities using security_review/skill_review/plan_review skill"
```

**Why Red first**: Red identifies problems, Blue builds on those to propose solutions.

**Why parallel**: Both can work simultaneously, reducing total time.

### Step 2: Review Responses

Read both Red and Blue responses:
- Red identifies: Gaps, weaknesses, risks, failures
- Blue identifies: Strengths, opportunities, mitigations, quick wins

### Step 3: Follow-Up (Optional)

If you need clarification or deeper analysis:

```
Task(lab-tech-red): "You identified [X] as critical. Can you provide more detail on [specific aspect]?"
Task(lab-tech-blue): "For the risk Red identified about [X], can you expand on mitigation approach [Y]?"
```

Iterate as needed until you have sufficient understanding.

### Step 4: Synthesis

Combine both perspectives:
- Start with Red's critical findings (what must be addressed)
- Integrate Blue's mitigations (how to address them)
- Add Blue's opportunities (how to go beyond just fixing)
- Prioritize: Critical issues first, then improvements

## Unified Skills

Both Red and Blue use the same three unified skills, but load different perspective resources:

### security_review

**Purpose**: Security vulnerability assessment and defense

**Red loads**: `resources/red_perspective.md`
- Attack surface enumeration
- Vulnerability patterns (buffer overflows, injection, etc.)
- Exploit chain construction
- Severity assessment

**Blue loads**: `resources/blue_perspective.md`
- Mitigation strategies
- Defense-in-depth layers
- Detection and monitoring
- Hardening recommendations

**Shared**: `resources/vulnerability_database.md`
- CWE/CVE mappings
- Common vulnerabilities with examples
- Platform-specific issues (C++, CUDA, wxWidgets)

### skill_review

**Purpose**: Claude Code skill quality assessment

**Red loads**: `resources/red_perspective.md`
- Clarity issues
- Completeness gaps
- Failure modes
- Usability problems

**Blue loads**: `resources/blue_perspective.md`
- Strengths identification
- Improvement opportunities (quick wins → long-term)
- Patterns worth replicating
- Enhancement recommendations

**Shared**: `resources/skill_best_practices.md`
- Anthropic progressive disclosure guidelines
- Skill structure best practices
- Common anti-patterns

### plan_review

**Purpose**: Project plan and architecture assessment

**Red loads**: `resources/red_perspective.md`
- Risky assumptions
- Missing elements
- Dependency risks
- Failure scenarios

**Blue loads**: `resources/blue_perspective.md`
- Strengths in plan
- Risk mitigation strategies
- Incremental delivery approaches
- Quick wins and opportunities

**Shared**: `resources/planning_frameworks.md`
- Estimation techniques
- Risk assessment frameworks
- Planning anti-patterns
- Incremental delivery patterns

## Example: Security Review

**Scenario**: Review socket protocol implementation for security vulnerabilities

### Step 1: Invoke in Parallel

Main agent:
```
I need to review src/core/socket_protocol.cpp for security vulnerabilities.
Let me invoke Lab Tech Red for vulnerability analysis and Lab Tech Blue for mitigation strategies.
```

**Red's task**: "Analyze src/core/socket_protocol.cpp for security vulnerabilities using the security_review skill. Load resources/red_perspective.md and resources/vulnerability_database.md. Identify attack surfaces, vulnerability patterns, and severity."

**Blue's task**: "For src/core/socket_protocol.cpp, provide defensive recommendations using the security_review skill. Load resources/blue_perspective.md. Propose mitigations for common socket vulnerabilities and hardening strategies."

### Step 2: Red Responds

```
CRITICAL: Buffer overflow in parseMessage()
- Location: socket_protocol.cpp:127
- Pattern: CWE-119, no bounds checking on message length
- Attack: Send length=0xFFFFFFFF, causes huge allocation
- Impact: DoS or code execution
- Severity: 9.5/10 (Critical)

MAJOR: No input validation on message type
- Location: socket_protocol.cpp:156
- Pattern: CWE-20, trusts client-provided type field
- Attack: Send invalid type, causes switch() to hit default with undefined behavior
- Impact: Crashes or unexpected state
- Severity: 7.0/10 (High)
```

### Step 3: Blue Responds

```
Current strengths:
- Uses TLS for transport (good)
- Error handling present (partially)

For CRITICAL buffer overflow:
Priority 0 mitigation:
- Add maximum message size check (MAX_MSG_SIZE = 1MB)
- Validate length < MAX_MSG_SIZE before allocation
- Code:
  if (length > MAX_MSG_SIZE) { return ERROR_TOO_LARGE; }
- Effort: 30 minutes
- Testing: Unit test with oversized messages

For MAJOR input validation:
Priority 1 mitigation:
- Whitelist valid message types
- Reject unknown types immediately
- Code:
  if (type < MSG_MIN || type > MSG_MAX) { return ERROR_INVALID_TYPE; }
- Effort: 1 hour
- Testing: Fuzz testing with random types

Quick wins:
- Enable -D_FORTIFY_SOURCE=2 compiler flag (5 min, catches some overflows)
- Add bounds checking to all recv() calls (2 hours, defense in depth)
```

### Step 4: Synthesis

Main agent synthesizes:
```
Security analysis of socket_protocol.cpp:

CRITICAL issues requiring immediate fix:
1. Buffer overflow (line 127): Add MAX_MSG_SIZE validation
   - Mitigation: Check length before allocation
   - Effort: 30 minutes
   - Status: Must fix before deployment

HIGH priority issues:
2. Input validation missing (line 156): Whitelist message types
   - Mitigation: Range check on type field
   - Effort: 1 hour
   - Status: Fix in current sprint

Quick wins to implement:
- Enable _FORTIFY_SOURCE (5 min)
- Bounds checking on recv() calls (2 hours)

Recommendation: Address CRITICAL issue immediately. HIGH and quick wins can follow.
```

## Example: Skill Review

**Scenario**: Review methodological_skill_creation skill for quality

### Step 1: Invoke in Parallel

**Red's task**: "Review .claude/skills/methodological_skill_creation/SKILL.md using skill_review skill. Load resources/red_perspective.md. Identify clarity issues, completeness gaps, and failure modes."

**Blue's task**: "Review .claude/skills/methodological_skill_creation/SKILL.md using skill_review skill. Load resources/blue_perspective.md. Identify strengths, patterns worth replicating, and improvement opportunities."

### Step 2-4: [Similar flow to security example]

Red identifies: Missing examples, unclear progressive disclosure, no "when NOT to use"
Blue proposes: Quick wins (add examples, 30 min each), medium-term (improve structure)
Main agent synthesizes: Prioritized improvement list with effort estimates

## Example: Plan Review

**Scenario**: Review plan to implement GPU acceleration

### Step 1: Invoke in Parallel

**Red's task**: "Review GPU acceleration plan using plan_review skill. Load resources/red_perspective.md. Identify risky assumptions, missing elements, and failure scenarios."

**Blue's task**: "Review GPU acceleration plan using plan_review skill. Load resources/blue_perspective.md. Identify strengths, risk mitigation strategies, and incremental delivery opportunities."

### Step 2-4: [Similar flow]

Red identifies: Optimistic timeline, learning curve underestimated, no fallback plan
Blue proposes: Spike work to validate (1 week), phased approach (CPU → simple GPU → optimized), fallback to CPU
Main agent synthesizes: Revised plan with spike, phases, and decision points

## Iteration Guidelines

### When to Follow Up

Follow up with Red when:
- Finding is vague ("security issues") → need specifics
- Severity unclear → need impact assessment
- Multiple failure scenarios → need prioritization
- Want deeper analysis of specific risk

Follow up with Blue when:
- Mitigation approach unclear → need implementation details
- Effort estimate needed → need time/resource assessment
- Multiple options presented → need recommendation
- Want to explore alternative approaches

### Example Follow-Up

**After initial Red response**:
```
Task(lab-tech-red): "You identified buffer overflow as CRITICAL. Can you provide:
1. Specific exploit scenario
2. Proof-of-concept code demonstrating the vulnerability
3. Assessment of exploitability (local vs. remote, authentication needed?)"
```

**After initial Blue response**:
```
Task(lab-tech-blue): "For the buffer overflow mitigation, you suggested MAX_MSG_SIZE check.
1. What should MAX_MSG_SIZE be set to and why?
2. Are there other bounds we should check simultaneously?
3. What testing strategy ensures this mitigation is complete?"
```

## Anti-Patterns

### Don't: Use for simple tasks

❌ Bad:
```
Task(lab-tech-red): "Check if this variable name is good"
```

✓ Good: Answer directly without lab techs

### Don't: Use sequentially when parallel works

❌ Bad:
```
Task(lab-tech-red): [wait for response]
[After Red completes]
Task(lab-tech-blue): [wait for response]
```

✓ Good:
```
Task(lab-tech-red): [analysis task]
Task(lab-tech-blue): [analysis task]
[Both work in parallel]
```

### Don't: Forget to synthesize

❌ Bad:
```
[Red and Blue respond]
[Main agent just forwards both responses to user]
```

✓ Good:
```
[Red and Blue respond]
[Main agent synthesizes, prioritizes, creates action plan]
[Main agent presents coherent recommendation]
```

### Don't: Over-iterate

❌ Bad:
```
[10 rounds of back-and-forth questions]
```

✓ Good:
```
[Initial analysis]
[1-2 rounds of clarification if needed]
[Synthesize and decide]
```

## Best Practices

### 1. Frame the Question Clearly

**Instead of**: "Look at this code"
**Better**: "Analyze src/core/parser.cpp for memory safety vulnerabilities (buffer overflows, use-after-free, integer overflows)"

**Why**: Clear scope helps Red and Blue focus their analysis

### 2. Specify the Skill

**Always tell them which skill to use**:
- Security? → `security_review`
- Skill quality? → `skill_review`
- Plan assessment? → `plan_review`

### 3. Synthesize Thoughtfully

Don't just concatenate Red and Blue responses. Instead:
- Extract key findings from Red
- Map Blue's mitigations to Red's findings
- Prioritize by severity and effort
- Create actionable recommendations
- Present as coherent analysis

### 4. Use Parallel Invocation

Launch both in single message when possible:
```markdown
I'm going to use Lab Tech Red and Blue to analyze this security issue.

[Invoke both with Task tool in same message]
```

### 5. Trust Their Expertise

Red and Blue are expert analysts. If they identify an issue or propose a solution:
- Take it seriously
- Don't dismiss without investigation
- Ask for clarification if needed
- Trust their severity assessments

## Context Management

**Key principle**: Lab techs preserve YOUR context by working in isolated sub-agent contexts.

**When you invoke a lab tech**:
1. Sub-agent spawns with ~20k token baseline
2. Loads specified skill + resources
3. Performs analysis
4. Returns findings
5. Context released

**You retain context for**:
- User conversation
- Code under review
- Synthesis of findings
- Final recommendations

**Don't load skills yourself** - delegate to lab techs. That's what they're for.

## Summary

**Pattern**: Main agent → Red (critical) + Blue (constructive) → Synthesis

**Skills**: security_review, skill_review, plan_review (unified, red/blue perspectives)

**Invocation**: Parallel via Task tool, Red listed first

**Iteration**: Follow up for clarification, but don't over-iterate

**Synthesis**: Combine perspectives into actionable recommendations

**Trust**: Lab techs are experts, take their findings seriously

The lab tech system provides balanced, thorough analysis while preserving your context for coordination and synthesis.
