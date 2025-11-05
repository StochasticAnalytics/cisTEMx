# Claude Vocabulary Guidance

## Purpose

This file guides vocabulary choices to enhance scientific communication quality and precision. Academic and technical writing demands careful word selection that conveys appropriate certainty levels, avoids exaggeration, and maintains professional objectivity.

Hyperbolic or absolutist language undermines credibility in scientific contexts by:
- Overstating certainty where evidence is limited
- Creating unrealistic expectations about solutions or approaches
- Reducing precision in describing outcomes and constraints
- Conflating enthusiasm with rigor

Effective scientific communication uses measured, precise language that accurately represents both confidence and uncertainty.

## Practical Implications

Beyond scientific rigor, imprecise vocabulary creates real workflow problems:

**Version Control Impact**: Commit messages with terms like "final fix", "completely resolved", or "critical bug" when inaccurate create false signals in git history. This leads to:
- Wasted time investigating supposedly "final" fixes that weren't
- Misallocated urgency based on overstated severity
- Reduced trust in commit message accuracy
- False search hits when hunting for actual critical issues

**Communication Overhead**: Exaggerated language in documentation, error messages, or status updates forces readers to:
- Discount claims and verify independently
- Spend mental effort filtering signal from noise
- Question whether terms have specific meaning or are simply emphasis

**Debugging Efficiency**: When "impossible" errors occur or "guaranteed" solutions fail, the mismatch between stated confidence and reality wastes debugging time chasing assumptions rather than investigating actual behavior.

Precise vocabulary isn't about being less enthusiastic—it's about being more useful.

## Epistemological Framework

As scientists and engineers, we maintain a Bayesian outlook: we do not believe in fixed, knowable truth, but rather that our understanding is inherently limited and probabilistic. We update our beliefs in the face of new evidence. This perspective shapes how we communicate.

Claims of absolute certainty, completeness, or universality conflict with this framework. Our vocabulary should reflect appropriate epistemic humility—acknowledging what we know, what we don't know, and the conditional nature of our conclusions.

## Anti-Patterns: Words and Phrases to Avoid

### Category 1: Epistemic Certainty Claims (Almost Never Appropriate)

These terms claim absolute knowledge, completeness, or universality. They conflict with Bayesian thinking where we update beliefs probabilistically with evidence.

**Totality/Completeness Claims**:
- comprehensive (implies exhaustive coverage, but edge cases and future changes exist)
- complete (claims nothing is missing)
- all (universal quantifier rarely justified)
- fully (claims no gaps remain)
- entirely (claims absolute scope)
- final (implies no future revision needed)

**Absolute Determiners**:
- always (universal claim across all cases)
- never (negative universal, equally problematic)
- certain (claims zero uncertainty)
- definitely (removes possibility of being wrong)
- guaranteed (promises invariant outcome)
- impossible (claims absolute constraint)
- infallible (claims perfection)
- foolproof (claims no failure mode exists)

**Universality of Perspective**:
- obvious (assumes universal understanding, dismisses different expertise levels)
- clearly (similar assumption about shared perspective)

**Why these are problematic**: They claim epistemic completeness in a domain where:
- Edge cases always exist
- Future changes invalidate "final" states
- Different expertise levels mean "obvious" is relative
- Stochastic systems have irreducible uncertainty
- Our understanding is always provisional and updating

**Better alternatives**: "covers major cases", "addresses known issues", "tested scenarios", "current implementation", "observed in these conditions", "suggests", "indicates", "in most cases"

### Category 2: Severity/Priority Terms (Appropriate with Clear Evidence)

These describe urgency or importance, not certainty. They CAN be appropriate when evidence supports the claim.

**"Critical"**:
- **Appropriate**: Security vulnerability with active exploitation risk, data corruption/loss, failure blocking all users from core functionality, build system preventing compilation
- **Inappropriate**: Edge case bugs, performance slowdowns, missing features, non-essential test failures
- **Usage principle**: Requires specific evidence of severity (CVE number, scope metrics, impact assessment)

**Other severity terms**: Similar principles apply to "urgent", "severe", "high-priority" - they're defensible when justified with evidence, problematic when used for emphasis.

### Category 3: Other Problematic Patterns

**Superlatives and Hyperbole**:
Excessive enthusiasm that diminishes objectivity without adding information:
- amazing, incredible, fantastic, awesome, spectacular
- revolutionary, groundbreaking, paradigm-shifting, game-changing

**Dismissive Simplifiers**:
Undermine complexity of work and can alienate readers:
- trivial (dismisses effort and complexity)
- simple, straightforward, easy (relative to expertise)
- just (as in "just do X" - minimizes actual complexity)

**Overpromising Adverbs**:
Suggest unrealistic ease or perfection:
- seamlessly (claims no integration friction)
- effortlessly (claims no work required)
- flawlessly (claims perfection)
- instantly (claims zero latency)
- automatically (when conditions/configuration required)

### Better Alternatives

Instead of absolutist language, prefer:
- Qualified statements: "typically", "generally", "often", "in most cases"
- Measured uncertainty: "likely", "probable", "suggests", "indicates"
- Conditional phrasing: "when configured correctly", "given these constraints"
- Precise descriptors: "reduced by 40%", "completed in 3 steps", "requires 2 dependencies"
- Accurate scope: "addresses Issue #123", "fixes segfault in parser", "updates three related functions"

---

## Future Extensions

This document may be extended to include:
- Positive vocabulary patterns for effective scientific communication
- Domain-specific terminology guidance
- Context-dependent phrasing recommendations
