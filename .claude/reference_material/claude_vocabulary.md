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

## Anti-Patterns: Words and Phrases to Avoid

### Absolute Qualifiers
These suggest certainty beyond what evidence typically supports:
- absolutely
- completely
- totally
- entirely
- perfectly
- fully

### Superlatives and Hyperbole
Excessive enthusiasm that diminishes objectivity:
- amazing
- incredible
- fantastic
- awesome
- spectacular
- revolutionary
- groundbreaking
- paradigm-shifting
- game-changing

### Absolute Determiners
Rarely accurate in complex technical contexts:
- always
- never
- impossible
- guaranteed
- certain
- infallible
- foolproof

### Dismissive Simplifiers
Undermine the complexity of work and can alienate readers:
- trivial
- obvious
- simple
- straightforward
- easy
- just (as in "just do X")

### Overpromising Adverbs
Suggest unrealistic ease or perfection:
- seamlessly
- effortlessly
- flawlessly
- instantly
- automatically (when qualified conditions exist)

### Version Control Red Flags
Terms that create false expectations in commit messages:
- final (when there may be follow-up work)
- critical (when severity is moderate)
- complete (when implementation is partial)
- definitely (when uncertainty exists)
- all (when scope is limited)

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
