# Commit Message Examples

Real-world examples demonstrating best practices for cisTEMx commits.

## New Features

```
feat: Add GPU memory pool for FFT operations

Implements CUDA memory pool to reduce allocation overhead
during iterative refinement. Reduces memory fragmentation
and improves performance by ~15% for large datasets.

Refs: #156
```

```
feat: Support MRC2014 file format in image I/O
```

## Bug Fixes

```
fix: Prevent segfault when loading corrupted STAR files

Add validation for required columns before accessing data.
Corrupted files now fail gracefully with error message
instead of crashing.

Fixes: #203
```

```
fix: Correct orientation angle calculation for particle extraction
```

## Refactoring

```
refactor: Extract FFT wrapper into separate class

Moves FFT initialization and execution logic from Image
class to dedicated FFTWrapper. No behavior change, improves
testability and prepares for multi-backend support.
```

```
refactor: Simplify database connection pooling logic
```

## Performance

```
perf: Optimize CTF calculation with SIMD intrinsics

Replace scalar loop with AVX2 vectorized implementation.
Measured 3.8x speedup for typical CTF parameter ranges.
```

```
perf: Cache particle mask generation during classification
```

## Tests

```
test: Add unit tests for socket protocol parser

Covers binary encoding/decoding, endianness handling,
and error cases for malformed messages.
```

```
test: Add regression test for issue #187
```

## Documentation

```
docs: Document GPU memory requirements for 3D refinement

Add memory estimation formulas and configuration
recommendations based on volume size.
```

```
docs: Update CLAUDE.md with lab tech consultation protocol
```

## Build System

```
build: Update wxWidgets to 3.2.4

Includes fixes for high-DPI display support on Linux
and memory leak in grid control.
```

```
build: Add clang-format pre-commit hook
```

## Style (Formatting)

```
style: Apply clang-format to src/core/*.cpp

No functional changes, whitespace only.
```

## Chores

```
chore: Remove deprecated socket timeout parameter

Unused since v2.1.0, cleanup for v3.0 release.
```

```
chore: Update copyright year to 2025
```

## Multi-line Examples

### Complex Feature

```
feat: Implement adaptive particle mask generation

Adds automatic mask generation based on particle variance
and local SNR estimation. Improves classification accuracy
for heterogeneous datasets.

Algorithm:
1. Compute variance map across particle stack
2. Estimate local SNR from variance
3. Generate soft-edged mask with SNR-weighted threshold
4. Apply morphological smoothing

Performance: ~200ms per 1000 particles (256² boxes)

Refs: #178, #192
```

### Breaking Change

```
feat!: Change database schema for classification metadata

BREAKING CHANGE: Adds new required columns to ParticleStack
table. Existing databases must be migrated using the provided
migrate_db.py script before upgrading.

Migration: python scripts/migrate_db.py --database path/to/db.sqlite

Refs: #210
```

## Anti-patterns (Avoid These)

### Too Vague
```
❌ Update files
❌ Fix bug
❌ Changes
```

### Too Detailed
```
❌ fix: Change line 42 from foo(x) to bar(x) and line 56 from...
```
(Details go in commit body or code review, not summary)

### Mixed Concerns
```
❌ feat: Add new feature and fix old bug and update docs
```
(Split into separate commits)

### Past Tense
```
❌ Added new feature
❌ Fixed bug
```
(Use imperative: "Add", "Fix")

### Missing Type
```
❌ new particle picker algorithm
```
(Should be: `feat: Add new particle picker algorithm`)

## cisTEMx Specific Conventions

### GPU Code
- Mention GPU vs CPU in description when relevant
- Note memory implications for large changes
- Include performance comparisons when available

### File Format Changes
- Always document in commit body
- Note backward compatibility impact
- Reference specification if applicable

### Scientific Algorithms
- Brief algorithm description in body
- Reference papers in comments, not commits
- Performance characteristics when measured

### GUI Changes
- Mention which dialog/panel affected
- Note if wxFormBuilder files regenerated
- Screenshot in PR, not commit message
