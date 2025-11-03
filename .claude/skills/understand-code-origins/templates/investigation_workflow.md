# Code Investigation Workflow

Systematic process for understanding unfamiliar or legacy code through git history.

## Phase 1: Initial Assessment

### ☐ Identify Investigation Goal

**What do you need to know?**
- [ ] Why does this code exist?
- [ ] What problem does it solve?
- [ ] Who wrote it originally?
- [ ] When was it introduced?
- [ ] Has it been refactored?
- [ ] What's the design rationale?

**Document your question**: Write it down clearly before starting.

### ☐ Locate the Code

```bash
# Find file
find . -name "*pattern*"

# Or search by content
git grep -n "specificCode"
```

## Phase 2: Quick Blame

### ☐ Run Basic Blame

```bash
# Start with noise reduction
git blame -w file.cpp
```

**Look for**:
- Recent authors (may be maintainers)
- Commit messages (clues about purpose)
- Commit dates (age of code)

### ☐ Check for Refactorings

**Red flags in blame output**:
- "Refactor..."
- "Rename..."
- "Extract method..."
- "Move..."
- "Reformat..."

**If you see refactorings**: Proceed to Phase 3.
**If blame looks meaningful**: Skip to Phase 4 (examine commits).

## Phase 3: Drill Through Refactorings

### ☐ Use Interactive git gui blame

```bash
git gui blame file.cpp
```

**Workflow**:
1. Find line with refactoring commit
2. Right-click → "Blame Parent Commit"
3. Repeat until find meaningful commit
4. Note the commit hash

### ☐ OR: Use Pickaxe

```bash
# Find when code was actually introduced
git log -S"codeSnippet" --oneline -- file.cpp

# Show the introduction commit
git log -S"codeSnippet" -p -- file.cpp | head -100
```

## Phase 4: Examine Introduction Commit

### ☐ View Full Commit

```bash
COMMIT=<hash-from-blame-or-pickaxe>

# Full details
git show $COMMIT

# With more context
git show $COMMIT -U10
```

### ☐ Read Commit Message Carefully

**Look for**:
- Purpose statement ("Fix...", "Add...", "Implement...")
- Related work (mentions of other commits/issues)
- Trailers (Fixes: #123, Reviewed-by:, etc.)
- Design rationale

### ☐ Check Commit Context

```bash
# What else changed in that commit?
git show --stat $COMMIT

# Commits around same time
git log --oneline --graph $COMMIT~5..$COMMIT^+5
```

## Phase 5: Find Related Context

### ☐ Search for Related Discussions

**If commit references issue**:
```bash
# GitHub
gh issue view <number>

# Or search commit messages
git log --grep="<issue-number>" --oneline
```

**If no issue reference**:
```bash
# Search for keywords from commit message
git log --grep="<keyword>" --oneline

# Find author's other work around same time
git log --author="<author>" \
  --since="<commit-date>" \
  --until="1.week.later" \
  --oneline
```

### ☐ Check for Documentation

```bash
# Find related docs committed same time
git diff-tree --no-commit-id --name-only -r $COMMIT | grep -E "\.(md|txt|doc)"

# Search docs for mentions
git grep -i "feature-name" -- "*.md"
```

## Phase 6: Trace Evolution

### ☐ See How Code Changed Over Time

**For specific function**:
```bash
git log -L :functionName:file.cpp
```

**For broader context**:
```bash
# All changes to file
git log --oneline -- file.cpp

# With patches
git log -p -- file.cpp
```

### ☐ Count How Many Times Modified

```bash
# Total commits touching file
git log --oneline -- file.cpp | wc -l

# Commits modifying specific code
git log -G"pattern" --oneline -- file.cpp | wc -l
```

**Interpretation**:
- High count → frequently changed (possibly problematic)
- Low count → stable (probably well-designed)

## Phase 7: Understand Design Decision

### ☐ Reconstruct the Problem

**Questions to answer**:
- What state was codebase in before this change?
- What problem needed solving?
- What alternatives were considered?
- Why this approach?

**Commands**:
```bash
# Checkout state before change
git checkout $COMMIT^

# Review relevant files
ls -la
# Examine code...

# Return to present
git checkout -
```

### ☐ Check for Related Patterns

```bash
# Find similar code patterns
git grep -n "similarPattern"

# Find other work by same author
git log --author="<author>" --oneline

# Check if pattern used elsewhere
git log -S"pattern" --all --oneline
```

## Phase 8: Verify Understanding

### ☐ Summarize Findings

**Write down**:
1. **What**: What does the code do?
2. **Why**: Why was it added?
3. **When**: When was it introduced?
4. **Who**: Who wrote it originally?
5. **How**: How has it evolved?
6. **Context**: What problem did it solve?

### ☐ Cross-Check with Others

**If still uncertain**:
- Find original author (if available)
- Check team documentation
- Ask in team chat with context
- Review related PRs/issues

## Common Scenarios

### Scenario A: "Why This Strange Check?"

```bash
# 1. Blame the line
git blame -wCCC file.cpp | grep "strangeCheck"

# 2. View commit
git show <commit>

# 3. Often reveals: Was fixing specific bug
# Look for: Fixes: #123 in commit message
```

### Scenario B: "Why Use This Pattern?"

```bash
# 1. Find introduction
git log -S"PatternName" --diff-filter=A

# 2. Check author's other work
git log --author="<author>" --since="<date>" --until="1.month.later"

# 3. Often reveals: Part of larger refactoring or architectural change
```

### Scenario C: "Where Did This Come From?"

```bash
# 1. Aggressive blame
git blame -wCCC file.cpp

# 2. If shows recent commit, check for copies
git log -S"code" --all --oneline

# 3. Often reveals: Copied from another file or imported from library
```

## Troubleshooting

**"Can't find original commit"**:
- Code may have been migrated from another repo
- Try searching all branches: `git log --all -S"code"`
- Check for imports: `git log --grep="import\|migrate"`

**"Blame shows only refactorings"**:
- Use `git gui blame` interactively
- Or use pickaxe: `git log -S"code"`

**"Too many results"**:
- Narrow time range: `--since="1.year.ago"`
- Narrow scope: `-- specific/path/`
- Be more specific in search terms

**"Still unclear why code exists"**:
- Check related test files
- Look for design docs in repo
- Review architectural decision records (ADRs)
- Ask team members

## Success Criteria

✅ **Investigation successful when you can answer**:
1. What problem does this code solve?
2. Why this approach vs. alternatives?
3. What constraints led to this design?
4. Who has context if questions arise?
5. How has it evolved over time?

## Documentation

### ☐ Document Findings

**For your own reference**:
```markdown
# Investigation: [Feature/Code]

**Question**: Why does X do Y?

**Answer**: Added in commit abc1234 (2023-05-15) by Alice to solve Z problem.

**Context**: Part of larger refactoring for performance. See issue #123.

**Evolution**: Modified 3 times since, last change was optimization in def5678.

**Key insight**: Original constraint was memory limitation, now less relevant.
```

**For team knowledge base**:
- Add inline comments with historical context
- Update README or docs with design rationale
- Create ADR (Architecture Decision Record) if appropriate

## Next Steps

- **Found bug introduction?** Use `find-bug-introduction` skill
- **Planning refactoring?** Use `identify-refactoring-targets` skill
- **Need deeper techniques?** Review `resources/*.md` files
- **Want automation?** Use `scripts/investigate_code.sh`
