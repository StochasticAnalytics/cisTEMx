# Expert Workflows for Code Archaeology

Proven investigation patterns from experienced developers.

## John Firebaugh's Multi-Tool Workflow

**Problem**: Blame often lands on refactoring commits, hiding real origins.

**Solution**: Three-tool approach

### Step 1: Start with Blame (with noise reduction)

```bash
git blame -wCCC path/to/file.rb
```

**Flags**:
- `-w`: Ignore whitespace changes
- `-CCC`: Detect moves/copies aggressively

**Result**: May show refactoring commit, but filtered most noise.

### Step 2: If Refactoring, Use Pickaxe

When blame shows "Extract method" or "Rename variable":

```bash
git log -S'suspicious_code' path/to/file.rb
```

**Result**: Finds when code was actually introduced (not moved).

### Step 3: Interactive Drilling with git gui blame

```bash
git gui blame path/to/file.rb
```

**Workflow**:
1. Right-click on line showing refactoring
2. Select "Blame Parent Commit"
3. Git re-blames from before refactoring
4. Repeat until find real origin

**Quote**: "git gui blame is my go-to tool for code archeology" - John Firebaugh

**Why it works**: Interactively skips refactorings without manual bisecting.

## General Investigation Process

**5-phase systematic approach:**

### Phase 1: Identify the Question

- What code are you investigating?
- What do you need to know?
  - Why was it added?
  - When did it break?
  - Who knows about it?

### Phase 2: Broad Search

```bash
# When was string added?
git log -S'code_snippet' --all

# Find modifications
git log -G'regex_pattern' --all

# Search commit messages
git log --all --grep="feature name"
```

### Phase 3: Narrow Down

```bash
# Who wrote it?
git blame -wM file.c

# Function history
git log -L :function_name:file.c

# Specific time range
git log --since="6.month" -- file.c
```

### Phase 4: Gather Context

```bash
# Full commit
git show <commit-hash>

# Related commits
git log --oneline --graph --all -- file.c

# What else changed in commit
git diff-tree --no-commit-id --name-only -r <commit-hash>
```

### Phase 5: Verification (optional)

```bash
# If looking for bug introduction
git bisect start
git bisect bad <bad-commit>
git bisect good <good-commit>
git bisect run ./test.sh
```

## File Rename Workflow

**Problem**: File was renamed multiple times.

**Solution**: Iterative following

```bash
# 1. Start with current filename
git log --follow current-name.c

# 2. Find rename commit
git log --follow --name-status current-name.c | grep "^R"

# Output: R100  old-name.c  current-name.c

# 3. Continue with old name
git log --follow old-name.c

# 4. Combine results
git log --all -- current-name.c old-name.c
```

## Understanding Design Decisions

**Goal**: Understand why code is designed a certain way.

```bash
# 1. Find introduction commit
git log -S'class DesignPattern' --diff-filter=A

# 2. Read commit message and diff
git show <commit-hash>

# 3. Find related discussions (GitHub/GitLab)
gh pr list --search="<commit-hash>"

# 4. Check for related issues
git log --grep="issue.*[0-9]" --oneline

# 5. Look at before/after context
git log --oneline --graph <commit-hash>~5..<commit-hash>+5
```

## Security Audit Workflow

**Goal**: Audit who touched sensitive code.

```bash
# 1. All commits to security code
git log --all -- src/auth/ src/crypto/

# 2. Detailed blame
git blame -e -l --date=iso src/auth/login.c

# 3. Check for sensitive patterns
git log -G'password|secret|token' --all -- src/

# 4. Find reviewers
git log --format=%B -- src/auth/ | grep "Reviewed-by:" | sort | uniq -c

# 5. Generate audit report
git log --format='%H|%an|%ae|%ai|%s' -- src/auth/ > security-audit.csv
```

## Interactive Tools

### git gui blame

**Best for**: Iteratively skipping refactorings.

```bash
git gui blame <file>
```

**Key features**:
- Dual columns (original author + mover)
- Right-click menu: "Blame Parent Commit"
- Visual commit browsing
- Built-in git show integration

### tig

**Best for**: Browsing history interactively.

```bash
# Browse commits
tig

# Blame mode
tig blame file.c

# Search mode
tig grep "pattern"
```

**Navigation**:
- `j/k`: Up/down
- `Enter`: View commit
- `q`: Back/quit
- `/`: Search

### GitLens (VS Code)

**Best for**: In-editor investigation.

**Features**:
- Inline blame annotations
- File history sidebar
- Line history view
- Commit graph
- Heatmap visualization

## Performance Optimization

### For Large Files

```bash
# Limit to relevant section
git blame -wCCC -L :functionName large_file.cpp

# Limit to recent history
git blame --since="1.year.ago" file.cpp
```

### For Large Repositories

```bash
# Enable commit-graph
git config core.commitGraph true
git commit-graph write --reachable --changed-paths

# Use Bloom filters for faster blame
git config commitGraph.generationVersion 2
```

### Limit Pickaxe Scope

```bash
# Specific paths only
git log -S"text" -- src/

# Single branch
git log -S"text" main

# Time range
git log -S"text" --since="1.year.ago"
```

## Common Patterns Summary

| Goal | Primary Tool | Backup Tool |
|------|-------------|-------------|
| Find author | `git blame -wCCC` | `git gui blame` |
| Find when added | `git log -S` | `git log -G` |
| Skip refactoring | `git gui blame` | Manual pickaxe |
| Function evolution | `git log -L` | `git log -S` + `git show` |
| Find context | `git show` | `git log --grep` |
| Track renames | `git log --follow` | Manual `-S` search |

## Anti-Patterns to Avoid

❌ **Don't**:
- Trust blame without checking for refactorings
- Use `git log` without paths on huge repos (too slow)
- Blame without `-w` (whitespace hides real authors)
- Use `-CCC` without good reason on massive repos (slow!)
- Forget `--follow` when file history seems incomplete
- Manually bisect when automation is possible

## Tool Selection Decision Tree

```
Need to find who wrote code?
  → Start: git blame -wCCC
  → Lands on refactoring? → git gui blame (interactive)
  → Still unclear? → git log -S

Need to find when code changed?
  → String added/removed? → git log -S
  → Line modified? → git log -G
  → Function evolution? → git log -L

Need context?
  → Commit details → git show <hash>
  → Related work → git log --grep
  → Issue tracking → Search trailers

Interactive exploration?
  → Visual drilling → git gui blame
  → History browsing → tig
  → In-editor → GitLens (VS Code)
```

## Real-World Success Stories

### Case 1: Linux Kernel Bug Hunt

**Tool**: git bisect + git blame
**Result**: Found 10-year-old commit introducing race condition
**Time**: 45 minutes (would have been weeks of manual review)

### Case 2: API Design Decision

**Tool**: git log -S + commit trailers
**Result**: Found original RFC and mailing list discussion
**Context**: Linked to 5-year-old design document

### Case 3: Security Vulnerability

**Tool**: git blame -CCC + git log -G
**Result**: Traced vulnerable code through 3 refactorings to original copy-paste from old library
**Action**: Fixed root cause, not just symptom

## Next Steps

- **Learn techniques?** See other `resources/*.md` files
- **Ready to investigate?** See `../templates/investigation_workflow.md`
- **Need automation?** See `../scripts/investigate_code.sh`
