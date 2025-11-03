# Commit Message Mining

Extract structured information and context from commit messages using trailers and patterns.

## Git Trailers

**Trailers** are structured metadata lines at the end of commit messages, like RFC 822 email headers.

### Common Trailers

```
Signed-off-by: Alice <alice@example.com>
Co-authored-by: Bob <bob@example.com>
Reviewed-by: Charlie <charlie@example.com>
Fixes: #123
Closes: #456
Resolves: #789
See: https://issue-tracker.example.com/item
Tested-by: David <david@example.com>
```

## Searching by Trailers

### Find Specific Trailer

```bash
# Commits with "Fixes" trailer
git log --grep="^Fixes:"

# Reviewed by specific person
git log --grep="^Reviewed-by: Alice"
```

### Multiple Trailers (OR)

```bash
# Commits that fix OR close issues
git log --grep="^Closes:" --grep="^Fixes:"
```

### Multiple Trailers (AND)

```bash
# Commits that close AND were reviewed
git log --all-match --grep="^Closes:" --grep="^Reviewed-by:"
```

## Extracting Trailers

### Using git interpret-trailers

```bash
# Show trailers from a commit
git show --format=%B <commit> | git interpret-trailers

# Only show trailers (no commit message)
git interpret-trailers --only-trailers commit-message.txt

# Parse specific trailer
git interpret-trailers --only-trailers --trailer="Fixes" commit-message.txt
```

## Issue Tracking Integration

### Find Commits Related to Issue

```bash
# All references to issue #123
git log --grep="#123"

# Only commits that fix issues
git log --grep="^Fixes: #[0-9]"

# Extract all issue references
git log --format=%B | grep -oE "(Fixes|Closes|Resolves): #[0-9]+" | sort | uniq
```

### Auto-Close Integration

GitHub, GitLab, and Bitbucket auto-close issues when commits with these trailers are merged:

- `Fixes #123`
- `Closes #456`
- `Resolves #789`

**Best practice**: Always link commits to issues.

## Author Analysis with Trailers

### Count Reviewers

```bash
# Who reviews code most?
git shortlog -ns --group=trailer:reviewed-by
```

### Count Co-Authors

```bash
# Who collaborates most?
git shortlog -ns --group=trailer:co-authored-by
```

### Combined Stats

```bash
# Count both authors and co-authors (pair programming)
git shortlog -ns --group=author --group=trailer:co-authored-by
```

## Custom Trailers

Define your own structured metadata:

```
Performance-Impact: High
Breaking-Change: Yes
Migration-Guide: docs/migrations/v2-to-v3.md
Security-Fix: CVE-2025-1234
Benchmark: 2x faster than previous
```

Then search:

```bash
git log --grep="^Breaking-Change: Yes"
git log --grep="^Security-Fix: CVE"
```

## Temporal Analysis

### Commits Over Time

```bash
# Commits per month
git log --date=format:%Y-%m --format=%ad | sort | uniq -c

# Fixes per month
git log --grep="^Fixes:" --date=format:%Y-%m --format=%ad | sort | uniq -c
```

### Issue Resolution Rate

```bash
# Count closed issues per week
git log --grep="^Closes:" --since="3.months" --format="%ai" | \
  cut -d' ' -f1 | cut -d'-' -f1,2 | uniq -c
```

## Pattern Matching

### Find Keywords in Messages

```bash
# Case-insensitive search
git log --grep="refactor" -i

# Multiple keywords
git log --grep="performance" --grep="optimization" -i

# Regex
git log --grep="fix.*bug" -E
```

### Exclude Patterns

```bash
# Exclude merge commits
git log --grep="feature" --no-merges

# Exclude specific authors
git log --grep="bug" --invert-grep --grep="^Revert"
```

## Tally All Trailers

### Repository-Wide Analysis

```bash
# List all trailer types used
git log --format=%B | git interpret-trailers --parse | \
  cut -d: -f1 | sort | uniq -c | sort -rn

# Count specific trailer
git log --format=%B | grep -c "^Signed-off-by:"
```

### Example Output

```
    342 Signed-off-by
     89 Reviewed-by
     56 Fixes
     34 Co-authored-by
     12 Tested-by
```

## Workflow Integration

### Pre-Commit Hook

Enforce trailer requirements:

```bash
#!/bin/bash
# .git/hooks/commit-msg

commit_msg=$(cat "$1")

# Require issue reference
if ! echo "$commit_msg" | grep -qE "Fixes: #[0-9]|Closes: #[0-9]"; then
    echo "ERROR: Commit must reference an issue (Fixes: #123)"
    exit 1
fi
```

### PR Templates

GitHub PR template with trailers:

```markdown
## Description
...

## Related Issues
Fixes: #
Closes: #

## Reviewers
Reviewed-by:
Tested-by:
```

## Common Patterns

### Pattern 1: Find Related Work

```bash
# Author's commits around same time
git log --author="Alice" \
  --since="<commit-date>" \
  --until="1.week.later" \
  --oneline
```

### Pattern 2: Security Audit

```bash
# All security fixes
git log --grep="Security-Fix:" --format="%h %ai %s"

# Who worked on security
git log --grep="Security-Fix:" | \
  git shortlog -ns
```

### Pattern 3: Breaking Changes

```bash
# List all breaking changes
git log --grep="Breaking-Change: Yes" --format="- %ai: %s"

# Generate migration guide
git log --grep="Migration-Guide:" --format="%b" | \
  grep "Migration-Guide:" | cut -d: -f2-
```

## Best Practices

✅ **Do:**
- Use consistent trailer format
- Link commits to issues
- Document breaking changes
- Credit co-authors and reviewers
- Use structured metadata for automation

✅ **Trailer format**:
```
Key: Value
Key: Value with spaces
Key: #123
```

❌ **Don't:**
- Mix formats (e.g., "Fixes #123" and "fixes: #123")
- Use non-standard trailer keys without documentation
- Forget to document custom trailers
- Skip trailers for significant commits

## Integration Examples

### Generate Changelog

```bash
# Extract all "Fixes" commits since last release
git log v1.0..HEAD --grep="^Fixes:" --format="- %s (%h)"
```

### Security Report

```bash
# All security fixes with reviewers
git log --grep="Security-Fix:" --format="%H" | while read commit; do
  echo "Commit: $commit"
  git show --format=%B $commit | git interpret-trailers --only-trailers
  echo
done
```

### Team Collaboration Metrics

```bash
# Pair programming frequency
pairs=$(git log --format=%B | grep "Co-authored-by:" | wc -l)
total=$(git log --oneline | wc -l)
echo "Pair programming: $pairs/$total commits ($((pairs*100/total))%)"
```

## Next Steps

- **Need to find code origins?** See `git_blame_guide.md`
- **Need to find when code changed?** See `pickaxe_techniques.md`
- **Interactive workflows?** See `expert_workflows.md`
- **Systematic investigation?** See `../templates/investigation_workflow.md`
