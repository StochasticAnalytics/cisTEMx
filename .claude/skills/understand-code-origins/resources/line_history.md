# Line-Level History with git log -L

Track the evolution of specific lines or functions through history, following them through renames and refactorings.

## Basic Usage

### Track by Line Range

```bash
# Lines 10-20
git log -L 10,20:path/to/file.cpp

# Single line
git log -L 42,42:file.cpp

# From line 50 to end
git log -L 50,:file.cpp
```

### Track by Function

```bash
# Trace specific function
git log -L :functionName:path/to/file.cpp

# Works with C, C++, Python, Java, etc.
git log -L :calculateTotal:src/billing.py
```

**How it works**: Git uses language-specific heuristics to find function boundaries.

## Output

Shows every commit that modified the specified lines:

```
commit abc1234
Author: Alice
Date:   2024-10-15

    Optimize calculation

diff --git a/file.cpp b/file.cpp
--- a/file.cpp
+++ b/file.cpp
@@ @ ... @@
-    old implementation
+    new implementation
```

## Regex Ranges

### Relative Ranges

```bash
# Line matching regex, plus 5 lines after
git log -L /regex/,+5:file.cpp

# Between two regex matches
git log -L /start_pattern/,/end_pattern/:file.cpp

# From start of file to regex
git log -L ^/pattern/,+10:file.cpp
```

### Example: Track Error Handling

```bash
# Find all changes to error handling block
git log -L /try\s*{/,/catch\s*\(/file.cpp
```

## Advanced Options

### With Patches

```bash
# Show full patches for function changes
git log -L :function:file.cpp -p
```

### Across Renames (Experimental)

```bash
# Try to follow function across file renames
git log -L :function:file.cpp --follow
```

**Note**: --follow with -L is experimental and may not always work.

### All Branches

```bash
# Search function history across all branches
git log --all -L :function:file.cpp
```

### Time Range

```bash
# Function history for last year
git log --since="1.year.ago" -L :function:file.cpp
```

## Language Support

Git detects functions using built-in patterns for:

- C/C++
- Python
- Java
- C#
- Ruby
- Go
- PHP
- Perl
- And more...

### Custom Function Patterns

For unsupported languages, define patterns in `.gitattributes`:

```
*.mylang diff=mylang
```

Then in `.git/config`:
```
[diff "mylang"]
    xfuncname = "^function.*"
```

## Common Use Cases

### Use Case 1: Understanding Function Evolution

```bash
# See all changes to authentication logic
git log -L :authenticateUser:src/auth.cpp

# See who made each change
git log -L :authenticateUser:src/auth.cpp --format="%h %an %s"
```

### Use Case 2: Finding When Logic Changed

```bash
# Track specific algorithm block
git log -L 100,150:src/algorithm.cpp

# With context
git log -L 100,150:src/algorithm.cpp -p -U10
```

### Use Case 3: Refactoring History

```bash
# See how function was refactored over time
git log -L :complexFunction:src/core.cpp --oneline

# Count how many times it changed
git log -L :complexFunction:src/core.cpp --oneline | wc -l
```

## Limitations

**Known limitations**:
- Cannot specify multiple paths (single file only)
- Function detection may not work for unusual code styles
- --follow support is experimental
- Performance can be slow for heavily modified lines
- Limited to single line range per invocation

## Performance Optimization

### For Large Histories

```bash
# Limit to recent history
git log --since="6.months.ago" -L :function:file.cpp

# Limit to specific branch
git log main -L :function:file.cpp

# First N commits
git log -n 10 -L :function:file.cpp
```

### With Commit-Graph

Enable commit-graph for faster log operations:

```bash
git config core.commitGraph true
git commit-graph write --reachable --changed-paths
```

## Integration with Other Tools

### Combine with Blame

```bash
# Find current author
git blame -L :function file.cpp

# See full history
git log -L :function:file.cpp
```

### Combine with Bisect

```bash
# If function behavior changed
git bisect start HEAD v1.0
git bisect run sh -c "git show HEAD:file.cpp | grep 'expected_behavior'"
```

### Extract Specific Commit

```bash
# Get function at specific commit
git show <commit>:file.cpp | sed -n '/function_start/,/function_end/p'
```

## Troubleshooting

**"Function not found"**:
- Check exact function name and signature
- Try numeric line range instead
- Language may not be supported
- Function may be defined in unexpected way

**"Too many results"**:
- Add time constraints (--since)
- Limit to specific branch
- Use -n to limit result count

**"Performance too slow"**:
- Enable commit-graph with Bloom filters
- Limit time range
- Avoid --all unless necessary

**"Doesn't follow renames"**:
- --follow with -L is experimental
- Use pickaxe manually to track across renames:
  ```bash
  git log -S"functionName" --all --follow
  ```

## Real-World Examples

### Example 1: Security Audit

```bash
# Track all changes to authentication function
git log -L :verifyPassword:src/auth.cpp \
  --format="%h %an %ae %ai %s" \
  > security_audit_auth.csv
```

### Example 2: Performance Investigation

```bash
# See when performance-critical loop changed
git log -L 200,250:src/renderer.cpp

# Compare versions
git show abc1234:src/renderer.cpp | sed -n '200,250p'
git show def5678:src/renderer.cpp | sed -n '200,250p'
```

### Example 3: API Evolution

```bash
# Track public API function
git log -L :publicAPIcall:include/api.h

# Generate changelog
git log -L :publicAPIcall:include/api.h --format="- %ai: %s"
```

## Next Steps

- **Need to find who wrote code?** See `git_blame_guide.md`
- **Need to find when code changed?** See `pickaxe_techniques.md`
- **Want structured metadata?** See `commit_mining.md`
- **Systematic workflow?** See `expert_workflows.md`
