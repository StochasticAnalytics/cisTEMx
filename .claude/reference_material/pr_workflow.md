# Pull Request Workflow Reference
## Creating and Managing Pull Requests for cisTEMx

### Critical: Origin vs Upstream

**IMPORTANT**: This repository has separate `origin` and `upstream` remotes.

- **origin**: Your fork of the repository
- **upstream**: The main cisTEMx repository

**Pull requests must be created against `upstream`, not `origin`.**

### Pre-PR Checklist

Before creating a pull request:

- [ ] All commits compile successfully
- [ ] Relevant tests pass (unit, console, functional)
- [ ] All `// revert` marked debugging code removed
- [ ] Static analysis (at least blocker-level) passes
- [ ] Code follows style guidelines
- [ ] Documentation updated if needed

### Creating a Pull Request

#### Step 1: Ensure Clean Branch
```bash
# Check for uncommitted changes
git status

# Verify all commits compile
git rebase -i --exec "make -j16" HEAD~<number-of-commits>
```

#### Step 2: Push to Your Fork
```bash
# Push your branch to origin (your fork)
git push origin feature-branch-name
```

#### Step 3: Create PR via GitHub CLI
```bash
# Create PR against upstream/master
gh pr create \
  --repo upstream-owner/cisTEMx \
  --base master \
  --head your-username:feature-branch-name \
  --title "Brief description of changes" \
  --body "Detailed explanation"
```

#### Step 4: Use PR Template
The PR must follow the template at `.github/pull_request_template.md`:

```markdown
## Summary
Brief description of what this PR does

## Motivation
Why these changes are needed

## Changes
- List of specific changes made
- File modifications
- New features/fixes

## Testing
- [ ] Unit tests pass
- [ ] Console tests pass
- [ ] Functional tests pass
- [ ] Manual testing completed

## Screenshots (if applicable)
For GUI changes
```

### Interactive PR Creation Process

The repository supports an interactive PR drafting process:

1. **Initial Draft**: Create PR with comprehensive description
2. **Review Feedback**: Address reviewer comments
3. **Update**: Push additional commits to the branch
4. **Squash if Needed**: Clean up commit history before merge

See `.github/workflows/CLAUDE.md` for detailed interactive workflow.

### Common PR Issues

#### "Can't push to upstream"
You don't have write access to upstream. Always push to origin first, then create PR.

#### "PR shows too many commits"
Your branch might be behind master:
```bash
git fetch upstream
git rebase upstream/master
git push --force origin feature-branch-name
```

#### "Merge conflicts"
```bash
# Update your local master
git fetch upstream
git checkout master
git merge upstream/master

# Rebase your feature branch
git checkout feature-branch-name
git rebase master
# Resolve conflicts
git push --force origin feature-branch-name
```

### PR Review Process

#### As PR Author
- Respond to all review comments
- Mark conversations as resolved when addressed
- Request re-review after making changes
- Don't force-push during review unless requested

#### Code Review Etiquette
- Be receptive to feedback
- Explain design decisions when questioned
- Make requested changes promptly
- Thank reviewers for their time

### CI and Automated Checks

Pull requests automatically trigger:
- Compilation checks
- Static analysis (standard tier)
- Test suite execution
- Code coverage reporting

All checks must pass before merge.

### Merging Strategy

The project typically uses:
- **Squash and merge** for feature branches
- **Merge commits** for large features with meaningful history
- **Rebase and merge** rarely, only for linear history

### Post-Merge Cleanup

After PR is merged:
```bash
# Update local master
git checkout master
git pull upstream master

# Delete local feature branch
git branch -d feature-branch-name

# Delete remote branch on fork
git push origin --delete feature-branch-name
```

### Related Documentation
- PR template: `.github/pull_request_template.md`
- CI workflows: `.github/workflows/`
- Interactive workflow: `.github/workflows/CLAUDE.md`
- Commit best practices: See main CLAUDE.md