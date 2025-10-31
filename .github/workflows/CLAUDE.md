# CI/CD Workflows and Pull Request Creation

This file provides guidance for working with GitHub workflows and creating pull requests.

## Pull Request Creation Workflow

**CRITICAL: This repository uses `upstream` as the target for pull requests, NOT `origin`.**

When creating a pull request, follow this interactive workflow:

### Step 1: Draft PR According to Template

Read the PR template at `.github/pull_request_template.md` and create a complete PR description that addresses all sections:

1. **Description** - Explain what changed and **why** (motivation is critical)
2. **Rebase status** - Confirm branch is rebased to current master
3. **Compilers tested** - List which compilers were used
4. **Scope of changes** - Identify affected components (gui/core/gpu/programs)
5. **Testing performed** - Describe manual and automated testing
6. **Checklist** - Complete all checklist items

**Save the drafted PR description in `.claude/cache/pr_draft.md` for review.**

### Step 2: Present Draft to User for Review

After drafting the PR, present it to the user with these **easily selectable options**:

```
I've drafted the following pull request description (saved in .claude/cache/pr_draft.md):

[Display the full PR draft here]

Please select an option:
1. ✅ Accept - Create the PR with this description
2. ✏️ Edit - Make changes to the draft
3. 🔍 Preview in file - Open .claude/cache/pr_draft.md to review
```

### Step 3a: If User Selects "Accept"

Create the pull request immediately:

```bash
gh pr create --title "Your PR Title" --body "$(cat .claude/cache/pr_draft.md)" --base master --repo upstream
```

**Important:** Always use `--repo upstream` to ensure the PR targets the upstream repository, not origin.

### Step 3b: If User Selects "Edit"

**Enter planning mode** and follow this iterative editing process:

1. **Ask user to select a line or section** from the draft they want to edit
2. **Ask user to describe the desired change** for that line/section
3. **Add the edit to a TODO list** tracking all requested changes
4. **Ask if they want to:**
   - Select another line/section to edit (repeat from step 1)
   - Finalize and apply all edits

**Example TODO list during editing:**

```markdown
## PR Draft Edits

- [ ] Line 5: Change "Fixed bug" to "Fixed race condition in socket communication"
- [ ] Section "Testing performed": Add details about functional testing with 100 images
- [ ] Checklist: Mark "Passed console tests" as checked
```

**When user finalizes edits:**

1. Apply all changes from the TODO list to the draft
2. Save updated draft to `.claude/cache/pr_draft.md`
3. Present the updated draft again with the same three options (Accept/Edit/Preview)

### Step 4: Verify PR Target

Before creating the PR, verify remotes and target:

```bash
# Check remotes
git remote -v

# Verify upstream exists and is correct
git remote get-url upstream
```

**Expected remotes:**

- `origin` - Your fork (e.g., `github.com:YourUsername/cisTEMx.git`)
- `upstream` - Main repository (e.g., `github.com:StochasticAnalytics/cisTEMx.git`)

PRs must target `upstream/master`, not `origin/master`.

## Best Practices

### PR Title Format

Use clear, descriptive titles:

- ✅ "Fix memory leak in CTF estimation worker threads"
- ✅ "Add CUDA 12.0 support for sm_89 architecture"
- ❌ "Bug fix"
- ❌ "Updates"

### PR Description Guidelines

**Explain WHY, not just WHAT:**

- ✅ "Changed buffer size from 1024 to 4096 bytes because larger micrographs were causing truncation errors during socket transmission"
- ❌ "Changed buffer size"

**Include context:**

- What problem does this solve?
- What alternatives were considered?
- Are there any trade-offs?
- What should reviewers pay attention to?

### Testing Requirements

Every PR should document testing performed:

- **Manual GUI testing** - If GUI changes are involved
- **Manual CLI testing** - If command-line programs are affected
- **Console tests** - `./console_test` for core functionality
- **Functional tests** - `./samples_functional_testing` for workflows
- **Build testing** - Verify compilation with relevant compilers

### Common Pitfalls to Avoid

1. **Wrong target repository** - Always use `--repo upstream`, never create PRs against `origin`
2. **Incomplete template** - Every section of the template must be addressed
3. **Missing motivation** - Explain *why* changes were made, not just *what* changed
4. **Unchecked checklists** - Mark items as complete only when actually done
5. **Vague testing descriptions** - Be specific about what was tested and how

## Workflow File Organization

This directory (`.github/workflows/`) contains CI/CD workflow definitions:

- `run_builds.yml` - Main build and test workflow
- Other workflows as they are added

When modifying workflows, test them thoroughly as they affect all pull requests and merges.

## Emergency PR Procedures

If a critical bug fix needs to bypass the normal process:

1. **Still use the template** - Even emergency PRs need proper documentation
2. **Mark as urgent** - Add `[URGENT]` prefix to title
3. **Explain urgency** - In description, explain why normal review timeline cannot be followed
4. **Extra testing** - Emergency PRs require *more* testing, not less

## Example: Complete PR Creation Session

```
User: "Let's create a PR for this branch"

Claude: I'll create a PR following the template. Let me draft the description first.

[Claude drafts PR according to template and saves to .claude/cache/pr_draft.md]

Claude: I've drafted the following pull request description (saved in .claude/cache/pr_draft.md):

# Description

This PR removes leaked template matching queue code from master and improves
CUDA architecture targeting flexibility.

**Why these changes:**
- TMQ button leaked from experimental branch, causing UI confusion
- Different developers need different GPU architectures (sm_80, sm_86, sm_89)
...

[Full PR draft displayed]

Please select an option:
1. ✅ Accept - Create the PR with this description
2. ✏️ Edit - Make changes to the draft
3. 🔍 Preview in file - Open .claude/cache/pr_draft.md to review

User: 2 (Edit)

Claude: Entering planning mode for PR edits. Which line or section would you like to edit?

User: Line 5, change "UI confusion" to "user confusion and references non-existent functionality"

Claude: Added to TODO list:
- [ ] Line 5: Change "UI confusion" to "user confusion and references non-existent functionality"

Would you like to:
- Select another line/section to edit
- Finalize and apply all edits

User: Finalize

Claude: Applying edits and saving to .claude/cache/pr_draft.md...

[Claude shows updated draft with same three options]

User: 1 (Accept)

Claude: Creating pull request...
[Executes: gh pr create --title "..." --body "$(cat .claude/cache/pr_draft.md)" --base master --repo upstream]

PR created successfully at: https://github.com/StochasticAnalytics/cisTEMx/pull/XXX
```
