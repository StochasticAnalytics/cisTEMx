# Refactoring Target Assessment Workflow

## Purpose

This template provides a systematic workflow for identifying and prioritizing refactoring targets using data-driven analysis.

---

## Phase 1: Initial Analysis (30-60 minutes)

### 1.1 Quick Churn Check

**Goal**: Identify files that change most frequently

```bash
cd /path/to/repository

# Top 20 most-changed files
git log --format=format: --name-only --since=12.month \
  | egrep -v '^$' \
  | sort | uniq -c | sort -nr | head -20
```

**Record results**:
- [ ] Identified top 20 files by churn
- [ ] Noted any surprising entries
- [ ] Saved list for reference

### 1.2 Complexity Snapshot

**Goal**: Assess complexity of high-churn files

```bash
# Install if needed: pip install lizard
lizard --csv src/ > complexity.csv
```

**Record results**:
- [ ] Generated complexity data
- [ ] Identified files with CCN >20
- [ ] Noted files >500 lines

### 1.3 Initial Hotspot Identification

**Goal**: Find top 5-10 hotspots (high churn + high complexity)

**Manual calculation or use script**:
```bash
./scripts/identify_hotspots.sh /path/to/repo 12.month.ago
```

**Record results**:
- [ ] Top 5 hotspots identified
- [ ] Hotspot scores calculated
- [ ] Files prioritized by score

---

## Phase 2: Deep Dive Investigation (2-4 hours)

### 2.1 For Each Hotspot File

**File**: _____________________________
**Score**: _____ (CCN × commits)
**CCN**: _____
**Commits**: _____
**LOC**: _____

#### 2.1.1 Change Pattern Analysis

```bash
# View commit history
git log --oneline --follow -- <file>

# Analyze commit messages
git log --pretty=format:'%s' --follow -- <file> \
  | sort | uniq -c | sort -nr | head -10
```

**Patterns identified**:
- [ ] Repeated bug fixes (count: ___)
- [ ] Feature additions (count: ___)
- [ ] Configuration tweaks (count: ___)
- [ ] Other: _______________

**Root cause hypothesis**:
________________________________________________________________
________________________________________________________________

#### 2.1.2 Contributor Analysis

```bash
git shortlog -sn --follow -- <file>
```

**Contributors**:
- Primary owner: _______________ (commits: ___)
- Secondary contributors: ___ people
- Coordination complexity: [ ] Low  [ ] Medium  [ ] High

#### 2.1.3 Technical Debt Markers

```bash
git grep -n "TODO\|FIXME\|HACK" -- <file>
```

**SATD count**:
- TODO: ___
- FIXME: ___
- HACK: ___

**Key debt items**:
1. ________________________________________________________________
2. ________________________________________________________________
3. ________________________________________________________________

#### 2.1.4 Function-Level Complexity

```bash
lizard --verbose <file>
```

**High-complexity functions** (CCN >15):
1. Function: _______________ (CCN: ___, LOC: ___)
2. Function: _______________ (CCN: ___, LOC: ___)
3. Function: _______________ (CCN: ___, LOC: ___)

**Target for refactoring**: _______________

---

## Phase 3: Coupling Analysis (1-2 hours)

### 3.1 Generate Coupling Data

```bash
# Generate git log
git log --all --numstat --date=short \
  --pretty=format:'--%h--%ad--%aN' \
  --no-renames --after=6.month.ago \
  > git.log

# Run Code Maat (if available)
java -jar code-maat.jar -l git.log -c git2 -a coupling \
  --min-coupling 40 > coupling.csv
```

### 3.2 Check Coupling for Each Hotspot

**File**: _____________________________

**Coupled files** (degree >40%):
1. _______________  (degree: ___%)
2. _______________  (degree: ___%)
3. _______________  (degree: ___%)

**Coupling type**:
- [ ] Expected (UI + controller, header + impl)
- [ ] Unexpected (cross-layer, unrelated modules)

**Architectural issue**:
- [ ] No issue
- [ ] Layer violation
- [ ] Missing abstraction
- [ ] God object pattern

**Notes**:
________________________________________________________________
________________________________________________________________

---

## Phase 4: Prioritization (30-60 minutes)

### 4.1 Scoring Matrix

| File | Hotspot Score | Business Impact | Effort (weeks) | Priority Score |
|------|---------------|-----------------|----------------|----------------|
| 1. | | [ ]Low [ ]Med [ ]High | | |
| 2. | | [ ]Low [ ]Med [ ]High | | |
| 3. | | [ ]Low [ ]Med [ ]High | | |
| 4. | | [ ]Low [ ]Med [ ]High | | |
| 5. | | [ ]Low [ ]Med [ ]High | | |

**Priority formula**: (Hotspot Score × Impact Factor) / Effort
- Impact factors: Low=1, Medium=2, High=3

### 4.2 Final Prioritization

**P0 (Critical - Do Now)**:
1. ________________________________________________________________
   - Reason: ___________________________________________________________

**P1 (High - Next Sprint)**:
1. ________________________________________________________________
2. ________________________________________________________________

**P2 (Medium - This Quarter)**:
1. ________________________________________________________________
2. ________________________________________________________________

**P3 (Low - Backlog)**:
1. ________________________________________________________________

---

## Phase 5: Refactoring Plan (1-2 hours)

### For Each P0/P1 Item

**File**: _____________________________

#### 5.1 Refactoring Strategy

**Chosen approach**:
- [ ] Extract method/class
- [ ] Simplify conditional logic
- [ ] Replace conditionals with polymorphism
- [ ] Extract interface/abstraction
- [ ] Split large file into modules
- [ ] Other: _______________

**Rationale**:
________________________________________________________________
________________________________________________________________

#### 5.2 Risk Assessment

**Risks**:
- [ ] No test coverage
- [ ] Many dependencies
- [ ] Active development (conflicts likely)
- [ ] Unclear requirements
- [ ] Team lacks domain knowledge

**Mitigation**:
________________________________________________________________
________________________________________________________________

#### 5.3 Test Strategy

**Current test coverage**: ____%

**Testing plan**:
- [ ] Add unit tests before refactoring
- [ ] Write characterization tests
- [ ] Manual testing checklist
- [ ] Integration test updates needed

#### 5.4 Effort Estimate

**Developer time**: ___ days/weeks
**Review time**: ___ days
**Testing time**: ___ days
**Total**: ___ weeks

**Confidence**: [ ] Low  [ ] Medium  [ ] High

#### 5.5 Success Criteria

**Metrics to track**:
- [ ] CCN reduction target: ___ → ___ (___% reduction)
- [ ] LOC reduction target: ___ → ___ (___% reduction)
- [ ] Churn reduction (measure 3 months after): ___% expected
- [ ] Defect reduction: ___ fewer bugs
- [ ] Velocity improvement: ___% faster features

---

## Phase 6: Tracking & Measurement (Ongoing)

### 6.1 Before Refactoring Baseline

**Date**: _______________
**Version/Tag**: _______________

**Metrics captured**:
- [ ] Complexity (CCN, LOC)
- [ ] Churn (commits in past 3 months)
- [ ] Defect count (past 3 months)
- [ ] SATD count
- [ ] Test coverage

### 6.2 After Refactoring Measurement

**Date**: _______________
**Version/Tag**: _______________
**Time since refactoring**: ___ months

**Metrics captured**:
- [ ] Complexity (CCN, LOC)
- [ ] Churn (commits since refactoring)
- [ ] Defect count (since refactoring)
- [ ] SATD count
- [ ] Test coverage

### 6.3 Results Analysis

**CCN reduction**: ___% (target: 40-60%)
**Churn reduction**: ___% (target: 30-50%)
**Defect reduction**: ___%
**Velocity impact**: ___% (faster/slower)

**Lessons learned**:
________________________________________________________________
________________________________________________________________
________________________________________________________________

**Recommendation**:
- [ ] Success - replicate approach
- [ ] Partial success - adjust strategy
- [ ] Failure - investigate root cause

---

## Phase 7: Documentation & Communication

### 7.1 Stakeholder Report

**Executive summary** (2-3 sentences):
________________________________________________________________
________________________________________________________________
________________________________________________________________

**Key findings**:
1. ________________________________________________________________
2. ________________________________________________________________
3. ________________________________________________________________

**Top priorities**:
1. ________________________________________________________________ (ROI: High)
2. ________________________________________________________________ (ROI: Medium)
3. ________________________________________________________________ (ROI: Medium)

**Resource request**:
- Developer time: ___ weeks
- Budget: $_____
- Timeline: ___ weeks/months

### 7.2 Team Communication

**Announcement**:
- [ ] Shared analysis results with team
- [ ] Discussed findings in team meeting
- [ ] Created refactoring tickets/issues
- [ ] Assigned owners to P0/P1 items

**Documentation**:
- [ ] Added to wiki/docs
- [ ] Linked to relevant issues
- [ ] Created decision records (ADRs)

---

## Checklist Summary

**Analysis Phase**:
- [ ] Churn analysis completed
- [ ] Complexity analysis completed
- [ ] Hotspots identified (top 5-10)
- [ ] Coupling analysis completed

**Investigation Phase**:
- [ ] Top hotspots investigated
- [ ] Change patterns identified
- [ ] Technical debt documented
- [ ] Function-level targets identified

**Planning Phase**:
- [ ] Priorities assigned (P0-P3)
- [ ] Refactoring strategies defined
- [ ] Effort estimates completed
- [ ] Risk mitigation planned

**Execution Phase**:
- [ ] Baseline metrics captured
- [ ] Refactoring executed
- [ ] Post-refactoring metrics captured
- [ ] Results analyzed

**Communication Phase**:
- [ ] Stakeholder report created
- [ ] Team notified
- [ ] Documentation updated

---

## References

- **Detailed workflows**: `resources/practical_workflow.md`
- **Churn analysis**: `resources/churn_analysis.md`
- **Hotspot analysis**: `resources/hotspot_analysis.md`
- **Coupling analysis**: `resources/temporal_coupling.md`
- **Code Maat usage**: `resources/code_maat_guide.md`
- **Automation scripts**: `scripts/analyze_churn.sh`, `scripts/identify_hotspots.sh`
