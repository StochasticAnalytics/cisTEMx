# Claude Skills

## Overview

Skills are specialized capabilities that enable Claude to perform focused tasks within the cisTEMx project. Each skill packages domain knowledge, tool permissions, and methodologies for specific development workflows.

This page catalogs all available skills, organized by functional purpose. Click on any category to expand and view the skills.

---

??? example "Code Analysis & Understanding"

    <h2 style="color: #2E7D32; font-weight: bold;">identify-refactoring-targets</h2>

    === "Overview"
        **Description:** Identify code needing refactoring through churn analysis, hotspot detection, and temporal coupling. Use when prioritizing technical debt, planning refactoring sprints, or investigating maintenance burden. Combines git history with complexity metrics to find high-ROI refactoring targets using data-driven approach.

    === "Details"
        --8<-- "developer/skills/identify-refactoring-targets.md"

    ---

    <h2 style="color: #2E7D32; font-weight: bold;">understand-code-origins</h2>

    === "Overview"
        **Description:** Trace code back to its origins to understand why it exists and what problem it solved. Use when investigating legacy code, understanding design decisions, or finding original context lost through refactorings. Combines git blame, pickaxe, and commit mining.

    === "Details"
        --8<-- "developer/skills/understand-code-origins.md"

---

??? example "Testing & Quality"

    <h2 style="color: #2E7D32; font-weight: bold;">unit-testing</h2>

    === "Overview"
        **Description:** Write unit tests for individual functions/methods in isolation across C++/Catch2, Python/pytest, Bash/bats, and CUDA. Use when testing single components, implementing TDD, or adding regression tests. For Claude's direct use. Covers testing frameworks, best practices, and code organization. NOT for integration/functional tests.

    === "Details"
        --8<-- "developer/skills/unit-testing.md"

    ---

    <h2 style="color: #2E7D32; font-weight: bold;">analyze-test-coverage-gaps</h2>

    === "Overview"
        **Description:** Identify test coverage gaps by analyzing git history, test-to-production ratios, untested changes, and files that break repeatedly. Use when assessing test debt, planning test improvements, or investigating quality issues. Combines git analysis with coverage tools (diff-cover, gcov/lcov, coverage.py) to find high-risk untested code.

    === "Details"
        --8<-- "developer/skills/analyze-test-coverage-gaps.md"

---

??? example "Build & Compilation"

    <h2 style="color: #2E7D32; font-weight: bold;">compile-code</h2>

    === "Overview"
        **Description:** Build/compile the cisTEMx project using cmake, make, ninja. Provides build instructions, configuration, and compilation error diagnosis. Use when you need to build the project, compile code changes, or check for build errors.

    === "Details"
        --8<-- "developer/skills/compile-code.md"

---

??? example "Technical Review Frameworks"

    <h2 style="color: #2E7D32; font-weight: bold;">plan_review</h2>

    === "Overview"
        **Description:** Red/blue team plan analysis - critical risk assessment and constructive opportunity identification.

    === "Details"
        --8<-- "developer/skills/plan_review.md"

    ---

    <h2 style="color: #2E7D32; font-weight: bold;">security_review</h2>

    === "Overview"
        **Description:** Red/blue team security analysis framework for vulnerability assessment and defensive hardening.

    === "Details"
        --8<-- "developer/skills/security_review.md"

---

??? example "Technical Review Coordination"

    <h2 style="color: #2E7D32; font-weight: bold;">red-blue-tech-coordinator</h2>

    === "Overview"
        **Description:** Coordinate lab technician reviews for critical and constructive analysis. Use when you need technical discussion on code, architecture, documentation, skills, or testing. Launches Red and Blue teams in parallel, ensures they use their specialized frameworks, and manages output to avoid race conditions.

    === "Details"
        --8<-- "developer/skills/red-blue-tech-coordinator.md"

    ---

    <h2 style="color: #2E7D32; font-weight: bold;">red-blue-skill-review</h2>

    !!! warning "Sub-Agent Use Only"
        This skill is for Task agents ONLY. DO NOT load directly. Use `red-blue-tech-coordinator` to invoke red-blue reviews.

    === "Overview"
        **Description:** Collaborative red/blue team review framework for skill design and implementation. Combines critical analysis (red) with constructive improvement (blue). Task agents apply this methodology with frameworks provided by coordinators. Use when validating skill structure, frontmatter, resources, and implementation patterns.

    === "Details"
        --8<-- "developer/skills/red-blue-skill-review.md"

    ---

    <h2 style="color: #2E7D32; font-weight: bold;">red-blue-testing-review</h2>

    !!! warning "Sub-Agent Use Only"
        This skill is for Task agents ONLY. DO NOT load directly. Use `red-blue-tech-coordinator` to invoke red-blue reviews.

    === "Overview"
        **Description:** Collaborative red/blue team review framework for test quality across all test types (unit, integration, functional). Combines critical analysis of gaps and violations with constructive improvements.

    === "Details"
        --8<-- "developer/skills/red-blue-testing-review.md"

    ---

    <h2 style="color: #2E7D32; font-weight: bold;">red-blue-plan-review</h2>

    !!! warning "Sub-Agent Use Only"
        This skill is for Task agents ONLY. DO NOT load directly. Use `red-blue-tech-coordinator` to invoke red-blue reviews.

    === "Overview"
        **Description:** Collaborative red/blue team framework for plan and architecture analysis. Combines critical risk assessment with constructive opportunity identification.

    === "Details"
        --8<-- "developer/skills/red-blue-plan-review.md"

    ---

    <h2 style="color: #2E7D32; font-weight: bold;">red-blue-security-review</h2>

    !!! warning "Sub-Agent Use Only"
        This skill is for Task agents ONLY. DO NOT load directly. Use `red-blue-tech-coordinator` to invoke red-blue reviews.

    === "Overview"
        **Description:** Red/blue team security analysis framework for vulnerability assessment and defensive hardening.

    === "Details"
        --8<-- "developer/skills/red-blue-security-review.md"

---

??? example "Documentation & Knowledge Management"

    <h2 style="color: #2E7D32; font-weight: bold;">md-to-mkdocs</h2>

    === "Overview"
        **Description:** Full MkDocs documentation management for cisTEMx - add pages, manage navigation, interpret layout feedback, reorganize structure, validate integrity, and build/preview documentation. Handles integration of markdown sources including CLAUDE.md files while assisting with manual mkdocs.yml maintenance.

    === "Details"
        --8<-- "developer/skills/md-to-mkdocs.md"

    ---

    <h2 style="color: #2E7D32; font-weight: bold;">lab-notebook</h2>

    === "Overview"
        **Description:** Use for taking quick notes and recording observations while working. User feedback or corrections or encountering unexpected results are scenarios for writing a quick note. Quick notes are used to build up daily lab-notebook entries when prompted by the user and daily lab-notebook records are compiled into weekly notes on prompting. This is a frequently used skill as jotting down your thoughts as you work, live, real-time is essential.

        **Allowed Tools:** `Bash(.claude/skills/lab-notebook/scripts/take_quick_note.py:*)`, `Bash(.claude/skills/lab-notebook/scripts/list_notes.py:*)`, `Bash(.claude/skills/lab-notebook/scripts/search_notes.py:*)`, `Bash(.claude/skills/lab-notebook/scripts/note_stats.py:*)`, `Write(.claude/cache/**)`, `Read`, `Grep`, `Glob`

    === "Details"
        --8<-- "developer/skills/lab-notebook.md"

---

??? example "Skill Development & Creation"

    <h2 style="color: #2E7D32; font-weight: bold;">skill-builder</h2>

    === "Overview"
        **Description:** Systematic methodology for creating effective skills from domain knowledge. Use when designing new skills, converting documentation to skills, or teaching sub-agents skill creation patterns. Provides templates, decision frameworks, and quality criteria.

    === "Details"
        --8<-- "developer/skills/skill-builder.md"

---

## Using Skills

Skills are invoked through the Claude Code interface using the Skill tool. Each skill provides:

- **Focused expertise** in a specific domain
- **Tool permissions** necessary for the task
- **Methodologies** and best practices
- **Progressive disclosure** of information to preserve context

For detailed information about any skill, click the skill link in each skill's documentation section to view the full skill documentation.
