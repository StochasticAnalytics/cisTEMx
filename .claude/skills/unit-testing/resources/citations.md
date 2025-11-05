# Citations and Sources

Documentation of all external sources consulted during skill creation. This enables future maintenance and currency checks.

## Catch2 Framework

### Official Documentation
**Source**: GitHub - catchorg/Catch2
**URL**: https://github.com/catchorg/Catch2
**Accessed**: 2025-01-03
**Relevant Learnings**: Catch2 v3 architecture, TEST_CASE and SECTION syntax, assertion macros, tags for test filtering
**Linked Content**: `resources/cpp_catch2.md` - Framework overview, best practices, test structure patterns

### Catch2 v3 Migration Guide
**Source**: GitLab blog - "Develop C++ unit testing with Catch2, JUnit, and GitLab CI"
**Accessed**: 2024
**Relevant Learnings**: Catch2 v3 is no longer single-header, requires separate compilation, CMake integration patterns
**Linked Content**: `resources/cpp_catch2.md` - Version-specific guidance

### CMake Integration
**Source**: PHIPS BLOG - "CMake-Project with Catch2 v3 Unit Tests"
**Accessed**: 2024
**Relevant Learnings**: Modern Catch2 v3 build system integration, linking against Catch2::Catch2WithMain
**Linked Content**: `resources/build_integration.md` - Build system patterns

## pytest Framework

### Best Practices - Fixtures
**Source**: Pytest with Eric - "5 Best Practices For Organizing Tests"
**Accessed**: 2024
**Relevant Learnings**: Fixture organization in fixtures/ vs conftest.py, fixture scope selection, dependency injection patterns
**Linked Content**: `resources/python_pytest.md` - Fixture organization section

### Best Practices - Mocking
**Source**: Pytest with Eric - "Common Mocking Problems & How To Avoid Them"
**Accessed**: 2024
**Relevant Learnings**: Group mocks in fixtures, use spec for safety, avoid over-mocking, mimic real behavior
**Linked Content**: `resources/python_pytest.md` - Mocking best practices section

### Advanced Testing Techniques
**Source**: Expert Guide 2025 - "Advanced Integration Testing Techniques for Python Developers"
**Accessed**: 2025
**Relevant Learnings**: Test organization strategies, AAA pattern in Python, marker usage
**Linked Content**: `resources/python_pytest.md` - Test organization patterns

### Fixtures Deep Dive
**Source**: BinaryScripts - "Advanced Unit Testing in Python with Pytest Fixtures"
**Accessed**: 2024
**Relevant Learnings**: Fixture scopes (function/class/module/session), autouse fixtures, parameterized fixtures, fixture factories
**Linked Content**: `resources/python_pytest.md` - Fixtures section

## bats Framework

### Official Repository
**Source**: GitHub - bats-core/bats-core
**URL**: https://github.com/bats-core/bats-core
**Accessed**: 2025-01-03
**Relevant Learnings**: TAP-compliant testing, run helper, setup/teardown patterns, test isolation strategies
**Linked Content**: `resources/bash_bats.md` - Framework overview and patterns

### Practical Guide
**Source**: Code-Sage - "Automated Testing for BASH scripts with BATS"
**Accessed**: October 2024
**Relevant Learnings**: Real-world bats patterns, CI/CD integration, best practices for shell script testing
**Linked Content**: `resources/bash_bats.md` - Common patterns and CI integration

### Testing Patterns
**Source**: HackerOne/PullRequest - "Testing Bash Scripts with BATS: A Practical Guide"
**Accessed**: 2024
**Relevant Learnings**: Assertion patterns, file operation testing, environment isolation
**Linked Content**: `resources/bash_bats.md` - Testing patterns section

## CUDA Testing

### Host-Device Functions
**Source**: NVIDIA Developer Forums - "CUDA C++ unit testing in host and device code"
**Accessed**: 2024
**Relevant Learnings**: `__host__ __device__` strategy for testable GPU code, CPU+GPU testing patterns
**Linked Content**: `resources/cuda_testing.md` - Core strategy section

### Programming Patterns
**Source**: ACM - "Patterns for __host__ __device__ programming in Cuda"
**Accessed**: 2024
**Relevant Learnings**: Design patterns for dual CPU/GPU code, benefits for testing and debugging
**Linked Content**: `resources/cuda_testing.md` - Code decomposition patterns

### Testable Design
**Source**: Netherlands eScience Center blog - "Writing Testable GPU Code"
**Accessed**: 2024
**Relevant Learnings**: Code decomposition for testability, CPU reference implementations, property-based testing for GPU code
**Linked Content**: `resources/cuda_testing.md` - Strategy sections and design benefits

### Testing Tools
**Source**: GitHub - havogt/cuda_gtest_plugin
**Accessed**: 2024
**Relevant Learnings**: GoogleTest extension for CUDA, testing small device function fragments
**Linked Content**: `resources/cuda_testing.md` - Strategy 5 (small fragments)

## Test-Driven Development

### TDD Guide 2024
**Source**: Medium - "Mastering Test-Driven Development (TDD): The Ultimate Python Developer's Guide"
**Accessed**: December 2024
**Relevant Learnings**: TDD workflow (Red-Green-Refactor), AAA pattern, benefits backed by research, modern TDD practices
**Linked Content**: `resources/tdd_fundamentals.md` - TDD workflow and benefits sections

### TDD Relevance Study
**Source**: ScrumLaunch - "Test-Driven Development: Is TDD Relevant in 2024?"
**Accessed**: 2024
**Relevant Learnings**: Evidence-based TDD benefits (40-80% fewer bugs, 60% better coverage), cost-benefit analysis, modern applicability
**Linked Content**: `resources/tdd_fundamentals.md` - TDD benefits section

### Testing Types Comparison
**Source**: Software Testing Help - "Unit Testing Vs Integration Testing Vs Functional Testing"
**Accessed**: 2024
**Relevant Learnings**: Clear distinctions between test types, when to use each, scope boundaries
**Linked Content**: `SKILL.md` - Critical scope distinction section

## Testing Frameworks Comparison

### Catch2 vs GoogleTest
**Source**: Stack Overflow and Medium discussions
**Accessed**: 2024
**Relevant Learnings**: Catch2 advantages (descriptive test names, less boilerplate, SECTION blocks), comparison with GoogleTest patterns
**Linked Content**: `resources/cpp_catch2.md` - Framework overview

## Internal cisTEMx Sources

### unit-test-architect Agent
**Source**: `.claude/agents/unit-test-architect.md`
**Accessed**: 2025-01-03
**Relevant Learnings**:
- Testing philosophy (non-trivial tests, risk-based selection, property-based thinking)
- cisTEMx-specific Catch2 patterns and project structure
- GPU testing strategies (gating, skip messages, CPU fallbacks)
- Data realism requirements
- Reproducibility and CI-friendliness principles
- Negative testing requirements
- Quality gates checklist
**Linked Content**: `resources/cpp_catch2.md` - Extracted and generalized into comprehensive C++ testing resource

### Existing Test Files
**Source**: `src/test/core/test_matrix.cpp`
**Accessed**: 2025-01-03
**Relevant Learnings**: Excellent helper function patterns (MatricesAreAlmostEqual), SECTION usage, comprehensive coverage examples
**Linked Content**: `resources/cpp_catch2.md` - Helper functions and examples sections

**Source**: `src/test/core/test_job_packager.cpp`
**Accessed**: 2025-01-03
**Relevant Learnings**: File-level documentation patterns, type safety testing, property-based thinking in practice
**Linked Content**: `resources/cpp_catch2.md` - Documentation and data realism sections

**Source**: `src/test/gpu/test_pointers.cpp`
**Accessed**: 2025-01-03
**Relevant Learnings**: GPU test gating patterns, device memory testing, cisTEMx DevicePointerArray usage
**Linked Content**: `resources/cpp_catch2.md`, `resources/cuda_testing.md` - GPU testing sections

### Build System
**Source**: `src/Makefile.am`
**Accessed**: 2025-01-03
**Relevant Learnings**: Autotools test integration, adding test sources, build configuration
**Linked Content**: `resources/build_integration.md` - Autotools section

### Build Documentation
**Source**: Scripts CLAUDE.md
**Accessed**: 2025-01-03
**Relevant Learnings**: cisTEMx build system patterns, configure flags, compilation process
**Linked Content**: `resources/build_integration.md` - cisTEMx-specific guidance

## Research Process Notes

### Search Strategy
- Prioritized official documentation and framework repositories
- Consulted recent (2024-2025) articles for current best practices
- Cross-referenced multiple sources for validation
- Emphasized evidence-based claims (research studies, industry data)
- Checked community-trusted sources (high-voted Stack Overflow, official blogs)

### Quality Criteria
- Recent publication dates (prefer 2024-2025)
- Official sources when available
- Community consensus on best practices
- Evidence backing claims (not just opinion)
- Practical, actionable guidance

## Maintenance Notes

### Future Updates Needed When:

**Catch2:**
- Major version updates (currently v3)
- Breaking API changes
- New testing features

**pytest:**
- Major version updates
- New fixture patterns
- Plugin ecosystem changes

**bats:**
- Core functionality changes
- New assertion helpers

**CUDA:**
- New compute capabilities
- Changed device limits
- New testing tools

**TDD Research:**
- New studies on effectiveness
- Updated industry data
- Evolving best practices

### Recommended Review Schedule
- **Quarterly**: Check for major framework updates
- **Annually**: Review research citations for currency
- **As needed**: When encountering framework-specific issues

### Update Process
1. Review official framework changelogs
2. Check for breaking changes in patterns documented
3. Update resource files with new features/patterns
4. Add new citations with access dates
5. Archive or mark outdated patterns
6. Test examples still work with current versions

## Version Information (Skill Creation)

**Skill Created**: 2025-01-03
**Primary Author**: Claude (senior graduate student, Committee Chair: Anwar)
**Knowledge Base Cutoff**: January 2025
**Research Period**: November 2024 - January 2025

## Verification

All sources were:
- Cross-referenced against multiple sources
- Validated for technical accuracy
- Checked for recency and relevance
- Attributed clearly in linked content
- Selected for trustworthiness and authority

## Contact for Updates

If you discover outdated information:
1. Check framework official documentation first
2. Verify the change is not cisTEMx-specific
3. Update the relevant resource file
4. Add new citation with access date
5. Note the change in git commit message

## Related Skills

This skill may depend on:
- `compile-code` - For building and running C++ tests
- `git-commit` - For committing test changes
- `lab-notebook` - For documenting testing insights

This skill provides foundation for:
- Future integration testing skill
- Future functional testing skill
- Code quality and maintainability practices
