# Test Suite for Claude Code Skills

## Overview

This directory contains tests for shared utilities and scripts used across Claude Code skills. The test infrastructure was set up on November 3, 2025, with placeholder tests. Full test implementation is a future task as part of developing the testing skill.

## Running Tests

### Prerequisites
```bash
pip install pytest
```

### Run all tests
```bash
# From project root
pytest .claude/scripts/tests/

# Or with verbose output
pytest .claude/scripts/tests/ -v

# Run specific test file
pytest .claude/scripts/tests/test_path_validation.py -v
```

### Run placeholder tests only
```bash
pytest .claude/scripts/tests/ -k "placeholder"
```

## Test Coverage Goals

### test_path_validation.py
**Purpose**: Validate security-focused path utilities

**Test areas**:
- Path traversal detection (positive/negative cases)
- Within-project boundary validation
- Symlink handling
- Gitignore pattern matching
- Build directory comprehensive validation

**Priority**: High - these utilities prevent security issues

### test_build_paths.py
**Purpose**: Validate build directory gitignore coverage

**Test areas**:
- Extracting build configs from tasks.json
- Detecting uncovered directories
- Suggestion generation
- --fix mode operation
- Edge cases and error handling

**Priority**: Medium - prevents git pollution

## Future Test Development

### Testing Skill Development
Creating good tests is a skill itself. When developing the testing skill, consider:

1. **Test structure patterns**
   - Arrange-Act-Assert
   - Given-When-Then
   - Fixture usage

2. **Temporary file handling**
   - Using pytest `tmp_path` fixture
   - Creating realistic test scenarios
   - Cleanup strategies

3. **Mocking and isolation**
   - Mock subprocess calls (git, etc.)
   - Mock file system operations
   - Isolate units under test

4. **Edge case identification**
   - Boundary conditions
   - Error paths
   - Platform-specific issues (Windows vs Unix)

5. **Test data management**
   - Sample tasks.json files
   - Sample .gitignore patterns
   - Known good/bad inputs

### Integration with CI/CD
Once tests are implemented:
- Add to pre-commit hooks
- Run in GitHub Actions on PR
- Generate coverage reports
- Enforce minimum coverage thresholds

## Current Status

**Implemented**: ✓ Test infrastructure (directories, placeholders)
**TODO**: ✗ Actual test implementations
**TODO**: ✗ Testing skill development
**TODO**: ✗ CI/CD integration

## Notes for Future Development

### Why Placeholders?
Test-driven development (TDD) is valuable, but creating good tests requires its own skill development. By setting up the infrastructure now:
- Structure signals intent
- Makes it easy to add tests incrementally
- Documents what should be tested
- Provides examples for the testing skill to reference

### Test-First vs Test-After
For this project, we're taking a pragmatic approach:
- Security-critical code (path validation) should have tests ASAP
- Utility scripts can be tested as the testing skill develops
- Placeholders provide a roadmap for what needs coverage

### Reference Resources
When implementing tests, consult:
- pytest documentation: https://docs.pytest.org/
- Python testing best practices
- Lab notebook entries on testing patterns
- Weekly syntheses identifying test needs

## Lab Notebook Reference

See lab notebook entries for:
- Decision to create test infrastructure
- Rationale for placeholder approach
- Patterns discovered during test development
- Lessons learned from writing tests

---

*Test infrastructure created: 2025-11-03*
*Full implementation: TODO as part of testing skill development*
