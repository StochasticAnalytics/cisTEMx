# Citations and References

## External Sources

### GNU Autotools Linker Errors
**Sources:**
- https://stackoverflow.com/questions/12573816/what-is-an-undefined-reference-unresolved-external-symbol-error-and-how-do-i-fix
- https://stackoverflow.com/questions/57845812/autotools-build-fails-while-linking
- https://lists.gnu.org/archive/html/automake/2014-10/msg00014.html

**Accessed:** 2025-11-03

**Relevant Learnings:**
- LDADD vs LDFLAGS distinction in autotools
- Library order matters for single-pass linkers
- --as-needed linker flag behavior

**Linked Content:** `linker_errors_reference.md` sections on autotools gotchas and library order

### Clang Linker Issues
**Sources:**
- https://stackoverflow.com/questions/31397609/linker-error-with-clang-for-some-standard-library-classes
- https://stackoverflow.com/questions/18459894/clang-stdlib-libc-leads-to-undefined-reference

**Accessed:** 2025-11-03

**Relevant Learnings:**
- Standard library mixing (libstdc++ vs libc++)
- Clang-specific namespace mangling (std::__1::)
- -stdlib flag consistency requirements

**Linked Content:** `linker_errors_reference.md` section on standard library mismatch

### Build Script Design
**Source:** Extracted from cpp-build-expert.md agent definition (internal)

**Accessed:** 2025-11-03

**Relevant Learnings:**
- VS Code tasks.json structure for build configuration extraction
- Parallelism capping at 16 cores for stability
- Timestamped logging pattern

**Linked Content:** `run_build.py` script design and `build_execution.md`

### Error Analysis Patterns
**Source:** Extracted from cpp-build-expert.md agent definition (internal)

**Accessed:** 2025-11-03

**Relevant Learnings:**
- Case-sensitive "error" detection for compilation errors
- Case-insensitive "undefined reference" for linker errors
- Error cascading and root cause prioritization

**Linked Content:** `analyze_build_log.py` script logic and `error_diagnosis_basics.md`

## Platform Dependencies

### VS Code Tasks
**Component:** `.vscode/tasks.json`
**Version:** VS Code task schema 2.0.0
**Last Verified:** 2025-11-03
**Notes:** Build directory extraction depends on consistent task naming pattern: "BUILD cisTEMx <CONFIG>"

### GNU Make
**Component:** `make` with `-j` flag
**Version:** GNU Make (assumed recent)
**Last Verified:** 2025-11-03
**Notes:** Parallel build support required

### Python
**Component:** Python 3 for build scripts
**Version:** Python 3.6+ (uses f-strings, pathlib)
**Last Verified:** 2025-11-03
**Notes:** Scripts use subprocess, json, re, datetime, pathlib - all stdlib

## Internal Patterns

### Progressive Disclosure
Applied from `skill-builder` skill methodology:
- Concise SKILL.md entry point
- Detailed resources on-demand
- Script-based automation

### Workflow Skill Pattern
Template from `skill-builder/templates/workflow_skill_template.md`:
- Quick start section
- Resource directory
- Validation steps

## Future Maintenance Notes

### When C++20 Migration Happens
- Add template error diagnosis with concepts
- Update error message patterns
- Consider separate resource for concepts-related errors

### When Build System Changes
- Update VS Code tasks.json extraction logic
- Verify build directory patterns
- Test with new configuration names

### Periodically Review
- Check Stack Overflow links for updated solutions (every 6-12 months)
- Verify GNU autotools best practices haven't changed
- Test with newer clang/gcc versions
