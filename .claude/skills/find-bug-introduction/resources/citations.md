# Citations and Sources

All sources for find-bug-introduction skill. Organized by category for maintainability and future review.

## Official Git Documentation

**Primary Reference**:
- Git Bisect Manual: https://git-scm.com/docs/git-bisect (accessed 2025-11-03)
  - Complete command reference
  - Exit code conventions
  - All bisect options documented

**Historical Context**:
- Git bisect-lk2009: https://git-scm.com/docs/git-bisect-lk2009 (accessed 2025-11-03)
  - Original "Fighting regressions with git bisect" article
  - Historical perspective on bisect development

**Related Documentation**:
- Git Diff Core: https://git-scm.com/docs/gitdiffcore (accessed 2025-11-03)
  - Understanding how git compares commits
  - Relevant for understanding bisect internals

## Linux Kernel Development

**Practical Usage**:
- Kernel Bug Bisection Guide: https://docs.kernel.org/admin-guide/bug-bisect.html (accessed 2025-11-03)
  - Real-world bisect practices from Linux kernel team
  - Production-grade workflows
  - Ingo Molnar's automated bisecting referenced

**Context**: Linux kernel relies heavily on bisect for stability. Thousands of developers, millions of lines of code, bisect is essential tool.

## Expert Articles & Tutorials

**Automation Techniques**:
- LWN.net. "Fully automated bisecting with git bisect run." https://lwn.net/Articles/317154/ (accessed 2025-11-03)
  - Comprehensive guide to automation
  - Exit code conventions explained
  - Real-world examples

- Duffield, Jesse. "Git Bisect Run: Bug Hunting On Steroids." https://jesseduffield.com/Bisect/ (accessed 2025-11-03)
  - Practical examples from Lazygit development
  - ~100 manual checks → ~7 automated tests case study
  - Modern developer perspective (2023)

**Advanced Techniques**:
- Hoelz, Rob. "git-pisect - A Parallel Version of git-bisect." https://hoelz.ro/blog/git-pisect (accessed 2025-11-03)
  - Parallel bisecting implementation
  - Performance analysis (2.7x speedup)
  - Trade-offs documented

**Community Best Practices**:
- Atlassian Community. "Advanced Git bisect and blame techniques for tracking down bugs and regressions in your codebase." https://community.atlassian.com/forums/Bitbucket-articles/Advanced-Git-bisect-and-blame-techniques-for-tracking-down-bugs/ba-p/2387194 (accessed 2025-11-03)
  - Practical workflows
  - Integration with other git tools

## Research and Case Studies

**Performance Impact**:
- Andreas Ericsson's team study:
  - 88.6% reduction in bug resolution time
  - 142.6 hours → 16.2 hours average
  - Production environment measurements
  - Referenced in multiple git bisect discussions

**Theoretical Analysis**:
- Binary search efficiency: O(log₂ n)
  - Computer Science fundamentals
  - Applied to version control
  - Practical implications for large repositories

## Tools and Implementations

**git-pisect**:
- Repository: https://github.com/hoelzro/git-pisect (accessed 2025-11-03)
- Language: Perl
- License: Open source
- Status: Actively maintained

**CI/CD Integration Examples**:
- GitHub Actions examples from git community
- GitLab CI patterns for automated bisecting
- Industry standard practices

## Technical Details

**Exit Code Standards**:
- POSIX shell conventions
  - Exit 0: Success
  - Exit 1-255: Various failures
  - Exit 125: Specifically chosen for "skip" to avoid conflicts
  - Exit 126-127: Shell reserved
  - Exit 128-255: Signals

**Git Internals**:
- Binary search algorithm implementation
- Skip selection algorithm (PRNG with 1.5 power bias)
- Reference handling (BISECT_HEAD, bisect refs)

## Community Resources

**Stack Overflow**:
- Multiple highly-voted questions on git bisect
- Real-world problem-solving patterns
- Common pitfalls documented

**GitHub Discussions**:
- Best practices from open source maintainers
- Integration with various CI systems
- Automation scripts shared

## Verification Notes

**All URLs accessed**: November 3, 2025

**Content Stability**:
- Official git documentation: Stable, versioned
- Linux kernel docs: Stable, maintained
- Expert articles: Content snapshot recommended
- Tools: Version-specific documentation

**Future Review**:
- Check for git version updates (bisect options may expand)
- Monitor git-pisect for new features
- Review case studies for updated metrics
- Check for new automation patterns in CI/CD systems

## Related Skills

These skills may reference or be referenced by find-bug-introduction:

- **understand-code-origins**: Uses git blame, pickaxe after bisect identifies culprit
- **git-version-control**: General git workflow context
- **compile-code**: Build system integration for bisect test scripts

## Maintenance Schedule

**Quarterly Review**:
- Check official git documentation for updates
- Review tool status (git-pisect, etc.)
- Update performance benchmarks if new data available

**Annual Review**:
- Re-verify all external links
- Update case studies with new examples
- Check for new best practices in industry

## License Information

**Git Documentation**: GPLv2
**LWN Articles**: Copyright LWN.net, linked with permission via standard web citation
**Individual Blogs**: Cited under fair use for educational purposes
**Code Examples**: Public domain where not otherwise specified

---

**Last Updated**: November 3, 2025
**Maintainer**: Claude (cisTEMx project)
**Next Review Due**: February 3, 2026
