---
name: security_review
description: Red/blue team security analysis framework for vulnerability assessment and defensive hardening
---

# Security Review

Framework for adversarial security analysis (red team) and defensive hardening (blue team).

## Purpose

Systematic security evaluation using:
- **Red team perspective**: Identify vulnerabilities, attack surfaces, exploit chains
- **Blue team perspective**: Design mitigations, hardening strategies, detection

## When to Use

- Code security review (C++, Python, network protocols)
- Architecture threat modeling
- System hardening assessment
- Penetration testing preparation
- Defense-in-depth validation

## For Red Team (Attack/Vulnerability Focus)

Load `resources/red_perspective.md` for:
- **Attack surface enumeration**: Entry points, trust boundaries, untrusted input
- **Vulnerability pattern matching**: OWASP Top 10, CWE categories, common exploits
- **Exploit chain construction**: How vulnerabilities chain together
- **Severity assessment**: CVSS scoring, exploitability analysis

### Key Questions Red Asks

- What can an attacker control?
- Where are trust boundaries crossed?
- What input validation is missing?
- What security assumptions are risky?
- How could this fail catastrophically?

## For Blue Team (Defense/Mitigation Focus)

Load `resources/blue_perspective.md` for:
- **Mitigation strategies**: Input validation, output encoding, least privilege
- **Defense-in-depth layers**: Multiple independent security controls
- **Detection and monitoring**: Logging, alerting, anomaly detection
- **Hardening recommendations**: Compiler flags, sandboxing, isolation

### Key Questions Blue Asks

- What defenses are already present?
- How can we make exploitation harder?
- What additional layers would help?
- How do we detect attacks?
- What are the quick wins?

## Shared Reference Material

Load `resources/vulnerability_database.md` for:
- Common vulnerabilities (buffer overflows, injection attacks, etc.)
- CVE examples with exploit details
- CWE category mappings
- Platform-specific security considerations (C++, CUDA, wxWidgets, networking)

## Output Format

### Red Team Output

```markdown
# Security Analysis - Critical Findings

## Attack Surface

[Entry points, trust boundaries, untrusted input paths]

## Vulnerabilities Found

### CRITICAL: [Vulnerability Name]
- **Location**: [file:line]
- **Type**: [CWE-XXX: Description]
- **Attack vector**: [How to exploit]
- **Impact**: [What attacker gains]
- **Evidence**: [Code snippet or proof]

[Repeat for each finding with CRITICAL/MAJOR/MINOR severity]

## Exploit Chains

[How vulnerabilities can be chained together for greater impact]
```

### Blue Team Output

```markdown
# Security Analysis - Defensive Recommendations

## Current Defenses

[What's already working well]

## Recommended Mitigations

### Priority 1 (Immediate)
- **For [Vulnerability]**: [Specific mitigation]
- **Implementation**: [Code/config change]
- **Verification**: [How to test it works]

[Repeat with Priority 2, 3...]

## Defense-in-Depth Layers

[Additional security controls beyond fixing specific vulnerabilities]

## Detection & Monitoring

[How to detect attacks and respond]
```

## Progressive Disclosure

**Level 1** (this file): Framework and approach
**Level 2**: Load red_perspective.md or blue_perspective.md based on your role
**Level 3**: Load vulnerability_database.md for specific vulnerability details

## Version

1.0 - Initial unified security review framework
