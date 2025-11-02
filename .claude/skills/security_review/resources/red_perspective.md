# Red Team Security Perspective

Attack-focused vulnerability analysis framework.

## Attack Surface Enumeration

### Identify Entry Points

**Network-facing**:
- Socket connections (server/client)
- HTTP endpoints
- IPC mechanisms (shared memory, pipes, message queues)
- File parsers (user-provided data)

**User input**:
- Command-line arguments
- Configuration files
- Environment variables
- GUI input fields

**Data boundaries**:
- Serialization/deserialization
- Binary protocol parsers
- Image/file format parsers
- Database queries

### Trust Boundaries

- User process → Kernel
- Network → Local process
- GUI process → Worker process
- Host → GPU device
- Container → Host system

## Common Vulnerability Patterns

### Memory Safety (C/C++)

**Buffer overflows**:
```cpp
// Vulnerable
char buf[256];
strcpy(buf, user_input);  // No bounds checking

// Attack: user_input = "A" * 300 → buffer overflow
```

**Integer overflows**:
```cpp
// Vulnerable
size_t size = user_length * sizeof(Item);
void* ptr = malloc(size);  // If user_length is huge, size wraps

// Attack: user_length = 0xFFFFFFFF / sizeof(Item) + 1
```

**Use-after-free**:
```cpp
// Vulnerable
delete ptr;
// ... later ...
ptr->method();  // Use after free

// Attack: Spray heap, reclaim memory with attacker data
```

### Injection Attacks

**SQL injection**:
```cpp
// Vulnerable
string query = "SELECT * FROM users WHERE name='" + username + "'";

// Attack: username = "' OR '1'='1"
```

**Command injection**:
```cpp
// Vulnerable
system("convert " + user_filename + " output.png");

// Attack: user_filename = "input.jpg; rm -rf /"
```

**Path traversal**:
```cpp
// Vulnerable
open("/data/" + user_path);

// Attack: user_path = "../../../etc/passwd"
```

### Race Conditions

**TOCTOU (Time-of-Check, Time-of-Use)**:
```cpp
// Vulnerable
if (access(filename, R_OK) == 0) {  // Check
    FILE* f = fopen(filename, "r");  // Use
}

// Attack: Symlink race between check and use
```

**Lock-free data structures**:
```cpp
// Vulnerable
if (shared_ptr == nullptr) {  // Thread A checks
    shared_ptr = new Data();   // Thread B also creates
}

// Attack: Double initialization, memory leak, inconsistent state
```

### GPU/CUDA Specific

**Kernel parameter validation**:
```cpp
// Vulnerable
__global__ void kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = ...;  // No bounds check
}

// Attack: Launch with size < actual threads → out-of-bounds write
```

**Device-host trust boundary**:
```cpp
// Vulnerable
cudaMemcpy(host_buffer, device_buffer, user_size, ...);

// Attack: user_size exceeds actual allocation → buffer overflow
```

## Exploit Chain Construction

### Chain Components

1. **Initial access**: How attacker gets in (injection, overflow, etc.)
2. **Privilege escalation**: Moving from user → admin/root/kernel
3. **Persistence**: Maintaining access across reboots
4. **Data exfiltration**: Getting sensitive data out
5. **Lateral movement**: Spreading to other systems

### Example Chain: Web Service Compromise

```
1. SQL injection in login endpoint
   → Dump user credentials
2. Weak password hashing (MD5)
   → Crack admin password
3. Admin panel has command injection
   → Execute arbitrary commands as service user
4. Service runs as root (bad practice)
   → Full system compromise
5. Install backdoor in system startup
   → Persistent access
```

## Severity Assessment

### CVSS Scoring

**Critical (9.0-10.0)**:
- Remote code execution
- Authentication bypass
- Complete data breach

**High (7.0-8.9)**:
- Local privilege escalation
- Partial data breach
- Denial of service

**Medium (4.0-6.9)**:
- Information disclosure
- Limited DoS
- XSS

**Low (0.1-3.9)**:
- Minor information leak
- Low-impact vulnerabilities

### Exploitability Factors

- **Attack vector**: Network (worst) → Local → Physical
- **Attack complexity**: Low (worst) → High
- **Privileges required**: None (worst) → Admin
- **User interaction**: None (worst) → Required

## Red Team Mindset

### Questions to Ask

1. **Trust**: What does this code trust that it shouldn't?
2. **Validation**: What input validation is missing?
3. **Assumptions**: What assumptions can I violate?
4. **Boundaries**: Where do privilege levels change?
5. **Failure**: How does this behave when things go wrong?

### Common Developer Mistakes

- "Users won't send malformed input" → They will
- "This is internal code, no need for security" → Insider threats exist
- "Length checks are enough" → Integer overflows
- "This can't happen" → Race conditions make it happen
- "Sanitized on the frontend" → Never trust client-side validation

## Attack Techniques Reference

### Code Injection

- SQL injection
- Command injection
- LDAP injection
- XML injection
- Template injection

### Memory Corruption

- Stack buffer overflow
- Heap buffer overflow
- Integer overflow/underflow
- Use-after-free
- Double-free

### Logic Flaws

- Authentication bypass
- Authorization bypass
- TOCTOU race conditions
- Business logic errors

### Cryptographic

- Weak algorithms (MD5, SHA1)
- Hardcoded keys
- Insufficient entropy
- Improper key management

### Side Channels

- Timing attacks
- Cache timing
- Power analysis
- Spectre/Meltdown variants

## Platform-Specific Concerns

### C/C++ (cisTEMx)

- Manual memory management → leaks, corruption
- Pointer arithmetic → bounds violations
- Type confusion → vtable hijacking
- Integer promotions → unexpected overflow

### CUDA/GPU

- Kernel parameter validation
- Device memory bounds
- Host-device data races
- GPU privilege escalation

### wxWidgets/GUI

- Event handler injection
- Clipboard attacks
- Drag-and-drop exploits
- Window message spoofing

### Networking

- Protocol parsing bugs
- Message boundary issues
- State machine confusion
- TLS/SSL misconfiguration

## Output Template

```markdown
# Red Team Security Analysis

## Executive Summary
[1-2 sentences: highest severity findings]

## Attack Surface
- [Entry point 1]: [Description, trust level]
- [Entry point 2]: [Description, trust level]

## Vulnerabilities

### CRITICAL: [Vulnerability Name]
**CWE**: CWE-XXX
**Location**: file.cpp:123
**Attack Vector**: [Network/Local/Physical]
**Complexity**: [Low/High]
**Impact**: [What attacker achieves]
**Proof of Concept**:
```cpp
[Exploit code or steps]
```
**Recommendation**: [Brief mitigation - Blue will expand]

[Repeat for MAJOR and MINOR findings]

## Exploit Chains
[How vulnerabilities combine for greater impact]

## Risk Assessment
- **Exploitability**: [High/Medium/Low]
- **Blast radius**: [Complete compromise/Partial/Limited]
- **Detection difficulty**: [Hard/Moderate/Easy]
```
