# Code Style Standards Reference
## Coding Conventions and Formatting Rules for cisTEMx

### Automatic Formatting
The project uses `.clang-format` in the root directory for consistent code formatting.

```bash
# Format a single file
clang-format -i src/core/my_file.cpp

# Format all changed files
git diff --name-only | xargs clang-format -i
```

### Type Casting Conventions

**Always use modern C++ functional cast style:**
```cpp
// CORRECT - Functional style
int value = int(float_variable);
long count = long(size_variable);
float ratio = float(integer_value);

// INCORRECT - C-style casts
int value = (int)float_variable;    // Don't use
long count = (long)size_variable;   // Don't use
```

### wxWidgets-Specific Rules

#### Printf Formatting
**Critical**: Format specifier mismatches cause segfaults in wxFormatConverterBase

```cpp
// CORRECT - Exact type matching
wxString::Format(wxT("%ld items"), long(count));
wxString::Format(wxT("%d selected"), int(selection));
wxString::Format(wxT("%f ratio"), float(value));

// INCORRECT - Type mismatches
wxString::Format(wxT("%d items"), long(count));  // SEGFAULT!
```

#### Unicode Characters
**Never use Unicode characters in format strings** - causes segmentation faults

```cpp
// INCORRECT - Unicode characters
wxString::Format(wxT("Resolution: %f Å"), resolution);  // SEGFAULT!
wxString::Format(wxT("Angle: %f °"), angle);           // SEGFAULT!

// CORRECT - ASCII equivalents
wxString::Format(wxT("Resolution: %f A"), resolution);
wxString::Format(wxT("Angle: %f deg"), angle);
```

### Temporary Debugging Code

**All temporary debugging changes must be marked:**

```cpp
// revert - Added debug output to track socket state
printf("Socket state: %d\n", socket_state);

// revert - Commented out for testing without GPU
// if (runtime.has_gpu) {
//     RunGPUKernel();
// }
```

**Before committing:**
```bash
# Find all temporary changes
grep -r "// revert" src/

# Remove all temporary changes before commit
```

### Preprocessor Defines

**Project-specific defines must use `cisTEM_` prefix:**

```cpp
// CORRECT
#define cisTEM_ENABLE_PROFILING
#define cisTEM_MAX_THREADS 64
#ifdef cisTEM_DEBUG_MODE

// INCORRECT - No prefix
#define ENABLE_PROFILING      // Collision risk
#define MAX_THREADS 64        // Too generic
```

### Include Guards

**Use full path from project root:**

```cpp
// For src/gui/MyDialog.h
#ifndef _SRC_GUI_MYDIALOG_H_
#define _SRC_GUI_MYDIALOG_H_
// ... header content ...
#endif

// NOT:
#ifndef __MyDialog__      // Too generic
#ifndef MYDIALOG_H       // Missing path context
```

### File Organization

#### Header Files (.h)
1. Include guard
2. System includes
3. Library includes (wxWidgets, etc.)
4. Project includes
5. Forward declarations
6. Class definition

#### Implementation Files (.cpp)
1. Corresponding header include
2. System includes
3. Library includes
4. Other project includes
5. Anonymous namespace for file-local items
6. Implementation

### Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Classes | PascalCase | `SocketCommunicator` |
| Methods | PascalCase | `SendMessage()` |
| Variables | snake_case | `buffer_size` |
| Constants | UPPER_SNAKE | `MAX_BUFFER_SIZE` |
| Members | snake_case with trailing _ | `socket_id_` |

### Modern C++ Preferences

#### Prefer Modern Constructs
```cpp
// Prefer auto for complex types
auto result = ComplexFunction();

// Use range-based for loops
for (const auto& item : container) {
    ProcessItem(item);
}

// nullptr over NULL
MyClass* ptr = nullptr;

// Using declarations over typedefs
using BufferType = std::vector<uint8_t>;
```

### Philosophy

**Incremental Modernization**: Update and unify style as code is modified rather than wholesale changes.

**Legacy Compatibility**: Many legacy features exist; maintain compatibility while gradually improving.

### Related Documentation
- See `.clang-format` for formatting rules
- Check static_analysis.md for linting setup
- Review modern_cpp.md for container and memory guidelines