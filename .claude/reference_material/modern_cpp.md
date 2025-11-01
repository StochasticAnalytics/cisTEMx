# Modern C++ Practices Reference
## Container Usage and Memory Management Guidelines

### Container Selection Guide

**Use STL containers for new code.** wxWidgets containers exist only for legacy compatibility.

#### Recommended Container Choices

| Use Case | Recommended | Avoid |
|----------|-------------|-------|
| Dynamic arrays | `std::vector<T>` | `wxArray`, `wxVector` |
| Lists | `std::list<T>`, `std::deque<T>` | `wxList` |
| String lists | `std::vector<wxString>` | `wxArrayString` |
| Key-value pairs | `std::unordered_map<K,V>` | `wxHashMap` |
| Sorted maps | `std::map<K,V>` | `wxSortedArray` |
| Sets | `std::unordered_set<T>` | Custom implementations |
| Fixed arrays | `std::array<T,N>` | C-style arrays |

#### Container-Specific Guidance

##### std::vector
```cpp
// Reserve when size is known
std::vector<Image> images;
images.reserve(expected_count);

// Use emplace_back for in-place construction
images.emplace_back(width, height);

// Range-based iteration
for (const auto& image : images) {
    ProcessImage(image);
}
```

##### std::array
```cpp
// Compile-time size, stack allocation
std::array<float, 3> euler_angles{0.0f, 0.0f, 0.0f};

// Bounds-checked access in debug
float angle = euler_angles.at(1);
```

### Memory Management Patterns

#### GUI Objects (wxWidgets)
**Use raw pointers with parent-child ownership:**

```cpp
class MyDialog : public wxDialog {
    wxButton* ok_button;      // Parent owns, will delete
    wxTextCtrl* input_field;  // Parent owns, will delete

    MyDialog() {
        // Parent takes ownership
        ok_button = new wxButton(this, wxID_OK);
        input_field = new wxTextCtrl(this, wxID_ANY);
        // No manual delete needed - parent handles it
    }
};
```

**Rationale**: wxWidgets uses parent-child hierarchy for automatic cleanup.

#### Non-GUI Objects
**Use smart pointers for automatic memory management:**

```cpp
// Single ownership
std::unique_ptr<SocketCommunicator> comm =
    std::make_unique<SocketCommunicator>(port);

// Shared ownership
std::shared_ptr<Database> db =
    std::make_shared<Database>(db_path);

// Observer pattern (non-owning)
std::weak_ptr<Database> db_observer = db;
```

#### Large Arrays
**Use explicit new/delete for control:**

```cpp
class ImageBuffer {
    float* data;
    size_t size;

public:
    ImageBuffer(size_t n) : size(n) {
        data = new float[size];
    }

    ~ImageBuffer() {
        delete[] data;
    }

    // Rule of 5: Delete copy, implement move
    ImageBuffer(const ImageBuffer&) = delete;
    ImageBuffer& operator=(const ImageBuffer&) = delete;
    ImageBuffer(ImageBuffer&& other) noexcept { /* ... */ }
    ImageBuffer& operator=(ImageBuffer&& other) noexcept { /* ... */ }
};
```

### RAII (Resource Acquisition Is Initialization)

**Always use RAII for resource management:**

```cpp
// File handles
class FileGuard {
    FILE* file;
public:
    explicit FileGuard(const char* path, const char* mode)
        : file(fopen(path, mode)) {}
    ~FileGuard() { if (file) fclose(file); }
    operator FILE*() { return file; }
};

// Usage
{
    FileGuard file("data.bin", "rb");
    if (file) {
        // Use file
    }  // Automatically closed
}
```

### Move Semantics

**Implement move operations for expensive-to-copy types:**

```cpp
class Image {
    float* data;
    size_t size;

public:
    // Move constructor
    Image(Image&& other) noexcept
        : data(other.data), size(other.size) {
        other.data = nullptr;
        other.size = 0;
    }

    // Move assignment
    Image& operator=(Image&& other) noexcept {
        if (this != &other) {
            delete[] data;
            data = other.data;
            size = other.size;
            other.data = nullptr;
            other.size = 0;
        }
        return *this;
    }
};
```

### Algorithm Usage

**Prefer STL algorithms over manual loops:**

```cpp
// Finding
auto it = std::find_if(particles.begin(), particles.end(),
    [](const Particle& p) { return p.score > threshold; });

// Transforming
std::transform(input.begin(), input.end(), output.begin(),
    [](float val) { return val * scale_factor; });

// Sorting
std::sort(particles.begin(), particles.end(),
    [](const Particle& a, const Particle& b) {
        return a.score > b.score;
    });
```

### Lambda Expressions

**Use lambdas for local function objects:**

```cpp
// Capture by reference for local scope
float threshold = 0.5f;
auto filter = [&threshold](const Particle& p) {
    return p.score > threshold;
};

// Capture by value for async operations
auto task = [data = std::move(large_data)]() {
    ProcessData(data);
};
```

### Type Deduction

**Use auto judiciously:**

```cpp
// Good: Complex types
auto result = CalculateComplexResult();
auto it = container.begin();

// Good: Range-based loops
for (const auto& item : container) { }

// Avoid: When type is important for clarity
auto size = GetSize();  // int? size_t? float?
size_t size = GetSize(); // Better
```

### Constexpr and Compile-Time Computation

```cpp
// Compile-time constants
constexpr size_t BUFFER_SIZE = 1024;
constexpr float PI = 3.14159265f;

// Compile-time functions
constexpr int factorial(int n) {
    return (n <= 1) ? 1 : n * factorial(n - 1);
}
```

### Migration Strategy

**Incremental Modernization:**
1. Don't refactor working code just to modernize
2. When modifying code, update to modern style
3. New code should always use modern C++
4. Document mixed-style boundaries

### Related Documentation
- See code_style_standards.md for formatting
- Check static_analysis.md for modernization checks
- Review main CLAUDE.md for philosophy