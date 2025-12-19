# Testing Strategy

## Goal: C++/CUDA ↔ Python Parity

The cistemx Python package must produce results consistent with the C++/CUDA
implementations. Python tests validate against freshly-generated C++ output.

## Architecture

### 1. Catch2 Custom CLI Flag

Add a `--dump-python-fixtures` flag to the test runner:

```cpp
// In test runner main.cpp
#define CATCH_CONFIG_RUNNER
#include <catch2/catch_all.hpp>

bool g_dump_python_fixtures = false;
std::string g_fixture_dir = "/tmp/cistemx_fixtures";

int main(int argc, char* argv[]) {
    Catch::Session session;

    using namespace Catch::Clara;
    auto cli = session.cli()
        | Opt(g_dump_python_fixtures)
            ["--dump-python-fixtures"]
            ("Write output files for Python parity tests")
        | Opt(g_fixture_dir, "dir")
            ["--fixture-dir"]
            ("Directory for fixture output");

    session.cli(cli);

    int rc = session.applyCommandLine(argc, argv);
    if (rc != 0) return rc;

    return session.run();
}
```

### 2. Tests with Conditional Output

Tests tagged `[python-parity]` always validate, conditionally write fixtures:

```cpp
TEST_CASE("Euler rotation matrix", "[geometry][python-parity]") {
    AnglesAndShifts angles(45.0f, 30.0f, 90.0f, 0.0f, 0.0f);
    RotationMatrix R = angles.GenerateRotationMatrix();

    // Normal test assertions (always run)
    REQUIRE(R.m[0][0] == Approx(expected_00));

    // Conditional fixture output
    if (g_dump_python_fixtures) {
        auto path = g_fixture_dir + "/euler_rotation_45_30_90.txt";
        std::ofstream out(path);
        // ... write matrix
    }
}
```

### 3. Python Tests Call C++ Runner

```python
# python/tests/conftest.py
import subprocess
import tempfile
import pytest

@pytest.fixture(scope="session")
def cpp_fixtures():
    """Run C++ tests to generate fresh fixtures."""
    fixture_dir = tempfile.mkdtemp(prefix="cistemx_fixtures_")

    result = subprocess.run([
        "./build/unit_test_runner",
        "[python-parity]",
        "--dump-python-fixtures",
        f"--fixture-dir={fixture_dir}"
    ], capture_output=True)

    if result.returncode != 0:
        pytest.fail(f"C++ fixture generation failed: {result.stderr}")

    return fixture_dir
```

```python
# python/tests/test_geometry_parity.py
import numpy as np
from cistemx.geometry import euler_to_rotation_matrix

def test_euler_rotation_matches_cpp(cpp_fixtures):
    """Validate against freshly-generated C++ output."""
    fixture = np.loadtxt(f"{cpp_fixtures}/euler_rotation_45_30_90.txt").reshape(3, 3)
    R = euler_to_rotation_matrix(45.0, 30.0, 90.0)
    np.testing.assert_allclose(R, fixture, rtol=1e-6)
```

## Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ pytest python/tests/                                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. conftest.py runs C++ unit_test_runner                   │
│     with --dump-python-fixtures --fixture-dir=/tmp/...      │
│                                                             │
│  2. C++ tests execute, write fixtures to temp dir           │
│                                                             │
│  3. Python tests load fixtures, compare against             │
│     cistemx module output                                   │
│                                                             │
│  4. Temp directory cleaned up                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Benefits:**
- Always testing against *current* C++ implementation
- No stale fixtures to maintain
- Single source of truth (C++ code)
- Python tests fail immediately if implementations diverge

## Fixture Categories

| Category | C++ Source | Python Module | Fixture Format |
|----------|------------|---------------|----------------|
| Geometry | `AnglesAndShifts` | `cistemx.geometry` | Text (matrices) |
| MRC I/O | `MRCFile` | `cistemx.io.mrc` | MRC files |
| Template matching | `TemplateMatching` | Future | MRC + text |

## Open Questions

1. **Build path**: How does pytest find unit_test_runner? Environment variable
   or pytest config option.

2. **CUDA tests**: Skip if no GPU available, or require GPU for full parity.

3. **Performance**: Running C++ tests adds overhead. Consider caching fixtures
   within a pytest session (already done with `scope="session"`).

## Implementation Order

1. [ ] Add `--dump-python-fixtures` and `--fixture-dir` to test runner
2. [ ] Add `[python-parity]` tag to first C++ test (geometry)
3. [ ] Create conftest.py with cpp_fixtures fixture
4. [ ] Write first pytest parity test
5. [ ] Validate end-to-end

## Related

- Main project Catch2 tests: `src/tests/`
- Catch2 CLI customization: https://github.com/catchorg/Catch2/blob/devel/docs/own-main.md
