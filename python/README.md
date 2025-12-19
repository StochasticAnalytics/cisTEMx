# cistemx Python Package

Python utilities for cisTEMx cryo-EM analysis.

## Purpose

This package serves two goals:

1. **Accessible tools**: Provide researchers with Python interfaces to cisTEMx
   algorithms, enabling result reproduction and workflow customization without
   requiring C++/CUDA expertise.

2. **Reference implementations**: Even if core C++/CUDA code becomes proprietary,
   this package documents algorithm behavior through working Python code and
   validated test fixtures.

## Installation

```bash
cd python
pip install -e .
```

## Package Structure

```
cistemx/
├── io/                  # File I/O (MRC, STAR)
├── db/                  # Database queries (TemplateMatchAnalyzer)
├── geometry/            # Rotation math (Euler angles, SO(3) geodesics)
└── calculate_2dtm_p_value/  # Kexin Zhang's p-value package (isolated)
```

### Module Purposes

| Module | Purpose |
|--------|---------|
| `cistemx.io` | Reading/writing cryo-EM file formats |
| `cistemx.db` | Querying cisTEM SQLite databases |
| `cistemx.geometry` | Euler angle conversions, orientation comparisons |
| `cistemx.calculate_2dtm_p_value` | Statistical p-value calculation for 2DTM |

## CLI Entry Points

After installation, these commands are available:

- `cistemx-extract-particles` - Extract particles from 2DTM results
- `cistemx-filter-particles` - Filter particles by statistical criteria

## Experimental Scripts

Analysis scripts in `experimental_scripts/` are working but untested research code.
They demonstrate usage patterns and may be promoted to proper modules later.

## Testing Strategy

See `tests/README.md` for the parity testing approach that validates Python
implementations against C++/CUDA ground truth using Catch2-generated fixtures.

## Known Issues / TODO

- **Module redundancy**: `cistemx.geometry.rotations` and `cistemx.calculate_2dtm_p_value.geometry`
  both implement Euler angle utilities. Kexin's code is intentionally isolated pending
  proper test coverage before consolidation.

- **No test suite yet**: Need pytest tests before refactoring.

## References

- Zhang et al. (2024). https://doi.org/10.1107/S2052252524011771
