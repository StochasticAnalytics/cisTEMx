# cistemx.calculate_2dtm_p_value

Statistical p-value calculation for 2D template matching results.

**Author**: Kexin Zhang
**Reference**: Zhang et al. (2024). https://doi.org/10.1107/S2052252524011771

## Status: Isolated

This package is preserved as-is from the original `2DTM_postprocess_tools` repository.
Only import paths were updated (`tm_post` → `cistemx.calculate_2dtm_p_value`).

**No internal refactoring has been performed** because:
1. No test suite exists to validate changes
2. The statistical algorithms require domain expertise to verify
3. Breaking changes could silently corrupt scientific results

## CLI Tools

```bash
# Extract particles with p-value filtering
cistemx-extract-particles --help

# Filter existing particle sets
cistemx-filter-particles --help
```

## Module Overview

| Module | Purpose |
|--------|---------|
| `statistics.py` | Core p-value calculation algorithm |
| `extract.py` | Peak extraction from 2DTM results |
| `filters.py` | Particle filtering criteria |
| `database.py` | Database loading (separate from cistemx.db) |
| `geometry.py` | Euler angle utilities (redundant with cistemx.geometry) |
| `starfile.py` | STAR file I/O (custom implementation) |

## TODO Before Consolidation

1. Create comprehensive test suite with known-good outputs
2. Validate geometry functions match cistemx.geometry behavior
3. Evaluate starfile.py vs standard `starfile` package
4. Document expected inputs/outputs for statistical functions
