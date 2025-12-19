# Experimental Scripts

Research and analysis scripts for template matching workflows.

## Status

These scripts are **working but untested** - used for active research, not production.
They demonstrate cistemx usage patterns and may be promoted to proper modules later.

## Scripts

| Script | Purpose |
|--------|---------|
| `test_cross_search_comparison.py` | Compare peaks between two TM searches |
| `multi_search_consensus.py` | Find consensus peaks across multiple searches |
| `peak_recovery_fft.py` | FFT-based sub-pixel peak refinement |
| `peak_recovery_fft_components.py` | Component analysis for FFT refinement |
| `extract_particles_subpixel_test.py` | Particle extraction with sub-pixel positions |
| `analyze_resolution_height_correlation.py` | Correlate peak height with resolution |
| `analyze_subpixel_height_correlation.py` | Correlate sub-pixel refinement with height |
| `plot_threshold_resolution.py` | Visualize threshold vs resolution tradeoffs |

## Usage

Scripts use cistemx modules and can be run directly:

```bash
python experimental_scripts/test_cross_search_comparison.py project.db 1 8 --position-tolerance 3.0
```

## Dependencies

Scripts import from:
- `cistemx.db` - TemplateMatchAnalyzer
- `cistemx.geometry` - Euler angle utilities
- Standard scientific Python (numpy, pandas, matplotlib)
