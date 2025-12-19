# cistemx.db

Database utilities for querying cisTEM project databases.

## Modules

### database.py

`TemplateMatchAnalyzer` - Query template matching results from cisTEM SQLite databases.

```python
from cistemx.db import TemplateMatchAnalyzer

with TemplateMatchAnalyzer('/path/to/project.db') as analyzer:
    # Get peak count for a job
    count = analyzer.get_result_count(job_id=1)

    # Load all peaks as DataFrame
    peaks = analyzer.load_all_peaks_for_jobs([1, 2, 3])

    # Find overlapping images between jobs
    overlap = analyzer.get_overlapping_images(job_id1=1, job_id2=8)
```

## Database Schema

See docstring in `database.py` for schema documentation:
- `TEMPLATE_MATCH_LIST` - One row per image result
- `TEMPLATE_MATCH_PEAK_LIST_{id}` - Detected peaks per image

## TODO

- Add write operations (currently read-only)
- Consider caching for repeated queries
