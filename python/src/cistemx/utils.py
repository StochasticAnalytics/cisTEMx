"""
Utility functions for cistemx.

This module provides common utilities used across cistemx tools and scripts.
"""


def parse_job_range(range_str: str) -> list[int]:
    """
    Parse job range specification into list of job IDs.

    Supports mixed formats (order preserved):
    - Single value: "12" → [12]
    - Colon range: "12:18" → [12, 13, 14, 15, 16, 17, 18]
    - Comma list: "12,14,16" → [12, 14, 16]
    - Mixed: "14,2:4" → [14, 2, 3, 4]
    - Mixed: "1,5:7,10" → [1, 5, 6, 7, 10]

    Args:
        range_str: Range specification string

    Returns:
        List of job IDs in the order specified

    Raises:
        ValueError: If format is invalid or range is empty

    Examples:
        >>> parse_job_range("12:18")
        [12, 13, 14, 15, 16, 17, 18]
        >>> parse_job_range("14,2:4")
        [14, 2, 3, 4]
        >>> parse_job_range("1,5:7,10")
        [1, 5, 6, 7, 10]
    """
    range_str = range_str.strip()
    job_ids = []

    # Split by comma first, then handle each part
    parts = [p.strip() for p in range_str.split(',')]

    for part in parts:
        if ':' in part:
            # Range format: "12:18"
            range_parts = part.split(':')
            if len(range_parts) != 2:
                raise ValueError(f"Invalid range format '{part}'. Use 'start:end' (e.g., '12:18')")
            try:
                start, end = int(range_parts[0]), int(range_parts[1])
            except ValueError:
                raise ValueError(f"Invalid integers in range '{part}'")
            if start > end:
                raise ValueError(f"Start ({start}) must be <= end ({end})")
            job_ids.extend(range(start, end + 1))
        else:
            # Single value
            try:
                job_ids.append(int(part))
            except ValueError:
                raise ValueError(f"Invalid job ID '{part}'")

    if not job_ids:
        raise ValueError("No job IDs specified")

    return job_ids
