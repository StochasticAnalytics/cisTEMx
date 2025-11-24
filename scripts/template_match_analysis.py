#!/usr/bin/env python3
"""
Template Matching Analysis Module

Provides tools for analyzing template matching results stored in cisTEM databases.
Uses pandas DataFrames for efficient data caching and composable analysis operations.

Database Schema Overview
========================

TEMPLATE_MATCH_LIST (Static Table)
----------------------------------
One row per image result in a template matching job.

Key columns:
- TEMPLATE_MATCH_ID (PRIMARY KEY): Unique identifier for this result
- TEMPLATE_MATCH_JOB_ID: Groups multiple images from same job run (GUI shows this as 1-indexed)
- IMAGE_ASSET_ID: Which micrograph/tomogram was searched
- REFERENCE_VOLUME_ASSET_ID: Which 3D template was used
- IS_ACTIVE: Temporary hack for refinement workflow (ignore, will be removed)
- USED_THRESHOLD: Score threshold applied during search
- Plus: Search parameters (angular steps, defocus range, resolution limits, etc.)

TEMPLATE_MATCH_PEAK_LIST_{id} (Dynamic Tables)
-----------------------------------------------
One table per TEMPLATE_MATCH_ID, storing detected particles/features.

Table name format: TEMPLATE_MATCH_PEAK_LIST_<template_match_id>

Columns:
- PEAK_NUMBER (PRIMARY KEY): Sequential peak number
- X_POSITION, Y_POSITION: Peak location in image (pixels)
- PSI: In-plane rotation angle (degrees, 0-360)
- THETA: Out-of-plane tilt angle (degrees, 0-180)
- PHI: Out-of-plane azimuthal angle (degrees, 0-360)
- DEFOCUS: Refined defocus for this peak (Angstroms)
- PIXEL_SIZE: Refined pixel size for this peak
- PEAK_HEIGHT: Signal-to-noise ratio (SNR) score [NOT cross-correlation!]

Note: Empty tables (zero peaks) are valid and common - no detections above threshold.

IMAGE_ASSETS, IMAGE_GROUP_LIST, IMAGE_GROUP_{id}
-------------------------------------------------
Supporting tables for image metadata and grouping. See database_schema.h for details.

Usage Example
=============

    import template_match_analysis as tma

    # Create analyzer for job ID 1
    analyzer = tma.TemplateMatchAnalyzer(
        db_path='/path/to/project.db',
        job_id=1
    )

    # Get basic counts
    num_images = analyzer.get_completed_results_count()
    print(f"Analyzed {num_images} images")

    # Get comprehensive statistics
    stats = analyzer.get_peak_statistics()
    print(f"Total peaks: {stats['total_peaks']}")
    print(f"Mean SNR: {stats['global_stats']['mean_score']:.3f}")

    # Find which image groups match these results
    groups = analyzer.find_matching_groups()
    for group in groups['exact_matches']:
        print(f"Exact match: {group['group_name']}")

    # Access cached DataFrames directly for custom analysis
    peaks_df = analyzer.all_peaks
    high_snr_peaks = peaks_df[peaks_df['PEAK_HEIGHT'] > 5.0]

Dependencies
============
- pandas: DataFrame operations and statistics
- numpy: Numerical computations
- sqlite3: Database connectivity (standard library)

"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, Set, List, Tuple


class TemplateMatchAnalyzer:
    """
    Analyzer for template matching results from cisTEM database.

    Provides methods to load and analyze template matching data with composable operations.

    Attributes:
        db_path (str): Path to database file
    """

    def __init__(self, db_path: str):
        """
        Initialize analyzer and open database connection.

        Args:
            db_path: Path to cisTEM database file

        Raises:
            FileNotFoundError: If database file doesn't exist
            sqlite3.Error: If file exists but is not a valid SQLite database
        """
        import os

        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file not found: {db_path}")

        # Open read-only connection and keep it open
        try:
            self.conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        except sqlite3.Error as e:
            raise sqlite3.Error(f"Invalid or unreadable SQLite database: {e}")

        self.db_path = db_path

    def __enter__(self):
        """Context manager entry - connection already opened in __init__."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close database connection."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
        return False

    def __del__(self):
        """Destructor - close database connection if still open."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()

    def get_result_count(self, job_id: int) -> int:
        """
        Get the number of images with completed results for a job.

        Args:
            job_id: Template match job ID

        Returns:
            Count of distinct IMAGE_ASSET_IDs with results for this job
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT COUNT(DISTINCT IMAGE_ASSET_ID)
            FROM TEMPLATE_MATCH_LIST
            WHERE TEMPLATE_MATCH_JOB_ID = ?
        """, (job_id,))
        return cursor.fetchone()[0]

    def load_all_peaks_for_jobs(self, job_ids: List[int]) -> pd.DataFrame:
        """
        Load all peaks from one or more template matching jobs into a single DataFrame.

        This consolidates data from multiple TEMPLATE_MATCH_PEAK_LIST_{id} tables
        into a single long-format DataFrame for efficient analysis.

        Args:
            job_ids: List of template match job IDs to load (or single ID in list)

        Returns:
            Long-format DataFrame with columns:
            - TEMPLATE_MATCH_JOB_ID: Job identifier
            - IMAGE_ASSET_ID: Image identifier
            - TEMPLATE_MATCH_ID: Match result identifier
            - PEAK_NUMBER: Sequential peak number within image
            - X_POSITION, Y_POSITION: Peak location (pixels)
            - PSI, THETA, PHI: Euler angles (degrees)
            - DEFOCUS: Refined defocus (Angstroms)
            - PIXEL_SIZE: Refined pixel size
            - PEAK_HEIGHT: SNR score

        Example:
            # Load peaks from multiple jobs for comparison
            df = analyzer.load_all_peaks_for_jobs([1, 2, 3])

            # Filter to one job
            job1_peaks = df[df['TEMPLATE_MATCH_JOB_ID'] == 1]
        """
        cursor = self.conn.cursor()

        # Get all TEMPLATE_MATCH_IDs for these jobs with metadata
        placeholders = ','.join('?' * len(job_ids))
        query = f"""
            SELECT TEMPLATE_MATCH_JOB_ID, TEMPLATE_MATCH_ID, IMAGE_ASSET_ID
            FROM TEMPLATE_MATCH_LIST
            WHERE TEMPLATE_MATCH_JOB_ID IN ({placeholders})
        """
        cursor.execute(query, job_ids)
        match_info = cursor.fetchall()

        if not match_info:
            # No results for these jobs - return empty DataFrame
            return pd.DataFrame(columns=[
                'TEMPLATE_MATCH_JOB_ID', 'IMAGE_ASSET_ID', 'TEMPLATE_MATCH_ID',
                'PEAK_NUMBER', 'X_POSITION', 'Y_POSITION', 'PSI', 'THETA', 'PHI',
                'DEFOCUS', 'PIXEL_SIZE', 'PEAK_HEIGHT'
            ])

        # Load peaks from each image
        all_peaks = []
        for job_id, match_id, image_asset_id in match_info:
            table_name = f"TEMPLATE_MATCH_PEAK_LIST_{match_id}"

            # Query all peaks from this table
            query = f"SELECT * FROM {table_name}"
            peaks_df = pd.read_sql_query(query, self.conn)

            # Add metadata columns
            peaks_df['TEMPLATE_MATCH_JOB_ID'] = job_id
            peaks_df['TEMPLATE_MATCH_ID'] = match_id
            peaks_df['IMAGE_ASSET_ID'] = image_asset_id

            all_peaks.append(peaks_df)

        # Combine all peaks into single long-format DataFrame
        if all_peaks:
            combined_df = pd.concat(all_peaks, ignore_index=True)

            # Reorder columns for clarity (metadata first, then peak data)
            column_order = [
                'TEMPLATE_MATCH_JOB_ID', 'IMAGE_ASSET_ID', 'TEMPLATE_MATCH_ID',
                'PEAK_NUMBER', 'X_POSITION', 'Y_POSITION', 'PSI', 'THETA', 'PHI',
                'DEFOCUS', 'PIXEL_SIZE', 'PEAK_HEIGHT'
            ]
            return combined_df[column_order]
        else:
            return pd.DataFrame(columns=[
                'TEMPLATE_MATCH_JOB_ID', 'IMAGE_ASSET_ID', 'TEMPLATE_MATCH_ID',
                'PEAK_NUMBER', 'X_POSITION', 'Y_POSITION', 'PSI', 'THETA', 'PHI',
                'DEFOCUS', 'PIXEL_SIZE', 'PEAK_HEIGHT'
            ])

    def get_peaks_by_count(self, job_id: int, operator: str, threshold: int) -> pd.DataFrame:
        """
        Get all peak data for images in a job that meet a peak count condition.

        Args:
            job_id: Template match job ID
            operator: Comparison operator as string: '>', '>=', '==', '<', '<=', '!='
            threshold: Peak count threshold value

        Returns:
            Long-format DataFrame with columns:
            - TEMPLATE_MATCH_JOB_ID: Job identifier
            - IMAGE_ASSET_ID: Image identifier
            - TEMPLATE_MATCH_ID: Match result identifier
            - PEAK_NUMBER: Sequential peak number within image
            - X_POSITION, Y_POSITION: Peak location (pixels)
            - PSI, THETA, PHI: Euler angles (degrees)
            - DEFOCUS: Refined defocus (Angstroms)
            - PIXEL_SIZE: Refined pixel size
            - PEAK_HEIGHT: SNR score

        Example:
            # Get peaks from images with more than 10 detections
            df = analyzer.get_peaks_by_count(job_id=1, operator='>', threshold=10)
        """
        # Validate operator to prevent SQL injection
        valid_operators = {'>', '>=', '==', '<', '<=', '!='}
        if operator not in valid_operators:
            raise ValueError(f"Invalid operator '{operator}'. Must be one of: {valid_operators}")

        # Load all peaks for this job using consolidated loader
        all_peaks = self.load_all_peaks_for_jobs([job_id])

        if len(all_peaks) == 0:
            return all_peaks  # Already has correct empty structure

        # Count peaks per image
        peak_counts = all_peaks.groupby('TEMPLATE_MATCH_ID').size()

        # Apply condition to find matching images
        if operator == '>':
            matching_ids = peak_counts[peak_counts > threshold].index
        elif operator == '>=':
            matching_ids = peak_counts[peak_counts >= threshold].index
        elif operator == '==':
            matching_ids = peak_counts[peak_counts == threshold].index
        elif operator == '<':
            matching_ids = peak_counts[peak_counts < threshold].index
        elif operator == '<=':
            matching_ids = peak_counts[peak_counts <= threshold].index
        elif operator == '!=':
            matching_ids = peak_counts[peak_counts != threshold].index

        # Filter to only peaks from matching images
        filtered_peaks = all_peaks[all_peaks['TEMPLATE_MATCH_ID'].isin(matching_ids)]

        return filtered_peaks

    def get_all_image_groups(self) -> List[Tuple[int, str]]:
        """
        Get all image groups that exist (have both entry in IMAGE_GROUP_LIST and dynamic table).

        Excludes the default "All Images" group (GROUP_ID = -1).
        Validates that dynamic IMAGE_GROUP_{id} tables actually exist.

        Returns:
            List of (group_id, group_name) tuples for valid groups
        """
        cursor = self.conn.cursor()

        # Get all IMAGE_GROUP_* table names that exist in one query
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name LIKE 'IMAGE_GROUP_%'
        """)
        existing_tables = {row[0] for row in cursor.fetchall()}

        # Parse group IDs from table names
        existing_group_ids = set()
        for table_name in existing_tables:
            try:
                group_id = int(table_name.replace('IMAGE_GROUP_', ''))
                existing_group_ids.add(group_id)
            except ValueError:
                continue

        # Get all groups from IMAGE_GROUP_LIST (excluding "All Images" default)
        cursor.execute("""
            SELECT GROUP_ID, GROUP_NAME
            FROM IMAGE_GROUP_LIST
            WHERE GROUP_ID != -1
        """)
        all_groups = cursor.fetchall()

        # Return only groups that have both list entry AND existing table
        return [(gid, gname) for gid, gname in all_groups if gid in existing_group_ids]

    def find_matching_groups(self, job_id: int) -> Dict:
        """
        Identify which image groups could have been used for a template matching job.

        Compares the set of images with results against all defined image groups.
        Excludes the default "All Images" group (GROUP_ID = -1).

        Args:
            job_id: Template match job ID to analyze

        Returns:
            Dictionary with structure:
            {
                'exact_matches': [
                    {'group_id': int, 'group_name': str, 'image_count': int},
                    ...
                ],
                'superset_matches': [
                    {'group_id': int, 'group_name': str,
                     'total_images': int, 'matched_images': int},
                    ...
                ]
            }

        Match types:
        - exact_matches: Groups containing exactly the same images as results
        - superset_matches: Groups containing all result images plus additional images
        """
        # Get total number of images with results for this job
        total_results = self.get_result_count(job_id)

        if total_results == 0:
            return {'exact_matches': [], 'superset_matches': []}

        exact_matches = []
        superset_matches = []

        # Get all valid image groups (validated in get_all_image_groups)
        groups = self.get_all_image_groups()

        cursor = self.conn.cursor()
        for group_id, group_name in groups:
            table_name = f"IMAGE_GROUP_{group_id}"

            # Count total images in group
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            total_in_group = cursor.fetchone()[0]

            # Count how many images in this group have results for this job (via JOIN)
            cursor.execute(f"""
                SELECT COUNT(DISTINCT ig.IMAGE_ASSET_ID)
                FROM {table_name} ig
                INNER JOIN TEMPLATE_MATCH_LIST tml
                    ON ig.IMAGE_ASSET_ID = tml.IMAGE_ASSET_ID
                WHERE tml.TEMPLATE_MATCH_JOB_ID = ?
            """, (job_id,))
            matched_count = cursor.fetchone()[0]

            # Exact match: all results in group, all group images have results
            if matched_count == total_results and matched_count == total_in_group:
                exact_matches.append({
                    'group_id': group_id,
                    'group_name': group_name,
                    'image_count': total_in_group
                })
            # Superset: all results in group, but group has more images
            elif matched_count == total_results and matched_count < total_in_group:
                superset_matches.append({
                    'group_id': group_id,
                    'group_name': group_name,
                    'total_images': total_in_group,
                    'matched_images': matched_count
                })

        return {
            'exact_matches': exact_matches,
            'superset_matches': superset_matches
        }
