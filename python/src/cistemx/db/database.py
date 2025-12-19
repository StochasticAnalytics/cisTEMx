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

    from cistemx.io.database import TemplateMatchAnalyzer

    # Create analyzer for database
    analyzer = TemplateMatchAnalyzer(db_path='/path/to/project.db')

    # Get basic counts
    num_images = analyzer.get_result_count(job_id=1)
    print(f"Analyzed {num_images} images")

    # Load peaks for analysis
    peaks = analyzer.load_all_peaks_for_jobs([1, 2, 3])
    high_snr_peaks = peaks[peaks['PEAK_HEIGHT'] > 5.0]

Dependencies
============
- pandas: DataFrame operations and statistics
- numpy: Numerical computations
- sqlite3: Database connectivity (standard library)

"""

import sqlite3
import os
from typing import Dict, Set, List, Tuple

import pandas as pd
import numpy as np


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

    def add_metadata_columns(self, peaks_df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Add columns from TEMPLATE_MATCH_LIST metadata to peaks DataFrame.

        Enriches peak data with per-image metadata like defocus values, thresholds,
        search parameters, etc. Values are joined on TEMPLATE_MATCH_ID and will be
        redundant for all peaks from the same image.

        Args:
            peaks_df: DataFrame from load_all_peaks_for_jobs() or get_peaks_by_count()
            columns: List of column names from TEMPLATE_MATCH_LIST to add
                     e.g., ['USED_DEFOCUS1', 'USED_DEFOCUS2', 'USED_THRESHOLD']

        Returns:
            DataFrame with additional metadata columns

        Example:
            # Load peaks and add defocus information
            peaks = analyzer.load_all_peaks_for_jobs([8])
            peaks_with_defocus = analyzer.add_metadata_columns(
                peaks, ['USED_DEFOCUS1', 'USED_DEFOCUS2']
            )
        """
        if len(peaks_df) == 0:
            # Empty DataFrame - just add empty columns
            for col in columns:
                peaks_df[col] = pd.Series(dtype='float64')
            return peaks_df

        # Get unique TEMPLATE_MATCH_IDs from peaks
        match_ids = peaks_df['TEMPLATE_MATCH_ID'].unique()

        # Build query to select requested columns plus TEMPLATE_MATCH_ID for joining
        columns_str = ', '.join(['TEMPLATE_MATCH_ID'] + columns)
        placeholders = ','.join('?' * len(match_ids))
        query = f"""
            SELECT {columns_str}
            FROM TEMPLATE_MATCH_LIST
            WHERE TEMPLATE_MATCH_ID IN ({placeholders})
        """

        # Load metadata
        metadata_df = pd.read_sql_query(query, self.conn, params=match_ids.tolist())

        # Join metadata to peaks on TEMPLATE_MATCH_ID
        enriched_df = peaks_df.merge(metadata_df, on='TEMPLATE_MATCH_ID', how='left')

        return enriched_df

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

    def get_overlapping_images(self, job_id1: int, job_id2: int) -> Set[int]:
        """
        Find images that were processed by both template matching jobs.

        This is useful for cross-search comparisons where you need to
        analyze the same images under different search conditions.

        Args:
            job_id1: First template match job ID
            job_id2: Second template match job ID

        Returns:
            Set of IMAGE_ASSET_IDs that have results in both jobs

        Example:
            overlap = analyzer.get_overlapping_images(1, 8)
            print(f"{len(overlap)} images analyzed in both searches")
        """
        cursor = self.conn.cursor()

        # Single query using INTERSECT for efficiency
        cursor.execute("""
            SELECT IMAGE_ASSET_ID FROM TEMPLATE_MATCH_LIST
            WHERE TEMPLATE_MATCH_JOB_ID = ?
            INTERSECT
            SELECT IMAGE_ASSET_ID FROM TEMPLATE_MATCH_LIST
            WHERE TEMPLATE_MATCH_JOB_ID = ?
        """, (job_id1, job_id2))

        return {row[0] for row in cursor.fetchall()}

    def get_result_file_paths(self, job_id: int,
                               image_asset_ids: Set[int] = None) -> pd.DataFrame:
        """
        Get output file paths for template matching results.

        Returns paths to all output images (MIP, angular maps, defocus map, etc.)
        for each image result in the specified job.

        Args:
            job_id: Template match job ID
            image_asset_ids: Optional set of IMAGE_ASSET_IDs to filter by.
                           If None, returns paths for all images in job.

        Returns:
            DataFrame with columns:
            - IMAGE_ASSET_ID: Which image this result is for
            - TEMPLATE_MATCH_ID: Unique ID for this result (links to peak table)
            - MIP_OUTPUT_FILE: Maximum intensity projection (raw scores)
            - SCALED_MIP_OUTPUT_FILE: Normalized/scaled MIP = (MIP - AVG) / STD
            - AVG_OUTPUT_FILE: Per-pixel average across all correlations
            - STD_OUTPUT_FILE: Per-pixel standard deviation
            - PSI_OUTPUT_FILE: Best in-plane rotation angle per pixel
            - THETA_OUTPUT_FILE: Best out-of-plane tilt per pixel
            - PHI_OUTPUT_FILE: Best azimuthal angle per pixel
            - DEFOCUS_OUTPUT_FILE: Best defocus per pixel
            - PIXEL_SIZE_OUTPUT_FILE: Best pixel size per pixel

        Example:
            # Get paths for specific images
            overlap = analyzer.get_overlapping_images(1, 8)
            paths_job1 = analyzer.get_result_file_paths(1, overlap)
            paths_job8 = analyzer.get_result_file_paths(8, overlap)
        """
        columns = [
            'IMAGE_ASSET_ID',
            'TEMPLATE_MATCH_ID',
            'MIP_OUTPUT_FILE',
            'SCALED_MIP_OUTPUT_FILE',
            'AVG_OUTPUT_FILE',
            'STD_OUTPUT_FILE',
            'PSI_OUTPUT_FILE',
            'THETA_OUTPUT_FILE',
            'PHI_OUTPUT_FILE',
            'DEFOCUS_OUTPUT_FILE',
            'PIXEL_SIZE_OUTPUT_FILE'
        ]

        columns_str = ', '.join(columns)

        if image_asset_ids is not None and len(image_asset_ids) > 0:
            # Filter by specific image IDs
            placeholders = ','.join('?' * len(image_asset_ids))
            query = f"""
                SELECT {columns_str}
                FROM TEMPLATE_MATCH_LIST
                WHERE TEMPLATE_MATCH_JOB_ID = ?
                AND IMAGE_ASSET_ID IN ({placeholders})
            """
            params = [job_id] + list(image_asset_ids)
        else:
            # Return all images for this job
            query = f"""
                SELECT {columns_str}
                FROM TEMPLATE_MATCH_LIST
                WHERE TEMPLATE_MATCH_JOB_ID = ?
            """
            params = [job_id]

        return pd.read_sql_query(query, self.conn, params=params)

    def get_peaks_by_group(self, job_id: int, group_name: str) -> pd.DataFrame:
        """
        Get all peaks for images in a specific image group within a job.

        This combines job filtering (which images were analyzed) with group filtering
        (which images belong to a named group).

        Args:
            job_id: Template match job ID
            group_name: Name of the image group (from IMAGE_GROUP_LIST)

        Returns:
            Long-format DataFrame with same structure as load_all_peaks_for_jobs()

        Raises:
            ValueError: If group_name doesn't exist in IMAGE_GROUP_LIST

        Example:
            # Get peaks only for images in "Good Images" group
            df = analyzer.get_peaks_by_group(job_id=8, group_name="Good Images")
        """
        cursor = self.conn.cursor()

        # Find group_id from group_name
        cursor.execute("""
            SELECT GROUP_ID FROM IMAGE_GROUP_LIST WHERE GROUP_NAME = ?
        """, (group_name,))
        result = cursor.fetchone()
        if result is None:
            raise ValueError(f"Image group '{group_name}' not found")
        group_id = result[0]

        # Validate that dynamic table exists
        table_name = f"IMAGE_GROUP_{group_id}"
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name = ?
        """, (table_name,))
        if cursor.fetchone() is None:
            raise ValueError(f"Image group table '{table_name}' does not exist")

        # Get TEMPLATE_MATCH_IDs for this job that are in this group
        # Join TEMPLATE_MATCH_LIST with IMAGE_GROUP_{id} on IMAGE_ASSET_ID
        query = f"""
            SELECT tml.TEMPLATE_MATCH_JOB_ID, tml.TEMPLATE_MATCH_ID, tml.IMAGE_ASSET_ID
            FROM TEMPLATE_MATCH_LIST tml
            INNER JOIN {table_name} ig
                ON tml.IMAGE_ASSET_ID = ig.IMAGE_ASSET_ID
            WHERE tml.TEMPLATE_MATCH_JOB_ID = ?
        """
        cursor.execute(query, (job_id,))
        match_info = cursor.fetchall()

        if not match_info:
            # No results for this job in this group - return empty DataFrame
            return pd.DataFrame(columns=[
                'TEMPLATE_MATCH_JOB_ID', 'IMAGE_ASSET_ID', 'TEMPLATE_MATCH_ID',
                'PEAK_NUMBER', 'X_POSITION', 'Y_POSITION', 'PSI', 'THETA', 'PHI',
                'DEFOCUS', 'PIXEL_SIZE', 'PEAK_HEIGHT'
            ])

        # Load peaks from each matched image
        all_peaks = []
        for job_id_val, match_id, image_asset_id in match_info:
            peak_table = f"TEMPLATE_MATCH_PEAK_LIST_{match_id}"

            # Query all peaks from this table
            peaks_df = pd.read_sql_query(f"SELECT * FROM {peak_table}", self.conn)

            # Add metadata columns
            peaks_df['TEMPLATE_MATCH_JOB_ID'] = job_id_val
            peaks_df['TEMPLATE_MATCH_ID'] = match_id
            peaks_df['IMAGE_ASSET_ID'] = image_asset_id

            all_peaks.append(peaks_df)

        # Combine all peaks into single long-format DataFrame
        if all_peaks:
            combined_df = pd.concat(all_peaks, ignore_index=True)

            # Reorder columns for consistency
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

    def plot_defocus_analysis(self, peaks_df: pd.DataFrame, job_id: int,
                               output_dir: str = '.') -> Tuple[str, str]:
        """
        Generate 2D histogram analysis plots for peak height vs defocus parameters.

        Creates two plots:
        1. Peak height vs total defocus and astigmatism
        2. Peak height vs defocus angle and astigmatism

        Filtering applied before analysis:
        - Removes lowest 3 peaks per image
        - Removes outliers beyond +/-2.5 SD from per-image mean

        Args:
            peaks_df: DataFrame with peaks (must include USED_DEFOCUS1, USED_DEFOCUS2,
                      USED_DEFOCUS_ANGLE metadata columns)
            job_id: Job ID for plot titles and output filenames
            output_dir: Directory to save plots (default: current directory)

        Returns:
            Tuple of (defocus_plot_path, angle_plot_path)

        Raises:
            ImportError: If matplotlib is not installed
            ValueError: If required metadata columns are missing

        Example:
            # Load peaks with defocus metadata
            peaks = analyzer.get_peaks_by_count(8, '>=', 4)
            peaks_with_meta = analyzer.add_metadata_columns(
                peaks, ['USED_DEFOCUS1', 'USED_DEFOCUS2', 'USED_DEFOCUS_ANGLE']
            )

            # Generate analysis plots
            plot1, plot2 = analyzer.plot_defocus_analysis(peaks_with_meta, 8)
            print(f"Plots saved: {plot1}, {plot2}")
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

        # Validate required columns
        required_cols = ['IMAGE_ASSET_ID', 'PEAK_HEIGHT', 'USED_DEFOCUS1',
                         'USED_DEFOCUS2', 'USED_DEFOCUS_ANGLE']
        missing_cols = [col for col in required_cols if col not in peaks_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}. "
                           f"Use add_metadata_columns() to add defocus metadata first.")

        if len(peaks_df) == 0:
            raise ValueError("Empty peaks DataFrame provided")

        # Calculate total defocus and astigmatism
        peaks_with_metadata = peaks_df.copy()
        peaks_with_metadata['TOTAL_DEFOCUS'] = (
            peaks_with_metadata['USED_DEFOCUS1'] + peaks_with_metadata['USED_DEFOCUS2']
        ) / 2.0
        peaks_with_metadata['ASTIGMATISM'] = abs(
            peaks_with_metadata['USED_DEFOCUS1'] - peaks_with_metadata['USED_DEFOCUS2']
        )

        # Filter peaks: remove lowest 3 per image, then remove outliers
        filtered_peaks = []

        for image_id in peaks_with_metadata['IMAGE_ASSET_ID'].unique():
            img_peaks = peaks_with_metadata[peaks_with_metadata['IMAGE_ASSET_ID'] == image_id].copy()

            # Remove lowest 3 peaks by PEAK_HEIGHT
            if len(img_peaks) > 3:
                img_peaks = img_peaks.nlargest(len(img_peaks) - 3, 'PEAK_HEIGHT')

            # Calculate mean and std for this image
            if len(img_peaks) > 0:
                mean_height = img_peaks['PEAK_HEIGHT'].mean()
                std_height = img_peaks['PEAK_HEIGHT'].std()

                # Filter peaks within 2.5 standard deviations
                if std_height > 0:
                    lower_bound = mean_height - 2.5 * std_height
                    upper_bound = mean_height + 2.5 * std_height
                    img_peaks = img_peaks[
                        (img_peaks['PEAK_HEIGHT'] >= lower_bound) &
                        (img_peaks['PEAK_HEIGHT'] <= upper_bound)
                    ]

                filtered_peaks.append(img_peaks)

        # Combine filtered peaks
        if not filtered_peaks:
            raise ValueError("No peaks remaining after filtering")

        peaks_with_metadata = pd.concat(filtered_peaks, ignore_index=True)

        # Create 2D bins
        n_bins_x = 100  # Total defocus bins
        n_bins_y = 100  # Astigmatism bins

        peaks_with_metadata['DEFOCUS_BIN'] = pd.cut(
            peaks_with_metadata['TOTAL_DEFOCUS'],
            bins=n_bins_x
        )
        peaks_with_metadata['ASTIGMATISM_BIN'] = pd.cut(
            peaks_with_metadata['ASTIGMATISM'],
            bins=n_bins_y
        )

        # Calculate average peak height per 2D bin
        bin_stats_2d = peaks_with_metadata.groupby(
            ['DEFOCUS_BIN', 'ASTIGMATISM_BIN'],
            observed=True
        ).agg({
            'PEAK_HEIGHT': ['mean', 'count']
        }).reset_index()

        # Flatten column names
        bin_stats_2d.columns = ['DEFOCUS_BIN', 'ASTIGMATISM_BIN', 'MEAN_PEAK_HEIGHT', 'COUNT']

        # Get bin centers
        bin_stats_2d['DEFOCUS_CENTER'] = bin_stats_2d['DEFOCUS_BIN'].apply(lambda x: x.mid)
        bin_stats_2d['ASTIGMATISM_CENTER'] = bin_stats_2d['ASTIGMATISM_BIN'].apply(lambda x: x.mid)

        # Create pivot table for heatmap
        heatmap_data = bin_stats_2d.pivot_table(
            values='MEAN_PEAK_HEIGHT',
            index='ASTIGMATISM_CENTER',
            columns='DEFOCUS_CENTER'
        )

        # Create 2D histogram plot
        plt.figure(figsize=(14, 8))
        im = plt.imshow(heatmap_data, aspect='auto', origin='lower', cmap='RdBu_r')

        # Set axis labels with actual values
        plt.xlabel('Total Defocus (Angstroms)', fontsize=12)
        plt.ylabel('Astigmatism (Angstroms)', fontsize=12)
        plt.title(f'Average Peak Height by Defocus and Astigmatism (Job {job_id})', fontsize=14)

        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Average Peak Height (SNR)', fontsize=11)

        # Set tick labels to show actual defocus/astigmatism values
        x_ticks = range(0, len(heatmap_data.columns), max(1, len(heatmap_data.columns)//10))
        y_ticks = range(0, len(heatmap_data.index), max(1, len(heatmap_data.index)//10))

        plt.xticks(x_ticks, [f'{heatmap_data.columns[i]:.0f}' for i in x_ticks], rotation=45, ha='right')
        plt.yticks(y_ticks, [f'{heatmap_data.index[i]:.0f}' for i in y_ticks])

        # Save plot
        output_file = os.path.join(output_dir, f'peak_height_2d_histogram_job{job_id}.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        # Create second 2D histogram: Defocus Angle vs Astigmatism
        peaks_with_metadata['ANGLE_BIN'] = pd.cut(
            peaks_with_metadata['USED_DEFOCUS_ANGLE'],
            bins=n_bins_x
        )

        # Calculate average peak height per 2D bin
        bin_stats_angle = peaks_with_metadata.groupby(
            ['ANGLE_BIN', 'ASTIGMATISM_BIN'],
            observed=True
        ).agg({
            'PEAK_HEIGHT': ['mean', 'count']
        }).reset_index()

        # Flatten column names
        bin_stats_angle.columns = ['ANGLE_BIN', 'ASTIGMATISM_BIN', 'MEAN_PEAK_HEIGHT', 'COUNT']

        # Get bin centers
        bin_stats_angle['ANGLE_CENTER'] = bin_stats_angle['ANGLE_BIN'].apply(lambda x: x.mid)
        bin_stats_angle['ASTIGMATISM_CENTER'] = bin_stats_angle['ASTIGMATISM_BIN'].apply(lambda x: x.mid)

        # Create pivot table for heatmap
        heatmap_angle = bin_stats_angle.pivot_table(
            values='MEAN_PEAK_HEIGHT',
            index='ASTIGMATISM_CENTER',
            columns='ANGLE_CENTER'
        )

        # Create 2D histogram plot
        plt.figure(figsize=(14, 8))
        im = plt.imshow(heatmap_angle, aspect='auto', origin='lower', cmap='RdBu_r')

        # Set axis labels
        plt.xlabel('Defocus Angle (degrees)', fontsize=12)
        plt.ylabel('Astigmatism (Angstroms)', fontsize=12)
        plt.title(f'Average Peak Height by Defocus Angle and Astigmatism (Job {job_id})', fontsize=14)

        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Average Peak Height (SNR)', fontsize=11)

        # Set tick labels
        x_ticks_angle = range(0, len(heatmap_angle.columns), max(1, len(heatmap_angle.columns)//10))
        y_ticks_angle = range(0, len(heatmap_angle.index), max(1, len(heatmap_angle.index)//10))

        plt.xticks(x_ticks_angle, [f'{heatmap_angle.columns[i]:.0f}' for i in x_ticks_angle], rotation=45, ha='right')
        plt.yticks(y_ticks_angle, [f'{heatmap_angle.index[i]:.0f}' for i in y_ticks_angle])

        # Save plot
        output_file_angle = os.path.join(output_dir, f'peak_height_angle_astig_job{job_id}.png')
        plt.savefig(output_file_angle, dpi=150, bbox_inches='tight')
        plt.close()

        return (output_file, output_file_angle)
