#!/usr/bin/env python3
"""
find_fragile_code.py

Identify high-risk files by correlating:
- Low test coverage
- High bug-fix frequency
- High change frequency (churn)

This script helps prioritize which test gaps to address first.

Usage:
    python3 find_fragile_code.py <coverage_file> [options]

Arguments:
    coverage_file    Path to coverage report (LCOV .info or Cobertura XML)

Options:
    --since PERIOD   Time period for git analysis (default: "6 months ago")
    --format FORMAT  Output format: table, csv, json (default: table)
    --top N          Show top N files (default: 20)
    --threshold N    Minimum risk score to display (default: 50)

Examples:
    python3 find_fragile_code.py coverage.info
    python3 find_fragile_code.py coverage.xml --since "3 months ago" --top 30
    python3 find_fragile_code.py coverage.info --format csv > risk_report.csv
"""

import sys
import subprocess
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def parse_lcov_coverage(coverage_file: str) -> Dict[str, float]:
    """Parse LCOV .info file to extract coverage percentages."""
    coverage = {}

    try:
        result = subprocess.run(
            ['lcov', '--list', coverage_file],
            capture_output=True,
            text=True,
            check=True
        )

        for line in result.stdout.split('\n'):
            parts = line.split()
            if len(parts) >= 2 and '%' in parts[1]:
                filename = parts[0]
                coverage_pct = float(parts[1].rstrip('%'))
                coverage[filename] = coverage_pct

    except subprocess.CalledProcessError:
        print("Error: Failed to parse LCOV coverage file", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: lcov command not found. Install with: sudo apt-get install lcov", file=sys.stderr)
        sys.exit(1)

    return coverage


def parse_xml_coverage(coverage_file: str) -> Dict[str, float]:
    """Parse Cobertura XML coverage file."""
    try:
        import xml.etree.ElementTree as ET
    except ImportError:
        print("Error: xml.etree.ElementTree not available", file=sys.stderr)
        sys.exit(1)

    coverage = {}

    try:
        tree = ET.parse(coverage_file)
        root = tree.getroot()

        # Cobertura format
        for package in root.findall('.//package'):
            for cls in package.findall('.//class'):
                filename = cls.get('filename')
                line_rate = float(cls.get('line-rate', 0))
                coverage[filename] = line_rate * 100

    except ET.ParseError:
        print(f"Error: Failed to parse XML coverage file: {coverage_file}", file=sys.stderr)
        sys.exit(1)

    return coverage


def get_coverage_data(coverage_file: str) -> Dict[str, float]:
    """Detect format and parse coverage file."""
    if coverage_file.endswith('.info'):
        return parse_lcov_coverage(coverage_file)
    elif coverage_file.endswith('.xml'):
        return parse_xml_coverage(coverage_file)
    else:
        print("Error: Unsupported coverage file format. Use .info (LCOV) or .xml (Cobertura)", file=sys.stderr)
        sys.exit(1)


def get_bug_fix_count(filepath: str, since: str = "6 months ago") -> int:
    """Count bug-fix commits for a file."""
    try:
        # Find commits with bug-related messages
        result = subprocess.run(
            ['git', 'log', '--no-merges', f'--since={since}',
             '--format=%H %s', '--', filepath],
            capture_output=True,
            text=True,
            check=True
        )

        bug_commits = [
            line for line in result.stdout.split('\n')
            if any(word in line.lower() for word in ['fix', 'bug', 'issue', 'defect'])
        ]

        return len(bug_commits)

    except subprocess.CalledProcessError:
        return 0


def get_change_frequency(filepath: str, since: str = "6 months ago") -> int:
    """Get number of times file was modified."""
    try:
        result = subprocess.run(
            ['git', 'log', '--no-merges', f'--since={since}',
             '--format=%H', '--', filepath],
            capture_output=True,
            text=True,
            check=True
        )
        commits = [c for c in result.stdout.split('\n') if c]
        return len(commits)

    except subprocess.CalledProcessError:
        return 0


def get_file_complexity(filepath: str) -> int:
    """Estimate complexity (lines of code as proxy)."""
    try:
        if Path(filepath).exists():
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return len(f.readlines())
    except:
        pass
    return 0


def calculate_risk_score(coverage_pct: float, bug_count: int, churn: int, loc: int) -> float:
    """
    Calculate risk score based on multiple factors.

    Formula:
        Risk = (bug_count × 20) + (churn × 10) + ((100 - coverage) × 2) + (loc / 100)

    Weights:
        - Bug count: 20 (most important - actual problems)
        - Churn: 10 (instability indicator)
        - Coverage gap: 2 (missing tests)
        - LOC: 0.01 (size/complexity proxy)
    """
    coverage_gap = 100 - coverage_pct
    risk = (bug_count * 20) + (churn * 10) + (coverage_gap * 2) + (loc / 100)
    return risk


def analyze_files(coverage_data: Dict[str, float], since: str) -> List[Tuple[str, Dict]]:
    """Analyze all files and calculate risk scores."""
    results = []

    print("Analyzing files...", file=sys.stderr)

    for filepath, coverage_pct in coverage_data.items():
        # Skip if coverage is already high (>80%)
        if coverage_pct > 80:
            continue

        # Get git metrics
        bug_count = get_bug_fix_count(filepath, since)
        churn = get_change_frequency(filepath, since)
        loc = get_file_complexity(filepath)

        # Calculate risk
        risk_score = calculate_risk_score(coverage_pct, bug_count, churn, loc)

        results.append((filepath, {
            'coverage': coverage_pct,
            'bugs': bug_count,
            'churn': churn,
            'loc': loc,
            'risk_score': risk_score
        }))

    # Sort by risk score descending
    results.sort(key=lambda x: x[1]['risk_score'], reverse=True)

    return results


def format_table(results: List[Tuple[str, Dict]], top_n: int, threshold: float) -> None:
    """Print results as formatted table."""
    print()
    print("Risk Score | Coverage | Bugs | Churn | LOC  | File")
    print("-----------|----------|------|-------|------|" + "-" * 50)

    count = 0
    for filepath, metrics in results:
        if metrics['risk_score'] < threshold:
            continue

        if count >= top_n:
            break

        # Truncate long file paths
        display_path = filepath[-48:] if len(filepath) > 48 else filepath
        if len(filepath) > 48:
            display_path = "..." + display_path

        print(f"{metrics['risk_score']:>10.1f} | "
              f"{metrics['coverage']:>7.1f}% | "
              f"{metrics['bugs']:>4d} | "
              f"{metrics['churn']:>5d} | "
              f"{metrics['loc']:>4d} | "
              f"{display_path}")

        count += 1

    print()
    print(f"Showing top {count} files with risk score ≥ {threshold}")
    print()


def format_csv(results: List[Tuple[str, Dict]], top_n: int, threshold: float) -> None:
    """Print results as CSV."""
    print("File,Risk_Score,Coverage,Bugs,Churn,LOC")

    count = 0
    for filepath, metrics in results:
        if metrics['risk_score'] < threshold:
            continue

        if count >= top_n:
            break

        print(f"{filepath},"
              f"{metrics['risk_score']:.1f},"
              f"{metrics['coverage']:.1f},"
              f"{metrics['bugs']},"
              f"{metrics['churn']},"
              f"{metrics['loc']}")

        count += 1


def format_json(results: List[Tuple[str, Dict]], top_n: int, threshold: float) -> None:
    """Print results as JSON."""
    output = []

    count = 0
    for filepath, metrics in results:
        if metrics['risk_score'] < threshold:
            continue

        if count >= top_n:
            break

        output.append({
            'file': filepath,
            'risk_score': round(metrics['risk_score'], 1),
            'coverage': round(metrics['coverage'], 1),
            'bugs': metrics['bugs'],
            'churn': metrics['churn'],
            'loc': metrics['loc']
        })

        count += 1

    print(json.dumps(output, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description='Identify fragile code with low coverage and high bug/churn rates',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s coverage.info
  %(prog)s coverage.xml --since "3 months ago" --top 30
  %(prog)s coverage.info --format csv > risk_report.csv
        """
    )

    parser.add_argument('coverage_file', help='Coverage report file (.info or .xml)')
    parser.add_argument('--since', default='6 months ago', help='Time period for git analysis')
    parser.add_argument('--format', choices=['table', 'csv', 'json'], default='table',
                        help='Output format')
    parser.add_argument('--top', type=int, default=20, help='Number of top files to show')
    parser.add_argument('--threshold', type=float, default=50.0,
                        help='Minimum risk score to display')

    args = parser.parse_args()

    # Verify coverage file exists
    if not Path(args.coverage_file).exists():
        print(f"Error: Coverage file not found: {args.coverage_file}", file=sys.stderr)
        sys.exit(1)

    # Verify we're in a git repository
    try:
        subprocess.run(['git', 'rev-parse', '--is-inside-work-tree'],
                       capture_output=True, check=True)
    except subprocess.CalledProcessError:
        print("Error: Not in a git repository", file=sys.stderr)
        sys.exit(1)

    # Parse coverage data
    if args.format == 'table':
        print(f"Parsing coverage data from {args.coverage_file}...", file=sys.stderr)
    coverage_data = get_coverage_data(args.coverage_file)

    if not coverage_data:
        print("Error: No coverage data found", file=sys.stderr)
        sys.exit(1)

    if args.format == 'table':
        print(f"Found {len(coverage_data)} files with coverage data", file=sys.stderr)
        print(f"Analyzing git history since: {args.since}", file=sys.stderr)

    # Analyze files
    results = analyze_files(coverage_data, args.since)

    # Output results
    if args.format == 'table':
        format_table(results, args.top, args.threshold)
    elif args.format == 'csv':
        format_csv(results, args.top, args.threshold)
    elif args.format == 'json':
        format_json(results, args.top, args.threshold)


if __name__ == '__main__':
    main()
