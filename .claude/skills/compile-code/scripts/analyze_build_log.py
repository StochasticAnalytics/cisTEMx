#!/usr/bin/env python3
"""
Analyze build log files for errors and generate concise summaries.

Searches for compilation and linking errors, extracting file:line:error
information for efficient troubleshooting.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple


class ErrorEntry:
    """Represents a single error found in the build log."""
    
    def __init__(self, line_num: int, content: str, error_type: str):
        self.line_num = line_num
        self.content = content.strip()
        self.error_type = error_type  # "compile" or "link"
    
    def __str__(self) -> str:
        return f"Line {self.line_num}: {self.content}"


def find_errors(log_file: Path) -> Tuple[List[ErrorEntry], List[ErrorEntry]]:
    """
    Scan log file for compilation and linking errors.
    
    Args:
        log_file: Path to build log file
    
    Returns:
        Tuple of (compile_errors, link_errors)
    """
    compile_errors = []
    link_errors = []
    
    # Patterns to match
    # "error" must be case-sensitive per requirements
    error_pattern = re.compile(r'\berror\b')  # Word boundary ensures "error" not "Error"
    undefined_ref_pattern = re.compile(r'undefined reference', re.IGNORECASE)
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
            for line_num, line in enumerate(f, 1):
                # Check for undefined reference (linking error)
                if undefined_ref_pattern.search(line):
                    link_errors.append(ErrorEntry(line_num, line, "link"))
                # Check for "error" (case-sensitive, compilation error)
                elif error_pattern.search(line):
                    compile_errors.append(ErrorEntry(line_num, line, "compile"))
    
    except FileNotFoundError:
        print(f"Error: Log file not found: {log_file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading log file: {e}", file=sys.stderr)
        sys.exit(1)
    
    return compile_errors, link_errors


def generate_summary(
    log_file: Path,
    compile_errors: List[ErrorEntry],
    link_errors: List[ErrorEntry]
) -> str:
    """
    Generate human-readable error summary.
    
    Args:
        log_file: Path to log file
        compile_errors: List of compilation errors
        link_errors: List of linking errors
    
    Returns:
        Formatted summary string
    """
    summary_lines = []
    summary_lines.append(f"BUILD LOG ANALYSIS: {log_file.name}")
    summary_lines.append("=" * 70)
    summary_lines.append("")
    
    total_errors = len(compile_errors) + len(link_errors)
    
    if total_errors == 0:
        summary_lines.append("✓ No errors found in build log")
        summary_lines.append("")
        summary_lines.append("Note: This analysis searches for:")
        summary_lines.append('  - "error" (case-sensitive)')
        summary_lines.append('  - "undefined reference" (case-insensitive)')
        summary_lines.append("")
        summary_lines.append("If build failed for other reasons, review the full log.")
    else:
        summary_lines.append(f"✗ ERRORS FOUND: {total_errors}")
        summary_lines.append("")
        
        # Compilation errors
        if compile_errors:
            summary_lines.append(f"COMPILATION ERRORS: {len(compile_errors)}")
            summary_lines.append("-" * 70)
            for i, error in enumerate(compile_errors[:20], 1):  # Limit to first 20
                summary_lines.append(f"{i}. {error}")
            
            if len(compile_errors) > 20:
                summary_lines.append(f"... and {len(compile_errors) - 20} more")
            summary_lines.append("")
        
        # Linking errors
        if link_errors:
            summary_lines.append(f"LINKING ERRORS: {len(link_errors)}")
            summary_lines.append("-" * 70)
            for i, error in enumerate(link_errors[:20], 1):  # Limit to first 20
                summary_lines.append(f"{i}. {error}")
            
            if len(link_errors) > 20:
                summary_lines.append(f"... and {len(link_errors) - 20} more")
            summary_lines.append("")
    
    summary_lines.append("=" * 70)
    summary_lines.append(f"Full log: {log_file}")
    summary_lines.append("")
    summary_lines.append("To view specific errors in context:")
    summary_lines.append(f"  sed -n '<line_number>p' {log_file}")
    summary_lines.append("Or use Read tool with offset/limit parameters")
    
    return "\n".join(summary_lines)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze build log for compilation and linking errors'
    )
    parser.add_argument(
        'log_file',
        type=Path,
        help='Path to build log file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Write summary to file instead of stdout'
    )
    
    args = parser.parse_args()
    
    # Analyze log
    compile_errors, link_errors = find_errors(args.log_file)
    
    # Generate summary
    summary = generate_summary(args.log_file, compile_errors, link_errors)
    
    # Output summary
    if args.output:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, 'w') as f:
                f.write(summary)
            print(f"Summary written to: {args.output}")
        except Exception as e:
            print(f"Error writing summary file: {e}", file=sys.stderr)
            return 1
    else:
        print(summary)
    
    # Return exit code based on error count
    return 1 if (compile_errors or link_errors) else 0


if __name__ == "__main__":
    sys.exit(main())
