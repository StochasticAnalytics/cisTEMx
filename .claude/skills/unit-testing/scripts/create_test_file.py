#!/usr/bin/env python3
"""Create a test file from template

This utility helps create new test files from templates for different
languages/frameworks. It handles:
- Template selection based on file type
- Placeholder substitution
- Test file placement following project conventions

Usage:
    python create_test_file.py --source src/core/matrix.cpp --output src/test/core/test_matrix.cpp
    python create_test_file.py --source scripts/build.sh --output scripts/tests/test_build.bats
    python create_test_file.py --source .claude/skills/foo/scripts/bar.py --output .claude/skills/foo/tests/test_bar.py
"""

import argparse
import sys
from pathlib import Path

# Add .claude directory to path for imports
claude_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(claude_dir))

try:
    from scripts.common.path_validation import (
        validate_path_within_project,
        find_git_root
    )
    HAS_VALIDATION = True
except ImportError:
    print("Warning: Could not import path validation utilities", file=sys.stderr)
    print("Falling back to basic path validation.", file=sys.stderr)
    HAS_VALIDATION = False


class ValidationError(Exception):
    """Path validation error"""
    pass


# Template directory
TEMPLATE_DIR = Path(__file__).parent.parent / "templates"

# Template mapping
TEMPLATES = {
    ".cpp": "catch2_test_template.cpp",
    ".py": "pytest_test_template.py",
    ".bats": "bats_test_template.bats",
    ".cu": "cuda_test_template.cpp",
    ".cuh": "cuda_test_template.cpp",
}


def detect_template(output_file: Path) -> Path:
    """Detect which template to use based on file extension

    Args:
        output_file: Path to the output test file

    Returns:
        Path to the template file

    Raises:
        ValueError: If no template found for file extension
    """
    ext = output_file.suffix

    if ext not in TEMPLATES:
        raise ValueError(
            f"No template found for extension '{ext}'. "
            f"Supported extensions: {', '.join(TEMPLATES.keys())}"
        )

    template_file = TEMPLATE_DIR / TEMPLATES[ext]

    if not template_file.exists():
        raise FileNotFoundError(f"Template file not found: {template_file}")

    return template_file


def extract_component_name(source_file: Path) -> str:
    """Extract component name from source file

    Examples:
        matrix.cpp -> Matrix
        job_packager.cpp -> JobPackager
        build_helper.sh -> BuildHelper

    Args:
        source_file: Path to source file

    Returns:
        Component name in PascalCase
    """
    # Get stem without extension
    stem = source_file.stem

    # Remove test_ prefix if present
    if stem.startswith("test_"):
        stem = stem[5:]

    # Convert snake_case to PascalCase
    parts = stem.split('_')
    return ''.join(word.capitalize() for word in parts)


def generate_test_file(template_file: Path, source_file: Path, output_file: Path) -> str:
    """Generate test file content from template

    Args:
        template_file: Path to template
        source_file: Path to source file being tested
        output_file: Path to output test file

    Returns:
        Generated test file content
    """
    # Read template
    with open(template_file, 'r') as f:
        content = f.read()

    # Extract component information
    component_name = extract_component_name(source_file)
    component_lower = source_file.stem

    # Basic substitutions
    replacements = {
        "COMPONENT_NAME": component_name,
        "COMPONENT": component_lower,
        "MODULE_NAME": component_lower,
        "SCRIPT_NAME": component_lower,
        "YEAR": "2025",
        # Placeholder features - user should fill in
        "FEATURE_1": "Feature 1 description (replace with actual feature)",
        "FEATURE_2": "Feature 2 description (replace with actual feature)",
        "ARGUMENT_NAME": "argument (replace with actual argument name)",
    }

    for placeholder, replacement in replacements.items():
        content = content.replace(placeholder, replacement)

    return content


def main():
    parser = argparse.ArgumentParser(
        description="Create a test file from template",
        epilog="Examples:\n"
               "  %(prog)s --source src/core/matrix.cpp --output src/test/core/test_matrix.cpp\n"
               "  %(prog)s --source scripts/build.sh --output scripts/tests/test_build.bats\n",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--source",
        type=str,
        help="Path to source file being tested (for component name extraction)"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output test file"
    )

    parser.add_argument(
        "--template",
        type=str,
        help="Template to use (auto-detected from output extension if not specified)"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing file"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be created without creating it"
    )

    args = parser.parse_args()

    try:
        # Paths
        output_path = Path(args.output)
        source_path = Path(args.source) if args.source else output_path

        # Validate output path
        if HAS_VALIDATION:
            git_root = find_git_root()
            if git_root:
                valid, error = validate_path_within_project(
                    output_path, git_root, "Output file"
                )
                if not valid:
                    print(error, file=sys.stderr)
                    return 1
        else:
            # Basic fallback validation
            if not output_path.parent.exists():
                print(f"Error: Parent directory does not exist: {output_path.parent}", file=sys.stderr)
                return 1

        # Check if output exists
        if output_path.exists() and not args.force:
            print(f"Error: Output file already exists: {output_path}", file=sys.stderr)
            print("Use --force to overwrite", file=sys.stderr)
            return 1

        # Detect or use specified template
        if args.template:
            template_path = TEMPLATE_DIR / args.template
        else:
            template_path = detect_template(output_path)

        print(f"Source: {source_path}")
        print(f"Output: {output_path}")
        print(f"Template: {template_path.name}")

        # Generate content
        content = generate_test_file(template_path, source_path, output_path)

        if args.dry_run:
            print("\n--- Generated Content (Dry Run) ---\n")
            print(content)
            print(f"\nWould write to: {output_path}")
            return 0

        # Create parent directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file
        with open(output_path, 'w') as f:
            f.write(content)

        print(f"\nTest file created: {output_path}")
        print("\nNext steps:")
        print("1. Edit the generated file and replace placeholders")
        print("2. Add actual test cases")

        if output_path.suffix == ".cpp":
            print("3. Add to src/Makefile.am:")
            print(f"   unit_test_runner_SOURCES += {output_path}")
            print("4. Run: ./autogen.sh && ./configure && make unit_test_runner")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
