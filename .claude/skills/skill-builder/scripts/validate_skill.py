#!/usr/bin/env python3
"""
Skill Validator - Checks REQUIREMENTS, not guidelines

Validates YAML frontmatter constraints and skill structure for cisTEMx skills.

VALIDATES (errors):
- YAML syntax validity
- Required frontmatter fields (name, description)
- Frontmatter character limit (1024 max)
- Directory structure exists

SHOULD NOT VALIDATE (warnings at most):
- Line count recommendations (200-line guideline)
- Organizational suggestions
- Style preferences

Philosophy: Distinguish "must be true" from "best practices"
"""

import sys
import yaml
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class SkillValidator:
    """Validates skill files against Anthropic constraints."""

    # Anthropic's constraints from documentation
    MAX_NAME_LENGTH = 64
    MAX_DESCRIPTION_LENGTH = 1024
    ALLOWED_YAML_FIELDS = {'name', 'description', 'version', 'author', 'tags'}

    def __init__(self, skill_path: Path):
        """Initialize validator with skill directory path."""
        self.skill_path = Path(skill_path)
        self.skill_file = self.skill_path / 'SKILL.md'
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate(self) -> bool:
        """Run all validation checks."""
        if not self.skill_file.exists():
            self.errors.append(f"SKILL.md not found at {self.skill_file}")
            return False

        # Read the skill file
        with open(self.skill_file, 'r') as f:
            content = f.read()

        # Extract and validate YAML frontmatter
        frontmatter = self._extract_frontmatter(content)
        if frontmatter:
            self._validate_frontmatter(frontmatter)

        # Validate file structure
        self._validate_structure(content)

        # Validate references
        self._validate_references(content)

        # Check skill size
        self._check_size_constraints(content)

        return len(self.errors) == 0

    def _extract_frontmatter(self, content: str) -> Optional[Dict]:
        """Extract YAML frontmatter from content."""
        match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
        if not match:
            self.errors.append("No YAML frontmatter found")
            return None

        try:
            return yaml.safe_load(match.group(1))
        except yaml.YAMLError as e:
            self.errors.append(f"Invalid YAML frontmatter: {e}")
            return None

    def _validate_frontmatter(self, fm: Dict):
        """Validate frontmatter against Anthropic constraints."""
        # Check required fields
        if 'name' not in fm:
            self.errors.append("Missing required field: name")
        elif len(fm['name']) > self.MAX_NAME_LENGTH:
            self.errors.append(f"Name exceeds {self.MAX_NAME_LENGTH} characters: {len(fm['name'])}")

        if 'description' not in fm:
            self.errors.append("Missing required field: description")
        elif len(fm['description']) > self.MAX_DESCRIPTION_LENGTH:
            self.errors.append(f"Description exceeds {self.MAX_DESCRIPTION_LENGTH} characters: {len(fm['description'])}")

        # Check for disallowed fields
        extra_fields = set(fm.keys()) - self.ALLOWED_YAML_FIELDS
        if extra_fields:
            self.errors.append(f"Disallowed YAML fields: {', '.join(extra_fields)}")

        # Validate name format (letters, numbers, hyphens, underscores only)
        if 'name' in fm and not re.match(r'^[a-z0-9_-]+$', fm['name']):
            self.errors.append(f"Name contains invalid characters. Use only: a-z, 0-9, -, _")

    def _validate_structure(self, content: str):
        """Check skill file structure."""
        lines = content.split('\n')

        # Check for main heading after frontmatter
        in_frontmatter = False
        content_start = False
        for line in lines:
            if line == '---':
                if not in_frontmatter:
                    in_frontmatter = True
                else:
                    content_start = True
                    in_frontmatter = False
            elif content_start and line.startswith('# '):
                break
        else:
            self.warnings.append("No main heading (# Title) found after frontmatter")

        # Check line count (warning if over 200)
        if len(lines) > 200:
            self.warnings.append(f"SKILL.md has {len(lines)} lines. Consider delegating more to resources/")

    def _validate_references(self, content: str):
        """Check that referenced files exist."""
        # Find all references to resources/ and templates/
        resource_pattern = r'(?:resources|templates)/[\w/-]+\.(?:md|py|ya?ml)'
        references = re.findall(resource_pattern, content)

        for ref in references:
            ref_path = self.skill_path / ref
            if not ref_path.exists():
                self.errors.append(f"Referenced file not found: {ref}")

    def _check_size_constraints(self, content: str):
        """Check size-related constraints."""
        # Check total size
        size_kb = len(content.encode('utf-8')) / 1024
        if size_kb > 10:
            self.warnings.append(f"SKILL.md is {size_kb:.1f}KB. Consider more aggressive delegation")

        # Check resources directory
        resources_dir = self.skill_path / 'resources'
        if resources_dir.exists():
            resource_files = list(resources_dir.glob('*.md'))
            if len(resource_files) > 10:
                self.warnings.append(f"Resources directory has {len(resource_files)} files. Consider consolidation")

    def print_report(self):
        """Print validation report."""
        print(f"\n🔍 Validating skill: {self.skill_path.name}")
        print("=" * 60)

        if self.errors:
            print("\n❌ ERRORS (must fix):")
            for error in self.errors:
                print(f"  • {error}")

        if self.warnings:
            print("\n⚠️  WARNINGS (should consider):")
            for warning in self.warnings:
                print(f"  • {warning}")

        if not self.errors and not self.warnings:
            print("\n✅ All checks passed!")

        print("\n" + "=" * 60)
        return len(self.errors) == 0

def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python validate_skill.py <skill_directory>")
        print("Example: python validate_skill.py /path/to/skill")
        sys.exit(1)

    skill_path = Path(sys.argv[1])
    if not skill_path.exists():
        print(f"Error: Path does not exist: {skill_path}")
        sys.exit(1)

    validator = SkillValidator(skill_path)
    validator.validate()
    success = validator.print_report()

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()