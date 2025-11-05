#!/bin/bash
# identify_hotspots.sh - Complete hotspot analysis (churn + complexity)
#
# Usage: ./identify_hotspots.sh [repo_path] [start_date]
#
# Arguments:
#   repo_path   - Path to git repository (default: current directory)
#   start_date  - Analysis start date (default: 12.month.ago)
#
# Requirements:
#   - git
#   - lizard (install: pip install lizard)
#   - python3 + pandas (install: pip install pandas)
#
# Example:
#   ./identify_hotspots.sh /path/to/repo 2024-01-01
#   ./identify_hotspots.sh . 6.month.ago

set -e  # Exit on error

# === Configuration ===
REPO_PATH="${1:-.}"
START_DATE="${2:-12.month.ago}"
OUTPUT_DIR="hotspot_analysis_$(date +%Y%m%d_%H%M%S)"

# === Colors ===
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# === Functions ===
print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        print_error "$1 is not installed"
        echo "Install: $2"
        exit 1
    fi
}

# === Validation ===
if [ ! -d "$REPO_PATH/.git" ]; then
    print_error "Not a git repository: $REPO_PATH"
    exit 1
fi

check_command "lizard" "pip install lizard"
check_command "python3" "Install Python 3"

# Check for pandas
if ! python3 -c "import pandas" 2>/dev/null; then
    print_error "pandas is not installed"
    echo "Install: pip install pandas"
    exit 1
fi

# === Setup ===
mkdir -p "$OUTPUT_DIR"
cd "$REPO_PATH"

print_header "Hotspot Analysis (Churn + Complexity)"
echo "Repository: $REPO_PATH"
echo "Analysis period: $START_DATE to present"
echo "Output directory: $OUTPUT_DIR"
echo ""

# === 1. Calculate Churn ===
print_header "Step 1/4: Calculating code churn..."

git log --format=format: --name-only --since="$START_DATE" \
  | egrep -v '^$' \
  | egrep -v '\.(md|txt|json|xml|yml|yaml)$' \
  | egrep -v '^test/' \
  | egrep -v '^tests/' \
  | egrep -v '^docs/' \
  | sort \
  | uniq -c \
  | sort -nr \
  | head -100 \
  > "$OUTPUT_DIR/churn_raw.txt"

# Convert to CSV
echo "file,commits" > "$OUTPUT_DIR/churn.csv"
cat "$OUTPUT_DIR/churn_raw.txt" \
  | awk '{print $2","$1}' \
  >> "$OUTPUT_DIR/churn.csv"

CHURN_FILES=$(wc -l < "$OUTPUT_DIR/churn.csv")
print_success "Analyzed churn for $((CHURN_FILES - 1)) files"

# === 2. Calculate Complexity ===
print_header "Step 2/4: Calculating code complexity..."

# Determine source directories (customize for your project)
SRC_DIRS=$(find . -type d -name "src" -o -name "lib" -o -name "app" 2>/dev/null | head -1)
if [ -z "$SRC_DIRS" ]; then
    SRC_DIRS="."
fi

lizard --csv "$SRC_DIRS" > "$OUTPUT_DIR/complexity_raw.csv" 2>/dev/null || {
    print_error "Lizard failed. Trying current directory..."
    lizard --csv . > "$OUTPUT_DIR/complexity_raw.csv" 2>/dev/null
}

# Clean up complexity CSV (lizard adds header comments)
grep -v "^#" "$OUTPUT_DIR/complexity_raw.csv" > "$OUTPUT_DIR/complexity.csv"

COMPLEXITY_FILES=$(tail -n +2 "$OUTPUT_DIR/complexity.csv" | wc -l)
print_success "Analyzed complexity for $COMPLEXITY_FILES files"

# === 3. Join Churn + Complexity ===
print_header "Step 3/4: Combining churn and complexity data..."

python3 << 'PYTHON_SCRIPT'
import pandas as pd
import sys
import os

try:
    # Load data
    churn = pd.read_csv('OUTPUT_DIR/churn.csv')
    complexity = pd.read_csv('OUTPUT_DIR/complexity.csv')

    # Normalize paths (remove leading ./)
    churn['file'] = churn['file'].str.replace('^\./', '', regex=True)
    complexity['file'] = complexity['file'].str.replace('^\./', '', regex=True)

    # Join on file path
    hotspots = pd.merge(
        churn,
        complexity[['file', 'NLOC', 'CCN']],
        on='file',
        how='inner'
    )

    # Handle missing values
    hotspots['NLOC'] = hotspots['NLOC'].fillna(0)
    hotspots['CCN'] = hotspots['CCN'].fillna(1)

    # Calculate hotspot score (CCN × commits)
    hotspots['hotspot_score'] = hotspots['CCN'] * hotspots['commits']

    # Sort by score
    hotspots = hotspots.sort_values('hotspot_score', ascending=False)

    # Save
    hotspots.to_csv('OUTPUT_DIR/hotspots.csv', index=False)

    # Summary statistics
    print(f"Successfully joined {len(hotspots)} files")
    print(f"\nTop 20 Hotspots:")
    print(hotspots[['file', 'commits', 'CCN', 'NLOC', 'hotspot_score']].head(20).to_string(index=False))

    # Save summary
    with open('OUTPUT_DIR/hotspot_summary.txt', 'w') as f:
        f.write("=== Hotspot Analysis Summary ===\n\n")
        f.write(f"Total files analyzed: {len(hotspots)}\n\n")
        f.write("Top 20 Hotspots:\n")
        f.write(hotspots[['file', 'commits', 'CCN', 'NLOC', 'hotspot_score']].head(20).to_string(index=False))
        f.write("\n\n=== Thresholds ===\n")

        # Count by severity
        critical = len(hotspots[(hotspots['CCN'] > 50) & (hotspots['commits'] > 50)])
        high = len(hotspots[(hotspots['CCN'] > 20) & (hotspots['commits'] > 20)])
        medium = len(hotspots[(hotspots['CCN'] > 10) & (hotspots['commits'] > 10)])

        f.write(f"Critical hotspots (CCN>50, commits>50): {critical}\n")
        f.write(f"High hotspots (CCN>20, commits>20): {high}\n")
        f.write(f"Medium hotspots (CCN>10, commits>10): {medium}\n")

except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)

PYTHON_SCRIPT

# Replace OUTPUT_DIR placeholder in Python script
python3 << PYTHON_SCRIPT
import pandas as pd
import sys

try:
    # Load data
    churn = pd.read_csv('${OUTPUT_DIR}/churn.csv')
    complexity = pd.read_csv('${OUTPUT_DIR}/complexity.csv')

    # Normalize paths
    churn['file'] = churn['file'].str.replace('^\./', '', regex=True)
    complexity['file'] = complexity['file'].str.replace('^\./', '', regex=True)

    # Join
    hotspots = pd.merge(
        churn,
        complexity[['file', 'NLOC', 'CCN']],
        on='file',
        how='inner'
    )

    # Handle missing
    hotspots['NLOC'] = hotspots['NLOC'].fillna(0)
    hotspots['CCN'] = hotspots['CCN'].fillna(1)

    # Score
    hotspots['hotspot_score'] = hotspots['CCN'] * hotspots['commits']
    hotspots = hotspots.sort_values('hotspot_score', ascending=False)

    # Save
    hotspots.to_csv('${OUTPUT_DIR}/hotspots.csv', index=False)

    print(f"Successfully joined {len(hotspots)} files")

    # Save summary
    with open('${OUTPUT_DIR}/hotspot_summary.txt', 'w') as f:
        f.write("=== Hotspot Analysis Summary ===\n\n")
        f.write(f"Total files analyzed: {len(hotspots)}\n\n")
        f.write("Top 20 Hotspots:\n")
        f.write(hotspots[['file', 'commits', 'CCN', 'NLOC', 'hotspot_score']].head(20).to_string(index=False))
        f.write("\n\n=== Severity Classification ===\n")

        critical = len(hotspots[(hotspots['CCN'] > 50) & (hotspots['commits'] > 50)])
        high = len(hotspots[(hotspots['CCN'] > 20) & (hotspots['commits'] > 20)])
        medium = len(hotspots[(hotspots['CCN'] > 10) & (hotspots['commits'] > 10)])

        f.write(f"Critical hotspots (CCN>50, commits>50): {critical}\n")
        f.write(f"High hotspots (CCN>20, commits>20): {high}\n")
        f.write(f"Medium hotspots (CCN>10, commits>10): {medium}\n")

    print("\nTop 20 Hotspots:")
    print(hotspots[['file', 'commits', 'CCN', 'NLOC', 'hotspot_score']].head(20).to_string(index=False))

except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
PYTHON_SCRIPT

print_success "Generated: $OUTPUT_DIR/hotspots.csv"

# === 4. Generate Summary Report ===
print_header "Step 4/4: Generating summary report..."

{
    echo "=== Hotspot Analysis Report ==="
    echo "Date: $(date)"
    echo "Repository: $REPO_PATH"
    echo "Analysis period: $START_DATE to present"
    echo ""
    echo "Files analyzed:"
    echo "  - Churn: $((CHURN_FILES - 1)) files"
    echo "  - Complexity: $COMPLEXITY_FILES files"
    echo ""
    cat "$OUTPUT_DIR/hotspot_summary.txt" 2>/dev/null || echo "Summary generation failed"
} > "$OUTPUT_DIR/report.txt"

print_success "Generated: $OUTPUT_DIR/report.txt"

# === Completion ===
print_header "Analysis Complete"
echo ""
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Key files:"
echo "  - hotspots.csv           : Complete hotspot data (import to spreadsheet)"
echo "  - report.txt             : Executive summary"
echo "  - churn.csv              : Raw churn data"
echo "  - complexity.csv         : Raw complexity data"
echo ""
print_success "Next steps:"
echo "  1. Review report.txt for top hotspots"
echo "  2. Import hotspots.csv into Google Sheets/Excel"
echo "  3. Create scatter plot (X=commits, Y=CCN)"
echo "  4. Investigate top-right quadrant files"
echo "  5. Check for temporal coupling: grep 'hotspot_file' coupling.csv"
echo ""
print_warning "For visualization, consider:"
echo "  - Google Sheets: Import hotspots.csv, create scatter chart"
echo "  - Python: Use matplotlib (see practical_workflow.md)"
