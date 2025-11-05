#!/bin/bash
# analyze_churn.sh - Quick code churn analysis for refactoring prioritization
#
# Usage: ./analyze_churn.sh [repo_path] [start_date] [top_n]
#
# Arguments:
#   repo_path   - Path to git repository (default: current directory)
#   start_date  - Analysis start date (default: 12.month.ago)
#   top_n       - Number of top files to display (default: 20)
#
# Example:
#   ./analyze_churn.sh /path/to/repo 2024-01-01 50
#   ./analyze_churn.sh . 6.month.ago 20

set -e  # Exit on error

# === Configuration ===
REPO_PATH="${1:-.}"
START_DATE="${2:-12.month.ago}"
TOP_N="${3:-20}"
OUTPUT_DIR="churn_analysis_$(date +%Y%m%d_%H%M%S)"

# === Colors for output ===
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# === Validation ===
if [ ! -d "$REPO_PATH/.git" ]; then
    print_error "Not a git repository: $REPO_PATH"
    exit 1
fi

# === Setup ===
mkdir -p "$OUTPUT_DIR"
cd "$REPO_PATH"

print_header "Code Churn Analysis"
echo "Repository: $REPO_PATH"
echo "Analysis period: $START_DATE to present"
echo "Top files: $TOP_N"
echo "Output directory: $OUTPUT_DIR"
echo ""

# === 1. Overall Churn (All Files) ===
print_header "Analyzing overall churn..."

git log --format=format: --name-only --since="$START_DATE" \
  | egrep -v '^$' \
  | sort \
  | uniq -c \
  | sort -nr \
  | head -"$TOP_N" \
  > "$OUTPUT_DIR/churn_all.txt"

print_success "Generated: $OUTPUT_DIR/churn_all.txt"

# === 2. Production Code Churn (Filtered) ===
print_header "Analyzing production code churn..."

git log --format=format: --name-only --since="$START_DATE" \
  | egrep -v '^$' \
  | egrep -v '\.(md|txt|json|xml|yml|yaml)$' \
  | egrep -v '^test/' \
  | egrep -v '^tests/' \
  | egrep -v '^docs/' \
  | egrep -v '^vendor/' \
  | egrep -v '^node_modules/' \
  | sort \
  | uniq -c \
  | sort -nr \
  | head -"$TOP_N" \
  > "$OUTPUT_DIR/churn_production.txt"

print_success "Generated: $OUTPUT_DIR/churn_production.txt"

# === 3. Export to CSV ===
print_header "Exporting to CSV format..."

echo "file,commits" > "$OUTPUT_DIR/churn.csv"
cat "$OUTPUT_DIR/churn_production.txt" \
  | awk '{print $2","$1}' \
  >> "$OUTPUT_DIR/churn.csv"

print_success "Generated: $OUTPUT_DIR/churn.csv"

# === 4. Summary Statistics ===
print_header "Summary Statistics"

TOTAL_COMMITS=$(git log --oneline --since="$START_DATE" | wc -l)
TOTAL_FILES=$(git log --format=format: --name-only --since="$START_DATE" | egrep -v '^$' | sort -u | wc -l)
TOTAL_AUTHORS=$(git shortlog -sn --since="$START_DATE" | wc -l)

echo "Total commits: $TOTAL_COMMITS"
echo "Total files changed: $TOTAL_FILES"
echo "Total contributors: $TOTAL_AUTHORS"
echo ""

# Top 10 files summary
echo "Top 10 files by churn (production code):"
head -10 "$OUTPUT_DIR/churn_production.txt" | awk '{printf "  %4d commits: %s\n", $1, $2}'
echo ""

# Save summary
{
    echo "=== Code Churn Analysis Summary ==="
    echo "Date: $(date)"
    echo "Repository: $REPO_PATH"
    echo "Period: $START_DATE to present"
    echo ""
    echo "Total commits: $TOTAL_COMMITS"
    echo "Total files changed: $TOTAL_FILES"
    echo "Total contributors: $TOTAL_AUTHORS"
    echo ""
    echo "Top 10 files by churn (production code):"
    head -10 "$OUTPUT_DIR/churn_production.txt" | awk '{printf "  %4d commits: %s\n", $1, $2}'
} > "$OUTPUT_DIR/summary.txt"

print_success "Generated: $OUTPUT_DIR/summary.txt"

# === 5. Churn by Directory ===
print_header "Analyzing churn by directory..."

git log --format=format: --name-only --since="$START_DATE" \
  | egrep -v '^$' \
  | awk -F'/' '{if (NF>1) print $1"/"$2}' \
  | sort \
  | uniq -c \
  | sort -nr \
  | head -20 \
  > "$OUTPUT_DIR/churn_by_directory.txt"

print_success "Generated: $OUTPUT_DIR/churn_by_directory.txt"

# === 6. Churn Trend (Monthly) ===
print_header "Calculating monthly churn trend..."

{
    echo "month,commits"
    git log --format=format:%ad --date=format:%Y-%m --since="$START_DATE" \
      | sort \
      | uniq -c \
      | awk '{print $2","$1}'
} > "$OUTPUT_DIR/churn_trend_monthly.csv"

print_success "Generated: $OUTPUT_DIR/churn_trend_monthly.csv"

# === 7. High-Churn Thresholds ===
print_header "Identifying high-churn files..."

{
    echo "=== High-Churn Files (Potential Hotspots) ==="
    echo ""
    echo "Thresholds:"
    echo "  - Critical: >50 commits"
    echo "  - High: 20-50 commits"
    echo "  - Medium: 5-20 commits"
    echo ""

    CRITICAL=$(awk '$1 > 50 {count++} END {print count+0}' "$OUTPUT_DIR/churn_production.txt")
    HIGH=$(awk '$1 >= 20 && $1 <= 50 {count++} END {print count+0}' "$OUTPUT_DIR/churn_production.txt")
    MEDIUM=$(awk '$1 >= 5 && $1 < 20 {count++} END {print count+0}' "$OUTPUT_DIR/churn_production.txt")

    echo "Critical churn files (>50 commits): $CRITICAL"
    echo "High churn files (20-50 commits): $HIGH"
    echo "Medium churn files (5-20 commits): $MEDIUM"
    echo ""

    if [ "$CRITICAL" -gt 0 ]; then
        echo "=== Critical Churn Files ==="
        awk '$1 > 50 {printf "  %4d commits: %s\n", $1, $2}' "$OUTPUT_DIR/churn_production.txt"
        echo ""
    fi
} > "$OUTPUT_DIR/high_churn_analysis.txt"

print_success "Generated: $OUTPUT_DIR/high_churn_analysis.txt"

# === 8. Completion ===
print_header "Analysis Complete"
echo ""
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Files generated:"
echo "  - churn_all.txt               : All files ranked by churn"
echo "  - churn_production.txt        : Production code only"
echo "  - churn.csv                   : CSV format for spreadsheet import"
echo "  - churn_by_directory.txt      : Churn by directory"
echo "  - churn_trend_monthly.csv     : Monthly churn trend"
echo "  - high_churn_analysis.txt     : Files exceeding thresholds"
echo "  - summary.txt                 : Executive summary"
echo ""
print_success "Next steps:"
echo "  1. Review high_churn_analysis.txt for potential hotspots"
echo "  2. Run complexity analysis on top files (e.g., lizard)"
echo "  3. Use identify_hotspots.sh for combined churn+complexity analysis"
echo "  4. Import churn.csv into spreadsheet for visualization"
