#!/bin/bash
# analyze_coverage_gaps.sh
# Comprehensive test coverage gap analysis combining git history and coverage tools
#
# Usage:
#   ./analyze_coverage_gaps.sh [time_period] [coverage_file]
#
# Arguments:
#   time_period  - Time period for analysis (default: "3 months ago")
#   coverage_file - Path to coverage report (optional, for correlation analysis)
#
# Example:
#   ./analyze_coverage_gaps.sh "6 months ago"
#   ./analyze_coverage_gaps.sh "1 month ago" build/coverage.info

set -e

# Configuration
TIME_PERIOD="${1:-3 months ago}"
COVERAGE_FILE="${2:-}"
SRC_DIR="src"
TEST_DIR="test"

# Colors for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Verify we're in a git repository
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    print_error "Not in a git repository"
    exit 1
fi

echo "==================================================================="
echo "      Test Coverage Gap Analysis"
echo "==================================================================="
echo "Time Period: $TIME_PERIOD"
echo "Source Directory: $SRC_DIR/"
echo "Test Directory: $TEST_DIR/"
if [ -n "$COVERAGE_FILE" ]; then
    echo "Coverage File: $COVERAGE_FILE"
fi
echo "==================================================================="

# ============================================================================
# GAP CATEGORY 1: Test-to-Production Change Ratio
# ============================================================================

print_header "1. Test-to-Production Change Ratio"

echo "Analyzing line changes in last $(echo "$TIME_PERIOD" | sed 's/ ago//')..."

# Production changes
prod_stats=$(git log --stat --since="$TIME_PERIOD" -- "$SRC_DIR/" | \
  grep -E "^ (.*)\|" | \
  awk '{added+=$(NF-3); deleted+=$(NF-1)} END {printf "%d %d", added, deleted}')

prod_added=$(echo "$prod_stats" | awk '{print $1}')
prod_deleted=$(echo "$prod_stats" | awk '{print $2}')
prod_net=$((prod_added - prod_deleted))

# Test changes
test_stats=$(git log --stat --since="$TIME_PERIOD" -- "$TEST_DIR/" | \
  grep -E "^ (.*)\|" | \
  awk '{added+=$(NF-3); deleted+=$(NF-1)} END {printf "%d %d", added, deleted}')

test_added=$(echo "$test_stats" | awk '{print $1}')
test_deleted=$(echo "$test_stats" | awk '{print $2}')
test_net=$((test_added - test_deleted))

echo ""
echo "Production code:"
echo "  Lines added:    $prod_added"
echo "  Lines deleted:  $prod_deleted"
echo "  Net change:     $prod_net"
echo ""
echo "Test code:"
echo "  Lines added:    $test_added"
echo "  Lines deleted:  $test_deleted"
echo "  Net change:     $test_net"
echo ""

# Calculate ratio
if [ "$prod_net" -gt 0 ]; then
    ratio=$(echo "scale=2; $test_net / $prod_net" | bc)
    echo "Test:Production Ratio: 1:$ratio"
    echo ""

    if (( $(echo "$ratio >= 1.0" | bc -l) )); then
        print_success "Healthy ratio (≥1:1) - Tests keeping pace with production code"
    elif (( $(echo "$ratio >= 0.8" | bc -l) )); then
        print_warning "Acceptable ratio (≥1:0.8) - Consider increasing test coverage"
    else
        print_error "Warning ratio (<1:0.8) - Tests falling behind production code"
    fi
else
    echo "No significant production code changes in this period"
fi

# ============================================================================
# GAP CATEGORY 2: Commits Changing Production Without Tests
# ============================================================================

print_header "2. Recent Commits Without Test Changes"

echo "Finding commits that modified $SRC_DIR/ but not $TEST_DIR/..."
echo ""

untested_count=0
git log --name-only --format="%H|%an|%ad|%s" --since="$TIME_PERIOD" --no-merges | \
awk -v src="$SRC_DIR" -v test="$TEST_DIR" '
  BEGIN { commit=""; has_prod=0; has_test=0; }
  /\|/ {
    if (commit != "" && has_prod && !has_test) {
      print commit
    }
    commit = $0
    has_prod = 0
    has_test = 0
    next
  }
  {
    if ($0 ~ "^" src "/") has_prod = 1
    if ($0 ~ "^" test "/") has_test = 1
  }
  END {
    if (commit != "" && has_prod && !has_test) {
      print commit
    }
  }
' | head -10 | while IFS='|' read -r hash author date subject; do
    untested_count=$((untested_count + 1))
    echo "Commit: ${hash:0:8}"
    echo "  Author:  $author"
    echo "  Date:    $date"
    echo "  Subject: $subject"
    echo ""
done

if [ "$untested_count" -eq 0 ]; then
    print_success "No recent commits changed production without tests"
else
    print_warning "Found commits with production changes but no test updates (showing first 10)"
    echo "  Review these commits to ensure adequate test coverage"
fi

# ============================================================================
# GAP CATEGORY 3: High-Churn Files
# ============================================================================

print_header "3. High-Churn Production Files"

echo "Identifying frequently changed files (potential risk areas)..."
echo ""
echo "Top 15 most frequently changed production files:"
echo "Churn | File"
echo "------|------------------------------------------------------"

git log --no-merges --since="$TIME_PERIOD" --name-only --format='' | \
  grep "^$SRC_DIR/" | grep -v "$TEST_DIR" | \
  sort | uniq -c | sort -r -k1 -n | head -n 15 | \
  while read count file; do
      printf "%5d | %s\n" "$count" "$file"
  done

echo ""
print_warning "High-churn files are high-risk if inadequately tested"
echo "  Prioritize test coverage for files with >10 changes"

# ============================================================================
# GAP CATEGORY 4: Files with Bug Fixes
# ============================================================================

print_header "4. Files with Repeated Bug Fixes"

echo "Finding files most frequently associated with bug fixes..."
echo ""
echo "Top 10 files with bug-fix commits:"
echo "Bugs | File"
echo "-----|------------------------------------------------------"

git log --no-merges --since="$TIME_PERIOD" --format="%H|%s" | \
  grep -iE "fix|bug|issue|defect" | cut -d'|' -f1 | \
  while read commit; do
      git show --name-only --format= "$commit"
  done | \
  grep "^$SRC_DIR/" | grep -v "$TEST_DIR" | \
  sort | uniq -c | sort -rn | head -10 | \
  while read count file; do
      printf "%4d | %s\n" "$count" "$file"
  done

echo ""
print_warning "Files with repeated bugs may have insufficient test coverage"
echo "  Files with ≥3 bug fixes should have test quality review"

# ============================================================================
# COVERAGE CORRELATION (if coverage file provided)
# ============================================================================

if [ -n "$COVERAGE_FILE" ] && [ -f "$COVERAGE_FILE" ]; then
    print_header "5. Coverage Correlation Analysis"

    echo "Correlating git analysis with coverage data..."
    echo ""

    # Check if lcov
    if command -v lcov > /dev/null 2>&1; then
        echo "Zero-coverage files:"
        echo ""
        lcov --list "$COVERAGE_FILE" 2>/dev/null | grep "0.0%" | head -10 | \
          awk '{printf "  %s\n", $1}'

        echo ""
        echo "Low-coverage files (<50%):"
        echo ""
        lcov --list "$COVERAGE_FILE" 2>/dev/null | \
          awk '$2 ~ /%/ && $2 != "0.0%" {
            cov = $2;
            sub(/%/, "", cov);
            if (cov + 0 < 50 && cov + 0 > 0) {
              printf "  %6s | %s\n", $2, $1
            }
          }' | head -10
    else
        echo "lcov not available - skipping detailed coverage analysis"
        echo "Install lcov for coverage correlation: sudo apt-get install lcov"
    fi
fi

# ============================================================================
# SUMMARY & RECOMMENDATIONS
# ============================================================================

print_header "Summary & Recommendations"

echo "Gap Analysis Complete. Key Findings:"
echo ""

# Ratio assessment
if [ "$prod_net" -gt 0 ]; then
    ratio=$(echo "scale=2; $test_net / $prod_net" | bc)
    if (( $(echo "$ratio < 0.8" | bc -l) )); then
        echo "1. ${YELLOW}Action Required${NC}: Test-to-production ratio is low ($ratio)"
        echo "   → Increase test coverage in upcoming sprints"
        echo "   → Enforce diff-coverage on new PRs"
        echo ""
    else
        echo "1. ${GREEN}Good${NC}: Test-to-production ratio is healthy ($ratio)"
        echo ""
    fi
fi

# Commit assessment
echo "2. Review commits that changed production without test updates"
echo "   → Use diff-cover to ensure new changes are tested"
echo "   → See: resources/diff_cover_workflow.md"
echo ""

# High churn
echo "3. Prioritize test coverage for high-churn files"
echo "   → Focus on files changed >10 times in period"
echo "   → See: resources/prioritization_strategies.md"
echo ""

# Bug fixes
echo "4. Review test quality for files with repeated bugs"
echo "   → Consider mutation testing for critical areas"
echo "   → See: resources/tooling_integration.md"
echo ""

echo "Next Steps:"
echo "  • Set up diff-cover enforcement: resources/diff_cover_workflow.md"
echo "  • Prioritize gaps using risk scoring: scripts/find_fragile_code.py"
echo "  • Review workflow checklist: templates/gap_assessment_workflow.md"
echo ""

echo "==================================================================="
echo "Report generated: $(date)"
echo "==================================================================="
