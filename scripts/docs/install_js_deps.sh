#!/bin/bash
# Install JavaScript dependencies for MkDocs documentation
# This script downloads npm packages and copies built files to docs/javascripts/vendor/
# for offline-capable documentation deployment

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENDOR_DIR="$PROJECT_ROOT/docs/javascripts/vendor"

echo "=========================================="
echo "Installing JavaScript Dependencies"
echo "=========================================="
echo ""

# Navigate to project root
cd "$PROJECT_ROOT"

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "ERROR: Node.js is not installed"
    echo "Please install Node.js first (see scripts/containers/top_layer/install_node_22_and_claude.sh)"
    exit 1
fi

echo "✓ Node.js version: $(node --version)"
echo "✓ npm version: $(npm --version)"
echo ""

# Install npm dependencies
echo "Installing npm packages..."
npm install --production --no-audit --no-fund

# Create vendor directory
echo ""
echo "Creating vendor directory..."
mkdir -p "$VENDOR_DIR"

# Copy files to vendor directory
echo "Copying built files to docs/javascripts/vendor/..."

# Core Cytoscape
cp node_modules/cytoscape/dist/cytoscape.min.js "$VENDOR_DIR/" && \
    echo "  ✓ cytoscape.min.js"

# Dagre layout
cp node_modules/dagre/dist/dagre.min.js "$VENDOR_DIR/" && \
    echo "  ✓ dagre.min.js"
cp node_modules/cytoscape-dagre/cytoscape-dagre.js "$VENDOR_DIR/" && \
    echo "  ✓ cytoscape-dagre.js"

# Cola layout
cp node_modules/webcola/WebCola/cola.min.js "$VENDOR_DIR/" && \
    echo "  ✓ cola.min.js"
cp node_modules/cytoscape-cola/cytoscape-cola.js "$VENDOR_DIR/" && \
    echo "  ✓ cytoscape-cola.js"

# fCoSE layout (new)
cp node_modules/layout-base/layout-base.js "$VENDOR_DIR/" && \
    echo "  ✓ layout-base.js (fCoSE dependency)"
cp node_modules/cose-base/cose-base.js "$VENDOR_DIR/" && \
    echo "  ✓ cose-base.js (fCoSE dependency)"
cp node_modules/cytoscape-fcose/cytoscape-fcose.js "$VENDOR_DIR/" && \
    echo "  ✓ cytoscape-fcose.js"

# Grid-guide (new)
cp node_modules/cytoscape-grid-guide/cytoscape-grid-guide.js "$VENDOR_DIR/" && \
    echo "  ✓ cytoscape-grid-guide.js"

# Verify all files exist
echo ""
echo "Verifying files..."
REQUIRED_FILES=(
    "cytoscape.min.js"
    "dagre.min.js"
    "cytoscape-dagre.js"
    "cola.min.js"
    "cytoscape-cola.js"
    "layout-base.js"
    "cose-base.js"
    "cytoscape-fcose.js"
    "cytoscape-grid-guide.js"
)

MISSING=0
for file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$VENDOR_DIR/$file" ]]; then
        echo "  ✗ MISSING: $file"
        MISSING=1
    fi
done

if [[ $MISSING -eq 1 ]]; then
    echo ""
    echo "ERROR: Some files are missing. Installation failed."
    exit 1
fi

# Calculate total size
TOTAL_SIZE=$(du -sh "$VENDOR_DIR" | awk '{print $1}')

echo ""
echo "=========================================="
echo "✓ Installation Complete"
echo "=========================================="
echo "Location: $VENDOR_DIR"
echo "Files: ${#REQUIRED_FILES[@]}"
echo "Total size: $TOTAL_SIZE"
echo ""
echo "Next steps:"
echo "  1. Update mkdocs.yml to use local vendor files"
echo "  2. Test with: mkdocs build"
echo "  3. Commit vendor/ files for offline capability"
echo ""
