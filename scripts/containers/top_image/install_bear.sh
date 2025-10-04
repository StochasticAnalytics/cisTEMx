#!/usr/bin/env bash
# Install Bear (Build EAR) for compilation database generation
# Used by clang-tidy for static analysis

set -euo pipefail

echo "Installing Bear (Build EAR)..."

# Update package list
apt-get update

# Install Bear from Ubuntu repositories
# Version 2.4.3 is available in Ubuntu 20.04 repos
# Later versions (3.x) can be built from source if needed
apt-get install -y bear

# Verify installation
bear --version

# Clean up
rm -rf /var/lib/apt/lists/*

echo "✓ Bear installation complete"
