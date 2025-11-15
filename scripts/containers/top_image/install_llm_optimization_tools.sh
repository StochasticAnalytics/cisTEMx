#!/bin/bash
set -e  # Exit on any error

# install_llm_optimization_tools.sh - Install LLM optimization dependencies (Phase 7)
# Designed for Ubuntu Docker containers
# Assumes: Python 3.10 venv at /home/cisTEMdev already exists
# Required for: Phase 7 of documentation system implementation

echo "🤖 Installing LLM Optimization Tools (Phase 7)"
echo "=============================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Verify we're using the venv
print_status "Verifying Python venv..."
if [[ -z "$VIRTUAL_ENV" ]]; then
    print_error "VIRTUAL_ENV not set"
    exit 1
fi

print_status "Using Python: $(which python) ($(python --version))"
print_status "Using pip: $(which pip) ($(pip --version))"

# Install LLM optimization packages
print_status "Installing LLM optimization packages..."

# Token counting for LLM context windows
print_status "Installing tiktoken (OpenAI tokenizer)..."
pip install "tiktoken>=0.5.0"

# Lightweight embeddings using ONNX (no PyTorch dependency)
print_status "Installing fastembed (lightweight embeddings with ONNX)..."
pip install "fastembed>=0.2.0"

# Test the installation
print_status "Testing installation..."

# Test tiktoken
python -c "
import tiktoken
enc = tiktoken.get_encoding('cl100k_base')
tokens = enc.encode('Hello, world!')
print('✅ tiktoken: Successfully encoded test string ({} tokens)'.format(len(tokens)))
"

# Test fastembed
python -c "
from fastembed import TextEmbedding
print('✅ fastembed: Library loaded successfully')
print('   Note: Models will be downloaded on first use (~100-200MB)')
"

# Clean up to reduce image size
print_status "Cleaning up pip cache..."
pip cache purge

print_success "LLM optimization tools installed successfully!"
print_status "Installed packages:"
echo "  - tiktoken: Token counting for LLM context windows"
echo "  - fastembed: Lightweight embeddings using ONNX Runtime (no PyTorch)"
print_status ""
print_status "Note: FastEmbed will download models (~100-200MB) on first use"
print_status "Ready for Phase 7: LLM Optimization! 🤖"
