# LLM Optimization Tools Installation

## Overview

The `install_llm_optimization_tools.sh` script installs lightweight dependencies for Phase 7 (LLM Optimization) of the documentation system.

## When to Enable

**Enable this script when you're ready for Phase 7**, which includes:

- Token counting for LLM context window management
- Semantic embeddings for code similarity search
- LLM-friendly JSON schema generation

**Timeline:** Phase 7 is typically 5-6 weeks into the documentation system implementation.

## What Gets Installed

1. **tiktoken** (~10MB)
   - OpenAI's tokenization library
   - Used for accurate token counting across different LLM models
   - Ensures documentation chunks fit within LLM context windows

2. **fastembed** (~150-200MB total)
   - Lightweight embeddings using ONNX Runtime (no PyTorch!)
   - Uses quantized models optimized for CPU
   - Downloads model weights (~100-200MB) on first use
   - Used for semantic code search

**Total container size increase:** ~150-200MB (vs ~2GB for sentence-transformers)

## How to Enable

### Option 1: Add to Dockerfile (Recommended)

In `scripts/containers/top_image/Dockerfile`, add after the documentation tooling installation:

```dockerfile
# Install LLM optimization tools (Phase 7)
ARG build_llm_tools="false"
RUN if [[ "x${build_llm_tools}" == "xtrue" ]] ; then /tmp/install_llm_optimization_tools.sh ; fi
```

Then modify `COPY` line to include the new script:

```dockerfile
COPY install_wx_3.1.5.sh install_node_16.sh install_node_22_and_claude.sh \
     requirements.txt install_libtorch.sh install_documentation_tooling.sh \
     install_llm_optimization_tools.sh /tmp/
```

Build with:

```bash
./regenerate_containers.sh --build-llm-tools=true
```

### Option 2: Install Manually (Testing)

If you want to test without rebuilding the container:

```bash
# Inside the container
source ${HOME}/venv/bin/activate
bash /tmp/install_llm_optimization_tools.sh
```

### Option 3: Install Later via pip

You can also skip the script and install directly when needed:

```bash
pip install "tiktoken>=0.5.0" "fastembed>=0.2.0"
```

## Testing

After installation, verify:

```bash
# Test tiktoken
python -c "import tiktoken; print('✓ tiktoken works')"

# Test fastembed
python -c "from fastembed import TextEmbedding; print('✓ fastembed works')"
```

## Alternative: Skip LLM Optimization

Phase 7 (LLM Optimization) is **optional**. The documentation system works perfectly well with:

- Tag-based hierarchical search
- Full-text search via MkDocs
- Metadata filtering (complexity, GPU/CPU, etc.)

You can implement Phases 1-6 and decide later if you need semantic search and token optimization.

## Dependencies

These packages require:

- Python 3.10+ (already in container via /home/cisTEMdev/venv)
- ONNX Runtime (installed automatically)
- No CUDA or GPU needed

## See Also

- Main implementation plan: `docs/IMPLEMENTATION_PLAN.md`
- Phase 7 details: See "Phase 7: LLM Optimization" section
- Existing docs tooling: `install_documentation_tooling.sh`
