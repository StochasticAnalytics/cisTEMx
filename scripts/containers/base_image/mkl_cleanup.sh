#!/bin/bash
# mkl_cleanup.sh - Optimize Intel MKL installation for containerized deployment
# Usage: ./mkl_cleanup.sh [static|dynamic]
#
# This script removes unnecessary MKL components to reduce Docker image size
# while keeping either static (.a) or dynamic (.so) libraries based on your choice.

set -e  # Exit on error

LINK_TYPE="${1:-dynamic}"  # Default to dynamic if not specified
MKL_ROOT="/opt/intel/oneapi"

echo "=================================================="
echo "Intel MKL Cleanup Script"
echo "Link type: ${LINK_TYPE}"
echo "=================================================="

if [[ ! -d "${MKL_ROOT}" ]]; then
    echo "Error: ${MKL_ROOT} not found!"
    exit 1
fi

cd "${MKL_ROOT}"

# ============================================================================
# STEP 1: Remove components that are never needed (regardless of link type)
# ============================================================================
echo "Removing unnecessary components..."

# Remove SYCL/GPU libraries (you're using CUDA, not SYCL)
rm -rf mkl/latest/lib/intel64/*_sycl* 2>/dev/null || true
rm -rf mkl/latest/lib/intel64/*_blas95* 2>/dev/null || true
rm -rf mkl/latest/lib/intel64/*_lapack95* 2>/dev/null || true

# Remove 32-bit libraries (assuming 64-bit only)
rm -rf mkl/latest/lib/ia32 2>/dev/null || true

# Remove ScaLAPACK/cluster computing libraries (unless you need distributed MKL)
cd mkl/latest/lib/intel64 2>/dev/null || true
rm -f libmkl_scalapack*.a libmkl_scalapack*.so* 2>/dev/null || true
rm -f libmkl_blacs*.a libmkl_blacs*.so* 2>/dev/null || true
rm -f libmkl_cdft*.a libmkl_cdft*.so* 2>/dev/null || true

# Remove PGI compiler threading (unless you use PGI/NVIDIA compilers)
rm -f libmkl_pgi_thread.a libmkl_pgi_thread.so* 2>/dev/null || true

cd "${MKL_ROOT}"

# Remove tools, benchmarks, examples, documentation
rm -rf mkl/latest/tools 2>/dev/null || true
rm -rf mkl/latest/interfaces 2>/dev/null || true
rm -rf mkl/latest/benchmarks 2>/dev/null || true
rm -rf mkl/latest/examples 2>/dev/null || true

# Remove debugger and conda channel
rm -rf debugger/ 2>/dev/null || true
rm -rf conda_channel/ 2>/dev/null || true

# Remove licensing and modulefiles
rm -rf */latest/licensing 2>/dev/null || true
rm -rf */latest/modulefiles 2>/dev/null || true

# Remove documentation
find . -type d -name 'documentation' -exec rm -rf {} + 2>/dev/null || true

# Clean up compiler directory but keep OpenMP runtime
find compiler/latest/linux/lib -type f \
     ! -name 'libiomp5*' \
     ! -name 'libgomp*' \
     ! -name 'libtbb*' \
     ! -name 'libimf*' \
     ! -name 'libintlc*' \
     ! -name 'libirng*' \
     ! -name 'libsvml*' \
     -delete 2>/dev/null || true

# ============================================================================
# STEP 2: Remove either .a or .so files based on link type
# ============================================================================

cd mkl/latest/lib/intel64 2>/dev/null || cd "${MKL_ROOT}"

if [[ "${LINK_TYPE}" == "static" ]]; then
    echo "Configuring for STATIC linking..."
    echo "  - Keeping: .a files (static libraries)"
    echo "  - Removing: .so files (dynamic libraries)"
    
    # Remove all .so files and symlinks
    find . -type f -name "*.so*" -delete 2>/dev/null || true
    find . -type l -name "*.so*" -delete 2>/dev/null || true
    
    # Verify we kept the essential .a files
    if [[ -f libmkl_core.a ]] && [[ -f libmkl_intel_lp64.a ]]; then
        echo "✓ Static libraries verified"
    else
        echo "WARNING: Essential static libraries may be missing!"
    fi
    
elif [[ "${LINK_TYPE}" == "dynamic" ]]; then
    echo "Configuring for DYNAMIC linking..."
    echo "  - Keeping: .so files (dynamic libraries)"
    echo "  - Removing: .a files (static libraries)"
    
    # Remove all .a files
    find . -type f -name "*.a" -delete 2>/dev/null || true
    
    # Verify we kept the essential .so files
    if [[ -f libmkl_core.so ]] || [[ -f libmkl_rt.so ]]; then
        echo "✓ Dynamic libraries verified"
    else
        echo "WARNING: Essential dynamic libraries may be missing!"
    fi
    
else
    echo "Error: Invalid link type '${LINK_TYPE}'"
    echo "Usage: $0 [static|dynamic]"
    exit 1
fi

# ============================================================================
# STEP 3: Report space savings
# ============================================================================

cd "${MKL_ROOT}"
TOTAL_SIZE=$(du -sh . | cut -f1)
MKL_SIZE=$(du -sh mkl/latest 2>/dev/null | cut -f1 || echo "N/A")

echo ""
echo "=================================================="
echo "Cleanup complete!"
echo "Total oneAPI directory size: ${TOTAL_SIZE}"
echo "MKL directory size: ${MKL_SIZE}"
echo "Link type: ${LINK_TYPE}"
echo "=================================================="
echo ""
echo "Next steps:"
if [[ "${LINK_TYPE}" == "static" ]]; then
    echo "  Use static linking in your build (see Dockerfile comments)"
    echo "  Link with: -Wl,--start-group \$MKLROOT/lib/intel64/libmkl_*.a -Wl,--end-group"
else
    echo "  Use dynamic linking in your build (see Dockerfile comments)"
    echo "  Link with: -L\$MKLROOT/lib/intel64 -lmkl_rt"
fi

exit 0
